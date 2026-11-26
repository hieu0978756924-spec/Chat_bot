"""RAG chính sách: docs/policies/*.md + dataset/chinh_sach.csv -> FAISS.

- Chia đoạn theo mục Markdown, mỗi đoạn 300-500 ký tự, overlap ~60 ký tự; metadata: file, tiêu đề, mục.
- Embedding đa ngôn ngữ (config.EMBEDDING_MODEL), FAISS IndexFlatIP trên vector chuẩn hóa (cosine).
- Điểm cuối = 0.5 * cosine(câu hỏi, đoạn) + 0.2 * cosine(câu hỏi, tiêu đề điều khoản)
              + 0.3 * độ phủ từ khóa của câu hỏi trong tiêu đề điều khoản.
  (MiniLM đa ngôn ngữ yếu với tiếng Việt; tiêu đề điều khoản ngắn và rõ nghĩa nên giúp xếp hạng đúng hơn,
   và kéo điểm câu ngoài chủ đề xuống thấp.)
- Dưới ngưỡng -> không trả gì (policy sẽ nói "chưa tìm thấy" và đề nghị tạo phiếu hỗ trợ).
- Index tự build lại khi tài liệu hoặc model embedding thay đổi (manifest hash).
- KHÔNG dùng RAG cho bàn trống, giá, thực đơn (luôn truy vấn DB).
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from core import embeddings
from core.text import fold

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
_INDEX: Optional[Tuple[Any, List[Dict[str, Any]]]] = None
_TITLE_VECS: Optional[np.ndarray] = None

CHUNK_MIN, CHUNK_MAX, OVERLAP = 300, 500, 60
STOPWORDS = set("""toi minh ban em anh chi a ạ oi nhe nha thi la co khong duoc gi nao the nhu sao
va voi cua cho o tai neu khi thi ma hay hoac nay do kia vay roi nhung cac mot nhieu it bao
quan nha hang muon can biet hoi cho xin vui long giup quy dinh chinh sach dieu khoan ve gi
neu thi""".split())


# ---------------------------------------------------------------------- chia đoạn
def _split_sections(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Trả về (tiêu đề H1, [(tên mục, nội dung)])."""
    title_m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_m.group(1).strip() if title_m else ""
    body = re.sub(r"^#\s+.+$", "", text, count=1, flags=re.MULTILINE)
    parts = re.split(r"^##\s+(.+)$", body, flags=re.MULTILINE)
    sections: List[Tuple[str, str]] = []
    if parts[0].strip():
        sections.append(("", parts[0]))
    for i in range(1, len(parts), 2):
        sections.append((parts[i].strip(), parts[i + 1] if i + 1 < len(parts) else ""))
    return title, sections


def chunk_section(content: str) -> List[str]:
    """Gom các ý (dòng/đoạn) thành đoạn 300-500 ký tự, overlap OVERLAP ký tự."""
    items = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"\n(?=\s*(?:\d+\.|-|\*)\s)|\n\s*\n", content)]
    items = [x for x in items if x]
    chunks: List[str] = []
    cur = ""
    for it in items:
        while len(it) > CHUNK_MAX:  # ý quá dài -> cắt ở ranh giới câu/từ
            cut = it.rfind(". ", 0, CHUNK_MAX)
            cut = cut + 1 if cut > CHUNK_MIN else it.rfind(" ", 0, CHUNK_MAX)
            piece, it = it[:cut].strip(), it[cut:].strip()
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(piece)
        if cur and len(cur) + 1 + len(it) > CHUNK_MAX:
            chunks.append(cur)
            tail = cur[-OVERLAP:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = f"…{tail} {it}"
        else:
            cur = f"{cur} {it}".strip()
    if cur:
        if chunks and len(cur) < CHUNK_MIN // 2 and len(chunks[-1]) + len(cur) <= CHUNK_MAX + 100:
            chunks[-1] = f"{chunks[-1]} {cur}"
        else:
            chunks.append(cur)
    return chunks


def load_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(config.POLICIES_DIR, "*.md"))):
        with open(path, encoding="utf-8") as f:
            title, sections = _split_sections(f.read())
        for sec_name, content in sections:
            for i, c in enumerate(chunk_section(content)):
                chunks.append({"id": f"{os.path.basename(path)}#{sec_name or 'main'}#{i}",
                               "file": os.path.basename(path), "title": title, "section": sec_name,
                               "restaurant_id": "ALL", "content": c})
    csv_path = os.path.join(config.DATASET_DIR, "chinh_sach.csv")
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                chunks.append({"id": f"chinh_sach.csv#{r['ma_chinh_sach']}", "file": "chinh_sach.csv",
                               "title": f"Quy định riêng của nhà hàng {r['ma_nha_hang']}",
                               "section": r["loai_chinh_sach"], "restaurant_id": r["ma_nha_hang"],
                               "content": r["noi_dung"].strip()})
    return chunks


def _manifest(chunks: List[Dict[str, Any]]) -> str:
    h = hashlib.sha256(config.embedding_model().encode())
    for c in chunks:
        h.update(c["id"].encode())
        h.update(c["content"].encode())
    return h.hexdigest()


def _embed_text(c: Dict[str, Any]) -> str:
    return f"{c['title']}. {c['section']}. {c['content']}"


# ---------------------------------------------------------------------- build / load
def build_index(force: bool = False) -> bool:
    """Build (hoặc dùng lại) FAISS index. Trả False nếu không có embedding/faiss."""
    global _INDEX
    import faiss

    chunks = load_chunks()
    digest = _manifest(chunks)
    ipath, mpath = config.faiss_index_path(), config.faiss_meta_path()
    if not force and os.path.exists(ipath) and os.path.exists(mpath):
        try:
            with open(mpath, encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict) and meta.get("manifest") == digest:
                _INDEX = (faiss.read_index(ipath), meta["chunks"])
                return True
        except Exception:
            pass
    vecs = embeddings.encode([_embed_text(c) for c in chunks])
    if vecs is None:
        return False
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    os.makedirs(os.path.dirname(ipath), exist_ok=True)
    faiss.write_index(index, ipath)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump({"manifest": digest, "model": config.embedding_model(), "chunks": chunks},
                  f, ensure_ascii=False, indent=1)
    _INDEX = (index, chunks)
    logger.info("FAISS index: %d đoạn", len(chunks))
    return True


def _title_text(c: Dict[str, Any]) -> str:
    return f"{c['title']}. {c['section']}"


def _get_index() -> Optional[Tuple[Any, List[Dict[str, Any]]]]:
    global _TITLE_VECS
    with _LOCK:
        if _INDEX is None:
            build_index()
        if _INDEX is not None and _TITLE_VECS is None:
            _TITLE_VECS = embeddings.encode([_title_text(c) for c in _INDEX[1]])
    return _INDEX


def _content_tokens(text: str) -> set:
    return {t for t in re.findall(r"\w+", fold(text)) if len(t) >= 2 and t not in STOPWORDS}


def search(query: str, restaurant_id: Optional[str] = None, top_k: Optional[int] = None,
           threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Trả về tối đa top_k đoạn đạt ngưỡng: [{file, title, section, content, score, cosine}]."""
    top_k = top_k or config.rag_top_k()
    threshold = config.rag_threshold() if threshold is None else threshold
    loaded = _get_index()
    q = embeddings.encode([query])
    if loaded is None or q is None:
        raise RuntimeError("RAG_UNAVAILABLE: không nạp được embedding/FAISS")
    index, chunks = loaded
    sims, idx = index.search(q, len(chunks))
    title_sims = (_TITLE_VECS @ q.T).ravel() if _TITLE_VECS is not None else None
    q_tok = _content_tokens(query)
    hits = []
    for cos, i in zip(sims[0], idx[0]):
        if i < 0:
            continue
        c = chunks[i]
        if c["restaurant_id"] not in ("ALL", restaurant_id):
            continue
        t_tok = _content_tokens(_title_text(c))
        title_lex = len(q_tok & t_tok) / len(q_tok) if q_tok else 0.0
        t_cos = float(title_sims[i]) if title_sims is not None else float(cos)
        score = 0.5 * float(cos) + 0.2 * t_cos + 0.3 * title_lex
        hits.append({**{k: c[k] for k in ("id", "file", "title", "section", "content")},
                     "score": round(score, 4), "cosine": round(float(cos), 4)})
    hits.sort(key=lambda h: -h["score"])
    return [h for h in hits if h["score"] >= threshold][:top_k]
