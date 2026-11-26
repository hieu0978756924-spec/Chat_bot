"""Nạp model Sentence Transformers một lần, dùng chung cho RAG và NLU dự phòng."""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
_MODEL = None
_FAILED = False


def get_embedder():
    """Trả về SentenceTransformer hoặc None nếu không nạp được (ưu tiên cache offline)."""
    global _MODEL, _FAILED
    if _MODEL is not None or _FAILED:
        return _MODEL
    with _LOCK:
        if _MODEL is not None or _FAILED:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer

            name = config.embedding_model()
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                _MODEL = SentenceTransformer(name, local_files_only=True)
            except Exception:
                os.environ["HF_HUB_OFFLINE"] = "0"
                _MODEL = SentenceTransformer(name)
        except Exception as e:  # pragma: no cover - phụ thuộc môi trường
            logger.error("EMBEDDING_UNAVAILABLE: %s", e)
            _FAILED = True
            _MODEL = None
    return _MODEL


def encode(texts: List[str]) -> Optional[np.ndarray]:
    model = get_embedder()
    if model is None:
        return None
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False).astype("float32")
