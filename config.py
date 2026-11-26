"""Cấu hình tập trung cho toàn bộ chatbot (đọc từ biến môi trường / file .env).

Mọi module khác chỉ đọc cấu hình qua module này để tránh lệch giá trị mặc định.
Các giá trị đọc *tại thời điểm gọi* (hàm), nhờ đó test có thể đổi biến môi trường.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(BASE_DIR, path))


# ---------------------------------------------------------------- Đường dẫn
DATA_STORE_DIR = os.path.join(BASE_DIR, "data_store")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
POLICIES_DIR = os.path.join(BASE_DIR, "docs", "policies")
NLU_DATA_DIR = os.path.join(BASE_DIR, "data", "nlu")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "backup")


def db_path() -> str:
    return _abs(_env("SQLITE_DB_PATH", "./data_store/app.db"))


def faiss_index_path() -> str:
    return _abs(_env("FAISS_INDEX_PATH", "./data_store/faiss_policy.index"))


def faiss_meta_path() -> str:
    return os.path.splitext(faiss_index_path())[0] + "_meta.json"


# ---------------------------------------------------------------- Local LLM (Ollama)
def ollama_url() -> str:
    return _env("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def ollama_model() -> str:
    # Mặc định qwen2.5:3b: máy 8GB RAM / chỉ CPU không chạy nổi bản 7b.
    return _env("OLLAMA_MODEL", "qwen2.5:3b")


def ollama_timeout() -> float:
    return float(_env("OLLAMA_TIMEOUT", "90"))


def ollama_keep_alive() -> str:
    return _env("OLLAMA_KEEP_ALIVE", "30m")


def ollama_num_ctx() -> int:
    return int(_env("OLLAMA_NUM_CTX", "4096"))


def llm_disabled() -> bool:
    """LLM_PROVIDER=none -> bỏ qua Ollama, dùng NLU dự phòng (dùng cho test/máy yếu)."""
    return _env("LLM_PROVIDER", "ollama").lower() in ("none", "off", "fallback")


# ---------------------------------------------------------------- Embedding / RAG
def embedding_model() -> str:
    return _env("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def rag_threshold() -> float:
    return float(_env("RAG_THRESHOLD", "0.40"))


def rag_top_k() -> int:
    return int(_env("RAG_TOP_K", "3"))


# ---------------------------------------------------------------- Nghiệp vụ
CHEAP_BUDGET_VND = int(_env("CHEAP_BUDGET_VND", "150000"))   # "rẻ / bình dân"
DEMO_DAYS = int(_env("DEMO_DAYS", "14"))                     # số ngày sinh khung giờ demo
PENDING_ACTION_TTL_S = int(_env("PENDING_ACTION_TTL_S", "600"))
MAX_PARTY_SIZE = 50
TABLE_SIZES = (2, 4, 6, 10)
HISTORY_TURNS_FOR_NLU = 4
