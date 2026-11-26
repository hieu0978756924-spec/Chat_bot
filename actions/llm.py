"""Kiểm tra trạng thái Ollama cục bộ (chỉ Local LLM, không dùng API LLM bên ngoài)."""

from typing import Any, Dict, List

import requests

import config


def list_local_models() -> List[str]:
    try:
        r = requests.get(f"{config.ollama_url()}/api/tags", timeout=3)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return []


def llm_status() -> Dict[str, Any]:
    """Trạng thái thực: Ollama có chạy không VÀ model cấu hình đã được tải về chưa."""
    model = config.ollama_model()
    if config.llm_disabled():
        return {"ok": False, "code": "LLM_DISABLED",
                "message": "LLM tắt (LLM_PROVIDER=none) — dùng NLU dự phòng (embedding)."}
    models = list_local_models()
    if not models and not _online():
        return {"ok": False, "code": "LLM_UNAVAILABLE",
                "message": f"Ollama không chạy tại {config.ollama_url()} — dùng NLU dự phòng. Chạy 'ollama serve'."}
    if not any(m == model or m.startswith(model + ":") for m in models):
        return {"ok": False, "code": "MODEL_MISSING",
                "message": f"Chưa có model '{model}' trong Ollama — chạy 'ollama pull {model}'. Đang dùng NLU dự phòng."}
    return {"ok": True, "code": "OK", "message": f"Ollama {model} @ {config.ollama_url()}"}


def _online() -> bool:
    try:
        return requests.get(f"{config.ollama_url()}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False
