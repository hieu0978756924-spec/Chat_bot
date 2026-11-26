"""Lỗi hệ thống (mock): DB lỗi -> không bao giờ báo thành công; Ollama lỗi -> NLU dự phòng + báo người dùng."""

import json
import logging
import sqlite3

import pytest
import requests

from actions import db
from helpers import hallucination_checks

BOOKING_REQUESTS = [
    "Đặt bàn Lẩu Nhà Mình tối mai 7 giờ cho 4 người",
    "dat ban lau nha minh toi mai 7h 4 nguoi",
    "Book Lẩu Nhà Mình 19:00 ngày mai, 4 khách",
    "cho mình giữ 1 bàn ở Lẩu Nhà Mình lúc 7 giờ tối mai, nhà có 4 người",
    "Lẩu Nhà Mình, mai, 19h, 4 ng, đặt giúp",
    "đặt chỗ lẩu nhà mình tối mai 7h tối bốn người",
    "Mình muốn đặt bàn tại Lẩu Nhà Mình vào 7 giờ tối ngày mai cho gia đình 4 người",
    "dat lau nha minh 4 nguoi 7 gio toi mai nhe",
]


def _break_writes(monkeypatch):
    def broken_transaction():
        raise sqlite3.OperationalError("database is locked (giả lập)")
    monkeypatch.setattr(db, "transaction", broken_transaction)


@pytest.mark.parametrize("text", BOOKING_REQUESTS)
def test_db_write_error_never_reports_success(fallback_agent, monkeypatch, temp_db, text):
    a = fallback_agent
    r1 = a.ask(text)
    assert "confirm" in a.last_trace["blocks"], r1
    n_before = sqlite3.connect(temp_db).execute("SELECT COUNT(*) FROM dat_ban").fetchone()[0]
    _break_writes(monkeypatch)
    reply = a.ask("xác nhận")
    assert "action_failed" in a.last_trace["blocks"], reply
    assert "thành công" not in reply.lower() and "DB0" not in reply.split("Chưa thể")[0]
    assert "CHƯA ghi nhận" in reply
    assert sqlite3.connect(temp_db).execute("SELECT COUNT(*) FROM dat_ban").fetchone()[0] == n_before
    assert not hallucination_checks(a, reply)


SEARCHES = ["tìm quán lẩu", "quan nuong tan binh", "có quán chay nào không", "kiếm nhà hàng hải sản quận 1",
            "quán pizza phú nhuận", "tim quan an quan 5", "món Nhật ở đâu ngon", "quán ăn Việt quận 3"]


@pytest.mark.parametrize("text", SEARCHES)
def test_db_read_error_reports_failure(fallback_agent, monkeypatch, text):
    a = fallback_agent

    def broken_conn(*args, **kwargs):
        raise sqlite3.OperationalError("unable to open database file (giả lập)")
    monkeypatch.setattr(db, "get_connection", broken_conn)
    reply = a.ask(text)
    assert "tool_error" in a.last_trace["blocks"] or "ask_slots" in a.last_trace["blocks"], reply
    assert "restaurant_list" not in a.last_trace["blocks"]
    assert "⭐" not in reply   # không bịa danh sách quán


OLLAMA_MESSAGES = ["Tìm quán lẩu cho 4 người tối mai 7 giờ", "huy don DB003", "có món chay không",
                   "Nếu đến trễ thì giữ bàn bao lâu?", "đặt bàn giúp mình", "app không gửi OTP",
                   "phục vụ chậm quá", "giá vàng hôm nay"]


@pytest.mark.parametrize("text", OLLAMA_MESSAGES)
@pytest.mark.parametrize("failure", ["connection", "timeout"])
def test_ollama_down_uses_fallback_and_tells_user(llm_agent, monkeypatch, caplog, text, failure):
    def boom(*args, **kwargs):
        if failure == "timeout":
            raise requests.Timeout("read timed out (giả lập)")
        raise requests.ConnectionError("connection refused (giả lập)")
    monkeypatch.setattr(requests, "post", boom)
    with caplog.at_level(logging.WARNING):
        reply = llm_agent.ask(text)
    t = llm_agent.last_trace
    assert t["nlu_source"] == "fallback_embedding" and t["llm_error"] == "LLM_UNAVAILABLE"
    assert "LLM_UNAVAILABLE" in caplog.text
    assert reply.startswith("⚠️ Mô hình AI cục bộ đang không phản hồi")
    assert not hallucination_checks(llm_agent, reply)


def test_ollama_invalid_json_retries_once_then_fallback(llm_agent, monkeypatch, caplog):
    calls = []

    class Resp:
        status_code = 200

        def json(self):
            return {"message": {"content": json.dumps({"intents": ["không_phải_intent"]})}}

    def fake_post(*args, **kwargs):
        calls.append(1)
        return Resp()
    monkeypatch.setattr(requests, "post", fake_post)
    with caplog.at_level(logging.WARNING):
        llm_agent.ask("tìm quán lẩu")
    assert len(calls) == 2                      # 1 lần + 1 lần retry
    assert llm_agent.last_trace["llm_error"] == "LLM_INVALID_OUTPUT"
    assert llm_agent.last_trace["nlu_source"] == "fallback_embedding"
    assert "LLM_INVALID_OUTPUT" in caplog.text


def test_fewshot_and_examples_disjoint_from_test_sets():
    """Prompt few-shot và câu mẫu dự phòng KHÔNG được chứa câu của tập dev/held-out."""
    import glob
    import os

    import yaml

    from core.nlu import SYSTEM_PROMPT
    from core.text import fold
    from helpers import HERE

    seen = set()
    with open(os.path.join(os.path.dirname(HERE), "data", "nlu", "intent_examples.yaml"), encoding="utf-8") as f:
        for sents in yaml.safe_load(f).values():
            seen |= {fold(s).strip(" ?.!") for s in sents}
    prompt = fold(SYSTEM_PROMPT)
    for path in glob.glob(os.path.join(HERE, "eval", "nlu_*.yaml")):
        with open(path, encoding="utf-8") as f:
            for g in yaml.safe_load(f).values():
                for c in g["cases"]:
                    s = fold(c["text"]).strip(" ?.!")
                    if len(s.split()) <= 1:
                        continue   # câu 1 từ ("không", "yes") không thể tránh trùng
                    assert s not in seen, f"câu test nằm trong intent_examples: {c['text']}"
                    assert f'"{s}' not in prompt, f"câu test nằm trong few-shot: {c['text']}"
