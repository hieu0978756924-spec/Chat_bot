"""Cấu hình pytest: mỗi test dùng một CSDL tạm (không đụng data_store/app.db)."""

import datetime as dt
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config):
    config.addinivalue_line("markers", "ollama: cần Ollama đang chạy với model trong .env (chậm)")


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_OLLAMA_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="đặt RUN_OLLAMA_TESTS=1 để chạy test cần Ollama")
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip)


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    path = str(tmp_path / "app.db")
    monkeypatch.setenv("SQLITE_DB_PATH", path)
    from actions import db
    from core import entities

    db.init_database(reset=True, today=dt.date.today())
    entities.reset_restaurant_cache()
    yield path
    entities.reset_restaurant_cache()


@pytest.fixture()
def fallback_agent(temp_db, monkeypatch):
    """Agent chạy pipeline đầy đủ nhưng NLU = bộ dự phòng (không cần Ollama)."""
    monkeypatch.setenv("LLM_PROVIDER", "none")
    from agent.agent import RestaurantAgent

    return RestaurantAgent(warmup=False)


@pytest.fixture()
def llm_agent(temp_db, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    from agent.agent import RestaurantAgent

    return RestaurantAgent(warmup=False)
