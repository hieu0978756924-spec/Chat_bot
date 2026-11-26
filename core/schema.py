"""Schema dữ liệu: đầu ra NLU, JSON schema cho Ollama `format`, và bộ kiểm tra/chuẩn hóa giá trị slot."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

import config

VALID_INTENTS = [
    "greeting", "search_restaurant", "restaurant_details", "search_menu", "check_availability",
    "create_reservation", "get_reservation", "update_reservation", "cancel_reservation",
    "policy_question", "complaint", "app_issue", "confirm", "deny", "chitchat", "out_of_scope",
]

VALID_SLOTS = [
    "cuisine", "area", "date", "time", "party_size", "budget_max", "dietary",
    "restaurant_id", "booking_code", "customer_name", "phone", "occasion", "rating_min", "note",
]

INT_SLOTS = {"party_size", "budget_max"}

# Tiêu chí tìm kiếm chính: đổi các slot này -> kết quả tìm cũ không còn giá trị
SEARCH_CRITERIA = {"cuisine", "area", "budget_max", "dietary", "occasion", "rating_min"}


class SlotUpdate(BaseModel):
    op: Literal["set", "add", "remove"] = "set"
    value: Optional[Union[int, float, str]] = None


class NLUOutput(BaseModel):
    intents: List[str] = Field(default_factory=list)
    is_followup: bool = False
    slot_updates: Dict[str, SlotUpdate] = Field(default_factory=dict)
    slot_removals: List[str] = Field(default_factory=list)
    confirmation: Optional[Literal["yes", "no"]] = None
    needs_clarification: Optional[str] = None

    @field_validator("intents")
    @classmethod
    def _check_intents(cls, v: List[str]) -> List[str]:
        bad = [i for i in v if i not in VALID_INTENTS]
        if bad:
            raise ValueError(f"intent không hợp lệ: {bad}")
        if not v:
            raise ValueError("intents rỗng")
        return list(dict.fromkeys(v))[:3]

    @field_validator("slot_updates")
    @classmethod
    def _check_slots(cls, v: Dict[str, SlotUpdate]) -> Dict[str, SlotUpdate]:
        bad = [k for k in v if k not in VALID_SLOTS]
        if bad:
            raise ValueError(f"slot không hợp lệ: {bad}")
        return v

    @field_validator("slot_removals")
    @classmethod
    def _check_removals(cls, v: List[str]) -> List[str]:
        bad = [k for k in v if k not in VALID_SLOTS]
        if bad:
            raise ValueError(f"slot_removals không hợp lệ: {bad}")
        return v

    @field_validator("confirmation", mode="before")
    @classmethod
    def _empty_conf(cls, v: Any) -> Any:
        return None if v in ("", "null", "none") else v

    @field_validator("needs_clarification", mode="before")
    @classmethod
    def _empty_clar(cls, v: Any) -> Any:
        return None if v in ("", "null", "none", False) else v


def ollama_json_schema() -> Dict[str, Any]:
    """JSON schema ép đầu ra của LLM (Ollama structured outputs).

    slot_updates dùng additionalProperties thay vì liệt kê từng slot: đo trên qwen2.5:3b,
    schema liệt kê 11 slot khiến model điền bừa slot thừa và gây lỗi "token repeat limit".
    Tên slot được validator Python kiểm tra.
    """
    return {
        "type": "object",
        "properties": {
            "intents": {"type": "array", "items": {"type": "string", "enum": VALID_INTENTS}},
            "is_followup": {"type": "boolean"},
            "slot_updates": {"type": "object", "additionalProperties": {
                "type": "object",
                "properties": {"op": {"type": "string", "enum": ["set", "add", "remove"]},
                               "value": {"type": ["string", "number", "null"]}},
                "required": ["op", "value"]}},
            "slot_removals": {"type": "array", "items": {"type": "string"}},
            "confirmation": {"type": ["string", "null"]},
            "needs_clarification": {"type": ["string", "null"]},
        },
        "required": ["intents", "is_followup", "slot_updates", "slot_removals", "confirmation",
                     "needs_clarification"],
    }


# ------------------------------------------------------------------ chuẩn hóa giá trị slot
def coerce_slot_value(slot: str, value: Any, today: Optional[dt.date] = None) -> Any:
    """Chuẩn hóa/kiểm tra giá trị 1 slot. Trả None nếu không hợp lệ."""
    from core import entities
    from core.normalizer import parse_budget, parse_date, parse_time, _Spans

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if slot == "party_size":
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            m = re.search(r"\d+", str(value))
            if not m:
                return None
            n = int(m.group(0))
        return n if -config.MAX_PARTY_SIZE <= n <= config.MAX_PARTY_SIZE and n != 0 else None
    if slot == "budget_max":
        if isinstance(value, (int, float)):
            n = int(value)
            n = n * 1000 if 0 < n < 1000 else n
            return n if 10_000 <= n <= 50_000_000 else None
        return parse_budget(str(value), _Spans())
    if slot == "date":
        s = str(value).strip()
        try:
            return dt.date.fromisoformat(s).isoformat()
        except ValueError:
            return parse_date(s, today=today)
    if slot == "time":
        s = str(value).strip()
        m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", s)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            return f"{h:02d}:{mi:02d}" if h < 24 and mi < 60 else None
        return parse_time(s)[0]
    if slot == "cuisine":
        return entities.canonical_cuisine(str(value))
    if slot == "area":
        return entities.canonical_area(str(value))
    if slot == "dietary":
        return "chay" if "chay" in str(value).lower() or "vegan" in str(value).lower() else str(value).strip()
    if slot == "restaurant_id":
        hit = entities.canonical_restaurant(str(value))
        return hit[0] if hit else None
    if slot == "booking_code":
        m = re.search(r"DB\d{3,}", str(value).upper())
        return m.group(0) if m else None
    if slot == "rating_min":
        try:
            r = float(str(value).replace(",", ".").split()[0])
        except (ValueError, IndexError):
            return None
        return r if 1 <= r <= 5 else None
    if slot == "occasion":
        return entities.canonical_occasion(str(value))
    if slot == "note":
        return str(value).strip()[:120] or None
    if slot == "phone":
        m = re.search(r"0\d{9,10}", str(value))
        return m.group(0) if m else None
    return str(value).strip()
