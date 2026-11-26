"""Trạng thái hội thoại theo session_id và quy tắc merge slot (mục 2.2).

Quy tắc:
- set/add/remove theo NLU; chỉ đổi 1 slot thì giữ nguyên các slot khác.
- Đổi tiêu chí tìm kiếm chính (cuisine/area/budget/dietary) -> giữ ngày/giờ/số người,
  xóa restaurant_id và last_search_results cũ.
- "quán thứ 2", "quán đó" -> ánh xạ theo last_search_results (mơ hồ -> cờ để hỏi lại).
- Hỏi chính sách / khiếu nại / lỗi app giữa chừng -> đẩy việc đang dở vào task_stack.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

import config
from core.normalizer import NormalizedMessage
from core.schema import SEARCH_CRITERIA, VALID_SLOTS, NLUOutput

TASK_INTENTS = {"search_restaurant", "check_availability", "create_reservation", "update_reservation",
                "cancel_reservation", "get_reservation", "search_menu", "restaurant_details"}
INTERRUPT_INTENTS = {"policy_question", "complaint", "app_issue"}
SEARCH_FLOW_INTENTS = {"search_restaurant", "check_availability", "create_reservation", "update_reservation",
                       "confirm", "deny", "chitchat", "greeting"}


def empty_slots() -> Dict[str, Any]:
    return {s: None for s in VALID_SLOTS}


class ConversationState(BaseModel):
    session_id: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    turn: int = 0
    current_intent: Optional[str] = None
    previous_intent: Optional[str] = None
    active_task: Optional[str] = None
    slots: Dict[str, Any] = Field(default_factory=empty_slots)
    restaurant_name: Optional[str] = None
    slot_meta: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    last_search_results: List[Dict[str, Any]] = Field(default_factory=list)
    pending_action: Optional[Dict[str, Any]] = None
    task_stack: List[Dict[str, Any]] = Field(default_factory=list)
    awaiting_slots: List[str] = Field(default_factory=list)
    awaiting_resume: bool = False
    last_booking_code: Optional[str] = None
    last_ticket_code: Optional[str] = None

    def filled_slots(self) -> Dict[str, Any]:
        return {k: v for k, v in self.slots.items() if v is not None}

    def nlu_view(self) -> Dict[str, Any]:
        """Phần trạng thái gửi cho NLU (gọn)."""
        return {
            "slots": self.filled_slots(),
            "active_task": self.active_task,
            "pending_action": self.pending_action["type"] if self.pending_action else None,
            "awaiting_resume": self.awaiting_resume,
            "awaiting_slots": list(self.awaiting_slots),
            "last_results": [f"{i + 1}. {r['ten_nha_hang']} ({r['ma_nha_hang']})"
                             for i, r in enumerate(self.last_search_results)],
        }

    def debug_view(self) -> Dict[str, Any]:
        return {
            "turn": self.turn, "current_intent": self.current_intent, "previous_intent": self.previous_intent,
            "active_task": self.active_task, "slots": self.filled_slots(), "restaurant_name": self.restaurant_name,
            "last_search_results": [r["ma_nha_hang"] for r in self.last_search_results],
            "pending_action": self.pending_action, "task_stack": [t.get("task") for t in self.task_stack],
            "awaiting_slots": self.awaiting_slots, "awaiting_resume": self.awaiting_resume,
        }


class MergeResult(BaseModel):
    changed: Dict[str, Any] = Field(default_factory=dict)      # slot -> giá trị mới (None = bị xóa)
    criteria_changed: bool = False
    reference: Optional[str] = None                             # "resolved" | "ambiguous" | "out_of_range"
    interrupted_task: Optional[str] = None


class StateManager:
    def __init__(self, session_id: str = "default") -> None:
        self.state = ConversationState(session_id=session_id)

    def reset(self) -> None:
        self.state = ConversationState(session_id=self.state.session_id)

    # ------------------------------------------------------------------ merge
    def apply(self, nlu: NLUOutput, nm: NormalizedMessage) -> MergeResult:
        st = self.state
        st.turn += 1
        st.history.append({"role": "user", "text": nm.raw_text})
        st.previous_intent = st.current_intent
        st.current_intent = nlu.intents[0]
        res = MergeResult()

        # task_stack: câu hỏi xen ngang khi đang có việc dở
        if st.current_intent in INTERRUPT_INTENTS and st.active_task and (
                st.pending_action or st.filled_slots()):
            st.task_stack.append({"task": st.active_task, "pending_action": st.pending_action,
                                  "turn": st.turn})
            st.pending_action = None
            res.interrupted_task = st.active_task

        before = copy.deepcopy(st.slots)
        for slot in nlu.slot_removals:
            self._set(slot, None, "remove")
        for slot, up in nlu.slot_updates.items():
            if up.op == "remove":
                self._set(slot, None, "remove")
            elif up.op == "add" and slot == "party_size":
                base = st.slots.get("party_size") or 0
                self._set(slot, max(1, min(config.MAX_PARTY_SIZE, base + int(up.value))), "add")
            else:
                self._set(slot, up.value, "set")

        # đổi loại món sang món không chay -> bỏ yêu cầu chay cũ (nếu câu này không nhắc lại)
        if (before.get("cuisine") == "Chay" and st.slots.get("cuisine") not in (None, "Chay")
                and "dietary" not in nlu.slot_updates and st.slots.get("dietary") == "chay"):
            self._set("dietary", None, "auto_clear")

        res.changed = {k: st.slots[k] for k in VALID_SLOTS if st.slots.get(k) != before.get(k)}
        res.criteria_changed = any(k in SEARCH_CRITERIA for k in res.changed)
        if "restaurant_id" in res.changed and st.slots.get("restaurant_id"):
            self._sync_restaurant_name()
        elif res.criteria_changed and st.current_intent in SEARCH_FLOW_INTENTS:
            # hỏi thực đơn / thông tin quán thì GIỮ danh sách quán vừa tìm để lọc theo đó
            st.slots["restaurant_id"] = None
            st.restaurant_name = None
            st.last_search_results = []

        res.reference = self._resolve_reference(nm)
        if res.reference == "resolved":
            res.changed["restaurant_id"] = st.slots["restaurant_id"]
        if st.slots.get("booking_code"):
            st.last_booking_code = st.slots["booking_code"]
        return res

    def _set(self, slot: str, value: Any, source: str) -> None:
        self.state.slots[slot] = value
        self.state.slot_meta[slot] = {"value": value, "turn_set": self.state.turn, "source": source}

    def _sync_restaurant_name(self) -> None:
        from core import entities

        rid = self.state.slots.get("restaurant_id")
        for r in self.state.last_search_results:
            if r["ma_nha_hang"] == rid:
                self.state.restaurant_name = r["ten_nha_hang"]
                return
        hit = entities.find_restaurant(rid or "")
        self.state.restaurant_name = hit[1] if hit else None

    def _resolve_reference(self, nm: NormalizedMessage) -> Optional[str]:
        st = self.state
        results = st.last_search_results
        if nm.ordinal_ref:
            if not results:
                return "out_of_range"
            idx = len(results) - 1 if nm.ordinal_ref == -1 else nm.ordinal_ref - 1
            if not 0 <= idx < len(results):
                return "out_of_range"
            self._set("restaurant_id", results[idx]["ma_nha_hang"], "reference")
            st.restaurant_name = results[idx]["ten_nha_hang"]
            return "resolved"
        if nm.demonstrative_ref and not nm.restaurant:
            if st.slots.get("restaurant_id"):
                return None
            if len(results) == 1:
                self._set("restaurant_id", results[0]["ma_nha_hang"], "reference")
                st.restaurant_name = results[0]["ten_nha_hang"]
                return "resolved"
            if len(results) > 1:
                return "ambiguous"
        return None

    # ------------------------------------------------------------------ tiện ích cho policy
    def set_results(self, restaurants: List[Dict[str, Any]]) -> None:
        self.state.last_search_results = [{"ma_nha_hang": r["ma_nha_hang"], "ten_nha_hang": r["ten_nha_hang"]}
                                          for r in restaurants]

    def set_pending(self, action_type: str, params: Dict[str, Any], summary: Dict[str, Any]) -> None:
        self.state.pending_action = {"type": action_type, "params": params, "summary": summary,
                                     "expires_at": time.time() + config.PENDING_ACTION_TTL_S}

    def pending_expired(self) -> bool:
        p = self.state.pending_action
        return bool(p and time.time() > p["expires_at"])

    def record_bot(self, text: str) -> None:
        self.state.history.append({"role": "bot", "text": text})
