"""Bộ điều phối pipeline thống nhất cho mỗi tin nhắn (không rẽ nhánh ngầm):

User message
 -> Normalizer -> NLU (LLM + validator | dự phòng embedding) -> State Manager (merge)
 -> Dialogue Policy -> Tool Executor -> Response Generator
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from actions import db
from agent.tools import ToolExecutor
from core.nlu import OllamaNLU
from core.normalizer import normalize_message
from core.policy import DialoguePolicy
from core.response_gen import render
from core.state import StateManager

logger = logging.getLogger(__name__)


class RestaurantAgent:
    def __init__(self, session_id: str = "default", debug: bool = False, warmup: bool = True,
                 nlu: Optional[OllamaNLU] = None) -> None:
        db.ensure_database()
        self.session_id = session_id
        self.debug = debug
        self.sm = StateManager(session_id)
        self.tools = ToolExecutor()
        self.nlu = nlu or OllamaNLU()
        self.policy = DialoguePolicy(self.sm, self.tools)
        self.last_trace: Dict[str, Any] = {}
        self._notice_active = False
        if warmup:
            self.nlu.warmup()

    @property
    def state(self):
        return self.sm.state

    def greeting(self) -> str:
        return render([{"type": "greeting_full"}])

    def reset(self) -> None:
        self.sm.reset()
        self.last_trace = {}

    def ask(self, message: str) -> str:
        t0 = time.time()
        text = (message or "").strip()
        if not text:
            return "Dạ bạn cần mình hỗ trợ gì ạ?"

        nm = normalize_message(text)
        history = list(self.sm.state.history)
        nlu, meta = self.nlu.analyze(nm, self.sm.state.nlu_view(), history)
        merge = self.sm.apply(nlu, nm)
        self.tools.trace = []
        blocks = self.policy.decide(nlu, nm, merge, meta)

        if meta.get("llm_error") == "LLM_UNAVAILABLE" and meta["source"] == "fallback_embedding":
            if not self._notice_active:
                blocks.insert(0, {"type": "llm_notice"})
                self._notice_active = True
        elif meta.get("source", "").startswith("llm"):
            self._notice_active = False

        reply = render(blocks)
        self.sm.record_bot(reply)
        self.last_trace = {
            "input": text, "normalized": nm.normalized_text, "hints": nm.hints(),
            "nlu_source": meta.get("source"), "llm_error": meta.get("llm_error"),
            "llm_latency_s": meta.get("llm_latency_s"), "nlu_latency_s": meta.get("latency_s"),
            "dropped_slots": meta.get("dropped_slots"), "fallback_top": meta.get("fallback_top"),
            "intents": nlu.intents, "is_followup": nlu.is_followup,
            "slot_updates": {k: {"op": v.op, "value": v.value} for k, v in nlu.slot_updates.items()},
            "slot_removals": nlu.slot_removals, "confirmation": nlu.confirmation,
            "needs_clarification": nlu.needs_clarification,
            "changed": merge.changed, "state": self.sm.state.debug_view(),
            "tools": list(self.tools.trace), "blocks": [b["type"] for b in blocks],
            "total_s": round(time.time() - t0, 2),
        }
        logger.info("turn %s", json.dumps(self.last_trace, ensure_ascii=False, default=str))
        if self.debug:
            print(format_trace(self.last_trace))
        return reply


def format_trace(t: Dict[str, Any]) -> str:
    def j(x: Any) -> str:
        return json.dumps(x, ensure_ascii=False, default=str)

    lines = ["┌─ DEBUG " + "─" * 60,
             f"│ chuẩn hóa : {t['normalized']}   gợi ý={j(t['hints'])}",
             f"│ NLU       : nguồn={t['nlu_source']} lỗi_LLM={t['llm_error']} "
             f"LLM={t['llm_latency_s']}s NLU={t['nlu_latency_s']}s",
             f"│ intents   : {t['intents']} followup={t['is_followup']} xác_nhận={t['confirmation']}",
             f"│ slot_upd  : {j(t['slot_updates'])} xóa={t['slot_removals']}"]
    if t.get("dropped_slots"):
        lines.append(f"│ bị loại   : {t['dropped_slots']}")
    if t.get("needs_clarification"):
        lines.append(f"│ hỏi lại   : {t['needs_clarification']}")
    lines.append(f"│ state     : {j(t['state'])}")
    for c in t["tools"]:
        lines.append(f"│ tool      : {c['tool']}({j(c['params'])}) -> ok={c['ok']} {c['error_code'] or ''} {c['ms']}ms")
    lines.append(f"│ blocks    : {t['blocks']}   tổng={t['total_s']}s")
    lines.append("└" + "─" * 68)
    return "\n".join(lines)


class SessionStore:
    """Mỗi session_id có một agent/state riêng."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self._agents: Dict[str, RestaurantAgent] = {}
        self._nlu: Optional[OllamaNLU] = None

    def get(self, session_id: str) -> RestaurantAgent:
        if session_id not in self._agents:
            first = self._nlu is None
            self._nlu = self._nlu or OllamaNLU()
            self._agents[session_id] = RestaurantAgent(session_id, self.debug, warmup=first, nlu=self._nlu)
        return self._agents[session_id]
