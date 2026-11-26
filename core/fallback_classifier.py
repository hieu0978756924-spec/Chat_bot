"""NLU DỰ PHÒNG khi Local LLM không chạy: k-NN embedding trên tập câu mẫu theo intent.

- Ý định: bỏ phiếu có trọng số trên k câu mẫu gần nhất (data/nlu/intent_examples.yaml).
- Slot: lấy từ normalizer (luật chuẩn hóa dữ liệu), không có if/else theo từng câu.
- Câu chỉ chứa thông tin bổ sung (ví dụ "6 người nhé") khi đang có tác vụ dở -> tiếp nối tác vụ đó.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

import config
from core import embeddings
from core.normalizer import NormalizedMessage
from core.schema import NLUOutput, SlotUpdate

logger = logging.getLogger(__name__)

TASK_INTENTS = {"search_restaurant", "check_availability", "create_reservation",
                "update_reservation", "cancel_reservation", "get_reservation",
                "search_menu", "restaurant_details"}
WEAK_SCORE = 0.55
NEGATION_RE = re.compile(r"(?<!\w)(không cần|bỏ|khỏi|không quan trọng|không quan tâm|không nhất thiết|"
                         r"cũng được|cũng ok|sao cũng được|tùy|bất kỳ)(?!\w)")
CRITERION_WORDS = {
    "area": ["gần", "khu", "quận", "chỗ nào", "ở đâu", "lọc theo quận"],
    "budget_max": ["giá", "rẻ", "ngân sách", "tiền", "đắt"],
    "cuisine": ["món", "ăn gì", "loại"],
}


def slots_from_normalizer(nm: NormalizedMessage) -> Tuple[Dict[str, SlotUpdate], List[str]]:
    """Chuyển thực thể do normalizer trích ra thành slot_updates (dùng cho NLU dự phòng)."""
    up: Dict[str, SlotUpdate] = {}
    if nm.date:
        up["date"] = SlotUpdate(op="set", value=nm.date)
    if nm.time:
        up["time"] = SlotUpdate(op="set", value=nm.time)
    if nm.party_size is not None:
        up["party_size"] = SlotUpdate(op="set", value=nm.party_size)
    if nm.party_delta is not None:
        up["party_size"] = SlotUpdate(op="add", value=nm.party_delta)
    if nm.booking_code:
        up["booking_code"] = SlotUpdate(op="set", value=nm.booking_code)
    if nm.budget_max:
        up["budget_max"] = SlotUpdate(op="set", value=nm.budget_max)
    elif nm.budget_low_hint:
        up["budget_max"] = SlotUpdate(op="set", value=config.CHEAP_BUDGET_VND)
    if nm.cuisines:
        up["cuisine"] = SlotUpdate(op="set", value=nm.cuisines[-1])
    if nm.area:
        up["area"] = SlotUpdate(op="set", value=nm.area)
    if nm.restaurant:
        up["restaurant_id"] = SlotUpdate(op="set", value=nm.restaurant[0])
    if nm.phone:
        up["phone"] = SlotUpdate(op="set", value=nm.phone)
    if nm.dietary:
        up["dietary"] = SlotUpdate(op="set", value=nm.dietary)
    if nm.rating_min:
        up["rating_min"] = SlotUpdate(op="set", value=nm.rating_min)
    if nm.occasion:
        up["occasion"] = SlotUpdate(op="set", value=nm.occasion)
    if nm.note:
        up["note"] = SlotUpdate(op="set", value=nm.note)

    removals: List[str] = []
    low = nm.normalized_text.lower()
    if NEGATION_RE.search(low):
        for slot, words in CRITERION_WORDS.items():
            mentioned = any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", low) for w in words)
            if slot == "cuisine":
                mentioned = mentioned or bool(nm.cuisines)
            if slot == "area":
                mentioned = mentioned or bool(nm.area)
            if mentioned:
                removals.append(slot)
                up.pop(slot, None)
    return up, removals


STOP_RESIDUE = {"nhé", "nha", "nhe", "ạ", "à", "ừ", "đi", "thôi", "với", "cho", "mình", "em", "anh", "chị",
                "lúc", "vào", "tầm", "khoảng", "người", "khách", "giờ", "ngày", "mai", "tối", "trưa", "chiều",
                "sáng", "nay", "hôm", "rưỡi", "nhóm", "dưới", "k", "nghìn", "ngàn", "đó", "vậy", "luôn", "ok",
                "được", "quán", "mã", "đơn", "là", "có", "thì", "đổi", "sang", "thành", "thêm", "bớt", "nữa",
                "mấy", "bao", "nhiêu"}


def mask_entities(text: str) -> str:
    """Che thực thể để embedding tập trung vào từ chỉ ý định:
    mã DBxxx -> 'mã đơn', tên nhà hàng -> 'quán', con số -> bỏ."""
    from core import entities
    from core.text import fold

    t = re.sub(r"DB\d+", "mã đơn", text, flags=re.IGNORECASE)
    hit = entities.find_restaurant(t)
    if hit:
        name_f = fold(hit[1])
        words = t.split()
        fw = [fold(w) for w in words]
        n = len(name_f.split())
        for i in range(len(words) - n + 1):
            if " ".join(fw[i:i + n]) == name_f:
                words[i:i + n] = ["quán"]
                break
        t = " ".join(words)
    t = re.sub(r"\d+([:h.,]\d+)?", " ", t)
    return re.sub(r"\s+", " ", t).strip() or text


def residue_words(nm: NormalizedMessage) -> List[str]:
    """Từ còn lại sau khi bỏ thực thể + từ đệm: rỗng => câu chỉ chứa thông tin bổ sung."""
    from core.text import fold

    t = mask_entities(nm.normalized_text).lower()
    for c in nm.cuisines:
        t = t.replace(c.lower(), " ")
    if nm.area:
        t = t.replace(nm.area.lower(), " ")
    stop = {fold(w) for w in STOP_RESIDUE}
    return [w for w in re.findall(r"\w+", t) if fold(w) not in stop and len(w) > 1]


class SemanticFallbackClassifier:
    def __init__(self, path: Optional[str] = None) -> None:
        from core.normalizer import normalize_message

        path = path or os.path.join(config.NLU_DATA_DIR, "intent_examples.yaml")
        with open(path, encoding="utf-8") as f:
            data: Dict[str, List[str]] = yaml.safe_load(f)
        self.labels: List[str] = []
        self.texts: List[str] = []
        for label, sents in data.items():
            for sent in sents:
                self.labels.append(label)
                self.texts.append(mask_entities(normalize_message(str(sent)).normalized_text))
        self.matrix = embeddings.encode(self.texts)
        self.available = self.matrix is not None

    def rank(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Xếp hạng nhãn theo độ tương đồng với câu mẫu GẦN NHẤT của từng nhãn (1-NN theo nhãn)."""
        if not self.available:
            return []
        q = embeddings.encode([mask_entities(text)])
        sims = (self.matrix @ q.T).ravel()
        best: Dict[str, float] = {}
        for lab, sc in zip(self.labels, sims):
            if sc > best.get(lab, -1.0):
                best[lab] = float(sc)
        return sorted(best.items(), key=lambda kv: -kv[1])[:k]

    def classify(self, nm: NormalizedMessage, state: Dict[str, Any]) -> Tuple[NLUOutput, Dict[str, Any]]:
        text = nm.normalized_text
        ranked = self.rank(text)
        label, score = ranked[0] if ranked else ("out_of_scope", 0.0)
        info = {"top": ranked[:3], "help": False}

        updates, removals = slots_from_normalizer(nm)
        active_task = state.get("active_task")
        pending = state.get("pending_action")
        n_words = len(text.split())

        intent = label
        # điểm gần bằng nhau -> ưu tiên nhãn của thao tác đang chờ / việc đang làm (tiên nghiệm ngữ cảnh)
        prior = pending or active_task
        if prior and ranked:
            for lab, sc in ranked[1:]:
                if lab == prior and sc >= score - 0.03:
                    intent = lab
                    break
        if label == "help":
            intent, info["help"] = "chitchat", True

        confirmation = None
        if pending or state.get("awaiting_resume"):
            # đang chờ xác nhận: câu ngắn gần nhất với confirm/deny -> trả lời xác nhận
            cd = [(lab, sc) for lab, sc in ranked if lab in ("confirm", "deny")]
            if cd and n_words <= 6 and cd[0][1] >= 0.6 and not updates:
                intent = cd[0][0]
                confirmation = "yes" if intent == "confirm" else "no"
        elif intent in ("confirm", "deny") and active_task and (updates or removals):
            intent = active_task

        is_followup = False
        weak = score < WEAK_SCORE
        slot_only = (updates or removals) and not residue_words(nm)
        if (updates or removals or nm.ordinal_ref or nm.demonstrative_ref) and active_task:
            if slot_only or weak or intent in ("confirm", "deny", "chitchat", "greeting", "out_of_scope"):
                intent, is_followup = active_task, True
            elif intent in TASK_INTENTS:
                is_followup = True
        if removals and active_task in ("search_restaurant", "check_availability"):
            intent, is_followup = active_task, True

        out = NLUOutput(intents=[intent], is_followup=is_followup, slot_updates=updates,
                        slot_removals=removals, confirmation=confirmation)
        info["score"] = round(score, 3)
        return out, info


_INSTANCE: Optional[SemanticFallbackClassifier] = None


def get_fallback_classifier() -> SemanticFallbackClassifier:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SemanticFallbackClassifier()
    return _INSTANCE
