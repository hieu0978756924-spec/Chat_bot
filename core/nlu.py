"""NLU bằng Local LLM (Ollama) + validator + retry + NLU dự phòng (embedding).

Luồng: prompt (system TĨNH để Ollama tái dùng KV-cache, phần động nằm trong tin nhắn user)
 -> Ollama /api/chat với format = JSON schema, temperature thấp
 -> Pydantic validate (sai -> retry 1 lần kèm thông báo lỗi)
 -> vẫn sai / Ollama lỗi / timeout -> NLU dự phòng, log rõ LLM_UNAVAILABLE / LLM_INVALID_OUTPUT
 -> hậu xử lý chung: chuẩn hóa giá trị, đối chiếu với normalizer, chặn giá trị không có căn cứ,
    phát hiện câu mơ hồ ("đổi sang 8").
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydantic import ValidationError

import config
from core.fallback_classifier import get_fallback_classifier, slots_from_normalizer
from core.normalizer import NormalizedMessage, parse_time
from core.schema import NLUOutput, SlotUpdate, coerce_slot_value, ollama_json_schema

logger = logging.getLogger(__name__)

WEEKDAY_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

SYSTEM_PROMPT = """Bạn là bộ phân tích ngôn ngữ (NLU) của trợ lý đặt bàn nhà hàng tiếng Việt.
Nhiệm vụ: đọc TIN NHẮN MỚI cùng ngữ cảnh, trả về DUY NHẤT một JSON theo schema. Không trả lời khách.

INTENTS:
greeting (chào) | search_restaurant (tìm/gợi ý quán, kể cả khi đổi tiêu chí tìm) | restaurant_details (hỏi địa chỉ, giờ mở cửa, đánh giá 1 quán) |
search_menu (hỏi món, giá món, thực đơn, "có món gì rẻ/chay") | check_availability (hỏi còn bàn/chỗ không) | create_reservation (muốn đặt bàn, chọn quán để đặt) |
get_reservation (tra cứu đơn DBxxx) | update_reservation (sửa đơn ĐÃ ĐẶT có mã) | cancel_reservation (hủy đơn đã đặt) |
policy_question (quy định: cọc, hủy, đến trễ, phụ thu, mang đồ ngoài, trẻ em...) | complaint (phàn nàn, sự cố tại quán) |
app_issue (lỗi ứng dụng/website/tin nhắn) | confirm (đồng ý khi hệ thống đang chờ xác nhận) | deny (từ chối/khoan khi đang chờ xác nhận) |
chitchat (CHỈ: cảm ơn, tạm biệt, khen, hỏi bot là ai/làm được gì) | out_of_scope (mọi yêu cầu khác không liên quan nhà hàng/đặt bàn: viết bài, nấu ăn, thời tiết, tin tức...).
Câu hỏi về quy định/phí/cọc/trẻ em/gửi xe/đoàn đông là policy_question KỂ CẢ khi đang tìm quán hay đặt bàn.
Khách kể sự cố đã xảy ra (đến nơi không có bàn, chờ lâu, tính sai tiền) là complaint, không phải check_availability.
cancel_reservation CHỈ khi khách yêu cầu hủy đơn; hỏi phí hủy là policy_question; nhắc mã đơn khi phàn nàn vẫn là complaint.
Một câu có thể có tối đa 3 intents, theo thứ tự xuất hiện.

SLOTS: cuisine (Lẩu, Nướng, Nhật Bản, Hàn Quốc, Hải sản, Pizza, Chay, Món Việt, Cà phê, Âu, Á, Ý, Thái, Trung Hoa, Steak) | area (vd "Quận 1", "Tân Bình") |
date (YYYY-MM-DD) | time (HH:MM, 24h) | party_size (số nguyên) | budget_max (VNĐ, số nguyên; "rẻ/bình dân" = 150000) |
dietary ("chay" = quán CÓ món chay; còn "quán chay" là cuisine Chay) | restaurant_id (NHxxx) | booking_code (DBxxx) |
occasion (gia đình, hẹn hò, nhóm bạn, nhóm đông, gặp gỡ) | rating_min (số sao tối thiểu) | note (yêu cầu về bàn, vd "gần cửa sổ").

QUY TẮC:
1. slot_updates CHỈ chứa slot được nhắc trong TIN NHẮN MỚI. Không chép lại slot đã có trong trạng thái.
2. op: "set" = gán/ghi đè; "add" = cộng/trừ ("thêm 2 người" -> add 2, "bớt 1" -> add -1); "remove" = bỏ tiêu chí.
3. Bỏ tiêu chí ("không cần gần nhà", "bao nhiêu tiền cũng ok") -> đưa tên slot vào slot_removals.
4. Câu chỉ sửa/bổ sung thông tin cho việc đang làm -> is_followup=true và giữ intent của việc đang làm
   (đang tìm quán mà đổi số người -> search_restaurant, KHÔNG phải update_reservation khi chưa có mã đơn).
5. Đổi loại món (lẩu -> chay) là tìm kiếm mới: search_restaurant, set cuisine mới, KHÔNG nhắc lại ngày/giờ/số người.
6. Ngày/giờ: dùng "Gợi ý chuẩn hóa" nếu có; không tự bịa ngày giờ khi khách không nói.
7. Số không rõ đơn vị (vd "chuyển qua 9" có thể là 9 giờ hoặc 9 người) -> needs_clarification là câu hỏi lại ngắn, không set slot đó.
8. confirmation chỉ khác null khi hệ thống ĐANG CHỜ XÁC NHẬN: "yes" hoặc "no".
9. "quán thứ 2", "quán đó" -> hệ thống tự ánh xạ; chỉ cần đúng intent.

VÍ DỤ (ngữ cảnh -> JSON):
[Trạng thái: {}] "Cuối tuần có chỗ nào nướng ngon ở Tân Bình không?"
{"intents":["search_restaurant"],"is_followup":false,"slot_updates":{"cuisine":{"op":"set","value":"Nướng"},"area":{"op":"set","value":"Tân Bình"}},"slot_removals":[],"confirmation":null,"needs_clarification":null}
[Trạng thái: {"cuisine":"Nướng","party_size":4}, việc đang làm: search_restaurant] "cho thêm 2 bạn nữa"
{"intents":["search_restaurant"],"is_followup":true,"slot_updates":{"party_size":{"op":"add","value":2}},"slot_removals":[],"confirmation":null,"needs_clarification":null}
[Trạng thái: {"cuisine":"Nướng","area":"Tân Bình","date":"2026-01-10"}] "thôi đổi qua đồ Nhật đi"
{"intents":["search_restaurant"],"is_followup":true,"slot_updates":{"cuisine":{"op":"set","value":"Nhật Bản"}},"slot_removals":[],"confirmation":null,"needs_clarification":null}
[Trạng thái: {"cuisine":"Món Việt","area":"Quận 3"}] "khu nào cũng được, không cần quận 3"
{"intents":["search_restaurant"],"is_followup":true,"slot_updates":{},"slot_removals":["area"],"confirmation":null,"needs_clarification":null}
[Trạng thái: {"time":"19:00","party_size":4}, việc đang làm: check_availability] "đổi giúp mình sang 9"
{"intents":["check_availability"],"is_followup":true,"slot_updates":{},"slot_removals":[],"confirmation":null,"needs_clarification":"Dạ bạn muốn đổi sang 9 giờ hay 9 người ạ?"}
[Đang chờ xác nhận: create_reservation] "ừ chốt vậy đi"
{"intents":["confirm"],"is_followup":true,"slot_updates":{},"slot_removals":[],"confirmation":"yes","needs_clarification":null}
[Đang chờ xác nhận: cancel_reservation] "khoan, để mình hỏi lại vợ đã"
{"intents":["deny"],"is_followup":true,"slot_updates":{},"slot_removals":[],"confirmation":"no","needs_clarification":null}
[Trạng thái: {}] "tim quan nuong quan 7, ma quan co cho mang ruou vao khong"
{"intents":["search_restaurant","policy_question"],"is_followup":false,"slot_updates":{"cuisine":{"op":"set","value":"Nướng"},"area":{"op":"set","value":"Quận 7"}},"slot_removals":[],"confirmation":null,"needs_clarification":null}
[Trạng thái: {}] "đơn DB012 dời giúp mình qua thứ 6, 6 giờ chiều"
{"intents":["update_reservation"],"is_followup":false,"slot_updates":{"booking_code":{"op":"set","value":"DB012"},"time":{"op":"set","value":"18:00"}},"slot_removals":[],"confirmation":null,"needs_clarification":null}
[Trạng thái: {"cuisine":"Pizza","party_size":3}, việc đang làm: search_restaurant] "à cho hỏi đến muộn thì có bị hủy bàn không"
{"intents":["policy_question"],"is_followup":false,"slot_updates":{},"slot_removals":[],"confirmation":null,"needs_clarification":null}
[Trạng thái: {}] "mình đưa mã DB077 mà lễ tân bảo hệ thống không ghi nhận, giờ đứng ngoài cửa"
{"intents":["complaint"],"is_followup":false,"slot_updates":{"booking_code":{"op":"set","value":"DB077"}},"slot_removals":[],"confirmation":null,"needs_clarification":null}
[Trạng thái: {}] "hôm qua tới quán chờ mãi mà không ai dẫn vào bàn"
{"intents":["complaint"],"is_followup":false,"slot_updates":{},"slot_removals":[],"confirmation":null,"needs_clarification":null}
[Đang chờ xác nhận: create_reservation] "chuẩn rồi em"
{"intents":["confirm"],"is_followup":true,"slot_updates":{},"slot_removals":[],"confirmation":"yes","needs_clarification":null}
[Trạng thái: {"cuisine":"Pizza"}, có 3 kết quả tìm] "lấy quán số 3 nha"
{"intents":["create_reservation"],"is_followup":true,"slot_updates":{},"slot_removals":[],"confirmation":null,"needs_clarification":null}
[Trạng thái: {}] "bấm thanh toán cứ báo lỗi hoài"
{"intents":["app_issue"],"is_followup":false,"slot_updates":{},"slot_removals":[],"confirmation":null,"needs_clarification":null}
[Trạng thái: {}] "tối nay đá bóng mấy giờ"
{"intents":["out_of_scope"],"is_followup":false,"slot_updates":{},"slot_removals":[],"confirmation":null,"needs_clarification":null}
[Trạng thái: {}] "soạn giúp mình email xin nghỉ phép"
{"intents":["out_of_scope"],"is_followup":false,"slot_updates":{},"slot_removals":[],"confirmation":null,"needs_clarification":null}"""


def _fmt_today(now: dt.datetime) -> str:
    return f"{now.strftime('%Y-%m-%d')} ({WEEKDAY_VI[now.weekday()]}), {now.strftime('%H:%M')}"


def build_user_prompt(nm: NormalizedMessage, state: Dict[str, Any], history: List[Dict[str, str]],
                      now: Optional[dt.datetime] = None) -> str:
    now = now or dt.datetime.now()
    lines = [f"Hôm nay: {_fmt_today(now)}"]
    lines.append(f"Trạng thái: {json.dumps(state.get('slots', {}), ensure_ascii=False)}")
    if state.get("active_task"):
        lines.append(f"Việc đang làm: {state['active_task']}")
    if state.get("pending_action"):
        lines.append(f"Đang chờ xác nhận: {state['pending_action']}")
    elif state.get("awaiting_resume"):
        lines.append("Đang chờ xác nhận: có muốn tiếp tục việc đang dở không")
    if state.get("awaiting_slots"):
        lines.append(f"Bot vừa hỏi các slot: {state['awaiting_slots']}")
    if state.get("last_results"):
        lines.append("Kết quả tìm gần nhất: " + "; ".join(state["last_results"]))
    if history:
        lines.append("Hội thoại gần đây:")
        for h in history[-2 * config.HISTORY_TURNS_FOR_NLU:]:
            who = "Khách" if h["role"] == "user" else "Bot"
            text = h["text"].replace("\n", " ")
            lines.append(f"  {who}: {text[:160]}{'…' if len(text) > 160 else ''}")
    hints = nm.hints()
    if hints:
        lines.append(f"Gợi ý chuẩn hóa (luật, đáng tin): {json.dumps(hints, ensure_ascii=False)}")
    lines.append(f'TIN NHẮN MỚI: "{nm.raw_text}"')
    if nm.normalized_text != nm.raw_text:
        lines.append(f'(đã chuẩn hóa: "{nm.normalized_text}")')
    lines.append("JSON:")
    return "\n".join(lines)


class OllamaNLU:
    COOLDOWN_S = 60

    def __init__(self) -> None:
        self.fallback = get_fallback_classifier()
        self._down_until = 0.0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ Ollama
    def _payload(self, messages: List[Dict[str, str]], num_predict: int = 300) -> Dict[str, Any]:
        return {
            "model": config.ollama_model(),
            "messages": messages,
            "stream": False,
            "format": ollama_json_schema(),
            "keep_alive": config.ollama_keep_alive(),
            "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": config.ollama_num_ctx(),
                        "num_predict": num_predict},
        }

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Lỗi kết nối/timeout -> ném ConnectionError (LLM không dùng được).
        HTTP lỗi do quá trình sinh (vd "token repeat limit") -> trả chuỗi rỗng để validator retry."""
        try:
            r = requests.post(f"{config.ollama_url()}/api/chat", json=self._payload(messages),
                              timeout=config.ollama_timeout())
        except requests.RequestException as e:
            raise ConnectionError(str(e)) from e
        if r.status_code == 404:
            raise ConnectionError(f"Model {config.ollama_model()} chưa được tải: {r.text[:200]}")
        if r.status_code != 200:
            logger.warning("LLM_GENERATION_ERROR HTTP %s: %s", r.status_code, r.text[:200])
            return ""
        return r.json().get("message", {}).get("content", "")

    def warmup(self) -> None:
        """Nạp model + cache phần system prompt (chạy nền lúc khởi động)."""
        if config.llm_disabled():
            return

        def run() -> None:
            try:
                msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": 'TIN NHẮN MỚI: "xin chào"\nJSON:'}]
                p = self._payload(msgs, num_predict=1)
                requests.post(f"{config.ollama_url()}/api/chat", json=p, timeout=600)
            except Exception as e:  # pragma: no cover
                logger.warning("LLM_WARMUP_FAILED: %s", e)

        threading.Thread(target=run, daemon=True).start()

    def llm_available(self) -> bool:
        return not config.llm_disabled() and time.time() >= self._down_until

    # ------------------------------------------------------------------ main
    def analyze(self, nm: NormalizedMessage, state: Dict[str, Any],
                history: List[Dict[str, str]]) -> Tuple[NLUOutput, Dict[str, Any]]:
        t0 = time.time()
        meta: Dict[str, Any] = {"source": None, "llm_error": None, "llm_latency_s": None}
        out: Optional[NLUOutput] = None

        if self.llm_available():
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(nm, state, history)}]
            for attempt in (1, 2):
                try:
                    raw = self._chat(messages)
                except ConnectionError as e:
                    meta["llm_error"] = "LLM_UNAVAILABLE"
                    self.last_error = str(e)
                    self._down_until = time.time() + self.COOLDOWN_S
                    logger.warning("LLM_UNAVAILABLE (%s): %s", config.ollama_model(), e)
                    break
                try:
                    out = NLUOutput.model_validate(json.loads(raw))
                    meta["source"] = "llm" if attempt == 1 else "llm_retry"
                    break
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning("LLM_INVALID_OUTPUT (lần %d): %s | raw=%s", attempt, e, raw[:300])
                    meta["llm_error"] = "LLM_INVALID_OUTPUT"
                    messages = messages + [
                        {"role": "assistant", "content": raw[:1500]},
                        {"role": "user", "content": f"JSON sai schema: {str(e)[:300]}. Trả lại JSON đúng schema."},
                    ]
            meta["llm_latency_s"] = round(time.time() - t0, 2)
        elif config.llm_disabled():
            meta["llm_error"] = "LLM_DISABLED"
        else:
            meta["llm_error"] = "LLM_UNAVAILABLE"

        info: Dict[str, Any] = {}
        if out is None:
            out, info = self.fallback.classify(nm, state)
            meta["source"] = "fallback_embedding"
            meta["fallback_top"] = info.get("top")
        elif (state.get("pending_action") or state.get("awaiting_resume")):
            out = self._confirm_ensemble(out, nm, meta)
        meta["help"] = info.get("help", False)
        if meta["source"] != "fallback_embedding" and out.intents[0] == "chitchat":
            ranked = self.fallback.rank(nm.normalized_text, k=3)
            meta["help"] = bool(ranked and ranked[0][0] == "help" and ranked[0][1] >= 0.6)

        out = postprocess(out, nm, state, meta)
        meta["latency_s"] = round(time.time() - t0, 2)
        return out, meta


    def _confirm_ensemble(self, out: NLUOutput, nm: NormalizedMessage, meta: Dict[str, Any]) -> NLUOutput:
        """Đang chờ xác nhận + câu ngắn không có thông tin mới: kết hợp LLM với bộ phân loại
        embedding xác nhận/từ chối (LLM 3B hay lặp lại intent của việc đang chờ)."""
        if out.intents[0] in ("confirm", "deny"):
            conf = out.confirmation or ("yes" if out.intents[0] == "confirm" else "no")
            return out.model_copy(update={"confirmation": conf})
        norm_updates, _ = slots_from_normalizer(nm)
        if norm_updates or len(nm.normalized_text.split()) > 8:
            return out
        ranked = [(l, sc) for l, sc in self.fallback.rank(nm.normalized_text, k=20) if l in ("confirm", "deny")]
        if ranked and ranked[0][1] >= 0.6:
            lab = ranked[0][0]
            meta["intent_adjusted"] = f"confirm_ensemble:{lab}:{ranked[0][1]:.2f}"
            return out.model_copy(update={"intents": [lab], "confirmation": "yes" if lab == "confirm" else "no"})
        return out


# ---------------------------------------------------------------------- hậu xử lý chung
GROUNDED_SLOTS = ("date", "time", "party_size", "booking_code", "budget_max", "phone", "rating_min", "note")
REMOVABLE_SLOTS = ("cuisine", "area", "budget_max", "dietary", "restaurant_id", "occasion", "rating_min", "note")
import re as _re
MENU_Q = _re.compile(r"(?<!\w)(món\s+(?:\w+\s+){0,2}?(?:gì|nào)|những món|thực đơn|menu|các món)(?!\w)")
INTENTS_SELECTING_RESULT = ("search_restaurant", "check_availability", "restaurant_details", "chitchat")


def _neg_cue(nm: NormalizedMessage) -> bool:
    from core.fallback_classifier import NEGATION_RE as _NEG
    return bool(_NEG.search(nm.normalized_text.lower()))


def postprocess(out: NLUOutput, nm: NormalizedMessage, state: Dict[str, Any],
                meta: Dict[str, Any]) -> NLUOutput:
    slots_now = state.get("slots", {})
    dropped: List[str] = []
    fixed: Dict[str, SlotUpdate] = {}

    # 1) chuẩn hóa giá trị từng slot do NLU đưa ra
    for name, up in out.slot_updates.items():
        if up.op == "remove":
            if name in REMOVABLE_SLOTS and _neg_cue(nm):
                fixed[name] = up
            else:
                dropped.append(f"xóa {name} (không hợp lệ)")
            continue
        val = coerce_slot_value(name, up.value)
        if val is None:
            dropped.append(f"{name}={up.value!r} (không hợp lệ)")
            continue
        fixed[name] = SlotUpdate(op=up.op if name == "party_size" else "set", value=val)

    # 2) đối chiếu với normalizer: thực thể số/ngày/giờ do luật trích ra là nguồn tin cậy
    norm_updates, norm_removals = slots_from_normalizer(nm)
    for name in GROUNDED_SLOTS:
        if name in norm_updates:
            fixed[name] = norm_updates[name]
        elif name in fixed and fixed[name].op != "remove":
            # LLM đưa giá trị nhưng câu không hề nhắc -> chỉ chấp nhận nếu là số trơ đang chờ (xử lý ở bước 3)
            dropped.append(f"{name}={fixed[name].value!r} (không có căn cứ trong câu)")
            fixed.pop(name)
    # tiêu chí tìm kiếm do LLM đưa ra phải có căn cứ trong câu (LLM 3B hay "điền" tiêu chí từ few-shot)
    from core.text import fold as _fold
    from core import entities as _ent
    raw_s, clean_s = nm.raw_text, nm.normalized_text
    if nm.restaurant:   # tên quán ("Lẩu Biển Quy Nhơn") không phải căn cứ cho loại món / khu vực
        raw_s = _ent.strip_restaurant_name(raw_s, nm.restaurant[1])
        clean_s = _ent.strip_restaurant_name(clean_s, nm.restaurant[1])
    msg_f = _fold(raw_s + " " + clean_s)
    for name in ("cuisine", "area", "occasion", "dietary", "rating_min", "restaurant_id"):
        up = fixed.get(name)
        if up is None or up.op == "remove" or name in norm_updates:
            continue
        val_f = _fold(str(up.value))
        if name == "restaurant_id" or val_f not in msg_f:
            dropped.append(f"{name}={up.value!r} (không có căn cứ trong câu)")
            fixed.pop(name)
    for name in ("cuisine", "area", "restaurant_id", "dietary", "occasion"):
        # thực thể có căn cứ trong câu này thắng lệnh xóa của LLM (trừ khi chính câu có phủ định)
        if name in norm_updates and name not in fixed and name not in norm_removals:
            fixed[name] = norm_updates[name]
    # "có món chay" (luật) = dietary, không phải loại quán chay -> sửa nếu LLM đặt cuisine=Chay
    if nm.dietary == "chay" and fixed.get("cuisine") and fixed["cuisine"].value == "Chay" and "Chay" not in nm.cuisines:
        fixed.pop("cuisine")
        fixed["dietary"] = SlotUpdate(op="set", value="chay")
    if nm.budget_low_hint and "budget_max" not in fixed:
        fixed["budget_max"] = SlotUpdate(op="set", value=config.CHEAP_BUDGET_VND)
    # chỉ chấp nhận XÓA tiêu chí tìm kiếm / quán đã chọn; không xóa slot vừa được gán trong câu này
    # LLM chỉ được xóa tiêu chí khi câu có từ phủ định / bỏ ("không cần", "bỏ", "cũng được"...)
    from core.fallback_classifier import NEGATION_RE as _NEG
    llm_removals = list(out.slot_removals) if _NEG.search(nm.normalized_text.lower()) else []
    if out.slot_removals and not llm_removals:
        dropped.append(f"xóa {out.slot_removals} (câu không có ý bỏ tiêu chí)")
    removals = [r for r in dict.fromkeys(llm_removals + norm_removals)
                if r in REMOVABLE_SLOTS and not (r in fixed and fixed[r].op != "remove")
                and (r not in norm_updates or r in norm_removals)]
    dropped += [f"xóa {r} (không hợp lệ)" for r in llm_removals if r not in removals]

    # 3) số không rõ đơn vị: dùng slot bot vừa hỏi; nếu vẫn mơ hồ -> hỏi lại
    clarification = out.needs_clarification
    if nm.bare_numbers and not (nm.time or nm.party_size or nm.party_delta or nm.date):
        n = nm.bare_numbers[-1]
        awaiting = [s for s in state.get("awaiting_slots", []) if s in ("party_size", "time")]
        if len(awaiting) == 1:
            clarification = None
            if awaiting[0] == "party_size" and 1 <= n <= config.MAX_PARTY_SIZE:
                fixed["party_size"] = SlotUpdate(op="set", value=n)
            elif awaiting[0] == "time" and n <= 23:
                fixed["time"] = SlotUpdate(op="set", value=parse_time(f"{n} giờ")[0])
        elif n <= 23 and (state.get("active_task") or "time" in slots_now or "party_size" in slots_now):
            clarification = clarification or f"Dạ bạn muốn đổi sang {n} giờ hay {n} người ạ?"
            fixed.pop("time", None)
            fixed.pop("party_size", None)

    # Chỉ giữ câu hỏi lại của LLM khi câu có số không rõ đơn vị (mơ hồ thật). Các thông tin còn thiếu
    # khác do policy hỏi theo bảng slot bắt buộc; LLM 3B hay sinh câu hỏi lại thừa cho câu rõ nghĩa.
    if clarification and not nm.bare_numbers:
        meta["dropped_clarification"] = clarification
        clarification = None

    # 4) bỏ cập nhật không đổi gì
    for name in list(fixed):
        up = fixed[name]
        if up.op == "set" and slots_now.get(name) == up.value:
            fixed.pop(name)

    intents = list(out.intents)
    # câu hỏi "món gì / món nào / thực đơn / menu" là hỏi THỰC ĐƠN, kể cả khi có chữ "nhà hàng"
    if intents[0] in ("search_restaurant", "restaurant_details", "chitchat") and MENU_Q.search(nm.normalized_text.lower()):
        intents[0] = "search_menu"
        meta["intent_adjusted"] = "menu_question"
    # chọn 1 mục trong danh sách kết quả ("quán thứ 2", "lấy quán đầu") = chọn quán để đặt bàn
    if nm.ordinal_ref and state.get("last_results") and intents[0] in INTENTS_SELECTING_RESULT:
        intents = ["create_reservation"] + [i for i in intents[1:] if i != "create_reservation"]
        meta["intent_adjusted"] = "ordinal_selection"

    if dropped:
        meta["dropped_slots"] = dropped
    return NLUOutput(intents=intents, is_followup=out.is_followup, slot_updates=fixed,
                     slot_removals=removals, confirmation=out.confirmation,
                     needs_clarification=clarification)
