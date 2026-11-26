"""Dialogue Policy (mục 2.3): quyết định hành động từ intent + state.

- Bảng slot bắt buộc theo hành động; thiếu thì hỏi đúng slot thiếu (tối đa 2 slot/lần).
- search_restaurant đã có ngày/giờ/số người -> tự kiểm tra bàn trống cho các kết quả.
- create/update/cancel và tạo phiếu hỗ trợ: hiển thị tóm tắt, đặt pending_action, chỉ thực thi khi xác nhận.
- Không hiểu -> hỏi lại cụ thể. KHÔNG trả menu mặc định (chỉ khi chào hỏi lần đầu hoặc hỏi "bạn làm được gì").

Đầu ra: danh sách block {"type": ..., ...dữ liệu từ tool...} để Response Generator render.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.tools import ToolExecutor
from core.normalizer import NormalizedMessage
from core.schema import NLUOutput
from core.state import INTERRUPT_INTENTS, MergeResult, StateManager

SLOT_QUESTIONS = {
    "date": "ngày dùng bữa (vd: tối mai, thứ 7, 25/11)",
    "time": "giờ đến (vd: 18:30, 7 giờ tối)",
    "party_size": "số người",
    "restaurant_id": "nhà hàng bạn muốn chọn",
    "booking_code": "mã đặt bàn (dạng DBxxx)",
    "cuisine": "món bạn muốn ăn (lẩu, nướng, chay, đồ Nhật…)",
    "area": "khu vực (vd: Quận 1, Tân Bình)",
}
REQUIRED = {
    "check_availability": ["restaurant_id", "date", "time", "party_size"],
    "create_reservation": ["restaurant_id", "date", "time", "party_size"],
    "update_reservation": ["booking_code"],
    "cancel_reservation": ["booking_code"],
    "get_reservation": ["booking_code"],
    "restaurant_details": ["restaurant_id"],
}
BOOKING_TASKS = ("search_restaurant", "check_availability", "create_reservation")
import re as _re
EXPLICIT_CONFIRM = _re.compile(r"(?<!\w)(xác nhận|đồng ý|chắc chắn|ok|oke|yes|vâng|ừ)(?!\w)")
CHANGE_VERBS = _re.compile(r"(?<!\w)(đổi|tăng|giảm|thêm|bớt|lùi|dời|chuyển|sửa|thay đổi)(?!\w)")
CONFIRM_WORDS = set("""ừ ừm uh uk ok oke okie vâng dạ có đúng rồi chuẩn chắc chắn đồng ý xác nhận yes
hủy huỷ luôn đi nhé nha em anh chị mình giúp giùm làm được vậy thế ạ à cứ cancel thôi
tôi bàn đơn đặt mã này đó""".split())


def _extra_content(text: str) -> bool:
    """Câu trả lời xác nhận có chứa nội dung khác ngoài từ đồng ý/hủy (vd "ừ tạo phiếu giúp mình")."""
    import re

    words = re.findall(r"\w+", text.lower())
    return any(w not in CONFIRM_WORDS and not w.upper().startswith("DB") for w in words)
MAX_AVAILABILITY_CHECKS = 6


class DialoguePolicy:
    def __init__(self, sm: StateManager, tools: ToolExecutor) -> None:
        self.sm = sm
        self.tools = tools

    @property
    def st(self):
        return self.sm.state

    # ================================================================== entry
    def decide(self, nlu: NLUOutput, nm: NormalizedMessage, merge: MergeResult,
               meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        st = self.st
        self._nm = nm
        awaiting_before = list(st.awaiting_slots)
        st.awaiting_slots = []
        blocks: List[Dict[str, Any]] = []
        intents = list(nlu.intents)
        primary = intents[0]
        yes = nlu.confirmation == "yes" or primary == "confirm"
        no = nlu.confirmation == "no" or primary == "deny"

        if self.sm.pending_expired():
            blocks.append({"type": "pending_expired", "action": st.pending_action["type"]})
            st.pending_action = None

        # 1) đang chờ xác nhận
        if (st.pending_action and primary == st.pending_action["type"] and not merge.changed
                and EXPLICIT_CONFIRM.search(nm.normalized_text.lower())):
            yes = True     # "Tôi xác nhận hủy bàn" khi đang chờ chính thao tác đó
        if st.pending_action:
            if yes and not merge.changed and st.pending_action["type"] == "cancel_reservation"                     and _extra_content(nm.normalized_text):
                # thao tác không hoàn tác được + câu trả lời kèm nội dung khác -> hỏi lại cho chắc
                code = st.pending_action["params"]["booking_code"]
                return blocks + [{"type": "clarify",
                                  "text": f"Dạ để chắc chắn: bạn muốn HỦY đơn {code} đúng không ạ? (có / không)"}]
            if yes and not merge.changed:
                return blocks + self._execute_pending()
            if no and not merge.changed:
                action = st.pending_action["type"]
                st.pending_action = None
                return blocks + [{"type": "pending_cancelled", "action": action}]
            if merge.changed and st.pending_action["type"] == "create_reservation":
                st.pending_action = None          # khách sửa thông tin -> lập lại đơn với slot mới
                intents = ["create_reservation"]
            elif merge.changed and st.pending_action["type"] == "update_reservation":
                st.pending_action = None
                intents = ["update_reservation"]

        # 2) đang hỏi "có muốn tiếp tục việc dở không"
        if st.awaiting_resume:
            st.awaiting_resume = False
            if yes and not merge.changed:
                return blocks + self._resume()
            if no and not merge.changed:
                st.task_stack.pop() if st.task_stack else None
                return blocks + [{"type": "ack", "text": "Dạ vâng ạ. Bạn cần mình hỗ trợ gì thêm cứ nhắn nhé!"}]

        # 3) NLU yêu cầu hỏi lại (câu mơ hồ)
        if nlu.needs_clarification:
            st.awaiting_slots = ["time", "party_size"] if "giờ hay" in nlu.needs_clarification else awaiting_before
            return blocks + [{"type": "clarify", "text": nlu.needs_clarification}]

        if merge.reference == "ambiguous":
            return blocks + [self._ask_pick_restaurant("Dạ bạn muốn chọn quán nào ạ?")]
        if merge.reference == "out_of_range":
            if st.last_search_results:
                return blocks + [self._ask_pick_restaurant(
                    f"Dạ danh sách vừa rồi chỉ có {len(st.last_search_results)} quán ạ.")]
            return blocks + [{"type": "clarify",
                              "text": "Mình chưa có danh sách quán nào để chọn. Bạn muốn tìm quán món gì, ở khu vực nào ạ?"}]

        # tra cứu phiếu hỗ trợ ("cho tôi mã phiếu vừa tạo", "phiếu HT001")
        if nm.ticket_lookup:
            return blocks + self._ticket_info(nm)

        # 4) xử lý từng intent (tối đa 2)
        mutating = {"create_reservation", "update_reservation", "cancel_reservation"}
        if set(intents) & mutating:
            intents = [i for i in intents if i not in ("check_availability", "search_restaurant")] or intents
        if intents[0] == "search_menu":
            intents = [i for i in intents if i not in ("search_restaurant", "check_availability")]
        if "search_restaurant" in intents and "check_availability" in intents:
            first = intents[0] if intents[0] in ("search_restaurant", "check_availability") else "search_restaurant"
            intents = [i for i in intents if i not in ("search_restaurant", "check_availability") or i == first]
        handled: List[str] = []
        for intent in intents[:2]:
            intent = self._remap(intent, nlu, merge)
            if intent in handled:
                continue
            # intent phụ KHÔNG được ghi đè thao tác đang chờ xác nhận do intent chính tạo ra;
            # chỉ cho phép intent phụ chỉ-đọc (trả lời chính sách) khi đã có pending.
            if handled and st.pending_action and intent != "policy_question":
                break
            handled.append(intent)
            if intent == "policy_question" and len(handled) > 1:
                blocks.extend(self._policy(nm, secondary=True))
                continue
            blocks.extend(self._handle(intent, nlu, nm, merge, meta))

        if merge.changed.get("note") and not any(b["type"] == "confirm" for b in blocks):
            blocks.append({"type": "note_info", "note": st.slots["note"]})

        # 5) vừa trả lời câu hỏi xen ngang -> hỏi có muốn tiếp tục việc dở
        if (handled and handled[0] in INTERRUPT_INTENTS and st.task_stack
                and not st.pending_action and not st.awaiting_slots):
            st.awaiting_resume = True
            blocks.append({"type": "resume_question", "task": st.task_stack[-1]["task"]})
        return blocks

    # ================================================================== điều hướng
    def _remap(self, intent: str, nlu: NLUOutput, merge: MergeResult) -> str:
        st = self.st
        nm = getattr(self, "_nm", None)
        low = nm.normalized_text.lower() if nm is not None else ""
        if (nm is not None and nm.food and not nm.cuisines and not nm.restaurant and not st.pending_action
                and intent in ("search_restaurant", "check_availability", "create_reservation",
                               "restaurant_details", "search_menu")):
            return "search_restaurant"    # "muốn ăn bún cá" = tìm quán theo món mới, dù đang ở việc khác
        if (st.last_booking_code and st.active_task in ("get_reservation", "update_reservation")
                and intent in ("check_availability", "create_reservation", "search_restaurant")
                and not (nm is not None and nm.restaurant) and not merge.criteria_changed
                and CHANGE_VERBS.search(low)):
            return "update_reservation"   # "tăng lên 6 người", "đổi ngày" ngay sau khi vừa đặt/sửa đơn
        if (intent in ("search_menu", "restaurant_details")
                and any(k in merge.changed for k in ("date", "time", "party_size"))):
            return "check_availability"   # đang xem món/quán, khách đưa ngày/giờ/số người = hỏi bàn
        if (intent == "create_reservation" and nm is not None and nm.asks_tables and not st.pending_action
                and not _re.search(r"(?<!\w)(đặt|book|giữ|chốt|lấy)(?!\w)", low)):
            return "check_availability"   # "nhà hàng nào còn bàn trống" = hỏi, chưa phải đặt
        if (intent == "create_reservation" and nm is not None and len(nm.times) > 1
                and not st.pending_action):
            return "check_availability"   # "18h30 hết thì 19h có không" = hỏi so sánh giờ, chưa đặt
        if (intent == "search_restaurant" and nm is not None and nm.restaurant
                and "restaurant_id" in merge.changed and not merge.criteria_changed
                and any(st.slots.get(k) for k in ("date", "time", "party_size"))):
            return "check_availability"   # "có bàn ... ở <tên quán> không"
        has_code = bool(st.slots.get("booking_code") or st.last_booking_code)
        if intent == "update_reservation" and not has_code and st.active_task in BOOKING_TASKS:
            return st.active_task   # "đổi sang 6 người" khi đang tìm quán = sửa tiêu chí, không phải sửa đơn
        if intent in ("confirm", "deny", "chitchat", "greeting") and merge.changed and st.active_task:
            return st.active_task   # câu chỉ bổ sung thông tin cho việc đang làm
        if (intent in ("search_restaurant", "check_availability") and st.active_task == "create_reservation"
                and not merge.criteria_changed):
            return "create_reservation"   # đang đặt bàn, khách chỉ bổ sung quán/ngày/giờ/số người
        if intent == "check_availability" and not st.slots.get("restaurant_id") and not st.last_search_results:
            return "search_restaurant"
        return intent

    def _handle(self, intent: str, nlu: NLUOutput, nm: NormalizedMessage, merge: MergeResult,
                meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        h = {
            "search_restaurant": self._search,
            "check_availability": self._check,
            "create_reservation": self._create,
            "update_reservation": lambda: self._update(merge, nlu),
            "cancel_reservation": lambda: self._cancel(nm),
            "get_reservation": self._get_booking,
            "restaurant_details": self._details,
            "search_menu": self._menu,
            "policy_question": lambda: self._policy(nm),
            "complaint": lambda: self._ticket_flow(nm, "complaint"),
            "app_issue": lambda: self._ticket_flow(nm, "app_issue"),
        }.get(intent)
        if intent in ("search_restaurant", "check_availability", "create_reservation", "update_reservation",
                      "cancel_reservation", "get_reservation", "restaurant_details", "search_menu"):
            self.st.active_task = intent
        if h:
            return h()
        if intent == "greeting":
            return [{"type": "greeting_short"}]
        if intent == "chitchat":
            if meta.get("help"):
                return [{"type": "help"}]
            # LLM đôi khi xếp câu hỏi quy định vào chitchat: nếu khớp mạnh với điều khoản thì trả lời theo chính sách
            res = self.tools("search_policy", query=nm.normalized_text, restaurant_id=self.st.slots.get("restaurant_id"))
            if res["ok"] and res["data"][0]["score"] >= 0.5:
                return [{"type": "policy_answer", "hits": self._best_hits(res["data"])}]
            return [{"type": "chitchat"}]
        if intent in ("confirm", "deny"):
            return [{"type": "clarify", "text": "Dạ hiện mình không có thao tác nào đang chờ xác nhận. "
                                                "Bạn muốn tìm quán, kiểm tra bàn hay đặt bàn ạ?"}]
        if intent == "out_of_scope":
            return [{"type": "out_of_scope"}]
        return [{"type": "clarify", "text": "Mình chưa hiểu rõ ý bạn. Bạn muốn tìm quán, kiểm tra bàn trống, "
                                            "đặt/đổi/hủy bàn hay hỏi chính sách ạ?"}]

    # ================================================================== hỏi slot
    def _ask(self, missing: List[str], context: str) -> Dict[str, Any]:
        ask = missing[:2]
        self.st.awaiting_slots = ask
        return {"type": "ask_slots", "context": context, "slots": ask,
                "questions": [SLOT_QUESTIONS.get(s, s) for s in ask]}

    def _ask_pick_restaurant(self, lead: str) -> Dict[str, Any]:
        self.st.awaiting_slots = ["restaurant_id"]
        return {"type": "pick_restaurant", "lead": lead,
                "options": [r["ten_nha_hang"] for r in self.st.last_search_results]}

    def _missing(self, names: List[str]) -> List[str]:
        return [n for n in names if not self.st.slots.get(n)]

    def _criteria(self) -> Dict[str, Any]:
        s = self.st.slots
        return {k: s[k] for k in ("cuisine", "area", "budget_max", "dietary", "occasion", "rating_min") if s.get(k)}

    def _resolve_restaurant(self) -> Optional[str]:
        st = self.st
        if st.slots.get("restaurant_id"):
            return st.slots["restaurant_id"]
        if len(st.last_search_results) == 1:
            r = st.last_search_results[0]
            st.slots["restaurant_id"], st.restaurant_name = r["ma_nha_hang"], r["ten_nha_hang"]
            return r["ma_nha_hang"]
        return None

    # ================================================================== bàn trống (dùng chung)
    def _times_to_check(self) -> List[Optional[str]]:
        """Các mốc giờ cần kiểm tra: mọi giờ trong câu hiện tại (vd "18h30 hết thì 19h"), hoặc slot time,
        hoặc None (= liệt kê các giờ còn trống)."""
        nm = getattr(self, "_nm", None)
        if nm is not None and len(nm.times) > 1:
            return list(dict.fromkeys(nm.times))[:3]
        return [self.st.slots.get("time")]

    def _availability(self, rid: str) -> List[Dict[str, Any]]:
        s = self.st.slots
        nm = getattr(self, "_nm", None)
        period = nm.period if nm is not None else None
        out = []
        for t in self._times_to_check():
            out.append(self.tools("check_table_availability", restaurant_id=rid, date=s["date"], time=t,
                                  party_size=s["party_size"], **({"period": period} if t is None and period else {})))
        return out

    @staticmethod
    def _any_available(checks: List[Dict[str, Any]]) -> bool:
        return any(c["ok"] and c["data"].get("available") for c in checks)

    def _list_with_availability(self, restaurants: List[Dict[str, Any]], crit: Dict[str, Any],
                                ask_pick: bool) -> List[Dict[str, Any]]:
        st = self.st
        block: Dict[str, Any] = {"type": "restaurant_list", "criteria": crit, "restaurants": restaurants,
                                 "availability": {}, "slot_info": None}
        if not (st.slots.get("date") and st.slots.get("party_size")):
            missing = self._missing(["date", "time", "party_size"])
            return [block, self._ask(missing, "kiểm tra bàn trống")]
        block["slot_info"] = {k: st.slots.get(k) for k in ("date", "time", "party_size")}
        for r in restaurants[:MAX_AVAILABILITY_CHECKS]:
            block["availability"][r["ma_nha_hang"]] = self._availability(r["ma_nha_hang"])
        out = [block]
        if any(self._any_available(c) for c in block["availability"].values()):
            if st.slots.get("time"):
                st.awaiting_slots = ["restaurant_id"]
                out.append({"type": "next_step", "text": "Bạn muốn đặt quán nào ạ? (nhắn tên hoặc số thứ tự)"
                            if ask_pick or len(restaurants) > 1 else "Bạn có muốn mình đặt bàn luôn không ạ?"})
            else:
                st.awaiting_slots = ["time"]
                out.append({"type": "next_step", "text": "Bạn muốn đặt lúc mấy giờ, ở quán nào ạ?"})
        else:
            st.awaiting_slots = ["time"]
            out.append({"type": "next_step",
                        "text": "Bạn có muốn đổi sang khung giờ khác còn bàn ở trên, hoặc đổi ngày không ạ?"})
        return out

    # ================================================================== tìm quán
    def _search(self) -> List[Dict[str, Any]]:
        st = self.st
        crit = self._criteria()
        nm = getattr(self, "_nm", None)
        asking_tables = any(st.slots.get(k) for k in ("date", "time", "party_size"))
        if nm is not None and nm.food and not nm.cuisines and not nm.restaurant:
            # khách nêu MÓN MỚI ở lượt này ("muốn ăn bún cá"): thay loại món cũ, giữ khu vực / ngày / số người
            st.slots["cuisine"] = None
            st.slots["restaurant_id"], st.restaurant_name = None, None
            return self._search_by_food(nm)
        if not crit and not st.slots.get("restaurant_id") and nm is not None and (nm.food or nm.dish):
            return self._search_by_food(nm)              # "muốn ăn mì cay thì có quán nào"
        if not crit and not st.slots.get("restaurant_id") and not asking_tables:
            if nm is not None and nm.asks_tables:
                res = self.tools("search_restaurants", limit=6)   # "nhà hàng nào còn bàn trống"
                rows = res["data"] if res["ok"] else []
                self.sm.set_results(rows)
                return [{"type": "restaurant_list", "criteria": {}, "restaurants": rows, "availability": {},
                         "slot_info": None},
                        self._ask(["date", "party_size"], "kiểm tra quán nào còn bàn")]
            return [self._ask(["cuisine", "area"], "tìm quán")]
        if st.slots.get("restaurant_id") and not crit:
            rows_res = self.tools("get_restaurant_details", restaurant_id=st.slots["restaurant_id"])
            restaurants = [rows_res["data"]] if rows_res["ok"] else []
        else:
            res = self.tools("search_restaurants", **{k: crit.get(k) for k in
                             ("cuisine", "area", "budget_max", "dietary", "occasion", "rating_min")}, limit=6)
            if not res["ok"]:
                return [{"type": "tool_error", "action": "tìm nhà hàng", "message": res["message"]}]
            restaurants = res["data"]
            m_total = _re.search(r"\d+", res["message"] or "")
            self._last_total = int(m_total.group(0)) if m_total else len(restaurants)
        self.sm.set_results(restaurants)
        if not restaurants:
            return [{"type": "no_results", "criteria": crit}]
        if not asking_tables:
            block = {"type": "restaurant_list", "criteria": crit, "restaurants": restaurants,
                     "availability": {}, "slot_info": None, "total": getattr(self, "_last_total", None)}
            if not any(r.get("dat_ban", 1) for r in restaurants):
                return [block, {"type": "next_step", "text": "Các quán này hiện chỉ có thông tin tra cứu, chưa liên kết "
                                "đặt bàn. Bạn muốn lọc thêm theo món hoặc khu vực không ạ?"}]
            return [block, self._ask(["date", "time", "party_size"], "kiểm tra bàn trống")]
        return self._list_with_availability(restaurants, crit, ask_pick=True)

    def _search_by_food(self, nm: NormalizedMessage) -> List[Dict[str, Any]]:
        """Món không thuộc nhóm ẩm thực đã biết ("bún cá", "mì cay"): tìm quán có tên/loại món chứa cụm đó
        và món trong thực đơn; không có đúng cụm thì thử từ chính ("bún"), và nói rõ là không có đúng món."""
        wanted = nm.food or nm.dish
        tries = [wanted]
        if nm.dish and nm.dish not in tries:
            tries.append(nm.dish)
        head = wanted.split()[0]
        if head not in tries:
            tries.append(head)
        area = self.st.slots.get("area")
        for kw in tries:
            rest = self.tools("search_restaurants", keyword=kw, area=area, limit=6)
            menu = self.tools("search_menu", keyword=kw, limit=8)
            rows = rest["data"] if rest["ok"] else []
            dishes = menu["data"] if menu["ok"] else []
            if not rows and not dishes:
                continue
            out: List[Dict[str, Any]] = []
            found = {r["ma_nha_hang"]: r["ten_nha_hang"] for r in rows}
            for d in dishes:
                found.setdefault(d["ma_nha_hang"], d["ten_nha_hang"])
            self.sm.set_results([{"ma_nha_hang": k, "ten_nha_hang": v} for k, v in found.items()])
            if kw != wanted:
                out.append({"type": "clarify", "text": f"Mình chưa thấy quán hay món nào đúng \"{wanted}\". "
                                                       f"Kết quả gần nhất cho \"{kw}\":"})
            if rows:
                m_total = _re.search(r"\d+", rest["message"] or "")
                out.append({"type": "restaurant_list", "criteria": {"cuisine": kw}, "restaurants": rows,
                            "availability": {}, "slot_info": None,
                            "total": int(m_total.group(0)) if m_total else None})
            if dishes:
                out.append({"type": "menu_list", "dishes": dishes, "vegetarian": False, "budget_max": None,
                            "keyword": kw, "restaurant_name": None})
            self.st.awaiting_slots = ["restaurant_id"]
            return out
        return [{"type": "menu_list", "dishes": [], "vegetarian": False, "budget_max": None, "keyword": wanted,
                 "restaurant_name": None},
                self._ask(["cuisine", "area"], "gợi ý quán khác")]

    # ================================================================== kiểm tra bàn
    def _check(self) -> List[Dict[str, Any]]:
        st = self.st
        rid = self._resolve_restaurant()
        if not rid:
            if st.last_search_results:
                missing = self._missing(["date", "party_size"])
                if missing:
                    return [self._ask(missing, "kiểm tra bàn trống")]
            return self._search()   # kiểm tra cho cả danh sách đang có / mọi quán
        missing = self._missing(["date", "party_size"])
        if missing:
            return [self._ask(missing, "kiểm tra bàn trống")]
        checks = self._availability(rid)
        out = [{"type": "availability", "results": checks}]
        if self._any_available(checks):
            st.active_task = "create_reservation"
            out.append({"type": "next_step", "text": "Bạn có muốn mình đặt bàn này luôn không ạ?"
                        if st.slots.get("time") else "Bạn muốn đặt lúc mấy giờ ạ?"})
            if not st.slots.get("time"):
                st.awaiting_slots = ["time"]
        return out

    # ================================================================== đặt bàn
    def _create(self) -> List[Dict[str, Any]]:
        st = self.st
        rid = self._resolve_restaurant()
        if not rid:
            have_all = all(st.slots.get(k) for k in ("date", "time", "party_size"))
            candidates = st.last_search_results
            if not candidates and (self._criteria() or have_all):
                return self._search()
            if candidates and have_all:
                avail = [r for r in candidates if self._any_available(self._availability(r["ma_nha_hang"]))]
                if len(avail) == 1:   # chỉ 1 quán còn bàn -> đề xuất luôn quán đó
                    st.slots["restaurant_id"] = avail[0]["ma_nha_hang"]
                    st.restaurant_name = avail[0]["ten_nha_hang"]
                    rid = avail[0]["ma_nha_hang"]
                else:
                    res = self.tools("search_restaurants", **{k: self._criteria().get(k) for k in
                                     ("cuisine", "area", "budget_max", "dietary", "occasion", "rating_min")}, limit=6)
                    rows = [r for r in (res["data"] or []) if r["ma_nha_hang"] in
                            {c["ma_nha_hang"] for c in candidates}] if res["ok"] else []
                    if rows:
                        return self._list_with_availability(rows, self._criteria(), ask_pick=True)
            if not rid and len(candidates) > 1:
                return [self._ask_pick_restaurant("Dạ bạn muốn đặt bàn ở quán nào ạ?")]
        missing = self._missing(REQUIRED["create_reservation"])
        if missing:
            return [self._ask(missing, "đặt bàn")]
        checks = self._availability(rid)
        chk = next((c for c in checks if c["ok"] and c["data"].get("available")
                    and c["data"].get("time") == st.slots["time"]), None)
        if chk is None:
            return [{"type": "availability", "results": checks}]
        params = {"restaurant_id": rid, "date": st.slots["date"], "time": st.slots["time"],
                  "party_size": st.slots["party_size"], "customer_name": st.slots.get("customer_name"),
                  "phone": st.slots.get("phone"), "note": st.slots.get("note") or ""}
        summary = {**params, "ten_nha_hang": chk["data"]["ten_nha_hang"], "table_size": chk["data"]["table_size"]}
        self.sm.set_pending("create_reservation", params, summary)
        return [{"type": "confirm", "action": "create_reservation", "summary": summary}]

    # ================================================================== sửa đơn
    def _booking_code(self) -> Optional[str]:
        return self.st.slots.get("booking_code") or self.st.last_booking_code

    def _load_booking(self) -> Any:
        code = self._booking_code()
        if not code:
            return None, [self._ask(["booking_code"], "xử lý đơn đặt bàn")]
        res = self.tools("get_reservation", booking_code=code)
        if not res["ok"]:
            self.st.slots["booking_code"] = None
            self.st.last_booking_code = None
            return None, [{"type": "tool_error", "action": "tra cứu đơn", "message": res["message"],
                           "code": res["error_code"]}]
        if res["data"]["status"] == "Đã hủy":
            return None, [{"type": "booking_inactive", "booking": res["data"]}]
        return res["data"], []

    def _update(self, merge: MergeResult, nlu: NLUOutput) -> List[Dict[str, Any]]:
        bk, err = self._load_booking()
        if bk is None:
            return err
        changes = {k: merge.changed[k] for k in ("date", "time", "party_size") if merge.changed.get(k)}
        pu = nlu.slot_updates.get("party_size")
        if pu is not None and pu.op == "add":   # "thêm 2 khách" -> cộng vào số khách CỦA ĐƠN
            changes["party_size"] = max(1, int(bk["party_size"]) + int(pu.value))
            self.st.slots["party_size"] = changes["party_size"]
        if not changes:
            low = self._nm.normalized_text.lower() if getattr(self, "_nm", None) else ""
            if "ngày" in low:
                self.st.awaiting_slots, q = ["date"], "Bạn muốn đổi sang ngày nào ạ? (giờ và số người giữ nguyên)"
            elif "giờ" in low:
                self.st.awaiting_slots, q = ["time"], "Bạn muốn đổi sang mấy giờ ạ?"
            elif "người" in low or "khách" in low:
                self.st.awaiting_slots, q = ["party_size"], "Bạn muốn đổi thành bao nhiêu người ạ?"
            else:
                self.st.awaiting_slots, q = ["time", "party_size"], "Bạn muốn đổi ngày, giờ hay số người thành bao nhiêu ạ?"
            return [{"type": "booking_info", "booking": bk}, {"type": "clarify", "text": q}]
        new = {"date": changes.get("date", bk["date"]), "time": changes.get("time", bk["time"]),
               "party_size": changes.get("party_size", bk["party_size"])}
        if (new["date"], new["time"]) != (bk["date"], bk["time"]) or new["party_size"] > bk["party_size"]:
            chk = self.tools("check_table_availability", restaurant_id=bk["restaurant_id"], **new)
            same_slot = (new["date"], new["time"]) == (bk["date"], bk["time"])
            # cùng khung giờ: bàn hiện tại của đơn sẽ được trả lại khi đổi, nên chỉ chặn khi DB báo lỗi
            if not chk["ok"] or (not chk["data"]["available"] and not same_slot):
                return [{"type": "availability", "results": [chk]}]
        params = {"booking_code": bk["booking_code"], **changes}
        self.sm.set_pending("update_reservation", params, {"booking": bk, "new": new})
        return [{"type": "confirm", "action": "update_reservation", "summary": {"booking": bk, "new": new}}]

    def _cancel(self, nm: NormalizedMessage) -> List[Dict[str, Any]]:
        bk, err = self._load_booking()
        if bk is None:
            return err
        self.sm.set_pending("cancel_reservation", {"booking_code": bk["booking_code"], "reason": nm.raw_text[:200]},
                            {"booking": bk})
        return [{"type": "confirm", "action": "cancel_reservation", "summary": {"booking": bk}}]

    def _get_booking(self) -> List[Dict[str, Any]]:
        code = self._booking_code()
        if not code:
            return [self._ask(["booking_code"], "tra cứu đơn")]
        res = self.tools("get_reservation", booking_code=code)
        if not res["ok"]:
            return [{"type": "tool_error", "action": "tra cứu đơn", "message": res["message"],
                     "code": res["error_code"]}]
        return [{"type": "booking_info", "booking": res["data"]}]

    # ================================================================== thông tin quán / thực đơn
    def _details(self) -> List[Dict[str, Any]]:
        rid = self._resolve_restaurant()
        if not rid:
            if len(self.st.last_search_results) > 1:
                return [self._ask_pick_restaurant("Dạ bạn muốn xem thông tin quán nào ạ?")]
            return [self._ask(["restaurant_id"], "xem thông tin quán")]
        res = self.tools("get_restaurant_details", restaurant_id=rid)
        if not res["ok"]:
            return [{"type": "tool_error", "action": "xem thông tin quán", "message": res["message"]}]
        return [{"type": "restaurant_details", "restaurant": res["data"]}]

    def _menu(self) -> List[Dict[str, Any]]:
        s = self.st.slots
        nm = getattr(self, "_nm", None)
        if nm is not None and nm.food and nm.food != nm.dish and not nm.restaurant:
            return self._search_by_food(nm)   # món tự do (vd "mì cay"): báo thật nếu không có đúng món
        veg = s.get("dietary") == "chay" or s.get("cuisine") == "Chay"
        keyword = nm.dish if nm is not None else None
        if not keyword and nm is not None and nm.kids:
            keyword = "bé"   # tên món trong DB: "... cho bé"
        in_results = [r["ma_nha_hang"] for r in self.st.last_search_results] if not s.get("restaurant_id") else None
        # giá trong câu hỏi món là giá MÓN; ngân sách/người của tìm quán chỉ áp dụng khi câu này nhắc giá
        budget = s.get("budget_max") if (nm is not None and (nm.budget_max or nm.budget_low_hint)) else None
        res = self.tools("search_menu", restaurant_id=s.get("restaurant_id"), vegetarian=veg, budget_max=budget,
                         cuisine=None if (keyword or s.get("cuisine") == "Chay") else s.get("cuisine"),
                         keyword=keyword, restaurant_ids=in_results or None)
        if not res["ok"]:
            return [{"type": "tool_error", "action": "tìm món", "message": res["message"]}]
        scope = (self.st.restaurant_name if s.get("restaurant_id") else
                 ("các quán vừa tìm" if in_results else None))
        out = [{"type": "menu_list", "dishes": res["data"], "vegetarian": veg, "budget_max": budget,
                "keyword": keyword, "restaurant_name": scope}]
        if res["data"] and in_results and veg:
            names = sorted({d["ten_nha_hang"] for d in res["data"]})
            out.append({"type": "next_step", "text": "Quán có món chay: " + ", ".join(names) +
                        ". Bạn muốn đặt bàn quán nào ạ?"})
        return out

    # ================================================================== chính sách / hỗ trợ
    @staticmethod
    def _best_hits(hits: List[Dict[str, Any]], n: int = 2) -> List[Dict[str, Any]]:
        top = hits[0]["score"]
        return [h for h in hits if h["score"] >= top - 0.06][:n]

    def _policy(self, nm: NormalizedMessage, secondary: bool = False) -> List[Dict[str, Any]]:
        res = self.tools("search_policy", query=nm.normalized_text, restaurant_id=self.st.slots.get("restaurant_id"))
        if res["ok"]:
            return [{"type": "policy_answer", "hits": self._best_hits(res["data"])}]
        if secondary:
            return []   # intent phụ: không tìm thấy thì không làm phiền bằng đề nghị tạo phiếu
        if res["error_code"] == "POLICY_NOT_FOUND":
            st = self.st
            if st.task_stack and st.task_stack[-1]["turn"] == st.turn and st.task_stack[-1].get("pending_action"):
                # vừa cất thao tác đang chờ vì tưởng là câu hỏi xen ngang, nhưng không có chính sách nào khớp
                # -> nhiều khả năng phân loại nhầm: khôi phục thao tác và hỏi xác nhận lại
                p = st.task_stack.pop()["pending_action"]
                st.pending_action = p
                return [{"type": "confirm", "action": p["type"], "summary": p["summary"]}]
            # câu bị hiểu là "chính sách" nhưng chứa thực thể nghiệp vụ -> xử lý theo đúng nghiệp vụ
            if nm.dietary or nm.dish or nm.kids:
                return self._menu()
            if nm.note and self.st.active_task in BOOKING_TASKS:
                return self._handle(self.st.active_task, NLUOutput(intents=[self.st.active_task]), nm,
                                    MergeResult(), {})
            if (nm.time or nm.date or nm.party_size) and self.st.active_task:
                return self._handle(self.st.active_task, NLUOutput(intents=[self.st.active_task]), nm,
                                    MergeResult(), {})
        if res["error_code"] == "POLICY_NOT_FOUND" and self.st.pending_action:
            return [{"type": "policy_not_found"}]   # không ghi đè thao tác đang chờ xác nhận
        if res["error_code"] == "POLICY_NOT_FOUND":
            cat = ("Khiếu nại" if _re.search(r"(?<!\w)(bị tính|tính sai|tính thừa|không đúng|sai|chậm|tệ|kém)(?!\w)",
                                             nm.normalized_text.lower()) else "Câu hỏi chính sách")
            params = {"category": cat, "description": nm.raw_text}
            self.sm.set_pending("create_support_ticket", params, params)
            return [{"type": "policy_not_found" if cat != "Khiếu nại" else "apology"},
                    {"type": "confirm", "action": "create_support_ticket", "summary": params}]
        return [{"type": "tool_error", "action": "tra cứu chính sách", "message": res["message"]}]

    def _ticket_flow(self, nm: NormalizedMessage, kind: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        res = self.tools("search_policy", query=nm.normalized_text, restaurant_id=self.st.slots.get("restaurant_id"))
        if res["ok"] and res["data"][0]["score"] >= 0.55:
            out.append({"type": "policy_answer", "hits": self._best_hits(res["data"], 1), "lead": "Thông tin liên quan:"})
        if self.st.pending_action:   # không ghi đè thao tác đang chờ xác nhận
            out.insert(0, {"type": "apology" if kind == "complaint" else "app_issue_ack"})
            return out
        low = nm.normalized_text.lower()
        if _re.search(r"(?<!\w)(gặp|nói chuyện với|kết nối|chuyển|gọi|liên hệ)\s+(?:với\s+)?"
                      r"(nhân viên|người hỗ trợ|quản lý|tổng đài)(?!\w)", low) and not _re.search(
                r"(?<!\w)(app|ứng dụng|web|otp|đăng nhập)(?!\w)", low):
            kind, category = "complaint", "Yêu cầu gặp nhân viên"
        else:
            category = "Khiếu nại" if kind == "complaint" else "Lỗi ứng dụng"
        params = {"category": category, "description": nm.raw_text,
                  "booking_code": self.st.slots.get("booking_code"),
                  "restaurant_id": self.st.slots.get("restaurant_id"), "phone": self.st.slots.get("phone")}
        self.sm.set_pending("create_support_ticket", params, params)
        out.insert(0, {"type": "apology" if kind == "complaint" else "app_issue_ack"})
        out.append({"type": "confirm", "action": "create_support_ticket", "summary": params})
        return out

    def _ticket_info(self, nm: NormalizedMessage) -> List[Dict[str, Any]]:
        code = nm.ticket_code or self.st.last_ticket_code
        if not code:
            return [{"type": "clarify", "text": "Dạ trong cuộc trò chuyện này chưa có phiếu hỗ trợ nào. "
                                                "Bạn cho mình mã phiếu (dạng HTxxx) để mình tra cứu nhé."}]
        res = self.tools("get_support_ticket", ticket_code=code)
        if not res["ok"]:
            return [{"type": "tool_error", "action": "tra cứu phiếu", "message": res["message"]}]
        return [{"type": "ticket_info", "ticket": res["data"]}]

    # ================================================================== thực thi sau xác nhận
    def _execute_pending(self) -> List[Dict[str, Any]]:
        st = self.st
        p = st.pending_action
        st.pending_action = None
        res = self.tools(p["type"], **p["params"])
        if not res["ok"]:
            return [{"type": "action_failed", "action": p["type"], "message": res["message"],
                     "code": res["error_code"]}]
        out: List[Dict[str, Any]] = [{"type": "action_done", "action": p["type"], "data": res["data"]}]
        if p["type"] in ("create_reservation", "update_reservation", "cancel_reservation"):
            st.last_booking_code = res["data"]["booking_code"]
            st.slots["booking_code"] = res["data"]["booking_code"]
            st.active_task = "get_reservation"
        if p["type"] == "create_support_ticket":
            st.last_ticket_code = res["data"]["ma_yeu_cau"]
        if p["type"] == "create_support_ticket" and st.task_stack:
            st.awaiting_resume = True
            out.append({"type": "resume_question", "task": st.task_stack[-1]["task"]})
        return out

    def _resume(self) -> List[Dict[str, Any]]:
        st = self.st
        if not st.task_stack:
            return [{"type": "ack", "text": "Dạ bạn muốn tiếp tục việc gì ạ?"}]
        task = st.task_stack.pop()
        st.active_task = task["task"]
        if task.get("pending_action"):
            st.pending_action = task["pending_action"]
            if self.sm.pending_expired():
                st.pending_action = None
            else:
                p = st.pending_action
                return [{"type": "confirm", "action": p["type"], "summary": p["summary"]}]
        handler = {"search_restaurant": self._search, "check_availability": self._check,
                   "create_reservation": self._create}.get(task["task"])
        if handler:
            return handler()
        return [{"type": "ack", "text": "Dạ mình tiếp tục nhé, bạn cần mình làm gì tiếp ạ?"}]
