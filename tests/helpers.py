"""Hàm dùng chung cho pytest và script đánh giá: dựng ngữ cảnh, kiểm tra 'bịa thông tin'."""

from __future__ import annotations

import datetime as dt
import os
import time
from typing import Any, Dict, List

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FULL_MENU_MARK = "Mình có thể:"


def subst(v: Any) -> Any:
    today = dt.date.today()
    rep = {"{today}": today.isoformat(), "{tomorrow}": (today + dt.timedelta(days=1)).isoformat(),
           "{d2}": (today + dt.timedelta(days=2)).isoformat()}
    if isinstance(v, str):
        for k, x in rep.items():
            v = v.replace(k, x)
        return v
    if isinstance(v, dict):
        return {k: subst(x) for k, x in v.items()}
    if isinstance(v, list):
        return [subst(x) for x in v]
    return v


def load_yaml(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return subst(yaml.safe_load(f))


def seed_context(agent, ctx: Dict[str, Any]) -> None:
    """Nạp ngữ cảnh dựng sẵn vào state của agent (sau khi reset)."""
    from actions import db
    from agent import tools as T

    agent.reset()
    st = agent.state
    for k, v in (ctx.get("slots") or {}).items():
        st.slots[k] = v
    st.active_task = ctx.get("active_task")
    names = {r["ma_nha_hang"]: r["ten_nha_hang"] for r in db.list_restaurants()}
    st.last_search_results = [{"ma_nha_hang": r, "ten_nha_hang": names[r]} for r in ctx.get("results", [])]
    if st.slots.get("restaurant_id"):
        st.restaurant_name = names.get(st.slots["restaurant_id"])
    if st.slots.get("booking_code"):
        st.last_booking_code = st.slots["booking_code"]
    if st.last_search_results:
        st.history = [{"role": "user", "text": "tìm quán giúp mình"},
                      {"role": "bot", "text": "Dạ em tìm được: " + "; ".join(
                          f"{i + 1}. {r['ten_nha_hang']}" for i, r in enumerate(st.last_search_results))}]
    p = ctx.get("pending")
    if p == "create_reservation":
        params = {k: st.slots[k] for k in ("date", "time", "party_size")}
        params.update(restaurant_id=st.slots["restaurant_id"], customer_name=None, phone=None)
        summary = {**params, "ten_nha_hang": names[params["restaurant_id"]], "table_size": 4}
        agent.sm.set_pending("create_reservation", params, summary)
        st.history.append({"role": "bot", "text": "📋 XÁC NHẬN ĐẶT BÀN ... Bạn xác nhận chứ ạ? (có / không)"})
    elif p == "cancel_reservation":
        bk = T.get_reservation(st.slots["booking_code"])["data"]
        agent.sm.set_pending("cancel_reservation", {"booking_code": bk["booking_code"], "reason": "test"},
                             {"booking": bk})
        st.history = [{"role": "user", "text": f"hủy {bk['booking_code']}"},
                      {"role": "bot", "text": "⚠️ XÁC NHẬN HỦY ĐƠN ... Bạn xác nhận chứ ạ? (có / không)"}]
    if ctx.get("awaiting_resume"):
        st.awaiting_resume = True
        st.task_stack = [{"task": t, "pending_action": None, "turn": 0} for t in ctx.get("task_stack", [])]
        st.history.append({"role": "bot", "text": "Bạn có muốn tiếp tục tìm quán lúc nãy không ạ?"})


def hallucination_checks(agent, reply: str) -> List[str]:
    """Các lỗi 'bịa / sai sự thật' trong phản hồi so với DB và vết tool."""
    from actions import db
    from agent import tools as T

    t = agent.last_trace
    errs: List[str] = []
    if FULL_MENU_MARK in reply and "help" not in t["blocks"] and t["intents"][0] != "greeting":
        errs.append("MENU_MẶC_ĐỊNH: trả menu mặc định cho câu có nghĩa")
    success_marks = ("Đặt bàn thành công", "Đã hủy đơn", "Đã cập nhật đơn", "Đã tạo phiếu hỗ trợ")
    if any(m in reply for m in success_marks) and not any(
            c["tool"] in ("create_reservation", "update_reservation", "cancel_reservation", "create_support_ticket")
            and c["ok"] for c in t["tools"]):
        errs.append("BÁO_THÀNH_CÔNG_KHI_CHƯA_COMMIT")
    for c in t["tools"]:
        if c["tool"] != "check_table_availability" or not c["ok"]:
            continue
        res = T.check_table_availability(**c["params"])
        if not res["ok"]:
            continue
        avail = res["data"]["available"]
        if "confirm" in t["blocks"] and "XÁC NHẬN ĐẶT BÀN" in reply and not avail:
            errs.append(f"XÁC_NHẬN_ĐẶT_KHI_HẾT_BÀN: {c['params']}")
        if not ({"restaurant_list", "availability"} & set(t["blocks"])):
            continue
        line_ok = "🟢" in reply if avail else ("🔴" in reply or "🟠" in reply)
        if not line_ok:
            errs.append(f"SAI_TÌNH_TRẠNG_BÀN: DB available={res['data']['available']} cho {c['params']}")
    names = {r["ten_nha_hang"] for r in db.list_restaurants()}
    for line in reply.splitlines():
        s = line.strip()
        if s[:2].rstrip(".").isdigit() and ". " in s[:4] and " — " in s:
            name = s.split(". ", 1)[1].split(" — ")[0].strip()
            if name not in names:
                errs.append(f"QUÁN_KHÔNG_CÓ_TRONG_DB: {name}")
    return errs


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t0
