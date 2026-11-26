"""Response Generator (mục 2.6): render block của policy bằng template Python.

Chỉ dùng dữ liệu trong block (đến từ tool/DB/RAG). Không tự thêm số liệu.
Câu "thành công" chỉ xuất hiện ở block action_done — block này chỉ được tạo khi tool trả ok=True
(tức DB đã COMMIT).
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Tuple

WEEKDAY_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
ACTION_VI = {"create_reservation": "đặt bàn", "update_reservation": "cập nhật đơn",
             "cancel_reservation": "hủy đơn", "create_support_ticket": "tạo phiếu hỗ trợ"}
TASK_VI = {"search_restaurant": "tìm quán", "check_availability": "kiểm tra bàn trống",
           "create_reservation": "đặt bàn", "update_reservation": "đổi đơn đặt bàn",
           "cancel_reservation": "hủy đơn", "get_reservation": "tra cứu đơn",
           "search_menu": "xem thực đơn", "restaurant_details": "xem thông tin quán"}

GREETING_FULL = (
    "Chào bạn! Mình là trợ lý đặt bàn nhà hàng. Mình có thể:\n"
    "  • Tìm quán theo món, khu vực, ngân sách\n"
    "  • Tìm theo phong cách: món Tây (Âu), Nhật, Hàn, đồ Á, Thái, Ý, steak, hải sản, lẩu, chay…\n"
    "  • Xem thực đơn và giá món\n"
    "  • Kiểm tra bàn trống, đặt / đổi / hủy bàn\n"
    "  • Giải đáp chính sách (cọc, hủy, đến trễ…) và tiếp nhận khiếu nại\n"
    "Bạn cần mình giúp gì ạ?"
)


def vnd(n: Any) -> str:
    try:
        return f"{int(n):,}".replace(",", ".") + "đ"
    except (TypeError, ValueError):
        return str(n)


def vi_date(iso: str) -> str:
    try:
        d = dt.date.fromisoformat(iso)
    except (TypeError, ValueError):
        return str(iso)
    today = dt.date.today()
    rel = {0: " (hôm nay)", 1: " (ngày mai)"}.get((d - today).days, "")
    return f"{WEEKDAY_VI[d.weekday()]} {d.strftime('%d/%m/%Y')}{rel}"


def _criteria_text(c: Dict[str, Any]) -> str:
    parts = []
    if c.get("cuisine"):
        parts.append(f"món {c['cuisine']}")
    if c.get("dietary") == "chay" and c.get("cuisine") != "Chay":
        parts.append("có món chay")
    if c.get("area"):
        parts.append(f"ở {c['area']}")
    if c.get("budget_max"):
        parts.append(f"giá TB ≤ {vnd(c['budget_max'])}/người")
    if c.get("occasion"):
        parts.append(f"phù hợp {c['occasion']}")
    if c.get("rating_min"):
        parts.append(f"từ {c['rating_min']:g} sao")
    return ", ".join(parts) or "trong hệ thống"


def _chunk_no(h: Dict[str, Any]) -> int:
    tail = str(h.get("id", "")).rsplit("#", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _merge_policy_hits(hits: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Gộp các đoạn cùng một mục chính sách theo đúng thứ tự trong văn bản, bỏ phần overlap."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits:
        src = h["title"] + (f" › {h['section']}" if h.get("section") else "")
        groups.setdefault(src, []).append(h)
    out: List[Tuple[str, str]] = []
    for src, hs in groups.items():
        text = ""
        for h in sorted(hs, key=_chunk_no):
            piece = h["content"].replace("**", "").strip()
            if text and piece.startswith("…"):
                piece = piece.lstrip("… ")
                # đoạn sau bắt đầu bằng phần đuôi (overlap) của đoạn trước -> cắt phần trùng
                for k in range(min(len(piece), 120), 10, -1):
                    if text.endswith(piece[:k]):
                        piece = piece[k:].lstrip()
                        break
            else:
                piece = piece.lstrip("… ")
            text = f"{text} {piece}".strip()
        out.append((src, text))
    return out


def _facts_line(r: Dict[str, Any], with_hours: bool = True) -> str:
    """Giá / đánh giá / giờ mở cửa; trường trống ghi rõ "chưa có", không bao giờ tự điền."""
    est = " (ước tính)" if r.get("uoc_tinh") else ""
    parts = [f"Giá TB ~{vnd(r['gia_trung_binh'])}/người{est}" if r.get("gia_trung_binh") else "Giá: chưa có thông tin",
             (f"⭐ {r['diem_danh_gia']} ({r.get('so_luot_danh_gia') or 0} đánh giá)" if r.get("diem_danh_gia")
              else "Đánh giá: chưa có")]
    if with_hours:
        parts.append(f"Mở cửa {r['gio_mo_cua']}–{r['gio_dong_cua']}{est}" if r.get("gio_mo_cua") and r.get("gio_dong_cua")
                     else "Giờ mở cửa: chưa có")
    return " · ".join(parts)


def _source_tag(r: Dict[str, Any]) -> str:
    if not r.get("nguon_du_lieu") or r["nguon_du_lieu"] == "demo":
        return ""
    return f"\n   ⓘ Nguồn: {r['nguon_du_lieu']} · " + ("đã xác minh" if r.get("xac_minh") else "chưa xác minh")


def _availability_line(chk: Dict[str, Any]) -> str:
    if not chk["ok"]:
        return f"⚠️ {chk['message']}"
    d = chk["data"]
    if d.get("time") is None:   # liệt kê giờ còn trống
        when = f"buổi {d['period']}" if d.get("period") else "trong ngày"
        if d.get("reason") == "LARGE_GROUP":
            return f"🟠 Nhóm {d['party_size']} người cần liên hệ quán để xếp bàn đoàn"
        if d["available"]:
            return f"🟢 Còn bàn cho {d['party_size']} người {when} lúc: " + ", ".join(d["free_times"])
        return f"🔴 Hết bàn cho {d['party_size']} người {when}"
    if d["available"]:
        return f"🟢 Còn bàn {d['table_size']} chỗ lúc {d['time']} cho {d['party_size']} người ({d['free_tables']} bàn trống)"
    if d.get("reason") == "LARGE_GROUP":
        return f"🟠 Nhóm {d['party_size']} người cần liên hệ quán để xếp bàn đoàn"
    sugg = ", ".join(d.get("suggestions") or []) or "không còn khung giờ trống trong ngày"
    reason = "hết bàn phù hợp" if d.get("reason") == "FULL" else "không nhận đặt khung giờ này"
    return f"🔴 Lúc {d['time']} {reason} cho {d['party_size']} người — giờ còn bàn: {sugg}"


def _booking_lines(b: Dict[str, Any]) -> List[str]:
    return [f"   • Mã đặt bàn: {b['booking_code']}",
            f"   • Nhà hàng: {b['ten_nha_hang']} — {b['dia_chi']}",
            f"   • Thời gian: {b['time']}, {vi_date(b['date'])}",
            f"   • Số khách: {b['party_size']} người",
            f"   • Trạng thái: {b['status']}"]


# ---------------------------------------------------------------------- từng loại block
def _render(b: Dict[str, Any]) -> str:
    t = b["type"]
    if t == "greeting_full":
        return GREETING_FULL
    if t == "greeting_short":
        return "Chào bạn! Bạn muốn tìm quán hay đặt bàn hôm nay ạ?"
    if t == "help":
        return GREETING_FULL.replace("Chào bạn! ", "")
    if t == "chitchat":
        return "Mình luôn sẵn lòng ạ! Khi cần tìm quán hay đặt bàn, bạn cứ nhắn mình nhé."
    if t == "out_of_scope":
        return ("Dạ câu này nằm ngoài phạm vi mình hỗ trợ ạ. Mình chỉ giúp được về nhà hàng: tìm quán, thực đơn, "
                "bàn trống, đặt/đổi/hủy bàn và chính sách.")
    if t in ("clarify", "ack", "next_step"):
        return b["text"]
    if t == "ask_slots":
        qs = b["questions"]
        if len(qs) == 1:
            return f"Để {b['context']}, bạn cho mình xin {qs[0]} nhé?"
        return f"Để {b['context']}, bạn cho mình xin thêm:\n" + "\n".join(f"   • {q}" for q in qs)
    if t == "pick_restaurant":
        opts = "\n".join(f"   {i + 1}. {name}" for i, name in enumerate(b["options"]))
        return f"{b['lead']}\n{opts}" if opts else b["lead"]

    if t == "restaurant_list":
        rs = b["restaurants"]
        total = b.get("total") or len(rs)
        lines = [f"Mình tìm được {total} quán {_criteria_text(b['criteria'])}"
                 + (f" (hiển thị {len(rs)} quán đầu):" if total > len(rs) else ":")]
        if b.get("slot_info"):
            si = b["slot_info"]
            lines[0] += (f"\n(Kiểm tra bàn: {si['time'] + ', ' if si.get('time') else ''}"
                         f"{vi_date(si['date'])}, {si['party_size']} người)")
        for i, r in enumerate(rs, 1):
            lines.append(f"\n{i}. {r['ten_nha_hang']} — {r['loai_am_thuc']}, {r['quan_huyen']}")
            lines.append("   " + _facts_line(r))
            lines.append(f"   {r['dia_chi']}")
            if b.get("criteria", {}).get("dietary") == "chay" or (r.get("co_mon_chay") and b.get("show_veg")):
                lines[-3] += " · 🌱 có món chay"
            for chk in b["availability"].get(r["ma_nha_hang"], []):
                lines.append("   " + _availability_line(chk))
        return "\n".join(lines)
    if t == "no_results":
        return (f"Dạ hiện chưa có quán nào khớp {_criteria_text(b['criteria'])} trong hệ thống. "
                "Bạn có muốn bỏ bớt một điều kiện (khu vực, giá…) để mình tìm rộng hơn không ạ?")
    if t == "availability":
        lines: List[str] = []
        head = None
        for chk in b["results"]:
            if not chk["ok"]:
                name = (chk.get("data") or {}).get("ten_nha_hang")
                lines.append(f"Mình chưa kiểm tra được bàn{(' tại ' + name) if name else ''}: {chk['message']}")
                continue
            d = chk["data"]
            if head is None:
                head = f"{d['ten_nha_hang']} — {vi_date(d['date'])}:"
                lines.insert(0, head)
            lines.append(f"   {_availability_line(chk)}")
        return "\n".join(lines)

    if t == "confirm":
        a, s = b["action"], b["summary"]
        if a == "create_reservation":
            body = [f"   • Nhà hàng: {s['ten_nha_hang']}", f"   • Thời gian: {s['time']}, {vi_date(s['date'])}",
                    f"   • Số khách: {s['party_size']} người (bàn {s['table_size']} chỗ)"]
            if s.get("note"):
                body.append(f"   • Ghi chú: {s['note']} (hệ thống chưa quản lý vị trí bàn, yêu cầu sẽ được "
                            "ghi vào đơn để nhà hàng sắp xếp)")
            head = "📋 XÁC NHẬN ĐẶT BÀN"
        elif a == "update_reservation":
            bk, new = s["booking"], s["new"]
            body = [f"   • Đơn {bk['booking_code']} — {bk['ten_nha_hang']}",
                    f"   • Hiện tại: {bk['time']}, {vi_date(bk['date'])}, {bk['party_size']} người",
                    f"   • Đổi thành: {new['time']}, {vi_date(new['date'])}, {new['party_size']} người"]
            head = "📋 XÁC NHẬN ĐỔI ĐƠN"
        elif a == "cancel_reservation":
            body = _booking_lines(s["booking"])
            head = "⚠️ XÁC NHẬN HỦY ĐƠN"
        else:
            body = [f"   • Loại: {s['category']}", f"   • Nội dung: {s['description']}"]
            if s.get("booking_code"):
                body.append(f"   • Mã đặt bàn: {s['booking_code']}")
            head = "📝 TẠO PHIẾU HỖ TRỢ"
        return f"{head}\n" + "\n".join(body) + "\nBạn xác nhận chứ ạ? (có / không)"

    if t == "action_done":
        a, d = b["action"], b["data"]
        if a == "create_reservation":
            return "✅ Đặt bàn thành công!\n" + "\n".join(_booking_lines(d)) + \
                "\nKhi đến quán bạn đọc mã đặt bàn cho lễ tân nhé."
        if a == "update_reservation":
            return "✅ Đã cập nhật đơn:\n" + "\n".join(_booking_lines(d))
        if a == "cancel_reservation":
            return f"✅ Đã hủy đơn {d['booking_code']} tại {d['ten_nha_hang']}."
        return (f"✅ Đã tạo phiếu hỗ trợ {d['ma_yeu_cau']} ({d['loai_su_co']}). "
                "Nhân viên sẽ liên hệ lại bạn sớm nhất ạ.")
    if t == "action_failed":
        return (f"❌ Chưa thể {ACTION_VI.get(b['action'], 'thực hiện')}: {b['message']}\n"
                "Hệ thống CHƯA ghi nhận thay đổi nào.")
    if t == "pending_cancelled":
        return f"Mình đã hủy thao tác {ACTION_VI.get(b['action'], '')}, chưa có gì thay đổi ạ."
    if t == "pending_expired":
        return f"(Yêu cầu {ACTION_VI.get(b['action'], '')} trước đó đã hết hạn xác nhận.)"
    if t == "tool_error":
        return f"Mình chưa {b['action']} được: {b['message']}"
    if t == "booking_info":
        bk = b["booking"]
        extra = f"\n   • Ghi chú: {bk['note']}" if bk.get("note") else ""
        return "Thông tin đơn:\n" + "\n".join(_booking_lines(bk)) + extra
    if t == "note_info":
        return (f"📝 Mình đã ghi nhận yêu cầu: \"{b['note']}\". Hệ thống chưa có dữ liệu vị trí từng bàn, "
                "nên yêu cầu này sẽ được ghi chú vào đơn khi đặt để nhà hàng sắp xếp ạ.")
    if t == "ticket_info":
        tk = b["ticket"]
        return (f"Phiếu hỗ trợ {tk['ma_yeu_cau']}:\n   • Loại: {tk['loai_su_co']}\n   • Nội dung: {tk['mo_ta']}\n"
                f"   • Trạng thái: {tk['trang_thai']}\n   • Tạo lúc: {tk['thoi_gian_tao']}"
                + (f"\n   • Mã đặt bàn liên quan: {tk['ma_dat_ban']}" if tk.get("ma_dat_ban") else ""))
    if t == "booking_inactive":
        bk = b["booking"]
        return f"Dạ đơn {bk['booking_code']} tại {bk['ten_nha_hang']} đã ở trạng thái '{bk['status']}', không thể thay đổi."

    if t == "restaurant_details":
        r = b["restaurant"]
        na = "chưa có thông tin"
        est = " (ước tính)" if r.get("uoc_tinh") else ""
        hours = f"{r['gio_mo_cua']}–{r['gio_dong_cua']}{est}" if r.get("gio_mo_cua") and r.get("gio_dong_cua") else na
        lines = [f"{r['ten_nha_hang']} — {r['loai_am_thuc']}", f"   • Địa chỉ: {r['dia_chi']}",
                 f"   • Mở cửa: {hours} · ĐT: {r.get('so_dien_thoai') or na}",
                 "   • " + _facts_line(r, with_hours=False)]
        if r.get("mo_ta"):
            lines.append(f"   • {r['mo_ta']}")
        if r.get("mon_an"):
            lines.append("   • Món: " + "; ".join(f"{m['ten_mon']} {vnd(m['gia_vnd'])}" for m in r["mon_an"]))
        return "\n".join(lines)
    if t == "menu_list":
        ds = b["dishes"]
        if not ds:
            what = f"món \"{b['keyword']}\"" if b.get("keyword") else "món nào khớp yêu cầu"
            where = f" ở {b['restaurant_name']}" if b.get("restaurant_name") else ""
            return f"Dạ hiện chưa có {what}{where} trong thực đơn của hệ thống."
        if b.get("wanted"):
            lead = (f"Dạ hiện chưa có món \"{b['wanted']}\" trong thực đơn. "
                    f"Các quán có món \"{b['keyword']}\" gần nhất:")
            return lead + "\n" + "\n".join(
                f"   • {d['ten_mon']} — {vnd(d['gia_vnd'])} ({d['ten_nha_hang']})" for d in ds)
        head = ("Các món" + (f" \"{b['keyword']}\"" if b.get("keyword") else "")) + (" chay" if b["vegetarian"] else "") + \
               (f" giá ≤ {vnd(b['budget_max'])}" if b.get("budget_max") else "") + \
               (f" tại {b['restaurant_name']}" if b.get("restaurant_name") else "") + ":"
        return head + "\n" + "\n".join(
            f"   • {d['ten_mon']} — {vnd(d['gia_vnd'])} ({d['ten_nha_hang']})" + (" 🌱" if d["chay"] else "")
            for d in ds)

    if t == "policy_answer":
        lines = [b.get("lead", "Dạ theo chính sách hiện hành:")]
        for src, content in _merge_policy_hits(b["hits"]):
            lines.append(f"   • {content}\n     (Nguồn: {src})")
        return "\n".join(lines)
    if t == "policy_not_found":
        return "Mình chưa tìm thấy thông tin chính sách phù hợp với câu hỏi này."
    if t == "apology":
        return "Mình rất xin lỗi về trải nghiệm này ạ."
    if t == "app_issue_ack":
        return "Mình ghi nhận lỗi ứng dụng bạn gặp phải."
    if t == "resume_question":
        return f"Bạn có muốn tiếp tục {TASK_VI.get(b['task'], 'việc')} lúc nãy không ạ?"
    if t == "llm_notice":
        return ("⚠️ Mô hình AI cục bộ đang không phản hồi, mình tạm dùng chế độ hiểu câu dự phòng "
                "nên có thể hiểu chưa chính xác.")
    return ""


def render(blocks: List[Dict[str, Any]]) -> str:
    parts = [_render(b) for b in blocks]
    return "\n\n".join(p for p in parts if p).strip() or \
        "Mình chưa hiểu rõ ý bạn. Bạn muốn tìm quán, kiểm tra bàn hay đặt bàn ạ?"
