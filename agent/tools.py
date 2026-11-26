"""Tools nghiệp vụ (hàm Python). LLM không gọi trực tiếp và không viết SQL:
policy chọn tool + tham số -> tool kiểm tra tham số -> truy vấn DB (có tham số) / RAG.

Mọi tool trả về: {"ok": bool, "data": ..., "error_code": str|None, "message": str}
Tool ghi DB chỉ trả ok=True sau khi transaction COMMIT thành công.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional

import config
from actions import db
from core import entities
from core.text import fold

logger = logging.getLogger(__name__)


def result(ok: bool, data: Any = None, error_code: Optional[str] = None, message: str = "") -> Dict[str, Any]:
    return {"ok": ok, "data": data, "error_code": error_code, "message": message}


class ParamError(ValueError):
    pass


# ---------------------------------------------------------------------- kiểm tra tham số
def _v_restaurant(rid: Optional[str]) -> Dict[str, Any]:
    if not rid or not re.fullmatch(r"(?:NH|QN)\d{3}", str(rid).upper()):
        raise ParamError(f"Mã nhà hàng không hợp lệ: {rid!r}")
    nh = db.get_restaurant(str(rid).upper())
    if not nh:
        raise LookupError(f"Không tìm thấy nhà hàng {rid}")
    return nh


def _v_date(s: Optional[str]) -> dt.date:
    try:
        d = dt.date.fromisoformat(str(s))
    except (TypeError, ValueError):
        raise ParamError(f"Ngày không hợp lệ: {s!r}")
    if d < dt.date.today():
        raise ParamError("PAST_DATE")
    return d


def _v_time(s: Optional[str]) -> str:
    m = re.fullmatch(r"(\d{2}):(\d{2})", str(s or ""))
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ParamError(f"Giờ không hợp lệ: {s!r}")
    return str(s)


def _v_party(n: Any) -> int:
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise ParamError(f"Số người không hợp lệ: {n!r}")
    if not 1 <= n <= config.MAX_PARTY_SIZE:
        raise ParamError(f"Số người phải từ 1 đến {config.MAX_PARTY_SIZE}")
    return n


def _v_code(code: Optional[str]) -> str:
    c = str(code or "").strip().upper()
    if not re.fullmatch(r"DB\d{3,}", c):
        raise ParamError(f"Mã đặt bàn không hợp lệ: {code!r}")
    return c


def _guard(fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Chuyển exception thành kết quả có cấu trúc. Không bao giờ để lỗi DB lọt thành 'thành công'."""
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except ParamError as e:
            code = "PAST_DATE" if str(e) == "PAST_DATE" else "INVALID_PARAM"
            msg = "Ngày này đã qua." if code == "PAST_DATE" else str(e)
            return result(False, None, code, msg)
        except LookupError as e:
            return result(False, None, "NOT_FOUND", str(e))
        except sqlite3.Error as e:
            logger.exception("DB_ERROR in %s", fn.__name__)
            return result(False, None, "DB_ERROR", f"Lỗi cơ sở dữ liệu: {e}")
        except Exception as e:  # pragma: no cover
            logger.exception("TOOL_ERROR in %s", fn.__name__)
            return result(False, None, "TOOL_ERROR", str(e))
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _public_restaurant(r: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("ma_nha_hang", "ten_nha_hang", "dia_chi", "quan_huyen", "tinh_thanh", "loai_am_thuc",
            "gia_trung_binh", "diem_danh_gia", "so_luot_danh_gia", "gio_mo_cua", "gio_dong_cua", "so_dien_thoai",
            "mo_ta", "loai_hinh", "nguon_du_lieu", "trang_thai_du_lieu", "xac_minh", "dat_ban", "uoc_tinh")
    return {k: r.get(k) for k in keys}


# ---------------------------------------------------------------------- 1. tìm nhà hàng
@_guard
def search_restaurants(cuisine: Optional[str] = None, area: Optional[str] = None,
                       budget_max: Optional[int] = None, dietary: Optional[str] = None,
                       occasion: Optional[str] = None, rating_min: Optional[float] = None,
                       keyword: Optional[str] = None, limit: int = 6) -> Dict[str, Any]:
    """Tìm nhà hàng theo loại món, khu vực/thành phố, ngân sách/người, có món chay, dịp đi ăn, số sao."""
    if budget_max is not None and int(budget_max) <= 0:
        raise ParamError("Ngân sách phải > 0")
    if rating_min is not None and not 0 < float(rating_min) <= 5:
        raise ParamError("Số sao phải từ 1 đến 5")
    cuisine_c = entities.canonical_cuisine(cuisine) if cuisine else None
    out = []
    for r in db.list_restaurants():
        if cuisine_c and not entities.cuisine_matches(cuisine_c, r["loai_am_thuc"], r["mo_ta"]):
            continue
        if area and not entities.area_matches(area, r["quan_huyen"], r["dia_chi"], r["tinh_thanh"]):
            continue
        if budget_max and (r["gia_trung_binh"] is None or r["gia_trung_binh"] > int(budget_max)):
            continue   # quán chưa có giá không được coi là "rẻ"
        if dietary == "chay" and not r["co_mon_chay"]:
            continue
        if occasion and not entities.occasion_matches(occasion, r["mo_ta"]):
            continue
        if rating_min and (r["diem_danh_gia"] is None or r["diem_danh_gia"] < float(rating_min)):
            continue
        if keyword and not entities.text_contains(f"{r['ten_nha_hang']} {r['loai_am_thuc']}", keyword):
            continue
        item = _public_restaurant(r)
        item["co_mon_chay"] = bool(r["co_mon_chay"])
        out.append(item)
    total = len(out)
    return result(True, out[: max(1, min(int(limit), 10))], None, f"Tìm thấy {total} nhà hàng.")


# ---------------------------------------------------------------------- 2. chi tiết
@_guard
def get_restaurant_details(restaurant_id: str) -> Dict[str, Any]:
    """Thông tin chi tiết 1 nhà hàng: địa chỉ, giờ mở cửa, món, đánh giá."""
    nh = _v_restaurant(restaurant_id)
    data = _public_restaurant(nh)
    data["mon_an"] = [{"ten_mon": m["ten_mon"], "gia_vnd": m["gia_vnd"], "chay": bool(m["phu_hop_an_chay"])}
                      for m in nh["mon_an"]]
    data["danh_gia"] = [{"diem": d["diem"], "noi_dung": d["noi_dung"]} for d in nh["danh_gia"]]
    return result(True, data)


# ---------------------------------------------------------------------- 3. thực đơn
@_guard
def search_menu(restaurant_id: Optional[str] = None, vegetarian: bool = False,
                budget_max: Optional[int] = None, cuisine: Optional[str] = None,
                keyword: Optional[str] = None, restaurant_ids: Optional[List[str]] = None,
                limit: int = 10) -> Dict[str, Any]:
    """Tìm món ăn (luôn từ DB, không dùng RAG). keyword = tên món; restaurant_ids = chỉ trong các quán này."""
    if restaurant_id:
        _v_restaurant(restaurant_id)
    dishes = db.search_dishes(int(budget_max) if budget_max else None, vegetarian,
                              str(restaurant_id).upper() if restaurant_id else None)
    if restaurant_ids and not restaurant_id:
        allowed = {x.upper() for x in restaurant_ids}
        dishes = [d for d in dishes if d["ma_nha_hang"] in allowed]
    if keyword:
        dishes = [d for d in dishes
                  if entities.text_contains(f"{d['ten_mon']} {d['loai_mon'] or ''} {d['mo_ta'] or ''}", keyword)]
    elif cuisine and not restaurant_id and not restaurant_ids:
        canon = entities.canonical_cuisine(cuisine)
        rest_ok = {r["ma_nha_hang"] for r in db.list_restaurants()
                   if entities.cuisine_matches(canon, r["loai_am_thuc"], r["mo_ta"])}
        dishes = [d for d in dishes if d["ma_nha_hang"] in rest_ok
                  or entities.cuisine_matches(canon, d["loai_mon"] or "", d["ten_mon"])]
    data = [{"ten_mon": d["ten_mon"], "gia_vnd": d["gia_vnd"], "loai_mon": d["loai_mon"],
             "chay": bool(d["phu_hop_an_chay"]), "ma_nha_hang": d["ma_nha_hang"],
             "ten_nha_hang": d["ten_nha_hang"]} for d in dishes]
    return result(True, data[: int(limit)], None, f"Tìm thấy {len(data)} món.")


# ---------------------------------------------------------------------- 4. bàn trống
PERIODS = {"sáng": ("06:00", "10:59"), "trưa": ("11:00", "14:00"), "chiều": ("14:00", "17:29"),
           "tối": ("17:00", "22:00")}


@_guard
def check_table_availability(restaurant_id: str, date: str, time: Optional[str], party_size: int,
                             period: Optional[str] = None) -> Dict[str, Any]:
    """Kiểm tra bàn trống thật trong DB theo (nhà hàng, ngày, giờ, số khách).
    time=None -> trả danh sách giờ còn bàn phù hợp trong ngày (lọc theo buổi nếu có)."""
    nh = _v_restaurant(restaurant_id)
    if not nh.get("dat_ban"):
        return result(False, {"restaurant_id": nh["ma_nha_hang"], "ten_nha_hang": nh["ten_nha_hang"]},
                      "NOT_BOOKABLE",
                      f"{nh['ten_nha_hang']} chưa liên kết đặt bàn trên hệ thống (thông tin quán chưa xác minh). "
                      "Bạn vui lòng liên hệ trực tiếp quán.")
    d = _v_date(date)
    n = _v_party(party_size)
    now_hm = dt.datetime.now().strftime("%H:%M") if d == dt.date.today() else "00:00"
    if time is None:
        base = {"restaurant_id": nh["ma_nha_hang"], "ten_nha_hang": nh["ten_nha_hang"], "date": d.isoformat(),
                "time": None, "party_size": n, "period": period}
        if n > max(config.TABLE_SIZES):
            return result(True, {**base, "available": False, "reason": "LARGE_GROUP", "free_times": []},
                          None, "Nhóm đông, cần liên hệ quán.")
        slots = db.get_slots(nh["ma_nha_hang"], d.isoformat())
        if not slots:
            rng = db.data_date_range()
            return result(False, base, "NO_SLOT_DATA", f"Chưa có lịch bàn ngày {d.isoformat()}"
                          + (f"; hệ thống có dữ liệu từ {rng[0]} đến {rng[1]}." if rng else "."))
        lo, hi = PERIODS.get(period or "", ("00:00", "23:59"))
        free = sorted({x["gio"] for x in slots if x["loai_ban"] >= n and x["so_ban_trong"] > 0
                       and lo <= x["gio"] <= hi and x["gio"] > now_hm})
        return result(True, {**base, "available": bool(free), "free_times": free}, None,
                      "Các giờ còn bàn." if free else "Không còn giờ trống phù hợp.")
    t = _v_time(time)
    if d == dt.date.today() and t <= now_hm:
        return result(False, None, "PAST_TIME", "Giờ này hôm nay đã qua.")
    base = {"restaurant_id": nh["ma_nha_hang"], "ten_nha_hang": nh["ten_nha_hang"],
            "date": d.isoformat(), "time": t, "party_size": n}
    slots = db.get_slots(nh["ma_nha_hang"], d.isoformat())
    if not slots:
        rng = db.data_date_range()
        msg = (f"Chưa có lịch bàn ngày {d.isoformat()}; hệ thống có dữ liệu từ {rng[0]} đến {rng[1]}."
               if rng else "Chưa có dữ liệu lịch bàn.")
        return result(False, base, "NO_SLOT_DATA", msg)
    if n > max(config.TABLE_SIZES):
        return result(True, {**base, "available": False, "reason": "LARGE_GROUP", "suggestions": []},
                      None, f"Nhóm {n} người vượt bàn lớn nhất ({max(config.TABLE_SIZES)} chỗ), cần liên hệ quán.")
    fits = [s for s in slots if s["loai_ban"] >= n]
    at_time = [s for s in fits if s["gio"] == t and s["so_ban_trong"] > 0]
    times_free = sorted({s["gio"] for s in fits if s["so_ban_trong"] > 0})
    suggestions = sorted(times_free, key=lambda x: abs(_minutes(x) - _minutes(t)))[:3]
    if at_time:
        best = min(at_time, key=lambda s: s["loai_ban"])
        return result(True, {**base, "available": True, "table_size": best["loai_ban"],
                             "free_tables": best["so_ban_trong"], "suggestions": []},
                      None, "Còn bàn.")
    known_time = any(s["gio"] == t for s in slots)
    return result(True, {**base, "available": False,
                         "reason": "FULL" if known_time else "NO_SLOT_AT_TIME",
                         "suggestions": sorted(suggestions)},
                  None, "Hết bàn phù hợp." if known_time else "Không có khung giờ này.")


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


# ---------------------------------------------------------------------- 5-8. đặt bàn
def _booking_view(b: Dict[str, Any]) -> Dict[str, Any]:
    return {"booking_code": b["ma_dat_ban"], "restaurant_id": b["ma_nha_hang"],
            "ten_nha_hang": b["ten_nha_hang"], "dia_chi": b["dia_chi"], "date": b["ngay_dat"],
            "time": b["gio_dat"], "party_size": b["so_nguoi"], "status": b["trang_thai"],
            "customer_name": b.get("ten_khach"), "note": b.get("ghi_chu")}


@_guard
def create_reservation(restaurant_id: str, date: str, time: str, party_size: int,
                       customer_name: Optional[str] = None, phone: Optional[str] = None,
                       note: str = "") -> Dict[str, Any]:
    """Tạo đơn đặt bàn trong transaction. Chỉ trả mã khi COMMIT thành công."""
    nh = _v_restaurant(restaurant_id)
    if not nh.get("dat_ban"):
        return result(False, None, "NOT_BOOKABLE", f"{nh['ten_nha_hang']} chưa liên kết đặt bàn trên hệ thống.")
    d, t, n = _v_date(date), _v_time(time), _v_party(party_size)
    try:
        b = db.insert_booking(nh["ma_nha_hang"], d.isoformat(), t, n,
                              (customer_name or "Khách").strip()[:80], phone, note[:200])
    except LookupError:
        return result(False, None, "FULL", "Khung giờ này vừa hết bàn phù hợp, chưa tạo đơn.")
    return result(True, _booking_view(b), None, "Đã ghi nhận đơn đặt bàn.")


@_guard
def get_reservation(booking_code: str) -> Dict[str, Any]:
    """Tra cứu đơn đặt bàn theo mã."""
    b = db.get_booking(_v_code(booking_code))
    if not b:
        return result(False, None, "NOT_FOUND", f"Không tìm thấy mã {booking_code}.")
    return result(True, _booking_view(b))


@_guard
def update_reservation(booking_code: str, date: Optional[str] = None, time: Optional[str] = None,
                       party_size: Optional[int] = None) -> Dict[str, Any]:
    """Đổi ngày/giờ/số người của đơn trong 1 transaction."""
    code = _v_code(booking_code)
    b = db.get_booking(code)
    if not b:
        return result(False, None, "NOT_FOUND", f"Không tìm thấy mã {code}.")
    new_d = _v_date(date or b["ngay_dat"]).isoformat()
    new_t = _v_time(time or b["gio_dat"])
    new_n = _v_party(party_size or b["so_nguoi"])
    try:
        nb = db.modify_booking(code, new_d, new_t, new_n)
    except LookupError as e:
        reason = str(e).strip("'")
        msgs = {"FULL": "Khung giờ mới không còn bàn phù hợp, đơn được giữ nguyên.",
                "INACTIVE": "Đơn này đã hủy/không còn hiệu lực.", "NOT_FOUND": f"Không tìm thấy mã {code}."}
        return result(False, None, reason, msgs.get(reason, reason))
    return result(True, _booking_view(nb), None, "Đã cập nhật đơn.")


@_guard
def cancel_reservation(booking_code: str, reason: str = "Khách yêu cầu hủy") -> Dict[str, Any]:
    """Hủy đơn trong transaction và trả bàn."""
    code = _v_code(booking_code)
    try:
        b = db.cancel_booking(code, reason[:200])
    except LookupError as e:
        r = str(e).strip("'")
        msgs = {"INACTIVE": f"Đơn {code} đã hủy trước đó.", "NOT_FOUND": f"Không tìm thấy mã {code}."}
        return result(False, None, r, msgs.get(r, r))
    return result(True, _booking_view(b), None, "Đã hủy đơn.")


# ---------------------------------------------------------------------- 9. chính sách (RAG)
@_guard
def search_policy(query: str, restaurant_id: Optional[str] = None) -> Dict[str, Any]:
    """Tra cứu chính sách bằng RAG (FAISS). Dưới ngưỡng -> ok=False, POLICY_NOT_FOUND."""
    from actions import rag

    if not query or not query.strip():
        raise ParamError("Câu hỏi rỗng")
    hits = rag.search(query, restaurant_id=restaurant_id)
    if not hits:
        return result(False, [], "POLICY_NOT_FOUND", "Chưa tìm thấy thông tin chính sách phù hợp.")
    return result(True, hits)


@_guard
def get_support_ticket(ticket_code: str) -> Dict[str, Any]:
    """Tra cứu phiếu hỗ trợ HTxxx."""
    code = str(ticket_code or "").strip().upper()
    if not re.fullmatch(r"HT\d{3,}", code):
        raise ParamError(f"Mã phiếu không hợp lệ: {ticket_code!r}")
    t = db.get_ticket(code)
    if not t:
        return result(False, None, "NOT_FOUND", f"Không tìm thấy phiếu {code}.")
    return result(True, t)


# ---------------------------------------------------------------------- 10. phiếu hỗ trợ
@_guard
def create_support_ticket(category: str, description: str, booking_code: Optional[str] = None,
                          restaurant_id: Optional[str] = None, phone: Optional[str] = None) -> Dict[str, Any]:
    """Tạo phiếu hỗ trợ (transaction)."""
    if not description or not description.strip():
        raise ParamError("Thiếu mô tả sự cố")
    code = _v_code(booking_code) if booking_code else None
    t = db.insert_ticket(category[:80], description.strip()[:1000], code,
                         str(restaurant_id).upper() if restaurant_id else None, phone)
    return result(True, t, None, "Đã tạo phiếu hỗ trợ.")


TOOL_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "search_restaurants": search_restaurants,
    "get_restaurant_details": get_restaurant_details,
    "search_menu": search_menu,
    "check_table_availability": check_table_availability,
    "create_reservation": create_reservation,
    "get_reservation": get_reservation,
    "update_reservation": update_reservation,
    "cancel_reservation": cancel_reservation,
    "search_policy": search_policy,
    "create_support_ticket": create_support_ticket,
    "get_support_ticket": get_support_ticket,
}


class ToolExecutor:
    """Gọi tool theo tên, ghi lại vết (tên, tham số, kết quả) cho debug & kiểm thử."""

    def __init__(self, registry: Optional[Dict[str, Callable[..., Dict[str, Any]]]] = None) -> None:
        self.registry = registry or TOOL_REGISTRY
        self.trace: List[Dict[str, Any]] = []

    def __call__(self, name: str, **params: Any) -> Dict[str, Any]:
        if name not in self.registry:
            return result(False, None, "UNKNOWN_TOOL", name)
        t0 = time.time()
        res = self.registry[name](**params)
        self.trace.append({"tool": name, "params": params, "ok": res["ok"],
                           "error_code": res["error_code"], "ms": int((time.time() - t0) * 1000)})
        return res
