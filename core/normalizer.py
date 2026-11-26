"""Bộ chuẩn hóa câu tiếng Việt (luật cứng chỉ cho DỮ LIỆU: số, ngày, giờ, tiền, chính tả).

Không phân loại ý định ở đây. Đầu ra là văn bản đã chuẩn hóa + các "gợi ý thực thể"
(hint) để NLU đối chiếu, và để bộ NLU dự phòng dùng khi LLM không chạy.

Các điểm chính:
- Câu KHÔNG dấu -> thêm dấu theo từ điển cụm từ của miền đặt bàn ("toi mai" -> "tối mai").
- Số viết bằng chữ chỉ được đổi khi đứng trước đơn vị ("sáu người" -> "6 người"),
  tránh lỗi "tư vấn" -> "4 vấn", "sau đó" -> "6 đó".
- "tối mai" -> ngày cụ thể; "7h tối"/"tầm 7 giờ" -> 19:00.
- Số không có đơn vị ("đổi sang 8") được đánh dấu là MƠ HỒ (bare_numbers) để hỏi lại.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config
from core import entities
from core.text import fold, has_diacritics, nfc

# ----------------------------------------------------------------------------- chính tả / viết tắt
SLANG = {  # áp dụng cho mọi câu (so khớp theo từ, không phân biệt hoa thường)
    "conf": "còn", "coon": "còn", "ko": "không", "k": "không", "hok": "không", "khum": "không", "kh": "không",
    "dc": "được", "đc": "được", "j": "gì", "mik": "mình", "mk": "mình", "mn": "mọi người",
    "ng": "người", "ngừoi": "người", "nguời": "người", "ngưòi": "người",
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn", "ok": "ok", "oke": "ok", "okie": "ok",
    "bn": "bao nhiêu", "z": "vậy", "v": "vậy", "r": "rồi", "ah": "à", "ạh": "ạ",
}

# Từ điển thêm dấu cho câu không dấu (khóa đã bỏ dấu). Cụm dài được áp dụng trước.
ACCENT_PHRASES = {
    "toi mai": "tối mai", "toi nay": "tối nay", "trua mai": "trưa mai", "trua nay": "trưa nay",
    "chieu mai": "chiều mai", "chieu nay": "chiều nay", "sang mai": "sáng mai", "sang nay": "sáng nay",
    "ngay mai": "ngày mai", "ngay mot": "ngày mốt", "ngay kia": "ngày kia", "hom nay": "hôm nay",
    "cuoi tuan": "cuối tuần", "tuan sau": "tuần sau", "tuan nay": "tuần này", "chu nhat": "chủ nhật",
    "ma dat ban": "mã đặt bàn", "dat ban": "đặt bàn", "huy ban": "hủy bàn", "huy don": "hủy đơn",
    "doi gio": "đổi giờ", "doi ngay": "đổi ngày", "doi sang": "đổi sang", "doi lai": "đổi lại",
    "xac nhan": "xác nhận", "dong y": "đồng ý", "chinh sach": "chính sách", "tien coc": "tiền cọc",
    "dat coc": "đặt cọc", "den tre": "đến trễ", "den muon": "đến muộn", "hai san": "hải sản",
    "an chay": "ăn chay", "do chay": "đồ chay", "mon chay": "món chay", "thuc don": "thực đơn",
    "nha hang": "nhà hàng", "binh dan": "bình dân", "gia re": "giá rẻ", "mon an": "món ăn",
    "khuyen mai": "khuyến mãi", "phong rieng": "phòng riêng", "bao lau": "bao lâu",
    "quan nao": "quán nào", "con ban": "còn bàn", "ban trong": "bàn trống", "nhan vien": "nhân viên",
    "khieu nai": "khiếu nại", "ung dung": "ứng dụng", "bi loi": "bị lỗi", "dang nhap": "đăng nhập",
    "thanh toan": "thanh toán", "tra cuu": "tra cứu", "khong can": "không cần", "tang len": "tăng lên",
    "giam xuong": "giảm xuống", "mon viet": "món Việt", "ca phe": "cà phê", "ban oi": "bạn ơi",
    "cam on": "cảm ơn", "xin chao": "xin chào", "tam biet": "tạm biệt", "gan day": "gần đây",
    "phuc vu": "phục vụ", "thai do": "thái độ", "tinh sai": "tính sai", "hoa don": "hóa đơn",
    "nuong han": "nướng Hàn", "thit nuong": "thịt nướng", "an toi": "ăn tối", "an trua": "ăn trưa",
    "lam duoc gi": "làm được gì", "giu ban": "giữ bàn", "bao nhieu": "bao nhiêu", "the nao": "thế nào",
    "nhu the nao": "như thế nào", "co the": "có thể", "tre em": "trẻ em", "sinh nhat": "sinh nhật",
    "gui xe": "gửi xe", "mang theo": "mang theo", "thoi tiet": "thời tiết",
    "ho tro": "hỗ trợ", "trang tri": "trang trí", "phu thu": "phụ thu", "hoan coc": "hoàn cọc",
    "di ung": "dị ứng", "giu xe": "giữ xe", "do xe": "đỗ xe", "diem thuong": "điểm thưởng",
    "uu dai": "ưu đãi", "ngay le": "ngày lễ", "tre em": "trẻ em", "ghe tre em": "ghế trẻ em",
    "mon chay": "món chay", "an kieng": "ăn kiêng", "phong vip": "phòng VIP", "dat tiec": "đặt tiệc",
    "nguoi": "người", "khach": "khách", "lau": "lẩu", "nuong": "nướng", "re": "rẻ", "gio": "giờ",
    "huy": "hủy", "doi": "đổi", "them": "thêm", "bot": "bớt", "duoi": "dưới", "khong": "không",
    "co": "có", "mon": "món", "gia": "giá", "tim": "tìm", "kiem": "kiếm", "giup": "giúp",
    "minh": "mình", "duoc": "được", "gi": "gì", "nao": "nào", "coc": "cọc", "tre": "trễ",
    "nhom": "nhóm", "ban": "bàn", "dat": "đặt", "ngay": "ngày", "thu": "thứ", "toi": "tôi",
    "quan": "quán", "chon": "chọn", "dau tien": "đầu tiên", "cuoi": "cuối", "do": "đó", "nay": "này",
    "muon": "muốn", "can": "cần", "nua": "nữa", "thoi": "thôi", "di": "đi", "nhe": "nhé", "a": "ạ",
    "em": "em", "sang": "sang", "tam": "tầm", "khoang": "khoảng", "luc": "lúc", "vao": "vào",
    "cho": "cho", "trua": "trưa", "chieu": "chiều", "sach": "sách", "gan": "gần", "vua": "vừa",
    "hen": "hẹn", "xem": "xem", "giam": "giảm", "tang": "tăng", "len": "lên",
    "thanh": "thành", "rui": "rồi", "roi": "rồi", "dang": "đang", "nhieu": "nhiều", "it": "ít",
}

NUM_WORDS = {
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "sáu": 6, "bảy": 7, "bẩy": 7,
    "tám": 8, "chín": 9, "mười": 10, "mười một": 11, "mười hai": 12, "mười lăm": 15,
    "hai mươi": 20, "hai chục": 20, "chục": 10, "mươi": 10,
}
PARTY_UNITS = r"(?:người|khách|vị|suất|chỗ|đứa|mạng|ông|bạn)"
TIME_UNITS = r"(?:giờ|h|g|tiếng)"

WEEKDAYS = {  # Python weekday(): Thứ 2 = 0
    "thứ 2": 0, "thứ hai": 0, "thứ 3": 1, "thứ ba": 1, "thứ 4": 2, "thứ tư": 2,
    "thứ 5": 3, "thứ năm": 3, "thứ 6": 4, "thứ sáu": 4, "thứ 7": 5, "thứ bảy": 5,
    "chủ nhật": 6, "cn": 6,
}

CHEAP_WORDS = ["rẻ", "bình dân", "giá mềm", "hạt dẻ", "sinh viên", "tiết kiệm", "không đắt", "giá tốt"]

ORDINAL_WORDS = {"nhất": 1, "đầu": 1, "đầu tiên": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5}


@dataclass
class NormalizedMessage:
    raw_text: str
    normalized_text: str
    date: Optional[str] = None
    time: Optional[str] = None
    time_assumed_pm: bool = False
    party_size: Optional[int] = None
    party_delta: Optional[int] = None
    booking_code: Optional[str] = None
    budget_max: Optional[int] = None
    budget_low_hint: bool = False
    bare_numbers: List[int] = field(default_factory=list)
    ordinal_ref: Optional[int] = None          # 1-based; -1 = cuối cùng
    demonstrative_ref: bool = False            # "quán đó", "nhà hàng này"
    cuisines: List[str] = field(default_factory=list)
    area: Optional[str] = None
    restaurant: Optional[Tuple[str, str]] = None
    phone: Optional[str] = None
    dietary: Optional[str] = None              # "chay" = quán CÓ món chay (khác cuisine "Chay" = quán chay)
    rating_min: Optional[float] = None
    occasion: Optional[str] = None
    note: Optional[str] = None                 # yêu cầu về bàn: "gần cửa sổ", "không gần cửa ra vào"
    times: List[str] = field(default_factory=list)   # mọi mốc giờ trong câu (theo thứ tự)
    period: Optional[str] = None               # "trưa" | "tối"... khi không có giờ cụ thể
    dish: Optional[str] = None                 # cụm từ trùng tên món trong DB
    ticket_code: Optional[str] = None
    ticket_lookup: bool = False
    kids: bool = False                         # hỏi món/chỗ cho trẻ em
    food: Optional[str] = None                 # cụm món tự do: "ăn mì cay" -> "mì cay"
    asks_tables: bool = False                  # hỏi "còn bàn / bàn trống / còn chỗ"

    def hints(self) -> Dict[str, object]:
        """Gợi ý gửi cho LLM (chỉ các trường có giá trị)."""
        h: Dict[str, object] = {}
        for k in ("date", "time", "party_size", "party_delta", "booking_code", "budget_max", "area",
                  "dietary", "rating_min", "occasion", "note"):
            v = getattr(self, k)
            if v is not None:
                h[k] = v
        if self.budget_low_hint and self.budget_max is None:
            h["budget_max"] = config.CHEAP_BUDGET_VND
        if self.cuisines:
            h["cuisine"] = self.cuisines[-1]
        if self.restaurant:
            h["restaurant_id"] = self.restaurant[0]
        if self.ordinal_ref:
            h["chọn_kết_quả_thứ"] = self.ordinal_ref
        if self.bare_numbers:
            h["số_không_rõ_đơn_vị"] = self.bare_numbers
        return h


# ----------------------------------------------------------------------------- bước chuẩn hóa văn bản
def _sub_words(text: str, mapping: Dict[str, str], folded_keys: bool) -> str:
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", re.IGNORECASE)

    def repl(m: re.Match) -> str:
        key = m.group(1).lower()
        return mapping.get(key, m.group(1))

    return pattern.sub(repl, text)


def restore_accents(text: str) -> str:
    """Thêm dấu cho câu không dấu. Câu đã có dấu được giữ nguyên."""
    if has_diacritics(text):
        return text
    t = text
    # "quan 5" -> "quận 5"; "7h toi"/"7 gio toi" -> tối
    t = re.sub(r"(?<!\w)quan\s*(\d{1,2})(?!\d)", r"quận \1", t, flags=re.IGNORECASE)
    t = re.sub(r"(\d{1,2}\s*(?:h|gio|g)(?:\s*\d{2})?\s+)toi(?!\w)", r"\1tối", t, flags=re.IGNORECASE)
    t = re.sub(r"(\d{1,2}\s*(?:h|gio|g)(?:\s*\d{2})?\s+)sang(?!\w)", r"\1sáng", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\w)sau(\s+(?:nguoi|khach|gio))", r"sáu\1", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\w)bon(\s+(?:nguoi|khach|gio))", r"bốn\1", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\w)bay(\s+(?:nguoi|khach|gio))", r"bảy\1", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\w)tam(\s+(?:nguoi|khach|gio))", r"tám\1", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\w)nam(\s+(?:nguoi|khach|gio))", r"năm\1", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\w)chin(\s+(?:nguoi|khach|gio))", r"chín\1", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\w)muoi(\s+(?:nguoi|khach|gio))", r"mười\1", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\w)mot(\s+(?:nguoi|khach|gio))", r"một\1", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\w)thu\s+(\d)", r"thứ \1", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\w)ma(\s+DB\d)", r"mã\1", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\w)so(\s*\d)", r"số\1", t, flags=re.IGNORECASE)
    return _sub_words(t, ACCENT_PHRASES, folded_keys=True)


def words_to_digits(text: str) -> str:
    """Đổi số chữ -> số chỉ khi đứng trước đơn vị người/giờ hoặc sau 'thêm/bớt/nhóm'."""
    keys = sorted(NUM_WORDS, key=len, reverse=True)
    alt = "|".join(re.escape(k) for k in keys)
    pat = re.compile(rf"(?<!\w)({alt})(?=\s+(?:{PARTY_UNITS[3:-1]}|{TIME_UNITS[3:-1]}|rưỡi)(?!\w))",
                     re.IGNORECASE)
    text = pat.sub(lambda m: str(NUM_WORDS[m.group(1).lower()]), text)
    pat2 = re.compile(rf"(?<!\w)(thêm|bớt|nhóm|giảm)\s+({alt})(?!\w)", re.IGNORECASE)
    return pat2.sub(lambda m: f"{m.group(1)} {NUM_WORDS[m.group(2).lower()]}", text)


def clean_text(text: str) -> str:
    t = nfc(text or "")
    t = re.sub(r"\s+", " ", t).strip()
    t = restore_accents(t)          # trước khi thay viết tắt (viết tắt có thể sinh dấu)
    t = _sub_words(t, SLANG, folded_keys=False)
    t = words_to_digits(t)
    return t


# ----------------------------------------------------------------------------- trích xuất
class _Spans:
    def __init__(self) -> None:
        self.used: List[Tuple[int, int]] = []

    def add(self, m: re.Match, group: int = 0) -> None:
        self.used.append(m.span(group))

    def covered(self, pos: int) -> bool:
        return any(a <= pos < b for a, b in self.used)


def parse_date(text: str, today: Optional[dt.date] = None, spans: Optional[_Spans] = None) -> Optional[str]:
    today = today or dt.date.today()
    low = text.lower()
    spans = spans or _Spans()

    m = re.search(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)", low)
    if m:
        spans.add(m)
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return None
    m = re.search(r"(?<!\d)(\d{1,2})\s*[/\-.]\s*(\d{1,2})(?:\s*[/\-.]\s*(\d{2,4}))?(?!\d)", low)
    if m and not re.match(r"\s*(?:người|khách)", low[m.end():]):
        spans.add(m)
        d, mo = int(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else today.year
        if y < 100:
            y += 2000
        try:
            cand = dt.date(y, mo, d)
            if not m.group(3) and cand < today:
                cand = dt.date(y + 1, mo, d)
            return cand.isoformat()
        except ValueError:
            return None
    m = re.search(r"(?<!\d)(\d{1,2})\s*tháng\s*(\d{1,2})(?!\d)", low)
    if m:
        spans.add(m)
        try:
            cand = dt.date(today.year, int(m.group(2)), int(m.group(1)))
            return (cand if cand >= today else cand.replace(year=today.year + 1)).isoformat()
        except ValueError:
            return None
    m = re.search(r"(?<!\w)ngày\s+(\d{1,2})(?!\s*(?:người|khách|giờ|h|:|tháng))(?!\d)", low)
    if m:
        spans.add(m, 1)
        d = int(m.group(1))
        try:
            cand = today.replace(day=d)
            if cand < today:
                nm = (today.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
                cand = nm.replace(day=d)
            return cand.isoformat()
        except ValueError:
            return None

    if re.search(r"(?<!\w)(ngày mốt|ngày kia|mốt)(?!\w)", low):
        return (today + dt.timedelta(days=2)).isoformat()
    if re.search(r"(?<!\w)(?<!khuyến )(?<!ban )mai(?!\w)(?! mốt)", low):
        return (today + dt.timedelta(days=1)).isoformat()
    if re.search(r"(?<!\w)(hôm nay|tối nay|trưa nay|chiều nay|sáng nay|bây giờ|ngay bây giờ)(?!\w)", low):
        return today.isoformat()
    for name, wd in sorted(WEEKDAYS.items(), key=lambda kv: -len(kv[0])):
        m = re.search(rf"(?<!\w){re.escape(name)}(?!\w)", low)
        if m and not re.search(r"(?:quán|nhà hàng|cái|chỗ|lựa chọn|kết quả|phương án|option)\s*$",
                               low[:m.start()]):
            spans.add(m)
            delta = (wd - today.weekday()) % 7
            if "tuần sau" in low or "tuần tới" in low:
                delta = delta + 7 if delta != 0 else 7
            return (today + dt.timedelta(days=delta)).isoformat()
    if "cuối tuần" in low:
        delta = (5 - today.weekday()) % 7
        return (today + dt.timedelta(days=delta)).isoformat()
    return None


def _apply_period(hour: int, period: str, whole: str) -> Tuple[int, bool]:
    period = period or ""
    pm_words = ("chiều", "tối", "đêm", "pm")
    if period in pm_words or (not period and any(w in whole for w in ("tối", "chiều", "đêm"))):
        return (hour + 12 if hour < 12 else hour), False
    if period == "trưa" or (not period and "trưa" in whole):
        return (hour + 12 if 1 <= hour <= 3 else hour), False
    if period in ("sáng", "am") or (not period and "sáng" in whole):
        return hour, False
    # không có buổi: giờ 1-9 trong ngữ cảnh nhà hàng hiểu là chiều/tối
    if 1 <= hour <= 9:
        return hour + 12, True
    return hour, False


def parse_times(text: str, spans: Optional[_Spans] = None) -> Tuple[List[str], Optional[str], bool]:
    """Mọi mốc giờ trong câu. Trả (danh sách giờ, giờ chính, có_giả_định_chiều_tối).
    Giờ chính = giờ đứng sau "sang/thành/qua/lên/xuống" nếu có (vd "từ 19h sang 20h" -> 20:00)."""
    low = text.lower()
    spans = spans or _Spans()
    found: List[Tuple[int, str, bool]] = []
    rest = low
    offset = 0
    while True:
        local = _Spans()
        val, assumed = parse_time(rest, local)
        if not val:
            break
        a, b = local.used[-1]
        found.append((offset + a, val, assumed))
        spans.used.append((offset + a, offset + b))
        offset += b
        rest = low[offset:]
    if not found:
        return [], None, False
    main = found[0]
    for pos, val, assumed in found:
        if re.search(r"(?:sang|thành|qua|lên|xuống|đổi)\s*(?:lúc|tầm|khoảng)?\s*$", low[:pos]):
            main = (pos, val, assumed)
            break
    return [v for _, v, _ in found], main[1], main[2]


def parse_time(text: str, spans: Optional[_Spans] = None) -> Tuple[Optional[str], bool]:
    low = text.lower()
    spans = spans or _Spans()
    period_re = r"(?:\s*(sáng|trưa|chiều|tối|đêm|am|pm))?"
    patterns = [
        rf"(?<!\d)(\d{{1,2}})\s*(?::|h|giờ|g)\s*(\d{{2}})(?!\d){period_re}",   # 19:30, 7h30, 7 giờ 30
        rf"(?<!\d)(\d{{1,2}})\s*(?:h|giờ|g)?\s*rưỡi{period_re}",               # 7 rưỡi, 6 giờ rưỡi
        rf"(?<!\d)(\d{{1,2}})\s*(?:h|giờ|g)(?![a-zà-ỹ]){period_re}",           # 7h, 7 giờ tối
    ]
    for i, p in enumerate(patterns):
        m = re.search(p, low)
        if not m:
            continue
        hour = int(m.group(1))
        minute = int(m.group(2)) if i == 0 else (30 if i == 1 else 0)
        period = m.group(3) if i == 0 else m.group(2)
        if hour > 23 or minute > 59:
            continue
        spans.add(m)
        hour, assumed = _apply_period(hour, period or "", low)
        if hour > 23:
            continue
        return f"{hour:02d}:{minute:02d}", assumed
    return None, False


def parse_party(text: str, spans: _Spans) -> Tuple[Optional[int], Optional[int]]:
    low = text.lower()
    m = re.search(rf"(?<!\w)(thêm|tăng thêm|bớt|giảm|giảm bớt|bớt đi)\s*(\d{{1,2}})\s*{PARTY_UNITS}?(?!\w)", low)
    if m and (re.search(PARTY_UNITS, low[m.end(2):m.end(2) + 8]) or "người" in low or "khách" in low):
        spans.add(m)
        n = int(m.group(2))
        return None, (-n if m.group(1).startswith(("bớt", "giảm")) else n)
    for m in re.finditer(rf"(?<!\d)(\d{{1,3}})\s*{PARTY_UNITS}(?!\w)", low):
        if re.search(r"(?:nghìn|ngàn|k|đ|đồng|vnd)\s*/?\s*$", low[:m.start()]):
            continue   # "350 nghìn một người" là đơn giá, không phải số khách
        spans.add(m)
        n = int(m.group(1))
        return (n if 1 <= n <= config.MAX_PARTY_SIZE else None), None
    m = re.search(r"(?<!\w)(?:nhóm|bàn cho|bàn)\s*(\d{1,2})(?!\s*(?:giờ|h\b|:|/))(?!\d)", low)
    if m:
        spans.add(m, 1)
        n = int(m.group(1))
        return (n if 1 <= n <= config.MAX_PARTY_SIZE else None), None
    return None, None


def parse_budget(text: str, spans: _Spans) -> Optional[int]:
    low = text.lower()
    m = re.search(r"(?<!\w)(\d+(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(k|nghìn|ngàn|ngan|nghin|tr|triệu|củ|đồng|đ|vnd)(?!\w)", low)
    if not m:
        m2 = re.search(r"(?:dưới|tối đa|không quá|khoảng|tầm)\s*(\d{5,7})(?!\d)", low)
        if m2:
            spans.add(m2, 1)
            return int(m2.group(1))
        return None
    spans.add(m)
    num_s, unit = m.group(1), m.group(2)
    if unit in ("k", "nghìn", "ngàn", "ngan", "nghin"):
        return int(float(num_s.replace(",", ".")) * 1000)
    if unit in ("tr", "triệu", "củ"):
        return int(float(num_s.replace(",", ".")) * 1_000_000)
    return int(re.sub(r"[.,]", "", num_s))


def parse_ordinal(text: str) -> Tuple[Optional[int], bool]:
    low = text.lower()
    head = r"(?:quán|nhà hàng|cái|chỗ|lựa chọn|phương án|option|kết quả)"
    m = re.search(rf"{head}\s*(?:thứ|số)\s*(\d|nhất|hai|ba|bốn|tư|năm)(?!\w)", low) \
        or re.search(r"(?:chọn|lấy|đặt)\s*(?:quán\s*)?(?:số|thứ)\s*(\d)(?!\w)", low)
    if m:
        v = m.group(1)
        return (int(v) if v.isdigit() else ORDINAL_WORDS.get(v)), False
    m = re.search(rf"{head}\s*(đầu tiên|đầu|cuối cùng|cuối)(?!\w)", low)
    if m:
        return (1 if m.group(1).startswith("đầu") else -1), False
    if re.search(rf"{head}\s*(?:đó|này|ấy|kia|vừa rồi|vừa nãy|trên)(?!\w)", low):
        return None, True
    return None, False


def normalize_message(text: str, today: Optional[dt.date] = None) -> NormalizedMessage:
    clean = clean_text(text)
    spans = _Spans()
    nm = NormalizedMessage(raw_text=text, normalized_text=clean)

    bm = re.search(r"\b(DB\d{3,})\b", clean, re.IGNORECASE)
    if bm:
        spans.add(bm)
        nm.booking_code = bm.group(1).upper()
    pm = re.search(r"(?<!\d)(0\d{9,10})(?!\d)", clean)
    if pm:
        spans.add(pm)
        nm.phone = pm.group(1)

    nm.budget_max = parse_budget(clean, spans)
    nm.date = parse_date(clean, today=today, spans=spans)
    nm.times, nm.time, nm.time_assumed_pm = parse_times(clean, spans)
    if not nm.time:
        pm = re.search(r"(?<!\w)(trưa|tối|chiều|sáng)(?!\w)", clean.lower())
        nm.period = pm.group(1) if pm else None
    rm = re.search(r"(?<!\d)(\d(?:[.,]\d)?)\s*(?:sao|⭐)", clean.lower())
    if rm:
        spans.used.append(rm.span())
        nm.rating_min = float(rm.group(1).replace(",", "."))
    nm.party_size, nm.party_delta = parse_party(clean, spans)
    low = clean.lower()
    nm.budget_low_hint = any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", low) for w in CHEAP_WORDS)
    nm.ordinal_ref, nm.demonstrative_ref = parse_ordinal(clean)
    nm.restaurant = entities.find_restaurant(clean) or entities.find_restaurant(text)
    src_raw, src_clean = text, clean
    if nm.restaurant:
        src_raw = entities.strip_restaurant_name(text, nm.restaurant[1])
        src_clean = entities.strip_restaurant_name(clean, nm.restaurant[1])
    # thực thể tìm trên câu gốc trước (câu không dấu được so khớp không dấu), rồi mới đến câu đã chuẩn hóa
    nm.cuisines = entities.find_cuisines(src_raw) or entities.find_cuisines(src_clean)
    nm.area = entities.find_area(src_clean) or entities.find_area(src_raw)
    # "có món chay" = lọc quán CÓ món chay; "quán chay / đồ chay / ăn chay" = loại quán chay
    if "Chay" in nm.cuisines and not re.search(
            r"(?<!\w)(quán|nhà hàng|chỗ|đồ|ăn|thuần|bếp|tiệm)\s+chay(?!\w)", low):
        nm.cuisines = [c for c in nm.cuisines if c != "Chay"]
        nm.dietary = "chay"
    nm.occasion = entities.find_occasion(clean)
    notes = re.findall(r"(?:(?:đừng|không|tránh|không muốn)\s+)?(?:ngồi\s+)?(?:gần|cạnh|sát|xa)\s+"
                       r"(?:cửa sổ|cửa ra vào|cửa chính|cửa|nhà vệ sinh|toilet|bếp|điều hòa|quầy|sân khấu|lối đi)"
                       r"|(?:chỗ|bàn|góc)\s+(?:yên tĩnh|ngoài trời|trong nhà|ban công|view biển)"
                       r"|ghế (?:cho )?(?:trẻ em|em bé)", low)
    if notes:
        nm.note = ", ".join(dict.fromkeys(n.strip() for n in notes))
    nm.dish = entities.find_dish_keyword(text) or entities.find_dish_keyword(clean)
    fm = re.search(r"(?<!\w)(?:muốn ăn|thèm|ăn|món)\s+((?:[^\W\d_]+\s*){1,3}?)"
                   r"(?=\s*(?:thì|ở|có|không|nào|đâu|nha|nhé|ngon|giá|quán|nhà|cho|tại|$|[,.?!]))", low)
    if fm:
        phrase = fm.group(1).strip()
        generic = {"gì", "quán", "nhà hàng", "món", "đồ", "uống", "chỗ", "tối", "trưa", "sáng", "luôn", "đi"}
        first = fold(phrase.split()[0]) if phrase else ""
        if (phrase and phrase not in generic and not entities.find_cuisines(phrase)
                and (first in entities.FOOD_HEADS or entities.find_dish_keyword(phrase))):
            nm.food = phrase   # chỉ nhận khi giống tên món ăn ("mì cay", "bún cá"), không nhận "tiếp đi"
    nm.asks_tables = bool(re.search(r"(?<!\w)(còn bàn|bàn trống|còn chỗ|còn trống|trống bàn|chỗ trống|hết bàn)(?!\w)", low))
    nm.kids = bool(re.search(r"(?<!\w)(trẻ em|trẻ nhỏ|em bé|cho bé|các bé|con nít|con nhỏ)(?!\w)", low))
    tm = re.search(r"(HT\d{3,})", clean, re.IGNORECASE)
    if tm:
        nm.ticket_code = tm.group(1).upper()
    nm.ticket_lookup = bool(tm) or bool(re.search(
        r"(?<!\w)(mã|xem|thông tin|tra cứu|kiểm tra|cho .{0,6}biết)(?!\w).{0,25}phiếu", low))

    # số không có đơn vị và chưa thuộc thực thể nào -> có thể mơ hồ (8 giờ hay 8 người?)
    area_m = re.search(r"quận\s*\d{1,2}", low)
    for m in re.finditer(r"(?<![\w/:.-])(\d{1,2})(?![\w/:.-])", clean):
        if spans.covered(m.start()) or (area_m and area_m.start() <= m.start() < area_m.end()):
            continue
        if re.search(r"(?:thứ|số|quận|top)\s*$", low[:m.start()]):
            continue
        if re.match(r"\s*(?:phút|tiếng|ngày|tuổi|%|lần|sao|năm|tháng|món|phần)(?!\w)", low[m.end():]):
            continue
        nm.bare_numbers.append(int(m.group(1)))
    return nm


def fold_text(text: str) -> str:
    return fold(text)
