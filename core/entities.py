"""Từ điển thực thể (gazetteer) dùng để CHUẨN HÓA giá trị slot.

Đây là luật chuẩn hóa dữ liệu (được phép), không phải luật phân loại câu:
- Ẩm thực: từ đồng nghĩa -> giá trị chuẩn.
- Khu vực: suy ra từ cột quan_huyen trong DB + cách viết tắt (q1, quan 1...).
- Tên nhà hàng: đối sánh mờ với tên trong DB.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from core.text import fold, has_diacritics, nfc, word_in

# giá trị chuẩn -> các cách nói (CÓ DẤU). Câu có dấu được so khớp có dấu (tránh "bao lâu" -> lẩu,
# "ngon nhất" -> Nhật); câu không dấu được so khớp sau khi bỏ dấu cả hai phía.
CUISINE_SYNONYMS: Dict[str, List[str]] = {
    "Lẩu": ["lẩu", "hotpot"],
    "Nướng": ["nướng", "bbq", "thịt nướng", "đồ nướng"],
    "Nhật Bản": ["nhật bản", "món nhật", "đồ nhật", "quán nhật", "nhà hàng nhật", "nhật", "sushi", "sashimi", "ramen"],
    "Hàn Quốc": ["hàn quốc", "món hàn", "đồ hàn", "quán hàn", "nhà hàng hàn"],
    "Hải sản": ["hải sản", "ốc", "tôm cua", "đồ biển"],
    "Pizza": ["pizza", "pizzeria"],
    "Âu": ["món âu", "đồ âu", "món tây", "đồ tây", "kiểu âu", "kiểu tây", "châu âu", "phương tây", "âu"],
    "Á": ["món á", "đồ á", "kiểu á", "châu á", "á đông", "món á đông"],
    "Ý": ["món ý", "đồ ý", "mì ý", "spaghetti", "pasta"],
    "Thái": ["món thái", "đồ thái", "thái lan"],
    "Trung Hoa": ["món trung", "đồ trung", "món hoa", "dimsum", "dim sum", "trung hoa"],
    "Steak": ["steak", "bít tết", "bò bít tết", "beefsteak"],
    "Chay": ["chay", "ăn chay", "đồ chay", "món chay", "thuần chay", "vegan"],
    "Món Việt": ["món việt", "cơm nhà", "cơm tấm", "món ăn việt", "đồ việt"],
    "Cà phê": ["cà phê", "cafe", "coffee", "trà sữa", "bánh ngọt", "quán nước"],
}

# khi tìm trong DB, giá trị chuẩn được đối sánh với các chuỗi này (fold) trong loai_am_thuc/mo_ta
CUISINE_DB_TERMS: Dict[str, List[str]] = {
    "Lẩu": ["lau"],
    "Nướng": ["nuong"],
    "Nhật Bản": ["nhat ban", "sushi"],
    "Hàn Quốc": ["han quoc"],
    "Hải sản": ["hai san"],
    "Pizza": ["pizza"],
    # phong cách: khớp với loại ẩm thực của quán (DB demo + dataset Quy Nhơn)
    "Âu": ["mon au", "pizza", "mon y", "steak", "bit tet", "pizzeria"],
    "Á": ["mon a", "nhat ban", "han quoc", "sushi", "ramen", "thai", "trung hoa"],
    "Ý": ["mon y", "pizza", "mi y"],
    "Thái": ["thai"],
    "Trung Hoa": ["trung hoa"],
    "Steak": ["steak", "bit tet"],
    "Chay": ["chay"],
    "Món Việt": ["mon viet", "am thuc viet", "com gia dinh", "am thuc dia phuong"],
    "Cà phê": ["ca phe", "banh"],
}


def canonical_cuisine(value: Optional[str]) -> Optional[str]:
    """'đồ chay' -> 'Chay', 'sushi' -> 'Nhật Bản'. Không nhận ra -> giữ nguyên (đã title)."""
    if not value:
        return None
    found = find_cuisines(str(value))
    if found:
        return found[0]
    f = fold(str(value)).strip()
    for canon in CUISINE_SYNONYMS:
        if f == fold(canon):
            return canon
    return str(value).strip()


def _match_space(text: str):
    """Trả về (chuỗi để tìm, hàm biến đổi từ khóa) theo việc câu có dấu hay không."""
    if has_diacritics(text):
        return nfc(text).lower(), lambda w: nfc(w).lower()
    return fold(text), fold


def find_cuisines(text: str) -> List[str]:
    hay, conv = _match_space(text)
    found: List[Tuple[int, str]] = []
    for canon, syns in CUISINE_SYNONYMS.items():
        for s in sorted(syns, key=len, reverse=True):
            m = re.search(r"(?<!\w)" + re.escape(conv(s)) + r"(?!\w)", hay)
            if m:
                found.append((m.start(), canon))
                break
    return [c for _, c in sorted(found)]


def cuisine_matches(canon: str, loai_am_thuc: str, mo_ta: str) -> bool:
    terms = CUISINE_DB_TERMS.get(canon) or [fold(canon)]
    hay = fold(f"{loai_am_thuc} {mo_ta}")
    return any(word_in(t, hay) for t in terms)


# ------------------------------------------------------------------ khu vực
_AREA_RE = re.compile(r"(?<!\w)(?:quan|q\.?)\s*(\d{1,2})(?!\w)")
_NAMED_AREAS = ["tan binh", "phu nhuan", "binh thanh", "go vap", "thu duc", "tan phu", "binh tan",
                "nhon ly", "nhon hai", "ghenh rang", "eo gio", "ky co", "bai xep", "quy nhon nam",
                "quy nhon tay", "trung tam quy nhon", "xuan dieu"]
# thành phố: cách nói (đã bỏ dấu) -> tên chuẩn
_CITIES = {"quy nhon": "Quy Nhơn", "sai gon": "TP. Hồ Chí Minh", "ho chi minh": "TP. Hồ Chí Minh",
           "hcm": "TP. Hồ Chí Minh", "tphcm": "TP. Hồ Chí Minh", "ha noi": "Hà Nội", "da nang": "Đà Nẵng"}


def canonical_area(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    f = fold(str(value)).strip()
    for k, v in _CITIES.items():
        if word_in(k, f):
            return v
    m = _AREA_RE.search(f) or re.fullmatch(r"(\d{1,2})", f)
    if m:
        return f"Quận {int(m.group(1))}"
    for a in _NAMED_AREAS:
        if word_in(a, f):
            return _area_display(a)
    return None   # không phải tên quận/thành phố đã biết (vd "gần cửa sổ") -> không nhận làm khu vực


def _area_display(folded: str) -> str:
    table = {"nhon ly": "Nhơn Lý", "nhon hai": "Nhơn Hải", "ghenh rang": "Ghềnh Ráng", "eo gio": "Eo Gió",
             "ky co": "Kỳ Co", "bai xep": "Bãi Xếp", "quy nhon nam": "Quy Nhơn Nam",
             "quy nhon tay": "Quy Nhơn Tây", "trung tam quy nhon": "Trung tâm Quy Nhơn", "xuan dieu": "Xuân Diệu",
             "tan binh": "Tân Bình", "phu nhuan": "Phú Nhuận", "binh thanh": "Bình Thạnh",
             "go vap": "Gò Vấp", "thu duc": "Thủ Đức", "tan phu": "Tân Phú", "binh tan": "Bình Tân"}
    return table.get(folded, folded.title())


def find_area(text: str) -> Optional[str]:
    f = fold(text)
    for k, v in _CITIES.items():
        if word_in(k, f):
            return v
    m = _AREA_RE.search(f)
    if m:
        return f"Quận {int(m.group(1))}"
    for a in _NAMED_AREAS:
        if word_in(a, f):
            return _area_display(a)
    return None


def area_matches(area: str, quan_huyen: str, dia_chi: str, tinh_thanh: str = "") -> bool:
    hay = fold(f"{quan_huyen}, {dia_chi}, {tinh_thanh}")
    a = fold(area)
    if a in hay:
        return True
    return a.startswith("tp. ") and a[4:] in hay


# ------------------------------------------------------------------ dịp / mục đích đi ăn
OCCASIONS: Dict[str, List[str]] = {   # giá trị chuẩn -> cách nói (có dấu); so khớp với mo_ta nhà hàng
    "gia đình": ["gia đình", "cả nhà", "bố mẹ", "ông bà", "con nhỏ", "trẻ nhỏ"],
    "hẹn hò": ["hẹn hò", "lãng mạn", "người yêu", "cặp đôi", "kỷ niệm"],
    "nhóm bạn": ["nhóm bạn", "bạn bè", "đi nhóm", "hội bạn"],
    "nhóm đông": ["nhóm đông", "đoàn đông", "tiệc công ty", "liên hoan", "tất niên"],
    "gặp gỡ": ["gặp gỡ", "gặp đối tác", "tiếp khách", "nói chuyện"],
}
OCCASION_DB_TERMS = {"gia đình": ["gia dinh"], "hẹn hò": ["cap doi", "hen ho", "gap go"],
                     "nhóm bạn": ["nhom ban", "nhom"], "nhóm đông": ["nhom dong", "tiec"],
                     "gặp gỡ": ["gap go"]}


def find_occasion(text: str) -> Optional[str]:
    hay, conv = _match_space(text)
    for canon, syns in OCCASIONS.items():
        if any(re.search(r"(?<!\w)" + re.escape(conv(x)) + r"(?!\w)", hay) for x in syns):
            return canon
    return None


def canonical_occasion(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return find_occasion(str(value)) or (str(value).strip() if fold(str(value)) in
                                         {fold(k) for k in OCCASIONS} else None)


def occasion_matches(occ: str, mo_ta: str) -> bool:
    hay = fold(mo_ta)
    return any(word_in(t, hay) for t in OCCASION_DB_TERMS.get(occ, [fold(occ)]))


# ------------------------------------------------------------------ món ăn (tên trong DB)
_DISH_STOP = {"mon", "quan", "nha", "hang", "co", "khong", "nao", "gi", "an", "cho", "toi", "minh", "gia",
              "bao", "nhieu", "the", "nay", "do", "ban", "phan", "suat", "duoi", "tren", "chay",
              "nguoi", "set", "combo", "tre", "em", "be", "loai", "ngon", "re", "hai", "ba", "bon",
              "nhung", "cac", "vai", "may", "thu", "mot", "duoc", "phu", "hop", "khach", "hang"}


@lru_cache(maxsize=1)
def _dish_index() -> List[Tuple[str, str]]:
    """(tên món chữ thường có dấu, tên món bỏ dấu)."""
    try:
        from actions.db import read_conn

        with read_conn() as c:
            return [(nfc(r[0]).lower(), fold(r[0])) for r in c.execute("SELECT ten_mon FROM mon_an")]
    except Exception:
        return []


def text_contains(hay: str, needle: str) -> bool:
    """So khớp theo từ; câu có dấu thì so có dấu ("phở" không khớp "phô mai"), không dấu thì bỏ dấu."""
    if has_diacritics(needle):
        return word_in(nfc(needle).lower(), nfc(hay).lower())
    return word_in(fold(needle), fold(hay))


FOOD_HEADS = {"mi", "pho", "bun", "com", "chao", "ga", "bo", "ca", "oc", "tom", "muc", "lau", "banh", "xoi",
              "hu tieu", "nem", "goi", "cua", "ghe", "vit", "heo", "de", "sushi", "pizza", "ramen", "tra", "che"}


def find_dish_keyword(text: str) -> Optional[str]:
    """Cụm từ trong câu trùng với (một phần) tên món trong DB, vd "lẩu thái" -> khớp "Lẩu Thái hải sản"."""
    words = [w for w in re.findall(r"\w+", fold(text))]
    orig_words = re.findall(r"\w+", nfc(text).lower())
    accented = has_diacritics(text) and len(orig_words) == len(words)
    names = _dish_index()
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if any(w.isdigit() for w in gram):
                continue   # "2 người" không phải tên món (dù có món "Set nướng 2 người")
            if all(w in _DISH_STOP for w in gram) or (
                    n == 1 and gram[0] not in FOOD_HEADS and (len(gram[0]) < 4 or gram[0] in _DISH_STOP)):
                continue
            phrase = " ".join(gram)
            if accented:
                ophrase = " ".join(orig_words[i:i + n])
                hit = any(word_in(ophrase, name_l) for name_l, _ in names)
            else:
                hit = any(word_in(phrase, name_f) for _, name_f in names)
            if hit:
                # trả lại đúng chữ (có dấu) trong câu gốc
                orig = re.findall(r"\w+", text)
                return " ".join(orig[i:i + n]) if len(orig) == len(words) else phrase
    return None


# ------------------------------------------------------------------ nhà hàng
@lru_cache(maxsize=1)
def _restaurant_index() -> List[Tuple[str, str, str]]:
    try:
        from actions.db import list_restaurants

        return [(r["ma_nha_hang"], r["ten_nha_hang"], fold(r["ten_nha_hang"])) for r in list_restaurants()]
    except Exception:
        return []


def reset_restaurant_cache() -> None:
    _restaurant_index.cache_clear()
    _dish_index.cache_clear()


def find_restaurant(text: str) -> Optional[Tuple[str, str]]:
    """Tìm mã/tên nhà hàng được nhắc trong câu: theo mã NHxxx hoặc tên (cho phép thiếu dấu)."""
    if not text:
        return None
    m = re.search(r"\b(?:NH|QN)\d{3}\b", text.upper())
    idx = _restaurant_index()
    if m:
        for code, name, _ in idx:
            if code == m.group(0):
                return code, name
    f = fold(text)
    best = None
    generic = {"am", "thuc", "viet", "nha", "hang", "quan", "mon", "an", "com", "hai", "san", "chay", "lau"}
    for code, name, fname in idx:
        if all(w in generic for w in fname.split()):
            continue   # tên chung chung ("Ẩm thực Việt") trùng với cách nói thường ngày -> chỉ nhận qua mã
        if word_in(fname, f):
            return code, name
        # bỏ các từ chung để bắt "quán An Nhiên", "Sushi Sora"
        core = " ".join(w for w in fname.split() if w not in {"nha", "hang", "quan", "&", "restaurant"})
        if len(core) >= 6 and word_in(core, f):
            best = best or (code, name)
    return best


def strip_restaurant_name(text: str, name: str) -> str:
    """Bỏ tên quán khỏi câu (để "Lẩu Biển Quy Nhơn" không bị hiểu thành món Lẩu + khu Quy Nhơn)."""
    words = text.split()
    fw = [fold(w).strip(".,!?") for w in words]
    nf = fold(name).split()
    for i in range(len(words) - len(nf) + 1):
        if fw[i:i + len(nf)] == nf:
            return " ".join(words[:i] + ["quán"] + words[i + len(nf):])
    core = [w for w in nf if w not in {"nha", "hang", "quan", "&"}]
    for i in range(len(words) - len(core) + 1):
        if core and fw[i:i + len(core)] == core:
            return " ".join(words[:i] + ["quán"] + words[i + len(core):])
    return text


def canonical_restaurant(value: Optional[str]) -> Optional[Tuple[str, str]]:
    if not value:
        return None
    return find_restaurant(str(value))
