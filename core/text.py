"""Tiện ích xử lý chuỗi tiếng Việt dùng chung."""

import re
import unicodedata


def fold(text: str) -> str:
    """Bỏ dấu + chữ thường: 'Lẩu Nhà Mình' -> 'lau nha minh'."""
    if not text:
        return ""
    t = unicodedata.normalize("NFD", text)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return t.replace("đ", "d").replace("Đ", "D").lower()


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def has_diacritics(text: str) -> bool:
    return fold(text) != (text or "").lower()


def tokens(text: str):
    return re.findall(r"\w+", fold(text))


def word_in(phrase: str, text_folded: str) -> bool:
    """Tìm cụm từ (đã fold) theo ranh giới từ trong chuỗi đã fold."""
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text_folded) is not None
