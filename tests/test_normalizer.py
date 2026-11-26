import datetime as dt

import pytest

from core.normalizer import normalize_message

TODAY = dt.date(2026, 11, 17)  # Thứ Ba


def n(text):
    return normalize_message(text, today=TODAY)


@pytest.mark.parametrize("text,date,time,party", [
    ("Tối mai kiếm giúp mình quán lẩu 4 người, tầm 7 giờ nhé.", "2026-11-18", "19:00", 4),
    ("toi mai 7h toi 4 nguoi", "2026-11-18", "19:00", 4),
    ("12h trưa mai 3 ng", "2026-11-18", "12:00", 3),
    ("ngày 20/11 6 rưỡi tối, sáu người", "2026-11-20", "18:30", 6),
    ("thứ 7 lúc 18h30 cho nhóm 10", "2026-11-21", "18:30", 10),
    ("hôm nay 19:15 hai người", "2026-11-17", "19:15", 2),
    ("25 tháng 11 lúc 8 giờ tối", "2026-11-25", "20:00", None),
])
def test_date_time_party(text, date, time, party):
    m = n(text)
    assert (m.date, m.time, m.party_size) == (date, time, party)


@pytest.mark.parametrize("text", [
    "Tôi muốn tư vấn quán ăn", "Sau đó mình đi ăn", "Nhà hàng ở miền Nam", "mình muốn bay qua quận 1",
])
def test_no_spurious_number_conversion(text):
    m = n(text)
    assert m.normalized_text.split()[0] == text.split()[0]
    assert m.party_size is None and m.time is None


def test_khuyen_mai_is_not_tomorrow():
    assert n("khuyen mai gi khong").date is None
    assert n("Có khuyến mãi gì không").date is None


def test_bao_lau_is_not_lau_cuisine():
    assert n("Đến trễ thì giữ bàn bao lâu?").cuisines == []
    assert n("món nào ngon nhất").cuisines == []


def test_accent_restoration_unaccented_sentence():
    m = n("quan nao re ma co do chay ko")
    assert "quán" in m.normalized_text and "chay" in m.normalized_text
    assert m.budget_low_hint and m.cuisines == ["Chay"]


def test_party_delta():
    assert n("thêm 2 người").party_delta == 2
    assert n("bớt một người nha").party_delta == -1
    assert n("thêm 2 người").party_size is None


def test_ambiguous_bare_number():
    m = n("đổi sang 8")
    assert m.bare_numbers == [8] and m.time is None and m.party_size is None


def test_ordinal_not_weekday():
    m = n("chọn quán thứ 2")
    assert m.ordinal_ref == 2 and m.date is None


def test_budget_and_area():
    m = n("tim quan lau quan 5 duoi 250k")
    assert m.budget_max == 250000 and m.area == "Quận 5" and m.cuisines == ["Lẩu"]


def test_booking_code():
    assert n("huy ma DB002 giup minh").booking_code == "DB002"
