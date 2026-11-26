import datetime as dt
import sqlite3

from actions import db
from agent import tools as T

TOMORROW = (dt.date.today() + dt.timedelta(days=1)).isoformat()


def _free(rid, date, time):
    return {(s["loai_ban"]): s["so_ban_trong"] for s in db.get_slots(rid, date) if s["gio"] == time}


def test_availability_depends_on_party_size(temp_db):
    small = T.check_table_availability("NH001", TOMORROW, "19:00", 2)
    big = T.check_table_availability("NH001", TOMORROW, "19:00", 12)
    assert small["ok"] and big["data"]["reason"] == "LARGE_GROUP" and not big["data"]["available"]


def test_validation_errors(temp_db):
    assert T.check_table_availability("NH001", "2000-01-01", "19:00", 2)["error_code"] == "PAST_DATE"
    assert T.check_table_availability("NH001", TOMORROW, "25:00", 2)["error_code"] == "INVALID_PARAM"
    assert T.check_table_availability("NH999", TOMORROW, "19:00", 2)["error_code"] == "NOT_FOUND"
    assert T.check_table_availability("NH001", "2099-01-01", "19:00", 2)["error_code"] == "NO_SLOT_DATA"
    assert T.create_reservation("NH001", TOMORROW, "19:00", 0)["error_code"] == "INVALID_PARAM"


def test_create_update_cancel_transactions(temp_db):
    before = _free("NH002", TOMORROW, "18:00")
    r = T.create_reservation("NH002", TOMORROW, "18:00", 2)
    assert r["ok"] and r["data"]["booking_code"].startswith("DB")
    code = r["data"]["booking_code"]
    assert T.get_reservation(code)["data"]["status"] == "Đã xác nhận"
    assert sum(_free("NH002", TOMORROW, "18:00").values()) == sum(before.values()) - 1
    u = T.update_reservation(code, party_size=3)
    assert u["ok"] and u["data"]["party_size"] == 3
    c = T.cancel_reservation(code)
    assert c["ok"] and c["data"]["status"] == "Đã hủy"
    assert sum(_free("NH002", TOMORROW, "18:00").values()) == sum(before.values())
    assert T.cancel_reservation(code)["error_code"] == "INACTIVE"


def test_db_error_never_reports_success(temp_db, monkeypatch):
    """Lỗi DB giữa transaction -> rollback, ok=False, không có mã đặt bàn, không đổi bàn trống."""
    before = _free("NH002", TOMORROW, "18:00")
    n_before = len(sqlite3.connect(temp_db).execute("SELECT * FROM dat_ban").fetchall())

    def boom(conn, counter, prefix):
        raise sqlite3.OperationalError("disk I/O error (giả lập)")

    monkeypatch.setattr(db, "_next_code", boom)
    r = T.create_reservation("NH002", TOMORROW, "18:00", 2)
    assert not r["ok"] and r["error_code"] == "DB_ERROR" and r["data"] is None
    assert _free("NH002", TOMORROW, "18:00") == before
    assert len(sqlite3.connect(temp_db).execute("SELECT * FROM dat_ban").fetchall()) == n_before


def test_search_filters(temp_db):
    r = T.search_restaurants(cuisine="đồ chay", budget_max=150000)
    assert {"NH005", "NH011", "QN006"} <= {x["ma_nha_hang"] for x in r["data"]}   # QN006: giá ước tính 90k
    assert T.search_restaurants(cuisine="lẩu", area="Quận 1")["data"] == []


def test_rag_threshold(temp_db):
    assert T.search_policy("Đến trễ 20 phút thì quán còn giữ bàn không?")["ok"]
    assert T.search_policy("thời tiết ngày mai thế nào")["error_code"] == "POLICY_NOT_FOUND"


def test_extend_slots_adds_future_days_without_touching_existing(temp_db):
    """Chạy demo sau khi hết lịch: run.bat tự bổ sung ngày mới, không xóa đơn / không đổi khung giờ cũ."""
    future = dt.date.today() + dt.timedelta(days=20)
    r = T.create_reservation("NH002", TOMORROW, "18:00", 2)
    before = db.get_slots("NH002", TOMORROW)
    assert db.get_slots("NH001", future.isoformat()) == []
    assert db.extend_slots(today=future) > 0
    assert db.get_slots("NH001", future.isoformat())
    assert db.get_slots("NH002", TOMORROW) == before
    assert T.get_reservation(r["data"]["booking_code"])["ok"]
    assert db.extend_slots(today=future) == 0   # chạy lại không thêm trùng


def test_cuisine_styles_and_quy_nhon_dataset(temp_db):
    """Tìm theo phong cách (Âu / Á) và dataset nhà hàng thật ở Quy Nhơn."""
    au = {r["ma_nha_hang"] for r in T.search_restaurants(cuisine="món Âu", area="Quy Nhơn", limit=10)["data"]}
    assert {"QN020", "QN022", "QN026"} <= au            # TRE, 4Q (steak), Sisters Pizzeria
    a = {r["ma_nha_hang"] for r in T.search_restaurants(cuisine="đồ Á", limit=10)["data"]}
    assert {"QN017", "QN023", "NH003"} <= a             # món Á + Nhật
    nhon_ly = T.search_restaurants(cuisine="hải sản", area="Nhơn Lý", limit=10)["data"]
    assert len(nhon_ly) == 4 and all(r["quan_huyen"] == "Nhơn Lý" for r in nhon_ly)
    bun = T.search_restaurants(keyword="bún cá", limit=10)["data"]
    assert [r["ten_nha_hang"] for r in bun] == ["Bún cá Ngọc Liên"]


def test_quy_nhon_estimates_are_labeled(temp_db):
    """Dataset Quy Nhơn để trống giá/giờ: dùng số ƯỚC TÍNH (cờ uoc_tinh) để demo; số sao không bịa."""
    r = T.get_restaurant_details("QN020")["data"]            # TRE Restaurant (steak)
    assert r["uoc_tinh"] == 1 and r["gia_trung_binh"] == 400000 and r["gio_mo_cua"] == "11:00"
    assert r["diem_danh_gia"] is None                         # đánh giá sao để trống
    assert T.check_table_availability("QN020", TOMORROW, "19:00", 4)["ok"]
    from core.response_gen import render
    txt = render([{"type": "restaurant_details", "restaurant": r}])
    assert "(ước tính)" in txt and "Đánh giá: chưa có" in txt
