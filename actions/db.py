"""Tầng truy cập dữ liệu SQLite (schema v2).

Nguyên tắc:
- Mọi truy vấn dùng tham số (?), không bao giờ ghép chuỗi SQL từ dữ liệu người dùng / LLM.
- Mọi thao tác ghi chạy trong transaction `BEGIN IMMEDIATE`; lỗi -> rollback -> ném exception.
  Mã đặt bàn chỉ được trả về sau khi COMMIT thành công.
- Bàn trống được quản lý theo (nhà hàng, ngày, giờ, loại bàn = số chỗ ngồi), nên số khách
  thực sự ảnh hưởng đến kết quả kiểm tra bàn trống.
- Dữ liệu khung giờ demo được sinh tất định cho DEMO_DAYS ngày kể từ ngày khởi tạo.

Module KHÔNG tự xóa/khởi tạo lại DB khi import. Khởi tạo lại: `python scripts/init_db.py --reset`
(script đó sao lưu DB cũ trước).
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import sqlite3
import zlib
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import config

SCHEMA_VERSION = "4"
ACTIVE_STATUSES = ("Đã xác nhận", "Chờ xác nhận")


class SchemaOutdatedError(RuntimeError):
    """DB tồn tại nhưng không phải schema v2 -> cần chạy scripts/init_db.py --reset."""


# --------------------------------------------------------------------------- kết nối
def get_connection(path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or config.db_path(), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def read_conn() -> Iterator[sqlite3.Connection]:
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Transaction ghi: BEGIN IMMEDIATE ... COMMIT; mọi exception -> ROLLBACK và ném lại."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------------- schema
SCHEMA_SQL = """
CREATE TABLE meta (khoa TEXT PRIMARY KEY, gia_tri TEXT NOT NULL);
CREATE TABLE bo_dem (ten TEXT PRIMARY KEY, gia_tri INTEGER NOT NULL);
CREATE TABLE nha_hang (
    ma_nha_hang TEXT PRIMARY KEY, ten_nha_hang TEXT NOT NULL, dia_chi TEXT NOT NULL,
    quan_huyen TEXT, tinh_thanh TEXT, vi_do REAL, kinh_do REAL, loai_am_thuc TEXT,
    gia_trung_binh INTEGER, diem_danh_gia REAL, so_luot_danh_gia INTEGER DEFAULT 0,
    suc_chua INTEGER, gio_mo_cua TEXT, gio_dong_cua TEXT, so_dien_thoai TEXT, mo_ta TEXT,
    loai_hinh TEXT, nguon_du_lieu TEXT, ma_nguon TEXT, trang_thai_du_lieu TEXT,
    xac_minh INTEGER DEFAULT 0,      -- 1 = thông tin đã xác minh
    dat_ban INTEGER DEFAULT 0,       -- 1 = có lịch bàn trên hệ thống (nhận đặt bàn)
    uoc_tinh INTEGER DEFAULT 0       -- 1 = giá / giờ mở cửa / sức chứa là số ƯỚC TÍNH (CSV để trống)
);
CREATE TABLE mon_an (
    ma_mon TEXT PRIMARY KEY, ma_nha_hang TEXT NOT NULL REFERENCES nha_hang(ma_nha_hang),
    ten_mon TEXT NOT NULL, gia_vnd INTEGER NOT NULL, loai_mon TEXT, mo_ta TEXT,
    phu_hop_an_chay INTEGER DEFAULT 0
);
CREATE TABLE chinh_sach (
    ma_chinh_sach TEXT PRIMARY KEY, ma_nha_hang TEXT NOT NULL, loai_chinh_sach TEXT NOT NULL,
    noi_dung TEXT NOT NULL
);
CREATE TABLE danh_gia (
    ma_danh_gia TEXT PRIMARY KEY, ma_nha_hang TEXT NOT NULL REFERENCES nha_hang(ma_nha_hang),
    diem INTEGER NOT NULL, noi_dung TEXT, ngay_danh_gia TEXT
);
CREATE TABLE khung_gio_ban (
    ma_nha_hang TEXT NOT NULL REFERENCES nha_hang(ma_nha_hang),
    ngay TEXT NOT NULL, gio TEXT NOT NULL, loai_ban INTEGER NOT NULL,
    so_ban_tong INTEGER NOT NULL, so_ban_trong INTEGER NOT NULL,
    PRIMARY KEY (ma_nha_hang, ngay, gio, loai_ban),
    CHECK (so_ban_trong >= 0 AND so_ban_trong <= so_ban_tong)
);
CREATE TABLE dat_ban (
    ma_dat_ban TEXT PRIMARY KEY, ma_khach_hang TEXT, ten_khach TEXT, so_dien_thoai TEXT,
    ma_nha_hang TEXT NOT NULL REFERENCES nha_hang(ma_nha_hang),
    ngay_dat TEXT NOT NULL, gio_dat TEXT NOT NULL, so_nguoi INTEGER NOT NULL,
    loai_ban INTEGER, trang_thai TEXT NOT NULL, ghi_chu TEXT, tao_luc TEXT
);
CREATE TABLE yeu_cau_ho_tro (
    ma_yeu_cau TEXT PRIMARY KEY, ma_dat_ban TEXT, ma_nha_hang TEXT, so_dien_thoai TEXT,
    loai_su_co TEXT NOT NULL, mo_ta TEXT NOT NULL, trang_thai TEXT DEFAULT 'Chờ xử lý',
    thoi_gian_tao TEXT NOT NULL
);
"""


def schema_version(path: Optional[str] = None) -> Optional[str]:
    p = path or config.db_path()
    if not os.path.exists(p):
        return None
    conn = get_connection(p)
    try:
        row = conn.execute("SELECT gia_tri FROM meta WHERE khoa='schema_version'").fetchone()
        return row[0] if row else "1"
    except sqlite3.OperationalError:
        return "1"
    finally:
        conn.close()


def ensure_database() -> None:
    """Tạo DB nếu chưa có. Nếu DB cũ (schema v1) -> báo lỗi, KHÔNG tự xóa."""
    ver = schema_version()
    if ver is None:
        init_database(reset=False)
    elif ver != SCHEMA_VERSION:
        raise SchemaOutdatedError(
            f"CSDL {config.db_path()} đang ở schema v{ver}. "
            "Chạy: python scripts/init_db.py --reset (script tự sao lưu DB cũ)."
        )


def _read_csv(name: str) -> List[Dict[str, str]]:
    path = os.path.join(config.DATASET_DIR, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _table_mix(suc_chua: int) -> Dict[int, int]:
    """Số bàn mỗi loại suy ra từ sức chứa nhà hàng (tất định)."""
    return {
        2: max(2, suc_chua // 10),
        4: max(2, suc_chua // 8),
        6: max(1, suc_chua // 20),
        10: max(1, suc_chua // 35),
    }


def estimate_profile(loai_hinh: str, am_thuc: str) -> Dict[str, Any]:
    """Ước tính giá TB / giờ mở cửa / sức chứa theo LOẠI QUÁN khi dataset để trống.
    Đây là số ước tính để demo (luôn hiển thị kèm "(ước tính)"), không phải số liệu thật của quán."""
    from core.text import fold

    t = fold(f"{loai_hinh} {am_thuc}")
    if "steak" in t:
        return {"gia": 400000, "mo": "11:00", "dong": "22:30", "suc_chua": 50}
    if "pizza" in t or "mon y" in t:
        return {"gia": 200000, "mo": "11:00", "dong": "22:30", "suc_chua": 40}
    if "chay" in t:
        return {"gia": 90000, "mo": "07:00", "dong": "21:00", "suc_chua": 40}
    if "quan an" in t:
        if "oc" in t.split() or "hai san" in t:
            return {"gia": 120000, "mo": "15:00", "dong": "22:30", "suc_chua": 40}
        return {"gia": 60000, "mo": "06:30", "dong": "21:00", "suc_chua": 30}
    if "com gia dinh" in t:
        return {"gia": 100000, "mo": "10:00", "dong": "21:00", "suc_chua": 60}
    if "hai san" in t:
        return {"gia": 300000, "mo": "10:00", "dong": "22:30", "suc_chua": 80}
    if "nuong" in t or "bbq" in t:
        return {"gia": 220000, "mo": "15:00", "dong": "23:00", "suc_chua": 60}
    return {"gia": 180000, "mo": "10:00", "dong": "22:00", "suc_chua": 60}


def _time_slots(open_s: str, close_s: str) -> List[str]:
    """Khung giờ trưa 11:00-13:30 và tối 17:00-21:00 (mỗi 30'), nằm trong giờ mở cửa,
    và kết thúc trước giờ đóng cửa ít nhất 60 phút."""
    def to_min(s: str) -> int:
        h, m = s.strip().split(":")[:2]
        return int(h) * 60 + int(m)

    o, c = to_min(open_s or "10:00"), to_min(close_s or "22:00")
    cands = list(range(11 * 60, 13 * 60 + 31, 30)) + list(range(17 * 60, 21 * 60 + 1, 30))
    return [f"{m // 60:02d}:{m % 60:02d}" for m in cands if o <= m <= c - 60]


def _occupancy(key: str, gio: str) -> float:
    """Tỉ lệ bàn đã kín (tất định theo key), cao điểm 18:30-19:30."""
    r = (zlib.crc32(key.encode("utf-8")) % 1000) / 1000.0
    peak = gio in ("18:30", "19:00", "19:30")
    base = 0.45 if peak else 0.2          # loại bàn có nhiều bàn luôn còn >= 1; bàn lớn ít bàn có thể kín
    return min(0.9, base + r * (0.45 if peak else 0.4))


def init_database(reset: bool = False, today: Optional[dt.date] = None) -> None:
    """Khởi tạo schema v2 và nạp dữ liệu demo. reset=True sẽ xóa DB hiện có (gọi qua init_db.py)."""
    path = config.db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        if not reset:
            return
        os.remove(path)
    today = today or dt.date.today()

    conn = get_connection(path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
        conn.execute("INSERT INTO meta VALUES ('seed_date', ?)", (today.isoformat(),))

        # 1) quán DEMO (hư cấu, đủ thông tin, có lịch bàn để demo đặt / đổi / hủy)
        restaurants = _read_csv("nha_hang.csv")
        for r in restaurants:
            conn.execute(
                "INSERT INTO nha_hang VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["ma_nha_hang"], r["ten_nha_hang"], r["dia_chi"], r["quan_huyen"], r["tinh_thanh"],
                 float(r["vi_do"] or 0), float(r["kinh_do"] or 0), r["loai_am_thuc"],
                 int(r["gia_trung_binh"] or 0), float(r["diem_danh_gia"] or 0),
                 int(r["so_luot_danh_gia"] or 0), int(r["suc_chua"] or 0),
                 r["gio_mo_cua"], r["gio_dong_cua"], r["so_dien_thoai"], r["mo_ta"],
                 "Nhà hàng", "demo", r["ma_nha_hang"], "Dữ liệu demo", 1, 1, 0),
            )
        # 2) quán THẬT ở Quy Nhơn (dataset_nha_hang_quy_nhon): chỉ tra cứu, trường trống giữ NULL,
        #    KHÔNG sinh lịch bàn (không có giờ mở cửa / sức chứa thật -> không được bịa)
        for r in _read_csv("nha_hang_quy_nhon.csv"):
            def num(v, cast):
                v = (v or "").strip()
                return cast(v) if v else None
            ma = "QN" + r["ma_nha_hang"].strip()[2:]
            gio = (r.get("gio_mo_cua") or "").strip()
            mo, dong = ([x.strip() for x in gio.split("-", 1)] if "-" in gio else (None, None))
            verified = 1 if (r.get("trang_thai_hoat_dong") or "").strip().lower() == "đã xác minh" else 0
            est = estimate_profile(r["loai_hinh"], r["am_thuc_chinh"])
            gia = num(r.get("gia_tham_khao"), int)
            estimated = 1 if (gia is None or not mo) else 0
            gia = gia if gia is not None else est["gia"]
            mo, dong = (mo, dong) if mo and dong else (est["mo"], est["dong"])
            conn.execute(
                "INSERT INTO nha_hang VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ma, r["ten_nha_hang"].strip(), r["dia_chi"].strip(), r["khu_vuc"].strip(), "Quy Nhơn",
                 num(r.get("toa_do_vi_do"), float), num(r.get("toa_do_kinh_do"), float),
                 r["am_thuc_chinh"].strip(), gia, num(r.get("diem_danh_gia"), float),
                 num(r.get("so_luong_danh_gia"), int), est["suc_chua"], mo, dong,
                 (r.get("so_dien_thoai") or "").strip() or None,
                 f"{r['loai_hinh'].strip()} {r['am_thuc_chinh'].strip().lower()} tại {r['khu_vuc'].strip()}",
                 r["loai_hinh"].strip(), r["nguon_tham_khao"].strip(), r["ma_nha_hang"].strip(),
                 r["trang_thai_du_lieu"].strip(), verified, 1, estimated),
            )
        for r in _read_csv("mon_an.csv"):
            chay = 1 if r["phu_hop_an_chay"].strip().lower() in ("true", "1", "yes") else 0
            conn.execute("INSERT INTO mon_an VALUES (?,?,?,?,?,?,?)",
                         (r["ma_mon"], r["ma_nha_hang"], r["ten_mon"], int(r["gia_vnd"]),
                          r["loai_mon"], r["mo_ta"], chay))
        for r in _read_csv("chinh_sach.csv"):
            conn.execute("INSERT INTO chinh_sach VALUES (?,?,?,?)",
                         (r["ma_chinh_sach"], r["ma_nha_hang"], r["loai_chinh_sach"], r["noi_dung"]))
        for r in _read_csv("danh_gia_mau.csv"):
            conn.execute("INSERT INTO danh_gia VALUES (?,?,?,?,?)",
                         (r["ma_danh_gia"], r["ma_nha_hang"], int(r["diem"]), r["noi_dung"],
                          r["ngay_danh_gia"]))

        # Khung giờ theo loại bàn cho DEMO_DAYS ngày kể từ hôm nay
        for r in restaurants:
            mix = _table_mix(int(r["suc_chua"] or 0))
            for d in range(config.DEMO_DAYS):
                ngay = (today + dt.timedelta(days=d)).isoformat()
                for gio in _time_slots(r["gio_mo_cua"], r["gio_dong_cua"]):
                    for loai, tong in mix.items():
                        busy = round(tong * _occupancy(f"{r['ma_nha_hang']}|{d}|{gio}|{loai}", gio))
                        conn.execute("INSERT INTO khung_gio_ban VALUES (?,?,?,?,?,?)",
                                     (r["ma_nha_hang"], ngay, gio, loai, tong, max(0, tong - busy)))

        # Đơn đặt bàn mẫu: dời ngày để đơn sớm nhất rơi vào ngày mai (giữ khoảng cách giữa các đơn)
        bookings = _read_csv("dat_ban_mau.csv")
        if bookings:
            first = min(dt.date.fromisoformat(b["ngay_dat"]) for b in bookings)
            shift = (today + dt.timedelta(days=1)) - first
            max_num = 0
            for b in bookings:
                ngay = (dt.date.fromisoformat(b["ngay_dat"]) + shift).isoformat()
                so_nguoi = int(b["so_nguoi"])
                loai = next((t for t in config.TABLE_SIZES if t >= so_nguoi), None)
                conn.execute(
                    "INSERT INTO dat_ban VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (b["ma_dat_ban"], b["ma_khach_hang"], "Khách mẫu", None, b["ma_nha_hang"], ngay,
                     b["gio_dat"][:5], so_nguoi, loai, b["trang_thai"], b["ghi_chu"],
                     dt.datetime.now().isoformat(timespec="seconds")),
                )
                if b["trang_thai"] in ACTIVE_STATUSES and loai:
                    conn.execute(
                        "UPDATE khung_gio_ban SET so_ban_trong = so_ban_trong - 1 "
                        "WHERE ma_nha_hang=? AND ngay=? AND gio=? AND loai_ban=? AND so_ban_trong > 0",
                        (b["ma_nha_hang"], ngay, b["gio_dat"][:5], loai))
                max_num = max(max_num, int(b["ma_dat_ban"][2:]))
            conn.execute("INSERT INTO bo_dem VALUES ('dat_ban', ?)", (max_num,))
        else:
            conn.execute("INSERT INTO bo_dem VALUES ('dat_ban', 0)")
        conn.execute("INSERT INTO bo_dem VALUES ('yeu_cau_ho_tro', 0)")
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()
    extend_slots(today)   # lịch bàn cho mọi quán nhận đặt bàn (kể cả dataset Quy Nhơn)


def extend_slots(today: Optional[dt.date] = None) -> int:
    """Bổ sung khung giờ cho các ngày còn thiếu trong [today, today + DEMO_DAYS).
    KHÔNG xóa/sửa dữ liệu có sẵn (INSERT OR IGNORE). Trả về số dòng thêm mới."""
    today = today or dt.date.today()
    added = 0
    with transaction() as conn:
        restaurants = conn.execute("SELECT ma_nha_hang, suc_chua, gio_mo_cua, gio_dong_cua FROM nha_hang "
                                   "WHERE dat_ban = 1").fetchall()
        for r in restaurants:
            mix = _table_mix(int(r["suc_chua"] or 0))
            for d in range(config.DEMO_DAYS):
                day = today + dt.timedelta(days=d)
                ngay = day.isoformat()
                if conn.execute("SELECT 1 FROM khung_gio_ban WHERE ma_nha_hang=? AND ngay=? LIMIT 1",
                                (r["ma_nha_hang"], ngay)).fetchone():
                    continue
                for gio in _time_slots(r["gio_mo_cua"], r["gio_dong_cua"]):
                    for loai, tong in mix.items():
                        key = f"{r['ma_nha_hang']}|{day.toordinal()}|{gio}|{loai}"
                        busy = round(tong * _occupancy(key, gio))
                        cur = conn.execute("INSERT OR IGNORE INTO khung_gio_ban VALUES (?,?,?,?,?,?)",
                                           (r["ma_nha_hang"], ngay, gio, loai, tong, max(0, tong - busy)))
                        added += cur.rowcount
    return added


# --------------------------------------------------------------------------- đọc
def _next_code(conn: sqlite3.Connection, counter: str, prefix: str) -> str:
    conn.execute("UPDATE bo_dem SET gia_tri = gia_tri + 1 WHERE ten = ?", (counter,))
    val = conn.execute("SELECT gia_tri FROM bo_dem WHERE ten = ?", (counter,)).fetchone()[0]
    return f"{prefix}{val:03d}"


def list_restaurants() -> List[Dict[str, Any]]:
    with read_conn() as c:
        rows = c.execute("""
            SELECT n.*, EXISTS(SELECT 1 FROM mon_an m WHERE m.ma_nha_hang = n.ma_nha_hang
                               AND m.phu_hop_an_chay = 1) AS co_mon_chay
            FROM nha_hang n ORDER BY diem_danh_gia DESC, so_luot_danh_gia DESC""").fetchall()
    return [dict(r) for r in rows]


def get_restaurant(ma_nha_hang: str) -> Optional[Dict[str, Any]]:
    with read_conn() as c:
        row = c.execute("SELECT * FROM nha_hang WHERE ma_nha_hang = ?", (ma_nha_hang,)).fetchone()
        if not row:
            return None
        nh = dict(row)
        nh["mon_an"] = [dict(r) for r in c.execute(
            "SELECT * FROM mon_an WHERE ma_nha_hang = ? ORDER BY gia_vnd", (ma_nha_hang,))]
        nh["danh_gia"] = [dict(r) for r in c.execute(
            "SELECT * FROM danh_gia WHERE ma_nha_hang = ?", (ma_nha_hang,))]
    return nh


def search_dishes(ngan_sach_max: Optional[int], chay: Optional[bool],
                  ma_nha_hang: Optional[str]) -> List[Dict[str, Any]]:
    sql = """SELECT m.*, n.ten_nha_hang, n.quan_huyen FROM mon_an m
             JOIN nha_hang n ON n.ma_nha_hang = m.ma_nha_hang WHERE 1=1"""
    params: List[Any] = []
    if ngan_sach_max:
        sql += " AND m.gia_vnd <= ?"
        params.append(ngan_sach_max)
    if chay:
        sql += " AND m.phu_hop_an_chay = 1"
    if ma_nha_hang:
        sql += " AND m.ma_nha_hang = ?"
        params.append(ma_nha_hang)
    sql += " ORDER BY m.gia_vnd"
    with read_conn() as c:
        return [dict(r) for r in c.execute(sql, params)]


def get_slots(ma_nha_hang: str, ngay: str) -> List[Dict[str, Any]]:
    with read_conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT gio, loai_ban, so_ban_tong, so_ban_trong FROM khung_gio_ban "
            "WHERE ma_nha_hang = ? AND ngay = ? ORDER BY gio, loai_ban", (ma_nha_hang, ngay))]


def get_booking(ma_dat_ban: str) -> Optional[Dict[str, Any]]:
    with read_conn() as c:
        row = c.execute("""SELECT d.*, n.ten_nha_hang, n.dia_chi FROM dat_ban d
                           JOIN nha_hang n ON n.ma_nha_hang = d.ma_nha_hang
                           WHERE d.ma_dat_ban = ?""", (ma_dat_ban.upper(),)).fetchone()
    return dict(row) if row else None


def get_ticket(ma_yeu_cau: str) -> Optional[Dict[str, Any]]:
    with read_conn() as c:
        row = c.execute("SELECT * FROM yeu_cau_ho_tro WHERE ma_yeu_cau = ?", (ma_yeu_cau.upper(),)).fetchone()
    return dict(row) if row else None


def data_date_range() -> Optional[tuple]:
    with read_conn() as c:
        row = c.execute("SELECT MIN(ngay), MAX(ngay) FROM khung_gio_ban").fetchone()
    return (row[0], row[1]) if row and row[0] else None


# --------------------------------------------------------------------------- ghi (transaction)
def _pick_table(conn: sqlite3.Connection, ma_nh: str, ngay: str, gio: str,
                so_nguoi: int) -> Optional[int]:
    row = conn.execute(
        "SELECT loai_ban FROM khung_gio_ban WHERE ma_nha_hang=? AND ngay=? AND gio=? "
        "AND loai_ban >= ? AND so_ban_trong > 0 ORDER BY loai_ban LIMIT 1",
        (ma_nh, ngay, gio, so_nguoi)).fetchone()
    return row[0] if row else None


def insert_booking(ma_nha_hang: str, ngay: str, gio: str, so_nguoi: int,
                   ten_khach: str, so_dien_thoai: Optional[str], ghi_chu: str) -> Dict[str, Any]:
    """Trả về dict đơn mới (đã COMMIT) hoặc ném LookupError('FULL') nếu hết bàn."""
    with transaction() as conn:
        loai = _pick_table(conn, ma_nha_hang, ngay, gio, so_nguoi)
        if loai is None:
            raise LookupError("FULL")
        conn.execute("UPDATE khung_gio_ban SET so_ban_trong = so_ban_trong - 1 "
                     "WHERE ma_nha_hang=? AND ngay=? AND gio=? AND loai_ban=?",
                     (ma_nha_hang, ngay, gio, loai))
        code = _next_code(conn, "dat_ban", "DB")
        conn.execute("INSERT INTO dat_ban VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (code, None, ten_khach, so_dien_thoai, ma_nha_hang, ngay, gio, so_nguoi, loai,
                      "Đã xác nhận", ghi_chu, dt.datetime.now().isoformat(timespec="seconds")))
    return get_booking(code)  # type: ignore[return-value]


def modify_booking(ma_dat_ban: str, ngay: str, gio: str, so_nguoi: int) -> Dict[str, Any]:
    """Đổi ngày/giờ/số người trong 1 transaction. Ném LookupError('NOT_FOUND'|'INACTIVE'|'FULL')."""
    with transaction() as conn:
        b = conn.execute("SELECT * FROM dat_ban WHERE ma_dat_ban = ?", (ma_dat_ban,)).fetchone()
        if not b:
            raise LookupError("NOT_FOUND")
        if b["trang_thai"] not in ACTIVE_STATUSES:
            raise LookupError("INACTIVE")
        if b["loai_ban"]:
            conn.execute("UPDATE khung_gio_ban SET so_ban_trong = MIN(so_ban_tong, so_ban_trong + 1) "
                         "WHERE ma_nha_hang=? AND ngay=? AND gio=? AND loai_ban=?",
                         (b["ma_nha_hang"], b["ngay_dat"], b["gio_dat"], b["loai_ban"]))
        loai = _pick_table(conn, b["ma_nha_hang"], ngay, gio, so_nguoi)
        if loai is None:
            raise LookupError("FULL")  # rollback trả lại trạng thái cũ
        conn.execute("UPDATE khung_gio_ban SET so_ban_trong = so_ban_trong - 1 "
                     "WHERE ma_nha_hang=? AND ngay=? AND gio=? AND loai_ban=?",
                     (b["ma_nha_hang"], ngay, gio, loai))
        conn.execute("UPDATE dat_ban SET ngay_dat=?, gio_dat=?, so_nguoi=?, loai_ban=? "
                     "WHERE ma_dat_ban=?", (ngay, gio, so_nguoi, loai, ma_dat_ban))
    return get_booking(ma_dat_ban)  # type: ignore[return-value]


def cancel_booking(ma_dat_ban: str, ly_do: str) -> Dict[str, Any]:
    with transaction() as conn:
        b = conn.execute("SELECT * FROM dat_ban WHERE ma_dat_ban = ?", (ma_dat_ban,)).fetchone()
        if not b:
            raise LookupError("NOT_FOUND")
        if b["trang_thai"] not in ACTIVE_STATUSES:
            raise LookupError("INACTIVE")
        note = ((b["ghi_chu"] or "") + f" | Hủy: {ly_do}").strip(" |")
        conn.execute("UPDATE dat_ban SET trang_thai='Đã hủy', ghi_chu=? WHERE ma_dat_ban=?",
                     (note, ma_dat_ban))
        if b["loai_ban"]:
            conn.execute("UPDATE khung_gio_ban SET so_ban_trong = MIN(so_ban_tong, so_ban_trong + 1) "
                         "WHERE ma_nha_hang=? AND ngay=? AND gio=? AND loai_ban=?",
                         (b["ma_nha_hang"], b["ngay_dat"], b["gio_dat"], b["loai_ban"]))
    return get_booking(ma_dat_ban)  # type: ignore[return-value]


def insert_ticket(loai_su_co: str, mo_ta: str, ma_dat_ban: Optional[str],
                  ma_nha_hang: Optional[str], so_dien_thoai: Optional[str]) -> Dict[str, Any]:
    now = dt.datetime.now().isoformat(sep=" ", timespec="seconds")
    with transaction() as conn:
        code = _next_code(conn, "yeu_cau_ho_tro", "HT")
        conn.execute("INSERT INTO yeu_cau_ho_tro VALUES (?,?,?,?,?,?,?,?)",
                     (code, ma_dat_ban, ma_nha_hang, so_dien_thoai, loai_su_co, mo_ta,
                      "Chờ xử lý", now))
    return {"ma_yeu_cau": code, "loai_su_co": loai_su_co, "ma_dat_ban": ma_dat_ban,
            "thoi_gian_tao": now, "trang_thai": "Chờ xử lý"}
