"""Khởi tạo CSDL demo (schema v2) và FAISS index.

    python scripts/init_db.py            # tạo nếu chưa có
    python scripts/init_db.py --reset    # SAO LƯU DB cũ vào backup/<timestamp>/ rồi tạo lại
    python scripts/init_db.py --index    # build lại FAISS index chính sách
"""

import argparse
import datetime as dt
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
from actions import db  # noqa: E402


def backup_db() -> str:
    src = config.db_path()
    dest_dir = os.path.join(config.BACKUP_DIR, dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))
    return dest_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="sao lưu rồi tạo lại DB")
    ap.add_argument("--index", action="store_true", help="đảm bảo FAISS index mới nhất (tự build nếu tài liệu đổi)")
    ap.add_argument("--force-index", action="store_true", help="build lại FAISS index bắt buộc")
    ap.add_argument("--if-outdated", action="store_true", help="chỉ reset nếu DB không phải schema v2")
    args = ap.parse_args()

    ver = db.schema_version()
    reset = args.reset or (args.if_outdated and ver not in (None, db.SCHEMA_VERSION))
    if reset and ver is not None:
        print(f"Đã sao lưu DB cũ (schema v{ver}) vào: {backup_db()}")
    db.init_database(reset=reset)
    added = db.extend_slots()   # luôn giữ đủ lịch bàn 14 ngày tới, không xóa dữ liệu cũ
    if added:
        print(f"Đã bổ sung {added} khung giờ cho các ngày còn thiếu.")
    print(f"CSDL: {config.db_path()} (schema v{db.schema_version()}), dữ liệu bàn: {db.data_date_range()}")

    if args.index or args.force_index or reset:
        from actions import rag
        ok = rag.build_index(force=args.force_index)
        print("FAISS index:", "OK" if ok else "LỖI (không nạp được model embedding)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
