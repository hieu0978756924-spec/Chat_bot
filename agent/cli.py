"""Giao diện dòng lệnh.

    python main.py            # chạy thường
    python main.py --debug    # in intent, slot, state, tool + tham số, thời gian LLM mỗi lượt
"""

import argparse
import logging
import os
import sys
import uuid

for _s in (sys.stdout, sys.stderr, sys.stdin):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from actions.db import SchemaOutdatedError  # noqa: E402

HELP = """Lệnh:  /moi  hội thoại mới   /state  xem trạng thái   /debug  bật/tắt debug   /thoat  thoát"""


def setup_logging(debug: bool) -> None:
    os.makedirs(os.path.join(config.BASE_DIR, "logs"), exist_ok=True)
    handlers = [logging.FileHandler(os.path.join(config.BASE_DIR, "logs", "chatbot.log"), encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers)
    # chỉ cảnh báo quan trọng ra màn hình, không làm rối hội thoại
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.ERROR if not debug else logging.WARNING)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(console)
    for noisy in ("sentence_transformers", "urllib3", "httpx", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description="Trợ lý đặt bàn nhà hàng (Local LLM)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    setup_logging(args.debug)

    print("Đang khởi động (nạp model embedding, kết nối Ollama)…", flush=True)
    from agent.agent import RestaurantAgent
    from actions.llm import llm_status

    try:
        agent = RestaurantAgent(session_id=uuid.uuid4().hex[:8], debug=args.debug)
    except SchemaOutdatedError as e:
        print(f"❌ {e}")
        return
    status = llm_status()
    print("=" * 70)
    print(f"Mô hình NLU: {status['message']}")
    if args.debug:
        print("Chế độ DEBUG: BẬT")
    print(HELP)
    print("=" * 70)
    print(f"\n🤖 {agent.greeting()}\n")       # lời chào chỉ in MỘT lần

    while True:
        try:
            text = input("Bạn > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt bạn!")
            return
        if not text:
            continue
        cmd = text.lower()
        if cmd in ("/thoat", "/exit", "/quit", "/stop"):
            print("Tạm biệt bạn!")
            return
        if cmd in ("/moi", "/new", "/reset"):
            agent.reset()
            print("\n🔄 Đã bắt đầu hội thoại mới.\n")
            continue
        if cmd == "/state":
            import json
            print(json.dumps(agent.state.debug_view(), ensure_ascii=False, indent=2))
            continue
        if cmd == "/debug":
            agent.debug = not agent.debug
            print(f"Debug: {'BẬT' if agent.debug else 'TẮT'}")
            continue
        if cmd in ("/help", "/trogiup"):
            print(HELP)
            continue
        print("⏳ Đang xử lý…", end="\r", flush=True)
        try:
            reply = agent.ask(text)
        except Exception as e:  # lỗi không lường trước: báo thân thiện, ghi log
            logging.getLogger(__name__).exception("Lỗi xử lý tin nhắn")
            reply = f"❌ Xin lỗi, hệ thống gặp lỗi khi xử lý ({type(e).__name__}). Chưa có thay đổi nào được ghi nhận."
        print(" " * 20, end="\r")
        print(f"\n🤖 {reply}\n")


if __name__ == "__main__":
    main()
