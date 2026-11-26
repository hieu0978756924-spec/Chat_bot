"""Giao diện web cho chatbot (chạy trên localhost, chỉ dùng thư viện chuẩn của Python).

    venv\\Scripts\\python web\\server.py            # mở http://127.0.0.1:8000
    venv\\Scripts\\python web\\server.py --port 8080

API:
    GET  /api/greeting?sid=...   -> {"reply": lời chào}
    POST /api/chat   {"sid", "message"} -> {"reply", "latency_s"}
    POST /api/reset  {"sid"}           -> {"ok": true}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from agent.cli import setup_logging  # noqa: E402

log = logging.getLogger("web")
STORE = None                 # SessionStore, tạo khi khởi động
LOCK = threading.Lock()      # Ollama xử lý tuần tự -> mỗi lúc chỉ 1 lượt


def _sid(v: object) -> str:
    s = str(v or "default")
    return "".join(ch for ch in s if ch.isalnum() or ch in "-_")[:40] or "default"


class Handler(BaseHTTPRequestHandler):
    server_version = "TableNow/1.0"

    def log_message(self, fmt: str, *args) -> None:  # tắt log truy cập ra console
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 20000:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if url.path == "/api/greeting":
            sid = _sid(parse_qs(url.query).get("sid", [""])[0])
            with LOCK:
                agent = STORE.get(sid)
                return self._json({"reply": agent.greeting()})
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        data = self._body()
        sid = _sid(data.get("sid"))
        if self.path == "/api/reset":
            with LOCK:
                STORE.get(sid).reset()
            return self._json({"ok": True})
        if self.path == "/api/chat":
            text = str(data.get("message") or "").strip()[:1000]
            if not text:
                return self._json({"reply": "Bạn nhập nội dung giúp mình nhé."})
            t0 = time.time()
            try:
                with LOCK:
                    reply = STORE.get(sid).ask(text)
            except Exception as e:  # lỗi bất ngờ: báo thân thiện, ghi log
                log.exception("Lỗi xử lý tin nhắn")
                reply = (f"❌ Xin lỗi, hệ thống gặp lỗi khi xử lý ({type(e).__name__}). "
                         "Chưa có thay đổi nào được ghi nhận.")
            return self._json({"reply": reply, "latency_s": round(time.time() - t0, 1)})
        self._send(404, b"Not found", "text/plain; charset=utf-8")


def main() -> None:
    global STORE
    ap = argparse.ArgumentParser(description="Giao diện web TableNow")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    setup_logging(args.debug)

    print("Đang khởi động (nạp model embedding, kết nối Ollama)…", flush=True)
    from agent.agent import SessionStore
    from actions.llm import llm_status

    STORE = SessionStore(debug=args.debug)
    STORE.get("warmup")      # nạp sẵn model, tránh lượt chat đầu tiên bị chậm
    print(f"Mô hình NLU: {llm_status()['message']}")

    url = f"http://127.0.0.1:{args.port}"
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"✅ Giao diện web đã chạy: {url}   (Ctrl+C để dừng)", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng máy chủ.")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
