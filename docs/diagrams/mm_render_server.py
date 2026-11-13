"""四卡全景图的本地渲染服务：给 headless Chrome 提供同源页面 + 接收 POST 落盘。

用法::

    python mm_render_server.py <diagrams 目录> [port]

为什么需要它：SVG 必须由浏览器渲染导出，而 **Windows 上 chrome.exe 是 GUI 子系统程序，
stdout 抓不到**，所以不能用 --dump-dom 收结果；本服务让页面把 SVG / 指标 POST 回磁盘。
"""
from __future__ import annotations

import http.server
import os
import sys
from urllib.parse import parse_qs, urlparse

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8791


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_POST(self) -> None:  # noqa: N802
        name = os.path.basename(parse_qs(urlparse(self.path).query).get("name", ["out.bin"])[0])
        data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        with open(os.path.join(ROOT, name), "wb") as fh:
            fh.write(data)
        print(f"SAVED {name} {len(data)} bytes", flush=True)
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print("HTTP " + (fmt % args), flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    print(f"serving {ROOT} on http://127.0.0.1:{PORT}/", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
