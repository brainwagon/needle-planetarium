"""Static server for the prototype, run from the repo root:

    python3 spike/webgpu/serve.py 8767     # then open http://localhost:8767/spike/webgpu/

Sends cross-origin isolation headers (needed for multi-threaded WASM) and accepts
POST /log, appending the page's results to spike/webgpu/results.log, so a run in
another browser (e.g. Windows Chrome with a real GPU) can be read back here.
"""
import sys
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG = Path(__file__).with_name("results.log")


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def do_POST(self):
        if self.path != "/log":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8", "replace")
        with LOG.open("a") as fh:
            fh.write(f"--- {datetime.now():%Y-%m-%d %H:%M:%S} {self.headers.get('User-Agent', '')}\n{body}\n")
        self.send_response(204)
        self.end_headers()


ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1]) if len(sys.argv) > 1 else 8767), Handler).serve_forever()
