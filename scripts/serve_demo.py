"""Serve the Nomos demo site with no-cache headers.

python3 -m http.server lets browsers heuristically cache index.html/app.js,
which can serve a stale viewer after edits. This wrapper sends
`Cache-Control: no-cache` so every load revalidates against disk.

Usage:
    python3 scripts/serve_demo.py [--port 8141]
    # open http://127.0.0.1:8141/cesium/index.html  (landing: /landing.html)
"""
from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

DEMO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "smoothride", "demo"))


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8141)
    args = ap.parse_args()

    if not os.path.isdir(DEMO_DIR):
        raise SystemExit(f"demo directory not found: {DEMO_DIR}")

    handler = lambda *a, **kw: NoCacheHandler(*a, directory=DEMO_DIR, **kw)  # noqa: E731
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as e:
        if e.errno == 48:  # EADDRINUSE
            raise SystemExit(
                f"port {args.port} is already in use — the demo server is probably "
                f"already running.\nOpen http://127.0.0.1:{args.port}/cesium/index.html, "
                f"or stop the old one:  lsof -ti tcp:{args.port} | xargs kill")
        raise
    with httpd:
        print(f"Serving {DEMO_DIR}")
        print(f"  sim:     http://127.0.0.1:{args.port}/cesium/index.html")
        print(f"  landing: http://127.0.0.1:{args.port}/landing.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
