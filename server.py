"""Local web app: the WindGap page (templates/windgap.html) with our Python model as its engine.

    python server.py                  # http://localhost:8000
    python server.py --port 8080 --data data
    python server.py --reload         # re-import src/ on every request (while changing the model)

GET  /             the page, switched to the backend engine
GET  /health       status + turbines the server has on disk
POST /api/analyze  page request -> estimate_loss() -> page response (see src/webapi.py)
POST /api/export   analysis request (+ targets) -> ZIP: CSV per turbine, park total, metrics (see src/export.py)
POST /api/claim    analysis request + claim form -> email draft, PDF report, CSVs (see src/claimapi.py)
GET  /api/claim/<id>/<file>   generated report (inline preview; ?download=1 to download)
"""
import argparse
import importlib
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import src.assessment
import src.claimapi
import src.data_io
import src.energy
import src.export
import src.model
import src.pricing
import src.report
import src.webapi

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "templates" / "windgap.html"
BACKEND_JS = '<script>window.WINDGAP_BACKEND={base:"",path:"/api/analyze"};</script>'


class Handler(BaseHTTPRequestHandler):
    store = None
    reload = False

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")      # page opened from file:// also works
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            html = PAGE.read_text(encoding="utf-8").replace("</head>", BACKEND_JS + "</head>", 1)
            self._send(200, html, "text/html; charset=utf-8")
        elif path.startswith("/api/claim/"):
            parts = path.split("/")
            got = src.claimapi.claim_file(parts[3], parts[4]) if len(parts) == 5 else None
            if got is None:
                return self._send(404, json.dumps({"error": "not found (reports live in memory - generate again)"}))
            data, ctype = got
            disposition = "attachment" if "download=1" in self.path else "inline"   # inline = preview in the page
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'{disposition}; filename="{parts[4]}"')
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        elif path == "/health":
            self._send(200, json.dumps({"status": "ok", "turbines": sorted(self.store.turbines),
                                        "power_curves": sorted(self.store.curves)}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        route = self.path.split("?")[0]
        if route not in ("/api/analyze", "/api/claim", "/api/export"):
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.reload:
                for mod in (src.data_io, src.energy, src.model, src.webapi,
                            src.pricing, src.report, src.assessment, src.claimapi, src.export):
                    importlib.reload(mod)
            if route == "/api/export":
                data, name = src.export.export_zip(payload, self.store)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.send_header("Access-Control-Expose-Headers", "Content-Disposition")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
                return
            if route == "/api/claim":
                result = src.claimapi.claim(payload, self.store)
            else:
                result = src.webapi.analyze(payload, self.store)
            self._send(200, json.dumps(result))
        except ValueError as e:            # expected: bad window, no reference data, ...
            self._send(400, json.dumps({"error": str(e)}))
        except Exception as e:             # bugs: full traceback in this terminal
            traceback.print_exc()
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def log_message(self, fmt, *args):
        if self.command == "POST" or not str(args[1]).startswith("2"):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = this computer only; 0.0.0.0 = reachable from the network / cloud hosting")
    args = ap.parse_args()
    Handler.store = src.webapi.Store(args.data)
    Handler.reload = args.reload
    print(f"Loaded {len(Handler.store.turbines)} turbines from {args.data}")
    print(f"WindGap running on http://localhost:{args.port}  (listening on {args.host})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
