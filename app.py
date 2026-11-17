"""Serve the local demo and its prediction endpoint."""
import argparse
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from model import DigitModel

ROOT = Path(__file__).resolve().parent
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/sample.json": ("sample.json", "application/json"),
}
MAX_PAYLOAD = 4096


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, model, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def reply(self, status, data, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def json_reply(self, status, data):
        self.reply(status, json.dumps(data).encode("utf-8"), "application/json")

    def do_GET(self):
        if self.path == "/api/metrics":
            self.json_reply(HTTPStatus.OK, self.model.metrics)
        elif self.path in STATIC:
            name, content_type = STATIC[self.path]
            self.reply(HTTPStatus.OK, (ROOT / name).read_bytes(), content_type)
        else:
            self.json_reply(HTTPStatus.NOT_FOUND, {"error": "Route not found."})

    def do_POST(self):
        if self.path != "/api/predict":
            self.json_reply(HTTPStatus.NOT_FOUND, {"error": "Route not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self.json_reply(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Send JSON."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_PAYLOAD:
                raise ValueError("Request body must be between 1 and 4096 bytes.")
            data = json.loads(self.rfile.read(size))
            result = self.model.explain(data.get("pixels") if isinstance(data, dict) else None)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
            self.json_reply(HTTPStatus.BAD_REQUEST, {"error": "Malformed JSON."})
        except ValueError as error:
            self.json_reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        else:
            self.json_reply(HTTPStatus.OK, result)


def make_server(port, model):
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, model=model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with make_server(args.port, DigitModel()) as server:
        print(f"Open http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
