#!/usr/bin/env python3
"""Video upload test page + same-origin proxy to OpenAI-compatible endpoints.

Stdlib only. Serves index.html and forwards POST /proxy and GET /models to the
chosen endpoint. API keys come from env VIDEOTEST_KEY_<NAME> (never files) or,
for the "custom" endpoint, the X-Custom-Key request header (held in memory only).
Request/response bodies are never logged. Redirects are not followed.
"""
import http.client
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BODY = 200 * 1024 * 1024
TIMEOUT = 300
PORT = int(os.environ.get("VIDEOTEST_PORT", "18810"))


def load_endpoints():
    with open(os.path.join(HERE, "endpoints.json")) as f:
        eps = json.load(f)
    # VIDEOTEST_BASE_<NAME> overrides a preset's base URL (e.g. in a container).
    for n, v in eps.items():
        o = os.environ.get("VIDEOTEST_BASE_" + n.upper().replace("-", "_"))
        if o and v.get("base") is not None:
            v["base"] = o
    return eps


def resolve(ep_name, query, headers):
    """Return (scheme, host, port, path_prefix, key) or raise ValueError."""
    eps = load_endpoints()
    if ep_name not in eps:
        raise ValueError("unknown endpoint")
    base = eps[ep_name].get("base")
    key = os.environ.get("VIDEOTEST_KEY_" + ep_name.upper().replace("-", "_"))
    if base is None:  # user-supplied
        base = (query.get("base") or [""])[0]
        key = headers.get("X-Custom-Key") or key
    u = urllib.parse.urlsplit(base)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("base URL must be http(s)")
    port = u.port or (443 if u.scheme == "https" else 80)
    return u.scheme, u.hostname, port, u.path.rstrip("/"), key


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "videotest/1"

    def log_message(self, fmt, *args):  # request line only, never bodies
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._send(code, json.dumps({"error": {"message": msg}}))

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/app.js":
            with open(os.path.join(HERE, "app.js"), "rb") as f:
                return self._send(200, f.read(), "text/javascript; charset=utf-8")
        if u.path == "/endpoints":
            eps = load_endpoints()
            return self._send(200, json.dumps(
                [{"name": n, "custom": v.get("base") is None} for n, v in eps.items()]))
        if u.path == "/models":
            return self._forward("GET", "/models", q, None)
        self._err(404, "not found")

    def do_POST(self):
        u = urllib.parse.urlsplit(self.path)
        if u.path != "/proxy":
            return self._err(404, "not found")
        q = urllib.parse.parse_qs(u.query)
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._err(411, "Content-Length required")
        if n > MAX_BODY:
            return self._err(413, "body exceeds %d MB" % (MAX_BODY >> 20))
        self._forward("POST", "/chat/completions", q, n)

    def _forward(self, method, sub, q, length):
        try:
            scheme, host, port, prefix, key = resolve(
                (q.get("endpoint") or [""])[0], q, self.headers)
        except ValueError as e:
            if length:  # drain so the connection stays sane
                self.rfile.read(length)
            return self._err(400, str(e))
        cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        conn = cls(host, port, timeout=TIMEOUT)
        try:
            conn.putrequest(method, prefix + sub)
            conn.putheader("Accept", "application/json")
            if key:
                conn.putheader("Authorization", "Bearer " + key)
            if length is not None:
                conn.putheader("Content-Type", "application/json")
                conn.putheader("Content-Length", str(length))
            conn.endheaders()
            left = length or 0
            while left:  # stream-forward in chunks, no full buffering
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    raise ConnectionError("client closed during upload")
                conn.send(chunk)
                left -= len(chunk)
            r = conn.getresponse()  # 3xx are returned as-is, not followed
            data = r.read()
            self._send(r.status, data, r.getheader("Content-Type", "application/json"))
        except Exception as e:
            try:
                self._err(502, "upstream error: %s" % type(e).__name__ + (": %s" % e if str(e) else ""))
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    print("videotest listening on 0.0.0.0:%d" % PORT, flush=True)
    srv.serve_forever()
