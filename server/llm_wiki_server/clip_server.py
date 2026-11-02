"""Port of clip_server.rs: the web-clipper HTTP receiver (default port 19827).

Hosted as a second, thread-based stdlib HTTP server (the Rust one is a separate
tiny_http listener on its own port, and the browser extension in extension/
hard-codes http://127.0.0.1:19827 / http://localhost:19827). Endpoints, exact
URL matching (query strings -> 404) and response bodies match the Rust code:

  GET  /status         {"ok":true,"version":"0.1.0"}
  GET  /project        {"ok":true,"path":...}
  POST /project        {"path": ...}              -> {"ok":true}
  GET  /projects       {"ok":true,"projects":[{name,path,current}]}
  POST /projects       {"projects":[{name,path}]} -> {"ok":true}
  GET  /clips/pending  {"ok":true,"clips":[{projectPath,filePath}]} (drains the list)
  POST /clip           {title,url,content,projectPath?} -> {"ok":true,"path":<relative>}

Non-loopback callers need the API token (X-LLM-Wiki-Token / Bearer).
"""
from __future__ import annotations

import hmac
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import cors, server_bind

PORT = 19827
MAX_BIND_RETRIES = 3
MAX_RESTART_RETRIES = 10
BIND_RETRY_DELAY_SECS = 2
RESTART_DELAY_SECS = 5
ALLOW_HEADERS = "Content-Type, Authorization, X-LLM-Wiki-Token"

_lock = threading.RLock()
_current_project = ""
_all_projects: List[Tuple[str, str]] = []  # (name, path)
_pending_clips: List[Tuple[str, str]] = []  # (projectPath, filePath)
_status = 0  # 0 starting, 1 running, 2 port_conflict, 3 error
_server: Optional[ThreadingHTTPServer] = None
_bound_port: Optional[int] = None


def get_daemon_status() -> str:
    return {0: "starting", 1: "running", 2: "port_conflict"}.get(_status, "error")


def current_project_path() -> str:
    with _lock:
        return _current_project


def set_current_project_path(path: str) -> None:
    global _current_project
    with _lock:
        _current_project = path


def all_projects() -> List[Tuple[str, str]]:
    with _lock:
        return list(_all_projects)


def bound_port() -> Optional[int]:
    return _bound_port


def next_restart_count(current: int) -> Optional[int]:
    nxt = current + 1
    return None if nxt > MAX_RESTART_RETRIES else nxt


def address_is_loopback(addr: Optional[str]) -> bool:
    if not addr:
        return False
    if addr.startswith("::ffff:"):
        addr = addr[7:]
    import ipaddress

    try:
        return ipaddress.ip_address(addr.split("%")[0]).is_loopback
    except ValueError:
        return False


def api_token() -> Optional[str]:
    t = os.environ.get("LLM_WIKI_API_TOKEN", "").strip()
    if t:
        return t
    try:
        from .state import app_state

        cfg = app_state().get("apiConfig")
        tok = cfg.get("token") if isinstance(cfg, dict) else None
        if isinstance(tok, str) and tok:
            return tok
    except Exception:
        pass
    return None


def is_token_authorized(headers: Dict[str, str]) -> bool:
    token = api_token()
    if token is None:
        return False
    tb = token.encode()
    v = headers.get("x-llm-wiki-token")
    if v is not None and hmac.compare_digest(v.encode(), tb):
        return True
    a = headers.get("authorization")
    if a is not None and a.startswith("Bearer ") and hmac.compare_digest(a[7:].encode(), tb):
        return True
    return False


def _err(msg: str) -> str:
    return json.dumps({"ok": False, "error": msg}, separators=(",", ":"), ensure_ascii=False)


def _field_str(parsed: Any, key: str) -> Optional[str]:
    if isinstance(parsed, dict) and isinstance(parsed.get(key), str):
        return parsed[key]
    return None


def handle_set_project(body: str) -> str:
    try:
        parsed = json.loads(body)
    except ValueError as e:
        return _err(f"Invalid JSON: {e}")
    p = _field_str(parsed, "path")
    if p is None:
        return '{"ok":false,"error":"path field is required"}'
    set_current_project_path(p.replace("\\", "/"))
    return '{"ok":true}'


def _slugify(title: str) -> str:
    mapped = "".join(c if (c.isalnum() or c == " " or c == "-") else " " for c in title)
    return "-".join(mapped.split()).lower()[:50]


def handle_clip(body: str) -> str:
    try:
        parsed = json.loads(body)
    except ValueError as e:
        return _err(f"Invalid JSON: {e}")
    t = _field_str(parsed, "title")
    title = "Untitled" if t is None else t
    url = _field_str(parsed, "url") or ""
    content = _field_str(parsed, "content") or ""
    project_path = _field_str(parsed, "projectPath") or ""
    if not project_path:
        project_path = current_project_path()
    project_path = project_path.replace("\\", "/")
    if not project_path:
        return ('{"ok":false,"error":"projectPath is required (set via POST /project or '
                'include in request body)"}')
    if not content:
        return '{"ok":false,"error":"content is required"}'

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    date_compact = now.strftime("%Y%m%d")
    base_name = f"{_slugify(title)}-{date_compact}"
    dir_path = Path(project_path) / "raw" / "sources"
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return _err(f"Failed to create directory: {e}")
    file_path = dir_path / f"{base_name}.md"
    counter = 2
    while file_path.exists():
        file_path = dir_path / f"{base_name}-{counter}.md"
        counter += 1
    file_str = str(file_path).replace("\\", "/")
    markdown = (
        f'---\ntype: clip\ntitle: "{title.replace(chr(34), chr(92) + chr(34))}"\n'
        f'url: "{url.replace(chr(34), chr(92) + chr(34))}"\nclipped: {date}\n'
        f"origin: web-clip\nsources: []\ntags: [web-clip]\n---\n\n# {title}\n\n"
        f"Source: {url}\n\n{content}\n"
    )
    try:
        with open(file_str, "w", encoding="utf-8", newline="") as f:
            f.write(markdown)
    except OSError as e:
        return _err(f"Failed to write file: {e}")
    try:
        relative = str(Path(file_str).relative_to(Path(project_path))).replace("\\", "/")
    except ValueError:
        relative = file_str
    with _lock:
        _pending_clips.append((project_path, file_str))
    return json.dumps({"ok": True, "path": relative}, separators=(",", ":"), ensure_ascii=False)


def _json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llm-wiki-clip"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _respond(self, status: int, body: str, extra: Optional[Dict[str, str]] = None) -> None:
        origin = self.headers.get("Origin")
        headers = cors.local_cors_headers(origin, ALLOW_HEADERS)
        if extra:
            headers.update(extra)
        data = body.encode("utf-8")
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _read_body(self) -> Optional[str]:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n).decode("utf-8")
        except (ValueError, OSError, UnicodeDecodeError) as e:
            self._respond(400, _err(f"Failed to read body: {e}"))
            return None

    def _dispatch(self) -> None:
        method = self.command
        if method == "OPTIONS":
            self._respond(204, "", {"Access-Control-Max-Age": "600"})
            return
        if not address_is_loopback(self.client_address[0]):
            hdrs = {k.lower(): v for k, v in self.headers.items()}
            if not is_token_authorized(hdrs):
                self._respond(401, '{"ok":false,"error":"Missing or invalid API token"}')
                return
        url = self.path
        if method == "GET" and url == "/status":
            self._respond(200, '{"ok":true,"version":"0.1.0"}')
        elif method == "GET" and url == "/project":
            self._respond(200, _json({"ok": True, "path": current_project_path()}))
        elif method == "POST" and url == "/project":
            body = self._read_body()
            if body is None:
                return
            result = handle_set_project(body)
            self._respond(200 if '"ok":true' in result else 400, result)
        elif method == "GET" and url == "/projects":
            cur = current_project_path()
            items = [{"name": n, "path": p, "current": p == cur} for n, p in all_projects()]
            self._respond(200, _json({"ok": True, "projects": items}))
        elif method == "POST" and url == "/projects":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n).decode("utf-8")
                parsed = json.loads(body)
                arr = parsed.get("projects") if isinstance(parsed, dict) else None
                if isinstance(arr, list):
                    with _lock:
                        _all_projects.clear()
                        for item in arr:
                            if not isinstance(item, dict):
                                continue
                            name = item.get("name") if isinstance(item.get("name"), str) else ""
                            path = item.get("path") if isinstance(item.get("path"), str) else ""
                            if path:
                                _all_projects.append((name, path))
            except (ValueError, OSError):
                pass
            self._respond(200, '{"ok":true}')
        elif method == "GET" and url == "/clips/pending":
            with _lock:
                clips = [{"projectPath": p, "filePath": f} for p, f in _pending_clips]
                _pending_clips.clear()
            self._respond(200, _json({"ok": True, "clips": clips}))
        elif method == "POST" and url == "/clip":
            body = self._read_body()
            if body is None:
                return
            result = handle_clip(body)
            self._respond(200 if '"ok":true' in result else 500, result)
        else:
            self._respond(404, '{"ok":false,"error":"Not found"}')

    do_GET = do_POST = do_OPTIONS = do_PUT = do_DELETE = do_PATCH = do_HEAD = _dispatch


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_server(host: str, port: int) -> ThreadingHTTPServer:
    h = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    cls = _Server
    if ":" in h:
        class _V6(_Server):
            address_family = socket.AF_INET6
        cls = _V6
    return cls((h, port), _Handler)


def start_clip_server(port: Optional[int] = None, host: Optional[str] = None,
                      retry_delay: float = BIND_RETRY_DELAY_SECS) -> threading.Thread:
    """Spawn the daemon thread (idempotent per call; used by the startup hook)."""
    if port is None:
        try:
            port = int(os.environ.get("LLM_WIKI_CLIP_PORT", PORT))
        except ValueError:
            port = PORT

    def run() -> None:
        global _status, _server, _bound_port
        restarts = 0
        while True:
            bind_host = host or server_bind.configured_bind_host()
            addr = server_bind.bind_addr(bind_host, port)
            srv = None
            last_err = ""
            for attempt in range(1, MAX_BIND_RETRIES + 1):
                try:
                    srv = _make_server(bind_host, port)
                    break
                except OSError as e:
                    last_err = str(e)
                    print(f"[Clip Server] Bind attempt {attempt}/{MAX_BIND_RETRIES} failed for {addr}: {e}",
                          file=sys.stderr)
                    if attempt < MAX_BIND_RETRIES:
                        time.sleep(retry_delay)
            if srv is None:
                print(f"[Clip Server] Address {addr} unavailable after {MAX_BIND_RETRIES} attempts: {last_err}",
                      file=sys.stderr)
                _status = 2
                return
            _server = srv
            _bound_port = srv.server_address[1]
            _status = 1
            print(f"[Clip Server] Listening on http://{addr}")
            try:
                srv.serve_forever()
            except Exception as e:
                print(f"[Clip Server] serve error: {e}", file=sys.stderr)
            if _server is None:  # shutdown requested
                return
            _status = 3
            nxt = next_restart_count(restarts)
            if nxt is None:
                print(f"[Clip Server] Exceeded max restarts ({MAX_RESTART_RETRIES}). Giving up.", file=sys.stderr)
                return
            restarts = nxt
            time.sleep(RESTART_DELAY_SECS)

    t = threading.Thread(target=run, name="clip-server", daemon=True)
    t.start()
    return t


def stop_clip_server() -> None:
    global _server, _status
    srv, _server = _server, None
    if srv is not None:
        srv.shutdown()
        srv.server_close()
    _status = 0
