"""Port of api_server.rs: the `/api/v1` REST surface.

The Rust version was a hand-rolled tiny_http server; here it is a FastAPI
`APIRouter` (`router`, mounted by app.py) with a single catch-all dispatcher
so that routing, status codes (404/405/401/503/429), error bodies
(`{"ok": false, "error": ...}`) and ordering of checks stay identical.

Dependencies on modules ported by other agents (search, page_embedding,
file_sync, agent.runtime / agent.tools / agent.session) are imported lazily
inside the handlers.

Agent runtime contract (used by the chat endpoints only):

    runtime.run_chat(project, request, config, cancel_event, on_event) -> dict
        project: {"id","name","path","current"}; request: camelCase request dict
        (sessionId / runId already filled in); config: {"embedding","llm",
        "webSearch","anytxt"} dicts; cancel_event: threading.Event;
        on_event: callable(dict) | None (streaming sink). May be async.
        Returns the aggregate response dict {projectId, sessionId, mode, message,
        references, toolEvents, events, usage}. Raises Exception(str) on
        failure ("Agent turn cancelled" for cancellation).
    runtime.redact_for_external_api(event) -> dict   (optional)
"""
from __future__ import annotations

import asyncio
import dataclasses
import hmac
import importlib
import inspect
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import unquote_to_bytes

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .. import __version__
from ..state import app_state

router = APIRouter()

# --------------------------------------------------------------------------
# Constants (identical to api_server.rs)
# --------------------------------------------------------------------------
PORT = 19828
API_PREFIX = "/api/v1"
MAX_BODY_BYTES = 1024 * 1024
MAX_CHAT_BODY_BYTES = 40 * 1024 * 1024
MAX_PAGE_WRITE_BODY_BYTES = 6 * 2 * 1024 * 1024 + 16 * 1024
MAX_FILE_CONTENT_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_FILES = 2_000
HARD_MAX_FILES = 10_000
DEFAULT_MAX_REVIEWS = 200
HARD_MAX_REVIEWS = 1_000
MAX_SEARCH_RESULTS = 50
RATE_LIMIT_WINDOW = 1.0
RATE_LIMIT_MAX_REQUESTS = 120
MAX_IN_FLIGHT_REQUESTS = 64
MAX_IN_FLIGHT_CHAT_STREAMS = 8
MAX_IN_FLIGHT_PAGE_EMBEDS = 4
SSE_QUEUE_CAPACITY = 64
SSE_HEARTBEAT_INTERVAL = 10.0
GRAPH_CACHE_TTL = 5.0
GRAPH_CACHE_MAX_PROJECTS = 8

AGENT_UNAVAILABLE = "agent runtime not available"
DEFAULT_BIND_HOST = "127.0.0.1"
PUBLIC_BIND_HOST = "0.0.0.0"
BIND_HOST_ENV = "LLM_WIKI_BIND_HOST"


class ApiError(Exception):
    """Short-circuit a handler with `{"ok": false, "error": message}` and `status`."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _err_body(message: str) -> Dict[str, Any]:
    return {"ok": False, "error": message}


# --------------------------------------------------------------------------
# Counters / limits
# --------------------------------------------------------------------------
_counter_lock = threading.Lock()
_in_flight = {"requests": 0, "chat": 0, "embed": 0}
_rate_hits: "deque[float]" = deque()
_rate_lock = threading.Lock()


def _try_acquire(kind: str, limit: int) -> bool:
    with _counter_lock:
        if _in_flight[kind] >= limit:
            return False
        _in_flight[kind] += 1
        return True


def _release(kind: str) -> None:
    with _counter_lock:
        _in_flight[kind] = max(0, _in_flight[kind] - 1)


def allow_request() -> bool:
    now = time.monotonic()
    start = now - RATE_LIMIT_WINDOW
    with _rate_lock:
        while _rate_hits and _rate_hits[0] < start:
            _rate_hits.popleft()
        if len(_rate_hits) >= RATE_LIMIT_MAX_REQUESTS:
            return False
        _rate_hits.append(now)
        return True


def should_rate_limit(method: str, path: str) -> bool:
    if method == "OPTIONS":
        return False
    return not (path == "/health" or path == f"{API_PREFIX}/health")


# --------------------------------------------------------------------------
# App-state / config access
# --------------------------------------------------------------------------
_state_cache: Dict[str, Any] = {"sig": None, "value": None}
_state_lock = threading.Lock()


def invalidate_config_cache() -> None:
    with _state_lock:
        _state_cache["sig"] = None
        _state_cache["value"] = None


def load_app_state() -> Optional[dict]:
    """Parsed app-state.json (re-read only when the file's mtime/size changes)."""
    st = app_state()
    try:
        s = os.stat(st.path)
        sig = (s.st_mtime_ns, s.st_size, str(st.path))
    except OSError:
        with _state_lock:
            prev = _state_cache["value"]
        return prev
    with _state_lock:
        if _state_cache["sig"] == sig:
            return _state_cache["value"]
        previous = _state_cache["value"]
    try:
        loaded = json.loads(Path(st.path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            loaded = None
    except (OSError, ValueError):
        loaded = None
    value = loaded if loaded is not None else previous
    with _state_lock:
        _state_cache["sig"] = sig
        _state_cache["value"] = value
    return value


def _api_config() -> dict:
    parsed = load_app_state()
    cfg = parsed.get("apiConfig") if parsed else None
    return cfg if isinstance(cfg, dict) else {}


def _cfg_bool(key: str, default: bool) -> bool:
    v = _api_config().get(key)
    return v if isinstance(v, bool) else default


def api_token() -> Optional[str]:
    env = os.environ.get("LLM_WIKI_API_TOKEN")
    if env is not None and env.strip():
        return env.strip()
    tok = _api_config().get("token")
    return tok if isinstance(tok, str) and tok else None


def api_token_source() -> str:
    env = os.environ.get("LLM_WIKI_API_TOKEN")
    if env is not None and env.strip():
        return "env"
    tok = _api_config().get("token")
    if isinstance(tok, str) and tok:
        return "store"
    return "none"


def api_allow_unauthenticated() -> bool:
    return _cfg_bool("allowUnauthenticated", False)


def api_auth_required() -> bool:
    return not api_allow_unauthenticated()


def api_allow_lan_access() -> bool:
    return _cfg_bool("allowLanAccess", False)


def api_enabled() -> bool:
    return _cfg_bool("enabled", True)


def api_mcp_enabled() -> bool:
    return _cfg_bool("mcpEnabled", False)


def sanitize_bind_host(value: str) -> Optional[str]:
    host = (value or "").strip()
    if not host:
        return None
    if all(c.isascii() and (c.isalnum() or c in ".-_:[]") for c in host):
        return host
    return None


def configured_bind_host() -> str:
    """Bind host policy of server_bind.rs: env LLM_WIKI_BIND_HOST, else 0.0.0.0 when
    apiConfig.allowLanAccess is true, else 127.0.0.1. For use by `python -m llm_wiki_server`
    when `--host` is not given."""
    env = sanitize_bind_host(os.environ.get(BIND_HOST_ENV, ""))
    if env:
        return env
    if api_allow_lan_access():
        return PUBLIC_BIND_HOST
    return DEFAULT_BIND_HOST


# --------------------------------------------------------------------------
# URL helpers / auth
# --------------------------------------------------------------------------
def percent_decode(s: str) -> str:
    return unquote_to_bytes(s.replace("+", " ")).decode("utf-8", errors="replace")


def parse_query(query: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pair in query.split("&"):
        if not pair:
            continue
        k, _, v = pair.partition("=")
        out[percent_decode(k)] = percent_decode(v)
    return out


def api_path_parts(path: str) -> Optional[List[str]]:
    if not path.startswith(API_PREFIX):
        return None
    rem = path[len(API_PREFIX):]
    if rem and not rem.startswith("/"):
        return None
    return [p for p in rem.lstrip("/").split("/") if p]


def is_agent_chat_request(method: str, path: str) -> bool:
    parts = api_path_parts(path)
    if parts is None or method != "POST":
        return False
    if len(parts) == 3 and parts[0] == "projects" and parts[2] == "chat":
        return True
    return len(parts) == 5 and parts[0] == "projects" and parts[2] == "chat" and parts[4] == "cancel"


def is_token_required_request(method: str, path: str) -> bool:
    if is_agent_chat_request(method, path):
        return True
    parts = api_path_parts(path)
    if parts is None or method != "POST":
        return False
    return (
        len(parts) == 4
        and parts[0] == "projects"
        and parts[2] == "pages"
        and parts[3] in ("embed", "write")
    )


def chat_project_id(method: str, path: str) -> Optional[str]:
    if method != "POST":
        return None
    parts = api_path_parts(path)
    if parts and len(parts) == 3 and parts[0] == "projects" and parts[2] == "chat":
        return parts[1]
    return None


def wants_streaming_chat(method: str, path: str, body: str, headers: List[Tuple[str, str]]) -> bool:
    if chat_project_id(method, path) is None:
        return False
    accepts_sse = False
    for k, v in headers:
        if k == "accept":
            for part in v.split(","):
                if part.strip().split(";")[0].strip().lower() == "text/event-stream":
                    accepts_sse = True
    try:
        parsed = json.loads(body)
        stream = parsed.get("stream") if isinstance(parsed, dict) else None
    except ValueError:
        stream = None
    if isinstance(stream, bool):
        return stream
    return accepts_sse


def body_limit_for_request(method: str, path: str) -> int:
    parts = api_path_parts(path)
    if method != "POST" or parts is None:
        return MAX_BODY_BYTES
    if len(parts) == 3 and parts[0] == "projects" and parts[2] == "chat":
        return MAX_CHAT_BODY_BYTES
    if len(parts) == 4 and parts[0] == "projects" and parts[2] == "pages" and parts[3] == "write":
        return MAX_PAGE_WRITE_BODY_BYTES
    return MAX_BODY_BYTES


def constant_time_eq(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)


def is_token_authorized(query: str, headers: List[Tuple[str, str]]) -> bool:
    token = api_token()
    if token is None:
        return False
    tb = token.encode()
    params = parse_query(query)
    if "token" in params and constant_time_eq(params["token"].encode(), tb):
        return True
    for key, value in headers:
        if key == "x-llm-wiki-token":
            if constant_time_eq(value.encode(), tb):
                return True
        elif key == "authorization":
            if value.startswith("Bearer ") and constant_time_eq(value[7:].encode(), tb):
                return True
    return False


def is_authorized(query: str, headers: List[Tuple[str, str]]) -> bool:
    if not api_auth_required():
        return True
    return is_token_authorized(query, headers)


# --------------------------------------------------------------------------
# Lazy imports / helpers for sibling modules
# --------------------------------------------------------------------------
def _lazy(rel: str):
    """Import `llm_wiki_server.<rel>` at call time (respects sys.modules monkeypatching)."""
    root = __name__.rsplit(".", 2)[0]  # llm_wiki_server
    return importlib.import_module(f"{root}.{rel}")


def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a sync-or-async function from a worker thread and return its result."""
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        async def _wait():
            return await res

        return asyncio.run(_wait())
    return res


def _jsonable(v: Any) -> Any:
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return dataclasses.asdict(v)
    if hasattr(v, "model_dump"):
        return v.model_dump(by_alias=True)
    if hasattr(v, "dict") and callable(v.dict) and not isinstance(v, dict):
        return v.dict()
    return v


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    return default


def _exc_message(e: BaseException) -> str:
    return getattr(e, "message", None) or str(e)


# --------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------
def normalize_path(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def project_name_from_path(path: str) -> str:
    return PurePath(path).name or "Project"


def read_project_id(path: str) -> Optional[str]:
    try:
        parsed = json.loads((Path(path) / ".llm-wiki/project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = parsed.get("id") if isinstance(parsed, dict) else None
    return pid if isinstance(pid, str) else None


def _clip_current() -> str:
    try:
        cur = str(_lazy("clip_server").current_project_path() or "")
        if cur:
            return cur
    except Exception:
        pass
    parsed = load_app_state() or {}
    last = parsed.get("lastProject")
    if isinstance(last, dict) and isinstance(last.get("path"), str):
        return last["path"]
    if isinstance(last, str):
        return last
    return ""


def _clip_all() -> List[Tuple[str, str]]:
    try:
        mod = _lazy("clip_server")
        return [(str(n), str(p)) for n, p in mod.all_projects()]
    except Exception:
        return []


def project_path_matches(stored: str, candidate: str) -> bool:
    a, b = normalize_path(stored), normalize_path(candidate)
    if sys.platform == "win32":
        return a.lower() == b.lower()
    return a == b


def load_projects() -> List[dict]:
    current = normalize_path(_clip_current())
    by_path: Dict[str, dict] = {}
    parsed = load_app_state()
    if parsed:
        registry = parsed.get("projectRegistry")
        if isinstance(registry, dict):
            for pid, value in registry.items():
                if not isinstance(value, dict):
                    continue
                path = value.get("path")
                if not isinstance(path, str) or not path:
                    continue
                path = normalize_path(path)
                name = value.get("name") if isinstance(value.get("name"), str) else project_name_from_path(path)
                by_path[path] = {"id": pid, "name": name, "path": path, "current": path == current}
        recents = parsed.get("recentProjects")
        if isinstance(recents, list):
            for value in recents:
                if not isinstance(value, dict):
                    continue
                path = value.get("path")
                if not isinstance(path, str) or not path:
                    continue
                path = normalize_path(path)
                if path in by_path:
                    continue
                name = value.get("name") if isinstance(value.get("name"), str) else project_name_from_path(path)
                by_path[path] = {
                    "id": read_project_id(path) or path,
                    "name": name,
                    "path": path,
                    "current": path == current,
                }
    for name, path in _clip_all():
        path = normalize_path(path)
        if path not in by_path:
            by_path[path] = {
                "id": read_project_id(path) or path,
                "name": name if name else project_name_from_path(path),
                "path": path,
                "current": path == current,
            }
    if current and current not in by_path:
        by_path[current] = {
            "id": read_project_id(current) or current,
            "name": project_name_from_path(current),
            "path": current,
            "current": True,
        }
    return [by_path[k] for k in sorted(by_path)]


def resolve_project(project_id: str) -> dict:
    pid = percent_decode(project_id)
    wants_current = pid.lower() == "current"
    for p in load_projects():
        if p["id"] == pid or project_path_matches(p["path"], pid) or (wants_current and p["current"]):
            return p
    raise ApiError(404, f"Unknown project: {pid}")


def handle_projects() -> dict:
    projects = load_projects()
    current = next((p for p in projects if p["current"]), None)
    return {"ok": True, "projects": projects, "currentProject": current}


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------
def safe_join(project_path: str, rel: str) -> Path:
    root = Path(project_path)
    rel = rel.lstrip("/")
    rel_path = Path(rel)
    if rel_path.is_absolute() or rel_path.drive:
        raise ApiError(400, "Absolute paths are not allowed")
    for part in rel_path.parts:
        if part == "..":
            raise ApiError(400, "Path traversal is not allowed")
    joined = root / rel_path
    try:
        root_canon = root.resolve(strict=True)
    except OSError as e:
        raise ApiError(400, f"Failed to resolve project path: {e}")

    def inside(p: Path) -> bool:
        try:
            p.relative_to(root_canon)
            return True
        except ValueError:
            return False

    if joined.exists():
        try:
            canon = joined.resolve(strict=True)
        except OSError as e:
            raise ApiError(400, f"Failed to resolve path: {e}")
        if not inside(canon):
            raise ApiError(400, "Resolved path escapes the project directory")
        return canon
    parent = joined.parent
    if parent.exists():
        try:
            pc = parent.resolve(strict=True)
        except OSError as e:
            raise ApiError(400, f"Failed to resolve parent path: {e}")
        if not inside(pc):
            raise ApiError(400, "Resolved parent escapes the project directory")
    return joined


def is_public_project_rel(rel: str) -> bool:
    rel = normalize_path(rel).lstrip("/")
    if any(part == "" or part.startswith(".") for part in rel.split("/")):
        return False
    lower = rel.lower()
    return (
        lower == "purpose.md"
        or lower == "schema.md"
        or lower.startswith("wiki/")
        or lower.startswith("raw/sources/")
    )


_TEXT_EXTS = {"md", "mdx", "txt", "csv", "json", "yaml", "yml", "xml", "html", "htm", "log"}


def is_text_content_rel(rel: str) -> bool:
    base = normalize_path(rel).lower().rsplit("/", 1)[-1]
    return PurePath(base).suffix.lstrip(".") in _TEXT_EXTS


def relative_to_project(project_path: str, path: Path) -> str:
    candidates = [Path(project_path)]
    try:
        candidates.append(Path(project_path).resolve())
    except OSError:
        pass
    for root in candidates:
        try:
            return str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            continue
    return str(path).replace("\\", "/")


def _push_file_node(project_path: str, path: Path, recursive: bool, max_files: int,
                    count: List[int], out: List[dict]) -> None:
    name = path.name
    if name.startswith("."):
        return
    try:
        st = os.lstat(path)
    except OSError as e:
        raise ApiError(500, f"Failed to read metadata: {e}")
    import stat as _stat

    if _stat.S_ISLNK(st.st_mode):
        return
    count[0] += 1
    if count[0] > max_files:
        raise ApiError(413, f"File listing exceeds maxFiles limit ({max_files})")
    is_dir = _stat.S_ISDIR(st.st_mode)
    children = _list_tree(project_path, path, True, max_files, count) if (recursive and is_dir) else None
    out.append({
        "name": name,
        "path": relative_to_project(project_path, path),
        "isDir": is_dir,
        "size": None if is_dir else st.st_size,
        "children": children,
    })


def _list_tree(project_path: str, path: Path, recursive: bool, max_files: int, count: List[int]) -> List[dict]:
    out: List[dict] = []
    try:
        entries = list(os.scandir(path))
    except OSError as e:
        raise ApiError(500, f"Failed to list directory: {e}")
    for entry in entries:
        _push_file_node(project_path, Path(entry.path), recursive, max_files, count, out)
    out.sort(key=lambda n: (not n["isDir"], n["name"].encode("utf-8")))
    return out


def handle_files(project_id: str, query: str) -> dict:
    project = resolve_project(project_id)
    params = parse_query(query)
    root = params.get("root", "wiki")
    recursive = params.get("recursive", "true") != "false"
    mf = _parse_usize(params.get("maxFiles"))
    max_files = min(max(mf if mf is not None else DEFAULT_MAX_FILES, 1), HARD_MAX_FILES)
    if root == "wiki":
        rel = "wiki"
    elif root in ("sources", "raw", "raw/sources"):
        rel = "raw/sources"
    elif root in ("all", ""):
        rel = ""
    else:
        raise ApiError(400, "root must be wiki, sources, or all")
    count = [0]
    if rel == "":
        roots: List[dict] = []
        for r in ("purpose.md", "schema.md", "wiki", "raw/sources"):
            p = safe_join(project["path"], r)
            if not p.exists():
                continue
            _push_file_node(project["path"], p, recursive, max_files, count, roots)
        return {"ok": True, "projectId": project["id"], "root": "all", "files": roots, "truncated": False}
    d = safe_join(project["path"], rel)
    files = _list_tree(project["path"], d, recursive, max_files, count)
    return {"ok": True, "projectId": project["id"], "root": rel, "files": files, "truncated": False}


def handle_file_content(project_id: str, query: str) -> dict:
    project = resolve_project(project_id)
    params = parse_query(query)
    rel = params.get("path")
    if rel is None:
        raise ApiError(400, "Missing path query parameter")
    if not is_public_project_rel(rel):
        raise ApiError(403, "Path is not exposed by the local API")
    if not is_text_content_rel(rel):
        raise ApiError(415, "Only text-like project files can be read via this endpoint")
    path = safe_join(project["path"], rel)
    try:
        size = os.stat(path).st_size
    except OSError as e:
        raise ApiError(404, f"File not found: {e}")
    if size > MAX_FILE_CONTENT_BYTES:
        raise ApiError(413, "File is too large to return via API")
    try:
        content = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        raise ApiError(415, "File is not valid UTF-8 text")
    return {"ok": True, "projectId": project["id"], "path": rel, "content": content}


def _parse_usize(v: Optional[str]) -> Optional[int]:
    if v is None or not re.fullmatch(r"\+?[0-9]+", v):
        return None
    return int(v)


# --------------------------------------------------------------------------
# Reviews
# --------------------------------------------------------------------------
_TITLE_PREFIXES = [
    "missing page", "missing-page", "missingpage",
    "duplicate page", "duplicate-page", "duplicatepage",
    "possible duplicate", "possible-duplicate", "possibleduplicate",
    "缺失页面", "缺少页面", "重复页面", "疑似重复",
]


def normalize_review_title(title: str) -> str:
    trimmed = title.lstrip()
    lower = trimmed.lower()
    rest = trimmed
    for prefix in _TITLE_PREFIXES:
        if not lower.startswith(prefix):
            continue
        suffix = trimmed[len(prefix):]
        if not suffix:
            continue
        if suffix[0] in (":", "："):
            rest = suffix[1:].lstrip()
            break
    return " ".join(rest.split()).lower()


def review_id_for_parts(item_type: str, title: str) -> str:
    key = f"{item_type}::{normalize_review_title(title)}"
    h = 0x811C9DC5
    data = key.encode("utf-16-le")
    for i in range(0, len(data), 2):
        unit = data[i] | (data[i + 1] << 8)
        h ^= unit
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"review-{h:08x}"


def _str(item: Any, key: str) -> Optional[str]:
    if isinstance(item, dict) and isinstance(item.get(key), str):
        return item[key]
    return None


def stable_review_id(item: Any) -> Optional[str]:
    t, title = _str(item, "type"), _str(item, "title")
    if t is None or title is None:
        return None
    return review_id_for_parts(t, title)


def review_id_matches(item: Any, requested: str) -> bool:
    return _str(item, "id") == requested or stable_review_id(item) == requested


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def sanitize_review_item(item: Any) -> dict:
    out: Dict[str, Any] = {}
    sid = stable_review_id(item)
    if sid is not None:
        out["id"] = sid
    elif _str(item, "id") is not None:
        out["id"] = item["id"]
    for key in ("type", "title", "description", "sourcePath"):
        if _str(item, key) is not None:
            out[key] = item[key]
    for key in ("affectedPages", "searchQueries"):
        vals = item.get(key) if isinstance(item, dict) else None
        if isinstance(vals, list):
            out[key] = [v for v in vals if isinstance(v, str)]
    opts = item.get("options") if isinstance(item, dict) else None
    if isinstance(opts, list):
        options = []
        for o in opts:
            if not isinstance(o, dict):
                continue
            s: Dict[str, Any] = {}
            if isinstance(o.get("label"), str):
                s["label"] = o["label"]
            if isinstance(o.get("action"), str):
                s["action"] = o["action"]
            if s:
                options.append(s)
        out["options"] = options
    if isinstance(item, dict) and isinstance(item.get("resolved"), bool):
        out["resolved"] = item["resolved"]
    if _str(item, "resolvedAction") is not None:
        out["resolvedAction"] = item["resolvedAction"]
    if isinstance(item, dict) and _is_num(item.get("createdAt")):
        v = float(item["createdAt"])
        if v == v and v not in (float("inf"), float("-inf")):
            out["createdAt"] = v
    return out


def _merge_str_array(existing: dict, incoming: dict, key: str) -> None:
    ev = existing.get(key)
    values = [v for v in ev if isinstance(v, str)] if isinstance(ev, list) else []
    iv = incoming.get(key)
    for v in (iv if isinstance(iv, list) else []):
        if isinstance(v, str) and v not in values:
            values.append(v)
    if values:
        existing[key] = values


def merge_sanitized_review(existing: dict, incoming: dict) -> None:
    resolved = bool(existing.get("resolved") is True or incoming.get("resolved") is True)
    existing["resolved"] = resolved
    if resolved and "resolvedAction" not in existing:
        if isinstance(incoming.get("resolvedAction"), str):
            existing["resolvedAction"] = incoming["resolvedAction"]
    for key in ("description", "sourcePath"):
        cur = existing.get(key)
        if not isinstance(cur, str) or cur == "":
            if isinstance(incoming.get(key), str):
                existing[key] = incoming[key]
    _merge_str_array(existing, incoming, "affectedPages")
    _merge_str_array(existing, incoming, "searchQueries")
    options = list(existing["options"]) if isinstance(existing.get("options"), list) else []
    inc_opts = incoming.get("options")
    for o in (inc_opts if isinstance(inc_opts, list) else []):
        action = o.get("action") if isinstance(o, dict) else None
        present = isinstance(action, str) and any(
            isinstance(e, dict) and e.get("action") == action for e in options
        )
        if not present:
            options.append(o)
    if options:
        existing["options"] = options
    if _is_num(incoming.get("createdAt")):
        ic = float(incoming["createdAt"])
        ec = float(existing["createdAt"]) if _is_num(existing.get("createdAt")) else ic
        existing["createdAt"] = min(ec, ic)


def _review_path(project_path: str) -> Path:
    return Path(project_path) / ".llm-wiki" / "review.json"


def read_raw_review_array(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ApiError(500, f"Failed to read review state: {e}")
    try:
        return json.loads(raw)
    except ValueError as e:
        raise ApiError(500, f"Invalid review state JSON: {e}")


def write_raw_review_array(path: Path, parsed: Any) -> None:
    try:
        path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        raise ApiError(500, f"Failed to write review state: {e}")


def parse_review_query(query: str) -> dict:
    params = parse_query(query)
    status = params.get("status", "unresolved")
    if status in ("unresolved", "pending"):
        status = "unresolved"
    elif status in ("resolved", "all"):
        pass
    else:
        raise ApiError(400, f"Invalid review status '{status}'. Expected unresolved, resolved, or all")
    t = params.get("type")
    item_type = t.strip() if t is not None and t.strip() else None
    lim = _parse_usize(params.get("limit"))
    limit = min(max(lim if lim is not None else DEFAULT_MAX_REVIEWS, 1), HARD_MAX_REVIEWS)
    return {"status": status, "type": item_type, "limit": limit}


def load_review_items(project_path: str, query: dict) -> List[dict]:
    parsed = read_raw_review_array(_review_path(project_path))
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        raise ApiError(500, "Invalid review state JSON: expected an array")
    normalized: List[dict] = []
    index_by_id: Dict[str, int] = {}
    for item in parsed:
        s = sanitize_review_item(item)
        rid = s.get("id")
        if isinstance(rid, str):
            if rid in index_by_id:
                merge_sanitized_review(normalized[index_by_id[rid]], s)
                continue
            index_by_id[rid] = len(normalized)
        normalized.append(s)
    reviews: List[dict] = []
    for item in normalized:
        resolved = item.get("resolved") is True
        st = query["status"]
        if (st == "unresolved" and resolved) or (st == "resolved" and not resolved):
            continue
        if query["type"] is not None and item.get("type") != query["type"]:
            continue
        if len(reviews) >= query["limit"]:
            break
        reviews.append(item)
    return reviews


def handle_reviews(project_id: str, query: str) -> dict:
    project = resolve_project(project_id)
    q = parse_review_query(query)
    reviews = load_review_items(project["path"], q)
    return {"ok": True, "projectId": project["id"], "status": q["status"],
            "count": len(reviews), "reviews": reviews}


def apply_resolution(item: Any, resolved: bool, action: Optional[str]) -> None:
    if isinstance(item, dict):
        item["resolved"] = resolved
        if not resolved:
            item.pop("resolvedAction", None)
        elif action is not None:
            item["resolvedAction"] = action


def patch_review_item(project_path: str, review_id: str, resolved: bool, action: Optional[str]) -> bool:
    path = _review_path(project_path)
    parsed = read_raw_review_array(path)
    if parsed is None:
        return False
    if not isinstance(parsed, list):
        raise ApiError(500, "Invalid review state JSON: expected an array")
    found = False
    for item in parsed:
        if not review_id_matches(item, review_id):
            continue
        apply_resolution(item, resolved, action)
        sid = stable_review_id(item)
        if sid is not None and isinstance(item, dict):
            item["id"] = sid
        found = True
    if not found:
        return False
    write_raw_review_array(path, parsed)
    return True


def resolve_review_items(project_path: str, ids: List[str], action: Optional[str]) -> Tuple[List[str], List[str]]:
    path = _review_path(project_path)
    parsed = read_raw_review_array(path)
    if parsed is None:
        return [], list(ids)
    if not isinstance(parsed, list):
        raise ApiError(500, "Invalid review state JSON: expected an array")
    found: set = set()
    for item in parsed:
        raw_id = _str(item, "id")
        sid = stable_review_id(item)
        matched = next((i for i in ids if raw_id == i or sid == i), None)
        if matched is None:
            continue
        apply_resolution(item, True, action)
        if sid is not None and isinstance(item, dict):
            item["id"] = sid
        found.add(matched)
    if found:
        write_raw_review_array(path, parsed)
    return [i for i in ids if i in found], [i for i in ids if i not in found]


def _json_body(body: str, prefix: str, allow_empty: bool = False) -> dict:
    if allow_empty and not body.strip():
        return {}
    try:
        parsed = json.loads(body)
    except ValueError as e:
        raise ApiError(400, f"{prefix}: {e}")
    if not isinstance(parsed, dict):
        raise ApiError(400, f"{prefix}: invalid type: expected a JSON object")
    return parsed


def _opt_type(req: dict, key: str, typ: Any, prefix: str, what: str) -> Any:
    v = req.get(key)
    if v is None:
        return None
    if typ is int:
        ok = isinstance(v, int) and not isinstance(v, bool) and v >= 0
    else:
        ok = isinstance(v, typ)
    if not ok:
        raise ApiError(400, f"{prefix}: invalid type for field `{key}`, expected {what}")
    return v


def _req_str(req: dict, key: str, prefix: str) -> str:
    if key not in req:
        raise ApiError(400, f"{prefix}: missing field `{key}`")
    if not isinstance(req[key], str):
        raise ApiError(400, f"{prefix}: invalid type for field `{key}`, expected a string")
    return req[key]


def handle_patch_review(project_id: str, review_id: str, body: str) -> dict:
    project = resolve_project(project_id)
    p = "Invalid request body"
    req = _json_body(body, p, allow_empty=True)
    resolved = _opt_type(req, "resolved", bool, p, "a boolean")
    action = _opt_type(req, "action", str, p, "a string")
    resolved = True if resolved is None else resolved
    if patch_review_item(project["path"], review_id, resolved, action):
        return {"ok": True, "projectId": project["id"], "reviewId": review_id, "resolved": resolved}
    raise ApiError(404, f"Review item '{review_id}' not found")


def handle_bulk_resolve_reviews(project_id: str, body: str) -> dict:
    project = resolve_project(project_id)
    p = "Invalid request body"
    req = _json_body(body, p)
    if "ids" not in req:
        raise ApiError(400, f"{p}: missing field `ids`")
    ids = req["ids"]
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ApiError(400, f"{p}: invalid type for field `ids`, expected an array of strings")
    action = _opt_type(req, "action", str, p, "a string")
    if not ids:
        raise ApiError(400, "ids must be a non-empty array")
    resolved, not_found = resolve_review_items(project["path"], ids, action)
    return {"ok": True, "projectId": project["id"], "resolved": resolved,
            "notFound": not_found, "count": len(resolved)}


# --------------------------------------------------------------------------
# Search / page write / embed / rescan
# --------------------------------------------------------------------------
def load_embedding_config() -> Optional[dict]:
    parsed = load_app_state()
    v = parsed.get("embeddingConfig") if parsed else None
    return v if isinstance(v, dict) else None


SEARCH_NOTE = (
    "Search uses the shared backend hybrid retrieval service, combining keyword, vector, and "
    "one-hop knowledge-graph candidates. When embeddingConfig is enabled, the API automatically "
    "includes LanceDB vector results; clients may also pass queryEmbedding explicitly."
)


def handle_search(project_id: str, body: str) -> dict:
    project = resolve_project(project_id)
    p = "Invalid JSON"
    req = _json_body(body, p)
    query = _req_str(req, "query", p)
    top_k = _opt_type(req, "topK", int, p, "a non-negative integer")
    include_content = _opt_type(req, "includeContent", bool, p, "a boolean")
    emb = req.get("queryEmbedding")
    if emb is not None and not (isinstance(emb, list) and all(_is_num(x) for x in emb)):
        raise ApiError(400, f"{p}: invalid type for field `queryEmbedding`, expected an array of numbers")
    if not query.strip():
        raise ApiError(400, "query is required")
    top_k = min(max(top_k if top_k is not None else 10, 1), MAX_SEARCH_RESULTS)
    try:
        search = _lazy("commands.search")
    except ImportError:
        raise ApiError(501, "search backend not available")
    try:
        qe = _call(search.resolve_query_embedding, query, emb, load_embedding_config())
    except Exception as e:
        raise ApiError(400, _exc_message(e))
    try:
        res = _call(search.search_project_inner, project["path"], query, top_k,
                    bool(include_content), qe)
    except Exception as e:
        raise ApiError(500, _exc_message(e))
    res = _jsonable(res)
    return {
        "ok": True,
        "projectId": project["id"],
        "mode": _field(res, "mode"),
        "note": SEARCH_NOTE,
        "tokenHits": _field(res, "tokenHits", "token_hits"),
        "vectorHits": _field(res, "vectorHits", "vector_hits"),
        "graphHits": _field(res, "graphHits", "graph_hits"),
        "results": _jsonable(_field(res, "results", default=[])),
    }


def handle_write_page(project_id: str, body: str) -> dict:
    project = resolve_project(project_id)
    p = "Invalid JSON"
    req = _json_body(body, p)
    unknown = set(req) - {"path", "content", "allowOverwrite"}
    if unknown:
        raise ApiError(400, f"{p}: unknown field `{sorted(unknown)[0]}`, expected one of `path`, `content`, `allowOverwrite`")
    path = _req_str(req, "path", p)
    content = _req_str(req, "content", p)
    allow = req.get("allowOverwrite", False)
    if not isinstance(allow, bool):
        raise ApiError(400, f"{p}: invalid type for field `allowOverwrite`, expected a boolean")
    if not path.strip():
        raise ApiError(400, "path is required")
    try:
        tools = _lazy("agent.tools")
        fn = tools.write_wiki_page_verified
    except (ImportError, AttributeError):
        raise ApiError(501, AGENT_UNAVAILABLE)
    try:
        result = _call(fn, project["path"], path, content, allow)
    except Exception as e:
        msg = _exc_message(e)
        lower = msg.lower()
        if "without allowoverwrite" in lower:
            status = 409
        elif "path" in lower or "markdown file" in lower or "too large" in lower:
            status = 400
        else:
            status = 500
        raise ApiError(status, msg)
    return {"ok": True, "projectId": project["id"], "result": _jsonable(result)}


_EMBED_STATUS = {"invalidrequest": 400, "notfound": 404, "provider": 502, "storage": 500, "timeout": 504}


def handle_embed_page(project_id: str, body: str) -> dict:
    project = resolve_project(project_id)
    p = "Invalid JSON"
    req = _json_body(body, p)
    path = _req_str(req, "path", p)
    force = req.get("force", False)
    if not isinstance(force, bool):
        raise ApiError(400, f"{p}: invalid type for field `force`, expected a boolean")
    if not path.strip():
        raise ApiError(400, "path is required")
    config = load_embedding_config()
    if config is None:
        raise ApiError(400, "Embedding is not configured")
    if not _try_acquire("embed", MAX_IN_FLIGHT_PAGE_EMBEDS):
        raise ApiError(503, "Too many page indexing requests are already running")
    try:
        try:
            pe = _lazy("commands.page_embedding")
            fn = pe.embed_wiki_page
        except (ImportError, AttributeError):
            raise ApiError(501, "page embedding backend not available")
        try:
            result = _call(fn, project["path"], path, config, force)
        except ApiError:
            raise
        except Exception as e:
            kind = str(getattr(getattr(e, "kind", None), "name", getattr(e, "kind", "")) or "")
            kind = kind.replace("_", "").replace("-", "").lower()
            raise ApiError(_EMBED_STATUS.get(kind, 500), _exc_message(e))
    finally:
        _release("embed")
    return {"ok": True, "projectId": project["id"], "result": _jsonable(result)}


def load_source_watch_config(project_id: str) -> Optional[dict]:
    parsed = load_app_state()
    if not parsed:
        return None
    settings = parsed.get("sourceWatchConfig")
    if isinstance(settings, dict):
        value = settings.get(project_id) if project_id in settings else settings.get("default")
        if isinstance(value, dict):
            return value
    legacy = parsed.get("projectFileSyncEnabled")
    if isinstance(legacy, dict):
        v = legacy.get(project_id) if project_id in legacy else legacy.get("default")
        if isinstance(v, bool):
            return {"enabled": v}
    return None


def handle_rescan(project_id: str) -> dict:
    project = resolve_project(project_id)
    invalidate_graph_cache(project["path"])
    cfg = load_source_watch_config(project["id"])
    try:
        fs_mod = _lazy("commands.file_sync")
        fn = fs_mod.rescan_project_files
    except (ImportError, AttributeError):
        raise ApiError(501, "file sync backend not available")
    try:
        result = _call(fn, project_id=project["id"], project_path=project["path"],
                       source_watch_config=cfg, watch_roots_only=None)
    except Exception as e:
        raise ApiError(500, _exc_message(e))
    return {"ok": True, "projectId": project["id"], "result": _jsonable(result)}


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def extract_title(content: str, file_name: str) -> str:
    has_fm = content.startswith("---")
    in_fm = has_fm
    closed = False
    for line in content.splitlines()[1 if has_fm else 0:]:
        t = line.strip()
        if in_fm and t == "---":
            in_fm = False
            closed = True
            continue
        if in_fm and t.startswith("title:"):
            return t[len("title:"):].strip().strip('"').strip("'")
        if has_fm and not closed:
            continue
        if t.startswith("# "):
            return t[2:].strip()
    name = file_name
    while name.endswith(".md"):
        name = name[:-3]
    return name.replace("-", " ")


def _normalize_related_value(value: str) -> Optional[str]:
    v = value.strip().strip('"').strip("'").strip()
    if v.startswith("[[") and v.endswith("]]") and len(v) >= 4:
        v = v[2:-2]
    v = v.strip()
    return v or None


def extract_related_frontmatter(content: str) -> List[str]:
    normalized = content.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return []
    rest = normalized[4:]
    end = rest.find("\n---")
    if end < 0:
        return []
    lines = rest[:end].split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("related:"):
            i += 1
            continue
        value = line[len("related:"):].strip()
        if value.startswith("[") and value.endswith("]"):
            for item in value[1:-1].split(","):
                n = _normalize_related_value(item)
                if n:
                    out.append(n)
        elif value == "":
            i += 1
            while i < len(lines):
                t = lines[i].strip()
                if not t.startswith("-"):
                    break
                n = _normalize_related_value(t[1:])
                if n:
                    out.append(n)
                i += 1
            continue
        i += 1
    return out


def extract_graph_links(content: str) -> List[str]:
    links: List[str] = []
    rest = content
    while True:
        s = rest.find("[[")
        if s < 0:
            break
        rest = rest[s + 2:]
        e = rest.find("]]")
        if e < 0:
            break
        target = rest[:e].split("|")[0].strip()
        if target:
            links.append(target)
        rest = rest[e + 2:]
    for rel in extract_related_frontmatter(content):
        if rel not in links:
            links.append(rel)
    return links


def extract_type(content: str) -> str:
    for line in content.splitlines():
        t = line.strip()
        if t.startswith("type:"):
            return t[5:].strip().strip('"').strip("'").lower()
    return "other"


def name_alias_keys(raw: str) -> List[str]:
    n = raw.strip().replace("\\", "/").lower().strip("/")
    if n.startswith("./"):
        n = n[2:]
    if n.startswith("wiki/"):
        n = n[5:]
    if n.endswith(".md"):
        n = n[:-3]
    if not n:
        return []
    slug = n.replace(" ", "-")
    return [n] if slug == n else [n, slug]


def link_alias_keys(raw: str) -> List[str]:
    return name_alias_keys(raw.split("|")[0].split("#")[0].strip())


class GraphAliases:
    def __init__(self) -> None:
        self.exact_ids: Dict[str, str] = {}
        self.ids: Dict[str, Optional[str]] = {}
        self.stems: Dict[str, Optional[str]] = {}
        self.titles: Dict[str, Optional[str]] = {}


def _insert_alias(table: Dict[str, Optional[str]], keys: List[str], ident: str) -> None:
    for k in keys:
        if k not in table:
            table[k] = ident
        elif table[k] is not None and table[k] != ident:
            table[k] = None


def graph_aliases(raw: Dict[str, Tuple[str, str, str, List[str]]]) -> GraphAliases:
    a = GraphAliases()
    for ident, (title, _t, path, _l) in raw.items():
        a.exact_ids[ident] = ident
        _insert_alias(a.ids, name_alias_keys(ident), ident)
        canonical = path[5:] if path.startswith("wiki/") else path
        while canonical.endswith(".md"):
            canonical = canonical[:-3]
        _insert_alias(a.ids, name_alias_keys(canonical), ident)
        _insert_alias(a.stems, name_alias_keys(ident.rsplit("/", 1)[-1]), ident)
        _insert_alias(a.titles, name_alias_keys(title), ident)
    return a


def resolve_link(raw: str, aliases: GraphAliases) -> Optional[str]:
    exact = raw.split("|")[0].split("#")[0].strip()
    if exact in aliases.exact_ids:
        return aliases.exact_ids[exact]
    keys = link_alias_keys(raw)
    for tier in (aliases.ids, aliases.stems, aliases.titles):
        for k in keys:
            if k in tier:
                return tier[k]
    return None


def graph_files(project_path: str) -> Tuple[int, List[Path]]:
    wiki_root = Path(project_path) / "wiki"
    files: List[Path] = []
    for dirpath, _dirs, names in os.walk(wiki_root, followlinks=False):
        for n in names:
            p = Path(dirpath) / n
            if p.suffix.lower() == ".md" and not p.is_symlink() and p.is_file():
                files.append(p)
    files.sort(key=lambda p: p.parts)
    fp: List[Any] = []
    for p in files:
        try:
            st = os.stat(p)
            fp.append((str(p), st.st_size, st.st_mtime_ns))
        except OSError:
            fp.append((str(p),))
    return hash(tuple(fp)), files


def build_graph_from_files(project_path: str, files: List[Path]) -> Tuple[List[dict], List[dict]]:
    wiki_root = Path(project_path) / "wiki"
    stem_counts: Dict[str, int] = {}
    for p in files:
        stem_counts[p.stem] = stem_counts.get(p.stem, 0) + 1
    raw: Dict[str, Tuple[str, str, str, List[str]]] = {}
    for p in files:
        try:
            content = p.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            rel = p.relative_to(wiki_root)
        except ValueError:
            rel = p
        relative_id = str(rel.with_suffix("")).replace("\\", "/")
        ident = p.stem if stem_counts.get(p.stem) == 1 else relative_id
        if not ident:
            continue
        title = extract_title(content, p.name)
        ntype = extract_type(content)
        raw[ident] = (title, ntype, relative_to_project(project_path, p), extract_graph_links(content))
    raw = {k: v for k, v in raw.items() if v[1] != "query"}
    raw = {k: raw[k] for k in sorted(raw)}
    aliases = graph_aliases(raw)
    link_count: Dict[str, int] = {k: 0 for k in raw}
    seen: set = set()
    edges: List[dict] = []
    for source, (_t, _ty, _p, links) in raw.items():
        for link in links:
            target = resolve_link(link, aliases)
            if target is None or target == source:
                continue
            key = f"{source}::{target}" if source < target else f"{target}::{source}"
            if key not in seen:
                seen.add(key)
                link_count[source] = link_count.get(source, 0) + 1
                link_count[target] = link_count.get(target, 0) + 1
                edges.append({"source": source, "target": target, "weight": 1.0})
    nodes = [
        {"id": i, "label": t, "nodeType": ty, "path": pa, "linkCount": link_count.get(i, 0)}
        for i, (t, ty, pa, _l) in raw.items()
    ]
    return nodes, edges


class CachedGraph:
    def __init__(self, fingerprint: int, nodes: List[dict], edges: List[dict]):
        self.loaded_at = time.monotonic()
        self.fingerprint = fingerprint
        self.nodes = nodes
        self.edges = edges


_graph_cache: Dict[str, CachedGraph] = {}
_graph_lock = threading.Lock()


def invalidate_graph_cache(project_path: str) -> None:
    with _graph_lock:
        _graph_cache.pop(project_path, None)


def load_graph(project_path: str) -> CachedGraph:
    with _graph_lock:
        e = _graph_cache.get(project_path)
        if e is not None and time.monotonic() - e.loaded_at < GRAPH_CACHE_TTL:
            return e
    fingerprint, files = graph_files(project_path)
    with _graph_lock:
        e = _graph_cache.get(project_path)
        if e is not None and e.fingerprint == fingerprint:
            return e
    nodes, edges = build_graph_from_files(project_path, files)
    graph = CachedGraph(fingerprint, nodes, edges)
    with _graph_lock:
        now = time.monotonic()
        for k in [k for k, v in _graph_cache.items() if now - v.loaded_at >= GRAPH_CACHE_TTL]:
            del _graph_cache[k]
        if len(_graph_cache) >= GRAPH_CACHE_MAX_PROJECTS and project_path not in _graph_cache:
            oldest = min(_graph_cache, key=lambda k: _graph_cache[k].loaded_at)
            del _graph_cache[oldest]
        _graph_cache[project_path] = graph
    return graph


def select_graph_page(nodes: List[dict], edges: List[dict], q: Optional[str], node_type: Optional[str],
                      offset: int, limit: int, paged_edges: bool) -> dict:
    def matches(n: dict) -> bool:
        return ((q is None or q in n["id"].lower() or q in n["label"].lower())
                and (node_type is None or n["nodeType"] == node_type))

    matched = [n for n in nodes if matches(n)]
    total = len(matched)
    filtered_ids = {n["id"] for n in matched} if paged_edges else None
    page = matched[offset:offset + limit]
    page_ids = {n["id"] for n in page}
    out_edges = []
    for e in edges:
        if paged_edges:
            if e["source"] in page_ids and e["target"] in filtered_ids:  # type: ignore[operator]
                out_edges.append(e)
        elif e["source"] in page_ids and e["target"] in page_ids:
            out_edges.append(e)
    return {"nodes": page, "edges": out_edges, "total_count": total,
            "has_more": offset + len(page) < total}


def handle_graph(project_id: str, query: str) -> dict:
    project = resolve_project(project_id)
    params = parse_query(query)
    q = params["q"].lower() if "q" in params else None
    node_type = params["nodeType"].lower() if "nodeType" in params else None
    scope = params.get("edgeScope")
    if scope == "filtered":
        paged_edges = True
    elif scope in ("page", None):
        paged_edges = False
    else:
        raise ApiError(400, "edgeScope must be 'page' or 'filtered'")
    if "offset" in params:
        offset = _parse_usize(params["offset"])
        if offset is None:
            raise ApiError(400, "offset must be a non-negative integer")
    else:
        offset = 0
    if "limit" in params:
        lim = _parse_usize(params["limit"])
        if lim is None or lim <= 0:
            raise ApiError(400, "limit must be a positive integer")
        limit = min(max(lim, 1), 1000)
    else:
        limit = 200
    try:
        graph = load_graph(project["path"])
    except ApiError:
        raise
    except Exception as e:
        raise ApiError(500, str(e))
    page = select_graph_page(graph.nodes, graph.edges, q, node_type, offset, limit, paged_edges)
    return {"ok": True, "projectId": project["id"], "nodes": page["nodes"], "edges": page["edges"],
            "offset": offset, "limit": limit, "totalCount": page["total_count"],
            "hasMore": page["has_more"]}


# --------------------------------------------------------------------------
# Chat (thin adapters over the agent runtime, ported in a later wave)
# --------------------------------------------------------------------------
class CancellationRegistry:
    """(project, session, run) -> threading.Event; mirrors agent::cancel::AgentCancellationRegistry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: Dict[Tuple[str, str, str], threading.Event] = {}

    def start(self, project_id: str, session_id: str, run_id: str) -> threading.Event:
        ev = threading.Event()
        with self._lock:
            self._runs[(project_id, session_id, run_id)] = ev
        return ev

    def finish(self, project_id: str, session_id: str, run_id: str) -> None:
        with self._lock:
            self._runs.pop((project_id, session_id, run_id), None)

    def cancel(self, project_id: str, session_id: str, run_id: Optional[str] = None) -> bool:
        hit = False
        with self._lock:
            for (p, s, r), ev in self._runs.items():
                if p == project_id and s == session_id and (run_id is None or r == run_id):
                    ev.set()
                    hit = True
        return hit


cancellations = CancellationRegistry()


def _runtime_module():
    try:
        return _lazy("agent.runtime")
    except ImportError:
        raise ApiError(501, AGENT_UNAVAILABLE)


def project_llm_config(parsed: dict, project_id: str) -> Optional[dict]:
    global_cfg = parsed.get("llmConfig")
    global_cfg = global_cfg if isinstance(global_cfg, dict) else None
    overrides = parsed.get("projectLlmOverrides")
    project = overrides.get(project_id) if isinstance(overrides, dict) else None
    if not isinstance(project, dict) or project.get("enabled") is not True:
        return global_cfg
    profile = project.get("profile")
    preset_id = project.get("presetId")
    if not isinstance(profile, dict) or not isinstance(preset_id, str):
        return None
    profile = dict(profile)
    configs = parsed.get("providerConfigs")
    provider = configs.get(preset_id) if isinstance(configs, dict) else None
    provider = provider if isinstance(provider, dict) else None
    presets = parsed.get("customLlmPresets")
    custom_exists = isinstance(presets, list) and any(
        isinstance(x, dict) and x.get("id") == preset_id for x in presets)
    if preset_id.startswith("custom-") and not custom_exists:
        return global_cfg
    if provider is not None:
        for key in ("apiKey", "apiMode", "azureApiVersion", "maxContextSize", "reasoning",
                    "streamingEnabled", "customHeaders"):
            if key in provider:
                profile[key] = provider[key]
        model = project.get("model")
        if not (isinstance(model, str) and model.strip()):
            if "model" in provider:
                profile["model"] = provider["model"]
        if "baseUrl" in provider:
            profile["ollamaUrl" if profile.get("provider") == "ollama" else "customEndpoint"] = provider["baseUrl"]
    return profile


def load_agent_runtime_config(project_id: Optional[str]) -> dict:
    parsed = load_app_state()
    if not parsed:
        return {"embedding": None, "llm": None, "webSearch": None, "anytxt": None}

    def d(v: Any) -> Optional[dict]:
        return v if isinstance(v, dict) else None

    llm = project_llm_config(parsed, project_id) if project_id else None
    if llm is None:
        llm = d(parsed.get("llmConfig"))
    search_cfg = d(parsed.get("searchApiConfig"))
    return {
        "embedding": d(parsed.get("embeddingConfig")),
        "llm": llm,
        "webSearch": search_cfg,
        "anytxt": d(search_cfg.get("anyTxt")) if search_cfg else None,
    }


class PreparedChat:
    def __init__(self, project: dict, request: dict, config: dict, session_id: str, run_id: str,
                 cancel_event: threading.Event):
        self.project = project
        self.request = request
        self.config = config
        self.user_message = request["message"]
        self.persist_session = request.get("persistSession", True) is not False
        self.session_id = session_id
        self.run_id = run_id
        self.cancel_event = cancel_event


def _session_module():
    try:
        return _lazy("agent.session")
    except ImportError:
        return None


def prepare_chat(project_id: str, body: str) -> PreparedChat:
    project = resolve_project(project_id)
    p = "Invalid JSON"
    req = _json_body(body, p)
    msg = _req_str(req, "message", p)
    for key in ("sessionId", "runId"):
        if req.get(key) is not None and not isinstance(req[key], str):
            raise ApiError(400, f"{p}: invalid type for field `{key}`, expected a string")
    req["allowEmptyRetrieval"] = False
    if not msg.strip():
        raise ApiError(400, "message is required")
    if not (req.get("sessionId") or "").strip():
        req["sessionId"] = f"api_{uuid.uuid4()}"
    if not (req.get("runId") or "").strip():
        req["runId"] = f"run_{uuid.uuid4()}"
    session_id, run_id = req["sessionId"], req["runId"]
    history = req.get("history")
    if not history and not req.get("historyExplicit"):
        sess = _session_module()
        recent = getattr(sess, "recent_messages", None) if sess else None
        if recent is not None:
            try:
                msgs = _call(recent, project["path"], session_id, 12)
                req["history"] = [{"role": _field(m, "role"), "content": _field(m, "content")} for m in msgs]
            except Exception:
                pass
    _runtime_module()  # 501 early if the runtime is not importable
    config = load_agent_runtime_config(project["id"])
    ev = cancellations.start(project["id"], session_id, run_id)
    return PreparedChat(project, req, config, session_id, run_id, ev)


def _persist_chat_response(prepared: PreparedChat, response: dict) -> None:
    if not prepared.persist_session:
        return
    sess = _session_module()
    append = getattr(sess, "append_turn", None) if sess else None
    if append is None:
        return
    try:
        msg = response.get("message")
        content = msg.get("content") if isinstance(msg, dict) else msg
        _call(append, prepared.project["path"], prepared.project["id"],
              response.get("sessionId", prepared.session_id), prepared.user_message, content)
    except Exception:
        pass


def external_chat_response(response: dict) -> dict:
    rt = _runtime_module()
    redact = getattr(rt, "redact_for_external_api", None)
    events = response.get("events") or []
    if redact is not None:
        events = [redact(e) for e in events]
    msg = response.get("message")
    content = msg.get("content") if isinstance(msg, dict) else msg
    return {
        "ok": True,
        "projectId": response.get("projectId"),
        "sessionId": response.get("sessionId"),
        "mode": response.get("mode"),
        "message": {"role": "assistant", "content": content},
        "references": response.get("references"),
        "toolEvents": response.get("toolEvents"),
        "events": events,
        "usage": response.get("usage"),
    }


def _run_chat(prepared: PreparedChat, on_event: Optional[Callable[[dict], None]]) -> dict:
    rt = _runtime_module()
    return _call(rt.run_chat, prepared.project, prepared.request, prepared.config,
                 prepared.cancel_event, on_event)


def handle_chat(project_id: str, body: str) -> Tuple[int, dict]:
    prepared = prepare_chat(project_id, body)
    try:
        try:
            response = _run_chat(prepared, None)
        finally:
            cancellations.finish(prepared.project["id"], prepared.session_id, prepared.run_id)
    except ApiError:
        raise
    except Exception as e:
        msg = _exc_message(e)
        if msg == "message is required":
            raise ApiError(400, msg)
        if msg == "Agent turn cancelled":
            raise ApiError(499, msg)
        raise ApiError(502, msg)
    _persist_chat_response(prepared, response)
    return 200, external_chat_response(response)


def handle_cancel_chat(project_id: str, session_id: str) -> dict:
    project = resolve_project(project_id)
    _runtime_module()
    return {"ok": True, "cancelled": cancellations.cancel(project["id"], session_id, None),
            "sessionId": session_id}


def sse_frame(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


def _chat_sse_response(prepared: PreparedChat, slot_release: Callable[[], None]) -> StreamingResponse:
    q: "queue.Queue[Tuple[bytes, bool]]" = queue.Queue(maxsize=SSE_QUEUE_CAPACITY)
    closed = threading.Event()
    project_id = prepared.project["id"]
    q.put_nowait((sse_frame("meta", {"projectId": project_id, "sessionId": prepared.session_id,
                                     "runId": prepared.run_id}), False))
    rt = _runtime_module()
    redact = getattr(rt, "redact_for_external_api", None)

    def send_terminal(frame: bytes) -> None:
        while not closed.is_set():
            try:
                q.put((frame, True), timeout=0.5)
                return
            except queue.Full:
                continue

    def sink(event: Any) -> None:
        if redact is not None:
            event = redact(event)
        try:
            # Intermediate events may be dropped for a slow client; `done` carries everything.
            q.put_nowait((sse_frame("agent", event), False))
        except queue.Full:
            pass

    def worker() -> None:
        try:
            response = _run_chat(prepared, sink)
        except BaseException as e:  # noqa: BLE001
            cancellations.finish(project_id, prepared.session_id, prepared.run_id)
            msg = _exc_message(e) or "Agent chat task panicked"
            event = "cancelled" if msg == "Agent turn cancelled" else "error"
            send_terminal(sse_frame(event, {"ok": False, "error": msg}))
            return
        cancellations.finish(project_id, prepared.session_id, prepared.run_id)
        try:
            _persist_chat_response(prepared, response)
            send_terminal(sse_frame("done", external_chat_response(response)))
        except Exception as e:  # noqa: BLE001
            send_terminal(sse_frame("error", {"ok": False, "error": str(e)}))

    threading.Thread(target=worker, daemon=True, name="llm-wiki-chat-sse").start()

    async def gen():
        finished = False
        try:
            while True:
                try:
                    frame, terminal = await run_in_threadpool(q.get, True, SSE_HEARTBEAT_INTERVAL)
                except queue.Empty:
                    yield b": keepalive\n\n"
                    continue
                yield frame
                if terminal:
                    finished = True
                    return
        finally:
            closed.set()
            if not finished:  # client went away: cancel the run
                cancellations.cancel(project_id, prepared.session_id, prepared.run_id)
            slot_release()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
def health_body() -> dict:
    return {
        "ok": True,
        "status": "running",
        "version": __version__,
        "authRequired": api_auth_required(),
        "authConfigured": api_token() is not None,
        "tokenSource": api_token_source(),
        "enabled": api_enabled(),
        "mcpEnabled": api_mcp_enabled(),
        "allowUnauthenticated": api_allow_unauthenticated(),
        "allowLanAccess": api_allow_lan_access(),
        "agent": {"chat": True, "streaming": True, "streamProtocol": "sse"},
    }


def handle_request(method: str, path: str, query: str, body: str,
                   headers: List[Tuple[str, str]]) -> Tuple[int, dict]:
    """Port of `handle_request`: returns (status, json body)."""
    try:
        if path == "/health" or path == f"{API_PREFIX}/health":
            return 200, health_body()
        if not path.startswith(API_PREFIX):
            raise ApiError(404, "Not found")
        if not api_enabled():
            raise ApiError(503, "API server is disabled in Settings → API Server")
        if is_token_required_request(method, path) and not is_token_authorized(query, headers):
            raise ApiError(401, "Unauthorized")
        if not is_authorized(query, headers):
            raise ApiError(401, "Unauthorized")
        if method not in ("GET", "POST", "PATCH"):
            raise ApiError(405, "Method not allowed")
        parts = api_path_parts(path)
        if parts is None:
            raise ApiError(404, "Not found")
        n = len(parts)
        if parts[:1] == ["projects"]:
            if method == "GET" and n == 1:
                return 200, handle_projects()
            if n >= 3:
                pid, tail = parts[1], parts[2:]
                if method == "GET":
                    if tail == ["files"]:
                        return 200, handle_files(pid, query)
                    if tail == ["files", "content"]:
                        return 200, handle_file_content(pid, query)
                    if tail == ["reviews"]:
                        return 200, handle_reviews(pid, query)
                    if tail == ["graph"]:
                        return 200, handle_graph(pid, query)
                elif method == "POST":
                    if tail == ["reviews", "resolve"]:
                        return 200, handle_bulk_resolve_reviews(pid, body)
                    if tail == ["search"]:
                        return 200, handle_search(pid, body)
                    if tail == ["sources", "rescan"]:
                        return 200, handle_rescan(pid)
                    if tail == ["pages", "embed"]:
                        return 200, handle_embed_page(pid, body)
                    if tail == ["pages", "write"]:
                        return 200, handle_write_page(pid, body)
                    if tail == ["chat"]:
                        return handle_chat(pid, body)
                    if len(tail) == 3 and tail[0] == "chat" and tail[2] == "cancel":
                        return 200, handle_cancel_chat(pid, tail[1])
                elif method == "PATCH":
                    if len(tail) == 2 and tail[0] == "reviews":
                        return 200, handle_patch_review(pid, tail[1], body)
        raise ApiError(404, "Not found")
    except ApiError as e:
        return e.status, _err_body(e.message)
    except Exception as e:  # panic guard
        print(f"[API Server] request panicked: {e!r}", file=sys.stderr)
        return 500, _err_body("Internal API server error")


async def _read_body(request: Request, limit: int) -> str:
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > limit:
        raise ApiError(400, "Request body too large")
    chunks: List[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise ApiError(400, "Request body too large")
        chunks.append(chunk)
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError:
        raise ApiError(400, "Request body must be UTF-8")


def _json(status: int, body: dict) -> JSONResponse:
    return JSONResponse(body, status_code=status)


@router.api_route("/api/v1", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
                  include_in_schema=False)
@router.api_route("/api/v1/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
                  include_in_schema=False)
async def api_dispatch(request: Request):
    method = request.method.upper()
    raw = request.scope.get("raw_path")
    path = raw.decode("latin-1").split("?", 1)[0] if raw else request.url.path
    query = request.scope.get("query_string", b"").decode("latin-1")

    if should_rate_limit(method, path) and not allow_request():
        return _json(429, _err_body("Too many requests"))
    if not _try_acquire("requests", MAX_IN_FLIGHT_REQUESTS):
        return _json(503, _err_body("API server is busy"))
    release_request = True
    try:
        if method == "OPTIONS":
            return Response(status_code=204, headers={"Access-Control-Max-Age": "600"})
        headers = [(k.lower(), v) for k, v in request.headers.items()]
        try:
            body = await _read_body(request, body_limit_for_request(method, path))
        except ApiError as e:
            return _json(e.status, _err_body(e.message))

        if wants_streaming_chat(method, path, body, headers):
            if not api_enabled():
                return _json(503, _err_body("API server is disabled in Settings → API Server"))
            if not is_token_authorized(query, headers):
                return _json(401, _err_body("Unauthorized"))
            pid = chat_project_id(method, path)
            if pid is None:
                return _json(404, _err_body("Not found"))
            if not _try_acquire("chat", MAX_IN_FLIGHT_CHAT_STREAMS):
                return _json(503, _err_body("Too many concurrent Agent chat streams"))
            try:
                prepared = await run_in_threadpool(prepare_chat, pid, body)
                released = {"done": False}

                def release_all() -> None:
                    if not released["done"]:
                        released["done"] = True
                        _release("chat")
                        _release("requests")

                resp = _chat_sse_response(prepared, release_all)
                release_request = False
                return resp
            except ApiError as e:
                _release("chat")
                return _json(e.status, _err_body(e.message))
            except Exception as e:  # noqa: BLE001
                _release("chat")
                print(f"[API Server] chat setup panicked: {e!r}", file=sys.stderr)
                return _json(500, _err_body("Internal API server error"))

        status, payload = await run_in_threadpool(handle_request, method, path, query, body, headers)
        return _json(status, payload)
    finally:
        if release_request:
            _release("requests")
