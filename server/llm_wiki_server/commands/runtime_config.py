"""Runtime-config commands from lib.rs plus the SSE event endpoint.

Ports: set_proxy_env, set_close_behavior, api_server_status,
api_server_reload_config, clip_server_status, mcp_server_entry_path.
Helper ports live in ../proxy.py, ../cors.py, ../server_bind.py, ../clip_server.py.

SSE: ``GET /events?topics=a,b`` streams bus events (see ../events.py). Each
message is ``event: <topic>`` + ``data: <JSON payload>``; the topic names are
the Rust ``emit`` names (``file-sync://queue-updated``, ``file-sync://changed``,
``claude-cli:<id>`` / ``:done``, ``codex-cli:<id>`` / ``:done``). Omitting
``topics`` subscribes to everything; ``*`` wildcards are allowed in topics.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .. import clip_server, events, proxy
from ..errors import CommandError
from ..registry import command, on_startup

router = APIRouter()

_close_behavior = "minimize"
_close_lock = threading.Lock()


# ----------------------------------------------------------------- SSE

@router.get("/events")
async def sse_events(request: Request, topics: Optional[str] = None):
    wanted = [t.strip() for t in (topics or "").split(",") if t.strip()]
    sub = events.subscribe(wanted)

    async def gen():
        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    topic, payload = await asyncio.wait_for(sub.queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                data = json.dumps(payload, ensure_ascii=False)
                yield f"event: {topic}\ndata: {data}\n\n"
        finally:
            events.unsubscribe(sub)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ----------------------------------------------------------------- proxy / close behaviour

@command()
def set_proxy_env(config: dict):
    cfg = proxy.parse_proxy_config(config)
    if cfg is None:
        raise CommandError("invalid args `config`: expected a proxy config object")
    summary = proxy.apply_proxy_env(cfg)
    print(f"[proxy] live update: {summary}", file=sys.stderr)
    return summary


@command()
def set_close_behavior(value: str):
    global _close_behavior
    if value not in ("ask", "minimize", "exit"):
        raise CommandError(f"Invalid close behavior: {value}")
    with _close_lock:
        _close_behavior = value  # stored for parity; no window exists headless
    return value


def close_behavior() -> str:
    return _close_behavior


# ----------------------------------------------------------------- server status

@command()
def clip_server_status():
    return clip_server.get_daemon_status()


@command()
def api_server_status():
    try:
        from . import rest_api  # written by another module/agent
    except ImportError:
        return "error: REST API server is not available"
    try:
        fn = getattr(rest_api, "get_api_status", None)
        # The REST API is mounted on this very server, so if it imported it is running.
        return str(fn()) if fn else "running"
    except Exception as e:
        return f"error: {e}"


@command()
def api_server_reload_config():
    try:
        from . import rest_api

        rest_api.invalidate_config_cache()
    except (ImportError, AttributeError):
        pass  # nothing cached
    except Exception as e:
        return f"error: {e}"
    return "ok"


@command()
def mcp_server_entry_path():
    relative = Path("mcp-server") / "dist" / "src" / "index.js"
    candidates: List[Path] = []

    def push_repo(base: Path) -> None:
        candidates.append(base / relative)
        candidates.append(base / ".." / relative)
        candidates.append(base / ".." / ".." / relative)

    here = Path(__file__).resolve()
    override = os.environ.get("LLM_WIKI_MCP_SERVER_ENTRY")
    if override:
        candidates.append(Path(override))
    push_repo(here.parents[3] / "src-tauri")  # CARGO_MANIFEST_DIR equivalent
    push_repo(here.parents[3])
    push_repo(here.parents[2])
    try:
        push_repo(Path.cwd())
    except OSError:
        pass
    res = os.environ.get("LLM_WIKI_RESOURCE_DIR")
    if res:
        candidates.append(Path(res) / relative)
    exe_dir = Path(sys.executable).parent
    candidates.append(exe_dir / relative)
    candidates.append(exe_dir / ".." / "Resources" / relative)
    for c in candidates:
        if c.is_file():
            try:
                return str(c.resolve())
            except OSError:
                return str(c)
    raise CommandError(
        "MCP server entry was not found. Run `npm run mcp:build` from the LLM Wiki repository, "
        "then reopen Settings.")


# ----------------------------------------------------------------- startup (lib.rs setup hook)

@on_startup
def _apply_stored_proxy_and_start_clip_server() -> None:
    try:
        from ..state import app_state

        cfg = proxy.parse_proxy_config(app_state().get("proxyConfig"))
        if cfg is not None:
            print(f"[proxy] {proxy.apply_proxy_env(cfg)}", file=sys.stderr)
        else:
            print("[proxy] no proxyConfig in store, requests go direct", file=sys.stderr)
    except Exception as e:
        print(f"[proxy] failed to read stored config: {e}", file=sys.stderr)
    if os.environ.get("LLM_WIKI_DISABLE_CLIP_SERVER") != "1":
        clip_server.start_clip_server()
