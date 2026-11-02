"""Port of server_bind.rs: which interface the side HTTP servers bind to."""
from __future__ import annotations

import os
from typing import Any, Optional

DEFAULT_BIND_HOST = "127.0.0.1"
PUBLIC_BIND_HOST = "0.0.0.0"
BIND_HOST_ENV = "LLM_WIKI_BIND_HOST"


def sanitize_bind_host(value: str) -> Optional[str]:
    host = value.strip()
    if not host:
        return None
    if all((c.isascii() and c.isalnum()) or c in ".-_:[]" for c in host):
        return host
    return None


def allow_lan_access_from_state(value: Any) -> bool:
    cfg = value.get("apiConfig") if isinstance(value, dict) else None
    flag = cfg.get("allowLanAccess") if isinstance(cfg, dict) else None
    return flag if isinstance(flag, bool) else False


def bind_addr(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def configured_bind_host() -> str:
    env = os.environ.get(BIND_HOST_ENV)
    if env is not None:
        h = sanitize_bind_host(env)
        if h:
            return h
    try:
        from .state import app_state

        if allow_lan_access_from_state({"apiConfig": app_state().get("apiConfig")}):
            return PUBLIC_BIND_HOST
    except Exception:
        pass
    return DEFAULT_BIND_HOST
