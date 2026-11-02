"""Port of cors.rs: narrow CORS policy shared by the clip server and REST API."""
from __future__ import annotations

from typing import Dict, Optional

_EXACT = ("http://localhost", "http://127.0.0.1", "http://[::1]", "tauri://localhost",
          "http://tauri.localhost", "https://tauri.localhost")
_PREFIXES = ("chrome-extension://", "moz-extension://", "http://localhost:",
             "http://127.0.0.1:", "http://[::1]:")


def is_allowed_browser_origin(origin: str) -> bool:
    return origin in _EXACT or origin.startswith(_PREFIXES)


def local_cors_headers(origin: Optional[str], allow_headers: str) -> Dict[str, str]:
    headers = {
        "Access-Control-Allow-Methods": "GET, POST, PATCH, OPTIONS",
        "Access-Control-Allow-Headers": allow_headers,
        "Content-Type": "application/json",
    }
    if origin is not None and is_allowed_browser_origin(origin):
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
        headers["Access-Control-Allow-Private-Network"] = "true"
    return headers
