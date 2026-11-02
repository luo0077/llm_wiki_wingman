"""Port of src/lib/proxy-config.ts: global HTTP/HTTPS proxy configuration helpers.

A proxy config is a dict with the TS camelCase keys:
``{"enabled": bool, "url": str, "bypassLocal": bool, "acceptInvalidCerts": bool}``.
The server-side application of the config (env vars, TLS policy) lives in
``llm_wiki_server.proxy``; outbound HTTP honors it through ``http_fetch``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .llm_jscompat import js_trim, parse_url

DEFAULT_PROXY_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "url": "",
    "bypassLocal": True,
    "acceptInvalidCerts": False,
}

DEFAULT_BYPASS_LIST = "localhost,127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.local"

SUPPORTED_SCHEMES = ("http:", "https:")


def validate_proxy_url(url: str) -> Dict[str, Any]:
    """Return ``{"ok": True}`` or ``{"ok": False, "error": str}``."""
    trimmed = js_trim(url)
    if trimmed == "":
        return {"ok": False, "error": "URL is empty"}
    parsed = parse_url(trimmed)
    if parsed is None:
        return {"ok": False, "error": "Not a valid URL"}
    if not parsed.protocol:
        return {"ok": False, "error": "URL is missing a scheme (http:// or https://)"}
    if parsed.protocol not in SUPPORTED_SCHEMES:
        return {"ok": False, "error": f'Unsupported scheme "{parsed.protocol}". Use http:// or https://'}
    if not parsed.hostname:
        return {"ok": False, "error": "URL is missing a host"}
    return {"ok": True}


def build_no_proxy_value(bypass_local: bool) -> Optional[str]:
    """Value for NO_PROXY, or None when it should not be set."""
    return DEFAULT_BYPASS_LIST if bypass_local else None


def is_proxy_active(cfg: Dict[str, Any]) -> bool:
    if not cfg.get("enabled"):
        return False
    url = cfg.get("url") or ""
    if js_trim(url) == "":
        return False
    return validate_proxy_url(url)["ok"] is True
