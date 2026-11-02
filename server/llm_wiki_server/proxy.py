"""Port of proxy.rs: global outbound HTTP proxy via HTTP_PROXY/HTTPS_PROXY/NO_PROXY env vars."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

DEFAULT_BYPASS_LIST = "localhost,127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.local"
_accept_invalid_certs = False


def default_config() -> Dict[str, Any]:
    return {"enabled": False, "url": "", "bypassLocal": True, "acceptInvalidCerts": False}


def parse_proxy_config(raw: Any) -> Optional[Dict[str, Any]]:
    """serde-like parse with per-field defaults; None when not an object / wrong types."""
    if not isinstance(raw, dict):
        return None
    cfg = default_config()
    for key, typ in (("enabled", bool), ("url", str), ("bypassLocal", bool), ("acceptInvalidCerts", bool)):
        if key in raw and raw[key] is not None:
            if not isinstance(raw[key], typ):
                return None
            cfg[key] = raw[key]
    return cfg


def read_proxy_config_from_store(store_path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(store_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "proxyConfig" not in data:
        return None
    return parse_proxy_config(data["proxyConfig"])


def accept_invalid_certs() -> bool:
    """TLS policy for outbound clients (port of `configure_http_client`)."""
    return _accept_invalid_certs


def redact_url(url: str) -> str:
    i = url.find("://")
    if i == -1:
        return url
    scheme_end = i + 3
    after = url[scheme_end:]
    slash = after.find("/")
    path_start = len(after) if slash == -1 else slash
    at = after[:path_start].find("@")
    if at == -1:
        return url
    return url[:scheme_end] + "***" + after[at:]


def clear_proxy_env() -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        os.environ.pop(k, None)


def _supported(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.hostname)
    except ValueError:
        return False


def apply_proxy_env(config: Dict[str, Any]) -> str:
    global _accept_invalid_certs
    enabled = bool(config.get("enabled", False))
    url = str(config.get("url", "")).strip()
    bypass_local = bool(config.get("bypassLocal", True))
    accept = bool(config.get("acceptInvalidCerts", False))
    supported = _supported(url)
    active = enabled and supported
    _accept_invalid_certs = active and accept
    if not active:
        clear_proxy_env()
        if not enabled:
            return "disabled"
        if not url:
            return "disabled (empty url)"
        if not supported:
            return f"disabled (invalid or unsupported proxy url: {redact_url(url)})"
        return "disabled"
    os.environ["HTTP_PROXY"] = url
    os.environ["HTTPS_PROXY"] = url
    if bypass_local:
        os.environ["NO_PROXY"] = DEFAULT_BYPASS_LIST
    else:
        os.environ.pop("NO_PROXY", None)
    return (f"enabled ({redact_url(url)}, bypass_local={'true' if bypass_local else 'false'}, "
            f"accept_invalid_certs={'true' if accept else 'false'})")
