"""Port of src/lib/proxy-config.test.ts and the tauri-fetch proxy/TLS contract (http_fetch)."""
from __future__ import annotations

import pytest

from llm_wiki_server import proxy as server_proxy
from llm_wiki_server.ingest import http_fetch
from llm_wiki_server.ingest.proxy_config import (
    DEFAULT_BYPASS_LIST,
    build_no_proxy_value,
    is_proxy_active,
    validate_proxy_url,
)


def test_validate_accepts_http_https_and_auth():
    assert validate_proxy_url("http://127.0.0.1:7890") == {"ok": True}
    assert validate_proxy_url("http://proxy.corp.local:8080") == {"ok": True}
    assert validate_proxy_url("https://proxy.corp:443") == {"ok": True}
    assert validate_proxy_url("http://user:pass@127.0.0.1:7890") == {"ok": True}


def test_validate_rejects_missing_scheme():
    assert validate_proxy_url("127.0.0.1:7890")["ok"] is False
    assert validate_proxy_url("proxy.example.com:8080")["ok"] is False


def test_validate_rejects_unsupported_schemes():
    r = validate_proxy_url("socks5://127.0.0.1:1080")
    assert r["ok"] is False and 'Unsupported scheme "socks5:"' in r["error"]
    assert validate_proxy_url("ftp://x")["ok"] is False


def test_validate_rejects_empty_and_malformed():
    assert validate_proxy_url("") == {"ok": False, "error": "URL is empty"}
    assert validate_proxy_url("   ")["ok"] is False
    for bad in ("http://", "not-a-url", "http:"):
        assert validate_proxy_url(bad)["ok"] is False
    assert validate_proxy_url("http://:7890")["ok"] is False


def test_build_no_proxy_value():
    assert build_no_proxy_value(True) == DEFAULT_BYPASS_LIST
    assert build_no_proxy_value(False) is None
    for frag in ("localhost", "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "*.local"):
        assert frag in DEFAULT_BYPASS_LIST
    assert DEFAULT_BYPASS_LIST == server_proxy.DEFAULT_BYPASS_LIST


def test_is_proxy_active():
    assert is_proxy_active({"enabled": False, "url": "http://x:1", "bypassLocal": True}) is False
    assert is_proxy_active({"enabled": True, "url": "", "bypassLocal": True}) is False
    assert is_proxy_active({"enabled": True, "url": "   ", "bypassLocal": True}) is False
    assert is_proxy_active({"enabled": True, "url": "not-a-url", "bypassLocal": True}) is False
    assert is_proxy_active({"enabled": True, "url": "ftp://x:1", "bypassLocal": True}) is False
    assert is_proxy_active({"enabled": True, "url": "http://127.0.0.1:7890", "bypassLocal": True}) is True
    assert is_proxy_active({"enabled": True, "url": "https://proxy.corp:443", "bypassLocal": False}) is True


@pytest.fixture
def clean_proxy():
    yield
    server_proxy.apply_proxy_env({"enabled": False})


def test_http_fetch_honors_server_proxy_state(clean_proxy, monkeypatch):
    """Clients are built per call, so proxy env + the TLS policy of `llm_wiki_server.proxy` apply at once."""
    kw = http_fetch.client_kwargs()
    assert kw["trust_env"] is True and kw["verify"] is True
    server_proxy.apply_proxy_env({"enabled": True, "url": "http://proxy.corp:8080", "bypassLocal": True,
                                  "acceptInvalidCerts": True})
    kw = http_fetch.client_kwargs()
    assert kw["verify"] is False
    import os
    assert os.environ["HTTPS_PROXY"] == "http://proxy.corp:8080"
    server_proxy.apply_proxy_env({"enabled": False, "url": "http://proxy.corp:8080", "acceptInvalidCerts": True})
    assert http_fetch.client_kwargs()["verify"] is True


def test_transport_override_context_manager():
    import httpx
    t = httpx.MockTransport(lambda r: httpx.Response(200))
    assert http_fetch.get_default_transport() is None
    with http_fetch.use_transport(t):
        assert http_fetch.client_kwargs()["transport"] is t
    assert http_fetch.get_default_transport() is None
