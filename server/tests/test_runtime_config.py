from __future__ import annotations

import asyncio
import json
import os
import urllib.request

import pytest

from llm_wiki_server import clip_server, cors, events, proxy, server_bind
from llm_wiki_server.commands import runtime_config as rc
from llm_wiki_server.errors import CommandError


@pytest.fixture(autouse=True)
def clean_proxy_env(monkeypatch):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        monkeypatch.delenv(k, raising=False)
    yield
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        os.environ.pop(k, None)


def cfg(**kw):
    c = proxy.default_config()
    c.update(kw)
    return c


def test_proxy_enable_disable():
    s = rc.set_proxy_env(cfg(enabled=True, url="http://user:pw@127.0.0.1:7890", acceptInvalidCerts=True))
    assert os.environ["HTTP_PROXY"].endswith("127.0.0.1:7890") and "pw" not in s and "***" in s
    assert "localhost" in os.environ["NO_PROXY"] and proxy.accept_invalid_certs()
    assert rc.set_proxy_env(cfg(enabled=False)) == "disabled"
    assert "HTTP_PROXY" not in os.environ and not proxy.accept_invalid_certs()


@pytest.mark.parametrize("url", ["socks5://x:1", "http://", "   "])
def test_proxy_rejects(url):
    os.environ["HTTP_PROXY"] = "stale"
    s = rc.set_proxy_env(cfg(enabled=True, url=url))
    assert s.startswith("disabled") and "HTTP_PROXY" not in os.environ


def test_proxy_bypass_off_and_defaults():
    os.environ["NO_PROXY"] = "stale"
    rc.set_proxy_env(cfg(enabled=True, url="https://proxy.corp:443", bypassLocal=False))
    assert "NO_PROXY" not in os.environ
    assert proxy.parse_proxy_config({})["bypassLocal"] is True
    assert proxy.redact_url("http://user@h:1/a@b") == "http://***@h:1/a@b"
    assert proxy.redact_url("garbage") == "garbage"


def test_close_behavior():
    assert rc.set_close_behavior("exit") == "exit"
    with pytest.raises(CommandError, match="Invalid close behavior: nope"):
        rc.set_close_behavior("nope")


def test_cors_and_bind():
    assert cors.is_allowed_browser_origin("chrome-extension://abc")
    assert not cors.is_allowed_browser_origin("http://localhost.evil.com")
    assert not cors.is_allowed_browser_origin("https://localhost")
    h = cors.local_cors_headers("http://127.0.0.1:5500", "Content-Type")
    assert h["Access-Control-Allow-Origin"] == "http://127.0.0.1:5500" and h["Vary"] == "Origin"
    assert "Access-Control-Allow-Origin" not in cors.local_cors_headers("https://evil.com", "x")
    assert server_bind.bind_addr("::1", 1) == "[::1]:1" and server_bind.bind_addr("[::]", 1) == "[::]:1"
    assert server_bind.sanitize_bind_host("127.0.0.1;rm") is None
    assert server_bind.sanitize_bind_host("  [::]  ") == "[::]"


def test_mcp_entry_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_WIKI_MCP_SERVER_ENTRY", str(tmp_path / "x.js"))
    try:
        p = rc.mcp_server_entry_path()
        assert p.endswith("index.js")
    except CommandError as e:
        assert "MCP server entry was not found" in e.message
    f = tmp_path / "x.js"
    f.write_text("//")
    assert rc.mcp_server_entry_path() == str(f.resolve())


def test_api_status_without_rest_module():
    assert isinstance(rc.api_server_status(), str)
    assert rc.api_server_reload_config() == "ok"
    assert rc.clip_server_status() in ("starting", "running", "port_conflict", "error")


def test_event_bus_and_sse():
    async def go():
        sub = events.subscribe(["a:*", "exact"])
        events.publish("a:1", {"x": 1})
        events.publish("zzz", 1)
        events.publish("exact", "line")
        got = [await asyncio.wait_for(sub.queue.get(), 1) for _ in range(2)]
        assert got == [("a:1", {"x": 1}), ("exact", "line")]
        events.unsubscribe(sub)

        class Req:
            async def is_disconnected(self):
                return False

        resp = await rc.sse_events(Req(), "t1")
        it = resp.body_iterator
        assert await it.__anext__() == ": connected\n\n"
        events.publish("t1", {"k": "v"})
        events.publish("other", 1)
        chunk = await asyncio.wait_for(it.__anext__(), 2)
        assert chunk == 'event: t1\ndata: {"k": "v"}\n\n'
        await it.aclose()
        assert events.subscriber_count() == 0

    asyncio.run(go())


def test_event_bus_from_thread():
    import threading

    async def go():
        sub = events.subscribe(["thr"])
        threading.Thread(target=lambda: events.publish("thr", 5)).start()
        assert await asyncio.wait_for(sub.queue.get(), 2) == ("thr", 5)
        events.unsubscribe(sub)

    asyncio.run(go())


@pytest.fixture
def clip(tmp_path):
    import time
    clip_server.start_clip_server(port=0, host="127.0.0.1")
    for _ in range(100):
        if clip_server.bound_port():
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{clip_server.bound_port()}"
    clip_server.stop_clip_server()


def call(base, method, path, body=None, headers=None):
    req = urllib.request.Request(base + path, method=method, headers=headers or {},
                                 data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"null"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null"), dict(e.headers)


def test_clip_server_flow(clip, tmp_path):
    assert clip_server.get_daemon_status() == "running"
    assert call(clip, "GET", "/status")[1] == {"ok": True, "version": "0.1.0"}
    assert call(clip, "GET", "/status?x=1")[0] == 404
    s, _, h = call(clip, "OPTIONS", "/clip", headers={"Origin": "chrome-extension://abc"})
    assert s == 204 and h["Access-Control-Allow-Origin"] == "chrome-extension://abc"
    assert h["Access-Control-Max-Age"] == "600"
    assert call(clip, "POST", "/project", {"nope": 1})[0] == 400
    assert call(clip, "POST", "/clip", {"title": "T", "content": "c"})[0] == 500  # no project yet
    proj = str(tmp_path / "p")
    assert call(clip, "POST", "/project", {"path": proj})[0] == 200
    assert call(clip, "GET", "/project")[1] == {"ok": True, "path": proj}
    call(clip, "POST", "/projects", {"projects": [{"name": "P", "path": proj}, {"name": "x", "path": ""}]})
    assert call(clip, "GET", "/projects")[1]["projects"] == [{"name": "P", "path": proj, "current": True}]
    s, r, _ = call(clip, "POST", "/clip", {"title": 'He said "Hi"! Ünï', "url": "http://u", "content": "body"})
    assert s == 200 and r["path"].startswith("raw/sources/he-said-hi-ünï-")
    s, r2, _ = call(clip, "POST", "/clip", {"title": 'He said "Hi"! Ünï', "url": "http://u", "content": "body"})
    assert r2["path"].endswith("-2.md")
    text = (tmp_path / "p" / r["path"]).read_text()
    assert text.startswith('---\ntype: clip\ntitle: "He said \\"Hi\\"! Ünï"\nurl: "http://u"\n')
    assert text.endswith("# He said \"Hi\"! Ünï\n\nSource: http://u\n\nbody\n")
    pend = call(clip, "GET", "/clips/pending")[1]["clips"]
    assert len(pend) == 2 and pend[0]["projectPath"] == proj
    assert call(clip, "GET", "/clips/pending")[1]["clips"] == []
    assert call(clip, "POST", "/clip", {"projectPath": proj, "content": ""})[1]["error"] == "content is required"


def test_clip_loopback_and_token(monkeypatch):
    assert clip_server.address_is_loopback("127.0.0.1") and clip_server.address_is_loopback("::1")
    assert clip_server.address_is_loopback("::ffff:127.0.0.1")
    assert not clip_server.address_is_loopback("192.168.1.20") and not clip_server.address_is_loopback(None)
    monkeypatch.setenv("LLM_WIKI_API_TOKEN", "sekret")
    assert clip_server.is_token_authorized({"x-llm-wiki-token": "sekret"})
    assert clip_server.is_token_authorized({"authorization": "Bearer sekret"})
    assert not clip_server.is_token_authorized({"authorization": "Bearer nope"})
    assert clip_server.next_restart_count(clip_server.MAX_RESTART_RETRIES) is None
