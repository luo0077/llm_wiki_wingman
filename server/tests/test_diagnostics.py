from fastapi.testclient import TestClient

from llm_wiki_server.app import create_app

c = TestClient(create_app())


def test_info_has_no_secrets_and_reports_libs(monkeypatch):
    c.post("/rpc/app_state_set", json={"key": "llmConfig", "value": {"provider": "custom", "apiKey": "SECRET-KEY", "model": "m", "customEndpoint": "http://gw.example:8080/v1/chat/completions"}})
    r = c.post("/rpc/diagnostics_info", json={})
    assert r.status_code == 200
    body = r.json()
    assert "SECRET-KEY" not in r.text
    assert body["llmConfig"] == {"provider": "custom", "model": "m", "endpointHost": "gw.example:8080", "hasApiKey": True}
    assert body["libs"]["fastapi"]["installed"] and body["vectorBackend"] in ("sqlite", "lancedb")


def test_llm_ping_not_configured_and_unreachable():
    c.post("/rpc/app_state_set", json={"key": "llmConfig", "value": None})
    assert c.post("/rpc/diagnostics_llm_ping", json={}).json()["ok"] is False
    c.post("/rpc/app_state_set", json={"key": "llmConfig", "value": {"provider": "custom", "apiKey": "x", "model": "m", "customEndpoint": "http://127.0.0.1:1/v1/chat/completions"}})
    r = c.post("/rpc/diagnostics_llm_ping", json={"timeoutSecs": 10}).json()
    assert r["ok"] is False and r["error"]


def test_unregister_project(tmp_path):
    root = c.post("/rpc/create_project", json={"name": "u", "path": str(tmp_path)}).json()["path"]
    c.post("/rpc/register_project", json={"path": root})
    c.post("/rpc/unregister_project", json={"path": root})
    assert all(p["path"] != root for p in c.post("/rpc/app_state_get", json={"key": "recentProjects"}).json() or [])
    assert (tmp_path / "u" / "schema.md").exists()
