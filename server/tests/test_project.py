import pytest
from fastapi.testclient import TestClient

from llm_wiki_server.app import create_app

c = TestClient(create_app())


def test_health_and_unknown():
    assert c.get("/health").json()["ok"] is True
    assert c.post("/rpc/nope", json={}).status_code == 404


def test_create_open_project(tmp_path):
    r = c.post("/rpc/create_project", json={"name": "w", "path": str(tmp_path)})
    assert r.status_code == 200 and r.json()["name"] == "w"
    assert c.post("/rpc/create_project", json={"name": "w", "path": str(tmp_path)}).status_code == 400
    root = r.json()["path"]
    assert (tmp_path / "w/wiki/index.md").exists()
    assert c.post("/rpc/open_project", json={"path": root}).json()["name"] == "w"
    # camelCase args are accepted
    assert c.post("/rpc/open_path_in_project", json={"projectPath": root, "targetPath": "wiki"}).status_code == 200
    assert c.post("/rpc/open_path_in_project", json={"projectPath": root, "targetPath": "/etc"}).status_code == 400
    assert c.post("/rpc/open_project", json={"path": str(tmp_path)}).status_code == 400


def test_register_project_and_rest_listing(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_WIKI_API_TOKEN", "t")
    hdr = {"Authorization": "Bearer t"}
    root = c.post("/rpc/create_project", json={"name": "reg", "path": str(tmp_path)}, headers=hdr).json()["path"]
    assert c.post("/rpc/register_project", json={"path": root}, headers=hdr).json()["name"] == "reg"
    projects = c.get("/api/v1/projects", headers=hdr).json()
    assert any(p.get("name") == "reg" for p in projects["projects"]), projects


def test_rpc_requires_token_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_WIKI_API_TOKEN", "secret")
    assert c.post("/rpc/file_exists", json={"path": "/"}).status_code == 401
    ok = c.post("/rpc/file_exists", json={"path": "/"}, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    assert c.get("/health").status_code == 200  # health stays open


def test_rpc_remote_without_token_forbidden(monkeypatch):
    monkeypatch.delenv("LLM_WIKI_API_TOKEN", raising=False)
    remote = TestClient(create_app(), client=("10.1.2.3", 5000))
    assert remote.post("/rpc/file_exists", json={"path": "/"}).status_code == 403


def test_metadata_dir_exists_so_history_works(tmp_path):
    """Regression: the desktop frontend created .llm-wiki; headless must do it, or history calls hit ENOENT."""
    root = c.post("/rpc/create_project", json={"name": "h", "path": str(tmp_path)}).json()["path"]
    assert (tmp_path / "h" / ".llm-wiki").is_dir()
    f = root + "/wiki/concepts/x.md"
    assert c.post("/rpc/set_file_history_settings", json={"projectPath": root, "settings": {"enabled": True, "maxVersionsPerFile": 5}}).status_code == 200
    c.post("/rpc/write_file", json={"path": f, "contents": "a"})
    c.post("/rpc/write_file", json={"path": f, "contents": "b"})
    hist = c.post("/rpc/list_file_history", json={"projectPath": root, "filePath": f}).json()
    assert len(hist) >= 1
    # opening an older project without the folder also creates it
    import shutil
    shutil.rmtree(tmp_path / "h" / ".llm-wiki")
    c.post("/rpc/open_project", json={"path": root})
    assert (tmp_path / "h" / ".llm-wiki").is_dir()
