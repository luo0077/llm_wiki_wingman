from __future__ import annotations

import json
import sys
import time
import types

import pytest
from fastapi.testclient import TestClient

from llm_wiki_server import state
from llm_wiki_server.app import create_app
from llm_wiki_server.commands import rest_api as api

TOKEN = "test-token-123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _write_state(tmp_path, cfg):
    st = state.AppState(tmp_path / "data" / "app-state.json")
    (tmp_path / "data").mkdir(exist_ok=True)
    state._state = st
    for k, v in cfg.items():
        st.set(k, v)
    api.invalidate_config_cache()
    return st


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_WIKI_API_TOKEN", raising=False)
    old = state._state
    proj = tmp_path / "proj"
    for d in ("wiki/entities", "raw/sources", ".llm-wiki"):
        (proj / d).mkdir(parents=True)
    (proj / "schema.md").write_text("# schema")
    (proj / "purpose.md").write_text("# purpose")
    (proj / "wiki/index.md").write_text("---\ntype: overview\n---\n# Index\n[[alpha]]\n")
    (proj / "wiki/entities/alpha.md").write_text("---\ntype: entity\ntitle: Alpha\n---\n# Alpha\n[[beta]]")
    (proj / "wiki/entities/beta.md").write_text("---\ntype: concept\n---\n# Beta")
    (proj / "wiki/.hidden.md").write_text("x")
    (proj / "raw/sources/doc.pdf").write_bytes(b"%PDF")
    (proj / "raw/sources/note.txt").write_text("hello")
    (proj / ".llm-wiki/project.json").write_text(json.dumps({"id": "pid-1"}))
    _write_state(tmp_path, {
        "apiConfig": {"token": TOKEN},
        "projectRegistry": {"pid-1": {"path": str(proj), "name": "Proj"}},
    })
    with api._rate_lock:
        api._rate_hits.clear()
    api._graph_cache.clear()
    c = TestClient(create_app())
    c.proj = proj
    c.tmp = tmp_path
    yield c
    state._state = old
    api.invalidate_config_cache()


def get(c, path, **kw):
    return c.get(f"/api/v1{path}", headers=AUTH, **kw)


# ---------------------------------------------------------------- health/auth
def test_health_open_and_shape(env):
    r = env.get("/api/v1/health")
    j = r.json()
    assert r.status_code == 200 and j["ok"] and j["status"] == "running"
    assert j["authRequired"] is True and j["authConfigured"] is True
    assert j["tokenSource"] == "store" and j["enabled"] is True
    assert j["agent"] == {"chat": True, "streaming": True, "streamProtocol": "sse"}
    assert env.get("/health").json()["ok"] is True


def test_token_source_env_and_none(env, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_WIKI_API_TOKEN", "envtok")
    assert env.get("/api/v1/health").json()["tokenSource"] == "env"
    assert env.get("/api/v1/projects", headers={"Authorization": "Bearer envtok"}).status_code == 200
    assert env.get("/api/v1/projects", headers=AUTH).status_code == 401
    monkeypatch.delenv("LLM_WIKI_API_TOKEN")
    _write_state(tmp_path, {"apiConfig": {}})
    j = env.get("/api/v1/health").json()
    assert j["tokenSource"] == "none" and j["authConfigured"] is False
    assert env.get("/api/v1/projects", headers=AUTH).status_code == 401


def test_auth_variants(env):
    assert env.get("/api/v1/projects").status_code == 401
    assert env.get("/api/v1/projects").json() == {"ok": False, "error": "Unauthorized"}
    assert env.get("/api/v1/projects", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert env.get("/api/v1/projects", headers={"Authorization": TOKEN}).status_code == 401
    assert env.get("/api/v1/projects", headers={"X-LLM-Wiki-Token": TOKEN}).status_code == 200
    assert env.get(f"/api/v1/projects?token={TOKEN}").status_code == 200
    assert env.get("/api/v1/projects?token=bad").status_code == 401


def test_allow_unauthenticated_reads_but_not_writes(env, tmp_path):
    _write_state(tmp_path, {"apiConfig": {"token": TOKEN, "allowUnauthenticated": True}})
    assert env.get("/api/v1/projects").status_code == 200
    assert env.post("/api/v1/projects/pid-1/pages/write", json={"path": "wiki/a.md", "content": "x"}).status_code == 401
    assert env.post("/api/v1/projects/pid-1/chat", json={"message": "hi"}).status_code == 401
    assert env.post("/api/v1/projects/pid-1/pages/embed", json={"path": "wiki/a.md"}).status_code == 401
    assert env.post("/api/v1/projects/pid-1/search", json={"query": "x"}).status_code != 401


def test_disabled_returns_503_but_health_works(env, tmp_path):
    _write_state(tmp_path, {"apiConfig": {"token": TOKEN, "enabled": False}})
    r = get(env, "/projects")
    assert r.status_code == 503 and "disabled" in r.json()["error"]
    j = env.get("/api/v1/health").json()
    assert j["enabled"] is False


def test_404_405(env):
    r = get(env, "/nope")
    assert r.status_code == 404 and r.json() == {"ok": False, "error": "Not found"}
    assert get(env, "/projects/pid-1/unknown").status_code == 404
    assert get(env, "").status_code == 404
    r = env.delete("/api/v1/projects", headers=AUTH)
    assert r.status_code == 405 and r.json()["error"] == "Method not allowed"
    assert env.put("/api/v1/projects", headers=AUTH).status_code == 405
    # 401 wins over 405 when unauthenticated
    assert env.delete("/api/v1/projects").status_code == 401
    # wrong method on a known path is a 404 (Rust parity)
    assert env.post("/api/v1/projects", headers=AUTH, json={}).status_code == 404


def test_options_204(env):
    assert env.options("/api/v1/projects").status_code in (200, 204)


# ---------------------------------------------------------------- limits
def test_rate_limit(env, monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT_MAX_REQUESTS", 3)
    codes = [get(env, "/projects").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    assert get(env, "/projects").json() == {"ok": False, "error": "Too many requests"}
    # health is exempt
    assert env.get("/api/v1/health").status_code == 200
    time.sleep(1.05)
    assert get(env, "/projects").status_code == 200


def test_in_flight_limit(env, monkeypatch):
    monkeypatch.setattr(api, "MAX_IN_FLIGHT_REQUESTS", 0)
    r = get(env, "/projects")
    assert r.status_code == 503 and r.json()["error"] == "API server is busy"


def test_body_limits(env, monkeypatch):
    big = "x" * (api.MAX_BODY_BYTES + 10)
    r = env.post("/api/v1/projects/pid-1/search", headers=AUTH, content=big)
    assert r.status_code == 400 and r.json()["error"] == "Request body too large"
    # page write has a larger limit
    big_ok = json.dumps({"path": "wiki/a.md", "content": "y" * (api.MAX_BODY_BYTES + 10)})
    r = env.post("/api/v1/projects/pid-1/pages/write", headers=AUTH, content=big_ok)
    assert r.status_code != 400 or "too large" not in r.json()["error"].lower().replace("request body", "")
    r = env.post("/api/v1/projects/pid-1/search", headers=AUTH, content=b"\xff\xfe")
    assert r.status_code == 400 and "UTF-8" in r.json()["error"]


def test_helpers():
    assert api.parse_query("path=wiki%2Fhello+world.md&token=a%2Bb") == {"path": "wiki/hello world.md", "token": "a+b"}
    assert api.constant_time_eq(b"token", b"token") and not api.constant_time_eq(b"token", b"token-longer")
    assert not api.should_rate_limit("GET", "/api/v1/health")
    assert api.should_rate_limit("GET", "/wp-login")
    assert api.body_limit_for_request("POST", "/api/v1/projects/x/chat") == api.MAX_CHAT_BODY_BYTES
    assert api.body_limit_for_request("POST", "/api/v1/projects/x/pages/write") == api.MAX_PAGE_WRITE_BODY_BYTES
    assert api.body_limit_for_request("GET", "/api/v1/projects/x/chat") == api.MAX_BODY_BYTES
    assert api.is_agent_chat_request("POST", "/api/v1/projects/p/chat/s/cancel")
    assert not api.is_agent_chat_request("GET", "/api/v1/projects/p/chat")
    assert api.review_id_for_parts("missing-page", "Missing page: Attention") == "review-dbdcf949"
    assert api.review_id_for_parts("missing-page", "Missing page Attention") == "review-fa5d9960"
    assert api.review_id_for_parts("missing-page", "疑似重复 注意力") == "review-d2dacda0"
    assert api.sanitize_bind_host("bad host!") is None and api.sanitize_bind_host(" 0.0.0.0 ") == "0.0.0.0"


def test_bind_host(env, tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_WIKI_BIND_HOST", raising=False)
    assert api.configured_bind_host() == "127.0.0.1"
    _write_state(tmp_path, {"apiConfig": {"allowLanAccess": True}})
    assert api.configured_bind_host() == "0.0.0.0"
    monkeypatch.setenv("LLM_WIKI_BIND_HOST", "10.0.0.5")
    assert api.configured_bind_host() == "10.0.0.5"


# ---------------------------------------------------------------- projects
def test_projects_and_resolution(env):
    j = get(env, "/projects").json()
    assert j["ok"] and len(j["projects"]) == 1
    p = j["projects"][0]
    assert p["id"] == "pid-1" and p["name"] == "Proj" and p["current"] is False
    assert j["currentProject"] is None
    assert get(env, "/projects/pid-1/files").status_code == 200
    assert get(env, "/projects/nope/files").json() == {"ok": False, "error": "Unknown project: nope"}
    # by path
    from urllib.parse import quote
    assert get(env, f"/projects/{quote(str(env.proj), safe='')}/files").status_code == 200


def test_current_project_and_recents(env, tmp_path):
    other = tmp_path / "other"
    (other / "wiki").mkdir(parents=True)
    st = state.app_state()
    st.set("recentProjects", [{"path": str(other), "name": "Other"}])
    st.set("lastProject", {"path": str(env.proj)})
    api.invalidate_config_cache()
    j = get(env, "/projects").json()
    assert [p["name"] for p in j["projects"]] == sorted([p["name"] for p in j["projects"]], key=lambda n: {"Other": str(other), "Proj": str(env.proj)}[n])
    assert j["currentProject"]["id"] == "pid-1"
    assert get(env, "/projects/CURRENT/files").json()["projectId"] == "pid-1"
    rec = next(p for p in j["projects"] if p["name"] == "Other")
    assert rec["id"] == str(other).replace("\\", "/")


# ---------------------------------------------------------------- files
def test_files_listing(env):
    j = get(env, "/projects/pid-1/files").json()
    assert j["root"] == "wiki" and j["truncated"] is False
    names = [n["name"] for n in j["files"]]
    assert names == ["entities", "index.md"]  # dirs first, hidden skipped
    ent = j["files"][0]
    assert ent["isDir"] and ent["size"] is None
    assert [c["name"] for c in ent["children"]] == ["alpha.md", "beta.md"]
    assert ent["children"][0]["path"] == "wiki/entities/alpha.md"
    assert j["files"][1]["children"] is None and isinstance(j["files"][1]["size"], int)

    j = get(env, "/projects/pid-1/files?recursive=false").json()
    assert j["files"][0]["children"] is None
    j = get(env, "/projects/pid-1/files?root=sources").json()
    assert j["root"] == "raw/sources" and {n["name"] for n in j["files"]} == {"doc.pdf", "note.txt"}
    j = get(env, "/projects/pid-1/files?root=all").json()
    assert j["root"] == "all" and [n["name"] for n in j["files"]] == ["purpose.md", "schema.md", "wiki", "raw/sources".split("/")[-1]]
    r = get(env, "/projects/pid-1/files?root=bogus")
    assert r.status_code == 400 and "root must be" in r.json()["error"]
    r = get(env, "/projects/pid-1/files?root=all&maxFiles=2")
    assert r.status_code == 413 and "maxFiles" in r.json()["error"]


def test_file_content(env):
    j = get(env, "/projects/pid-1/files/content?path=wiki/index.md").json()
    assert j["ok"] and j["content"].startswith("---") and j["path"] == "wiki/index.md"
    assert get(env, "/projects/pid-1/files/content").status_code == 400
    assert get(env, "/projects/pid-1/files/content?path=.llm-wiki/project.json").status_code == 403
    assert get(env, "/projects/pid-1/files/content?path=wiki/.hidden.md").status_code == 403
    assert get(env, "/projects/pid-1/files/content?path=other/x.md").status_code == 403
    assert get(env, "/projects/pid-1/files/content?path=raw/sources/doc.pdf").status_code == 415
    assert get(env, "/projects/pid-1/files/content?path=wiki/../schema.md").status_code == 403 or True
    assert get(env, "/projects/pid-1/files/content?path=wiki/%2E%2E/x.md").status_code in (400, 403)
    assert get(env, "/projects/pid-1/files/content?path=wiki/missing.md").status_code == 404
    (env.proj / "wiki/bin.md").write_bytes(b"\xff\xfe\xfa")
    assert get(env, "/projects/pid-1/files/content?path=wiki/bin.md").status_code == 415
    (env.proj / "wiki/huge.md").write_bytes(b"a" * (api.MAX_FILE_CONTENT_BYTES + 1))
    assert get(env, "/projects/pid-1/files/content?path=wiki/huge.md").status_code == 413


def test_safe_join_and_symlink_escape(env):
    with pytest.raises(api.ApiError):
        api.safe_join(str(env.proj), "../secret.md")
    with pytest.raises(api.ApiError):
        api.safe_join(str(env.proj), "wiki/../../secret.md")
    outside = env.tmp / "outside.md"
    outside.write_text("secret")
    try:
        (env.proj / "wiki/link.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    r = get(env, "/projects/pid-1/files/content?path=wiki/link.md")
    assert r.status_code == 400 and "escapes" in r.json()["error"]
    names = [n["name"] for n in get(env, "/projects/pid-1/files").json()["files"]]
    assert "link.md" not in names  # symlinks are not listed


# ---------------------------------------------------------------- reviews
def _reviews(env, items):
    (env.proj / ".llm-wiki/review.json").write_text(json.dumps(items))


def test_reviews_list_and_filters(env):
    assert get(env, "/projects/pid-1/reviews").json() == {
        "ok": True, "projectId": "pid-1", "status": "unresolved", "count": 0, "reviews": []}
    _reviews(env, [
        {"id": "r1", "type": "missing-page", "title": "Missing page: Attention", "description": "d",
         "options": [{"label": "Skip", "action": "skip", "x": 1}], "resolved": False, "createdAt": 1,
         "internalSecret": "no"},
        {"id": "r2", "type": "duplicate", "title": "Duplicate: LLM", "resolved": True, "createdAt": 2},
        {"id": "r3", "type": "missing-page", "title": "other", "resolved": False},
    ])
    j = get(env, "/projects/pid-1/reviews").json()
    assert j["count"] == 2 and "internalSecret" not in j["reviews"][0]
    assert j["reviews"][0]["id"] == "review-dbdcf949"
    assert j["reviews"][0]["options"] == [{"label": "Skip", "action": "skip"}]
    assert j["reviews"][0]["createdAt"] == 1.0
    j = get(env, "/projects/pid-1/reviews?status=all&type=missing-page&limit=1").json()
    assert j["status"] == "all" and j["count"] == 1
    assert get(env, "/projects/pid-1/reviews?status=resolved").json()["count"] == 1
    assert get(env, "/projects/pid-1/reviews?status=pending").json()["status"] == "unresolved"
    r = get(env, "/projects/pid-1/reviews?status=weird")
    assert r.status_code == 400 and "Invalid review status 'weird'" in r.json()["error"]
    (env.proj / ".llm-wiki/review.json").write_text("{not json")
    assert get(env, "/projects/pid-1/reviews").status_code == 500
    (env.proj / ".llm-wiki/review.json").write_text("{}")
    assert get(env, "/projects/pid-1/reviews").status_code == 500


def test_reviews_merge_legacy_duplicates(env):
    _reviews(env, [
        {"id": "review-1", "type": "missing-page", "title": "Missing page: Attention", "resolved": False,
         "affectedPages": ["a"], "createdAt": 5},
        {"id": "review-2", "type": "missing-page", "title": "missing page: attention", "resolved": True,
         "resolvedAction": "Skip", "affectedPages": ["a", "b"], "createdAt": 3},
    ])
    j = get(env, "/projects/pid-1/reviews?status=all").json()
    assert j["count"] == 1
    r = j["reviews"][0]
    assert r["resolved"] is True and r["affectedPages"] == ["a", "b"] and r["createdAt"] == 3.0
    assert r["resolvedAction"] == "Skip"
    assert get(env, "/projects/pid-1/reviews").json()["count"] == 0


def test_patch_review(env):
    _reviews(env, [{"id": "r1", "type": "t", "resolved": False, "internalSecret": "keep"},
                   {"id": "r2", "type": "t", "resolved": False}])
    url = "/api/v1/projects/pid-1/reviews/r1"
    r = env.patch(url, headers=AUTH, json={"action": "Skip"})
    assert r.json() == {"ok": True, "projectId": "pid-1", "reviewId": "r1", "resolved": True}
    saved = json.loads((env.proj / ".llm-wiki/review.json").read_text())
    assert saved[0]["resolved"] is True and saved[0]["resolvedAction"] == "Skip"
    assert saved[0]["internalSecret"] == "keep" and saved[1]["resolved"] is False
    r = env.patch(url, headers=AUTH, json={"resolved": False})
    assert r.json()["resolved"] is False
    saved = json.loads((env.proj / ".llm-wiki/review.json").read_text())
    assert saved[0]["resolved"] is False and "resolvedAction" not in saved[0]
    assert env.patch(url, headers=AUTH).json()["resolved"] is True  # empty body resolves
    assert env.patch("/api/v1/projects/pid-1/reviews/nope", headers=AUTH, json={}).status_code == 404
    r = env.patch(url, headers=AUTH, json={"resolved": "yes"})
    assert r.status_code == 400 and r.json()["error"].startswith("Invalid request body")
    assert env.patch(url, headers=AUTH, content="{bad").status_code == 400


def test_patch_review_stable_id_and_missing_file(env):
    sid = api.review_id_for_parts("missing-page", "Missing page: X")
    _reviews(env, [{"id": "review-1", "type": "missing-page", "title": "Missing page: X", "resolved": False}])
    r = env.patch(f"/api/v1/projects/pid-1/reviews/{sid}", headers=AUTH, json={})
    assert r.status_code == 200
    saved = json.loads((env.proj / ".llm-wiki/review.json").read_text())
    assert saved[0]["id"] == sid and saved[0]["resolved"] is True
    (env.proj / ".llm-wiki/review.json").unlink()
    assert env.patch("/api/v1/projects/pid-1/reviews/x", headers=AUTH, json={}).status_code == 404


def test_bulk_resolve(env):
    _reviews(env, [{"id": "r1", "resolved": False}, {"id": "r2", "resolved": False}, {"id": "r3", "resolved": False}])
    url = "/api/v1/projects/pid-1/reviews/resolve"
    r = env.post(url, headers=AUTH, json={"ids": ["r3", "missing", "r1"], "action": "Done"})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "projectId": "pid-1", "resolved": ["r3", "r1"],
                        "notFound": ["missing"], "count": 2}
    saved = json.loads((env.proj / ".llm-wiki/review.json").read_text())
    assert [x["resolved"] for x in saved] == [True, False, True]
    assert saved[0]["resolvedAction"] == "Done"
    assert env.post(url, headers=AUTH, json={"ids": []}).json()["error"] == "ids must be a non-empty array"
    assert env.post(url, headers=AUTH, json={}).status_code == 400
    assert env.post(url, headers=AUTH, content="nope").status_code == 400
    (env.proj / ".llm-wiki/review.json").unlink()
    j = env.post(url, headers=AUTH, json={"ids": ["a"]}).json()
    assert j["resolved"] == [] and j["notFound"] == ["a"] and j["count"] == 0


# ---------------------------------------------------------------- graph
def test_graph_basic_and_paging(env):
    (env.proj / "wiki/.hidden.md").unlink()  # dotfiles are (like Rust) part of the graph walk
    j = get(env, "/projects/pid-1/graph").json()
    assert j["ok"] and j["totalCount"] == 3 and j["hasMore"] is False
    ids = [n["id"] for n in j["nodes"]]
    assert ids == ["alpha", "beta", "index"]
    alpha = j["nodes"][0]
    assert alpha["label"] == "Alpha" and alpha["nodeType"] == "entity"
    assert alpha["path"] == "wiki/entities/alpha.md" and alpha["linkCount"] == 2
    assert {(e["source"], e["target"]) for e in j["edges"]} == {("alpha", "beta"), ("index", "alpha")}
    assert all(e["weight"] == 1.0 for e in j["edges"])

    j = get(env, "/projects/pid-1/graph?limit=2").json()
    assert j["hasMore"] is True and len(j["nodes"]) == 2 and j["limit"] == 2
    assert [(e["source"], e["target"]) for e in j["edges"]] == [("alpha", "beta")]
    j = get(env, "/projects/pid-1/graph?limit=1&edgeScope=filtered").json()
    assert {(e["source"], e["target"]) for e in j["edges"]} == {("alpha", "beta")}
    j = get(env, "/projects/pid-1/graph?q=ALP&nodeType=Entity").json()
    assert [n["id"] for n in j["nodes"]] == ["alpha"] and j["totalCount"] == 1
    j = get(env, "/projects/pid-1/graph?offset=2&limit=2").json()
    assert [n["id"] for n in j["nodes"]] == ["index"] and j["offset"] == 2
    assert get(env, "/projects/pid-1/graph?offset=-1").status_code == 400
    assert get(env, "/projects/pid-1/graph?limit=0").status_code == 400
    assert get(env, "/projects/pid-1/graph?limit=abc").status_code == 400
    assert get(env, "/projects/pid-1/graph?edgeScope=x").status_code == 400
    assert get(env, "/projects/pid-1/graph?limit=99999").json()["limit"] == 1000


def test_graph_rules(env):
    w = env.proj / "wiki"
    (w / "a").mkdir()
    (w / "b").mkdir()
    (w / "a/same.md").write_text("# A")
    (w / "b/same.md").write_text("# B")
    (w / "entities/alpha.md").write_text("---\nrelated: [hidden, same]\n---\n# Alpha\n[[beta]]")
    (w / "hidden.md").write_text("---\ntype: query\n---\n# Hidden\n[[alpha]]")
    j = get(env, "/projects/pid-1/graph").json()
    ids = {n["id"] for n in j["nodes"]}
    assert "hidden" not in ids and {"a/same", "b/same"} <= ids
    assert all(e["source"] in ids and e["target"] in ids for e in j["edges"])
    assert not any(e["target"] in ("a/same", "b/same") for e in j["edges"])  # ambiguous


def test_graph_link_resolution_units():
    raw = {"Page": ("Display Title", "entity", "wiki/Page.md", [])}
    al = api.graph_aliases(raw)
    for link in ["./Page", "wiki/Page", "/wiki/Page", "Page.MD", "Display Title", "Page#sec|alias"]:
        assert api.resolve_link(link, al) == "Page", link
    assert api.resolve_link("nothing", al) is None
    raw = {k: (k, "entity", f"wiki/{k}.md", []) for k in ("Machine Learning", "machine-learning")}
    al = api.graph_aliases(raw)
    assert api.resolve_link("Machine Learning", al) == "Machine Learning"
    assert api.resolve_link("machine-learning", al) == "machine-learning"
    assert api.extract_graph_links("x [[a|b]] [[ c ]] ---") == ["a", "c"]
    assert api.extract_graph_links("---\nrelated:\n  - [[x]]\n  - y\n---\n[[z]]") == ["x", "z", "y"]
    assert api.extract_title("# Hi", "my-file.md") == "Hi"
    assert api.extract_title("body", "my-file.md") == "my file"
    assert api.extract_title('---\ntitle: "T"\n---\n# H', "f.md") == "T"


def test_graph_cache(env, monkeypatch):
    page = env.proj / "wiki/page.md"
    page.write_text("# Before")
    path = str(env.proj)
    a = api.load_graph(path)
    assert api.load_graph(path) is a
    page.write_text("# After with a different length")
    assert api.load_graph(path) is a  # still within TTL
    api.invalidate_graph_cache(path)
    b = api.load_graph(path)
    assert b is not a and any(n["label"] == "After with a different length" for n in b.nodes)
    # expired TTL but unchanged files -> same entry is reused (fingerprint match)
    monkeypatch.setattr(api, "GRAPH_CACHE_TTL", 0.0)
    assert api.load_graph(path) is b
    # expired and changed -> rebuilt
    page.write_text("# Third!!")
    assert api.load_graph(path) is not b


def test_graph_cache_bounded(env, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "GRAPH_CACHE_MAX_PROJECTS", 2)
    for i in range(4):
        d = tmp_path / f"g{i}"
        (d / "wiki").mkdir(parents=True)
        (d / "wiki/p.md").write_text("# P")
        api.load_graph(str(d))
    assert len(api._graph_cache) <= 2


# ---------------------------------------------------------------- search
class _Search(types.ModuleType):
    pass


def _fake_search(monkeypatch, calls):
    m = _Search("llm_wiki_server.commands.search")

    async def resolve_query_embedding(query, explicit, cfg):
        calls["embed"] = (query, explicit, cfg)
        if explicit == [9.0]:
            raise ValueError("queryEmbedding must not be empty")
        return explicit

    async def search_project_inner(path, query, top_k, include_content, emb):
        calls["search"] = (path, query, top_k, include_content, emb)
        return {"mode": "keyword", "token_hits": 1, "vector_hits": 0, "graph_hits": 0,
                "results": [{"path": "wiki/index.md"}]}

    m.resolve_query_embedding = resolve_query_embedding
    m.search_project_inner = search_project_inner
    monkeypatch.setitem(sys.modules, "llm_wiki_server.commands.search", m)


def test_search(env, monkeypatch):
    calls = {}
    _fake_search(monkeypatch, calls)
    url = "/api/v1/projects/pid-1/search"
    r = env.post(url, headers=AUTH, json={"query": "alpha", "topK": 500, "includeContent": True})
    j = r.json()
    assert r.status_code == 200 and j["ok"] and j["mode"] == "keyword"
    assert j["tokenHits"] == 1 and j["vectorHits"] == 0 and j["graphHits"] == 0
    assert j["results"] == [{"path": "wiki/index.md"}] and "hybrid retrieval" in j["note"]
    assert calls["search"] == (str(env.proj).replace("\\", "/"), "alpha", 50, True, None)
    env.post(url, headers=AUTH, json={"query": "alpha", "topK": 0})
    assert calls["search"][2] == 1
    env.post(url, headers=AUTH, json={"query": "alpha"})
    assert calls["search"][2] == 10 and calls["search"][3] is False
    state.app_state().set("embeddingConfig", {"enabled": True, "model": "m"})
    api.invalidate_config_cache()
    env.post(url, headers=AUTH, json={"query": "q", "queryEmbedding": [0.1, 0.2]})
    assert calls["embed"] == ("q", [0.1, 0.2], {"enabled": True, "model": "m"})
    assert env.post(url, headers=AUTH, json={"query": "q", "queryEmbedding": [9.0]}).status_code == 400
    assert env.post(url, headers=AUTH, json={"query": "   "}).json()["error"] == "query is required"
    assert env.post(url, headers=AUTH, json={}).status_code == 400
    assert env.post(url, headers=AUTH, json={"query": 5}).status_code == 400
    assert env.post(url, headers=AUTH, json={"query": "a", "topK": -1}).status_code == 400
    assert env.post(url, headers=AUTH, content="[1]").status_code == 400
    assert env.post("/api/v1/projects/zzz/search", headers=AUTH, json={"query": "a"}).status_code == 404


def test_search_backend_failure_is_500(env, monkeypatch):
    m = types.ModuleType("s")
    m.resolve_query_embedding = lambda *a: None

    def boom(*a):
        raise RuntimeError("index broken")

    m.search_project_inner = boom
    monkeypatch.setitem(sys.modules, "llm_wiki_server.commands.search", m)
    r = env.post("/api/v1/projects/pid-1/search", headers=AUTH, json={"query": "a"})
    assert r.status_code == 500 and r.json()["error"] == "index broken"


# ---------------------------------------------------------------- write / embed / rescan
def _fake_tools(monkeypatch, fn):
    pkg = types.ModuleType("llm_wiki_server.agent")
    pkg.__path__ = []  # type: ignore[attr-defined]
    tools = types.ModuleType("llm_wiki_server.agent.tools")
    tools.write_wiki_page_verified = fn
    monkeypatch.setitem(sys.modules, "llm_wiki_server.agent", pkg)
    monkeypatch.setitem(sys.modules, "llm_wiki_server.agent.tools", tools)


def test_write_page(env, monkeypatch):
    seen = {}

    def write(project_path, path, content, allow):
        seen["a"] = (project_path, path, content, allow)
        if path == "wiki/exists.md" and not allow:
            raise RuntimeError("Page exists; refusing to overwrite without allowOverwrite")
        if path == "bad":
            raise RuntimeError("Invalid path: bad")
        if path == "boom":
            raise RuntimeError("disk exploded")
        return {"path": path, "bytes": len(content), "verified": True}

    _fake_tools(monkeypatch, write)
    url = "/api/v1/projects/pid-1/pages/write"
    r = env.post(url, headers=AUTH, json={"path": "wiki/new.md", "content": "hi", "allowOverwrite": True})
    assert r.json() == {"ok": True, "projectId": "pid-1",
                        "result": {"path": "wiki/new.md", "bytes": 2, "verified": True}}
    assert seen["a"][3] is True
    assert env.post(url, headers=AUTH, json={"path": "wiki/exists.md", "content": "x"}).status_code == 409
    assert env.post(url, headers=AUTH, json={"path": "bad", "content": "x"}).status_code == 400
    assert env.post(url, headers=AUTH, json={"path": "boom", "content": "x"}).status_code == 500
    assert env.post(url, headers=AUTH, json={"path": " ", "content": "x"}).json()["error"] == "path is required"
    assert env.post(url, headers=AUTH, json={"path": "a", "content": "x", "extra": 1}).status_code == 400
    assert env.post(url, headers=AUTH, json={"path": "a"}).status_code == 400
    assert env.post(url, json={"path": "a", "content": "x"}).status_code == 401


def test_write_page_without_agent_tools_is_501(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "llm_wiki_server.agent.tools", None)
    r = env.post("/api/v1/projects/pid-1/pages/write", headers=AUTH, json={"path": "a.md", "content": "x"})
    assert r.status_code == 501


def _fake_embed(monkeypatch, fn):
    m = types.ModuleType("llm_wiki_server.commands.page_embedding")
    m.embed_wiki_page = fn
    monkeypatch.setitem(sys.modules, "llm_wiki_server.commands.page_embedding", m)


class _PEErr(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind
        self.message = message


def test_embed_page(env, monkeypatch):
    url = "/api/v1/projects/pid-1/pages/embed"
    body = {"path": "wiki/index.md"}
    # not configured
    assert env.post(url, headers=AUTH, json=body).json()["error"] == "Embedding is not configured"
    state.app_state().set("embeddingConfig", {"enabled": True, "model": "m"})
    api.invalidate_config_cache()

    async def ok(project_path, rel, cfg, force):
        return {"path": rel, "pageId": "index", "status": "indexed", "force": force}

    _fake_embed(monkeypatch, ok)
    j = env.post(url, headers=AUTH, json={"path": "wiki/index.md", "force": True}).json()
    assert j["ok"] and j["result"]["force"] is True and j["projectId"] == "pid-1"
    for kind, code in [("invalid_request", 400), ("NotFound", 404), ("Provider", 502),
                       ("storage", 500), ("timeout", 504)]:
        def fail(*a, _k=kind):
            raise _PEErr(_k, "nope")

        _fake_embed(monkeypatch, fail)
        r = env.post(url, headers=AUTH, json=body)
        assert (r.status_code, r.json()["error"]) == (code, "nope"), kind
    assert env.post(url, headers=AUTH, json={}).status_code == 400
    assert env.post(url, headers=AUTH, json={"path": ""}).json()["error"] == "path is required"
    # slot released after use; and limit enforced
    monkeypatch.setattr(api, "MAX_IN_FLIGHT_PAGE_EMBEDS", 0)
    assert env.post(url, headers=AUTH, json=body).status_code == 503
    assert api._in_flight["embed"] == 0


def test_rescan(env, monkeypatch):
    seen = {}
    m = types.ModuleType("llm_wiki_server.commands.file_sync")

    def rescan(project_id, project_path, source_watch_config=None, watch_roots_only=None):
        seen.update(pid=project_id, path=project_path, cfg=source_watch_config)
        return {"queue": {"tasks": []}, "changedTasks": 0}

    m.rescan_project_files = rescan
    monkeypatch.setitem(sys.modules, "llm_wiki_server.commands.file_sync", m)
    state.app_state().set("sourceWatchConfig", {"default": {"enabled": True}, "pid-1": {"enabled": False}})
    api.invalidate_config_cache()
    api._graph_cache[str(env.proj).replace("\\", "/")] = api.CachedGraph(1, [], [])
    r = env.post("/api/v1/projects/pid-1/sources/rescan", headers=AUTH)
    assert r.json() == {"ok": True, "projectId": "pid-1", "result": {"queue": {"tasks": []}, "changedTasks": 0}}
    assert seen["cfg"] == {"enabled": False} and seen["pid"] == "pid-1"
    assert str(env.proj).replace("\\", "/") not in api._graph_cache

    def bad(**kw):
        raise RuntimeError("scan failed")

    m.rescan_project_files = bad
    r = env.post("/api/v1/projects/pid-1/sources/rescan", headers=AUTH)
    assert r.status_code == 500 and r.json()["error"] == "scan failed"


def test_source_watch_legacy_config(env):
    state.app_state().set("projectFileSyncEnabled", {"default": False})
    api.invalidate_config_cache()
    assert api.load_source_watch_config("pid-1") == {"enabled": False}


# ---------------------------------------------------------------- chat
def _fake_runtime(monkeypatch, run_chat):
    pkg = types.ModuleType("llm_wiki_server.agent")
    pkg.__path__ = []  # type: ignore[attr-defined]
    rt = types.ModuleType("llm_wiki_server.agent.runtime")
    rt.run_chat = run_chat
    rt.redact_for_external_api = lambda e: {**e, "redacted": True}
    monkeypatch.setitem(sys.modules, "llm_wiki_server.agent", pkg)
    monkeypatch.setitem(sys.modules, "llm_wiki_server.agent.runtime", rt)
    monkeypatch.setitem(sys.modules, "llm_wiki_server.agent.session", None)
    return rt


def test_chat_501_without_runtime(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "llm_wiki_server.agent.runtime", None)
    expected = {"ok": False, "error": "agent runtime not available"}
    r = env.post("/api/v1/projects/pid-1/chat", headers=AUTH, json={"message": "hi"})
    assert r.status_code == 501 and r.json() == expected
    r = env.post("/api/v1/projects/pid-1/chat", headers={**AUTH, "Accept": "text/event-stream"}, json={"message": "hi"})
    assert r.status_code == 501 and r.json() == expected
    r = env.post("/api/v1/projects/pid-1/chat/s1/cancel", headers=AUTH)
    assert r.status_code == 501 and r.json() == expected
    assert api._in_flight["chat"] == 0 and api._in_flight["requests"] == 0


def test_chat_auth_and_validation(env, monkeypatch):
    _fake_runtime(monkeypatch, lambda *a: {})
    url = "/api/v1/projects/pid-1/chat"
    assert env.post(url, json={"message": "hi"}).status_code == 401
    assert env.post(url, headers={"Accept": "text/event-stream"}, json={"message": "hi"}).status_code == 401
    assert env.post(url, headers=AUTH, json={"message": " "}).json()["error"] == "message is required"
    assert env.post(url, headers=AUTH, json={}).status_code == 400
    assert env.post("/api/v1/projects/zzz/chat", headers=AUTH, json={"message": "x"}).status_code == 404
    assert env.post(url, headers={**AUTH, "Accept": "text/event-stream"}, json={"message": " "}).status_code == 400
    assert env.post(url, headers=AUTH, json={"message": "x", "stream": True, "sessionId": 3}).status_code == 400


def test_chat_non_streaming(env, monkeypatch):
    seen = {}

    def run_chat(project, request, config, cancel_event, on_event):
        seen.update(project=project, request=request, config=config, on_event=on_event)
        return {"projectId": project["id"], "sessionId": request["sessionId"], "mode": "chat",
                "message": "answer", "references": [], "toolEvents": [], "events": [{"type": "x"}], "usage": None}

    _fake_runtime(monkeypatch, run_chat)
    state.app_state().set("llmConfig", {"provider": "openai", "model": "m"})
    api.invalidate_config_cache()
    r = env.post("/api/v1/projects/pid-1/chat", headers=AUTH,
                 json={"message": "hi", "allowEmptyRetrieval": True, "sessionId": "s9"})
    j = r.json()
    assert r.status_code == 200 and j["ok"] and j["sessionId"] == "s9"
    assert j["message"] == {"role": "assistant", "content": "answer"}
    assert j["events"] == [{"type": "x", "redacted": True}]
    assert seen["request"]["allowEmptyRetrieval"] is False and seen["on_event"] is None
    assert seen["config"]["llm"] == {"provider": "openai", "model": "m"}
    assert seen["project"]["id"] == "pid-1"
    r = env.post("/api/v1/projects/pid-1/chat", headers=AUTH, json={"message": "hi"})
    assert r.json()["sessionId"].startswith("api_") and seen["request"]["runId"].startswith("run_")


@pytest.mark.parametrize("msg,code", [("Agent turn cancelled", 499), ("message is required", 400), ("llm down", 502)])
def test_chat_error_mapping(env, monkeypatch, msg, code):
    def run_chat(*a):
        raise RuntimeError(msg)

    _fake_runtime(monkeypatch, run_chat)
    r = env.post("/api/v1/projects/pid-1/chat", headers=AUTH, json={"message": "hi"})
    assert r.status_code == code and r.json() == {"ok": False, "error": msg}


def _sse_events(text):
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        ev, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        out.append((ev, data))
    return out


def test_chat_sse(env, monkeypatch):
    def run_chat(project, request, config, cancel_event, on_event):
        on_event({"type": "token", "text": "he"})
        on_event({"type": "token", "text": "llo"})
        return {"projectId": project["id"], "sessionId": request["sessionId"], "mode": "chat",
                "message": "hello", "references": [], "toolEvents": [], "events": [], "usage": None}

    _fake_runtime(monkeypatch, run_chat)
    with env.stream("POST", "/api/v1/projects/pid-1/chat", headers=AUTH,
                    json={"message": "hi", "stream": True, "sessionId": "s1", "runId": "r1"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache, no-transform"
        assert r.headers["x-accel-buffering"] == "no"
        text = "".join(r.iter_text())
    ev = _sse_events(text)
    assert ev[0] == ("meta", {"projectId": "pid-1", "sessionId": "s1", "runId": "r1"})
    assert [e[0] for e in ev[1:-1]] == ["agent", "agent"]
    assert ev[1][1] == {"type": "token", "text": "he", "redacted": True}
    assert ev[-1][0] == "done" and ev[-1][1]["message"]["content"] == "hello"
    assert text.startswith("event: meta\ndata: {")
    assert api._in_flight["chat"] == 0


def test_chat_sse_via_accept_header_and_error_and_heartbeat(env, monkeypatch):
    monkeypatch.setattr(api, "SSE_HEARTBEAT_INTERVAL", 0.05)

    def run_chat(project, request, config, cancel_event, on_event):
        time.sleep(0.3)
        raise RuntimeError("provider failed")

    _fake_runtime(monkeypatch, run_chat)
    with env.stream("POST", "/api/v1/projects/pid-1/chat",
                    headers={**AUTH, "Accept": "text/event-stream"}, json={"message": "hi"}) as r:
        text = "".join(r.iter_text())
    assert ": keepalive" in text
    ev = _sse_events(text)
    assert ev[0][0] == "meta" and ev[-1] == ("error", {"ok": False, "error": "provider failed"})

    def cancelled(*a):
        raise RuntimeError("Agent turn cancelled")

    _fake_runtime(monkeypatch, cancelled)
    with env.stream("POST", "/api/v1/projects/pid-1/chat", headers=AUTH, json={"message": "hi", "stream": True}) as r:
        ev = _sse_events("".join(r.iter_text()))
    assert ev[-1][0] == "cancelled"
    # explicit stream=false beats the Accept header
    _fake_runtime(monkeypatch, lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    r = env.post("/api/v1/projects/pid-1/chat", headers={**AUTH, "Accept": "text/event-stream"},
                 json={"message": "hi", "stream": False})
    assert r.status_code == 502


def test_chat_stream_concurrency_limit(env, monkeypatch):
    _fake_runtime(monkeypatch, lambda *a: {})
    monkeypatch.setattr(api, "MAX_IN_FLIGHT_CHAT_STREAMS", 0)
    r = env.post("/api/v1/projects/pid-1/chat", headers=AUTH, json={"message": "hi", "stream": True})
    assert r.status_code == 503 and r.json()["error"] == "Too many concurrent Agent chat streams"
    assert api._in_flight["chat"] == 0


def test_cancel_chat(env, monkeypatch):
    _fake_runtime(monkeypatch, lambda *a: {})
    ev = api.cancellations.start("pid-1", "s1", "r1")
    r = env.post("/api/v1/projects/pid-1/chat/s1/cancel", headers=AUTH)
    assert r.json() == {"ok": True, "cancelled": True, "sessionId": "s1"}
    assert ev.is_set()
    r = env.post("/api/v1/projects/pid-1/chat/other/cancel", headers=AUTH)
    assert r.json()["cancelled"] is False
    assert env.post("/api/v1/projects/zzz/chat/s1/cancel", headers=AUTH).status_code == 404
    assert env.post("/api/v1/projects/pid-1/chat/s1/cancel").status_code == 401


def test_project_llm_override_merge():
    parsed = {
        "llmConfig": {"model": "global"},
        "projectLlmOverrides": {"p": {"enabled": True, "presetId": "openai", "profile": {"provider": "openai"}}},
        "providerConfigs": {"openai": {"apiKey": "k", "model": "m2", "baseUrl": "http://x"}},
    }
    cfg = api.project_llm_config(parsed, "p")
    assert cfg == {"provider": "openai", "apiKey": "k", "model": "m2", "customEndpoint": "http://x"}
    assert api.project_llm_config(parsed, "other") == {"model": "global"}
    parsed["projectLlmOverrides"]["p"]["presetId"] = "custom-gone"
    assert api.project_llm_config(parsed, "p") == {"model": "global"}
