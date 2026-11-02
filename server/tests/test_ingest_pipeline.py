from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from llm_wiki_server import events, state
from llm_wiki_server.app import create_app
from llm_wiki_server.commands import rest_api as api
from llm_wiki_server.ingest import llm_client, pipeline, review_store
from llm_wiki_server.ingest import queue as iq
from llm_wiki_server.ingest.source_identity import (
    source_identity_for_path,
    source_summary_slug_from_identity,
)

TOKEN = "tok-ingest"
AUTH = {"Authorization": "Bearer " + TOKEN}

SOURCE = "# Transformers\n\nThe Transformer architecture uses attention. Vaswani introduced it.\n"


class FakeLlm:
    """Replaces llm_client.stream_chat; routes on the system prompt."""

    def __init__(self):
        self.calls = []
        self.generation_mode = "ok"  # ok | truncated | resume
        self.block_analysis = None  # threading/async gate for cancel tests
        self.resume_seen = False

    async def __call__(self, config, messages, callbacks, signal=None, overrides=None, **kw):
        cb = llm_client.StreamCallbacks.coerce(callbacks)
        system = messages[0]["content"]
        user = messages[-1]["content"]
        self.calls.append(system[:60])
        completion = None
        if system.startswith("You are an expert research analyst"):
            if self.block_analysis is not None:
                while not self.block_analysis.is_set():
                    if signal is not None and signal.aborted:
                        await llm_client._call(cb.on_done, None)
                        return
                    await asyncio.sleep(0.01)
            text = "## Key Entities\n- Transformer (page-worthy: yes)\n"
        elif system.startswith("You are a wiki maintainer"):
            if "RESUME COMPLETE" in user:
                self.resume_seen = True
                text = "---RESUME COMPLETE---"
            else:
                text = (
                    "---FILE: wiki/sources/whatever.md---\n---\ntype: source\ntitle: Source Transformers\n"
                    "tags: []\nrelated: []\n---\n# Source\nSummary of transformers.\n---END FILE---\n"
                    "---FILE: wiki/entities/transformer.md---\n---\ntype: entity\ntitle: Transformer\n"
                    "tags: [ml]\nrelated: []\n---\n# Transformer\nAn attention based architecture.\n---END FILE---\n"
                    "---REVIEW: missing-page | Attention mechanism---\nNo page for attention yet.\n"
                    "OPTIONS: Create Page | Skip\nSEARCH: attention mechanism\n---END REVIEW---\n"
                )
                if self.generation_mode == "truncated":
                    self.generation_mode = "ok"
                    completion = llm_client.StreamCompletion(finish_reason="length", truncated=True)
        elif system.startswith("You are merging"):
            text = "---\ntype: entity\ntitle: Transformer\n---\n# Transformer\nMerged.\n"
        else:
            text = ""
        if text:
            await llm_client._call(cb.on_token, text)
        await llm_client._call(cb.on_done, completion)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_WIKI_API_TOKEN", raising=False)
    monkeypatch.setattr(iq, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    old = state._state
    proj = tmp_path / "proj"
    for d in ("wiki/entities", "wiki/sources", "raw/sources", ".llm-wiki"):
        (proj / d).mkdir(parents=True)
    (proj / "schema.md").write_text("# schema")
    (proj / "purpose.md").write_text("# purpose")
    (proj / "wiki/index.md").write_text("# Wiki Index\n")
    (proj / "raw/sources/transformers.md").write_text(SOURCE)
    (proj / ".llm-wiki/project.json").write_text(json.dumps({"id": "pid-1"}))
    (tmp_path / "data").mkdir()
    st = state.AppState(tmp_path / "data" / "app-state.json")
    state._state = st
    st.set("apiConfig", {"token": TOKEN})
    st.set("projectRegistry", {"pid-1": {"path": str(proj), "name": "Proj"}})
    st.set("llmConfig", {"provider": "custom", "apiKey": "k", "model": "m",
                         "customEndpoint": "http://llm.invalid/v1", "maxContextSize": 128000})
    api.invalidate_config_cache()
    with api._rate_lock:
        api._rate_hits.clear()
    fake = FakeLlm()
    monkeypatch.setattr(llm_client, "stream_chat", fake)
    iq.manager().reset()
    seen = []
    remove = events.add_listener(lambda t, p: seen.append((t, p)))
    c = TestClient(create_app())
    c.proj, c.fake, c.events = proj, fake, seen
    yield c
    remove()
    iq.manager().reset()
    state._state = old
    api.invalidate_config_cache()


def _wait(pq, timeout=20):
    assert iq.manager().wait_idle(pq, timeout), pq.snapshot()


def test_full_ingest_through_queue(env):
    proj = env.proj
    pq = iq.manager().get("pid-1")
    ids = pq.enqueue_batch([{"sourcePath": "raw/sources/transformers.md", "folderContext": "ml"}])
    _wait(pq)
    assert len(ids) == 1
    identity = source_identity_for_path(str(proj), str(proj / "raw/sources/transformers.md"))
    slug = source_summary_slug_from_identity(identity)
    summary = proj / "wiki/sources" / (slug + ".md")
    assert summary.exists() and "Summary of transformers" in summary.read_text()
    assert identity in summary.read_text()  # sources field canonicalised
    entity = proj / "wiki/entities/transformer.md"
    assert "attention based" in entity.read_text()
    index = (proj / "wiki/index.md").read_text()
    assert "[[entities/transformer]]" in index and "## Recently Updated" in index
    assert "] ingest | " in (proj / "wiki/log.md").read_text()
    reviews = json.loads((proj / ".llm-wiki/review.json").read_text())
    assert len(reviews) == 1 and reviews[0]["type"] == "missing-page"
    assert reviews[0]["resolved"] is False and reviews[0]["id"].startswith("review-")
    assert reviews[0]["searchQueries"] == ["attention mechanism"]
    # the REST API reads the very same file
    r = env.get("/api/v1/projects/pid-1/reviews", headers=AUTH)
    assert r.json()["count"] == 1
    # queue: done task removed, nothing persisted
    assert pq.summary()["completed"] == 1 and pq.tasks == []
    assert json.loads((proj / ".llm-wiki/ingest-queue.json").read_text()) == []
    assert (proj / ".llm-wiki/ingest-cache.json").exists()
    topics = {t for t, _ in env.events}
    assert {"ingest://queue-updated", "ingest://activity", "ingest://task-done"} <= topics
    act = pipeline.list_activity(str(proj))[0]
    assert act["status"] == "done" and "files written" in act["detail"]
    # second run is served from the ingest cache: no new LLM calls
    count = lambda: len([c for c in env.fake.calls if "stale review" not in c])  # noqa: E731
    n = count()
    pq.enqueue_batch([{"sourcePath": "raw/sources/transformers.md", "folderContext": ""}])
    _wait(pq)
    assert count() == n
    assert "Skipped (unchanged)" in pipeline.list_activity(str(proj))[0]["detail"]


def test_cancel_then_retry(env):
    import threading

    gate = threading.Event()
    env.fake.block_analysis = gate
    pq = iq.manager().get("pid-1")
    (tid,) = pq.enqueue_batch([{"sourcePath": "raw/sources/transformers.md"}])
    deadline = time.time() + 10
    while pq.find(tid)["status"] != "processing" and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    assert pq.cancel([tid]) == 1
    _wait(pq)
    assert pq.find(tid)["status"] == "cancelled"
    persisted = json.loads((env.proj / ".llm-wiki/ingest-queue.json").read_text())
    assert persisted[0]["status"] == "cancelled" and persisted[0]["id"] == tid
    assert not (env.proj / "wiki/entities/transformer.md").exists()
    gate.set()
    assert pq.retry([tid]) == 1
    _wait(pq)
    assert pq.tasks == [] and (env.proj / "wiki/entities/transformer.md").exists()


def test_resume_after_restart(env):
    # queue file as the desktop app would leave it after a crash mid-task
    task = {"id": "ingest-1-abc", "projectId": "pid-1", "sourcePath": "raw/sources/transformers.md",
            "folderContext": "", "status": "processing", "addedAt": 1, "error": None, "retryCount": 0}
    (env.proj / ".llm-wiki/ingest-queue.json").write_text(json.dumps([task]))
    iq.manager().reset()
    assert iq.restore_all_queues() == 1
    pq = iq.manager().get("pid-1")
    _wait(pq)
    assert pq.tasks == [] and (env.proj / "wiki/entities/transformer.md").exists()


def test_interrupted_generation_resumes_from_checkpoint(env):
    env.fake.generation_mode = "truncated"
    pq = iq.manager().get("pid-1")
    (tid,) = pq.enqueue_batch([{"sourcePath": "raw/sources/transformers.md"}])
    _wait(pq)
    done = [p for t, p in env.events if t == "ingest://task-done"]
    # attempt 1: truncated output -> incomplete (pages kept, checkpoint saved) -> queue retry;
    # attempt 2: resume prompt lists the finished paths and the model confirms completion.
    assert [d["result"] for d in done] == ["retry", "done"]
    assert "Ingest incomplete" in done[0]["error"]
    assert env.fake.resume_seen
    assert pq.find(tid) is None
    assert "attention based" in (env.proj / "wiki/entities/transformer.md").read_text()
    assert not list((env.proj / ".llm-wiki/ingest-generation").glob("*.json"))


def test_failure_retries_then_fails(env, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(pipeline, "auto_ingest", boom)
    pq = iq.manager().get("pid-1")
    (tid,) = pq.enqueue_batch([{"sourcePath": "raw/sources/transformers.md"}])
    _wait(pq)
    t = pq.find(tid)
    assert t["status"] == "failed" and t["retryCount"] == iq.MAX_RETRIES and "exploded" in t["error"]


def test_blocked_without_llm(env):
    state.app_state().set("llmConfig", {"provider": "openai", "apiKey": ""})
    pq = iq.manager().get("pid-1")
    (tid,) = pq.enqueue_batch([{"sourcePath": "raw/sources/transformers.md"}])
    assert pq.blocked_on_llm_config and pq.find(tid)["status"] == "pending"
    assert "LLM not configured" in pq.find(tid)["error"]


def test_rest_and_rpc_surface(env):
    c = env
    assert c.post("/api/v1/projects/pid-1/ingest", json={"sourcePath": "raw/sources/transformers.md"}).status_code == 401
    r = c.post("/api/v1/projects/pid-1/ingest", headers=AUTH, json={"sourcePath": "../../etc/passwd"})
    assert r.status_code == 400
    r = c.post("/api/v1/projects/pid-1/ingest", headers=AUTH,
               json={"upload": {"name": "../up.md", "content": "# Uploaded\nHello world\n"}, "folderContext": "x"})
    assert r.status_code == 200, r.text
    tid = r.json()["taskIds"][0]
    assert (c.proj / "raw/sources/up.md").exists()
    pq = iq.manager().get("pid-1")
    _wait(pq)
    q = c.get("/api/v1/projects/pid-1/ingest/queue", headers=AUTH).json()
    assert q["ok"] and "summary" in q and q["summary"]["completed"] >= 1
    assert c.get("/api/v1/projects/pid-1/ingest/queue").status_code == 401
    assert c.get("/api/v1/projects/pid-1/ingest/nope", headers=AUTH).status_code == 404
    # RPC
    rpc = lambda name, **a: c.post("/rpc/" + name, json=a, headers=AUTH).json()  # noqa: E731
    out = rpc("ingest_enqueue", project="pid-1", sourcePath="raw/sources/transformers.md")
    tid2 = out["taskIds"][0]
    _wait(pq)
    assert rpc("ingest_list_queue", project="pid-1")["summary"]["pending"] == 0
    assert rpc("ingest_cancel", project="pid-1", taskId=tid2)["cancelled"] == 0  # already done
    assert rpc("ingest_retry", project="pid-1", all=True)["requeued"] == 0
    assert rpc("ingest_status", project="pid-1")["ok"]
    assert tid != tid2


def test_review_store_dedupes_and_keeps_resolved(env):
    item = {"type": "suggestion", "title": "Look into X", "description": "d", "options": []}
    review_store.add_items(str(env.proj), [item])
    (rid,) = [i["id"] for i in review_store.load_items(str(env.proj))]
    assert review_store.resolve_item(str(env.proj), rid, "Skip")
    review_store.add_items(str(env.proj), [dict(item, description="new", searchQueries=["q"])])
    items = review_store.load_items(str(env.proj))
    assert len(items) == 1 and items[0]["resolved"] is True and items[0]["description"] == "new"
