"""Translations of ingest-cache.test.ts and project-mutex.test.ts."""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from llm_wiki_server.ingest import ingest_cache as ic
from llm_wiki_server.ingest.project_mutex import (
    reset_project_locks_for_testing,
    with_project_lock,
    with_project_lock_sync,
)


# ── ingest-cache ──────────────────────────────────────────────────

@pytest.fixture
def fake_fs(monkeypatch):
    from llm_wiki_server.commands import fs

    state = {"persisted": "", "exists": lambda p: True, "writes": []}

    def read_file(p):
        return state["persisted"] or json.dumps({"entries": {}})

    def write_file(p, c):
        state["persisted"] = c
        state["writes"].append((p, c))

    monkeypatch.setattr(fs, "read_file", read_file)
    monkeypatch.setattr(fs, "write_file", write_file)
    monkeypatch.setattr(fs, "file_exists", lambda p: state["exists"](p))
    return state


def test_cache_no_entry(fake_fs):
    assert ic.check_ingest_cache("/project", "foo.pdf", "content") is None


def test_cache_hit(fake_fs):
    ic.save_ingest_cache("/project", "foo.pdf", "hello", ["wiki/sources/foo.md", "wiki/entities/bar.md"])
    assert ic.check_ingest_cache("/project", "foo.pdf", "hello") == ["wiki/sources/foo.md", "wiki/entities/bar.md"]


def test_cache_miss_when_file_deleted(fake_fs):
    ic.save_ingest_cache("/project", "foo.pdf", "hello", ["wiki/sources/foo.md", "wiki/entities/bar.md"])
    fake_fs["exists"] = lambda p: "entities/bar.md" not in p
    assert ic.check_ingest_cache("/project", "foo.pdf", "hello") is None


def test_cache_miss_on_content_change(fake_fs):
    ic.save_ingest_cache("/project", "foo.pdf", "hello", ["wiki/sources/foo.md"])
    assert ic.check_ingest_cache("/project", "foo.pdf", "different content") is None


def test_cache_miss_when_exists_raises(fake_fs):
    ic.save_ingest_cache("/project", "foo.pdf", "hello", ["wiki/sources/foo.md"])

    def boom(p):
        raise RuntimeError("stat failed")

    fake_fs["exists"] = boom
    assert ic.check_ingest_cache("/project", "foo.pdf", "hello") is None


def test_cache_absolute_file_paths_checked_as_is(fake_fs):
    seen = []
    fake_fs["exists"] = lambda p: seen.append(p) or True
    ic.save_ingest_cache("C:\\proj", "foo.pdf", "h", ["C:\\proj\\wiki\\a.md", "wiki/b.md"])
    assert ic.check_ingest_cache("C:\\proj", "foo.pdf", "h") is not None
    assert seen == ["C:/proj/wiki/a.md", "C:/proj/wiki/b.md"]


def test_cache_disk_format_matches_ts(fake_fs):
    ic.save_ingest_cache("/project", "foo.pdf", "hello", ["wiki/a.md"])
    path, written = fake_fs["writes"][-1]
    assert path == "/project/.llm-wiki/ingest-cache.json"
    data = json.loads(written)
    entry = data["entries"]["foo.pdf"]
    assert entry["hash"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"  # sha256("hello")
    assert isinstance(entry["timestamp"], int) and entry["timestamp"] > 1_600_000_000_000
    assert entry["filesWritten"] == ["wiki/a.md"]
    assert written == json.dumps(data, indent=2, ensure_ascii=False)
    assert ic.sha256("héllo 中") == __import__("hashlib").sha256("héllo 中".encode()).hexdigest()


def test_cache_list_remove_move(fake_fs):
    ic.save_ingest_cache("/project", "a/x.pdf", "c1", ["wiki/sources/x.md", "wiki/entities/e.md"])
    ic.save_ingest_cache("/project", "y.pdf", "c2", ["wiki/sources/y.md"])
    assert ic.list_ingested_source_identities("/project") == ["a/x.pdf", "y.pdf"]
    ic.move_ingest_cache_entry("/project", "a/x.pdf", "b/x.pdf", {"wiki/sources/x.md": "wiki/sources/b-x.md"})
    assert ic.list_ingested_source_identities("/project") == ["y.pdf", "b/x.pdf"]
    ic.save_ingest_cache  # noqa: B018
    fake_fs["exists"] = lambda p: True
    assert ic.check_ingest_cache("/project", "b/x.pdf", "c1") == ["wiki/sources/b-x.md", "wiki/entities/e.md"]
    ic.remove_from_ingest_cache("/project", "y.pdf")
    assert ic.list_ingested_source_identities("/project") == ["b/x.pdf"]
    # no-ops
    ic.move_ingest_cache_entry("/project", "nope", "other")
    ic.move_ingest_cache_entry("/project", "b/x.pdf", "b/x.pdf")
    assert ic.list_ingested_source_identities("/project") == ["b/x.pdf"]


def test_cache_real_filesystem_roundtrip(tmp_path):
    root = str(tmp_path)
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "a.md").write_text("x")
    ic.save_ingest_cache(root, "s.pdf", "body", ["wiki/a.md"])
    assert (tmp_path / ".llm-wiki" / "ingest-cache.json").exists()
    assert ic.check_ingest_cache(root, "s.pdf", "body") == ["wiki/a.md"]
    (tmp_path / "wiki" / "a.md").unlink()
    assert ic.check_ingest_cache(root, "s.pdf", "body") is None


def test_cache_corrupt_file_treated_as_empty(tmp_path):
    root = str(tmp_path)
    (tmp_path / ".llm-wiki").mkdir()
    (tmp_path / ".llm-wiki" / "ingest-cache.json").write_text("{not json")
    assert ic.list_ingested_source_identities(root) == []


# ── project-mutex ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_locks():
    reset_project_locks_for_testing()
    yield
    reset_project_locks_for_testing()


def test_same_project_serializes():
    async def main():
        order = []
        a_running = asyncio.Event()
        a_release = asyncio.Event()

        async def fn_a():
            order.append("A:start")
            a_running.set()
            await a_release.wait()
            order.append("A:end")
            return "A"

        async def fn_b():
            order.append("B:start")
            order.append("B:end")
            return "B"

        call_a = asyncio.ensure_future(with_project_lock("/proj", fn_a))
        await a_running.wait()
        assert order == ["A:start"]
        call_b = asyncio.ensure_future(with_project_lock("/proj", fn_b))
        for _ in range(5):
            await asyncio.sleep(0)
        assert order == ["A:start"]
        a_release.set()
        assert await call_a == "A"
        assert await call_b == "B"
        assert order == ["A:start", "A:end", "B:start", "B:end"]

    asyncio.run(main())


def test_exception_propagates_and_releases():
    async def main():
        order = []

        async def fails():
            order.append("failer:start")
            raise RuntimeError("boom")

        async def ok():
            order.append("ok:start")
            return "ok"

        c1 = asyncio.ensure_future(with_project_lock("/proj", fails))
        c2 = asyncio.ensure_future(with_project_lock("/proj", ok))
        with pytest.raises(RuntimeError, match="boom"):
            await c1
        assert await c2 == "ok"
        assert order == ["failer:start", "ok:start"]

    asyncio.run(main())


def test_third_caller_waits_behind_both():
    async def main():
        order = []
        a_release, b_release = asyncio.Event(), asyncio.Event()

        async def a():
            order.append("A:start")
            await a_release.wait()
            order.append("A:end")

        async def b():
            order.append("B:start")
            await b_release.wait()
            order.append("B:end")

        async def c():
            order.append("C:start")
            order.append("C:end")

        ta = asyncio.ensure_future(with_project_lock("/proj", a))
        tb = asyncio.ensure_future(with_project_lock("/proj", b))
        tc = asyncio.ensure_future(with_project_lock("/proj", c))
        for _ in range(10):
            await asyncio.sleep(0)
        assert order == ["A:start"]
        a_release.set()
        await ta
        for _ in range(10):
            await asyncio.sleep(0)
        assert order == ["A:start", "A:end", "B:start"]
        b_release.set()
        await tb
        await tc
        assert order == ["A:start", "A:end", "B:start", "B:end", "C:start", "C:end"]

    asyncio.run(main())


def test_different_projects_run_concurrently():
    async def main():
        order = []
        a_running, b_running, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def a():
            order.append("A:start")
            a_running.set()
            await release.wait()
            order.append("A:end")

        async def b():
            order.append("B:start")
            b_running.set()
            await release.wait()
            order.append("B:end")

        ta = asyncio.ensure_future(with_project_lock("/proj-A", a))
        tb = asyncio.ensure_future(with_project_lock("/proj-B", b))
        await asyncio.gather(a_running.wait(), b_running.wait())
        assert "A:end" not in order and "B:end" not in order
        release.set()
        await asyncio.gather(ta, tb)
        assert "A:end" in order and "B:end" in order

    asyncio.run(main())


def test_sync_callable_accepted():
    async def main():
        return await with_project_lock("/p", lambda: 42)

    assert asyncio.run(main()) == 42


def test_threads_and_coroutines_share_one_lock():
    """A worker thread holding the sync lock blocks the async variant, and vice versa."""
    events = []
    started = threading.Event()
    release = threading.Event()

    def worker():
        def body():
            events.append("thread:start")
            started.set()
            release.wait(5)
            events.append("thread:end")

        with_project_lock_sync("/shared", body)

    t = threading.Thread(target=worker)
    t.start()
    assert started.wait(5)

    async def main():
        async def fn():
            events.append("async:start")

        task = asyncio.ensure_future(with_project_lock("/shared", fn))
        await asyncio.sleep(0.05)
        assert events == ["thread:start"]
        release.set()
        await task

    asyncio.run(main())
    t.join(5)
    assert events == ["thread:start", "thread:end", "async:start"]


def test_sync_variant_serializes_threads():
    counter = {"in": 0, "max": 0}
    results = []

    def body():
        counter["in"] += 1
        counter["max"] = max(counter["max"], counter["in"])
        time.sleep(0.01)
        counter["in"] -= 1
        return 1

    threads = [threading.Thread(target=lambda: results.append(with_project_lock_sync("/q", body))) for _ in range(6)]
    [t.start() for t in threads]
    [t.join(5) for t in threads]
    assert counter["max"] == 1 and len(results) == 6


def test_cancelled_waiter_does_not_leak_lock():
    async def main():
        hold = asyncio.Event()

        async def holder():
            await hold.wait()

        t1 = asyncio.ensure_future(with_project_lock("/c", holder))
        await asyncio.sleep(0.01)
        t2 = asyncio.ensure_future(with_project_lock("/c", lambda: "never"))
        await asyncio.sleep(0.01)
        t2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t2
        hold.set()
        await t1
        assert await asyncio.wait_for(with_project_lock("/c", lambda: "after"), 2) == "after"

    asyncio.run(main())


def test_cancelled_waiter_on_thread_lock_does_not_leak():
    started, release = threading.Event(), threading.Event()

    def worker():
        with_project_lock_sync("/leak", lambda: (started.set(), release.wait(5)))

    t = threading.Thread(target=worker)
    t.start()
    assert started.wait(5)

    async def main():
        waiter = asyncio.ensure_future(with_project_lock("/leak", lambda: "never"))
        await asyncio.sleep(0.05)  # waiter is now parked in the executor acquiring the thread lock
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        t.join(5)
        await asyncio.sleep(0.1)  # let the late-release callback run
        assert await asyncio.wait_for(with_project_lock("/leak", lambda: "after"), 2) == "after"

    asyncio.run(main())
