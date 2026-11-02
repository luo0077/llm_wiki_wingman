from __future__ import annotations

import os
import time

import pytest

from llm_wiki_server import events
from llm_wiki_server.commands import file_sync as fs
from llm_wiki_server.errors import CommandError
from llm_wiki_server.registry import get_command

CFG = fs.normalize_source_watch_config(None)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "raw" / "sources").mkdir(parents=True)
    return tmp_path


def test_commands_registered():
    for n in ["start_project_file_watcher", "stop_project_file_watcher", "rescan_project_files",
              "invalidate_project_file_snapshot_paths", "get_file_change_queue",
              "retry_file_change_task", "ignore_file_change_task"]:
        assert get_command(n) is not None


def test_md5_detects_same_size_content_changes(root):
    rel = "raw/sources/a.md"
    (root / rel).write_text("aaaa")
    fs.ensure_sync_dir(root)
    fs.enqueue_rescan_changes(root, "p1", CFG)
    first = fs.read_queue(root)["tasks"][0]
    fs.with_queue_lock(root, lambda: fs.write_task_meta_to_snapshot(root, first, fs.read_meta(root, rel)))
    fs.write_queue(root, {"version": 1, "tasks": []})
    (root / rel).write_text("bbbb")
    fs.enqueue_paths(root, "p1", {rel})
    q = fs.read_queue(root)
    assert len(q["tasks"]) == 1
    assert q["tasks"][0]["kind"] == "modified"
    assert q["tasks"][0]["hashBefore"] != q["tasks"][0]["hashAfter"]


def test_repeated_changes_upsert_one_pending_task(root):
    rel = "raw/sources/a.md"
    (root / rel).write_text("one")
    fs.ensure_sync_dir(root)
    fs.enqueue_paths(root, "p1", {rel})
    (root / rel).write_text("two")
    fs.enqueue_paths(root, "p1", {rel})
    q = fs.read_queue(root)
    assert len(q["tasks"]) == 1 and q["tasks"][0]["status"] == "pending"


def test_directory_delete_expands_snapshot_children(root):
    a, b = "raw/sources/folder/a.md", "raw/sources/folder/b.md"
    (root / "raw/sources/folder").mkdir()
    (root / a).write_text("a")
    (root / b).write_text("b")
    fs.ensure_sync_dir(root)
    fs.sync_snapshot_paths(root, {a, b})
    import shutil
    shutil.rmtree(root / "raw/sources/folder")
    rels = set()
    fs.collect_known_paths(root, root / "raw/sources/folder", fs.read_snapshot(root), rels,
                           fs.SourceWatchRules(CFG))
    assert rels == {a, b}


def test_prefix_rescan_detects_created_and_deleted(root):
    (root / "raw/sources/old.md").write_text("old")
    fs.ensure_sync_dir(root)
    fs.sync_snapshot_paths(root, {"raw/sources/old.md"})
    (root / "raw/sources/old.md").unlink()
    (root / "raw/sources/new.md").write_text("new")
    fs.enqueue_rescan_changes_for_prefixes(root, "p1", ["raw/sources"], CFG)
    by = {t["path"]: t["kind"] for t in fs.read_queue(root)["tasks"]}
    assert by == {"raw/sources/old.md": "deleted", "raw/sources/new.md": "created"}


def test_invalidate_snapshot_paths(root):
    rel = "raw/sources/retry.md"
    (root / rel).write_text("retry me")
    fs.ensure_sync_dir(root)
    fs.sync_snapshot_paths(root, {rel})
    get_command("invalidate_project_file_snapshot_paths")(str(root), [rel])
    fs.enqueue_rescan_changes_for_prefixes(root, "p1", ["raw/sources"], CFG)
    assert any(t["path"] == rel and t["kind"] == "created" for t in fs.read_queue(root)["tasks"])
    with pytest.raises(CommandError, match="invalid project-relative snapshot path"):
        get_command("invalidate_project_file_snapshot_paths")(str(root), ["../x"])


def test_startup_rescan_scans_watch_roots_only(root):
    (root / "wiki/entities").mkdir(parents=True)
    (root / "unwatched/deep").mkdir(parents=True)
    (root / "raw/sources/source.md").write_text("s")
    (root / "wiki/entities/topic.md").write_text("t")
    (root / "purpose.md").write_text("p")
    (root / "schema.md").write_text("s")
    (root / "unwatched/deep/note.md").write_text("n")
    fs.ensure_sync_dir(root)
    fs.enqueue_startup_rescan_changes(root, "p1", CFG)
    paths = {t["path"] for t in fs.read_queue(root)["tasks"]}
    assert {"raw/sources/source.md", "wiki/entities/topic.md", "purpose.md", "schema.md"} <= paths
    assert "unwatched/deep/note.md" not in paths


def test_rescan_command_emits_events_and_clears_done(root):
    (root / "raw/sources/a.md").write_text("hello")
    (root / "raw/sources/skip.tmp").write_text("x")
    seen = []
    remove = events.add_listener(lambda t, p: seen.append((t, p)))
    try:
        res = get_command("rescan_project_files")("p1", str(root), None, None)
    finally:
        remove()
    assert [t["path"] for t in res["changedTasks"]] == ["raw/sources/a.md"]
    assert res["queue"]["tasks"] == []  # done tasks are dropped
    topics = [t for t, _ in seen]
    assert "file-sync://changed" in topics and "file-sync://queue-updated" in topics
    snap = fs.read_snapshot(root)
    assert "raw/sources/a.md" in snap["files"] and "raw/sources/skip.tmp" not in snap["files"]
    # unchanged -> nothing new
    res2 = get_command("rescan_project_files")("p1", str(root), None, None)
    assert res2["changedTasks"] == []


def test_ignore_and_retry(root):
    fs.ensure_sync_dir(root)
    task = {"id": "t1", "projectId": "p1", "path": "raw/sources/x.md", "kind": "created",
            "status": "failed", "hashBefore": None, "hashAfter": None, "size": 1, "mtimeMs": 1,
            "createdAt": 1, "updatedAt": 1, "retryCount": 3, "error": "boom", "needsRerun": False}
    fs.write_queue(root, {"version": 1, "tasks": [task]})
    q = get_command("ignore_file_change_task")("p1", str(root), "t1")
    assert q["tasks"] == []
    fs.write_queue(root, {"version": 1, "tasks": [task]})
    (root / "raw/sources/x.md").write_text("hi")
    q = get_command("retry_file_change_task")("p1", str(root), "t1")
    assert q["tasks"] == []  # processed -> done -> removed
    assert "raw/sources/x.md" in fs.read_snapshot(root)["files"]


def test_app_write_ignore(root):
    p = root / "wiki" / "a.md"
    assert not fs.is_app_write_ignored(p)
    fs.mark_app_write_path(p)
    assert fs.is_app_write_ignored(p)
    fs.mark_app_write_path(root / "dir")
    assert fs.is_app_write_ignored(root / "dir" / "child.md")


def test_should_watch_rules():
    r = fs.SourceWatchRules(CFG)
    assert fs.should_watch_rel("raw/sources/a.pdf", r)
    assert not fs.should_watch_rel("raw/sources/a.exe", r)
    assert not fs.should_watch_rel("raw/sources/node_modules/a.md", r)
    assert not fs.should_watch_rel("raw/sources/~$a.docx", r)
    assert not fs.should_watch_rel("wiki/media/a.md", r)
    assert fs.should_watch_rel("wiki/x/y.md", r) and not fs.should_watch_rel("wiki/x/y.txt", r)
    assert fs.wildcard_match("*.draft.*", "a.DRAFT.md")


def test_watcher_picks_up_changes_and_ignores_app_writes(root):
    (root / "wiki").mkdir()
    (root / "schema.md").write_text("s")
    fs_cmd = get_command("start_project_file_watcher")
    res = fs_cmd("p1", str(root), None)
    try:
        assert res["queue"]["tasks"] == []
        (root / "raw/sources/live.md").write_text("live")
        deadline = time.time() + 15
        while time.time() < deadline:
            if "raw/sources/live.md" in fs.read_snapshot(root)["files"]:
                break
            time.sleep(0.3)
        else:
            pytest.fail("watcher did not process new file")
        # app write: marked path is synced into snapshot silently (no task)
        p = root / "raw/sources/app.md"
        fs.mark_app_write_path(p)
        p.write_text("app")
        time.sleep(2.5)
        assert fs.read_queue(root)["tasks"] == []
    finally:
        get_command("stop_project_file_watcher")()
