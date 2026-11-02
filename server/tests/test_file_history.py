import json
import zipfile

from fastapi.testclient import TestClient

from llm_wiki_server.app import create_app
from llm_wiki_server.commands import file_history as fh

c = TestClient(create_app())


def rpc(name, **kw):
    return c.post(f"/rpc/{name}", json=kw)


def mk(tmp_path):
    (tmp_path / ".llm-wiki").mkdir()
    (tmp_path / "wiki").mkdir()
    f = tmp_path / "wiki/page.md"
    f.write_text("v0")
    return str(tmp_path), f


def test_disabled_by_default(tmp_path):
    root, f = mk(tmp_path)
    fh.record_file_version(f, "agent", "t")
    assert not (tmp_path / ".llm-wiki/history").exists()
    assert rpc("get_file_history_settings", projectPath=root).json() == {"enabled": False, "maxVersionsPerFile": 10}


def test_record_list_restore_stats_clear(tmp_path):
    root, f = mk(tmp_path)
    r = rpc("set_file_history_settings", projectPath=root, settings={"enabled": True, "maxVersionsPerFile": 3})
    assert r.json() == {"enabled": True, "maxVersionsPerFile": 3}
    for i in range(5):
        f.write_text(f"v{i}")
        fh.record_file_version(f, "agent", "test.write")
    fh.record_file_version(f, "agent", "test.write")  # duplicate ignored
    ents = rpc("list_file_history", projectPath=root, filePath=str(f)).json()
    assert [e["content"] for e in ents] == ["v4", "v3", "v2"]
    assert set(ents[0]) == {"id", "path", "timestamp", "author", "tool", "content"}
    f.write_text("changed")
    out = rpc("restore_file_history", projectPath=root, filePath=str(f), entryId=ents[2]["id"])
    assert out.json() == "v2" and f.read_text() == "v2"
    ents = rpc("list_file_history", projectPath=root, filePath=str(f)).json()
    assert ents[0]["tool"] == "before.history.restore"
    st = rpc("get_file_history_stats", projectPath=root).json()
    assert st["files"] == 1 and st["entries"] == 3 and st["bytes"] > 0
    assert rpc("restore_file_history", projectPath=root, filePath=str(f), entryId="x").status_code == 400
    assert rpc("clear_file_history", projectPath=root).status_code == 200
    assert not (tmp_path / ".llm-wiki/history").exists()


def test_fnv_and_format(tmp_path):
    root, f = mk(tmp_path)
    rpc("set_file_history_settings", projectPath=root, settings={"enabled": True, "maxVersionsPerFile": 5})
    fh.record_file_version(f, "agent", "x")
    files = list((tmp_path / ".llm-wiki/history").glob("*.json"))
    assert len(files) == 1
    assert files[0].name == f"{fh._history_path(tmp_path, f).stem}.json"
    assert json.loads(files[0].read_text())[0]["content"] == "v0"


def test_rejects_paths_outside(tmp_path):
    root, f = mk(tmp_path)
    other = tmp_path.parent / "outside.md"
    other.write_text("x")
    assert rpc("list_file_history", projectPath=root, filePath=str(other)).status_code == 400
    meta = tmp_path / ".llm-wiki/x.md"
    meta.write_text("x")
    assert rpc("list_file_history", projectPath=root, filePath=str(meta)).status_code == 400
    assert rpc("get_file_history_stats", projectPath=str(tmp_path / "nope")).status_code == 400


def test_record_never_raises(tmp_path):
    fh.record_file_version("/nonexistent/zzz/file.md", "a", "b")
    fh.record_file_version(None, "a", "b")
