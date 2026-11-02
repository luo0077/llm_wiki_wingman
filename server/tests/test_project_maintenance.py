import zipfile

from fastapi.testclient import TestClient

from llm_wiki_server.app import create_app

c = TestClient(create_app())


def rpc(name, **kw):
    return c.post(f"/rpc/{name}", json=kw)


def test_rebuild_index(tmp_path):
    (tmp_path / "wiki/entities").mkdir(parents=True)
    (tmp_path / "wiki/concepts").mkdir()
    (tmp_path / "wiki/entities/a.md").write_text("---\ntype: entity\ntitle: Alpha\n---\nBody")
    (tmp_path / "wiki/concepts/a.md").write_text("---\ntype: concept\ntitle: 'Also Alpha'\n---\nBody")
    (tmp_path / "wiki/log.md").write_text("x")
    r = rpc("rebuild_wiki_index", projectPath=str(tmp_path))
    assert r.json() == {"pages": 2, "groups": 2}
    idx = (tmp_path / "wiki/index.md").read_text()
    assert "## entity" in idx and "[[entities/a|Alpha]]" in idx and "[[concepts/a|Also Alpha]]" in idx
    assert rpc("rebuild_wiki_index", projectPath=str(tmp_path / "zz")).status_code == 400


def test_roundtrip(tmp_path):
    src, dst, arc = tmp_path / "s", tmp_path / "t", tmp_path / "a.zip"
    (src / "wiki").mkdir(parents=True)
    (src / ".llm-wiki").mkdir()
    (src / "wiki/index.md").write_text("# Index")
    (src / ".llm-wiki/ingest-cache.json").write_text("{}")
    assert rpc("export_project_archive", projectPath=str(src), destination=str(arc)).status_code == 200
    r = rpc("import_project_archive", archivePath=str(arc), destination=str(dst))
    assert r.status_code == 200 and r.json() == str(dst)
    assert (dst / ".llm-wiki/ingest-cache.json").read_text() == "{}"
    # non-empty destination
    assert rpc("import_project_archive", archivePath=str(arc), destination=str(dst)).status_code == 400


def test_export_destination_inside_project(tmp_path):
    (tmp_path / "wiki").mkdir()
    r = rpc("export_project_archive", projectPath=str(tmp_path), destination=str(tmp_path / "wiki/x.zip"))
    assert r.status_code == 400
    assert rpc("export_project_archive", projectPath="rel", destination="x").status_code == 400


def test_zip_slip_and_missing_index(tmp_path):
    for name in ["../evil.txt", "/abs.txt", "./x.txt"]:
        arc = tmp_path / "bad.zip"
        with zipfile.ZipFile(arc, "w") as z:
            z.writestr("wiki/index.md", "x")
            z.writestr(name, "x")
        r = rpc("import_project_archive", archivePath=str(arc), destination=str(tmp_path / "out"))
        assert r.status_code == 400 and "Unsafe archive path" in r.text
    arc = tmp_path / "noidx.zip"
    with zipfile.ZipFile(arc, "w") as z:
        z.writestr("a.txt", "x")
    r = rpc("import_project_archive", archivePath=str(arc), destination=str(tmp_path / "out2"))
    assert "wiki/index.md is missing" in r.text
    assert not (tmp_path / "out").exists()
