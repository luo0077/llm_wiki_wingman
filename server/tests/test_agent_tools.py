import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient

from llm_wiki_server.app import create_app
from llm_wiki_server.agent import tools as T
from llm_wiki_server.errors import CommandError

c = TestClient(create_app())


def mk(tmp_path):
    r = c.post("/rpc/create_project", json={"name": "w", "path": str(tmp_path)})
    return r.json()["path"]


def run(coro):
    return asyncio.run(coro)


def put(root, rel, content):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p


def ctx(root, **kw):
    return T.ToolContext(root, **kw)


def err(fn, *a):
    with pytest.raises(CommandError) as e:
        fn(*a)
    return e.value.message


def test_specs():
    specs = T.builtin_tool_specs()
    names = [s.name for s in specs]
    for n in ["wiki.search", "wiki.read_page", "source.search", "graph.search", "anytxt.search",
              "wiki.write_page", "llm.generate", "skills.load", "skill.read_file",
              "workspace.write_file", "workspace.append_file", "shell.exec", "web.search",
              "deep_research.run"]:
        assert n in names
    llm = next(s for s in specs if s.name == "llm.generate")
    assert "parameters" not in json.loads(json.dumps(llm))
    assert next(s for s in specs if s.name == "shell.exec")["effects"] == ["read", "process"]
    assert T.BuiltinToolRegistry().specs()[0]["name"] == "wiki.search"


def test_write_page_rules(tmp_path):
    root = mk(tmp_path)
    ref = T.write_wiki_page_with_options(root, "wiki/queries/new-page.md", "# Hello\n\nBody", False)
    assert ref["title"] == "Hello" and ref["kind"] == "wiki" and ref["path"] == "wiki/queries/new-page.md"
    assert "existedBefore" not in ref
    assert open(os.path.join(root, "wiki/queries/new-page.md")).read() == "# Hello\n\nBody"
    m = err(T.write_wiki_page_with_options, root, "wiki/queries/new-page.md", "x", False)
    assert "without allowOverwrite=true" in m
    out = T.write_wiki_page_verified(root, "wiki/queries/new-page.md", "新内容", True)
    assert out == {"path": "wiki/queries/new-page.md", "bytes": 9, "allowOverwrite": True,
                   "verified": True, "existedBefore": True}
    # title fallback
    r = T.write_wiki_page_with_options(root, "wiki/my-page.md", "no heading here", False)
    assert r["title"] == "my page"
    for bad, frag in [
        ("../secret.md", "Markdown file under wiki/"),
        ("wiki/../x.md", "hidden paths"),
        ("/etc/x.md", "stay inside"),
        ("wiki/a/.hid.md", "hidden"),
        ("wiki/a.txt", "Markdown file"),
        ("wiki/CON.md", "reserved"),
        ("wiki/a:b.md", "invalid on Windows"),
        ("wiki//a.md", "empty segment"),
        ("wiki/a /b.md", "not portable"),
        ("wiki/a/../b.md", "hidden paths"),
    ]:
        assert frag in err(T.write_wiki_page_verified, root, bad, "x", False), bad
    assert "too large" in err(T.write_wiki_page_verified, root, "wiki/big.md",
                              "x" * (2 * 1024 * 1024 + 1), False)


def test_write_page_symlink_escape(tmp_path):
    root = mk(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(str(outside), os.path.join(root, "wiki", "link"))
    assert "escapes" in err(T.write_wiki_page_verified, root, "wiki/link/x.md", "x", False)
    assert not (outside / "x.md").exists()
    target = outside / "t.md"
    target.write_text("t")
    os.symlink(str(target), os.path.join(root, "wiki", "sl.md"))
    m = err(T.write_wiki_page_verified, root, "wiki/sl.md", "x", True)
    assert "escapes" in m or "symlink" in m
    assert target.read_text() == "t"


def test_read_page(tmp_path):
    root = mk(tmp_path)
    assert "wiki.read_page" in err(T.read_wiki_page, root, "../secret.md")
    assert "wiki.read_page" in err(T.read_wiki_page, root, "raw/sources/a.md")
    assert "wiki.read_page" in err(T.read_wiki_page, root, "wiki/.hidden.md")
    put(root, "wiki/a.md", "# A\n")
    assert T.read_wiki_page(root, "/wiki/a.md") == "# A\n"
    os.makedirs(os.path.join(root, "wiki", "dir.md"))
    assert "not a file" in err(T.read_wiki_page, root, "wiki/dir.md")
    put(root, "wiki/big.md", "x" * (2 * 1024 * 1024 + 1))
    assert "too large" in err(T.read_wiki_page, root, "wiki/big.md")


def test_knowledge_graph_and_read_tool(tmp_path):
    root = mk(tmp_path)
    os.makedirs(os.path.join(root, ".llm-wiki"), exist_ok=True)
    put(root, ".llm-wiki/history-settings.json", '{"enabled":true,"maxVersionsPerFile":10}')
    put(root, "wiki/entities/alpha.md", "---\ntags: [x, 'y']\n---\n# Alpha\nSee [[Beta]] and [[beta|b]].")
    put(root, "wiki/entities/beta.md", "---\ntags:\n  - one\n  - \"two\"\n---\n# Beta\nback to [[alpha]]")
    put(root, "wiki/entities/gamma.md", "# Gamma\nlinks [[Alpha]]")
    idx = T.build_knowledge_context_index(root)
    a = idx["wiki/entities/alpha.md"]
    assert a["tags"] == ["x", "y"]
    assert a["outgoingLinks"] == ["Beta", "beta"]
    assert a["backlinks"] == ["wiki/entities/beta.md", "wiki/entities/gamma.md"]
    assert a["linkCount"] == 4
    assert idx["wiki/entities/beta.md"]["tags"] == ["one", "two"]
    assert "tags" not in idx["wiki/entities/gamma.md"]

    T.write_wiki_page_verified(root, "wiki/entities/alpha.md", "# Alpha\n[[Beta]]", True)
    out = run(T.execute_tool("wiki.read_page", {"path": " wiki/entities/alpha.md "}, ctx(root)))
    assert out["path"] == "wiki/entities/alpha.md"
    kc = out["knowledgeContext"]
    assert kc["latestVersion"]["author"] == "agent" and kc["latestVersion"]["tool"] == "wiki.write_page"


def test_search_graph(tmp_path):
    root = mk(tmp_path)
    put(root, "wiki/a.md", "# Alpha Thing\nlinks [[b]]")
    put(root, "wiki/b.md", "# Bee\nplain")
    put(root, "wiki/c.md", "# Sea\nplain")
    refs = T.search_graph(root, "alpha", 5)
    assert [r["path"] for r in refs] == ["wiki/a.md", "wiki/b.md"]
    assert refs[0]["score"] == 10001.0 and refs[1]["score"] == 5001.0
    assert refs[0]["snippet"] == "matched entity; 1 related link(s)"
    assert refs[1]["snippet"].startswith("direct neighbor")
    assert refs[0]["kind"] == "graph" and refs[0]["knowledgeContext"]["linkCount"] == 1
    assert T.search_graph(root, "zzzz", 5) == []
    assert T.search_graph(root, "  ", 5) == []


def test_search_sources(tmp_path):
    root = mk(tmp_path)
    put(root, "raw/sources/a.txt", "x" * 600 + " needle here " + "y" * 600)
    put(root, "raw/sources/.hidden.txt", "needle")
    put(root, "raw/sources/b.bin", "needle")
    put(root, "raw/sources/c.md", "nothing")
    refs = T.search_sources(root, "needle", 5)
    assert len(refs) == 1
    r = refs[0]
    assert r["title"] == "a.txt" and r["path"] == "raw/sources/a.txt" and r["kind"] == "source"
    assert r["snippet"].startswith("...") and r["snippet"].endswith("...") and "needle here" in r["snippet"]
    assert "source.search query is required" in err(T.search_sources, root, " ", 5)
    # term fallback, stop words ignored
    put(root, "raw/sources/d.csv", "foo,bar\n")
    assert [x["title"] for x in T.search_sources(root, "sources bar", 5)] == ["d.csv"]
    assert T.search_sources(root, "raw", 5) == []


def test_workspace_write_append(tmp_path):
    root = mk(tmp_path)
    o = T.write_workspace_file(root, "cover/cover.svg", "<svg/>")
    assert o == {"path": "agent-workspace/cover/cover.svg", "bytes": 6, "existedBefore": False}
    o = T.append_workspace_file(root, "cover/cover.svg", "ab")
    assert o["bytes"] == 8 and o["existedBefore"] is True and o["previousContent"] == "<svg/>"
    o = T.write_workspace_file(root, "cover/cover.svg", "new")
    assert o["previousContent"] == "<svg/>ab" and o["bytes"] == 3
    for bad in ["", "wiki/x.md", "raw/x", ".hid/x", "../x", "a/../b"]:
        m = err(T.write_workspace_file, root, bad, "x")
        assert "workspace.write_file" in m, bad
    assert "workspace.append_file" in err(T.append_workspace_file, root, "../x", "x")
    os.makedirs(os.path.join(root, "agent-workspace"), exist_ok=True)
    outside = tmp_path / "out"
    outside.mkdir()
    os.symlink(str(outside), os.path.join(root, "agent-workspace", "lnk"))
    assert "escapes" in err(T.write_workspace_file, root, "lnk/x.txt", "x")
    assert "too large" in err(T.write_workspace_file, root, "big", "x" * (2 * 1024 * 1024 + 1))


def test_shell_exec(tmp_path):
    root = mk(tmp_path)
    out = run(T.execute_tool("shell.exec", {"command": "echo hi; echo err >&2; echo data > out.txt; echo $LLM_WIKI_PROJECT"},
                             ctx(root)))
    assert out["exitCode"] == 0 and out["stdout"].startswith("hi\n") and root in out["stdout"]
    assert out["stderr"] == "err\n" and out["timedOut"] is False
    assert out["generatedFiles"] == [{"path": "agent-workspace/out.txt", "bytes": 5, "existedBefore": False}]
    out = run(T.run_shell_exec(root, "exit 3", 5))
    assert out["exitCode"] == 3 and out["generatedFiles"] == []
    out = run(T.run_shell_exec(root, "sleep 5", 1))
    assert out["timedOut"] is True and out["exitCode"] is None
    assert out["stderr"].endswith("Command timed out after 1s")
    assert "empty" in err(lambda *a: run(T.run_shell_exec(*a)), root, "  ", 5)
    assert "too long" in err(lambda *a: run(T.run_shell_exec(*a)), root, "x" * 4001, 5)
    out = run(T.run_shell_exec(root, "yes | head -c 100000", 5))
    assert len(out["stdout"]) == 20003


def test_registry_dispatch(tmp_path):
    root = mk(tmp_path)
    put(root, "wiki/a.md", "# Alpha\nneedle text")
    x = ctx(root)
    out = run(T.execute_tool("wiki.write_page", {"path": "wiki/n.md", "content": "# N"}, x))
    assert out["existedBefore"] is False and out["path"] == "wiki/n.md"
    with pytest.raises(CommandError, match="refuses to overwrite"):
        run(T.execute_tool("wiki.write_page", {"path": "wiki/n.md", "content": "# N2"}, x))
    out = run(T.execute_tool("wiki.write_page", {"path": "wiki/n.md", "content": "# N2",
                                                 "allow_overwrite": True}, x))
    assert out["existedBefore"] is True
    with pytest.raises(CommandError, match="requires path"):
        run(T.execute_tool("wiki.write_page", {"content": "x"}, x))
    with pytest.raises(CommandError, match="requires content"):
        run(T.execute_tool("workspace.write_file", {"path": "a"}, x))
    with pytest.raises(CommandError, match="wiki.search requires query"):
        run(T.execute_tool("wiki.search", {"query": "  "}, x))
    with pytest.raises(CommandError, match="Unknown Agent tool: nope"):
        run(T.execute_tool("nope", {}, x))
    out = run(T.execute_tool("deep_research.run", {"query": " q "}, x))
    assert out == {"query": "q", "status": "orchestrated_by_agent_runtime"}
    out = run(T.execute_tool("graph.search", {"query": "alpha", "topK": 50}, x))
    assert out[0]["path"] == "wiki/a.md"
    assert T.tool_top_k({"topK": 99}) == 10 and T.tool_top_k({}) == 5 and T.tool_top_k({"top_k": 0}) == 1
    out = run(T.execute_tool("workspace.append_file", {"path": "a.txt", "content": "z"}, x))
    assert out["bytes"] == 1
    out = run(T.execute_tool("source.search", {"query": "x"}, x))
    assert out == []
    with pytest.raises(CommandError, match="no search provider"):
        run(T.execute_tool("web.search", {"query": "x"}, x))
    out = run(T.execute_tool("shell.exec", {"query": "echo q", "timeoutSeconds": 999}, x))
    assert out["stdout"] == "q\n"


def test_wiki_search_tool(tmp_path):
    root = mk(tmp_path)
    put(root, "wiki/a.md", "# Alpha\nneedle text [[b]]")
    put(root, "wiki/b.md", "# Bee\nother")
    out = run(T.run_wiki_search(root, "needle", 5, False, None))
    assert set(out) == {"mode", "tokenHits", "vectorHits", "graphHits", "references"}
    assert out["references"][0]["path"] == "wiki/a.md"
    ref = out["references"][0]
    assert ref["kind"] in ("wiki", "graph") and ref["knowledgeContext"]["linkCount"] == 1
    assert isinstance(ref["score"], float)
    out2 = run(T.execute_tool("wiki.search", {"query": "needle", "includeContent": True}, ctx(root)))
    assert out2["references"][0]["path"] == "wiki/a.md"
