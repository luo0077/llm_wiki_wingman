import asyncio
import os
import sys
import types

import pytest

os.environ["LLM_WIKI_VECTOR_BACKEND"] = "sqlite"

from llm_wiki_server.commands import page_embedding as pe  # noqa: E402
from llm_wiki_server.commands import vectorstore as vs  # noqa: E402

CFG = {"enabled": True, "endpoint": "https://example.com/v1/embeddings", "apiKey": "s", "model": "m",
       "outputDimensionality": 768, "extraHeaders": {"X-Route": "a"}, "maxChunkChars": 1000,
       "overlapChunkChars": 200}


@pytest.fixture
def fake_search(monkeypatch):
    calls = []
    mod = types.ModuleType("llm_wiki_server.commands.search")
    mod.supports_embedding_batch = lambda cfg: True

    async def batch(texts, cfg):
        calls.append(list(texts))
        return [[float(len(t) % 7), 1.0, 2.0, 3.0] for t in texts]

    mod.fetch_embedding_batch = batch
    monkeypatch.setitem(sys.modules, "llm_wiki_server.commands.search", mod)
    import llm_wiki_server.commands as pkg
    monkeypatch.setattr(pkg, "search", mod, raising=False)
    return calls


def project(tmp_path):
    (tmp_path / "wiki/nested").mkdir(parents=True)
    return tmp_path


def test_path_guard(tmp_path):
    root = project(tmp_path)
    (root / "wiki/nested/page.md").write_text("# Page")
    (root / "outside.md").write_text("# O")
    _, rel = pe.resolve_wiki_markdown_path(str(root), "wiki\\nested/page.md")
    assert rel == "wiki/nested/page.md"
    for bad in ["../outside.md", "outside.md", "wiki/missing.md", "wiki/../outside.md", "/etc/x.md", ""]:
        with pytest.raises(pe.PageEmbeddingError):
            pe.resolve_wiki_markdown_path(str(root), bad)


def test_chunker():
    inp = "---\r\ntitle: 测试页面\r\n---\r\n# 第一章\r\n\r\n内容一。\r\n## 第二节\r\n内容二。"
    chunks = pe.chunk_markdown(inp, 64, 8)
    assert pe.extract_title(inp, "fb") == "测试页面"
    assert all("title:" not in c["text"] for c in chunks)
    assert any("# 第一章" in c["heading_path"] for c in chunks)
    assert any("## 第二节" in c["heading_path"] for c in chunks)
    assert chunks[-1]["heading_path"] == "# 第一章 > ## 第二节"


def test_cjk_split_overlap():
    text = "中文内容用于验证字符边界不会发生截断异常。" * 20
    parts = pe._split_with_overlap(text, 64, 8)
    assert len(parts) > 1 and all(len(p) <= 64 for p in parts)
    for a, b in zip(parts, parts[1:]):
        assert b.startswith(a[-8:])


def test_atomic_blocks():
    code = "```rust\n" + "let value = 1;\n" * 12 + "```"
    table = "| Name | Value |\n| --- | --- |\n" + "| A | B |\n" * 12
    inp = f"# Page\n\nBefore text.\n\n{code}\n\n{table}\n\nAfter text."
    chunks = pe.chunk_markdown(inp, 64, 8)
    assert any(c["text"].strip() == code for c in chunks)
    assert any(c["text"].strip() == table.strip() for c in chunks)
    assert sum("```rust" in c["text"] for c in chunks) == 1
    big = pe.chunk_markdown("```text\n" + "x" * (pe.MAX_CHUNK_CHARS + 500) + "\n", 1000, 200)
    assert len(big) > 1 and all(len(c["text"]) <= pe.MAX_CHUNK_CHARS for c in big)


def test_frontmatter_close():
    inp = "---\ntitle: Kept\nnote: ---not-a-fence\n---\n# Body"
    assert pe.extract_title(inp, "fb") == "Kept"
    assert pe.strip_frontmatter(inp) == "# Body"


def test_vector_page_id():
    assert pe.vector_page_id("wiki/nested/page.md") == "nested/page"
    assert pe.vector_page_id("wiki/notes/Foo.MD") == "notes/Foo"
    assert pe.vector_page_id("outside/page.md") is None


def test_fingerprint():
    b = dict(CFG)
    m = dict(CFG, model="model-b")
    k = dict(CFG, apiKey="rotated")
    assert pe.embedding_fingerprint("r", b) != pe.embedding_fingerprint("r", m)
    assert pe.embedding_fingerprint("r", b) == pe.embedding_fingerprint("r", k)


def test_revision_metadata(tmp_path):
    p = str(project(tmp_path))
    pe.save_revision(p, "current", "sha256:c")
    pe.save_revision(p, "other", "sha256:o")
    assert pe.load_revision(p, "current") == "sha256:c"
    pe.invalidate_page_revision(p, "current")
    assert pe.load_revision(p, "current") is None
    assert pe.load_revision(p, "other") == "sha256:o"


def test_embed_flow(tmp_path, fake_search):
    root = project(tmp_path)
    (root / "wiki/nested/page.md").write_text("---\ntitle: T\n---\n# H\n\nhello world")
    (root / "wiki/index.md").write_text("# Index")
    p = str(root)
    r = asyncio.run(pe.embed_wiki_page(p, "wiki/nested/page.md", CFG))
    assert r["status"] == "indexed" and r["pageId"] == "nested/page" and r["vectorsWritten"] == 1
    assert r["path"] == "wiki/nested/page.md" and r["revision"].startswith("sha256:")
    assert vs.vector_count_chunks(p) == 1
    assert fake_search[0][0].startswith("T\n\n# H\n\n")
    r2 = asyncio.run(pe.embed_wiki_page(p, "wiki/nested/page.md", CFG))
    assert r2["status"] == "unchanged" and r2["vectorsWritten"] == 0 and r2["chunks"] == 1
    r3 = asyncio.run(pe.embed_wiki_page(p, "wiki/nested/page.md", CFG, True))
    assert r3["status"] == "indexed"
    for bad, status in [("wiki/index.md", 400), ("wiki/none.md", 404), ("outside.md", 400)]:
        with pytest.raises(pe.PageEmbeddingError) as ei:
            asyncio.run(pe.embed_wiki_page(p, bad, CFG))
        assert ei.value.status == status
    with pytest.raises(pe.PageEmbeddingError) as ei:
        asyncio.run(pe.embed_wiki_page(p, "wiki/nested/page.md", dict(CFG, enabled=False)))
    assert ei.value.kind == pe.INVALID_REQUEST


def test_legacy_id_removed_when_unique(tmp_path, fake_search):
    root = project(tmp_path)
    (root / "wiki/nested/page.md").write_text("hello there")
    p = str(root)
    vs.vector_upsert_chunks(p, "page", [{"chunk_index": 0, "chunk_text": "x", "heading_path": "",
                                         "embedding": [1.0, 2.0, 3.0, 4.0]}])
    asyncio.run(pe.embed_wiki_page(p, "wiki/nested/page.md", CFG))
    assert vs.vector_page_revision_match(p, "page", "x") is None
    assert vs.vector_count_chunks(p) == 1  # only nested/page remains
