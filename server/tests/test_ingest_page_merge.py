"""Translation of src/lib/page-merge.test.ts."""
from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki_server.ingest.page_merge import MergePageOptions, merge_page_content, strip_merge_scaffolding


def PAGE(fm: str, body: str) -> str:
    return f"---\n{fm}\n---\n\n{body}"


def base_opts(**kw):
    return MergePageOptions(sourceFileName="doc-B.pdf", pagePath="wiki/entities/foo.md", today=lambda: "2026-04-30", **kw)


def run(*args, **kw):
    return asyncio.run(merge_page_content(*args, **kw))


def merger_returning(value):
    return AsyncMock(return_value=value)


# ── fast paths ────────────────────────────────────────────────────

def test_new_page_returns_new_content():
    merger = AsyncMock()
    out = run(PAGE('type: entity\ntitle: Foo\nsources: ["doc.pdf"]', "body"), None, merger, base_opts())
    assert 'sources: ["doc.pdf"]' in out
    merger.assert_not_called()


def test_identical_returns_existing():
    merger = AsyncMock()
    c = PAGE("type: entity\ntitle: Foo", "body")
    assert run(c, c, merger, base_opts()) == c
    merger.assert_not_called()


def test_repairs_citations_on_identical_page():
    merger = AsyncMock()
    content = PAGE("type: entity\ntitle: Foo", "Evidence [[1]].")
    out = run(content, content, merger, base_opts())
    assert "Evidence [1]." in out
    merger.assert_not_called()


def test_skips_llm_when_bodies_identical():
    merger = AsyncMock()
    existing = PAGE('type: entity\ntitle: Foo\nsources: ["a.pdf"]', "same body")
    incoming = PAGE('type: entity\ntitle: Foo\nsources: ["b.pdf"]', "same body")
    out = run(incoming, existing, merger, base_opts())
    assert 'sources: ["a.pdf", "b.pdf"]' in out
    assert "same body" in out
    merger.assert_not_called()


# ── corrected single-source replacement ───────────────────────────

def test_replace_existing_body():
    existing = PAGE(
        'type: entity\ntitle: Stable title\ncreated: 2025-01-01\nupdated: 2025-01-02\nsources: ["doc.pdf"]\ntags: [manual]\nrelated: [kept-link]',
        "obsolete source wording plus a manual-era body",
    )
    incoming = PAGE(
        'type: concept\ntitle: Changed title\ncreated: 2026-01-01\nsources: ["doc.pdf"]\ntags: [generated]\nrelated: [new-link]',
        "corrected source wording",
    )
    merger = AsyncMock()
    backup = AsyncMock()
    out = run(incoming, existing, merger, base_opts(replaceExistingBody=True, backup=backup))
    assert "corrected source wording" in out
    assert "obsolete source wording" not in out
    assert "type: entity" in out
    assert "title: Stable title" in out
    assert "created: 2025-01-01" in out
    assert "updated: 2026-04-30" in out
    assert re.search(r'tags:\s*\[\s*"manual",\s*"generated"\s*\]', out)
    assert re.search(r'related:\s*\[\s*"kept-link",\s*"new-link"\s*\]', out)
    backup.assert_called_once_with(existing)
    merger.assert_not_called()


def test_normal_merge_for_page_with_another_source():
    existing = PAGE('type: entity\ntitle: Shared\nsources: ["doc.pdf", "other.pdf"]', "other source contribution")
    incoming = PAGE('type: entity\ntitle: Shared\nsources: ["doc.pdf"]', "corrected doc contribution")
    merger = merger_returning(
        PAGE(
            'type: entity\ntitle: Shared\nsources: ["doc.pdf", "other.pdf"]',
            "other source contribution and corrected doc contribution retained together",
        )
    )
    out = run(incoming, existing, merger, base_opts())
    merger.assert_called_once()
    assert "other source contribution" in out
    assert "corrected doc contribution" in out


# ── LLM merge ─────────────────────────────────────────────────────

def test_llm_merge_happy_path():
    existing = PAGE(
        'type: entity\ntitle: Accumulibacter\ncreated: 2026-04-09\ntags: [microbiology, ebpr]\nrelated: [dpao, vfa]\nsources: ["doc-A.pdf"]',
        "## Anaerobic Phase\n\nDescription from doc A.\n\n## Denitrification\n\nMore from doc A.",
    )
    incoming = PAGE(
        'type: entity\ntitle: Accumulibacter\ncreated: 2026-04-30\ntags: [paos, propionate]\nrelated: [pha]\nsources: ["doc-B.pdf"]',
        "## Carbon Source Preferences\n\nDescription from doc B.\n\n## Acetate vs Propionate\n\nMore from doc B.",
    )
    merged_body = "## Anaerobic Phase\n\nDescription from doc A.\n\n## Denitrification\n\nMore from doc A.\n\n## Carbon Source Preferences\n\nDescription from doc B.\n\n## Acetate vs Propionate\n\nMore from doc B."
    merger = merger_returning(
        PAGE(
            'type: entity\ntitle: Accumulibacter\ncreated: 2026-04-09\ntags: [paos, propionate]\nrelated: [pha]\nsources: ["doc-B.pdf"]',
            merged_body,
        )
    )
    out = run(incoming, existing, merger, base_opts())
    merger.assert_called_once()
    # merger receives (existing, array-merged incoming, sourceFileName, signal)
    args = merger.call_args[0]
    assert args[0] == existing and args[2] == "doc-B.pdf" and args[3] is None
    assert "Anaerobic Phase" in out
    assert "Carbon Source Preferences" in out
    assert "title: Accumulibacter" in out
    assert "created: 2026-04-09" in out
    assert "type: entity" in out
    assert "updated: 2026-04-30" in out
    assert re.search(r'sources:\s*\[\s*"doc-A.pdf",\s*"doc-B.pdf"\s*\]', out)
    assert re.search(r'tags:\s*\[\s*"microbiology",\s*"ebpr",\s*"paos",\s*"propionate"\s*\]', out)
    assert re.search(r'related:\s*\[\s*"dpao",\s*"vfa",\s*"pha"\s*\]', out)


def test_sync_merger_is_supported():
    existing = PAGE("type: entity\ntitle: Foo", "old body content here")
    incoming = PAGE("type: entity\ntitle: Foo", "very different new body here")
    merger = MagicMock(return_value=PAGE("type: entity\ntitle: Foo", "merged body that is reasonably long enough to pass the threshold check"))
    out = run(incoming, existing, merger, base_opts())
    assert "merged body that is reasonably long" in out


def test_locked_title_preserved():
    existing = PAGE("type: entity\ntitle: Accumulibacter", "old body content here")
    incoming = PAGE("type: entity\ntitle: Accumulibacter", "very different new body here")
    merger = merger_returning(
        PAGE("type: entity\ntitle: ACCUMULIBACTER (renamed)", "merged body that is reasonably long enough to pass the threshold check")
    )
    out = run(incoming, existing, merger, base_opts())
    assert "title: Accumulibacter" in out
    assert "ACCUMULIBACTER (renamed)" not in out


def test_locked_type_preserved():
    existing = PAGE("type: entity\ntitle: Foo", "original body content")
    incoming = PAGE("type: entity\ntitle: Foo", "new content from another source")
    merger = merger_returning(
        PAGE("type: concept\ntitle: Foo", "merged body that is long enough to clear the seventy percent threshold")
    )
    out = run(incoming, existing, merger, base_opts())
    assert "type: entity" in out
    assert "type: concept" not in out


def _simple(existing_body, incoming_body, merged_body, **opts):
    existing = PAGE("type: entity\ntitle: Foo", existing_body)
    incoming = PAGE("type: entity\ntitle: Foo", incoming_body)
    merger = merger_returning(PAGE("type: entity\ntitle: Foo", merged_body))
    return run(incoming, existing, merger, base_opts(**opts))


def test_normalizes_malformed_citations():
    out = _simple(
        "Existing detailed evidence for the topic.",
        "Incoming detailed evidence for the topic.",
        "Existing and incoming evidence are retained [[1]] with sources [[2], [3]].",
    )
    assert "retained [1] with sources [2], [3]." in out


def test_strips_internal_version_labels():
    out = _simple(
        "Existing detailed evidence for the topic.",
        "Incoming detailed evidence for the topic.",
        "## Classification（新生成版本）\n\nRetained evidence.\n\n*（来源：现有版本）*",
    )
    assert "## Classification" in out
    assert not re.search("新生成版本|现有版本", out)


def test_removes_invented_example_urls():
    backup = AsyncMock()
    out = _simple(
        "Existing detailed evidence for the topic.",
        "Incoming detailed evidence for the topic.",
        "Existing and incoming evidence retained in detail.\n\n## References\n- https://example.com/version1",
        backup=backup,
    )
    assert "Existing and incoming evidence retained in detail." in out
    assert "example.com" not in out
    backup.assert_not_called()


def test_removes_only_invented_url_range():
    legitimate = "https://example.com/docs/v1/guide"
    out = _simple(
        f"Existing evidence at {legitimate}.",
        "Incoming detailed evidence for the topic.",
        f"Existing evidence at {legitimate}.\n\nIncoming detailed evidence.\n\n[Invented citation](https://example.com/docs/v1)",
    )
    assert legitimate in out
    assert "Invented citation" in out
    assert "[Invented citation]()" not in out
    assert "https://example.com/docs/v1)" not in out


def test_lookalike_domains_preserved():
    out = _simple(
        "Existing detailed evidence for the topic.",
        "Incoming detailed evidence for the topic.",
        "\n".join(
            [
                "Existing and incoming evidence retained in detail.",
                "https://notexample.com/reference",
                "https://example.com.evil.test/reference",
                "<https://docs.example.org/invented>",
            ]
        ),
    )
    assert "https://notexample.com/reference" in out
    assert "https://example.com.evil.test/reference" in out
    assert "docs.example.org" not in out
    assert "<>" not in out


def test_preserves_version_headings_from_source_material():
    out = _simple(
        "## 旧版本\n\nHistorical product edition details.",
        "Incoming detailed evidence for the topic.",
        "## 旧版本\n\nHistorical product edition details.\n\n## Analysis（新生成版本）\n\nIncoming detailed evidence.",
    )
    assert "## 旧版本" in out
    assert "## Analysis\n" in out
    assert "Analysis（新生成版本）" not in out


def test_preserves_version_headings_when_depth_changes():
    out = _simple(
        "## iPhone 12 (old version)\n\nHistorical product details.",
        "Incoming detailed evidence for the topic.",
        "### iPhone 12 (old version)\n\nHistorical product details.\n\nIncoming detailed evidence.",
    )
    assert "### iPhone 12 (old version)" in out


def test_cleans_markdown_url_forms_but_not_fenced():
    out = _simple(
        "Existing detailed evidence for the topic.",
        "Incoming detailed evidence for the topic.",
        "\n".join(
            [
                "Existing and incoming evidence retained in detail.",
                "![diagram](https://example.com/image.png)",
                '[citation](https://example.org/page "Title")',
                "[1]: https://example.net/reference",
                "- https://example.com/bare",
                "```sh",
                "curl https://api.example.com/example",
                "```",
            ]
        ),
    )
    assert "diagram" in out
    assert "citation" in out
    assert "!diagram" not in out
    assert "[citation](" not in out
    assert not re.search(r"^\[1\]:\s*$", out, re.M)
    assert not re.search(r"^[-*+]\s*$", out, re.M)
    assert "curl https://api.example.com/example" in out


def test_strips_directory_prefixes_from_wikilinks_only():
    merged_body = "\n".join(
        [
            "Existing and incoming notes are retained with additional context.",
            "See [[clients/foo-overview]], [[concepts/bar#Details|Bar details]], and [[windows\\baz]].",
            "Keep the ordinary path clients/foo-overview unchanged.",
            "Keep the embed ![[media/charts/plot.png]] unchanged.",
            "Keep inline code `[[examples/inline-link]]` unchanged.",
            "Keep multi-backtick code ``[[examples/multi-backtick]]`` unchanged.",
            "```md",
            "[[examples/fenced-link]]",
            "```not-a-closing-fence",
            "[[examples/still-fenced-link]]",
            "```",
            "~~~md",
            "[[examples/tilde-fenced-link]]",
            "~~~",
            "    [[examples/indented-code-link]]",
            "Keep escaped \\[[examples/escaped-link]] unchanged.",
            "Keep URI-like [[https://developer.mozilla.org/wiki/page]] targets unchanged.",
            "Keep attachment [[attachments/report.pdf]] targets unchanged.",
        ]
    )
    out = _simple(
        "Existing notes link to [[clients/foo-overview]] and contain enough detail for merging.",
        "Incoming notes add a second detailed observation about the same subject.",
        merged_body,
    )
    assert "[[foo-overview]]" in out
    assert "[[bar#Details|Bar details]]" in out
    assert "[[baz]]" in out
    assert "[[clients/foo-overview]]" not in out
    assert "ordinary path clients/foo-overview unchanged" in out
    assert "![[media/charts/plot.png]]" in out
    assert "`[[examples/inline-link]]`" in out
    assert "``[[examples/multi-backtick]]``" in out
    assert "[[examples/fenced-link]]" in out
    assert "[[examples/still-fenced-link]]" in out
    assert "[[examples/tilde-fenced-link]]" in out
    assert "    [[examples/indented-code-link]]" in out
    assert "\\[[examples/escaped-link]]" in out
    assert "[[https://developer.mozilla.org/wiki/page]]" in out
    assert "[[attachments/report.pdf]]" in out


# ── stripMergeScaffolding ─────────────────────────────────────────

def test_strip_scaffolding_removes_notes_and_labels():
    inp = "\n".join(["## 分类标准（新生成版本）", "", "事实内容。", "", "*（来源：原始版本）*", "", "### GAP-3 — 旧版本"])
    assert strip_merge_scaffolding(inp) == "\n".join(["## 分类标准", "", "事实内容。", "", "", "### GAP-3"])


@pytest.mark.parametrize(
    "inp",
    [
        "The threshold (threshold = 0.5) remains factual.",
        "Values (gold standard) hold.",
        "The model (original paper by Smith) reported 90%.",
        "该方法（原始数据集）准确率 90%。",
        "[[page (old version)]]",
        "```js\nfoo(old_value)\n```",
        "---\ntitle: Foo (old)\nsources: [report (original).pdf]\n---\nBody",
        "## 旧版本",
    ],
)
def test_strip_scaffolding_preserves_content(inp):
    assert strip_merge_scaffolding(inp) == inp


# ── LLM failure fallback ──────────────────────────────────────────

def test_falls_back_when_llm_throws():
    existing = PAGE('type: entity\ntitle: Foo\ntags: [old]\nsources: ["a.pdf"]', "old body content")
    incoming = PAGE('type: entity\ntitle: Foo\ntags: [new]\nsources: ["b.pdf"]', "new body content")
    merger = AsyncMock(side_effect=Exception("LLM rate limited"))
    out = run(incoming, existing, merger, base_opts())
    assert re.search(r'tags:\s*\[\s*"old",\s*"new"\s*\]', out)
    assert re.search(r'sources:\s*\[\s*"a.pdf",\s*"b.pdf"\s*\]', out)
    assert "new body content" in out


def test_rejects_shrunk_body():
    long_body = "long body content " * 200
    existing = PAGE("type: entity\ntitle: Foo", long_body)
    incoming = PAGE("type: entity\ntitle: Foo", "incoming body that is also pretty long " + long_body)
    merger = merger_returning(PAGE("type: entity\ntitle: Foo", "tiny merged body"))
    out = run(incoming, existing, merger, base_opts())
    assert "tiny merged body" not in out
    assert "incoming body that is also pretty long" in out


def test_rejects_output_without_frontmatter():
    existing = PAGE("type: entity\ntitle: Foo", "old body content here")
    incoming = PAGE("type: entity\ntitle: Foo", "new body content here")
    merger = merger_returning("raw markdown with no frontmatter at all and definitely no opening triple-dash")
    out = run(incoming, existing, merger, base_opts())
    assert out.startswith("---")
    assert "new body content here" in out


def test_backup_called_on_fallback():
    existing = PAGE("type: entity\ntitle: Foo", "old body")
    incoming = PAGE("type: entity\ntitle: Foo", "new body")
    backup = AsyncMock()
    merger = AsyncMock(side_effect=Exception("network error"))
    run(incoming, existing, merger, base_opts(backup=backup))
    backup.assert_called_once_with(existing)


def test_backup_not_called_on_success():
    existing = PAGE("type: entity\ntitle: Foo", "old body")
    incoming = PAGE("type: entity\ntitle: Foo", "new body content")
    backup = AsyncMock()
    merger = merger_returning(PAGE("type: entity\ntitle: Foo", "merged body that is long enough to clear the threshold check"))
    run(incoming, existing, merger, base_opts(backup=backup))
    backup.assert_not_called()


def test_backup_failure_swallowed():
    existing = PAGE("type: entity\ntitle: Foo", "old body")
    incoming = PAGE("type: entity\ntitle: Foo", "new body content")
    backup = AsyncMock(side_effect=Exception("disk full"))
    merger = AsyncMock(side_effect=Exception("network error"))
    out = run(incoming, existing, merger, base_opts(backup=backup))
    assert "new body content" in out


def test_sync_backup_supported():
    existing = PAGE("type: entity\ntitle: Foo", "old body")
    incoming = PAGE("type: entity\ntitle: Foo", "new body")
    backup = MagicMock()
    merger = AsyncMock(side_effect=Exception("x"))
    run(incoming, existing, merger, base_opts(backup=backup))
    backup.assert_called_once_with(existing)
