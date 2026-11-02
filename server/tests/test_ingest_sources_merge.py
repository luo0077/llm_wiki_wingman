"""Translation of src/lib/sources-merge.test.ts."""
from __future__ import annotations

import re

import pytest

from llm_wiki_server.ingest.sources_merge import (
    merge_array_fields_into_content,
    merge_sources_into_content,
    merge_sources_lists,
    parse_frontmatter_array,
    parse_sources,
    write_sources,
)


def WRAP(fm, body="body\n"):
    return f"---\n{fm}\n---\n{body}"


@pytest.mark.parametrize(
    "fm,expected",
    [
        ('sources: ["a.md"]', ["a.md"]),
        ('sources: ["a.md", "b.md"]', ["a.md", "b.md"]),
        ("sources: ['a.md', 'b.md']", ["a.md", "b.md"]),
        ("sources: [a.md, b.md]", ["a.md", "b.md"]),
        ('sources: ["a.md","b.md"]', ["a.md", "b.md"]),
        ("sources: []", []),
        ("title: X", []),
        ('sources: ["测试.md", "test.md"]', ["测试.md", "test.md"]),
        ('sources: ["reports/Q1, revised.pdf", "测试，最终版.md"]', ["reports/Q1, revised.pdf", "测试，最终版.md"]),
    ],
)
def test_parse_sources_inline(fm, expected):
    assert parse_sources(WRAP(fm)) == expected


def test_parse_sources_no_frontmatter():
    assert parse_sources("# heading\n\nbody") == []


def test_parse_sources_crlf():
    content = '---\r\ntitle: Windows\r\nsources: ["BookA.pdf", "BookB.pdf"]\r\n---\r\n# Body\r\n'
    assert parse_sources(content) == ["BookA.pdf", "BookB.pdf"]


def test_round_trips_quotes_and_backslashes():
    after = write_sources(WRAP("title: X"), ['drafts/"quoted".md', "folder\\note.md"])
    assert 'sources: ["drafts/\\"quoted\\".md", "folder\\\\note.md"]' in after
    assert parse_sources(after) == ['drafts/"quoted".md', "folder\\note.md"]


def test_multiline_list():
    assert parse_sources(WRAP("\n".join(["sources:", "  - a.md", "  - b.md"]))) == ["a.md", "b.md"]


def test_quoted_multiline_entries():
    assert parse_sources(WRAP("\n".join(["sources:", '  - "a.md"', "  - 'b.md'"]))) == ["a.md", "b.md"]


# ── writeSources ──────────────────────────────────────────────────

def test_write_preserves_crlf():
    before = '---\r\ntitle: Windows\r\nsources: ["a.md"]\r\n---\r\n# Body\r\n'
    after = write_sources(before, ["a.md", "b.md"])
    assert '---\r\ntitle: Windows\r\nsources: ["a.md", "b.md"]\r\n---\r\n' in after
    assert parse_sources(after) == ["a.md", "b.md"]


def test_write_replaces_inline():
    after = write_sources(WRAP('title: X\nsources: ["a.md"]'), ["a.md", "b.md"])
    assert parse_sources(after) == ["a.md", "b.md"]
    assert "title: X" in after


def test_write_preserves_field_order():
    after = write_sources(WRAP('title: X\nsources: ["a.md"]\ntags: []'), ["a.md", "b.md"])
    fm = re.match(r"^---\n([\s\S]*?)\n---", after).group(1)
    assert [l.split(":")[0].strip() for l in fm.split("\n")] == ["title", "sources", "tags"]


def test_write_inserts_when_absent():
    after = write_sources(WRAP("title: X\ntags: []"), ["a.md"])
    assert parse_sources(after) == ["a.md"]
    assert "title: X" in after
    assert "tags: []" in after


def test_write_collapses_block_form():
    after = write_sources(WRAP("\n".join(["title: X", "sources:", "  - a.md", "  - b.md"])), ["a.md", "b.md", "c.md"])
    assert parse_sources(after) == ["a.md", "b.md", "c.md"]
    assert 'sources: ["a.md", "b.md", "c.md"]' in after
    assert not re.search(r"^\s+-\s+a\.md", after, re.M)


def test_write_preserves_following_field_lf():
    before = WRAP("\n".join(["title: X", "sources:", "  - a.md", "  - b.md", "tags: [keep]"]))
    after = write_sources(before, ["a.md", "b.md", "c.md"])
    assert 'sources: ["a.md", "b.md", "c.md"]\ntags: [keep]' in after


def test_write_preserves_following_field_crlf():
    before = "\r\n".join(["---", "title: Windows", "sources:", "  - a.md", "  - b.md", "tags: [keep]", "---", "# Body", ""])
    after = write_sources(before, ["a.md", "b.md", "c.md"])
    assert 'sources: ["a.md", "b.md", "c.md"]\r\ntags: [keep]' in after


def test_write_no_frontmatter_unchanged():
    before = "# heading\n\nbody"
    assert write_sources(before, ["a.md"]) == before


def test_write_preserves_body():
    after = write_sources(WRAP('title: X\nsources: ["a.md"]', "# Title\n\nLots of prose here.\nAnd more prose."), ["a.md", "b.md"])
    assert "# Title" in after and "Lots of prose here." in after and "And more prose." in after


def test_write_js_replacement_pattern_quirk():
    # JS `String.replace(re, "...$$...")` expands `$$` to `$`; the port reproduces that.
    after = write_sources(WRAP('sources: ["a"]'), ["x$$y"])
    assert 'sources: ["x$y"]' in after


def test_merge_array_fields_crlf_existing():
    existing = "\r\n".join(["---", "title: Windows", 'sources: ["old.pdf"]', "tags: [existing]", "---", "# Existing", ""])
    incoming = WRAP("\n".join(["title: Windows", 'sources: ["new.pdf"]', "tags: [incoming]"]), "# Updated\n")
    merged = merge_array_fields_into_content(incoming, existing, ["sources", "tags"])
    assert parse_frontmatter_array(merged, "sources") == ["old.pdf", "new.pdf"]
    assert parse_frontmatter_array(merged, "tags") == ["existing", "incoming"]
    assert "# Updated" in merged


# ── mergeSourcesLists ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (["a.md"], ["b.md"], ["a.md", "b.md"]),
        (["a.md"], ["a.md"], ["a.md"]),
        (["Test.md"], ["test.md"], ["Test.md"]),
        (["b.md", "a.md"], ["c.md"], ["b.md", "a.md", "c.md"]),
        ([], [], []),
        (["a.md", "b.md"], [], ["a.md", "b.md"]),
        ([], ["a.md", "b.md"], ["a.md", "b.md"]),
    ],
)
def test_merge_sources_lists(a, b, expected):
    assert merge_sources_lists(a, b) == expected


# ── mergeSourcesIntoContent ───────────────────────────────────────

def test_new_page_unchanged():
    nc = WRAP('sources: ["test2.md"]')
    assert merge_sources_into_content(nc, None) == nc


def test_merges_different_source():
    merged = merge_sources_into_content(WRAP('sources: ["test2.md"]', "fresh body from re-ingest"), WRAP('sources: ["test1.md"]', "old body"))
    assert parse_sources(merged) == ["test1.md", "test2.md"]
    assert "fresh body from re-ingest" in merged
    assert "old body" not in merged


def test_idempotent_subset():
    merged = merge_sources_into_content(WRAP('sources: ["test1.md"]'), WRAP('sources: ["test1.md", "test2.md"]'))
    assert parse_sources(merged) == ["test1.md", "test2.md"]


def test_short_circuit_returns_new_content():
    incoming = WRAP('sources: ["a.md"]', "body content")
    assert merge_sources_into_content(incoming, WRAP('sources: ["a.md"]')) == incoming


def test_case_insensitive_keeps_original_casing():
    merged = merge_sources_into_content(WRAP('sources: ["test1.md"]'), WRAP('sources: ["Test1.md"]'))
    assert parse_sources(merged) == ["Test1.md"]


def test_appends_to_multi_source():
    merged = merge_sources_into_content(WRAP('sources: ["c.md"]'), WRAP('sources: ["a.md", "b.md"]'))
    assert parse_sources(merged) == ["a.md", "b.md", "c.md"]


def test_existing_without_sources_noop():
    incoming = WRAP('sources: ["test2.md"]')
    assert merge_sources_into_content(incoming, WRAP("title: X")) == incoming


def test_existing_empty_string_noop():
    incoming = WRAP('sources: ["test2.md"]')
    assert merge_sources_into_content(incoming, "") == incoming


def test_new_missing_sources_keeps_existing():
    merged = merge_sources_into_content(WRAP("title: X"), WRAP('sources: ["test1.md"]'))
    assert parse_sources(merged) == ["test1.md"]


def test_existing_without_frontmatter():
    incoming = WRAP('sources: ["test2.md"]')
    assert merge_sources_into_content(incoming, "# legacy non-frontmatter page") == incoming


def test_regression_union_not_last_wins():
    first = merge_sources_into_content(WRAP('title: Attention\nsources: ["test1.md"]', "content from test1"), None)
    assert parse_sources(first) == ["test1.md"]
    second = merge_sources_into_content(WRAP('title: Attention\nsources: ["test2.md"]', "content from test2"), first)
    assert parse_sources(second) == ["test1.md", "test2.md"]
    assert [s for s in parse_sources(second) if s.lower() != "test2.md"] == ["test1.md"]


def test_three_way_reingest():
    page = merge_sources_into_content(WRAP('sources: ["a.md"]', "body v1"), None)
    page = merge_sources_into_content(WRAP('sources: ["b.md"]', "body v2"), page)
    page = merge_sources_into_content(WRAP('sources: ["c.md"]', "body v3"), page)
    assert parse_sources(page) == ["a.md", "b.md", "c.md"]
    assert "body v3" in page and "body v1" not in page and "body v2" not in page


# ── parseFrontmatterArray ─────────────────────────────────────────

def test_pfa_inline():
    assert parse_frontmatter_array(WRAP("tags: [a, b, c]"), "tags") == ["a", "b", "c"]


def test_pfa_block():
    assert parse_frontmatter_array(WRAP("related:\n  - foo\n  - bar"), "related") == ["foo", "bar"]


def test_pfa_strips_quotes():
    assert parse_frontmatter_array(WRAP('tags: ["a", "b"]'), "tags") == ["a", "b"]


def test_pfa_missing():
    assert parse_frontmatter_array(WRAP("title: X"), "tags") == []
    assert parse_frontmatter_array("# plain markdown", "tags") == []


def test_pfa_prefix_not_confused():
    c = WRAP("related: [a, b]")
    assert parse_frontmatter_array(c, "relate") == []
    assert parse_frontmatter_array(c, "related") == ["a", "b"]


# ── mergeArrayFieldsIntoContent ───────────────────────────────────

def test_maf_union():
    existing = WRAP('sources: ["a.md"]\ntags: [old-tag]\nrelated: [old-rel]')
    incoming = WRAP('sources: ["b.md"]\ntags: [new-tag]\nrelated: [new-rel]', "new body")
    merged = merge_array_fields_into_content(incoming, existing, ["sources", "tags", "related"])
    assert parse_frontmatter_array(merged, "sources") == ["a.md", "b.md"]
    assert parse_frontmatter_array(merged, "tags") == ["old-tag", "new-tag"]
    assert parse_frontmatter_array(merged, "related") == ["old-rel", "new-rel"]
    assert "new body" in merged


def test_maf_case_insensitive():
    merged = merge_array_fields_into_content(WRAP("tags: [foo, bar, baz]"), WRAP("tags: [Foo, BAR]"), ["tags"])
    assert parse_frontmatter_array(merged, "tags") == ["Foo", "BAR", "baz"]


def test_maf_preserves_field_missing_from_new():
    merged = merge_array_fields_into_content(WRAP("title: X"), WRAP("tags: [persistent]"), ["tags"])
    assert parse_frontmatter_array(merged, "tags") == ["persistent"]


def test_maf_noop():
    incoming = WRAP("title: Y")
    assert merge_array_fields_into_content(incoming, WRAP("title: X"), ["sources", "tags"]) == incoming


def test_maf_null_existing():
    incoming = WRAP("tags: [a]\nrelated: [b]")
    assert merge_array_fields_into_content(incoming, None, ["tags", "related"]) == incoming


def test_maf_only_requested_fields():
    merged = merge_array_fields_into_content(WRAP("sources: [b.md]\ntags: [new]"), WRAP("sources: [a.md]\ntags: [old]"), ["sources"])
    assert parse_frontmatter_array(merged, "sources") == ["a.md", "b.md"]
    assert parse_frontmatter_array(merged, "tags") == ["new"]


def test_maf_matches_sources_wrapper():
    existing = WRAP('sources: ["a.md"]')
    incoming = WRAP('sources: ["b.md"]', "new body")
    a = merge_sources_into_content(incoming, existing)
    b = merge_array_fields_into_content(incoming, existing, ["sources"])
    assert parse_sources(b) == parse_sources(a)
    assert "new body" in b
