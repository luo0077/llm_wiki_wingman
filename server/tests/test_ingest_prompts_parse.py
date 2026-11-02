"""Port of src/lib/ingest-parse.test.ts and the pure bits of
ingest-source-path-collision.test.ts."""
from __future__ import annotations

import re

from llm_wiki_server.ingest.ingest_core import (
    build_deterministic_ingest_log,
    build_fallback_source_summary,
    build_ingest_warning_log_text,
    canonicalize_sources_field,
    extract_mineru_image_rel_paths,
    filter_truncated_file_repair_output,
    has_mineru_image_refs,
    is_app_managed_aggregate_path,
    is_safe_ingest_path,
    parse_file_blocks,
    parse_review_blocks,
    rewrite_ingest_path_from_title_for_target_language,
    should_run_dedicated_review_stage,
    source_summary_media_refs_for_external_markdown,
    stamp_generated_frontmatter_dates,
    stamp_generated_log_date,
    update_bounded_recent_index_section,
)


def J(*lines, sep="\n"):
    return sep.join(lines)


class TestCanonical:
    def test_single(self):
        r = parse_file_blocks(J("---FILE: wiki/concepts/rope.md---", "# RoPE", "Rotary positional embedding.", "---END FILE---"))
        assert r.warnings == [] and len(r.blocks) == 1
        assert r.blocks[0].path == "wiki/concepts/rope.md"
        assert "# RoPE" in r.blocks[0].content

    def test_multiple(self):
        t = J("---FILE: wiki/entities/qwen.md---", "# Qwen", "---END FILE---", "",
              "---FILE: wiki/concepts/moe.md---", "# MoE", "---END FILE---", "",
              "---FILE: wiki/sources/paper.md---", "# S", "---END FILE---")
        r = parse_file_blocks(t)
        assert r.warnings == []
        assert [b.path for b in r.blocks] == ["wiki/entities/qwen.md", "wiki/concepts/moe.md", "wiki/sources/paper.md"]

    def test_preamble_ignored(self):
        t = J("Here are the wiki files:", "", "---FILE: wiki/concepts/foo.md---", "body", "---END FILE---")
        assert len(parse_file_blocks(t).blocks) == 1


class TestCrlf:
    def test_crlf(self):
        t = J("---FILE: wiki/entities/qwen.md---", "# Qwen", "---END FILE---", "",
              "---FILE: wiki/concepts/moe.md---", "# MoE", "---END FILE---", sep="\r\n")
        r = parse_file_blocks(t)
        assert r.warnings == [] and len(r.blocks) == 2
        assert all("\r" not in b.content for b in r.blocks)

    def test_mixed(self):
        r = parse_file_blocks("---FILE: wiki/concepts/foo.md---\nline1\r\nline2\r\n---END FILE---")
        assert r.blocks[0].content == "line1\nline2"


class TestTruncation:
    def test_warns(self):
        t = J("---FILE: wiki/entities/qwen.md---", "# Qwen", "---END FILE---", "",
              "---FILE: wiki/concepts/moe.md---", "# Mixture of Exp")
        r = parse_file_blocks(t)
        assert [b.path for b in r.blocks] == ["wiki/entities/qwen.md"]
        assert len(r.warnings) == 1
        assert "wiki/concepts/moe.md" in r.warnings[0]
        assert re.search("not closed", r.warnings[0], re.I)
        assert r.truncatedPaths == ["wiki/concepts/moe.md"]

    def test_only_block_unclosed(self):
        r = parse_file_blocks("---FILE: wiki/concepts/rope.md---\n# RoPE\nIt rotates")
        assert r.blocks == [] and len(r.warnings) == 1 and "rope.md" in r.warnings[0]


def test_filter_truncated_file_repair_output():
    req = "wiki/concepts/recovered.md"
    res = filter_truncated_file_repair_output(J(
        f"---FILE: {req}---", "# First complete repair", "---END FILE---",
        f"---FILE: {req}---", "# Duplicate repair", "---END FILE---",
        "---FILE: wiki/concepts/unrequested.md---", "# Unrequested", "---END FILE---"), [req])
    assert res.paths == [req]
    assert "# First complete repair" in res.text
    assert "# Duplicate repair" not in res.text and "# Unrequested" not in res.text
    assert any(re.search("duplicate FILE block", w) for w in res.warnings)
    assert any(re.search("unrequested FILE block", w) for w in res.warnings)
    assert res.text == f"---FILE: {req}---\n# First complete repair\n---END FILE---"


class TestMarkers:
    def test_variants(self):
        assert len(parse_file_blocks(J("---FILE: wiki/concepts/foo.md---", "body", "--- END FILE ---")).blocks) == 1
        assert len(parse_file_blocks(J("---FILE: wiki/concepts/foo.md---", "body", "---end file---")).blocks) == 1
        r = parse_file_blocks(J("--- FILE: wiki/concepts/foo.md ---", "body", "---END FILE---"))
        assert r.blocks[0].path == "wiki/concepts/foo.md"
        assert len(parse_file_blocks("---FILE: wiki/concepts/foo.md---   \nbody\n---END FILE---").blocks) == 1

    def test_prose_embedded_not_closer(self):
        r = parse_file_blocks(J("---FILE: wiki/concepts/foo.md---", "Not to be written:",
                                "- `---END FILE---` in backticks (this is prose)", "real content continues", "---END FILE---"))
        assert len(r.blocks) == 1 and "real content continues" in r.blocks[0].content


class TestFences:
    def test_closer_in_fence(self):
        r = parse_file_blocks(J(
            "---FILE: wiki/concepts/ingest-format.md---", "# Ingest Format", "", "Example of a FILE block:", "",
            "```plaintext", "---FILE: wiki/path/to/page.md---", "body content", "---END FILE---", "```", "",
            "More explanation after the example.", "---END FILE---"))
        assert r.warnings == [] and len(r.blocks) == 1
        assert "```plaintext" in r.blocks[0].content
        assert "More explanation after the example." in r.blocks[0].content

    def test_multiple_fences(self):
        r = parse_file_blocks(J("---FILE: wiki/concepts/foo.md---", "```", "---END FILE---", "```", "", "prose", "",
                                "~~~", "---END FILE---", "~~~", "", "more prose", "---END FILE---"))
        assert len(r.blocks) == 1 and "more prose" in r.blocks[0].content

    def test_nested_length(self):
        r = parse_file_blocks(J("---FILE: wiki/concepts/foo.md---", "````markdown", "```", "---END FILE---", "```", "````",
                                "", "real content after the outer fence closes", "---END FILE---"))
        assert len(r.blocks) == 1 and "real content after the outer fence closes" in r.blocks[0].content

    def test_three_does_not_close_four(self):
        r = parse_file_blocks(J("---FILE: wiki/concepts/foo.md---", "````", "```", "---END FILE---", "```", "````",
                                "", "real content", "---END FILE---"))
        assert len(r.blocks) == 1 and "real content" in r.blocks[0].content


def test_empty_path_warns():
    r = parse_file_blocks("---FILE:   ---\nsome body\n---END FILE---")
    assert r.blocks == [] and len(r.warnings) > 0


class TestSafePath:
    def test_accept(self):
        for p in ("wiki/concepts/foo.md", "wiki/index.md", "wiki/sources/some-paper.md", "wiki/entities/transformer.md"):
            assert is_safe_ingest_path(p)

    def test_reject(self):
        bad = ["", "   ", "\t\n", "notes/foo.md", "foo.md", "raw/sources/leaked.md", "/etc/passwd",
               "/Users/nash_su/.ssh/authorized_keys", "/wiki/foo.md", "C:/Windows/System32/config",
               "c:\\Users\\victim\\evil.txt", "\\Users\\victim\\evil.txt", "\\\\server\\share\\file.md",
               "wiki/../etc/passwd", "wiki/concepts/../../etc/passwd", "wiki/..", "..", "wiki\\..\\etc\\passwd",
               "wiki/concepts/foo\x00.md", "wiki/concepts/foo\nbar.md", "wiki/\x07alarm.md",
               "wiki/concepts/Article: Why It Matters.md", 'wiki/concepts/quoted"name.md', "wiki/concepts/a|b.md",
               "wiki/concepts/a?b.md", "wiki/concepts/a*b.md", "wiki/concepts/a<b>.md",
               "wiki/concepts/con.md", "wiki/concepts/NUL.pdf.md", "wiki/concepts/com1.md", "wiki/concepts/LPT9.notes.md",
               "wiki/concepts/topic.", "wiki/concepts/topic ", "wiki/concepts/folder./topic.md",
               "wiki/concepts/folder /topic.md", "wiki/.hidden"]
        for p in bad:
            assert not is_safe_ingest_path(p), p

    def test_allowed_edge_cases(self):
        for p in ("wiki/concepts/qwen-2.5..notes.md", "wiki/concepts/foo..bar.md", "wiki/concepts/auxiliary.md",
                  "wiki/concepts/topic .md"):
            assert is_safe_ingest_path(p), p

    def test_non_string(self):
        assert not is_safe_ingest_path(None)


class TestTraversalEndToEnd:
    def test_dotdot(self):
        r = parse_file_blocks(J("---FILE: wiki/concepts/legit.md---", "Real page.", "---END FILE---",
                                "---FILE: ../../etc/passwd---", "attacker:x:0:0::/root:/bin/bash", "---END FILE---"))
        assert [b.path for b in r.blocks] == ["wiki/concepts/legit.md"]
        assert any("../../etc/passwd" in w for w in r.warnings) and any("unsafe path" in w for w in r.warnings)

    def test_abs_and_outside(self):
        for p in ("/etc/passwd", "src-tauri/src/main.rs"):
            r = parse_file_blocks(J(f"---FILE: {p}---", "evil", "---END FILE---"))
            assert r.blocks == [] and any("unsafe path" in w for w in r.warnings)

    def test_mixed(self):
        r = parse_file_blocks(J("---FILE: wiki/concepts/topic-a.md---", "A", "---END FILE---",
                                "---FILE: ../config.json---", '{"hijacked": true}', "---END FILE---",
                                "---FILE: wiki/entities/topic-b.md---", "B", "---END FILE---"))
        assert [b.path for b in r.blocks] == ["wiki/concepts/topic-a.md", "wiki/entities/topic-b.md"]
        assert any("../config.json" in w for w in r.warnings)


def test_media_refs():
    inp = J("![Chart](media/report/001-chart.png)", "![Table](./media/report/table%201.png)",
            '<img src="media/report/raw.png" />', "![Already relative](../media/report/keep.png)")
    assert source_summary_media_refs_for_external_markdown(inp) == J(
        "![Chart](../media/report/001-chart.png)", "![Table](../media/report/table%201.png)",
        '<img src="../media/report/raw.png" />', "![Already relative](../media/report/keep.png)")


def test_mineru_refs():
    assert has_mineru_image_refs("![chart](media/%E6%B1%A1%E6%B0%B4%20paper/mineru/images/chart%281%29.png)", "污水 paper")
    assert has_mineru_image_refs("![chart](media/污水 paper/mineru/images/chart.png)", "污水 paper")
    assert not has_mineru_image_refs("![chart](media/other/mineru/images/chart.png)", "污水 paper")
    assert has_mineru_image_refs("media/a%27b%281%29/mineru/x", "a'b(1)")
    refs = extract_mineru_image_rel_paths(
        '![a](media/s%20x/mineru/i%281%29.png) ![b](<./media/s x/mineru/b.png> "t") '
        "![c](media/s x/mineru/b.png) ![d](x.png)", "s x")
    assert refs == ["media/s x/mineru/i(1).png", "media/s x/mineru/b.png"]


def test_fallback_summary_keeps_whole_analysis():
    analysis = f"begin-{'x' * 5000}-end"
    content = build_fallback_source_summary("long.md", analysis, "2026-07-11")
    assert analysis in content
    assert content.startswith('---\ntype: source\ntitle: "Source: long.md"\ncreated: 2026-07-11\n')
    assert "(Analysis not available)" in build_fallback_source_summary("a", "", "2026-07-11")


def test_deterministic_log():
    assert build_deterministic_ingest_log("", "raw/sources/a.pdf", "2026-07-20") == "# Wiki Log\n\n## [2026-07-20] ingest | raw/sources/a.pdf\n"
    assert build_deterministic_ingest_log("# Wiki Log\n", "raw/sources/b.pdf", "2026-07-20") == "# Wiki Log\n\n## [2026-07-20] ingest | raw/sources/b.pdf\n"


def test_warning_log_text_appends():
    out = build_ingest_warning_log_text("old\n\n", "s.md", ["w1"])
    assert out.startswith("old\n\n## ") and out.endswith("| s.md\n\n1. w1\n")
    assert build_ingest_warning_log_text("", "s.md", ["w1"]).startswith("## ")


class TestDates:
    def test_stamp_existing(self):
        out = stamp_generated_frontmatter_dates(
            J("---", "type: concept", "title: Wrong Date", "created: 2024-01-01", "updated: 2025-02-02", "tags: []", "---", "# Body"),
            "2026-06-07")
        assert "created: 2026-06-07" in out and "updated: 2026-06-07" in out
        assert "2024-01-01" not in out and "2025-02-02" not in out

    def test_adds_missing(self):
        out = stamp_generated_frontmatter_dates(J("---", "type: concept", "title: Missing Date", "tags: []", "---", "# Body"), "2026-06-07")
        assert "created: 2026-06-07" in out and "updated: 2026-06-07" in out

    def test_noop_without_frontmatter(self):
        assert stamp_generated_frontmatter_dates("no fm", "2026-01-02") == "no fm"

    def test_crlf_frontmatter(self):
        # Appended keys use "\n" (as in TS); the CRLF around the block is kept.
        out = stamp_generated_frontmatter_dates("---\r\ntype: x\r\nCreated : 1\r\n---\r\nbody", "2026-01-02")
        assert out == "---\r\ntype: x\r\nCreated : 2026-01-02\nupdated: 2026-01-02\r\n---\r\nbody"

    def test_log(self):
        assert stamp_generated_log_date("## [2024-01-01] ingest | Foo", "2026-06-07") == "## [2026-06-07] ingest | Foo"
        assert stamp_generated_log_date("## [YYYY-MM-DD] ingest | Foo", "2026-06-07") == "## [2026-06-07] ingest | Foo"


class TestRewrite:
    @staticmethod
    def _page(title, body):
        return J("---", "type: concept", f"title: {title}", "created: 2026-06-18", "---", "", f"# {title}", "", body)

    def test_cjk_target(self):
        assert rewrite_ingest_path_from_title_for_target_language(
            "wiki/concepts/denitrifying-phosphorus-removal.md", self._page("反硝化除磷技术", "正文。"), "Chinese"
        ) == "wiki/concepts/反硝化除磷技术.md"

    def test_auto_detects_cjk_title(self):
        assert rewrite_ingest_path_from_title_for_target_language(
            "wiki/concepts/denitrifying-phosphorus-removal.md", self._page("反硝化除磷技术", "这是一段中文正文，用于检测语言。"), "auto"
        ) == "wiki/concepts/反硝化除磷技术.md"

    def test_english_untouched(self):
        p = "wiki/concepts/denitrifying-phosphorus-removal.md"
        content = J("---", "type: concept", "title: Denitrifying phosphorus removal", "---", "", "# Denitrifying phosphorus removal", "", "This is English.")
        assert rewrite_ingest_path_from_title_for_target_language(p, content, "auto") == p

    def test_ascii_body_dominates(self):
        sql = "\n".join(f"CREATE TABLE audit_{i} (id BIGINT PRIMARY KEY, event_type VARCHAR(64));" for i in range(30))
        content = J("---", "type: concept", "title: 审计数据模型", "---", "", "# 审计数据模型", "", "```sql", sql, "```")
        assert rewrite_ingest_path_from_title_for_target_language(
            "wiki/concepts/audit-data-model.md", content, "auto") == "wiki/concepts/审计数据模型.md"

    def test_no_rewrite_sources_or_aggregate(self):
        content = "---\ntitle: 反硝化除磷技术\n---\n# 反硝化除磷技术"
        assert rewrite_ingest_path_from_title_for_target_language("wiki/sources/source-slug.md", content, "Chinese") == "wiki/sources/source-slug.md"
        assert rewrite_ingest_path_from_title_for_target_language("wiki/index.md", content, "Chinese") == "wiki/index.md"

    def test_heading_fallback_and_cjk_filename_kept(self):
        assert rewrite_ingest_path_from_title_for_target_language(
            "wiki/concepts/foo.md", "no fm\n# 标题 一\r\nx", "auto") == "wiki/concepts/标题-一.md"
        assert rewrite_ingest_path_from_title_for_target_language(
            "wiki/concepts/中文.md", self._page("标题", "x"), None) == "wiki/concepts/中文.md"


class TestSources:
    def test_removes_unsafe(self):
        content = J("---", "title: Entity",
                    'sources: ["wiki/log.md", "wiki/index.md", ".llm-wiki/state.json", "/tmp/secret.md", "../escape.md", "raw/sources/folder/source.md"]',
                    "---", "# Entity")
        r = canonicalize_sources_field(content, "folder/source.md")
        assert 'sources: ["folder/source.md"]' in r
        for s in ("wiki/log.md", ".llm-wiki", "/tmp/secret.md", "../escape.md"):
            assert s not in r

    def test_wiki_named_subfolder(self):
        content = '---\ntitle: Notes\nsources: ["raw/sources/wiki/notes.md"]\n---\n# Notes'
        assert 'sources: ["wiki/notes.md"]' in canonicalize_sources_field(content, "wiki/notes.md")

    def test_no_frontmatter_unchanged(self):
        assert canonicalize_sources_field("no fm", "b.md") == "no fm"


class TestAggregate:
    def test_variants(self):
        assert is_app_managed_aggregate_path("wiki/INDEX.md")
        assert is_app_managed_aggregate_path("wiki\\overview.MD")
        assert not is_app_managed_aggregate_path("wiki/entities/index.md")

    def test_bounded(self):
        existing = "\n".join(["# Wiki Index", "", "## Recently Updated",
                              *[f"- [[old-{i}]] — Old {i}" for i in range(205)], "", "## Other", "Keep me"])
        result = update_bounded_recent_index_section(existing, ["- [[new]] — New"])
        recent = result.split("## Recently Updated")[1].split("## Other")[0]
        assert len(re.findall(r"^- \[\[", recent, re.M)) == 200
        assert "[[new]]" in recent
        assert "## Other\nKeep me" in result

    def test_no_section_and_dedupe(self):
        assert update_bounded_recent_index_section("# Wiki\n\nfoo", ["- [[n]]"]) == "# Wiki\n\nfoo\n\n## Recently Updated\n- [[n]]\n"
        out = update_bounded_recent_index_section("## Recently Updated\n- [[a]]\n- [[b]]", ["- [[b]]"])
        assert out == "\n## Recently Updated\n- [[b]]\n- [[a]]\n"


def test_review_blocks():
    items = parse_review_blocks(
        "---REVIEW: Suggestion | Title A---\ndesc\nOPTIONS: Create Page | Skip\nPAGES: a, b\nSEARCH: q1 | q2 |\n---END REVIEW---\n"
        "---REVIEW: weird | T---\nbody\n---END REVIEW---", "s.md")
    assert items[0] == {
        "type": "suggestion", "title": "Title A", "description": "desc", "sourcePath": "s.md",
        "affectedPages": ["a", "b"], "searchQueries": ["q1", "q2"],
        "options": [{"label": "Create Page", "action": "Create Page"}, {"label": "Skip", "action": "Skip"}],
    }
    assert items[1]["type"] == "confirm"
    assert items[1]["options"] == [{"label": "Approve", "action": "Approve"}, {"label": "Skip", "action": "Skip"}]
    assert "affectedPages" not in items[1] and "searchQueries" not in items[1]


def test_dedicated_review_stage_trigger():
    assert not should_run_dedicated_review_stage("short")
    assert should_run_dedicated_review_stage("x" * 10_000)
    assert should_run_dedicated_review_stage("---REVIEW: suggestion | T---")
    assert should_run_dedicated_review_stage("".join(f"---FILE: wiki/p{i}.md---" for i in range(4)))
