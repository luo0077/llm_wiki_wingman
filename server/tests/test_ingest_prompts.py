"""Port of src/lib/ingest.prompt.test.ts plus pure-helper parity checks."""
from __future__ import annotations

from datetime import datetime, timezone

from llm_wiki_server.ingest.ingest_core import (
    build_analysis_prompt,
    build_chunk_analysis_system_prompt,
    build_generation_prompt,
    build_page_merge_system_prompt,
    build_page_merge_user_message,
    build_review_suggestion_prompt,
    compute_ingest_analysis_max_tokens,
    compute_ingest_generation_max_tokens,
    compute_ingest_review_max_tokens,
    compute_ingest_source_budget,
    extract_marked_section,
    format_ingest_warning_log_entry,
    hash_text_hex,
    len16,
    long_source_chunk_sizes,
    split_source_into_semantic_chunks,
    trim_long_text,
)


def _analysis(*a, lang="auto", **k):
    return build_analysis_prompt(*a, output_language=lang, **k)


def _gen(*a, lang="auto", **k):
    return build_generation_prompt(*a, output_language=lang, **k)


class TestAnalysisPrompt:
    def test_explicit_language(self):
        assert "MANDATORY OUTPUT LANGUAGE: Chinese" in _analysis("purpose", "index", "english source content", lang="Chinese")

    def test_user_setting_beats_source(self):
        p = _analysis("", "", "这段内容是中文", lang="Japanese")
        assert "MANDATORY OUTPUT LANGUAGE: Japanese" in p
        assert "OUTPUT LANGUAGE: Chinese" not in p

    def test_auto_detects_source(self):
        assert "MANDATORY OUTPUT LANGUAGE: Japanese" in _analysis("", "", "これは日本語の文章です")

    def test_auto_empty_is_english(self):
        assert "MANDATORY OUTPUT LANGUAGE: English" in _analysis("", "", "")

    def test_structural_sections(self):
        p = _analysis("", "", "")
        for s in ("## Key Entities", "## Key Concepts", "## Main Arguments & Findings", "## Recommendations"):
            assert s in p

    def test_subject_attachment(self):
        p = _analysis("", "", "")
        assert "Which named subject is each claim about" in p
        assert "Do not transfer claims, limits, or evaluations" in p

    def test_entity_threshold(self):
        p = _analysis("", "", "")
        for s in (
            "standalone-page significance threshold", "passing mentions", "ordinary story props",
            "unnamed or generic-role people", "Page-worthy: yes",
            "reuse the exact title of an existing wiki page first", "never invent or phonetically guess a name",
        ):
            assert s in p
        assert "central vs. peripheral" not in p

    def test_schema_injected(self):
        schema = "## Page Types\n| goal | wiki/goals/ | Outcomes |\n| habit | wiki/habits/ | Behaviours |"
        p = _analysis("", "", "", schema)
        assert "## Project Schema" in p and "wiki/goals/" in p and "goal, habit" in p

    def test_no_schema_section(self):
        assert "## Project Schema" not in _analysis("", "", "")

    def test_never_invent(self):
        assert "never invent" in _analysis("", "", "", "| goal | wiki/goals/ | x |")

    def test_blank_separator_lines_dropped_like_ts_filter_boolean(self):
        # Only the language directive (which contains its own blank line)
        # introduces a "\n\n"; the builder's "" separators are filtered out.
        assert _analysis("", "", "").count("\n\n") == 1


def test_warning_log_entry():
    entry = format_ingest_warning_log_entry(
        "book.pdf",
        ["FILE block was truncated", "Aggregate repair failed"],
        datetime(2026, 6, 30, 1, 2, 3, tzinfo=timezone.utc),
    )
    assert "## 2026-06-30T01:02:03.000Z | book.pdf" in entry
    assert "1. FILE block was truncated" in entry
    assert "2. Aggregate repair failed" in entry


class TestGenerationPrompt:
    def test_languages(self):
        assert "MANDATORY OUTPUT LANGUAGE: Chinese" in _gen("schema", "purpose", "index", "source.pdf", lang="Chinese")
        assert "MANDATORY OUTPUT LANGUAGE: Vietnamese" in _gen("", "", "", "file.pdf", lang="Vietnamese")
        assert "MANDATORY OUTPUT LANGUAGE: Chinese" in _gen("", "", "", "file.pdf", None, "这是中文源文档内容")

    def test_respects_setting_over_source(self):
        p = _gen("", "", "", "x.pdf", None, "私は日本語の文章を書きます", lang="English")
        assert "MANDATORY OUTPUT LANGUAGE: English" in p
        assert "OUTPUT LANGUAGE: Japanese" not in p

    def test_source_name_and_filename_rules(self):
        assert "my-paper.pdf" in _gen("", "", "", "my-paper.pdf")
        p = _gen("", "", "", "source.pdf", lang="Chinese")
        assert "Derive filenames from the page title in the mandatory output language" in p
        assert "keep readable CJK characters in the filename" in p
        assert "proper nouns and technical identifiers take precedence" in p
        assert "GPT-5" in p and "Transformer" in p and "standard original form" in p
        assert "Do not put raw URLs, citation strings, or full paper titles directly into file paths" in p
        assert "technical terms with no widely-used localized equivalent" in p
        assert "No exceptions — not even for page names" not in p

    def test_subject_boundaries(self):
        p = _gen("", "", "", "source.pdf")
        assert "Preserve subject boundaries" in p
        assert "Do not merge or generalize a claim about one subject into another subject's page" in p
        assert "cite which source/frontmatter `sources` entry supports that statement" in p

    def test_entity_rules(self):
        p = _gen("", "", "", "source.pdf")
        for s in (
            "only for a candidate explicitly marked `Page-worthy: yes`",
            "wikilinks only when the target already exists",
            "opening paragraph of every entity page must answer what the subject is",
            "stable, high-confidence public knowledge",
            "independently verifiable background in the mandatory output language",
            "do not create the standalone page",
            "reuse the exact title of an existing wiki page first",
            "Final Global Digest is authoritative",
        ):
            assert s in p

    def test_schema_authoritative(self):
        p = _gen("Use wiki/people/ for people. Use wiki/technologies/ for technical methods.", "", "", "source.pdf")
        assert "## Project Schema and Routing (AUTHORITATIVE)" in p
        assert "write pages into those schema-defined folders" in p
        assert "frontmatter type must match the schema directory" in p
        assert "otherwise use wiki/entities/" in p
        assert "Entity pages in wiki/entities/ for key entities" not in p

    def test_dates_and_structured_data(self):
        p = _gen("", "", "", "paper.pdf")
        assert "Today's date is" in p and "Use this exact date" in p
        assert "created: 2026-04-29" not in p
        p = _gen("", "", "", "schema.sql")
        a = _analysis("", "", "CREATE TABLE users (id BIGINT PRIMARY KEY);")
        assert "Preserve structured source data verbatim" in p and "DDL" in p
        assert "Preserve structured source data verbatim" in a and "constraints, keys, or indexes" in a

    def test_summary_path_and_base_name(self):
        assert "**wiki/sources/my paper.v2.md**" in _gen("", "", "", "my paper.v2.pdf")
        assert "**wiki/sources/x.md**" in _gen("", "", "", "a.pdf", None, "", "wiki/sources/x.md")

    def test_output_format_is_last_and_language_repeated(self):
        p = _gen("", "", "", "a.pdf", lang="Korean")
        assert p.count("MANDATORY OUTPUT LANGUAGE: Korean") == 2
        assert p.rindex("## Output Format") < p.rindex("---\n## ⚠️ MANDATORY")


class TestConsistency:
    def test_both_stages_same_language(self):
        a = _analysis("", "", "", lang="Korean")
        g = _gen("", "", "", "f.pdf", lang="Korean")
        assert "MANDATORY OUTPUT LANGUAGE: Korean" in a and "MANDATORY OUTPUT LANGUAGE: Korean" in g

    def test_auto_agrees(self):
        korean = "이것은 한국어 문장입니다"
        assert "MANDATORY OUTPUT LANGUAGE: Korean" in _analysis("", "", korean)
        assert "MANDATORY OUTPUT LANGUAGE: Korean" in _gen("", "", "", "f.pdf", None, korean)


def test_chunk_system_prompt_defers_entity_decisions():
    p = build_chunk_analysis_system_prompt("", "", "", "long source")
    chunk = p.index("## Chunk Analysis")
    digest = p.index("## Updated Global Digest")
    rules = p.index("Entity handling rules:")
    assert 0 <= chunk < digest < rules
    assert "without making a final standalone-page decision from one chunk alone" in p
    assert "accumulate recurrence across chunks" in p
    assert "supporting chunks" in p
    assert "Page-worthy: yes" in p


def test_page_merge_prompt():
    p = build_page_merge_system_prompt()
    for s in (
        "inputs are internal containers, not revisions or competing versions",
        "may mention additional subjects for comparison or context",
        "keep those comparisons attribution-exact",
        "do not fold them into claims about the main page subject",
        "prefer keeping them separate",
        "attribute a claim to a real source filename only when the supporting file is unambiguous",
        "otherwise do not guess",
        "Never create comparison sections/tables about the merge inputs",
        "Never invent URLs",
    ):
        assert s in p
    assert "describe the same entity" not in p


def test_page_merge_user_message_provenance():
    existing = '---\ntitle: T\nsources: ["a.md", "b ---c.md"]\n---\nold'
    msg = build_page_merge_user_message(existing, "new", "dir\n#x.md")
    assert "(provenance: a.md, b c.md)" in msg
    assert "(provenance: dir x.md)" in msg
    assert msg.endswith("Now output the merged file. Start with `---` on the first line.")
    msg2 = build_page_merge_user_message("no fm", "new", "")
    assert "(provenance: sources listed in the existing page frontmatter)" in msg2
    assert "(provenance: additional source)" in msg2


class TestReviewPrompt:
    def test_grounding_in_review(self):
        p = build_review_suggestion_prompt(
            "A private cognitive-analysis project", "- [[attention-filter]]", "internal-notes.md",
            "AF is the project's attention-filter signal.",
            "The pipeline dispatches AF events between internal modules.",
            "---FILE: wiki/concepts/attention-filter.md---", 128_000,
        )
        for s in (
            "project-local vocabulary", "Never emit a bare ambiguous term as a web query",
            "Do not substitute a popular public meaning", "underlying real-world concepts or comparisons",
            "AF is the project's attention-filter signal",
        ):
            assert s in p

    def test_grounding_in_generation(self):
        p = _gen("A private cognitive-analysis project", "- [[attention-filter]]", "Project overview",
                 "internal-notes.md", "AF is the project's attention-filter signal.")
        for s in ("project-local vocabulary", "Never emit a bare ambiguous term as a web query",
                  "Do not substitute a popular public meaning"):
            assert s in p

    def test_trims_sections(self):
        p = build_review_suggestion_prompt("", "", "s", "a" * 50_000, "", "", 64_000)
        assert "[...trimmed for prompt budget...]" in p


class TestPlanning:
    def test_tokens(self):
        assert compute_ingest_generation_max_tokens(64_000) == 8_192
        assert compute_ingest_generation_max_tokens(128_000) == 16_384
        assert compute_ingest_generation_max_tokens(256_000) == 24_576
        assert compute_ingest_generation_max_tokens(1_000_000) == 32_768
        assert compute_ingest_analysis_max_tokens(64_000) == 4_096
        assert compute_ingest_analysis_max_tokens(128_000) == 4_800
        assert compute_ingest_analysis_max_tokens(256_000) == 8_192
        assert compute_ingest_analysis_max_tokens(1_000_000) == 8_192
        assert compute_ingest_review_max_tokens(1_000_000) == 8_192

    def test_source_budget(self):
        small = compute_ingest_source_budget(64_000, 8_000)
        large = compute_ingest_source_budget(1_000_000, 8_000)
        assert small > 20_000
        assert 200_000 < large <= 300_000

    def test_chunk_sizes(self):
        assert long_source_chunk_sizes(100_000) == (55_000, 3_000)
        assert long_source_chunk_sizes(1) == (12_000, 960)

    def test_semantic_split(self):
        content = "\n".join(["# Chapter One", "", "A" * 1200, "", "B" * 1200, "", "## Section Two", "", "C" * 1200, "", "D" * 1200])
        chunks = split_source_into_semantic_chunks(content, 1800, 200)
        assert len(chunks) > 1
        assert chunks[0].headingPath == "Chapter One"
        assert any("Section Two" in c.headingPath for c in chunks)
        assert len(chunks[1].overlapBefore) > 0
        assert not chunks[1].main.startswith(chunks[0].main[-200:])
        assert [c.id for c in chunks] == [f"chunk-{i + 1}" for i in range(len(chunks))]
        assert all(c.total == len(chunks) for c in chunks)

    def test_split_empty(self):
        assert split_source_into_semantic_chunks("  \n\n ", 1800, 200) == []

    def test_split_counts_utf16_units(self):
        # 400 astral chars = 800 UTF-16 units; two of them (802 code points but
        # 1602 units with the joiner) exceed a 1000 target.
        content = "\U0001F600" * 400 + "\n\n" + "\U0001F601" * 400
        assert len(split_source_into_semantic_chunks(content, 1000, 100)) == 2
        assert len16("\U0001F600") == 2


def test_hash_text_hex_matches_js_fnv1a():
    assert hash_text_hex("") == "cbf29ce484222325"
    assert hash_text_hex("abc") == "e71fa2190541574b"
    assert hash_text_hex("héllo\U0001F600 世界\0x") == "e41b1fa8982d061c"


def test_extract_marked_section_and_trim():
    raw = "x\n## Chunk Analysis \nhello\n\n## Updated Global Digest\nd"
    assert extract_marked_section(raw, "Chunk Analysis") == "hello"
    assert extract_marked_section("## updated global digest\nD1\nD2", "Updated Global Digest") == "D1\nD2"
    assert extract_marked_section("nothing", "Chunk Analysis") == ""
    assert trim_long_text("abc", 10) == "abc"
    assert trim_long_text("abcdef  ", 5) == "abcde\n\n[...trimmed for prompt budget...]"
