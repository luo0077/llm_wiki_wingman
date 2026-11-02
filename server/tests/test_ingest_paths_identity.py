"""Translations of path-utils(.property).test.ts, source-identity.test.ts, wiki-filename.test.ts, wiki-page-types.test.ts."""
from __future__ import annotations

import random
import re
from datetime import datetime, timezone
from urllib.parse import quote

import pytest

from llm_wiki_server.ingest.path_utils import (
    get_file_name,
    get_file_stem,
    get_relative_path,
    is_absolute_path,
    join_path,
    normalize_path,
)
from llm_wiki_server.ingest.source_identity import (
    legacy_source_summary_slug_from_identity,
    source_identity_for_path,
    source_reference_identity,
    source_summary_slug_candidates_from_identity,
    source_summary_slug_from_identity,
)
from llm_wiki_server.ingest.wiki_filename import make_query_file_name, make_query_slug
from llm_wiki_server.ingest.wiki_page_types import GENERATION_WIKI_TYPES, infer_wiki_type_from_path, wiki_type_label


# ── path-utils ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "inp,out",
    [("C:\\Users\\foo\\bar", "C:/Users/foo/bar"), ("/Users/foo/bar", "/Users/foo/bar"), ("C:\\mixed/path\\here", "C:/mixed/path/here"), ("", ""), ("a\\\\b", "a//b")],
)
def test_normalize_path(inp, out):
    assert normalize_path(inp) == out


def test_join_path():
    assert join_path("a", "b", "c") == "a/b/c"
    assert join_path("a/", "/b", "c") == "a/b/c"
    assert join_path("C:\\Users", "foo", "bar") == "C:/Users/foo/bar"
    assert join_path("/abs", "path") == "/abs/path"
    assert join_path("only") == "only"


def test_get_file_name():
    assert get_file_name("/a/b/c.md") == "c.md"
    assert get_file_name("C:\\Users\\foo\\bar.pdf") == "bar.pdf"
    assert get_file_name("solo.md") == "solo.md"
    assert get_file_name("/dir/") == ""


def test_get_file_stem():
    assert get_file_stem("/a/b/note.md") == "note"
    assert get_file_stem("archive.tar.gz") == "archive.tar"
    assert get_file_stem(".env") == ".env"
    assert get_file_stem("C:\\dir\\notes.txt") == "notes"
    assert get_file_stem("README") == "README"


def test_get_relative_path():
    assert get_relative_path("/project/wiki/note.md", "/project") == "wiki/note.md"
    assert get_relative_path("/project/wiki/note.md", "/project/") == "wiki/note.md"
    assert get_relative_path("C:\\project\\wiki\\note.md", "C:\\project") == "wiki/note.md"
    assert get_relative_path("/other/path", "/project") == "/other/path"
    assert get_relative_path("/projector/a", "/project") == "/projector/a"
    assert get_relative_path("c:/users/me/inbox/sub/report.pdf", "C:/Users/Me/Inbox") == "sub/report.pdf"
    assert get_relative_path("//server/SHARE/sub/file.md", "//SERVER/share") == "sub/file.md"
    assert get_relative_path("c:/i\u0307/sub/file.md", "C:/\u0130") == "sub/file.md"
    assert get_relative_path("/Project/wiki/note.md", "/project") == "/Project/wiki/note.md"


def test_is_absolute_path():
    for p in ["/", "/home/user", "/a/b/c.md", "C:\\Users\\nash", "C:/Users/nash", "D:\\", "z:/project/file.pdf", "\\\\server\\share", "//server/share"]:
        assert is_absolute_path(p) is True, p
    for p in ["", "foo", "foo/bar.md", "./foo", "../foo", "wiki/concepts/attention.md", "C:foo"]:
        assert is_absolute_path(p) is False, p


def _rand_str(rng, n):
    alphabet = "ab/\\.:é中 -_"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, n)))


def test_path_properties():
    rng = random.Random(1234)
    for _ in range(300):
        p = _rand_str(rng, 30)
        assert normalize_path(normalize_path(p)) == normalize_path(p)
        assert "\\" not in normalize_path(p)
        assert len(normalize_path(p)) == len(p)
        segs = [_rand_str(rng, 10) for _ in range(rng.randint(1, 5))]
        joined = join_path(*segs)
        assert "//" not in joined and "\\" not in joined
        name = _rand_str(rng, 8).replace("/", "").replace("\\", "").replace(".", "") or "x"
        assert get_file_name(f"/some/path/{name}") == name
        assert get_file_name(f"C:\\a\\b\\{name}") == name
        assert get_file_stem(f"{name}.{name}") == name


# ── source-identity ───────────────────────────────────────────────

def test_identity_keeps_folder_context():
    assert source_identity_for_path("/tmp/project", "/tmp/project/raw/sources/project-a/config.yaml") == "project-a/config.yaml"


def test_reference_identity_prefixes():
    assert source_reference_identity("raw/sources/project-a/config.yaml") == "project-a/config.yaml"
    assert source_reference_identity("/tmp/project/raw/sources/project-a/config.yaml") == "project-a/config.yaml"


def test_case_insensitive_prefixes():
    assert source_identity_for_path("C:/Users/Me/Wiki", "c:/users/me/wiki/raw/sources/Project-A/Config.yaml") == "Project-A/Config.yaml"
    assert source_reference_identity("RAW/SOURCES/Project-A/Config.yaml") == "Project-A/Config.yaml"


def test_identity_fallbacks():
    assert source_identity_for_path("/p", "/elsewhere/dir/file.pdf") == "file.pdf"
    assert source_identity_for_path("/p", "/other/raw/sources/x/y.md") == "x/y.md"
    assert source_reference_identity("plain.md") == "plain.md"


def test_legacy_basename_slug():
    assert source_summary_slug_from_identity("config.yaml") == "config"
    assert source_summary_slug_from_identity("") == "source"


def test_slug_escapes_delimiters():
    slug = source_summary_slug_from_identity("a--b/config.yaml")
    assert re.match(r"^4-a-b--6-config--[a-z0-9]+$", slug)
    assert re.match(r"^1-a--1-b--6-config--[a-z0-9]+$", source_summary_slug_from_identity("a/b/config.yaml"))
    assert source_summary_slug_from_identity("4-a--b--6-config.yaml") != source_summary_slug_from_identity("a--b/config.yaml")
    assert re.sub(r"^4-a-b", "3-a-b", slug) in source_summary_slug_candidates_from_identity("a--b/config.yaml")


def test_slug_hash_is_fnv1a_base36():
    # Value computed with the TS implementation (FNV-1a over UTF-16 units, base36).
    assert source_summary_slug_from_identity("a/b.md") == "1-a--1-b--" + _expected_hash("a/b.md")


def _expected_hash(value):
    h = 0x811C9DC5
    data = value.encode("utf-16-le")
    for i in range(0, len(data), 2):
        h ^= data[i] | (data[i + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while h:
        h, r = divmod(h, 36)
        out = digits[r] + out
    return out or "0"


def test_slug_capped_for_nested_cjk():
    slug = source_summary_slug_from_identity("2024年/污水处理/反硝化除磷技术研究报告.pdf")
    assert len(slug) <= 120
    assert len(f"wiki/sources/{slug}.md") <= 136
    assert re.search(r"--[a-z0-9]+$", slug)
    assert "2024年" in slug and "污水处理" in slug and "反硝化除磷技术研究报告" in slug
    assert "%" not in slug


def test_slug_truncation_keeps_distinct_hashes():
    first = source_summary_slug_from_identity("2024年/污水处理/反硝化除磷技术研究报告.pdf")
    second = source_summary_slug_from_identity("2024年/污水处理/反硝化除磷技术研究报告修订版.pdf")
    assert first != second
    assert len(first) <= 120 and len(second) <= 120


def test_legacy_percent_encoded_candidates():
    identity = "2024年/污水处理/反硝化除磷技术研究报告.pdf"
    canonical = source_summary_slug_from_identity(identity)
    legacy = legacy_source_summary_slug_from_identity(identity)
    assert canonical != legacy
    assert "%" not in canonical
    assert "%E6" in legacy
    assert source_summary_slug_candidates_from_identity(identity) == [canonical, legacy]


def test_long_ascii_slug_truncates_with_utf16_length():
    identity = "x" * 100 + "/" + "y" * 100 + ".md"
    slug = source_summary_slug_from_identity(identity)
    assert len(slug) <= 120
    assert "--" in slug


# ── wiki-filename ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "title,slug",
    [
        ("Understanding RoPE in LLMs", "understanding-rope-in-llms"),
        ("What is GPT-4's context window?", "what-is-gpt-4s-context-window"),
        ("旋转位置编码", "旋转位置编码"),
        ("日本茶道", "日本茶道"),
        ("政策版本差异提示", "政策版本差异提示"),
        ("RoPE 旋转位置编码", "rope-旋转位置编码"),
        ("ＲｏＰＥ ２０２５", "rope-2025"),
        ("アテンション機構", "アテンション機構"),
        ("🎉🔥💯", "query"),
        ("!!! ??? ...", "query"),
        ("   ", "query"),
        ("", "query"),
        ("---foo   bar---", "foo-bar"),
        ("a - b - c", "a-b-c"),
    ],
)
def test_make_query_slug(title, slug):
    assert make_query_slug(title) == slug


def test_slug_truncation():
    assert len(make_query_slug("a" * 200)) == 50
    slug = make_query_slug("𠀀" * 60)
    assert len(slug) == 50  # code points
    quote(slug)  # encodable, no lone surrogates
    assert make_query_slug("RoPE") == make_query_slug("rope")
    assert make_query_slug("ATTENTION") == make_query_slug("attention")


NOW = datetime(2026, 4, 23, 14, 30, 52, 123000, tzinfo=timezone.utc)


def test_file_name_shape():
    r = make_query_file_name("Attention Is All You Need", NOW)
    assert r == {
        "slug": "attention-is-all-you-need",
        "date": "2026-04-23",
        "time": "143052",
        "fileName": "attention-is-all-you-need-2026-04-23-143052.md",
    }


def test_same_title_same_day_distinct():
    a = make_query_file_name("旋转位置编码", datetime(2026, 4, 23, 10, 0, 0, tzinfo=timezone.utc))
    b = make_query_file_name("旋转位置编码", NOW)
    c = make_query_file_name("旋转位置编码", datetime(2026, 4, 23, 23, 59, 59, tzinfo=timezone.utc))
    assert len({a["fileName"], b["fileName"], c["fileName"]}) == 3
    assert a["slug"] == "旋转位置编码"
    assert all(x["fileName"].startswith("旋转位置编码-2026-04-23-") for x in (a, b, c))


def test_emoji_title_distinct_per_save():
    a = make_query_file_name("🎉🔥", datetime(2026, 4, 23, 10, 0, 0, tzinfo=timezone.utc))
    b = make_query_file_name("🎉🔥", datetime(2026, 4, 23, 10, 0, 1, tzinfo=timezone.utc))
    assert a["fileName"] == "query-2026-04-23-100000.md"
    assert b["fileName"] == "query-2026-04-23-100001.md"


def test_file_name_uses_utc_for_aware_datetimes():
    from datetime import timedelta

    local = datetime(2026, 4, 23, 22, 30, 52, tzinfo=timezone(timedelta(hours=8)))
    assert make_query_file_name("x", local)["time"] == "143052"


def test_file_name_default_now():
    r = make_query_file_name("x")
    assert re.match(r"^x-\d{4}-\d{2}-\d{2}-\d{6}\.md$", r["fileName"])


# ── wiki-page-types ───────────────────────────────────────────────

def test_infer_core_dirs():
    for d, t in [("entities", "entity"), ("concepts", "concept"), ("sources", "source"), ("queries", "query"), ("comparisons", "comparison"), ("synthesis", "synthesis")]:
        assert infer_wiki_type_from_path(f"/project/wiki/{d}/x.md") == t
    assert infer_wiki_type_from_path("/project/wiki/findings/result.md") == "finding"
    assert infer_wiki_type_from_path("/project/wiki/thesis/main-claim.md") == "thesis"
    assert infer_wiki_type_from_path("/project/wiki/methodology/systematic-review.md") == "methodology"


def test_infer_windows_overview_custom():
    assert infer_wiki_type_from_path("C:\\wiki\\findings\\result.md") == "finding"
    assert infer_wiki_type_from_path("/project/wiki/overview.md") == "overview"
    assert infer_wiki_type_from_path("/project/wiki/people/ada-lovelace.md") == "people"
    assert infer_wiki_type_from_path("/project/wiki/technologies/vector-db.md") == "technologies"
    assert infer_wiki_type_from_path("/project/notes.md") is None


def test_type_label():
    assert wiki_type_label("finding") == "Finding"
    assert wiki_type_label("thesis") == "Thesis"
    assert wiki_type_label("methodology") == "Methodology"
    assert wiki_type_label("custom-topic") == "Custom Topic"
    assert wiki_type_label("a_b c") == "A B C"


def test_generation_types():
    for t in ("finding", "thesis", "methodology"):
        assert t in GENERATION_WIKI_TYPES
