"""Translation of src/lib/frontmatter.test.ts."""
from __future__ import annotations

import pytest

from llm_wiki_server.ingest.frontmatter import parse_frontmatter


def test_no_frontmatter():
    r = parse_frontmatter("# Hello\nbody text")
    assert r.frontmatter is None
    assert r.body == "# Hello\nbody text"
    assert r.rawBlock == ""


def test_basic_block():
    r = parse_frontmatter('---\ntype: entity\ntitle: "Foo"\n---\n\n# Body')
    assert r.frontmatter == {"type": "entity", "title": "Foo"}
    assert r.body == "# Body"


@pytest.mark.parametrize(
    "content",
    [
        "---\ntype: x\n---\nbody",
        "---\ntype: x\n---\n\nbody with leading blank line",
        "# No frontmatter at all",
        '---\ntype: x\ntitle: "Foo"\ntags: [a, b]\n---\n\n# Heading',
    ],
)
def test_rawblock_plus_body_roundtrip(content):
    r = parse_frontmatter(content)
    assert r.rawBlock + r.body == content


def test_recovered_fallback_is_lossy():
    content = "```yaml\n---\ntitle: Fixed\n---\n```\n\n# Body"
    r = parse_frontmatter(content)
    assert r.frontmatter == {"title": "Fixed"}
    assert r.rawBlock == "---\ntitle: Fixed\n---\n"
    assert r.rawBlock + r.body != content
    assert r.body == "# Body"


def test_inline_arrays():
    assert parse_frontmatter("---\ntags: [foo, bar, baz]\n---\nx").frontmatter == {"tags": ["foo", "bar", "baz"]}


def test_block_arrays():
    r = parse_frontmatter("---\nrelated:\n  - foo.md\n  - bar.md\n  - baz.md\n---\nbody")
    assert r.frontmatter == {"related": ["foo.md", "bar.md", "baz.md"]}
    assert r.body == "body"


def test_dates_stay_strings():
    assert parse_frontmatter("---\ncreated: 2026-04-09\n---\nbody").frontmatter == {"created": "2026-04-09"}


def test_cjk_tags():
    r = parse_frontmatter("---\ntags: [污水处理, 功能微生物, NOB优势菌]\n---\nbody")
    assert r.frontmatter == {"tags": ["污水处理", "功能微生物", "NOB优势菌"]}


def test_strips_quotes():
    assert parse_frontmatter('---\ntitle: "Hello World"\n---\nx').frontmatter == {"title": "Hello World"}


def test_frontmatter_only_file():
    r = parse_frontmatter("---\ntype: entity\n---\n")
    assert r.frontmatter == {"type": "entity"}
    assert r.body == ""


def test_crlf():
    r = parse_frontmatter("---\r\ntype: query\r\n---\r\n\r\nbody")
    assert r.frontmatter == {"type": "query"}
    assert r.body == "body"


def test_malformed_yaml_still_strips_fence():
    r = parse_frontmatter("---\nthis is not: : valid\n  malformed\n---\nbody")
    assert r.body == "body"


def test_real_accumulibacter():
    content = '---\ntype: entity\ntitle: "Accumulibacter"\ncreated: 2026-04-09\nupdated: 2026-04-09\ntags: [microorganism, PAO, DPAO, phosphorus-removal, Betaproteobacteria]\nrelated:\n  - wiki/entities/dpao.md\n  - wiki/entities/paos.md\n  - wiki/concepts/denitrifying-phosphorus-removal.md\nsources: ["research-foo-2026-04-09.md"]\n---\n\n# Body'
    r = parse_frontmatter(content)
    assert r.frontmatter["type"] == "entity"
    assert r.frontmatter["title"] == "Accumulibacter"
    assert r.frontmatter["related"] == [
        "wiki/entities/dpao.md",
        "wiki/entities/paos.md",
        "wiki/concepts/denitrifying-phosphorus-removal.md",
    ]
    assert r.frontmatter["sources"] == ["research-foo-2026-04-09.md"]
    assert r.body == "# Body"


def test_real_nitrospira():
    content = '---\ntype: entity\ntitle: Nitrospira\ncreated: 2026-04-07\nupdated: 2026-04-07\ntags: [污水处理, 功能微生物, 亚硝酸盐氧化细菌, NOB优势菌]\nrelated: [nitrite-oxidizing-bacteria, nitrifying-bacteria, nh3-n]\nsources: ["research--2026-04-07.md"]\n---\n'
    r = parse_frontmatter(content)
    assert r.frontmatter["title"] == "Nitrospira"
    assert r.frontmatter["tags"] == ["污水处理", "功能微生物", "亚硝酸盐氧化细菌", "NOB优势菌"]
    assert r.frontmatter["related"] == ["nitrite-oxidizing-bacteria", "nitrifying-bacteria", "nh3-n"]
    assert r.frontmatter["sources"] == ["research--2026-04-07.md"]


def test_yaml_fenced_block_closing_fence_stripped():
    content = '```yaml\n---\ntype: entity\ntitle: "Accumulibacter"\n---\n```\n\n# Accumulibacter\n\nbody'
    r = parse_frontmatter(content)
    assert r.frontmatter["type"] == "entity"
    assert r.frontmatter["title"] == "Accumulibacter"
    assert "```" not in r.body
    assert r.body.lstrip().startswith("# Accumulibacter")


def test_closing_fence_directly_after_dashes():
    content = "\n".join(
        ["```yaml", "---", "type: entity", 'title: "硝化菌"', "tags: [微生物]", "---", "```", "", "# 硝化菌", "", "## 基本信息", "", "- **类型**：自养型好氧细菌"]
    )
    r = parse_frontmatter(content)
    assert r.frontmatter["type"] == "entity"
    assert r.frontmatter["title"] == "硝化菌"
    assert "```" not in r.body
    assert "# 硝化菌" in r.body
    assert "## 基本信息" in r.body


def test_frontmatter_key_prefix():
    r = parse_frontmatter("frontmatter:\n---\ntype: entity\ntitle: LSTM\n---\n\n# LSTM\n\nbody")
    assert r.frontmatter["type"] == "entity"
    assert r.frontmatter["title"] == "LSTM"


def test_deep_horizontal_rule_not_frontmatter():
    content = "# Heading\n\nfirst paragraph\n\nsecond paragraph\n\n---\n\nthird paragraph after rule"
    r = parse_frontmatter(content)
    assert r.frontmatter is None
    assert r.body == content


def test_wikilink_list_repair():
    r = parse_frontmatter("---\ntype: entity\ntitle: LTTC\nrelated: [[riyuu-jiaoben]], [[zi-yuan-bu]], [[wai-wu]]\n---\nbody")
    assert r.frontmatter["type"] == "entity"
    assert r.frontmatter["related"] == ["[[riyuu-jiaoben]]", "[[zi-yuan-bu]]", "[[wai-wu]]"]


def test_nested_arrays_not_mangled():
    r = parse_frontmatter("---\nmatrix: [[1, 2], [3, 4]]\n---\nbody")
    assert r.frontmatter is not None
    assert r.frontmatter["matrix"] == ["[1,2]", "[3,4]"]  # JSON.stringify of each nested array
    assert r.body == "body"


def test_real_bod_entity():
    content = '---\ntype: entity\ntitle: BOD（生化需氧量）\ncreated: 2026-04-07\nupdated: 2026-04-07\ntags: [水质指标, 环境监测, 污水处理, 核心参数]\nrelated:\n  - "[[nh3-n]]"\n  - "[[soft-sensor-watertreatment]]"\n  - "[[ai-effluent-water-quality-prediction]]"\n  - "[[digital-twin-wastewater]]"\nsources: ["research-ai-2026-04-07.md"]\n---\n\n# Body'
    r = parse_frontmatter(content)
    assert r.frontmatter["title"] == "BOD（生化需氧量）"
    assert r.frontmatter["related"] == [
        "[[nh3-n]]",
        "[[soft-sensor-watertreatment]]",
        "[[ai-effluent-water-quality-prediction]]",
        "[[digital-twin-wastewater]]",
    ]
    assert r.body == "# Body"


# ── js-yaml JSON_SCHEMA parity (beyond the TS tests) ──────────────

def test_json_schema_scalars():
    r = parse_frontmatter(
        "---\na: yes\nb: true\nc: 0123\nd: 1e3\ne: 0x1F\nf: ~\ng: null\nh: 1.50\ni: .inf\nj: 1_000\nk: 12:30\nl: ''\n---\n"
    )
    assert r.frontmatter == {
        "a": "yes", "b": "true", "c": "123", "d": "1000", "e": "31", "f": "", "g": "",
        "h": "1.5", "i": "Infinity", "j": "1000", "k": "12:30", "l": "",
    }


def test_duplicate_keys_are_an_error():
    r = parse_frontmatter("---\na: 1\na: 2\n---\nbody")
    assert r.frontmatter is None
    assert r.body == "body"
    assert r.rawBlock == "---\na: 1\na: 2\n---\n"


def test_non_mapping_yaml_is_null():
    assert parse_frontmatter("---\n- a\n- b\n---\nbody").frontmatter is None
    assert parse_frontmatter("---\njust text\n---\nbody").frontmatter is None
