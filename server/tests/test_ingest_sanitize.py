"""Translation of src/lib/ingest-sanitize.test.ts."""
from __future__ import annotations

import pytest

from llm_wiki_server.ingest.ingest_sanitize import normalize_malformed_wikilinks, sanitize_ingested_file_content

S = sanitize_ingested_file_content
N = normalize_malformed_wikilinks


def test_clean_unchanged():
    inp = "---\ntype: entity\ntitle: Foo\n---\n\n# Foo\n\nbody"
    assert S(inp) == inp


def test_strips_yaml_wrapped_document():
    assert S("```yaml\n---\ntype: entity\ntitle: Accumulibacter\n---\n\n# Body\n```") == "---\ntype: entity\ntitle: Accumulibacter\n---\n\n# Body"


@pytest.mark.parametrize("lang", ["md", "markdown", ""])
def test_strips_other_wrapped_documents(lang):
    assert S(f"```{lang}\n---\ntype: x\n---\nbody\n```") == "---\ntype: x\n---\nbody"


def test_strips_fence_wrapping_only_frontmatter():
    assert S("```yaml\n---\ntype: x\n---\n```\n\n# Body") == "---\ntype: x\n---\n\n# Body"


def test_strips_fence_after_blank_lines_case_insensitive():
    assert S("\n  \n```YAML\n---\ntype: x\n---\n```\n# Body") == "---\ntype: x\n---\n# Body"


def test_strips_crlf_fence_around_empty_frontmatter():
    assert S("```yaml\r\n---\r\n---\r\n```\r\n\r\n# Body") == "---\r\n---\r\n\r\n# Body"


def test_does_not_strip_body_fence():
    inp = "---\ntype: x\n---\n\n# Heading\n\n```js\nconsole.log('hi')\n```\n\nmore body"
    assert S(inp) == inp


def test_does_not_strip_unclosed_fence():
    inp = "```yaml\n---\ntype: x\n---\nbody"
    assert S(inp) == inp


def test_strips_frontmatter_key_prefix():
    assert S("frontmatter:\n---\ntype: entity\ntitle: LSTM\n---\n\n# Body") == "---\ntype: entity\ntitle: LSTM\n---\n\n# Body"


def test_repairs_missing_opening_fence():
    inp = '\n\ntype: entity\ntitle: "Foo: Bar"\nsources: [foo.pdf]\n---\n\n# Foo\n\nBody'
    assert S(inp) == '---\ntype: entity\ntitle: "Foo: Bar"\nsources: [foo.pdf]\n---\n\n# Foo\n\nBody'


def test_does_not_invent_frontmatter():
    inp = "title: A research question\n\n# Notes\n\nBody"
    assert S(inp) == inp


def test_frontmatter_word_in_prose_untouched():
    inp = "---\ntype: x\n---\n\nThe frontmatter: of this doc is above."
    assert S(inp) == inp


def test_repairs_wikilink_list():
    assert S("---\ntype: entity\nrelated: [[a]], [[b]], [[c]]\n---\n\nbody") == '---\ntype: entity\nrelated: ["[[a]]", "[[b]]", "[[c]]"]\n---\n\nbody'


def test_repairs_wikilink_list_crlf():
    assert (
        S("---\r\ntype: entity\r\nrelated: [[a]], [[b]]\r\n---\r\n# Body\r\n")
        == '---\r\ntype: entity\r\nrelated: ["[[a]]", "[[b]]"]\r\n---\r\n# Body\r\n'
    )


def test_single_wikilink_untouched():
    inp = "---\nrelated: [[a]]\n---\nbody"
    assert S(inp) == inp


def test_body_wikilink_text_untouched():
    inp = "---\ntype: x\n---\n\nrelated: [[a]], [[b]] in body prose"
    assert S(inp) == inp


def test_composes_repairs():
    inp = "```yaml\nfrontmatter:\n---\ntype: entity\nrelated: [[a]], [[b]]\n---\n\n# Body\n```"
    assert S(inp) == '---\ntype: entity\nrelated: ["[[a]]", "[[b]]"]\n---\n\n# Body'


# ── normalizeMalformedWikilinks ───────────────────────────────────

def test_numeric_wikilinks_to_citations():
    assert N("One [[1]], several [[12], [14], [15]], and page [[2026]].") == "One [1], several [12], [14], [15], and page [[2026]]."


def test_aliased_missing_bracket():
    assert N("See **[[target|label]** now.") == "See **[[target|label]]** now."


def test_preserves_frontmatter_code_embeds_links():
    inp = "\n".join(
        [
            "---", "related: ['[[4]]']", "---", "`[[5]]` and ![[asset|preview]",
            "[label](https://example.test) with [[target|alias]", "```md", "[[6]] [[code|sample]", "```",
        ]
    )
    assert N(inp) == inp


@pytest.mark.parametrize(
    "inp", ["[[1]](https://example.test)", "![[1]]", "[[2024]]", "[[a|b]]", "[[a]", "[[a|see [1]]", "[[a|b]]]"]
)
def test_does_not_rewrite_valid_or_ambiguous(inp):
    assert N(inp) == inp


def test_preserves_inline_and_blockquoted_fenced_code():
    inp = "\n".join(
        ["a `` b `[[1]]` c", "> ```md", "> [[2]]", "> ```", "> > ~~~~md", "> > [[3]]", "> > ~~~~"]
    )
    assert N(inp) == inp


def test_empty_frontmatter_and_crlf():
    assert N("---\n---\nbody [[1]]\n---\nx") == "---\n---\nbody [1]\n---\nx"
    assert N("x\r\n[[1]]\r\n") == "x\r\n[1]\r\n"


def test_body_horizontal_rules_not_frontmatter():
    assert N("Intro\n\n---\n\nSee [[1]]\n\n---\n\nTail [[2]]\n") == "Intro\n\n---\n\nSee [1]\n\n---\n\nTail [2]\n"


def test_nested_lists_code_math():
    lines = ["- item", "    - nested [[1]]", "    const matrix = [[2]]", "$$", "[[3], [4]]", "$$", "<pre>", "[[5]]", "</pre>"]
    expected = list(lines)
    expected[1] = "    - nested [1]"
    assert N("\n".join(lines)) == "\n".join(expected)


def test_bom_frontmatter_and_idempotent_triple_brackets():
    inp = "﻿---\nrelated: ['[[1]]']\n---\n[[[1]]] and [[2]]"
    once = N(inp)
    assert once == "﻿---\nrelated: ['[[1]]']\n---\n[[[1]]] and [2]"
    assert N(once) == once


def test_public_sanitizer_applies_body_normalization():
    inp = "---\ntype: entity\nrelated: ['[[4]]']\n---\nBody [[1]] and [[target|alias]"
    assert S(inp) == "---\ntype: entity\nrelated: ['[[4]]']\n---\nBody [1] and [[target|alias]]"


def test_idempotent():
    once = N("[[1]] [[target|alias] [[2], [3]]")
    assert N(once) == once
