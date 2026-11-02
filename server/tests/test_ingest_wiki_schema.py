"""Translation of src/lib/wiki-schema.test.ts."""
from __future__ import annotations

from llm_wiki_server.ingest.wiki_schema import (
    correct_wiki_page_routing,
    load_project_wiki_schema_routing,
    parse_wiki_schema_routing,
    validate_wiki_page_routing,
)

SCHEMA = """# Wiki Schema

## Page Types

| Type | Directory | Purpose |
| ---- | --------- | ------- |
| source | wiki/sources/ | Source summaries |
| concept | wiki/concepts/ | Ideas |
| method | wiki/methods/ | Methods |
| overview | wiki/ | Top-level overview |
"""

ROUTING = parse_wiki_schema_routing(SCHEMA)


def test_extracts_type_directories():
    assert ROUTING.typeDirs == {
        "source": "wiki/sources",
        "concept": "wiki/concepts",
        "method": "wiki/methods",
        "overview": "wiki",
    }


def test_ignores_tables_outside_page_types():
    routing = parse_wiki_schema_routing(
        "\n".join(
            [
                "# Wiki Schema", "", "| Name | Directory |", "| ---- | --------- |", "| draft | wiki/drafts/ |", "",
                "## Page Types", "", "| Type | Directory | Purpose |", "| ---- | --------- | ------- |",
                "| concept | wiki/concepts/ | Ideas |", "", "## Examples", "", "| Type | Directory |",
                "| ---- | --------- |", "| person | wiki/people/ |",
            ]
        )
    )
    assert routing.typeDirs == {"concept": "wiki/concepts"}


def test_rejects_unsafe_directories():
    routing = parse_wiki_schema_routing(
        "\n".join(
            [
                "## Page Types", "| Type | Directory |", "| --- | --- |", "| escape | wiki/../outside/ |",
                "| windows | wiki/bad:name/ |", "| safe | wiki/安全/ |",
            ]
        )
    )
    assert routing.typeDirs == {"safe": "wiki/安全"}


def test_correct_moves_typed_page():
    out = correct_wiki_page_routing(
        "wiki/concepts/retrieval.md", "---\ntype: method\ntitle: Retrieval\n---\n# Retrieval", ROUTING
    )
    assert out.to_dict() == {
        "path": "wiki/methods/retrieval.md",
        "message": 'Auto-routed page type "method" from "wiki/concepts/" to "wiki/methods/".',
    }


def test_correct_leaves_unknown_and_correct_unchanged():
    assert correct_wiki_page_routing("wiki/methods/retrieval.md", "---\ntype: method\n---", ROUTING).message is None
    assert (
        correct_wiki_page_routing("wiki/custom/item.md", "---\ntype: custom\n---", ROUTING).path
        == "wiki/custom/item.md"
    )


def test_validate_reports_mismatch():
    issue = validate_wiki_page_routing(
        "wiki/concepts/flash-attention.md",
        "\n".join(["---", "type: source", "title: Flash Attention", "---", "", "# Flash Attention"]),
        ROUTING,
    )
    assert 'type "source" must be under "wiki/sources/"' in issue.message


def test_validate_allows_custom_types():
    assert (
        validate_wiki_page_routing(
            "wiki/methods/retrieval.md",
            "\n".join(["---", "type: method", "title: Retrieval", "---", "", "# Retrieval"]),
            ROUTING,
        )
        is None
    )


def test_validate_ignores_untyped():
    assert validate_wiki_page_routing("wiki/concepts/no-type.md", "# No Type", ROUTING) is None


def test_validate_path_implies_other_type():
    issue = validate_wiki_page_routing("wiki/methods/x.md", "---\ntype: weird\n---\n", ROUTING)
    assert issue.message == 'Pages under "wiki/methods/" must use type "method", but found "weird".'


def test_load_project_routing(tmp_path):
    assert load_project_wiki_schema_routing(str(tmp_path)) is None
    (tmp_path / "schema.md").write_text("   \n", encoding="utf-8")
    assert load_project_wiki_schema_routing(str(tmp_path)) is None
    (tmp_path / "schema.md").write_text(SCHEMA, encoding="utf-8")
    assert load_project_wiki_schema_routing(str(tmp_path) + "/").typeDirs["method"] == "wiki/methods"
