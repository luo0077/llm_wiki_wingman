"""Translations of context-budget, concurrency-limits, reasoning-capabilities, azure-openai,
research-query-grounding, source-filter and parsed-source-output tests."""
from __future__ import annotations

import math

import pytest

from llm_wiki_server.ingest import parsed_source_output as pso
from llm_wiki_server.ingest.azure_openai import build_azure_openai_url, is_azure_openai_endpoint, parse_azure_openai_endpoint
from llm_wiki_server.ingest.concurrency_limits import clamp_user_concurrency
from llm_wiki_server.ingest.context_budget import compute_context_budget
from llm_wiki_server.ingest.reasoning_capabilities import (
    is_adaptive_anthropic_model,
    is_gemini_thinking_level_model,
    normalize_reasoning_for_provider,
    resolve_ingest_reasoning,
    resolve_reasoning_capabilities,
)
from llm_wiki_server.ingest.research_query_grounding import ground_bare_research_queries, grounded_fallback_research_query
from llm_wiki_server.ingest.source_filter import filter_raw_source_tree, is_sensitive_config_source_file


# ── concurrency-limits ────────────────────────────────────────────

def test_clamp_user_concurrency():
    assert clamp_user_concurrency(64) == 64
    assert clamp_user_concurrency(100) == 64
    assert clamp_user_concurrency(0) == 1
    assert clamp_user_concurrency(2.9) == 2
    assert clamp_user_concurrency(math.nan, 4) == 4


# ── context-budget ────────────────────────────────────────────────

def test_budget_defaults():
    for v in (None, 0, math.nan):
        assert compute_context_budget(v).maxCtx == 204_800


def test_budget_8k():
    b = compute_context_budget(8_192)
    assert (b.pageBudget, b.maxPageSize, b.responseReserve, b.indexBudget) == (4_096, 4_096, 1_228, 409)


def test_budget_32k_floor():
    b = compute_context_budget(32_000)
    assert (b.pageBudget, b.maxPageSize) == (16_000, 5_000)


def test_budget_128k():
    b = compute_context_budget(131_072)
    assert (b.pageBudget, b.maxPageSize) == (65_536, 19_660)


def test_budget_200k():
    b = compute_context_budget(204_800)
    assert (b.pageBudget, b.maxPageSize, b.responseReserve, b.indexBudget) == (102_400, 30_720, 30_720, 10_240)


def test_budget_1m():
    b = compute_context_budget(1_000_000)
    assert (b.pageBudget, b.maxPageSize, b.responseReserve, b.indexBudget) == (500_000, 150_000, 150_000, 50_000)
    assert b.to_dict() == {"maxCtx": 1_000_000, "responseReserve": 150_000, "indexBudget": 50_000, "pageBudget": 500_000, "maxPageSize": 150_000}


def test_budget_sums_and_page_cap():
    for max_ctx in [4_000, 32_000, 128_000, 204_800, 1_000_000]:
        b = compute_context_budget(max_ctx)
        total = b.indexBudget + b.pageBudget + b.responseReserve
        assert total <= max_ctx
        assert max_ctx - total >= math.floor(max_ctx * 0.2)
    for max_ctx in [1_000, 2_000, 5_000, 8_192, 16_000, 32_000, 100_000, 1_000_000]:
        b = compute_context_budget(max_ctx)
        assert b.maxPageSize <= b.pageBudget
    assert compute_context_budget(100_000).responseReserve == 15_000


# ── reasoning-capabilities ────────────────────────────────────────

def cfg(provider, model, **kw):
    return {"provider": provider, "model": model, "apiKey": "key", "ollamaUrl": "http://localhost:11434",
            "customEndpoint": "https://gateway.example/v1", "maxContextSize": 128_000, **kw}


def test_custom_gateway_auto_only():
    c = cfg("custom", "Qwen3-thinking-only")
    assert list(resolve_reasoning_capabilities(c).modes) == ["auto"]
    assert normalize_reasoning_for_provider(c, {"mode": "off"}) == {"mode": "auto"}


def test_openrouter_controls():
    c = cfg("custom", "vendor/reasoning-model", customEndpoint="https://openrouter.ai/api/v1")
    assert list(resolve_reasoning_capabilities(c).modes) == ["auto", "off", "low", "medium", "high", "max", "custom"]
    assert normalize_reasoning_for_provider(c, {"mode": "low"}) == {"mode": "low"}


def test_thinking_required_models_have_no_off():
    assert "off" not in resolve_reasoning_capabilities(cfg("google", "gemini-2.5-pro")).modes
    assert "off" not in resolve_reasoning_capabilities(cfg("anthropic", "claude-opus-4-7")).modes
    assert "custom" not in resolve_reasoning_capabilities(cfg("anthropic", "claude-sonnet-4-6")).modes


def test_openai_reasoning_levels():
    assert list(resolve_reasoning_capabilities(cfg("openai", "gpt-5.4")).modes) == ["auto", "low", "medium", "high"]
    assert normalize_reasoning_for_provider(cfg("openai", "gpt-5.4"), {"mode": "max"}) == {"mode": "auto"}


def test_custom_budget_normalization():
    c = cfg("anthropic", "claude-sonnet-4-5")
    assert normalize_reasoning_for_provider(c, {"mode": "custom", "budgetTokens": 2048.9}) == {"mode": "custom", "budgetTokens": 2048}
    assert normalize_reasoning_for_provider(c, {"mode": "custom", "budgetTokens": 10}) == {"mode": "custom", "budgetTokens": 1024}
    assert normalize_reasoning_for_provider(c, {"mode": "custom", "budgetTokens": 99_999}) == {"mode": "custom", "budgetTokens": 32_768}
    assert normalize_reasoning_for_provider(c, {"mode": "custom", "budgetTokens": 0}) == {"mode": "auto"}
    assert normalize_reasoning_for_provider(c, {"mode": "custom"}) == {"mode": "auto"}


def test_misc_provider_rules():
    assert list(resolve_reasoning_capabilities(cfg("custom", "m", customEndpoint="https://api.deepseek.com/v1")).modes) == ["auto", "off", "high", "max"]
    assert list(resolve_reasoning_capabilities(cfg("custom", "m", customEndpoint="https://xiaomimimo.com")).modes) == ["auto", "off"]
    assert list(resolve_reasoning_capabilities(cfg("azure", "x", azureModelFamily="gpt5")).modes) == ["auto", "low", "medium", "high"]
    assert resolve_reasoning_capabilities(cfg("google", "gemini-2.5-pro")).customBudgetRange == {"min": 128, "max": 32_768}
    assert resolve_ingest_reasoning(cfg("openai", "x")) == {"mode": "off"}
    assert resolve_ingest_reasoning(cfg("openai", "x", ingestReasoning={"mode": "low"})) == {"mode": "low"}
    assert is_adaptive_anthropic_model(cfg("anthropic", "claude-opus-4-7"))
    assert not is_adaptive_anthropic_model(cfg("anthropic", "claude-opus-4-5"))
    assert is_gemini_thinking_level_model(cfg("google", "gemini-3-pro"))


# ── azure-openai ──────────────────────────────────────────────────

def test_azure_urls():
    assert build_azure_openai_url("https://my-resource.openai.azure.com", "gpt-5", "2024-10-21") == \
        "https://my-resource.openai.azure.com/openai/deployments/gpt-5/chat/completions?api-version=2024-10-21"
    assert build_azure_openai_url("https://my-resource.openai.azure.com/openai/deployments/my-gpt5", "wrong-model", "2024-10-21") == \
        "https://my-resource.openai.azure.com/openai/deployments/my-gpt5/chat/completions?api-version=2024-10-21"


def test_azure_endpoint_detection():
    assert is_azure_openai_endpoint("https://my-resource.openai.azure.com") is True
    assert is_azure_openai_endpoint("my-resource.openai.azure.com/openai/deployments/wiki") is True
    assert is_azure_openai_endpoint("https://my-resource.openai.azure.com.evil.com") is False
    assert is_azure_openai_endpoint("https://example.com/path/openai.azure.com") is False
    assert is_azure_openai_endpoint("  ") is False


def test_azure_parse_variants():
    p = parse_azure_openai_endpoint(
        "https://a.openai.azure.com/openai/deployments/d%20x/chat/completions?api-version=2024-02-01", "", ""
    )
    assert p.to_dict() == {"resourceBase": "https://a.openai.azure.com", "deployment": "d x", "apiVersion": "2024-02-01"}
    assert parse_azure_openai_endpoint("https://a.openai.azure.com", "", "") is None
    assert parse_azure_openai_endpoint("https://a.openai.azure.com", "dep", "").apiVersion == "2024-10-21"
    # non-azure endpoint falls back to plain concatenation
    assert build_azure_openai_url("https://x.com/", "my dep", "") == \
        "https://x.com/openai/deployments/my%20dep/chat/completions?api-version=2024-10-21"


# ── research-query-grounding ──────────────────────────────────────

def test_fallback_keeps_term_and_context():
    assert grounded_fallback_research_query("AF", "An attention-filter signal in the project pipeline.") == \
        "AF An attention-filter signal in the project pipeline."


def test_replaces_only_bare_queries():
    assert ground_bare_research_queries(["AF", "AF attention filter cognitive signal"], "AF", "attention-filter signal") == \
        ["AF attention-filter signal", "AF attention filter cognitive signal"]


def test_preserves_special_characters():
    qs = ["C# async await", "snake_case naming", "F# pattern matching", "site:https://example.com AF"]
    assert ground_bare_research_queries(qs, "AF", "attention filter") == qs


def test_normalizes_wrapped_terms_and_dedupes():
    assert ground_bare_research_queries(["AF", "af", "[[AF]]", '"AF"'], "AF", "attention filter") == ["AF attention filter"]


def test_removes_markdown_and_urls_from_context():
    q = grounded_fallback_research_query("[[Hana]]", "# Module `Hana` details https://example.test/private")
    assert q == "Hana Module Hana details"


def test_cjk_and_word_boundary_cap():
    q = grounded_fallback_research_query("灵犀", "context " * 100)
    assert q.startswith("灵犀 ")
    assert len(q) <= 280
    assert not q.endswith(" ")


def test_long_cjk_context_kept():
    q = grounded_fallback_research_query("灵犀", "AI " + "中" * 300)
    assert q.count("中") >= 150


# ── source-filter ─────────────────────────────────────────────────

def test_filter_raw_source_tree():
    tree = [
        {"name": ".cache", "path": "/project/raw/sources/.cache", "is_dir": True,
         "children": [{"name": "source.pdf.txt", "path": "/project/raw/sources/.cache/source.pdf.txt", "is_dir": False}]},
        {"name": ".DS_Store", "path": "/project/raw/sources/.DS_Store", "is_dir": False},
        {"name": ".claude", "path": "/project/raw/sources/.claude", "is_dir": True,
         "children": [
             {"name": "notes.md", "path": "/project/raw/sources/.claude/notes.md", "is_dir": False},
             {"name": "settings.json", "path": "/project/raw/sources/.claude/settings.json", "is_dir": False},
         ]},
        {"name": "empty-after-filter", "path": "/project/raw/sources/empty-after-filter", "is_dir": True,
         "children": [{"name": ".DS_Store", "path": "/project/raw/sources/empty-after-filter/.DS_Store", "is_dir": False}]},
    ]
    assert filter_raw_source_tree(tree) == [
        {"name": ".claude", "path": "/project/raw/sources/.claude", "is_dir": True,
         "children": [{"name": "notes.md", "path": "/project/raw/sources/.claude/notes.md", "is_dir": False}]},
    ]


def test_sensitive_config_files():
    for p in ["/external/.claude/settings.json", "/external/.codex/config.yaml", "/external/.codex/config.toml",
              "/external/.cursor/rules.yml", "/external/.gemini/config.xml", "/external/.mcp/.env",
              "/external/.mcp/servers.json", "/external/.CLAUDE/Settings.JSON"]:
        assert is_sensitive_config_source_file(p) is True, p
    assert is_sensitive_config_source_file("/external/data.json") is False
    assert is_sensitive_config_source_file("/external/.claude/research.md") is False


# ── parsed-source-output ──────────────────────────────────────────

def test_parsed_output_path():
    f = pso.parsed_markdown_output_path
    assert f("/project", "/project/raw/sources/reports/a.pdf") == "/project/raw/parsed/reports/a.pdf.md"
    assert f("C:\\Project", "c:\\project\\raw\\sources\\A.DOCX") == "C:/Project/raw/parsed/A.DOCX.md"
    assert f("/project", "raw/sources/a.pdf") == "/project/raw/parsed/a.pdf.md"
    assert f("/project", "/other/report.pdf") is None
    assert f("/project", "/project/raw/sources/note.md") is None
    assert f("/project", "/project/raw/sources/data.json") is None
    for name in ["a.docm", "a.ppt", "a.ppsm", "a.xlsb", "a.rtf"]:
        assert f("/project", f"/project/raw/sources/{name}") == f"/project/raw/parsed/{name}.md"


class _FsRecorder:
    def __init__(self):
        self.calls = []
        self.read_value = "# Parsed"

    def create_directory(self, p): self.calls.append(("create_directory", p))
    def write_file_atomic(self, p, c): self.calls.append(("write_file_atomic", p, c))
    def delete_file(self, p): self.calls.append(("delete_file", p))
    def read_file(self, p):
        self.calls.append(("read_file", p))
        return self.read_value


@pytest.fixture
def fsmock(monkeypatch):
    from llm_wiki_server.commands import fs

    rec = _FsRecorder()
    for name in ("create_directory", "write_file_atomic", "delete_file", "read_file"):
        monkeypatch.setattr(fs, name, getattr(rec, name))
    return rec


def test_persist_parsed_markdown(fsmock):
    out = pso.persist_parsed_markdown("/project", "/project/raw/sources/nested/report.pdf", "# Parsed")
    assert out == "/project/raw/parsed/nested/report.pdf.md"
    assert ("create_directory", "/project/raw/parsed/nested") in fsmock.calls
    assert ("write_file_atomic", "/project/raw/parsed/nested/report.pdf.md", "# Parsed") in fsmock.calls
    assert pso.persist_parsed_markdown("/project", "/project/raw/sources/a.md", "x") is None


def test_remove_and_move_parsed_markdown(fsmock):
    pso.remove_parsed_markdown("/project", "/project/raw/sources/old/report.pdf")
    assert fsmock.calls == [("delete_file", "/project/raw/parsed/old/report.pdf.md")]
    fsmock.calls.clear()
    pso.move_parsed_markdown("/project", "/project/raw/sources/old/report.pdf", "/project/raw/sources/new/report.pdf")
    assert ("read_file", "/project/raw/parsed/old/report.pdf.md") in fsmock.calls
    assert ("write_file_atomic", "/project/raw/parsed/new/report.pdf.md", "# Parsed") in fsmock.calls
    assert fsmock.calls[-1] == ("delete_file", "/project/raw/parsed/old/report.pdf.md")
