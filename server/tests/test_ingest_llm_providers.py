"""Wire-format coverage for src/lib/llm-providers.ts, azure-openai.ts, reasoning-*.ts and the JS compat helpers."""
from __future__ import annotations

import json

import pytest

from llm_wiki_server.ingest import llm_jscompat as js
from llm_wiki_server.ingest.azure_openai import (
    build_azure_openai_url,
    is_azure_openai_endpoint,
    parse_azure_openai_endpoint,
)
from llm_wiki_server.ingest.llm_providers import (
    build_anthropic_url,
    derive_anthropic_max_tokens,
    get_provider_config,
    is_local_or_private_http_endpoint,
    merge_llm_request_headers,
    parse_anthropic_line,
    parse_google_line,
    parse_openai_line,
    supports_image_input,
)
from llm_wiki_server.ingest.reasoning_capabilities import (
    is_openrouter_endpoint,
    normalize_reasoning_for_provider,
    resolve_ingest_reasoning,
)
from llm_wiki_server.ingest.reasoning_detector import (
    count_reasoning_chars_in_line,
    extract_reasoning_text_from_line,
)

U = [{"role": "user", "content": "hi"}]


def test_merge_headers_precedence_and_validation():
    out = merge_llm_request_headers(
        {"authorization": "mine", "X-Ok": " v ", "bad name": "x", "X-Empty": " ", "X-NL": "a\nb"},
        {"Authorization": "Bearer k", "Content-Type": "application/json", "Skip": ""})
    assert out == {"Authorization": "Bearer k", "X-Ok": "v", "Content-Type": "application/json"}


def test_local_endpoint_detection():
    for u in ("http://localhost:1", "http://x.localhost/", "http://127.0.0.1", "http://[::1]:8000",
              "http://10.1.2.3", "http://192.168.1.5", "http://172.16.0.1", "http://172.31.9.9"):
        assert is_local_or_private_http_endpoint(u), u
    for u in ("http://172.32.0.1", "https://api.openai.com", "nonsense", "http://172.x.1.1"):
        assert not is_local_or_private_http_endpoint(u), u


def test_anthropic_urls():
    assert build_anthropic_url("https://x/v1/messages") == "https://x/v1/messages"
    assert build_anthropic_url("https://x/v1") == "https://x/v1/messages"
    assert build_anthropic_url("https://x/api/paas/v4/") == "https://x/api/paas/v4/messages"
    assert build_anthropic_url("https://x/anthropic") == "https://x/anthropic/v1/messages"
    assert build_anthropic_url("https://api.minimax.io") == "https://api.minimax.io/anthropic/v1/messages"
    assert build_anthropic_url("https://api.minimaxi.com/v1") == "https://api.minimaxi.com/anthropic/v1/messages"


def test_minimax_bearer_auth_and_image_gate():
    p = get_provider_config({"provider": "minimax", "apiKey": "k", "model": "MiniMax-M2", "maxContextSize": 8192})
    assert p.url == "https://api.minimax.io/anthropic/v1/messages"
    assert p.headers["Authorization"] == "Bearer k" and "x-api-key" not in p.headers
    img = [{"role": "user", "content": [{"type": "image", "mediaType": "image/png", "dataBase64": "AA"}]}]
    with pytest.raises(Exception, match="MiniMax-M3"):
        p.build_body(img)
    assert supports_image_input({"provider": "minimax", "model": "MiniMax-M3"})
    assert not supports_image_input({"provider": "minimax", "model": "MiniMax-M2"})
    assert not supports_image_input({"provider": "codex-cli", "model": "x"})


def test_derive_anthropic_max_tokens():
    assert derive_anthropic_max_tokens(None) == 10240
    assert derive_anthropic_max_tokens(1) == 1
    assert derive_anthropic_max_tokens(10_000_000) == 16384


def test_anthropic_thinking_budget_and_adaptive():
    cfg = {"provider": "anthropic", "apiKey": "k", "model": "claude-3-7-sonnet", "maxContextSize": 8192}
    body = get_provider_config(cfg).build_body(U, {"temperature": 0.2, "max_tokens": 2000, "reasoning": {"mode": "medium"}})
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert body["max_tokens"] == 4097 and "temperature" not in body
    body = get_provider_config({**cfg, "model": "claude-opus-4-6"}).build_body(U, {"reasoning": {"mode": "max"}})
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"] == {"effort": "max"}
    # off keeps temperature
    body = get_provider_config(cfg).build_body(U, {"temperature": 0.2, "reasoning": {"mode": "off"}})
    assert body["temperature"] == 0.2 and "thinking" not in body
    # stop sequences
    assert get_provider_config(cfg).build_body(U, {"stop": "x"})["stop_sequences"] == ["x"]


def test_custom_openrouter_deepseek_mimo_kimi_adaptations():
    base = {"provider": "custom", "model": "m", "maxContextSize": 8192}
    b = get_provider_config({**base, "customEndpoint": "https://openrouter.ai/api/v1"}).build_body(
        U, {"reasoning": {"mode": "off"}})
    assert b["reasoning"] == {"effort": "none"}
    b = get_provider_config({**base, "customEndpoint": "https://openrouter.ai/api/v1"}).build_body(
        U, {"reasoning": {"mode": "custom", "budgetTokens": 2000}})
    assert b["reasoning"] == {"max_tokens": 2000}
    ds = {**base, "model": "deepseek-v4-pro", "customEndpoint": "https://api.deepseek.com"}
    assert get_provider_config(ds).build_body(U, {"reasoning": {"mode": "off"}})["thinking"] == {"type": "disabled"}
    b = get_provider_config(ds).build_body(U, {"reasoning": {"mode": "high"}})
    assert b["thinking"] == {"type": "enabled"} and b["reasoning_effort"] == "high"
    mimo = {**base, "customEndpoint": "https://api.xiaomimimo.com/v1"}
    b = get_provider_config(mimo).build_body(U, {"temperature": 0.1, "max_tokens": 5, "reasoning": {"mode": "off"}})
    assert b["thinking"] == {"type": "disabled"} and b["max_completion_tokens"] == 5 and b["temperature"] == 0.1
    b = get_provider_config(mimo).build_body(U, {"temperature": 0.1})
    assert "temperature" not in b
    kimi = {**base, "customEndpoint": "https://api.moonshot.ai/v1"}
    assert "temperature" not in get_provider_config(kimi).build_body(U, {"temperature": 0.1})


def test_custom_azure_endpoint_headers_and_body():
    cfg = {"provider": "custom", "apiKey": "k", "model": "gpt-5", "customEndpoint": "https://r.openai.azure.com",
           "maxContextSize": 8192, "azureModelFamily": "gpt5"}
    p = get_provider_config(cfg)
    assert p.url == "https://r.openai.azure.com/openai/deployments/gpt-5/chat/completions?api-version=2024-10-21"
    assert p.headers["api-key"] == "k" and "Origin" not in p.headers
    body = p.build_body(U, {"temperature": 0.1, "max_tokens": 9})
    assert "model" not in body and body["max_completion_tokens"] == 9 and "temperature" not in body


def test_azure_helpers():
    assert is_azure_openai_endpoint("res.openai.azure.com")
    assert not is_azure_openai_endpoint("https://evil.com/x.openai.azure.com")
    assert not is_azure_openai_endpoint("")
    p = parse_azure_openai_endpoint(
        "https://r.openai.azure.com/openai/deployments/my%20dep/chat/completions?api-version=2025-01-01", "fb", "x")
    p = p.to_dict() if hasattr(p, "to_dict") else p
    assert p == {"resourceBase": "https://r.openai.azure.com", "deployment": "my dep", "apiVersion": "2025-01-01"}
    assert parse_azure_openai_endpoint("https://r.openai.azure.com", "", "") is None
    assert build_azure_openai_url("https://r.openai.azure.com/", "a b", "") == \
        "https://r.openai.azure.com/openai/deployments/a%20b/chat/completions?api-version=2024-10-21"


def test_google_thinking_level_and_budget():
    g3 = {"provider": "google", "apiKey": "k", "model": "gemini-3-pro-preview", "maxContextSize": 8192}
    b = get_provider_config(g3).build_body(
        [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "u"}],
        {"reasoning": {"mode": "max"}, "stop": ["x"], "top_p": 0.5, "top_k": 3})
    assert b["generationConfig"] == {"topP": 0.5, "topK": 3, "stopSequences": ["x"],
                                     "thinkingConfig": {"thinkingLevel": "high"}}
    assert b["systemInstruction"] == {"parts": [{"text": "s"}]}
    assert [c["role"] for c in b["contents"]] == ["model", "user"]
    flash = {**g3, "model": "gemini-2.5-flash"}
    assert get_provider_config(flash).build_body(U, {"reasoning": {"mode": "custom", "budgetTokens": 777}})[
        "generationConfig"]["thinkingConfig"] == {"thinkingBudget": 777}
    assert "generationConfig" not in get_provider_config(flash).build_body(U)
    # non streaming url
    assert get_provider_config({**flash, "streamingEnabled": False}).url.endswith(":generateContent")


def test_reasoning_normalization():
    assert normalize_reasoning_for_provider({"provider": "custom", "customEndpoint": "http://x"}, {"mode": "high"}) == {"mode": "auto"}
    ol = {"provider": "ollama"}
    assert normalize_reasoning_for_provider(ol, {"mode": "off"}) == {"mode": "off"}
    assert normalize_reasoning_for_provider(ol, {"mode": "custom", "budgetTokens": 5}) == {"mode": "auto"}
    an = {"provider": "anthropic", "model": "claude-3-5-sonnet"}
    assert normalize_reasoning_for_provider(an, {"mode": "custom", "budgetTokens": 10}) == {"mode": "custom", "budgetTokens": 1024}
    assert normalize_reasoning_for_provider(an, {"mode": "custom", "budgetTokens": 999999.7}) == {"mode": "custom", "budgetTokens": 32768}
    assert normalize_reasoning_for_provider(an, {"mode": "custom"}) == {"mode": "auto"}
    assert resolve_ingest_reasoning({}) == {"mode": "off"}
    assert resolve_ingest_reasoning({"ingestReasoning": {"mode": "low"}}) == {"mode": "low"}
    assert is_openrouter_endpoint("https://sub.openrouter.ai/x") and not is_openrouter_endpoint("https://openrouter.ai.evil")


def test_stream_line_parsers():
    assert parse_openai_line('data: {"choices":[{"delta":{"content":"a"}}]}') == "a"
    assert parse_openai_line('data: {"choices":[{"delta":{"content":""}}]}') == ""
    assert parse_openai_line("data: [DONE]") is None and parse_openai_line("event: x") is None
    assert parse_openai_line("data: {bad") is None
    assert parse_anthropic_line('data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"t"}}') == "t"
    assert parse_anthropic_line('data: {"type":"message","content":[{"text":"a"},{"text":"b"}]}') == "ab"
    assert parse_anthropic_line('data: {"type":"message","content":[]}') is None
    assert parse_anthropic_line('data: {"choices":[{"delta":{"content":"oa"}}]}') == "oa"
    assert parse_google_line('data: {"candidates":[{"content":{"parts":[{"text":"x","thought":true},{"text":"y"},{"text":"z"}]}}]}') == "yz"
    assert parse_google_line('data: {"candidates":[{"content":{"parts":[]}}]}') is None


def test_reasoning_detector():
    line = 'data: {"choices":[{"delta":{"reasoning_content":"abc\\n","reasoning":"d"}}]}'
    assert count_reasoning_chars_in_line(line) == len("abc\\n") + 1
    assert extract_reasoning_text_from_line(line) == ["abc\n", "d"]
    assert extract_reasoning_text_from_line('data: {"delta":{"type":"thinking_delta","thinking":"t"}}') == ["t"]
    assert extract_reasoning_text_from_line('data: {"candidates":[{"content":{"parts":[{"text":"th","thought":true},{"text":"v"}]}}]}') == ["th"]
    assert extract_reasoning_text_from_line("data:{}") == []
    # supplementary characters count as two UTF-16 units
    assert count_reasoning_chars_in_line('{"reasoning":"\U0001F600"}') == 2


# ------------------------------------------------------------------ js compat

def test_js_trim_and_utf16():
    assert js.js_trim("﻿  x  ") == "x"
    assert js.utf16_len("a\U0001F600") == 3
    assert js.from_utf16(js.to_utf16("a\U0001F600b")) == "a\U0001F600b"
    assert js.from_utf16("\ud800") == "�"


def test_js_json_and_stringify():
    with pytest.raises(ValueError):
        js.js_json_loads("NaN")
    assert js.dumps_js({"t": 0.0, "u": 1.5, "n": float("nan"), "s": "é\ud800"}) == '{"t":0,"u":1.5,"n":null,"s":"é\\ud800"}'
    assert js.js_str(None) == "null" and js.js_str(400.0) == "400" and js.js_str(True) == "true"
    assert js.dumps_js_pretty({"a": [1]}).startswith('{\n  "a": [')


def test_uri_coding():
    assert js.encode_uri_component("a b/ü!'()*") == "a%20b%2F%C3%BC!'()*"
    assert js.decode_uri_component("a%20b%2F") == "a b/"
    assert js.decode_uri("a%20b%2Fc") == "a b%2Fc"
    with pytest.raises(ValueError):
        js.decode_uri_component("%E0%A4%A")
    with pytest.raises(ValueError):
        js.decode_uri_component("%FF")


def test_parse_url():
    u = js.parse_url("HTTP://User:Pw@Example.COM:8080/a/b?x=1#h")
    assert (u.protocol, u.username, u.password, u.hostname, u.port, u.pathname) == \
        ("http:", "User", "Pw", "example.com", "8080", "/a/b")
    assert js.parse_url("https://example.com:443").port == ""
    assert js.parse_url("http://[::1]:80/").hostname == "[::1]"
    assert js.parse_url("nope") is None and js.parse_url("http://") is None
