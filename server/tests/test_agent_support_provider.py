from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llm_wiki_server.agent import provider as pv
from llm_wiki_server.agent.provider import LlmClient, LlmConfig, LlmReasoningConfig, ProviderError
from llm_wiki_server.agent.types import AgentImage


def cfg(provider="openai", **kw):
    base = dict(provider=provider, api_key="key", model="model/name", ollama_url="http://localhost:11434/v1",
                custom_endpoint="https://example.com/v1")
    base.update(kw)
    return LlmConfig(**base)


def reasoning(mode, budget=None):
    return LlmReasoningConfig(mode, budget)


def client(config, handler):
    return LlmClient(config, httpx.MockTransport(handler))


# ---------------------------------------------------------------- config

def test_config_from_dict():
    c = LlmConfig.from_dict({"provider": "custom", "apiKey": "key", "model": "m", "ollamaUrl": "",
                             "customEndpoint": "https://example.com/v1", "apiMode": "anthropic_messages",
                             "reasoning": {"mode": "custom", "budgetTokens": 4096}})
    assert c.api_key == "key" and c.api_mode == "anthropic_messages" and c.reasoning.budget_tokens == 4096
    assert c.max_tokens is None and c.streaming_enabled is None
    assert LlmConfig.from_dict({"provider": "openai", "streamingEnabled": False}).streaming_enabled is False
    with pytest.raises(ValueError):
        LlmConfig.from_dict({})
    assert LlmConfig.from_dict(c.to_dict()) == c


def test_usability():
    assert cfg("openai").is_usable_for_backend_http()
    assert cfg("custom").is_usable_for_backend_http() and cfg("ollama").is_usable_for_backend_http()
    assert not cfg("claude-code").is_usable_for_backend_http()
    assert not cfg("openai", api_key="").is_usable_for_backend_http()
    assert not cfg("weird").is_usable_for_backend_http()


# ---------------------------------------------------------------- headers

def test_custom_headers_validated_and_auth_wins():
    c = cfg("openai", custom_headers={"X-Tenant-ID": "team-a", "authorization": "Custom secret"})
    h = pv.openai_headers(c, "https://api.openai.com/v1/chat/completions")
    assert h.get("x-tenant-id") == "team-a" and h.get("authorization") == "Bearer key"
    c.custom_headers["X-Bad"] = "ok\r\nInjected: yes"
    with pytest.raises(ProviderError):
        pv.custom_headers(c)
    with pytest.raises(ProviderError):
        pv.custom_headers(cfg(custom_headers={"bad name": "v"}))


def test_azure_uses_api_key_header():
    h = pv.openai_headers(cfg("azure"), "https://r.openai.azure.com/openai/deployments/x/chat/completions")
    assert h.get("api-key") == "key" and h.get("authorization") is None


def test_minimax_bearer_and_standard_x_api_key():
    c = cfg("custom", custom_headers={"authorization": "Custom secret", "x-api-key": "Stale"})
    h = pv.anthropic_headers(c, "https://api.minimaxi.com/anthropic/v1/messages")
    assert h.get("authorization") == "Bearer key" and h.get("x-api-key") is None
    assert h.get("anthropic-version") == pv.ANTHROPIC_VERSION
    h = pv.anthropic_headers(c, "https://api.minimax.io/anthropic/v1/messages")
    assert h.get("authorization") == "Bearer key"
    a = cfg("anthropic", custom_headers={"authorization": "Stale secret"})
    h = pv.anthropic_headers(a, "https://api.anthropic.com/v1/messages")
    assert h.get("x-api-key") == "key" and h.get("authorization") == "Stale secret"
    h = pv.anthropic_headers(cfg("custom"), "https://gateway.example.com/minimax.io/anthropic/v1/messages")
    assert h.get("x-api-key") == "key" and h.get("authorization") is None


@pytest.mark.parametrize("url", [
    "https://coding.dashscope.aliyuncs.com/apps/anthropic/v1/messages",
    "https://token-plan-cn.xiaomimimo.com/anthropic/v1/messages",
    "https://api.kimi.com/coding/v1/messages",
    "https://api.moonshot.ai/anthropic/v1/messages",
    "https://api.moonshot.cn/anthropic/v1/messages",
])
def test_bearer_detection(url):
    h = pv.anthropic_headers(cfg("custom"), url)
    assert h.get("authorization") == "Bearer key" and h.get("x-api-key") is None


def test_google_headers():
    assert pv.google_headers(cfg("google")).get("x-goog-api-key") == "key"


# ---------------------------------------------------------------- urls

def test_urls():
    assert pv.build_ollama_url("http://localhost:11434") == "http://localhost:11434/v1/chat/completions"
    assert pv.build_ollama_url("http://localhost:11434/v1") == "http://localhost:11434/v1/chat/completions"
    assert pv.build_ollama_url("http://localhost:11434/v1/chat/completions/") == "http://localhost:11434/v1/chat/completions"
    for b in ("https://api.anthropic.com", "https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"):
        assert pv.build_anthropic_url(b) == "https://api.anthropic.com/v1/messages"
    assert pv.build_anthropic_url("https://api.minimax.io") == "https://api.minimax.io/anthropic/v1/messages"
    assert pv.build_anthropic_url("https://api.minimaxi.com/v1/messages") == "https://api.minimaxi.com/anthropic/v1/messages"
    assert pv.build_anthropic_url("https://api.minimax.io/anthropic") == "https://api.minimax.io/anthropic/v1/messages"
    c = cfg("azure", custom_endpoint="https://resource.openai.azure.com", model="my deployment")
    assert pv.build_azure_url(c) == ("https://resource.openai.azure.com/openai/deployments/my%20deployment/"
                                     "chat/completions?api-version=2024-10-21")
    c2 = cfg("azure", custom_endpoint="https://x.openai.azure.com/openai/deployments/d?foo=1", azure_api_version="v9")
    assert pv.build_azure_url(c2).endswith("?foo=1&api-version=v9")
    with pytest.raises(ProviderError, match="Azure endpoint is required"):
        pv.build_azure_url(cfg("azure", custom_endpoint=""))
    assert pv.build_custom_openai_url(cfg("custom")) == "https://example.com/v1/chat/completions"
    assert pv.build_custom_openai_url(cfg("custom", custom_endpoint="https://e.com/chat/completions")) == "https://e.com/chat/completions"
    assert pv.url_encode_path_segment("a b/é") == "a%20b%2F%C3%A9"


# ---------------------------------------------------------------- bodies / reasoning

def body(c, stream=False, include_model=True):
    return LlmClient(c).openai_like_body("system", "user", [], include_model, stream)


def test_openai_body_basic_and_images():
    b = LlmClient(cfg()).openai_like_body("sys", "hi", [AgentImage("image/png", "AAA")], True, True)
    assert b["model"] == "model/name" and b["stream"] is True and b["max_tokens"] == 2048
    assert b["messages"][1]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}


def test_max_tokens_clamped():
    assert LlmClient(cfg(max_tokens=10)).max_output_tokens() == 256
    assert LlmClient(cfg(max_tokens=10**6)).max_output_tokens() == 32768


def test_reasoning_openai():
    assert "reasoning_effort" not in body(cfg(reasoning=reasoning("off")))
    b = body(cfg(model="o3-mini", reasoning=reasoning("high")))
    assert b["reasoning_effort"] == "high" and "max_tokens" not in b and "max_completion_tokens" in b
    assert "reasoning_effort" not in body(cfg("custom", model="qwen3", reasoning=reasoning("high")))
    assert body(cfg("ollama", reasoning=reasoning("off")))["reasoning_effort"] == "none"
    assert body(cfg("ollama", reasoning=reasoning("max")))["reasoning_effort"] == "high"


def test_structured_task_config():
    c = LlmClient(cfg(reasoning=reasoning("high"))).structured_task_config(16384)
    b = c.openai_like_body("s", "u", [], True, False)
    assert b["max_tokens"] == 16384 and "reasoning_effort" not in b and c.config.reasoning.mode == "off"
    d = LlmClient(cfg("custom", custom_endpoint="https://api.deepseek.com/v1/chat/completions",
                      model="deepseek-v4-flash")).structured_task_config(8192)
    assert d.openai_like_body("s", "u", [], True, False)["thinking"] == {"type": "disabled"}


def test_azure_max_completion_tokens():
    b = body(cfg("azure", model="dep", azure_model_family="gpt5"), include_model=False)
    assert b["max_completion_tokens"] == 2048 and "max_tokens" not in b
    b = body(cfg("azure", model="gpt-4o", azure_model_family="standard"), include_model=False)
    assert "max_tokens" in b and "max_completion_tokens" not in b
    b = body(cfg("custom", custom_endpoint="https://e.openai.azure.com/openai/deployments/prod",
                 model="dep", azure_model_family="gpt5"), include_model=False)
    assert "max_completion_tokens" in b


def test_anthropic_reasoning():
    for model, mode, effort in (("claude-opus-4-7", "high", "high"), ("claude-sonnet-4-6", "medium", "medium")):
        b = LlmClient(cfg("anthropic", model=model, reasoning=reasoning(mode))).anthropic_like_body("s", "u", [], False)
        assert b["thinking"] == {"type": "adaptive"} and b["output_config"] == {"effort": effort}
    b = LlmClient(cfg("anthropic", model="claude-sonnet-4-5", reasoning=reasoning("custom", 5000))) \
        .anthropic_like_body("s", "u", [], True)
    assert b["thinking"] == {"type": "enabled", "budget_tokens": 5000} and b["max_tokens"] == 6024 and b["stream"] is True
    b = LlmClient(cfg("anthropic", reasoning=reasoning("off"))).anthropic_like_body("s", "u", [], False)
    assert "thinking" not in b and "stream" not in b
    assert b["system"][0]["cache_control"] == {"type": "ephemeral"}
    b = LlmClient(cfg("custom", reasoning=reasoning("high"))).anthropic_like_body("s", "u", [], False)
    assert "thinking" not in b


def test_google_reasoning():
    g = lambda m, r: LlmClient(cfg("google", model=m, reasoning=r)).google_body("s", "u", [])["generationConfig"]
    assert g("gemini-3.5-flash", reasoning("medium"))["thinkingConfig"] == {"thinkingLevel": "medium"}
    assert "thinkingConfig" not in g("gemini-2.5-pro", reasoning("off"))
    assert g("gemini-2.5-flash", reasoning("off"))["thinkingConfig"] == {"thinkingBudget": 0}
    assert g("gemini-2.5-flash", reasoning("custom", 77))["thinkingConfig"] == {"thinkingBudget": 77}
    assert g("gemini-2.5-flash", None) == {"maxOutputTokens": 2048}


# ---------------------------------------------------------------- SSE parsing

def test_delta_parsers():
    assert pv.parse_openai_delta('{"choices":[{"delta":{"content":"hello"}}]}') == "hello"
    assert pv.parse_openai_delta('{"choices":[{"delta":{"role":"assistant"}}]}') is None
    assert pv.parse_openai_delta("nope") is None
    assert pv.parse_anthropic_delta('{"type":"content_block_delta","delta":{"type":"text_delta","text":"world"}}') == "world"
    assert pv.parse_anthropic_delta('{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"x"}}') is None
    assert pv.parse_google_delta('{"candidates":[{"content":{"parts":[{"text":"gem"},{"text":"ini"}]}}]}') == "gemini"


def test_sse_line_without_trailing_newline():
    got = []
    c = pv.SseCollector(pv.parse_openai_delta, got.append)
    c.feed(b'data: {"choices":[{"delta":{"content":"tail"}}]}')
    assert c.full == ""
    assert c.finish() == "tail" and got == ["tail"]


def test_sse_multibyte_across_chunks():
    line = 'data: {"choices":[{"delta":{"content":"煤矿"}}]}\r\n'.encode()
    split = line.index("矿".encode()) + 1
    got = []
    c = pv.SseCollector(pv.parse_openai_delta, got.append)
    c.feed(line[:split])
    assert c.full == ""
    c.feed(line[split:])
    assert c.full == "煤矿" and got == ["煤矿"]


def test_sse_ignores_done_and_empty_error():
    c = pv.SseCollector(pv.parse_openai_delta, lambda d: None)
    c.feed(b"event: x\ndata: [DONE]\n\n")
    with pytest.raises(ProviderError, match="did not contain assistant content"):
        c.finish()


def test_trim_error_body():
    t = pv.trim_error_body("错误" * 600)
    assert t.endswith("...") and len(t) == 803


# ---------------------------------------------------------------- end-to-end with MockTransport

def sse(*events):
    return ("".join(f"data: {e}\n\n" for e in events)).encode()


def test_openai_nonstream_request_shape():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["headers"] = req.headers
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "  answer \n"}}]})

    out = asyncio.run(client(cfg(), handler).generate_text("sys", "user", []))
    assert out == "answer"
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer key"
    assert seen["headers"]["content-type"] == "application/json"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"} and seen["body"]["stream"] is False


def test_sync_generate_and_errors():
    c = client(cfg(), lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
    assert c.generate_text_sync("s", "u") == "ok"
    c = client(cfg(), lambda r: httpx.Response(401, text="  bad key  "))
    with pytest.raises(ProviderError) as ei:
        c.generate_text_sync("s", "u")
    assert str(ei.value) == "LLM HTTP 401 Unauthorized: bad key"
    c = client(cfg(), lambda r: httpx.Response(200, text="not json"))
    with pytest.raises(ProviderError, match="^Invalid LLM JSON"):
        c.generate_text_sync("s", "u")
    c = client(cfg(), lambda r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(ProviderError, match="did not contain assistant content"):
        asyncio.run(c.generate_text("s", "u"))

    def boom(r):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError, match="^LLM request failed: refused"):
        client(cfg(), boom).generate_text_sync("s", "u")


def test_unsupported_provider_and_minimax_images():
    with pytest.raises(ProviderError, match="Provider 'claude-code' is not supported by the backend HTTP Agent yet"):
        client(cfg("claude-code"), lambda r: httpx.Response(200)).generate_text_sync("s", "u")
    with pytest.raises(ProviderError, match="MiniMax official"):
        client(cfg("minimax"), lambda r: httpx.Response(200)).generate_text_sync("s", "u", [AgentImage("image/png", "A")])


def test_openai_stream_async():
    def handler(req):
        assert json.loads(req.content)["stream"] is True
        data = sse('{"choices":[{"delta":{"content":"Hel"}}]}', '{"choices":[{"delta":{"content":"lo "}}]}',
                   '{"choices":[{"delta":{"reasoning_content":"think"}}]}', "[DONE]")
        return httpx.Response(200, content=data, headers={"content-type": "text/event-stream"})

    got = []
    out = asyncio.run(client(cfg(), handler).generate_text_stream("s", "u", [], got.append))
    assert out == "Hello" and got == ["Hel", "lo "]


def test_openai_stream_sync_http_error():
    c = client(cfg(), lambda r: httpx.Response(429, text="slow down"))
    with pytest.raises(ProviderError, match="^LLM HTTP 429 Too Many Requests: slow down$"):
        c.generate_text_stream_sync("s", "u", [], lambda d: None)


def test_streaming_disabled_falls_back_to_single_delta():
    def handler(req):
        assert json.loads(req.content)["stream"] is False
        return httpx.Response(200, json={"choices": [{"message": {"content": "whole"}}]})

    got = []
    out = asyncio.run(client(cfg(streaming_enabled=False), handler).generate_text_stream("s", "u", [], got.append))
    assert out == "whole" and got == ["whole"]


def test_anthropic_stream_and_complete():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["key"] = req.headers.get("x-api-key")
        if json.loads(req.content).get("stream"):
            return httpx.Response(200, content=sse(
                '{"type":"message_start"}',
                '{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hm"}}',
                '{"type":"content_block_delta","delta":{"type":"text_delta","text":"A"}}',
                '{"type":"content_block_delta","delta":{"type":"text_delta","text":"B"}}'))
        return httpx.Response(200, json={"content": [{"type": "thinking", "thinking": "x"},
                                                      {"type": "text", "text": "Hi "}, {"type": "text", "text": "there"}]})

    c = client(cfg("anthropic"), handler)
    got = []
    assert c.generate_text_stream_sync("s", "u", [], got.append) == "AB" and got == ["A", "B"]
    assert seen["url"] == "https://api.anthropic.com/v1/messages" and seen["key"] == "key"
    assert c.generate_text_sync("s", "u") == "Hi there"


def test_custom_anthropic_mode_and_minimax_default_url():
    seen = []

    def handler(req):
        seen.append((str(req.url), req.headers.get("authorization")))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "x"}]})

    client(cfg("custom", api_mode="anthropic_messages", custom_endpoint="https://api.kimi.com/coding"), handler) \
        .generate_text_sync("s", "u")
    client(cfg("minimax", custom_endpoint=""), handler).generate_text_sync("s", "u")
    assert seen[0] == ("https://api.kimi.com/coding/v1/messages", "Bearer key")
    assert seen[1] == ("https://api.minimax.io/anthropic/v1/messages", "Bearer key")


def test_google_urls_and_streaming():
    urls = []

    def handler(req):
        urls.append(str(req.url))
        assert req.headers["x-goog-api-key"] == "key"
        if "streamGenerateContent" in str(req.url):
            return httpx.Response(200, content=sse('{"candidates":[{"content":{"parts":[{"text":"g1"}]}}]}',
                                                   '{"candidates":[{"content":{"parts":[{"text":"g2"}]}}]}'))
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "full "}, {"text": "text"}]}}]})

    c = client(cfg("google"), handler)
    assert c.generate_text_sync("s", "u") == "full text"
    assert asyncio.run(c.generate_text_stream("s", "u", [], lambda d: None)) == "g1g2"
    assert urls[0] == "https://generativelanguage.googleapis.com/v1beta/models/model%2Fname:generateContent"
    assert urls[1].endswith("model%2Fname:streamGenerateContent?alt=sse")


def test_azure_and_ollama_requests():
    seen = []

    def handler(req):
        seen.append((str(req.url), json.loads(req.content), dict(req.headers)))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client(cfg("azure", custom_endpoint="https://r.openai.azure.com", model="dep"), handler).generate_text_sync("s", "u")
    client(cfg("ollama", api_key=""), handler).generate_text_sync("s", "u")
    assert seen[0][0].startswith("https://r.openai.azure.com/openai/deployments/dep/chat/completions?api-version=")
    assert "model" not in seen[0][1] and seen[0][2]["api-key"] == "key"
    assert seen[1][0] == "http://localhost:11434/v1/chat/completions"
    assert seen[1][1]["model"] == "model/name" and "authorization" not in seen[1][2]


def test_stream_cancel():
    class Cancel:
        def is_cancelled(self):
            return True

    c = client(cfg(), lambda r: httpx.Response(200, content=sse('{"choices":[{"delta":{"content":"a"}}]}')))
    with pytest.raises(ProviderError, match="Agent turn cancelled"):
        c.generate_text_stream_sync("s", "u", [], lambda d: None, cancel=Cancel())


def test_proxy_env_honored(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    # no transport => real client; connecting through the dead proxy must fail with a request error
    with pytest.raises(ProviderError, match="^LLM request failed"):
        LlmClient(cfg()).generate_text_sync("s", "u")
