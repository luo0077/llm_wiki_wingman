"""Port of src/lib/llm-client.test.ts (+ provider wire-format coverage) onto httpx.MockTransport."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import httpx
import pytest

from llm_wiki_server.ingest import llm_client
from llm_wiki_server.ingest.llm_client import (
    AbortSignal,
    StreamCallbacks,
    StreamCompletion,
    collect_stream_chat,
    is_fetch_network_error,
    is_reasoning_only_response_error,
    split_final_stream_records,
    stream_chat,
)


def run(coro):
    return asyncio.run(coro)


CFG = {
    "provider": "ollama", "apiKey": "", "model": "qwen3:8b", "ollamaUrl": "http://localhost:11434",
    "customEndpoint": "", "apiMode": "chat_completions", "maxContextSize": 8192,
}
CUSTOM = {**CFG, "provider": "custom", "model": "local-openai-model", "customEndpoint": "http://127.0.0.1:13305/v1"}
MSGS = [{"role": "user", "content": "hi"}]


def sse_token(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})


class Rec:
    def __init__(self):
        self.tokens: List[str] = []
        self.reasoning: List[str] = []
        self.done: List[Any] = []
        self.errors: List[Exception] = []

    def cb(self) -> StreamCallbacks:
        return StreamCallbacks(self.tokens.append, lambda *a: self.done.append(a[0] if a else None),
                               self.errors.append, self.reasoning.append)


def go(config, handler, overrides=None, signal=None, messages=None):
    rec = Rec()
    calls: List[httpx.Request] = []

    def wrapped(req: httpx.Request):
        calls.append(req)
        return handler(req)

    run(stream_chat(config, messages or MSGS, rec.cb(), signal, overrides,
                    transport=httpx.MockTransport(wrapped)))
    return rec, calls


# ------------------------------------------------------------------ error classifier

def test_is_fetch_network_error_shapes():
    assert is_fetch_network_error(Exception("Load failed"))
    assert is_fetch_network_error(TypeError("Failed to fetch"))
    assert is_fetch_network_error(TypeError("NetworkError when attempting to fetch resource."))
    assert is_fetch_network_error(Exception("The network error occurred while reading"))
    assert is_fetch_network_error(httpx.ConnectError("boom"))
    e = llm_client.AbortError("The operation was aborted.")
    assert not is_fetch_network_error(e)
    assert not is_fetch_network_error(Exception("HTTP 401: Unauthorized"))
    for v in ("boom", None, {"message": "Load failed"}):
        assert not is_fetch_network_error(v)


def test_is_reasoning_only_response_error():
    assert is_reasoning_only_response_error(Exception(
        "Model produced 2,176 characters of reasoning / chain-of-thought, but no actual response content. Try again."))
    assert not is_reasoning_only_response_error(Exception("plain provider error"))


# ------------------------------------------------------------------ buffered streaming

def test_json_endpoint_error_inside_http_200():
    rec, _ = go(CUSTOM, lambda r: httpx.Response(
        200, json={"error": {"code": 400, "message": "request exceeds available context"}},
        headers={"Content-Type": "text/event-stream"}))
    assert [str(e) for e in rec.errors] == ["LLM endpoint error 400: request exceeds available context"]
    assert rec.tokens == [] and rec.done == []


def test_retry_without_temperature_for_custom_endpoint():
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        if n["i"] == 1:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: temperature"}})
        return httpx.Response(200, text="\n\n".join([sse_token("retried"), "data: [DONE]"]))

    rec, calls = go(CUSTOM, handler, {"temperature": 0.1, "max_tokens": 512})
    assert len(calls) == 2
    first = json.loads(calls[0].content)
    assert first["temperature"] == 0.1 and first["max_tokens"] == 512
    retry = json.loads(calls[1].content)
    assert "temperature" not in retry and retry["max_tokens"] == 512
    assert rec.tokens == ["retried"] and len(rec.done) == 1 and rec.errors == []


def test_open_body_cancelled_after_sse_endpoint_error():
    closed = {"v": False}

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"error":{"message":"stream failed"}}\n'
            await asyncio.sleep(30)  # never finishes on its own

        async def aclose(self):
            closed["v"] = True

    rec, _ = go(CUSTOM, lambda r: httpx.Response(200, stream=Body(), headers={"Content-Type": "text/event-stream"}))
    assert [str(e) for e in rec.errors] == ["LLM endpoint error: stream failed"]
    assert closed["v"] is True
    assert rec.tokens == [] and rec.done == []


def test_parses_every_record_from_fully_buffered_sse_body():
    body = "\n".join([sse_token("Hello"), "", sse_token(" world"), "", "data: [DONE]", ""])
    rec, _ = go(CUSTOM, lambda r: httpx.Response(200, text=body))
    assert rec.tokens == ["Hello", " world"] and len(rec.done) == 1 and rec.errors == []


def test_token_limit_completion_keeps_content():
    body = "\n\n".join([sse_token("partial analysis"),
                        "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}),
                        "data: [DONE]"])
    rec, _ = go(CUSTOM, lambda r: httpx.Response(200, text=body))
    assert rec.tokens == ["partial analysis"]
    assert rec.done == [StreamCompletion("length", True)]
    assert rec.done[0] == {"finishReason": "length", "truncated": True}


@pytest.mark.parametrize("terminal,reason", [
    ('data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"}}', "max_tokens"),
    ('data: {"candidates":[{"finishReason":"MAX_TOKENS"}]}', "MAX_TOKENS"),
])
def test_provider_token_limit_reasons(terminal, reason):
    rec, _ = go(CUSTOM, lambda r: httpx.Response(200, text="\n\n".join([sse_token("partial"), terminal])))
    assert rec.done == [StreamCompletion(reason, True)]


def test_normal_stop_is_complete():
    body = "\n\n".join([sse_token("complete"), 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}', "data: [DONE]"])
    rec, _ = go(CUSTOM, lambda r: httpx.Response(200, text=body))
    assert rec.done == [StreamCompletion("stop", False)]


def test_token_limit_non_streaming_response():
    rec, _ = go({**CUSTOM, "streamingEnabled": False}, lambda r: httpx.Response(
        200, json={"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}))
    assert rec.done == [StreamCompletion("length", True)]


def test_empty_token_limited_non_streaming_response_is_truncated():
    rec, _ = go({**CUSTOM, "streamingEnabled": False}, lambda r: httpx.Response(
        200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}))
    assert rec.done == [StreamCompletion("length", True)] and rec.errors == []


def test_normalizes_escaped_separators_in_buffered_body():
    body = "\\n\\n".join([sse_token("Hello"), sse_token(" world"), "data: [DONE]"])
    rec, _ = go(CUSTOM, lambda r: httpx.Response(200, text=body))
    assert rec.tokens == ["Hello", " world"] and len(rec.done) == 1 and rec.errors == []


def test_does_not_split_separator_like_text_inside_json_content():
    content = "first line\n\ndata: still model output"
    body = "\\n\\n".join([sse_token(content), "data: [DONE]"])
    rec, _ = go(CUSTOM, lambda r: httpx.Response(200, text=body))
    assert rec.tokens == [content] and len(rec.done) == 1 and rec.errors == []


def test_split_final_stream_records_plain_newlines():
    assert split_final_stream_records("a\r\nb\nc") == ["a", "b", "c"]


# ------------------------------------------------------------------ non-streaming

def test_non_streaming_emits_one_token_and_sends_stream_false():
    rec, calls = go({**CFG, "streamingEnabled": False}, lambda r: httpx.Response(
        200, json={"choices": [{"message": {"content": "complete answer"}}]}))
    assert rec.tokens == ["complete answer"] and len(rec.done) == 1 and rec.errors == []
    assert json.loads(calls[0].content)["stream"] is False


def test_non_streaming_empty_response_is_an_error():
    rec, _ = go({**CFG, "streamingEnabled": False}, lambda r: httpx.Response(
        200, json={"choices": [{"message": {"content": ""}}]}))
    assert "empty non-streaming response" in str(rec.errors[0])


# ------------------------------------------------------------------ http / network errors

def test_http_error_detail_and_azure_404_hint():
    rec, _ = go(CUSTOM, lambda r: httpx.Response(500, text="boom"))
    assert str(rec.errors[0]) == "HTTP 500: Internal Server Error — boom"
    az = {**CFG, "provider": "azure", "customEndpoint": "https://res.openai.azure.com", "model": "dep", "apiKey": "k"}
    rec, calls = go(az, lambda r: httpx.Response(404, text="nope"))
    assert "Azure 404 usually means the deployment name is wrong" in str(rec.errors[0])
    assert str(calls[0].url).startswith("https://res.openai.azure.com/openai/deployments/dep/chat/completions")
    assert calls[0].headers["api-key"] == "k"


def test_network_error_message():
    def boom(req):
        raise httpx.ConnectError("refused")
    rec, _ = go(CUSTOM, boom)
    assert str(rec.errors[0]) == ("Network error reaching http://127.0.0.1:13305/v1/chat/completions. "
                                  "Check endpoint URL, API key, and connectivity.")


def test_mid_stream_network_error():
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (sse_token("a") + "\n").encode()
            raise httpx.ReadError("reset")

    rec, _ = go(CUSTOM, lambda r: httpx.Response(200, stream=Body()))
    assert rec.tokens == ["a"]
    assert str(rec.errors[0]) == "Connection lost during streaming. Try again."


def test_reasoning_only_stream_diagnostic():
    big = "x" * 250
    lines = "\n\n".join("data: " + json.dumps({"choices": [{"delta": {"reasoning_content": big}}]}) for _ in range(1))
    rec, _ = go(CUSTOM, lambda r: httpx.Response(200, text=lines))
    assert rec.reasoning == [big]
    assert is_reasoning_only_response_error(rec.errors[0])
    assert "250 characters of reasoning" in str(rec.errors[0])


def test_empty_content_token_is_forwarded():
    rec, _ = go(CUSTOM, lambda r: httpx.Response(200, text=sse_token("") + "\n"))
    assert rec.tokens == [""]


# ------------------------------------------------------------------ cancel / timeout

class Hang(httpx.AsyncByteStream):
    def __init__(self, ready: "asyncio.Event | None" = None):
        self.ready = ready

    async def __aiter__(self):
        if False:
            yield b""
        await asyncio.sleep(60)


def test_user_cancel_mid_stream_is_silent_done():
    sig = AbortSignal()

    async def main():
        rec = Rec()
        transport = httpx.MockTransport(lambda r: httpx.Response(200, stream=Hang()))
        task = asyncio.ensure_future(stream_chat(CFG, MSGS, rec.cb(), sig, {}, transport=transport))
        await asyncio.sleep(0.1)
        sig.abort()
        await task
        return rec

    rec = run(main())
    assert rec.done == [None] and rec.errors == []


def test_pre_aborted_signal_never_errors():
    sig = AbortSignal()
    sig.abort()
    rec, _ = go(CFG, lambda r: httpx.Response(200, stream=Hang()), signal=sig)
    assert rec.done == [None] and rec.errors == []


def test_backstop_timeout_maps_to_actionable_error(monkeypatch):
    monkeypatch.setattr(llm_client, "_SECONDS_PER_MINUTE", 0.01)
    rec, _ = go(CFG, lambda r: httpx.Response(200, stream=Hang()))
    assert len(rec.errors) == 1 and "timed out after 30 min" in str(rec.errors[0])
    assert rec.done == []


def test_request_timeout_minutes_is_configurable(monkeypatch):
    monkeypatch.setattr(llm_client, "_SECONDS_PER_MINUTE", 0.01)
    rec, _ = go({**CFG, "requestTimeoutMinutes": 90}, lambda r: httpx.Response(200, stream=Hang()))
    assert "timed out after 90 min" in str(rec.errors[0])


@pytest.mark.parametrize("msg", ["Request cancelled", "request cancelled", "Request canceled"])
def test_bare_cancel_messages_are_silent(msg):
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise Exception(msg)
            yield b""

    rec, _ = go(CFG, lambda r: httpx.Response(200, stream=Body()))
    assert rec.done == [None] and rec.errors == []


@pytest.mark.parametrize("msg", ["request cancelled", "Request canceled"])
def test_pre_fetch_cancel_spellings_are_silent(msg):
    def boom(req):
        raise Exception(msg)
    rec, _ = go(CFG, boom)
    assert rec.done == [None] and rec.errors == []


# ------------------------------------------------------------------ helpers / providers

def test_collect_stream_chat_returns_text_and_raises():
    body = "\n\n".join([sse_token("a"), sse_token("b"), "data: [DONE]"])
    out = run(collect_stream_chat(CUSTOM, MSGS, transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body))))
    assert out == "ab"
    with pytest.raises(Exception, match="HTTP 500"):
        run(collect_stream_chat(CUSTOM, MSGS, transport=httpx.MockTransport(lambda r: httpx.Response(500, text="x"))))


def test_async_callbacks_and_dict_callbacks():
    seen: List[str] = []

    async def on_token(t):
        seen.append(t)

    body = "\n\n".join([sse_token("z"), "data: [DONE]"])
    run(stream_chat(CUSTOM, MSGS, {"onToken": on_token, "onDone": lambda c: seen.append("done")}, None, None,
                    transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body))))
    assert seen == ["z", "done"]


def test_cli_provider_without_transport_reports_error():
    rec, _ = go({**CFG, "provider": "claude-code"}, lambda r: httpx.Response(200))
    assert "not available on the headless server" in str(rec.errors[0])


def test_anthropic_wire_headers_and_body():
    cfg = {"provider": "anthropic", "apiKey": "sk", "model": "claude-3-5-haiku", "maxContextSize": 8192}
    sse = "\n\n".join([
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}'])
    rec, calls = go(cfg, lambda r: httpx.Response(200, text=sse), {"temperature": 0.2, "reasoning": {"mode": "off"}},
                    messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}])
    req = calls[0]
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "sk" and req.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(req.content)
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["temperature"] == 0.2 and body["model"] == "claude-3-5-haiku" and body["stream"] is True
    assert rec.tokens == ["hi"] and rec.done == [StreamCompletion("end_turn", False)]


def test_google_wire_url_and_generation_config():
    cfg = {"provider": "google", "apiKey": "gk", "model": "google/gemini-2.5-flash", "maxContextSize": 8192}
    sse = 'data: {"candidates":[{"content":{"parts":[{"text":"a","thought":true},{"text":"b"}]}}]}'
    rec, calls = go(cfg, lambda r: httpx.Response(200, text=sse),
                    {"temperature": 0.1, "max_tokens": 9, "reasoning": {"mode": "off"}})
    assert str(calls[0].url) == ("https://generativelanguage.googleapis.com/v1beta/models/"
                                 "google%2Fgemini-2.5-flash:streamGenerateContent?alt=sse")
    assert calls[0].headers["x-goog-api-key"] == "gk"
    body = json.loads(calls[0].content)
    assert body["generationConfig"] == {"temperature": 0.1, "maxOutputTokens": 9, "thinkingConfig": {"thinkingBudget": 0}}
    assert rec.tokens == ["b"] and rec.reasoning == ["a"]


def test_multimodal_openai_body_and_vision_gate():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "d"},
                                         {"type": "image", "mediaType": "image/png", "dataBase64": "AAAA"}]}]
    rec, calls = go(CUSTOM, lambda r: httpx.Response(200, text="data: [DONE]\n"), messages=msgs)
    body = json.loads(calls[0].content)
    assert body["messages"][0]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    # BigModel non-vision model: error surfaces through on_error before any request
    big = {**CUSTOM, "customEndpoint": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4"}
    rec, calls = go(big, lambda r: httpx.Response(200), messages=msgs)
    assert calls == [] and "GLM vision models" in str(rec.errors[0])


def test_strict_openai_and_ollama_reasoning_adaptations():
    oa = {"provider": "openai", "apiKey": "k", "model": "gpt-5-mini", "maxContextSize": 8192}
    rec, calls = go(oa, lambda r: httpx.Response(200, text="data: [DONE]\n"),
                    {"temperature": 0.1, "max_tokens": 50, "reasoning": {"mode": "high"}})
    b = json.loads(calls[0].content)
    assert b["max_completion_tokens"] == 50 and "max_tokens" not in b and "temperature" not in b
    assert b["reasoning_effort"] == "high" and calls[0].headers["authorization"] == "Bearer k"
    rec, calls = go(CFG, lambda r: httpx.Response(200, text="data: [DONE]\n"), {"reasoning": {"mode": "off"}})
    assert json.loads(calls[0].content)["reasoning_effort"] == "none"
    assert str(calls[0].url) == "http://localhost:11434/v1/chat/completions"
    assert calls[0].headers["origin"] == "http://localhost"
