"""Port of src/lib/vision-caption.test.ts (stream_chat mocked)."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llm_wiki_server.ingest import llm_client, vision_caption
from llm_wiki_server.ingest.vision_caption import CAPTION_PROMPT, caption_image

CFG = {"provider": "custom", "apiKey": "", "model": "vl-model", "ollamaUrl": "", "customEndpoint": "http://example/v1",
       "apiMode": "chat_completions", "maxContextSize": 8192}
B64 = "iVBORw0KGgo="


@pytest.fixture
def mock_stream(monkeypatch):
    calls = []
    state = {"impl": None}

    async def fake(config, messages, callbacks, signal=None, overrides=None, **kw):
        calls.append((config, messages, callbacks, signal, overrides))
        if state["impl"]:
            await state["impl"](callbacks)
        else:
            callbacks.on_done()

    monkeypatch.setattr(llm_client, "stream_chat", fake)
    fake.calls = calls
    fake.state = state
    return fake


def run(coro):
    return asyncio.run(coro)


def set_impl(m, fn):
    m.state["impl"] = fn


def test_sends_one_user_message_with_text_and_image(mock_stream):
    async def impl(cb):
        cb.on_token("a red square")
        cb.on_done()
    set_impl(mock_stream, impl)
    assert run(caption_image(B64, "image/png", CFG)) == "a red square"
    assert len(mock_stream.calls) == 1
    msgs = mock_stream.calls[0][1]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert msgs[0]["content"] == [{"type": "text", "text": CAPTION_PROMPT},
                                  {"type": "image", "mediaType": "image/png", "dataBase64": B64}]


def test_joins_tokens_and_trims(mock_stream):
    async def impl(cb):
        for t in ("  Red ", "square ", "on white  "):
            cb.on_token(t)
        cb.on_done()
    set_impl(mock_stream, impl)
    assert run(caption_image(B64, "image/png", CFG)) == "Red square on white"


def test_rethrows_stream_error(mock_stream):
    async def impl(cb):
        cb.on_error(Exception("HTTP 500: model unavailable"))
    set_impl(mock_stream, impl)
    with pytest.raises(Exception, match="HTTP 500: model unavailable"):
        run(caption_image(B64, "image/png", CFG))


def test_override_passthrough_and_defaults(mock_stream):
    run(caption_image(B64, "image/png", CFG, None, {"temperature": 0.3, "maxTokens": 256}))
    assert mock_stream.calls[0][4] == {"temperature": 0.3, "max_tokens": 256, "reasoning": {"mode": "off"}}
    run(caption_image(B64, "image/png", CFG))
    assert mock_stream.calls[1][4] == {"temperature": 0, "max_tokens": 4096, "reasoning": {"mode": "off"}}


def test_reasoning_defaults_off_even_when_main_reasoning_is_high(mock_stream):
    run(caption_image(B64, "image/png", {**CFG, "reasoning": {"mode": "high"}}))
    assert mock_stream.calls[0][4]["reasoning"] == {"mode": "off"}
    run(caption_image(B64, "image/png", {**CFG, "ingestReasoning": {"mode": "low"}}))
    assert mock_stream.calls[1][4]["reasoning"] == {"mode": "low"}


def test_codex_cli_rejected(mock_stream):
    with pytest.raises(Exception, match="does not support image input"):
        run(caption_image(B64, "image/png", {**CFG, "provider": "codex-cli"}))
    assert mock_stream.calls == []


def test_signal_forwarded(mock_stream):
    sig = llm_client.AbortSignal()
    run(caption_image(B64, "image/png", CFG, sig))
    assert mock_stream.calls[0][3] is sig


def test_prompt_regression_guard():
    for frag in ("factually", "visible text verbatim", "Do NOT speculate", "no markdown"):
        assert frag in CAPTION_PROMPT


def _prompt(m, i=0):
    return m.calls[i][1][0]["content"][0]["text"]


def test_language_instruction(mock_stream):
    run(caption_image(B64, "image/png", CFG, None, {"outputLanguage": "German"}))
    t = _prompt(mock_stream)
    assert "Write the description in German" in t and "Preserve visible text verbatim" in t


def test_language_instruction_on_context_prompt(mock_stream):
    run(caption_image(B64, "image/png", CFG, None,
                      {"contextBefore": "Figure 3: Q2 revenue chart", "outputLanguage": "German"}))
    t = _prompt(mock_stream)
    assert "Figure 3: Q2 revenue chart" in t and "Write the description in German" in t


def test_auto_language_means_no_directive(mock_stream):
    run(caption_image(B64, "image/png", CFG, None, {"outputLanguage": "auto"}))
    assert _prompt(mock_stream) == CAPTION_PROMPT


def test_whitespace_context_uses_plain_prompt(mock_stream):
    run(caption_image(B64, "image/png", CFG, None, {"contextBefore": "  \n  ", "contextAfter": ""}))
    assert _prompt(mock_stream) == CAPTION_PROMPT


def test_context_aware_prompt_when_either_side_present(mock_stream):
    import re
    run(caption_image(B64, "image/png", CFG, None,
                      {"contextBefore": "Figure 3: Q2 revenue chart", "contextAfter": ""}))
    t = _prompt(mock_stream)
    for frag in ("Text before image", "Text after image", "MAY help describe the image", "MAY ALSO be unrelated"):
        assert frag in t
    assert "Figure 3: Q2 revenue chart" in t
    assert re.search(r"Text after image ---\s*\(none\)", t)


def test_end_to_end_over_mock_transport():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":" a cat "}}]}\n\ndata: [DONE]\n')

    out = run(caption_image(B64, "image/png", CFG, transport=httpx.MockTransport(handler)))
    assert out == "a cat"
    assert seen["body"]["messages"][0]["content"][1]["image_url"]["url"] == f"data:image/png;base64,{B64}"
    assert seen["body"]["temperature"] == 0 and seen["body"]["max_tokens"] == 4096
