"""Port of src/lib/llm-providers.ts: wire formats for every HTTP LLM provider.

Everything here is a pure function of an ``LlmConfig`` dict (TS camelCase keys:
``provider``, ``apiKey``, ``model``, ``ollamaUrl``, ``customEndpoint``, ``azureApiVersion``,
``azureModelFamily``, ``apiMode``, ``reasoning``, ``maxContextSize``, ``customHeaders``,
``streamingEnabled`` ...), a list of ``ChatMessage`` dicts (``{"role", "content"}`` where content
is a ``str`` or a list of ``{"type": "text", "text"}`` / ``{"type": "image", "mediaType",
"dataBase64"}`` blocks) and a ``RequestOverrides`` dict (``temperature``, ``top_p``, ``top_k``,
``max_tokens``, ``stop``, ``reasoning``).  A ``None`` value in the overrides means "unset"
(JS ``undefined``).

``get_provider_config`` returns a :class:`ProviderConfig` (url, headers, ``build_body``,
``parse_stream``, ``parse_response``, ``streaming``) exactly like the TS ``getProviderConfig``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from .azure_openai import AZURE_OPENAI_API_VERSION, build_azure_openai_url, is_azure_openai_endpoint
from .context_budget import compute_context_budget
from .llm_jscompat import (
    RE_I,
    encode_uri_component,
    is_js_number,
    js_json_loads,
    js_str,
    js_trim,
    parse_url,
)
from .reasoning_capabilities import (
    is_adaptive_anthropic_model,
    is_gemini_thinking_level_model,
    is_openrouter_endpoint,
    normalize_reasoning_for_provider,
)

ChatMessage = Dict[str, Any]
RequestOverrides = Dict[str, Any]
LlmConfigDict = Mapping[str, Any]

JSON_CONTENT_TYPE = "application/json"
_HTTP_HEADER_NAME_RE = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")


class LlmProviderError(Exception):
    """A plain `throw new Error(...)` from the provider layer."""


def _cfg_str(config: LlmConfigDict, key: str) -> str:
    v = config.get(key)
    return v if isinstance(v, str) else ""


# ----------------------------------------------------------------- headers

def merge_llm_request_headers(custom: Optional[Mapping[str, str]], required: Mapping[str, str]) -> Dict[str, str]:
    """Case-insensitive merge; required (protocol) headers win; invalid custom entries are dropped."""
    merged: Dict[str, List[str]] = {}
    for name, value in (custom or {}).items():
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        tn = js_trim(name)
        tv = js_trim(value)
        if not _HTTP_HEADER_NAME_RE.fullmatch(tn) or not tv or "\r" in tv or "\n" in tv:
            continue
        merged[tn.lower()] = [tn, tv]
    for name, value in required.items():
        if not value:
            continue
        merged[name.lower()] = [name, value]
    return {n: v for n, v in merged.values()}


def local_llm_origin_header() -> Dict[str, str]:
    return {"Origin": "http://localhost"}


def is_local_or_private_http_endpoint(endpoint: str) -> bool:
    url = parse_url(endpoint)
    if url is None:
        return False
    host = url.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    if host in ("127.0.0.1", "::1", "[::1]"):
        return True
    if re.match(r"^10\.", host):
        return True
    if re.match(r"^192\.168\.", host):
        return True
    m = re.match(r"^172\.([0-9]+)\.", host)
    if m:
        second = int(m.group(1))
        if 16 <= second <= 31:
            return True
    return False


# ----------------------------------------------------------------- JSON navigation helpers

def _get(obj: Any, key: Any) -> Any:
    """Optional-chaining step: `obj?.[key]` for dicts / lists (None when unreachable)."""
    if isinstance(key, int):
        if isinstance(obj, list) and -1 < key < len(obj):
            return obj[key]
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return None


def _path(obj: Any, *keys: Any) -> Any:
    for k in keys:
        obj = _get(obj, k)
        if obj is None:
            return None
    return obj


# ----------------------------------------------------------------- stream / response parsing

def parse_openai_line(line: str) -> Optional[str]:
    if not line.startswith("data:"):
        return None
    data = js_trim(line[5:])
    if data == "[DONE]":
        return None
    try:
        parsed = js_json_loads(data)
    except Exception:
        return None
    content = _path(parsed, "choices", 0, "delta", "content")
    return content if isinstance(content, str) else None


def _text_from_unknown_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for part in content:
        if not isinstance(part, dict):
            out.append("")
            continue
        t = part.get("text")
        out.append(t if isinstance(t, str) else "")
    return "".join(out)


def parse_openai_response(payload: Any) -> str:
    return _text_from_unknown_content(_path(payload, "choices", 0, "message", "content"))


def parse_anthropic_response(payload: Any) -> str:
    return _text_from_unknown_content(_get(payload, "content"))


def parse_google_response(payload: Any) -> str:
    parts = _path(payload, "candidates", 0, "content", "parts")
    if parts is None:
        parts = []
    if not isinstance(parts, list):
        raise LlmProviderError("parts.filter is not a function")
    out = []
    for part in parts:
        if part is None:
            raise LlmProviderError("Cannot read properties of null (reading 'thought')")
        if isinstance(part, dict):
            if part.get("thought"):
                continue
            t = part.get("text")
            out.append("" if t is None else js_str(t))
        else:
            out.append("")
    return "".join(out)


def parse_anthropic_line(line: str) -> Optional[str]:
    if not line.startswith("data:"):
        return None
    data = js_trim(line[5:])
    if data == "[DONE]":
        return None
    try:
        parsed = js_json_loads(data)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    delta = parsed.get("delta")
    if parsed.get("type") == "content_block_delta" and isinstance(delta, dict) and (
            delta.get("type") == "text_delta" or isinstance(delta.get("text"), str)):
        t = delta.get("text")
        return t if isinstance(t, str) else None

    content = parsed.get("content")
    if parsed.get("type") == "message" and isinstance(content, list):
        pieces = []
        for block in content:
            if block is None:
                return None  # JS: `block.text` on null throws -> caught
            t = block.get("text") if isinstance(block, dict) else None
            pieces.append(t if isinstance(t, str) else "")
        text = "".join(pieces)
        return text if len(text) > 0 else None

    choice_delta = _path(parsed, "choices", 0, "delta")
    if isinstance(choice_delta, dict) and isinstance(choice_delta.get("content"), str):
        return choice_delta["content"]
    return None


def parse_google_line(line: str) -> Optional[str]:
    if not line.startswith("data:"):
        return None
    data = js_trim(line[5:])
    try:
        parsed = js_json_loads(data)
    except Exception:
        return None
    parts = _path(parsed, "candidates", 0, "content", "parts")
    if not isinstance(parts, list) or len(parts) == 0:
        return None
    out = ""
    for p in parts:
        if p is None:
            return None  # JS TypeError -> caught
        if isinstance(p, dict):
            if p.get("thought"):
                continue
            t = p.get("text")
            if t:
                out += t if isinstance(t, str) else js_str(t)
    return out if len(out) > 0 else None


# ----------------------------------------------------------------- content translators

def _all_text(content: List[Dict[str, Any]]) -> bool:
    return all(b.get("type") == "text" for b in content)


def _block_text(b: Dict[str, Any]) -> str:
    return b.get("text") if b.get("type") == "text" else ""


def to_openai_content(content: Union[str, List[Dict[str, Any]]]) -> Any:
    if isinstance(content, str):
        return content
    if _all_text(content):
        return "".join(js_str(_block_text(b)) for b in content)
    out = []
    for b in content:
        if b.get("type") == "text":
            out.append({"type": "text", "text": b.get("text")})
        else:
            out.append({
                "type": "image_url",
                "image_url": {"url": f"data:{b.get('mediaType')};base64,{b.get('dataBase64')}"},
            })
    return out


def _strip_wire_agnostic_overrides(overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {k: v for k, v in (overrides or {}).items() if k != "reasoning" and v is not None}


def _effective_reasoning(config: LlmConfigDict, overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    requested = (overrides or {}).get("reasoning")
    if requested is None:
        requested = config.get("reasoning")
    if requested is None:
        requested = {"mode": "auto"}
    return normalize_reasoning_for_provider(config, requested)


def build_openai_body(messages: List[ChatMessage], overrides: Optional[RequestOverrides] = None,
                      streaming: bool = True) -> Dict[str, Any]:
    translated = [{"role": m["role"], "content": to_openai_content(m["content"])} for m in messages]
    body: Dict[str, Any] = {"messages": translated, "stream": streaming}
    body.update(_strip_wire_agnostic_overrides(overrides))
    return body


# ----------------------------------------------------------------- vendor detection

def _is_deepseek_endpoint(config: LlmConfigDict) -> bool:
    return re.search(r"api\.deepseek\.(?:com|cn)(?:[:/]|\Z)", _cfg_str(config, "customEndpoint"), RE_I) is not None


def _supports_deepseek_thinking_param(config: LlmConfigDict) -> bool:
    return re.search(r"deepseek[-_]?v4", _cfg_str(config, "model"), RE_I) is not None


def _is_kimi_endpoint(config: LlmConfigDict) -> bool:
    e = _cfg_str(config, "customEndpoint")
    return (re.search(r"api\.moonshot\.(ai|cn)(?:[:/]|\Z)", e, RE_I) is not None
            or re.search(r"api\.kimi\.com/coding(?:/|\Z)", e, RE_I) is not None)


def _is_xiaomi_mimo_endpoint(config: LlmConfigDict) -> bool:
    return re.search(r"\.?xiaomimimo\.com(?::|/|\Z)", _cfg_str(config, "customEndpoint"), RE_I) is not None


def _is_bigmodel_endpoint(config: LlmConfigDict) -> bool:
    return re.search(r"(?:^|//)open\.bigmodel\.cn(?:[:/]|\Z)", _cfg_str(config, "customEndpoint"), RE_I) is not None


def _is_glm_vision_model(model: str) -> bool:
    n = js_trim(model).lower()
    return (re.search(r"(?:^|[-_.])glm[-_.]5v[-_.]turbo(?:[-_.]|\Z)", n, RE_I) is not None
            or re.search(r"(?:^|[-_.])glm[-_.]4\.?6v(?:[-_.]|\Z)", n, RE_I) is not None
            or re.search(r"(?:^|[-_.])glm[-_.]4\.?5v(?:[-_.]|\Z)", n, RE_I) is not None
            or re.search(r"(?:^|[-_.])glm[-_.]4v(?:[-_.]|\Z)", n, RE_I) is not None)


def _is_openai_strict_completion_model(config: LlmConfigDict) -> bool:
    provider = config.get("provider")
    custom_azure = provider == "custom" and is_azure_openai_endpoint(_cfg_str(config, "customEndpoint"))
    if (provider == "azure" or custom_azure) and config.get("azureModelFamily") == "gpt5":
        return True
    model = js_trim(_cfg_str(config, "model")).lower()
    strict = (re.search(r"^gpt-5(?:[.\-_]|\Z)", model) is not None
              or re.search(r"^o[0-9]+(?:[.\-_]|\Z)", model) is not None)
    if not strict:
        return False
    if provider in ("openai", "azure"):
        return True
    return provider == "custom" and is_azure_openai_endpoint(_cfg_str(config, "customEndpoint"))


def _adapt_openai_strict_completion_body(config: LlmConfigDict, body: Dict[str, Any]) -> None:
    if not _is_openai_strict_completion_model(config):
        return
    if is_js_number(body.get("max_tokens")):
        body["max_completion_tokens"] = body.pop("max_tokens")
    body.pop("temperature", None)
    body.pop("top_p", None)
    body.pop("top_k", None)


def _adapt_kimi_body(config: LlmConfigDict, body: Dict[str, Any]) -> None:
    if not _is_kimi_endpoint(config):
        return
    body.pop("temperature", None)


def _adapt_xiaomi_mimo_body(config: LlmConfigDict, body: Dict[str, Any], reasoning: Dict[str, Any]) -> None:
    if not _is_xiaomi_mimo_endpoint(config):
        return
    if is_js_number(body.get("max_tokens")):
        body["max_completion_tokens"] = body.pop("max_tokens")
    if reasoning.get("mode") == "off":
        body["thinking"] = {"type": "disabled"}
    else:
        body.pop("temperature", None)


def _has_image_content(messages: List[ChatMessage]) -> bool:
    return any(isinstance(m["content"], list) and any(b.get("type") == "image" for b in m["content"])
               for m in messages)


def _assert_bigmodel_image_support(config: LlmConfigDict, messages: List[ChatMessage]) -> None:
    if (not _is_bigmodel_endpoint(config) or not _has_image_content(messages)
            or _is_glm_vision_model(_cfg_str(config, "model"))):
        return
    raise LlmProviderError(
        "Zhipu BigModel image input is supported only by GLM vision models. "
        "Switch to glm-5v-turbo, glm-4.6v, glm-4.5v, or glm-4v-plus.")


def build_openai_compatible_body(config: LlmConfigDict, messages: List[ChatMessage],
                                 overrides: Optional[RequestOverrides] = None,
                                 streaming: bool = True) -> Dict[str, Any]:
    _assert_bigmodel_image_support(config, messages)
    reasoning = _effective_reasoning(config, overrides)
    body = build_openai_body(messages, _strip_wire_agnostic_overrides(overrides), streaming)
    _adapt_openai_strict_completion_body(config, body)
    _adapt_kimi_body(config, body)
    _adapt_xiaomi_mimo_body(config, body, reasoning)

    provider = config.get("provider")
    mode = reasoning.get("mode")
    if provider == "custom" and is_openrouter_endpoint(_cfg_str(config, "customEndpoint")):
        if mode == "custom" and reasoning.get("budgetTokens") is not None:
            body["reasoning"] = {"max_tokens": reasoning["budgetTokens"]}
        elif mode != "auto":
            body["reasoning"] = {"effort": "none" if mode == "off" else mode}
        return body

    if _is_deepseek_endpoint(config):
        if _supports_deepseek_thinking_param(config):
            if mode == "off":
                body["thinking"] = {"type": "disabled"}
            elif mode != "auto":
                body["thinking"] = {"type": "enabled"}
                if mode in ("high", "max"):
                    body["reasoning_effort"] = mode
        return body

    if provider == "ollama":
        if mode == "off":
            body["reasoning_effort"] = "none"
        elif mode in ("low", "medium", "high"):
            body["reasoning_effort"] = mode
        elif mode == "max":
            body["reasoning_effort"] = "high"
        return body

    if provider == "openai" and mode != "auto" and mode != "off":
        if mode in ("low", "medium", "high"):
            body["reasoning_effort"] = mode
    return body


# ----------------------------------------------------------------- Anthropic

def to_anthropic_content(content: Union[str, List[Dict[str, Any]]]) -> Any:
    if isinstance(content, str):
        return content
    if _all_text(content):
        return "".join(js_str(_block_text(b)) for b in content)
    out = []
    for b in content:
        if b.get("type") == "text":
            out.append({"type": "text", "text": b.get("text")})
        else:
            out.append({"type": "image", "source": {
                "type": "base64", "media_type": b.get("mediaType"), "data": b.get("dataBase64")}})
    return out


def _flatten_text(content: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    return "".join(js_str(_block_text(b)) for b in content)


def _build_anthropic_system(system_text: str) -> Optional[List[Dict[str, Any]]]:
    if not system_text:
        return None
    return [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]


def build_anthropic_body(messages: List[ChatMessage], overrides: Optional[RequestOverrides] = None,
                         streaming: bool = True) -> Dict[str, Any]:
    ov = overrides or {}
    conversation = [{"role": m["role"], "content": to_anthropic_content(m["content"])}
                    for m in messages if m["role"] != "system"]
    system_text = "\n".join(_flatten_text(m["content"]) for m in messages if m["role"] == "system")
    system = _build_anthropic_system(system_text)

    body: Dict[str, Any] = {"messages": conversation}
    if system is not None:
        body["system"] = system
    body["stream"] = streaming
    max_tokens = ov.get("max_tokens")
    body["max_tokens"] = 4096 if max_tokens is None else max_tokens
    if ov.get("temperature") is not None:
        body["temperature"] = ov["temperature"]
    if ov.get("top_p") is not None:
        body["top_p"] = ov["top_p"]
    if ov.get("top_k") is not None:
        body["top_k"] = ov["top_k"]
    if ov.get("stop") is not None:
        stop = ov["stop"]
        body["stop_sequences"] = stop if isinstance(stop, list) else [stop]
    return body


def build_anthropic_body_with_reasoning(config: LlmConfigDict, messages: List[ChatMessage],
                                        overrides: Optional[RequestOverrides] = None,
                                        streaming: bool = True) -> Dict[str, Any]:
    body = build_anthropic_body(messages, overrides, streaming)
    reasoning = _effective_reasoning(config, overrides)
    mode = reasoning.get("mode")
    if mode == "auto" or mode == "off":
        return body

    if is_adaptive_anthropic_model(config):
        body["thinking"] = {"type": "adaptive"}
        effort = "high" if mode == "custom" else ("max" if mode == "max" else mode)
        body["output_config"] = {"effort": effort}
        for k in ("temperature", "top_p", "top_k"):
            body.pop(k, None)
        return body

    if mode == "custom" and reasoning.get("budgetTokens") is not None:
        budget = reasoning["budgetTokens"]
    elif mode == "low":
        budget = 1024
    elif mode == "medium":
        budget = 4096
    else:
        budget = 8192
    budget_tokens = max(1024, budget)
    if body["max_tokens"] <= budget_tokens:
        body["max_tokens"] = budget_tokens + 1
    body["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
    for k in ("temperature", "top_p", "top_k"):
        body.pop(k, None)
    return body


def _requires_bearer_auth(url: str) -> bool:
    normalized = re.sub(r"/+\Z", "", url.lower())
    return (
        normalized.startswith("https://api.minimax.io/anthropic")
        or normalized.startswith("https://api.minimaxi.com/anthropic")
        or normalized.startswith("https://coding.dashscope.aliyuncs.com/apps/anthropic")
        or re.search(r"(^https://|^)token-plan-cn\.xiaomimimo\.com/anthropic(?:/|\Z)", normalized, RE_I) is not None
        or normalized.startswith("https://api.kimi.com/coding")
        or normalized.startswith("https://api.moonshot.ai/anthropic")
        or normalized.startswith("https://api.moonshot.cn/anthropic")
    )


def _is_minimax_anthropic_host(url: str) -> bool:
    p = parse_url(url)
    return p is not None and p.hostname in ("api.minimax.io", "api.minimaxi.com")


def _normalize_minimax_anthropic_base(base: str) -> str:
    if not _is_minimax_anthropic_host(base):
        return base
    p = parse_url(base)
    assert p is not None
    path = re.sub(r"/+\Z", "", p.pathname)
    if path == "" or path == "/" or re.match(r"^/v[0-9]+(?:/messages)?\Z", path, RE_I):
        userinfo = ""
        if p.username or p.password:
            userinfo = p.username + (":" + p.password if p.password else "") + "@"
        port = f":{p.port}" if p.port else ""
        out = f"{p.protocol}//{userinfo}{p.hostname}{port}/anthropic"
        return re.sub(r"/+\Z", "", out)
    return base


def _is_official_minimax_anthropic_url(url: str) -> bool:
    if not _is_minimax_anthropic_host(url):
        return False
    p = parse_url(url)
    return p is not None and re.match(r"^/anthropic(?:/|\Z)", p.pathname, RE_I) is not None


def _is_minimax_m3_model(model: str) -> bool:
    return re.match(r"^minimax-m3(?:[-_.]|\Z)", js_trim(model), RE_I) is not None


def _assert_minimax_image_support(url: str, model: str, messages: List[ChatMessage]) -> None:
    if (not _is_official_minimax_anthropic_url(url) or not _has_image_content(messages)
            or _is_minimax_m3_model(model)):
        return
    raise LlmProviderError(
        "MiniMax image input is supported only by MiniMax-M3 on the official Anthropic-compatible "
        "endpoint. Switch the model to MiniMax-M3 or use another vision-capable provider.")


def supports_image_input(config: LlmConfigDict) -> bool:
    provider = config.get("provider")
    model = _cfg_str(config, "model")
    if provider == "codex-cli":
        return False
    if provider == "minimax":
        return _is_minimax_m3_model(model)
    if _is_bigmodel_endpoint(config):
        return _is_glm_vision_model(model)
    if provider == "custom" and (config.get("apiMode") or "chat_completions") == "anthropic_messages":
        url = build_anthropic_url(_cfg_str(config, "customEndpoint"))
        return not _is_official_minimax_anthropic_url(url) or _is_minimax_m3_model(model)
    return True


def build_anthropic_url(base: str) -> str:
    trimmed = re.sub(r"/+\Z", "", _normalize_minimax_anthropic_base(base))
    if re.search(r"/v[0-9]+/messages\Z", trimmed, RE_I):
        return trimmed
    if re.search(r"/v[0-9]+\Z", trimmed, RE_I):
        return f"{trimmed}/messages"
    return f"{trimmed}/v1/messages"


def _build_anthropic_headers(config: LlmConfigDict, url: str) -> Dict[str, str]:
    api_key = _cfg_str(config, "apiKey")
    base: Dict[str, str] = {"Content-Type": JSON_CONTENT_TYPE, "anthropic-version": "2023-06-01"}
    if _requires_bearer_auth(url):
        base["Authorization"] = f"Bearer {api_key}"
    else:
        base["x-api-key"] = api_key
        base["anthropic-dangerous-direct-browser-access"] = "true"
    return merge_llm_request_headers(config.get("customHeaders"), base)


# ----------------------------------------------------------------- Google

def _to_google_parts(content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    out = []
    for b in content:
        if b.get("type") == "text":
            out.append({"text": b.get("text")})
        else:
            out.append({"inline_data": {"mime_type": b.get("mediaType"), "data": b.get("dataBase64")}})
    return out


def build_google_body(config: LlmConfigDict, messages: List[ChatMessage],
                      overrides: Optional[RequestOverrides] = None) -> Dict[str, Any]:
    ov = overrides or {}
    system_messages = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": _to_google_parts(m["content"])} for m in conversation]
    system_instruction = None
    if system_messages:
        system_instruction = {"parts": [{"text": _flatten_text(m["content"])} for m in system_messages]}

    gc: Dict[str, Any] = {}
    if ov.get("temperature") is not None:
        gc["temperature"] = ov["temperature"]
    if ov.get("top_p") is not None:
        gc["topP"] = ov["top_p"]
    if ov.get("top_k") is not None:
        gc["topK"] = ov["top_k"]
    if ov.get("max_tokens") is not None:
        gc["maxOutputTokens"] = ov["max_tokens"]
    if ov.get("stop") is not None:
        stop = ov["stop"]
        gc["stopSequences"] = stop if isinstance(stop, list) else [stop]
    reasoning = _effective_reasoning(config, ov)
    mode = reasoning.get("mode")
    if mode == "off":
        gc["thinkingConfig"] = {"thinkingBudget": 0}
    elif mode != "auto":
        if is_gemini_thinking_level_model(config):
            level = mode if mode in ("low", "medium") else "high"
            gc["thinkingConfig"] = {"thinkingLevel": level}
        else:
            if mode == "custom" and reasoning.get("budgetTokens") is not None:
                budget = reasoning["budgetTokens"]
            elif mode == "low":
                budget = 1024
            elif mode == "medium":
                budget = 4096
            else:
                budget = 8192
            gc["thinkingConfig"] = {"thinkingBudget": budget}

    body: Dict[str, Any] = {"contents": contents}
    if system_instruction is not None:
        body["systemInstruction"] = system_instruction
    if gc:
        body["generationConfig"] = gc
    return body


def derive_anthropic_max_tokens(max_context_size: Optional[float]) -> int:
    response_reserve = compute_context_budget(max_context_size).responseReserve
    return max(1, min(16_384, math.floor(response_reserve / 3)))


# ----------------------------------------------------------------- getProviderConfig

@dataclass
class ProviderConfig:
    url: str
    headers: Dict[str, str]
    build_body: Callable[..., Dict[str, Any]]      # (messages, overrides=None) -> body dict
    parse_stream: Callable[[str], Optional[str]]
    parse_response: Callable[[Any], str]
    streaming: bool


def get_provider_config(config: LlmConfigDict) -> ProviderConfig:
    provider = config.get("provider")
    api_key = _cfg_str(config, "apiKey")
    model = _cfg_str(config, "model")
    ollama_url = _cfg_str(config, "ollamaUrl")
    custom_endpoint = _cfg_str(config, "customEndpoint")
    streaming = config.get("streamingEnabled") is not False
    custom_headers = config.get("customHeaders")
    anthropic_budget_tokens = derive_anthropic_max_tokens(config.get("maxContextSize"))

    def anthropic_config(url: str) -> ProviderConfig:
        def build(messages, overrides=None):
            _assert_minimax_image_support(url, model, messages)
            merged = {"max_tokens": anthropic_budget_tokens}
            merged.update(overrides or {})
            body = build_anthropic_body_with_reasoning(config, messages, merged, streaming)
            body["model"] = model
            return body
        return ProviderConfig(url, _build_anthropic_headers(config, url), build,
                              parse_anthropic_line, parse_anthropic_response, streaming)

    if provider == "openai":
        def build_openai(messages, overrides=None):
            body = build_openai_compatible_body(config, messages, overrides, streaming)
            body["model"] = model
            return body
        return ProviderConfig(
            "https://api.openai.com/v1/chat/completions",
            merge_llm_request_headers(custom_headers, {
                "Content-Type": JSON_CONTENT_TYPE, "Authorization": f"Bearer {api_key}"}),
            build_openai, parse_openai_line, parse_openai_response, streaming)

    if provider == "anthropic":
        return anthropic_config(build_anthropic_url("https://api.anthropic.com"))

    if provider == "google":
        encoded_model = encode_uri_component(model)
        base = f"https://generativelanguage.googleapis.com/v1beta/models/{encoded_model}"
        url = f"{base}:streamGenerateContent?alt=sse" if streaming else f"{base}:generateContent"

        def build_google(messages, overrides=None):
            ov = dict(overrides or {})
            ov["reasoning"] = _effective_reasoning(config, overrides)
            return build_google_body(config, messages, ov)
        return ProviderConfig(
            url,
            merge_llm_request_headers(custom_headers, {
                "Content-Type": JSON_CONTENT_TYPE, "x-goog-api-key": api_key}),
            build_google, parse_google_line, parse_google_response, streaming)

    if provider == "azure":
        return ProviderConfig(
            build_azure_openai_url(custom_endpoint, model,
                                   config.get("azureApiVersion") or AZURE_OPENAI_API_VERSION),
            merge_llm_request_headers(custom_headers, {
                "Content-Type": JSON_CONTENT_TYPE, "api-key": api_key}),
            lambda messages, overrides=None: build_openai_compatible_body(config, messages, overrides, streaming),
            parse_openai_line, parse_openai_response, streaming)

    if provider == "ollama":
        base = re.sub(r"/+\Z", "", ollama_url)
        if re.search(r"/v1/chat/completions\Z", base, RE_I):
            base = re.sub(r"/v1/chat/completions\Z", "", base, flags=RE_I)
        elif re.search(r"/v1\Z", base, RE_I):
            base = re.sub(r"/v1\Z", "", base, flags=RE_I)

        def build_ollama(messages, overrides=None):
            body = build_openai_compatible_body(config, messages, overrides, streaming)
            body["model"] = model
            return body
        headers = {"Content-Type": JSON_CONTENT_TYPE}
        headers.update(local_llm_origin_header())
        return ProviderConfig(f"{base}/v1/chat/completions",
                              merge_llm_request_headers(custom_headers, headers),
                              build_ollama, parse_openai_line, parse_openai_response, streaming)

    if provider == "minimax":
        return anthropic_config(build_anthropic_url(custom_endpoint or "https://api.minimax.io/anthropic"))

    if provider in ("claude-code", "codex-cli"):
        raise LlmProviderError(
            f"{provider} provider uses subprocess transport; get_provider_config should not be called for it")

    if provider == "custom":
        mode = config.get("apiMode") or "chat_completions"
        if mode == "anthropic_messages":
            return anthropic_config(build_anthropic_url(custom_endpoint))
        base = re.sub(r"/+\Z", "", custom_endpoint)
        if is_azure_openai_endpoint(base):
            url = build_azure_openai_url(base, model, config.get("azureApiVersion") or AZURE_OPENAI_API_VERSION)
        elif re.search(r"/chat/completions\Z", base, RE_I):
            url = base
        else:
            url = f"{base}/chat/completions"
        azure = is_azure_openai_endpoint(url)
        required: Dict[str, str] = {"Content-Type": JSON_CONTENT_TYPE}
        if api_key:
            if azure:
                required["api-key"] = api_key
            else:
                required["Authorization"] = f"Bearer {api_key}"
        if not azure and is_local_or_private_http_endpoint(url):
            required.update(local_llm_origin_header())

        def build_custom(messages, overrides=None):
            body = build_openai_compatible_body(config, messages, overrides, streaming)
            if not azure:
                body["model"] = model
            return body
        return ProviderConfig(url, merge_llm_request_headers(custom_headers, required), build_custom,
                              parse_openai_line, parse_openai_response, streaming)

    raise LlmProviderError(f"Unknown provider: {provider}")
