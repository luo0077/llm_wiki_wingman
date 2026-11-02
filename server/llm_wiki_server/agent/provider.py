"""Port of agent/provider.rs: LLM HTTP transports (OpenAI-compatible, Anthropic-compatible,
Google Gemini, Azure OpenAI, Ollama, MiniMax) with SSE streaming.

All failures raise `ProviderError` whose message equals the Rust `Err(String)` text.
Every request builds a fresh httpx client (so HTTP(S)_PROXY / NO_PROXY set by
`proxy.apply_proxy_env` are honored via `trust_env`, TLS verification follows
`proxy.accept_invalid_certs()`), with a 180s timeout.

The primary API is async (`generate_text`, `generate_text_stream`); `*_sync`
twins exist for callers running in worker threads. `on_delta` is a plain callable.
"""
from __future__ import annotations

import http
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import httpx

from .types import AgentImage

REQUEST_TIMEOUT_SECS = 180
DEFAULT_MAX_TOKENS = 2048
ANTHROPIC_VERSION = "2023-06-01"
AZURE_OPENAI_API_VERSION = "2024-10-21"
EMPTY_CONTENT_ERROR = "LLM response did not contain assistant content"


class ProviderError(Exception):
    """Rust `Err(String)`; `str(e)` is the message."""


def _lc(s: str) -> str:
    """ASCII-only lowercase (Rust to_ascii_lowercase)."""
    return "".join(c.lower() if c.isascii() else c for c in s)


# ----------------------------------------------------------------- config

@dataclass
class LlmReasoningConfig:
    mode: Optional[str] = None
    budget_tokens: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Any) -> "LlmReasoningConfig":
        if not isinstance(d, dict):
            raise ValueError("invalid reasoning config")
        mode = d.get("mode")
        bt = d.get("budgetTokens")
        if mode is not None and not isinstance(mode, str):
            raise ValueError("invalid reasoning.mode")
        if bt is not None and (isinstance(bt, bool) or not isinstance(bt, int) or bt < 0 or bt > 0xFFFFFFFF):
            raise ValueError("invalid reasoning.budgetTokens")
        return cls(mode, bt)

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "budgetTokens": self.budget_tokens}


def _s(d: dict, key: str, default: str = "") -> str:
    v = d.get(key)
    if v is None:
        return default
    if not isinstance(v, str):
        raise ValueError(f"invalid type for `{key}`: expected a string")
    return v


def _os(d: dict, key: str) -> Optional[str]:
    v = d.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"invalid type for `{key}`: expected a string")
    return v


@dataclass
class LlmConfig:
    provider: str
    api_key: str = ""
    model: str = ""
    ollama_url: str = ""
    custom_endpoint: str = ""
    azure_api_version: Optional[str] = None
    azure_model_family: Optional[str] = None
    api_mode: Optional[str] = None
    reasoning: Optional[LlmReasoningConfig] = None
    max_tokens: Optional[int] = None
    max_context_size: Optional[int] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    streaming_enabled: Optional[bool] = None

    @classmethod
    def from_dict(cls, d: Any) -> "LlmConfig":
        """serde Deserialize (camelCase keys); raises ValueError like a serde error."""
        if not isinstance(d, dict):
            raise ValueError("invalid LlmConfig: expected an object")
        if not isinstance(d.get("provider"), str):
            raise ValueError("missing field `provider`")
        mt = d.get("maxTokens")
        if mt is not None and (isinstance(mt, bool) or not isinstance(mt, int) or mt < 0 or mt > 0xFFFFFFFF):
            raise ValueError("invalid value for `maxTokens`")
        mcs = d.get("maxContextSize")
        if mcs is not None and (isinstance(mcs, bool) or not isinstance(mcs, int) or mcs < 0):
            raise ValueError("invalid value for `maxContextSize`")
        ch = d.get("customHeaders")
        if ch is None:
            ch = {}
        if not isinstance(ch, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in ch.items()):
            raise ValueError("invalid value for `customHeaders`")
        se = d.get("streamingEnabled")
        if se is not None and not isinstance(se, bool):
            raise ValueError("invalid value for `streamingEnabled`")
        r = d.get("reasoning")
        return cls(
            provider=d["provider"],
            api_key=_s(d, "apiKey"),
            model=_s(d, "model"),
            ollama_url=_s(d, "ollamaUrl"),
            custom_endpoint=_s(d, "customEndpoint"),
            azure_api_version=_os(d, "azureApiVersion"),
            azure_model_family=_os(d, "azureModelFamily"),
            api_mode=_os(d, "apiMode"),
            reasoning=LlmReasoningConfig.from_dict(r) if r is not None else None,
            max_tokens=mt,
            max_context_size=mcs,
            custom_headers=dict(ch),
            streaming_enabled=se,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider, "apiKey": self.api_key, "model": self.model,
            "ollamaUrl": self.ollama_url, "customEndpoint": self.custom_endpoint,
            "azureApiVersion": self.azure_api_version, "azureModelFamily": self.azure_model_family,
            "apiMode": self.api_mode,
            "reasoning": self.reasoning.to_dict() if self.reasoning else None,
            "maxTokens": self.max_tokens, "maxContextSize": self.max_context_size,
            "customHeaders": dict(self.custom_headers), "streamingEnabled": self.streaming_enabled,
        }

    def is_usable_for_backend_http(self) -> bool:
        has_model = bool(self.model.strip())
        p = self.provider
        if p in ("openai", "anthropic", "google", "azure", "minimax"):
            return has_model and bool(self.api_key.strip())
        if p == "ollama":
            return has_model and bool(self.ollama_url.strip())
        if p == "custom":
            return has_model and bool(self.custom_endpoint.strip())
        return False  # claude-code / codex-cli / unknown


# ----------------------------------------------------------------- headers

_TOKEN_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def _header_value_ok(v: str) -> bool:
    for b in v.encode("utf-8"):
        if not ((b >= 32 and b != 127) or b == 9):
            return False
    return True


class _Headers:
    """Case-insensitive insert/replace map (http::HeaderMap for single values)."""

    def __init__(self) -> None:
        self._d: Dict[str, Tuple[str, str]] = {}

    def insert(self, name: str, value: str) -> None:
        self._d[name.lower()] = (name, value)

    def remove(self, name: str) -> None:
        self._d.pop(name.lower(), None)

    def get(self, name: str) -> Optional[str]:
        e = self._d.get(name.lower())
        return e[1] if e else None

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._d

    def items(self) -> List[Tuple[str, str]]:
        return [(k, v) for k, (_, v) in self._d.items()]

    def as_dict(self) -> Dict[str, str]:
        return dict(self.items())


def custom_headers(config: LlmConfig) -> _Headers:
    headers = _Headers()
    for name, value in config.custom_headers.items():
        n = name.strip()
        if not _TOKEN_RE.match(n):
            raise ProviderError(f"Invalid custom header name '{name}': invalid HTTP header name")
        v = value.strip()
        if not _header_value_ok(v):
            raise ProviderError(f"Invalid custom header value for '{n.lower()}': failed to parse header value")
        headers.insert(n, v)
    return headers


def _checked_value(v: str, what: str) -> str:
    if not _header_value_ok(v):
        raise ProviderError(f"{what}: failed to parse header value")
    return v


def openai_headers(config: LlmConfig, url: str) -> _Headers:
    headers = custom_headers(config)
    headers.insert("Content-Type", "application/json")
    key = config.api_key.strip()
    if key:
        if is_azure_endpoint(url):
            headers.insert("api-key", _checked_value(key, "Invalid API key header"))
        else:
            headers.insert("Authorization", _checked_value(f"Bearer {key}", "Invalid authorization header"))
    return headers


def anthropic_headers(config: LlmConfig, url: str) -> _Headers:
    headers = custom_headers(config)
    headers.insert("Content-Type", "application/json")
    headers.insert("anthropic-version", ANTHROPIC_VERSION)
    key = config.api_key.strip()
    if key:
        if requires_bearer_auth(url):
            headers.remove("x-api-key")
            headers.insert("authorization", _checked_value(f"Bearer {key}", "Invalid API key header"))
        else:
            headers.insert("x-api-key", _checked_value(key, "Invalid API key header"))
    return headers


def google_headers(config: LlmConfig) -> _Headers:
    headers = custom_headers(config)
    headers.insert("Content-Type", "application/json")
    if config.api_key.strip():
        headers.insert("x-goog-api-key", _checked_value(config.api_key.strip(), "Invalid API key header"))
    return headers


# ----------------------------------------------------------------- content

def openai_user_content(user: str, images: Sequence[AgentImage]) -> Any:
    if not images:
        return user
    blocks: List[Dict[str, Any]] = [{"type": "text", "text": user}]
    for im in images:
        blocks.append({"type": "image_url",
                       "image_url": {"url": f"data:{im.media_type};base64,{im.data_base64}"}})
    return blocks


def anthropic_user_content(user: str, images: Sequence[AgentImage]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = [{"type": "text", "text": user}]
    for im in images:
        blocks.append({"type": "image",
                       "source": {"type": "base64", "media_type": im.media_type, "data": im.data_base64}})
    return blocks


# ----------------------------------------------------------------- SSE parsing

def parse_openai_delta(data: str) -> Optional[str]:
    try:
        parsed = json.loads(data)
    except ValueError:
        return None
    try:
        choice = parsed["choices"][0]
        # serde `get("delta").or_else(get("message"))`: a present-but-null delta wins and yields no content.
        if "delta" in choice:
            msg = choice["delta"]
        else:
            msg = choice.get("message")
        c = msg["content"]
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return c if isinstance(c, str) else None


def parse_anthropic_delta(data: str) -> Optional[str]:
    try:
        parsed = json.loads(data)
    except ValueError:
        return None
    if isinstance(parsed, dict) and parsed.get("type") == "content_block_delta":
        delta = parsed.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            return delta["text"]
    return None


def _google_text(parsed: Any) -> Optional[str]:
    try:
        parts = parsed["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(parts, list):
        return None
    return "".join(p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))


def parse_google_delta(data: str) -> Optional[str]:
    try:
        parsed = json.loads(data)
    except ValueError:
        return None
    text = _google_text(parsed)
    return text if text else None


class SseCollector:
    """Incremental SSE line collector (collect_sse_chunk_bytes / collect_sse_line)."""

    def __init__(self, parse_delta: Callable[[str], Optional[str]], on_delta: Callable[[str], Any]):
        self.parse_delta = parse_delta
        self.on_delta = on_delta
        self.buffer = bytearray()
        self.full = ""

    def feed(self, chunk: bytes) -> None:
        self.buffer.extend(chunk)
        while True:
            nl = self.buffer.find(b"\n")
            if nl < 0:
                break
            line = bytes(self.buffer[:nl])
            del self.buffer[: nl + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            self._line_bytes(line)

    def finish(self) -> str:
        if self.buffer.strip(b" \t\r\n\x0c"):
            self._line_bytes(bytes(self.buffer))
        self.buffer.clear()
        if not self.full.strip():
            raise ProviderError(EMPTY_CONTENT_ERROR)
        return self.full.strip()

    def _line_bytes(self, line: bytes) -> None:
        self.line(line.decode("utf-8", errors="replace").strip())

    def line(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        data = line[len("data:"):].strip()
        if data == "[DONE]" or not data:
            return
        delta = self.parse_delta(data)
        if delta:
            self.on_delta(delta)
            self.full += delta


def collect_sse_line(line: str, parse_delta, on_delta, full: List[str]) -> None:
    """Test helper mirroring the Rust fn; `full` is a 1-element list accumulator."""
    c = SseCollector(parse_delta, on_delta)
    c.full = full[0]
    c.line(line.strip())
    full[0] = c.full


# ----------------------------------------------------------------- URLs

def url_encode_path_segment(value: str) -> str:
    out = []
    for b in value.encode("utf-8"):
        ch = chr(b)
        if b < 128 and (ch.isalnum() or ch in "-_.~"):
            out.append(ch)
        else:
            out.append(f"%{b:02X}")
    return "".join(out)


def build_ollama_url(base: str) -> str:
    base = base.strip().rstrip("/")
    if _lc(base).endswith("/v1/chat/completions"):
        base = base[: -len("/v1/chat/completions")]
    elif _lc(base).endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/v1/chat/completions"


def build_custom_openai_url(config: LlmConfig) -> str:
    base = config.custom_endpoint.strip().rstrip("/")
    if is_azure_endpoint(base):
        try:
            return build_azure_url(config)
        except ProviderError:
            return base
    if _lc(base).endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def build_azure_url(config: LlmConfig) -> str:
    endpoint = config.custom_endpoint.strip().rstrip("/")
    if not endpoint:
        raise ProviderError("Azure endpoint is required")
    api_version = config.azure_api_version if (config.azure_api_version and config.azure_api_version.strip()) \
        else AZURE_OPENAI_API_VERSION
    if "/openai/deployments/" in _lc(endpoint):
        sep = "&" if "?" in endpoint else "?"
        return f"{endpoint}{sep}api-version={api_version}"
    return (f"{endpoint}/openai/deployments/{url_encode_path_segment(config.model)}"
            f"/chat/completions?api-version={api_version}")


def build_anthropic_url(base: str) -> str:
    normalized = normalize_minimax_anthropic_base(base.strip())
    base = normalized.rstrip("/")
    if _lc(base).endswith("/v1/messages"):
        return base
    if _lc(base).endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _parse_url(url: str):
    """Approximation of reqwest::Url::parse: needs scheme + authority + host."""
    try:
        p = urlsplit(url)
        host = p.hostname
    except ValueError:
        return None
    if not p.scheme or not p.netloc or not host:
        return None
    return p


def normalize_minimax_anthropic_base(base: str) -> str:
    p = _parse_url(base)
    if p is None:
        return base
    host = _lc(p.hostname or "")
    if host not in ("api.minimax.io", "api.minimaxi.com"):
        return base
    path = p.path.rstrip("/")
    version = None
    if path.startswith("/v"):
        version = path[2:]
        if version.endswith("/messages"):
            version = version[: -len("/messages")]
    is_generic = version is not None and version != "" and all(c in "0123456789" for c in version)
    if not (path == "" or is_generic):
        return base
    return f"{p.scheme.lower()}://{p.netloc}/anthropic"


def is_azure_endpoint(url: str) -> bool:
    lower = _lc(url)
    return ".openai.azure.com" in lower or "/openai/deployments/" in lower


def requires_bearer_auth(url: str) -> bool:
    p = _parse_url(url)
    if p is None:
        return False
    host = _lc(p.hostname or "")
    path = _lc(p.path.rstrip("/"))
    return (
        (host in ("api.minimax.io", "api.minimaxi.com") and path.startswith("/anthropic"))
        or (host == "coding.dashscope.aliyuncs.com" and path.startswith("/apps/anthropic"))
        or (host == "token-plan-cn.xiaomimimo.com" and path.startswith("/anthropic"))
        or (host == "api.kimi.com" and path.startswith("/coding"))
        or (host in ("api.moonshot.ai", "api.moonshot.cn") and path.startswith("/anthropic"))
    )


# ----------------------------------------------------------------- reasoning

def is_deepseek_endpoint(config: LlmConfig) -> bool:
    e = _lc(config.custom_endpoint)
    return "api.deepseek.com" in e or "api.deepseek.cn" in e


def supports_deepseek_thinking_param(config: LlmConfig) -> bool:
    return "deepseek-v4" in _lc(config.model).replace("_", "-")


def is_openai_reasoning_model(config: LlmConfig) -> bool:
    if config.provider == "azure" and config.azure_model_family == "gpt5":
        return True
    lower = _lc(config.model)
    if lower.startswith("gpt-5"):
        return True
    return len(lower) > 1 and lower[0] == "o" and lower[1].isascii() and lower[1].isdigit()


def is_gemini_3(model: str) -> bool:
    lower = _lc(model)
    return lower.startswith(("gemini-3-", "gemini-3.", "gemini_3_", "gemini_3."))


def is_gemini_thinking_required(model: str) -> bool:
    return _lc(model).startswith("gemini-2.5-pro") or is_gemini_3(model)


def is_claude_46_or_later(model: str) -> bool:
    lower = _lc(model)
    for prefix in ("claude-opus-4-", "claude-sonnet-4-", "claude-haiku-4-"):
        if lower.startswith(prefix):
            tail = lower[len(prefix):]
            first = re.split(r"[-_.]", tail, maxsplit=1)[0]
            if first.isascii() and first.isdigit() and int(first) <= 0xFFFFFFFF:
                return int(first) >= 6
            return False
    return False


def reasoning_budget(reasoning: LlmReasoningConfig) -> Optional[int]:
    m = reasoning.mode
    if m == "custom":
        return reasoning.budget_tokens
    if m == "low":
        return 1024
    if m == "medium":
        return 4096
    if m in ("high", "max"):
        return 8192
    return None


def apply_openai_reasoning(body: Dict[str, Any], config: LlmConfig) -> None:
    reasoning = config.reasoning
    if reasoning is None:
        return
    if is_deepseek_endpoint(config) and supports_deepseek_thinking_param(config):
        if reasoning.mode == "off":
            body["thinking"] = {"type": "disabled"}
        return
    if config.provider == "ollama":
        if reasoning.mode == "off":
            body["reasoning_effort"] = "none"
        elif reasoning.mode in ("low", "medium", "high"):
            body["reasoning_effort"] = reasoning.mode
        elif reasoning.mode == "max":
            body["reasoning_effort"] = "high"
        return
    if config.provider not in ("openai", "azure"):
        return
    if not is_openai_reasoning_model(config):
        return
    if reasoning.mode in ("low", "medium", "high"):
        body["reasoning_effort"] = reasoning.mode


def adapt_openai_strict_completion_body(body: Dict[str, Any], config: LlmConfig) -> None:
    custom_azure = config.provider == "custom" and is_azure_endpoint(config.custom_endpoint)
    strict = is_openai_reasoning_model(config) or (custom_azure and config.azure_model_family == "gpt5")
    if not strict or (config.provider not in ("openai", "azure") and not custom_azure):
        return
    if "max_tokens" in body:
        body["max_completion_tokens"] = body.pop("max_tokens")


def apply_anthropic_reasoning(body: Dict[str, Any], config: LlmConfig) -> None:
    reasoning = config.reasoning
    if reasoning is None or config.provider != "anthropic":
        return
    if reasoning.mode == "off":
        return
    budget = reasoning_budget(reasoning)
    if budget is None:
        return
    if is_claude_46_or_later(config.model):
        effort = {"low": "low", "medium": "medium", "max": "max"}.get(reasoning.mode or "", "high")
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": effort}
        return
    body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    body["max_tokens"] = max(min(budget + 1024, 0xFFFFFFFF), DEFAULT_MAX_TOKENS)


def apply_google_reasoning(body: Dict[str, Any], config: LlmConfig) -> None:
    reasoning = config.reasoning
    if reasoning is None:
        return
    gc = body["generationConfig"]
    if reasoning.mode == "off":
        if is_gemini_thinking_required(config.model):
            return
        gc["thinkingConfig"] = {"thinkingBudget": 0}
        return
    if is_gemini_3(config.model) and reasoning_budget(reasoning) is not None:
        level = {"low": "low", "medium": "medium"}.get(reasoning.mode or "", "high")
        gc["thinkingConfig"] = {"thinkingLevel": level}
        return
    budget = reasoning_budget(reasoning)
    if budget is not None:
        gc["thinkingConfig"] = {"thinkingBudget": budget}


def trim_error_body(text: str) -> str:
    MAX = 800
    t = text.strip()
    if len(t) <= MAX:
        return t
    return t[:MAX] + "..."


def _status_str(code: int) -> str:
    try:
        phrase = http.HTTPStatus(code).phrase
    except ValueError:
        phrase = "<unknown status code>"
    return f"{code} {phrase}"


def _http_error(resp: httpx.Response, text: str) -> ProviderError:
    return ProviderError(f"LLM HTTP {_status_str(resp.status_code)}: {trim_error_body(text)}")


# ----------------------------------------------------------------- client

class AgentLlmProvider:
    """Trait equivalent. LlmClient implements it; other implementations (tests, mocks)
    may subclass and override."""

    def provider_name(self) -> str:
        raise NotImplementedError

    def model_name(self) -> str:
        raise NotImplementedError

    async def generate_text(self, system: str, user: str, images: Sequence[AgentImage] = ()) -> str:
        raise NotImplementedError

    async def generate_text_stream(self, system: str, user: str, images: Sequence[AgentImage],
                                   on_delta: Callable[[str], Any]) -> str:
        raise NotImplementedError


@dataclass
class _Plan:
    """A fully prepared HTTP request."""
    url: str
    headers: Dict[str, str]
    body: Dict[str, Any]
    kind: str  # "openai" | "anthropic" | "google"


def _extract_openai(parsed: Any) -> str:
    try:
        c = parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        c = None
    return (c if isinstance(c, str) else "").strip()


def _extract_anthropic(parsed: Any) -> str:
    blocks = parsed.get("content") if isinstance(parsed, dict) else None
    if not isinstance(blocks, list):
        return ""
    return "".join(b["text"] for b in blocks
                   if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)).strip()


def _extract_google(parsed: Any) -> str:
    return (_google_text(parsed) or "").strip()


_EXTRACT = {"openai": _extract_openai, "anthropic": _extract_anthropic, "google": _extract_google}
_DELTA = {"openai": parse_openai_delta, "anthropic": parse_anthropic_delta, "google": parse_google_delta}

_MINIMAX_IMAGE_ERR = ("MiniMax official Anthropic-compatible endpoint does not support image input. "
                      "Use a vision-capable provider for image chat.")


class LlmClient(AgentLlmProvider):
    def __init__(self, config: LlmConfig, transport: Optional[httpx.BaseTransport] = None):
        """`transport` is for tests (httpx.MockTransport); None uses the real network + env proxies."""
        self.config = config
        self._transport = transport

    # -- trait --
    def provider_name(self) -> str:
        return self.config.provider

    def model_name(self) -> str:
        return self.config.model

    def structured_task_config(self, max_tokens: int) -> "LlmClient":
        cfg = replace(self.config, reasoning=LlmReasoningConfig(mode="off", budget_tokens=None),
                      max_tokens=max_tokens)
        return LlmClient(cfg, self._transport)

    def max_output_tokens(self) -> int:
        mt = self.config.max_tokens if self.config.max_tokens is not None else DEFAULT_MAX_TOKENS
        return max(256, min(32_768, mt))

    # -- bodies --
    def openai_like_body(self, system: str, user: str, images: Sequence[AgentImage],
                         include_model: bool, stream: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": openai_user_content(user, images)}],
            "stream": stream,
            "max_tokens": self.max_output_tokens(),
        }
        if include_model:
            body["model"] = self.config.model
        adapt_openai_strict_completion_body(body, self.config)
        apply_openai_reasoning(body, self.config)
        return body

    def anthropic_like_body(self, system: str, user: str, images: Sequence[AgentImage],
                            stream: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.config.model,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": anthropic_user_content(user, images)}],
            "max_tokens": self.max_output_tokens(),
        }
        if stream:
            body["stream"] = True
        apply_anthropic_reasoning(body, self.config)
        return body

    def google_body(self, system: str, user: str, images: Sequence[AgentImage]) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = [{"text": user}]
        for im in images:
            parts.append({"inlineData": {"mimeType": im.media_type, "data": im.data_base64}})
        body: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": self.max_output_tokens()},
        }
        apply_google_reasoning(body, self.config)
        return body

    # -- request planning --
    def _openai_plan(self, url, system, user, images, include_model, stream) -> _Plan:
        body = self.openai_like_body(system, user, images, include_model, stream)
        return _Plan(url, openai_headers(self.config, url).as_dict(), body, "openai")

    def _anthropic_plan(self, url, system, user, images, stream) -> _Plan:
        body = self.anthropic_like_body(system, user, images, stream)
        return _Plan(url, anthropic_headers(self.config, url).as_dict(), body, "anthropic")

    def _google_plan(self, system, user, images, stream) -> _Plan:
        enc = url_encode_path_segment(self.config.model)
        base = f"https://generativelanguage.googleapis.com/v1beta/models/{enc}"
        url = f"{base}:streamGenerateContent?alt=sse" if stream else f"{base}:generateContent"
        body = self.google_body(system, user, images)
        return _Plan(url, google_headers(self.config).as_dict(), body, "google")

    def plan(self, system: str, user: str, images: Sequence[AgentImage], stream: bool) -> _Plan:
        """Provider dispatch (generate_text / generate_text_stream match arms). Raises ProviderError."""
        c = self.config
        p = c.provider
        if p == "openai":
            return self._openai_plan("https://api.openai.com/v1/chat/completions", system, user, images, True, stream)
        if p == "azure":
            return self._openai_plan(build_azure_url(c), system, user, images, False, stream)
        if p == "ollama":
            return self._openai_plan(build_ollama_url(c.ollama_url), system, user, images, True, stream)
        if p == "custom" and c.api_mode == "anthropic_messages":
            return self._anthropic_plan(build_anthropic_url(c.custom_endpoint), system, user, images, stream)
        if p == "custom":
            url = build_custom_openai_url(c)
            return self._openai_plan(url, system, user, images, not is_azure_endpoint(url), stream)
        if p == "anthropic":
            return self._anthropic_plan("https://api.anthropic.com/v1/messages", system, user, images, stream)
        if p == "minimax":
            if images:
                raise ProviderError(_MINIMAX_IMAGE_ERR)
            base = c.custom_endpoint.strip() or "https://api.minimax.io/anthropic"
            return self._anthropic_plan(build_anthropic_url(base), system, user, images, stream)
        if p == "google":
            return self._google_plan(system, user, images, stream)
        raise ProviderError(f"Provider '{p}' is not supported by the backend HTTP Agent yet")

    # -- http --
    def _client_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"timeout": httpx.Timeout(float(REQUEST_TIMEOUT_SECS)), "trust_env": True}
        if self._transport is not None:
            kw["transport"] = self._transport
        else:
            try:
                from .. import proxy as _proxy
                kw["verify"] = not _proxy.accept_invalid_certs()
            except Exception:
                pass
        return kw

    @staticmethod
    def _content(plan: _Plan) -> bytes:
        return json.dumps(plan.body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _parse_complete(plan: _Plan, text: str) -> str:
        try:
            parsed = json.loads(text)
        except ValueError as e:
            raise ProviderError(f"Invalid LLM JSON: {e}")
        content = _EXTRACT[plan.kind](parsed)
        if not content:
            raise ProviderError(EMPTY_CONTENT_ERROR)
        return content

    @staticmethod
    def _check_cancel(cancel) -> None:
        if cancel is not None and cancel.is_cancelled():
            raise ProviderError("Agent turn cancelled")

    # async ------------------------------------------------------------
    async def _post_text(self, plan: _Plan) -> str:
        async with httpx.AsyncClient(**self._client_kwargs()) as client:
            try:
                resp = await client.post(plan.url, headers=plan.headers, content=self._content(plan))
            except httpx.HTTPError as e:
                raise ProviderError(f"LLM request failed: {e}")
            try:
                text = resp.text
            except Exception as e:
                raise ProviderError(f"Failed to read LLM response: {e}")
        if not resp.is_success:
            raise _http_error(resp, text)
        return self._parse_complete(plan, text)

    async def _post_stream(self, plan: _Plan, on_delta, cancel=None) -> str:
        collector = SseCollector(_DELTA[plan.kind], on_delta)
        async with httpx.AsyncClient(**self._client_kwargs()) as client:
            try:
                req = client.build_request("POST", plan.url, headers=plan.headers, content=self._content(plan))
                resp = await client.send(req, stream=True)
            except httpx.HTTPError as e:
                raise ProviderError(f"LLM request failed: {e}")
            try:
                if not resp.is_success:
                    try:
                        text = (await resp.aread()).decode(resp.encoding or "utf-8", errors="replace")
                    except httpx.HTTPError as e:
                        raise ProviderError(f"Failed to read LLM response: {e}")
                    raise _http_error(resp, text)
                try:
                    async for chunk in resp.aiter_bytes():
                        self._check_cancel(cancel)
                        collector.feed(chunk)
                except httpx.HTTPError as e:
                    raise ProviderError(f"LLM stream failed: {e}")
            finally:
                await resp.aclose()
        return collector.finish()

    async def generate_text(self, system: str, user: str, images: Sequence[AgentImage] = ()) -> str:
        return await self._post_text(self.plan(system, user, images, False))

    async def generate_text_stream(self, system: str, user: str, images: Sequence[AgentImage],
                                   on_delta: Callable[[str], Any], cancel=None) -> str:
        """`cancel`: optional object with is_cancelled(); checked between chunks."""
        if self.config.streaming_enabled is False:
            text = await self.generate_text(system, user, images)
            on_delta(text)
            return text
        if self.config.provider not in _STREAMING_PROVIDERS:
            return await self.generate_text(system, user, images)
        return await self._post_stream(self.plan(system, user, images, True), on_delta, cancel)

    # sync -------------------------------------------------------------
    def _post_text_sync(self, plan: _Plan) -> str:
        with httpx.Client(**self._client_kwargs()) as client:
            try:
                resp = client.post(plan.url, headers=plan.headers, content=self._content(plan))
            except httpx.HTTPError as e:
                raise ProviderError(f"LLM request failed: {e}")
            try:
                text = resp.text
            except Exception as e:
                raise ProviderError(f"Failed to read LLM response: {e}")
        if not resp.is_success:
            raise _http_error(resp, text)
        return self._parse_complete(plan, text)

    def _post_stream_sync(self, plan: _Plan, on_delta, cancel=None) -> str:
        collector = SseCollector(_DELTA[plan.kind], on_delta)
        with httpx.Client(**self._client_kwargs()) as client:
            try:
                req = client.build_request("POST", plan.url, headers=plan.headers, content=self._content(plan))
                resp = client.send(req, stream=True)
            except httpx.HTTPError as e:
                raise ProviderError(f"LLM request failed: {e}")
            try:
                if not resp.is_success:
                    try:
                        text = resp.read().decode(resp.encoding or "utf-8", errors="replace")
                    except httpx.HTTPError as e:
                        raise ProviderError(f"Failed to read LLM response: {e}")
                    raise _http_error(resp, text)
                try:
                    for chunk in resp.iter_bytes():
                        self._check_cancel(cancel)
                        collector.feed(chunk)
                except httpx.HTTPError as e:
                    raise ProviderError(f"LLM stream failed: {e}")
            finally:
                resp.close()
        return collector.finish()

    def generate_text_sync(self, system: str, user: str, images: Sequence[AgentImage] = ()) -> str:
        return self._post_text_sync(self.plan(system, user, images, False))

    def generate_text_stream_sync(self, system: str, user: str, images: Sequence[AgentImage],
                                  on_delta: Callable[[str], Any], cancel=None) -> str:
        if self.config.streaming_enabled is False:
            text = self.generate_text_sync(system, user, images)
            on_delta(text)
            return text
        if self.config.provider not in _STREAMING_PROVIDERS:
            return self.generate_text_sync(system, user, images)
        return self._post_stream_sync(self.plan(system, user, images, True), on_delta, cancel)


_STREAMING_PROVIDERS = ("openai", "azure", "ollama", "custom", "anthropic", "minimax", "google")
