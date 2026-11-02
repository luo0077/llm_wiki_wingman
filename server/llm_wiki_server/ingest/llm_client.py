"""Port of src/lib/llm-client.ts: ``stream_chat`` and friends.

Call-site translation (TS -> Python)::

    await streamChat(llmConfig, messages,
        { onToken, onDone, onError, onReasoningToken }, signal,
        { temperature: 0.1, reasoning, max_tokens })

    await stream_chat(llm_config, messages,
        StreamCallbacks(on_token=..., on_done=..., on_error=..., on_reasoning_token=...),
        signal, {"temperature": 0.1, "reasoning": reasoning, "max_tokens": n})

* ``llm_config``: dict with the TS ``LlmConfig`` camelCase keys (``llm_wiki_server.agent.provider.LlmConfig``
  instances are accepted too).
* ``messages``: ``[{"role": "system"|"user"|"assistant", "content": str | [blocks]}]`` where a block is
  ``{"type": "text", "text"}`` or ``{"type": "image", "mediaType", "dataBase64"}``.
* ``callbacks``: a :class:`StreamCallbacks` (or a dict with ``onToken``/``on_token`` style keys).  Each
  callback may be a plain function or a coroutine function.  ``on_done`` receives a
  :class:`StreamCompletion` or ``None`` (``None`` on user cancellation / non-token-limit early exits,
  exactly as the TS ``onDone()`` with no argument).
* ``signal``: ``None`` or anything abort-like: :class:`AbortSignal`, ``threading.Event`` /
  ``asyncio.Event`` (``is_set``), an object with ``is_cancelled()`` or an ``aborted`` attribute.
* ``request_overrides``: ``{"temperature", "top_p", "top_k", "max_tokens", "stop", "reasoning"}``;
  ``None`` values mean "unset".

Like the TS function, ``stream_chat`` reports failures through ``on_error`` and returns ``None``;
it only raises for programmer errors (e.g. an unknown provider).  ``collect_stream_chat`` wraps it
into the common "await the whole answer as a string, raise on error" shape.
"""
from __future__ import annotations

import asyncio
import codecs
import inspect
import math
import re
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Union

import httpx

from .http_fetch import (
    LLM_TIMEOUT,
    AbortError,
    is_fetch_network_error,
    make_async_client,
)
from .llm_jscompat import (
    JS_WS,
    dumps_js,
    js_json_loads,
    js_str,
    js_trim,
    utf16_len,
)
from .llm_providers import (
    ChatMessage,
    RequestOverrides,
    get_provider_config,
)
from .azure_openai import is_azure_openai_endpoint
from .reasoning_detector import count_reasoning_chars_in_line, extract_reasoning_text_from_line

__all__ = [
    "AbortError", "AbortSignal", "StreamCallbacks", "StreamCompletion", "stream_chat",
    "collect_stream_chat", "is_fetch_network_error", "is_reasoning_only_response_error",
    "buffered_stream_callbacks", "CLI_TRANSPORTS", "ChatMessage", "RequestOverrides",
]

# seconds per "minute" of requestTimeoutMinutes (tests shrink this)
_SECONDS_PER_MINUTE = 60.0
_POLL_INTERVAL = 0.02
REASONING_DIAGNOSTIC_THRESHOLD = 200


# ----------------------------------------------------------------- abort signal

class AbortSignal:
    """Minimal thread-safe AbortController/AbortSignal pair (``signal.abort()``; ``signal.aborted``)."""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def abort(self) -> None:
        self._event.set()

    # convenience aliases
    def is_set(self) -> bool:
        return self._event.is_set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def is_aborted(signal: Any) -> bool:
    """True when an abort-like object is signalled (see module docstring)."""
    if signal is None:
        return False
    for attr in ("aborted", "is_cancelled", "cancelled", "is_set"):
        v = getattr(signal, attr, None)
        if v is None:
            continue
        try:
            r = v() if callable(v) else v
        except Exception:
            continue
        if isinstance(r, bool):
            return r
    return False


# ----------------------------------------------------------------- callbacks

@dataclass
class StreamCompletion:
    finish_reason: Optional[str] = None
    truncated: bool = False

    @property
    def finishReason(self) -> Optional[str]:  # TS-style alias
        return self.finish_reason

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return (other.get("finishReason") == self.finish_reason
                    and other.get("truncated") == self.truncated)
        if isinstance(other, StreamCompletion):
            return self.finish_reason == other.finish_reason and self.truncated == other.truncated
        return NotImplemented


Callback = Optional[Callable[..., Union[None, Awaitable[None]]]]


@dataclass
class StreamCallbacks:
    on_token: Callback = None
    on_done: Callback = None
    on_error: Callback = None
    on_reasoning_token: Callback = None

    @classmethod
    def coerce(cls, value: Any) -> "StreamCallbacks":
        if isinstance(value, StreamCallbacks):
            return value
        if isinstance(value, Mapping):
            def pick(snake: str, camel: str) -> Callback:
                return value.get(snake) or value.get(camel)
            return cls(pick("on_token", "onToken"), pick("on_done", "onDone"),
                       pick("on_error", "onError"), pick("on_reasoning_token", "onReasoningToken"))
        return cls(getattr(value, "on_token", None), getattr(value, "on_done", None),
                   getattr(value, "on_error", None), getattr(value, "on_reasoning_token", None))


async def _call(cb: Callback, *args: Any) -> None:
    if cb is None:
        return
    r = cb(*args)
    if inspect.isawaitable(r):
        await r


def buffered_stream_callbacks(callbacks: StreamCallbacks) -> StreamCallbacks:
    """Accumulate everything and emit it once on completion (``streamingEnabled === false`` for CLIs)."""
    content: List[str] = []
    reasoning: List[str] = []

    def on_token(t: str) -> None:
        content.append(t)

    def on_reasoning(t: str) -> None:
        reasoning.append(t)

    async def on_done(completion: Optional[StreamCompletion] = None) -> None:
        if reasoning:
            await _call(callbacks.on_reasoning_token, "".join(reasoning))
        if content and "".join(content):
            await _call(callbacks.on_token, "".join(content))
        await _call(callbacks.on_done, completion)

    return StreamCallbacks(on_token, on_done, callbacks.on_error, on_reasoning)


# Subprocess transports (claude-code / codex-cli) do not exist in the headless server. A deployment
# may register `async def fn(config, messages, callbacks, signal, request_overrides)` here.
CLI_TRANSPORTS: Dict[str, Callable[..., Awaitable[None]]] = {}


# ----------------------------------------------------------------- stream parsing helpers

def _g(obj: Any, key: Any) -> Any:
    if isinstance(key, int):
        return obj[key] if isinstance(obj, list) and -1 < key < len(obj) else None
    return obj.get(key) if isinstance(obj, dict) else None


def _first_non_null(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def stream_finish_reason(record: str) -> Optional[str]:
    payload = js_trim(record[5:]) if record.startswith("data:") else js_trim(record)
    if not payload.startswith("{"):
        return None
    try:
        value = js_json_loads(payload)
    except Exception:
        return None
    return _first_non_null(
        _g(_g(_g(value, "choices"), 0), "finish_reason"),
        _g(_g(value, "delta"), "stop_reason"),
        _g(value, "stop_reason"),
        _g(_g(_g(value, "candidates"), 0), "finishReason"),
    )


def response_finish_reason(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    return _first_non_null(
        _g(_g(_g(payload, "choices"), 0), "finish_reason"),
        _g(payload, "stop_reason"),
        _g(_g(_g(payload, "candidates"), 0), "finishReason"),
    )


_TOKEN_LIMIT_RE = re.compile(r"(length|max_tokens|max_output_tokens)", re.IGNORECASE | re.ASCII)


def is_token_limit_finish_reason(reason: Any) -> bool:
    if reason is None or reason == "" or reason is False:
        return False
    return _TOKEN_LIMIT_RE.fullmatch(js_str(reason)) is not None


def parse_endpoint_error_envelope(record: str) -> Optional[Exception]:
    payload = js_trim(record[5:]) if record.startswith("data:") else record
    if not payload.startswith("{"):
        return None
    try:
        parsed = js_json_loads(payload)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    err = parsed.get("error")
    message = err if isinstance(err, str) else (err.get("message") if isinstance(err, dict) else None)
    if message is None or message == "" or message is False or message == 0:
        return None
    code = ""
    if isinstance(err, dict) and "code" in err:
        code = f" {js_str(err['code'])}"
    return Exception(f"LLM endpoint error{code}: {js_str(message)}")


_ESCAPED_SEP = re.compile(r"(?:\\r)?\\n(?:\\r)?\\n(?=data:)")
_LEADING_DATA = re.compile("^" + JS_WS + "*data:")


def split_final_stream_records(text: str) -> List[str]:
    """Split a fully buffered body; handles escaped record separators (see TS comments)."""
    if re.search(r"[\r\n]", text) or not _LEADING_DATA.search(text):
        return re.split(r"\r?\n", text)

    records: List[str] = []
    record_start = 0
    for match in _ESCAPED_SEP.finditer(text):
        candidate = js_trim(text[record_start:match.start()])
        payload = js_trim(candidate[5:]) if candidate.startswith("data:") else ""
        complete = payload == "[DONE]"
        if not complete and payload.startswith("{"):
            try:
                js_json_loads(payload)
                complete = True
            except Exception:
                pass
        if complete:
            records.append(candidate)
            record_start = match.end()
    records.append(text[record_start:])
    return records


_CANCELLED_RE = re.compile(r"request cancel(?:l)?ed", re.IGNORECASE | re.ASCII)


def is_request_cancelled_error(err: Any) -> bool:
    message = str(err) if isinstance(err, BaseException) else js_str(err)
    return _CANCELLED_RE.fullmatch(js_trim(message)) is not None


_REASONING_ONLY_RE = re.compile(
    r"^Model produced [0-9,]+ characters of reasoning / chain-of-thought, but no actual response content\.")


def is_reasoning_only_response_error(err: Any) -> bool:
    message = str(err) if isinstance(err, BaseException) else js_str(err)
    return _REASONING_ONLY_RE.search(message) is not None


def _should_retry_without_temperature(config: Mapping[str, Any], status: int, detail: str,
                                      overrides: Optional[Mapping[str, Any]]) -> bool:
    if config.get("provider") != "custom" or (overrides or {}).get("temperature") is None:
        return False
    if status != 400 and status != 422:
        return False
    d = detail.lower()
    return "temperature" in d and any(
        w in d for w in ("unsupported", "not support", "unknown", "not allowed", "only", "invalid"))


# ----------------------------------------------------------------- the abort/timeout race

class _UserAbort(Exception):
    pass


class _TimedOut(Exception):
    pass


class _State:
    timeout_fired = False


async def _race(coro: Awaitable[None], signal: Any, timeout_s: float, state: _State) -> None:
    """Run `coro`, cancelling it on user abort (-> _UserAbort) or the long-horizon backstop (-> _TimedOut)."""
    task = asyncio.ensure_future(coro)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    try:
        while True:
            remaining = deadline - loop.time()
            done, _ = await asyncio.wait({task}, timeout=max(0.0, min(_POLL_INTERVAL, remaining)))
            if done:
                return task.result()
            if is_aborted(signal):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise _UserAbort()
            if loop.time() >= deadline:
                state.timeout_fired = True
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise _TimedOut()
    except asyncio.CancelledError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


def _timeout_message(timeout_s_minutes: float) -> str:
    return (f"Request timed out after {int(math.floor(timeout_s_minutes + 0.5))} min. "
            "Try a faster model or a smaller context.")


def _timeout_minutes(config: Mapping[str, Any]) -> float:
    v = config.get("requestTimeoutMinutes")
    if v is None:
        v = 30
    return max(1, min(1440, v))


def _as_config_dict(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError("llm config must be a dict with the TS LlmConfig keys")


# ----------------------------------------------------------------- stream_chat

async def stream_chat(
    config: Any,
    messages: List[ChatMessage],
    callbacks: Any,
    signal: Any = None,
    request_overrides: Optional[RequestOverrides] = None,
    *,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> None:
    """Port of ``streamChat`` (see module docstring for argument shapes)."""
    cfg = _as_config_dict(config)
    cb = StreamCallbacks.coerce(callbacks)

    provider = cfg.get("provider")
    if provider in ("claude-code", "codex-cli"):
        cli = CLI_TRANSPORTS.get(provider)
        if cli is None:
            await _call(cb.on_error, Exception(
                f"The {provider} provider runs a local CLI subprocess and is not available on the "
                "headless server. Choose an HTTP provider."))
            return
        use_cb = buffered_stream_callbacks(cb) if cfg.get("streamingEnabled") is False else cb
        await cli(cfg, messages, use_cb, signal, request_overrides)
        return

    provider_config = get_provider_config(cfg)

    timeout_minutes = _timeout_minutes(cfg)
    timeout_s = timeout_minutes * _SECONDS_PER_MINUTE
    state = _State()

    try:
        await _race(
            _stream_attempt(cfg, provider_config, messages, cb, signal, request_overrides, state,
                            timeout_minutes, transport),
            signal, timeout_s, state)
    except _UserAbort:
        await _call(cb.on_done)
    except _TimedOut:
        await _call(cb.on_error, Exception(_timeout_message(timeout_minutes)))


async def _stream_attempt(cfg, provider_config, messages, cb: StreamCallbacks, signal, overrides,
                          state: _State, timeout_minutes: float, transport) -> None:
    on_token, on_done, on_error = cb.on_token, cb.on_done, cb.on_error
    timeout_msg = _timeout_message(timeout_minutes)

    # --- build + send -------------------------------------------------------------
    try:
        body = provider_config.build_body(messages, overrides)
    except Exception as err:
        if is_aborted(signal):
            await _call(on_done)
            return
        await _call(on_error, err)
        return

    client = make_async_client(timeout=LLM_TIMEOUT, follow_redirects=True, transport=transport)
    response: Optional[httpx.Response] = None
    try:
        try:
            request = client.build_request(
                "POST", provider_config.url, headers=provider_config.headers,
                content=dumps_js(body).encode("utf-8"))
            response = await client.send(request, stream=True)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            if is_aborted(signal):
                await _call(on_done)
                return
            if isinstance(err, AbortError) or is_request_cancelled_error(err):
                if state.timeout_fired:
                    await _call(on_error, Exception(timeout_msg))
                    return
                await _call(on_done)
                return
            if is_fetch_network_error(err):
                if state.timeout_fired:
                    await _call(on_error, Exception(timeout_msg))
                    return
                await _call(on_error, Exception(
                    f"Network error reaching {provider_config.url}. "
                    "Check endpoint URL, API key, and connectivity."))
                return
            await _call(on_error, err if isinstance(err, Exception) else Exception(str(err)))
            return

        await _handle_response(cfg, provider_config, messages, cb, signal, overrides, state,
                               timeout_minutes, transport, response)
    finally:
        if response is not None:
            try:
                await response.aclose()
            except Exception:
                pass
        await client.aclose()


async def _handle_response(cfg, provider_config, messages, cb: StreamCallbacks, signal, overrides,
                           state: _State, timeout_minutes: float, transport,
                           response: httpx.Response) -> None:
    on_token, on_done, on_error = cb.on_token, cb.on_done, cb.on_error
    timeout_msg = _timeout_message(timeout_minutes)

    if not (200 <= response.status_code < 300):
        error_detail = f"HTTP {response.status_code}: {response.reason_phrase}"
        try:
            text = (await response.aread()).decode(response.encoding or "utf-8", errors="replace")
            if text:
                error_detail += f" — {text}"
        except Exception:
            pass
        if _should_retry_without_temperature(cfg, response.status_code, error_detail, overrides):
            retry_overrides = {k: v for k, v in (overrides or {}).items() if k != "temperature"}
            await response.aclose()
            await _stream_attempt(cfg, provider_config, messages, cb, signal, retry_overrides, state,
                                  timeout_minutes, transport)
            return
        if response.status_code == 404 and (
                cfg.get("provider") == "azure"
                or (cfg.get("provider") == "custom"
                    and is_azure_openai_endpoint(cfg.get("customEndpoint") or ""))):
            await _call(on_error, Exception(
                f"{error_detail} — Azure 404 usually means the deployment name is wrong. "
                "Set Model to your Azure deployment name (not the model SKU), "
                "and Endpoint to https://<resource>.openai.azure.com "
                "or .../openai/deployments/<deployment-name>."))
            return
        await _call(on_error, Exception(error_detail))
        return

    # --- non-streaming wire -----------------------------------------------------------
    if not provider_config.streaming:
        try:
            raw = await response.aread()
            payload = js_json_loads(raw.decode("utf-8-sig", errors="replace"))
            content = provider_config.parse_response(payload)
            finish_reason = response_finish_reason(payload)
            if not content:
                if is_token_limit_finish_reason(finish_reason):
                    await _call(on_done, StreamCompletion(finish_reason, True))
                    return
                await _call(on_error, Exception("Model returned an empty non-streaming response"))
                return
            await _call(on_token, content)
            await _call(on_done, StreamCompletion(finish_reason, is_token_limit_finish_reason(finish_reason)))
        except asyncio.CancelledError:
            raise
        except Exception as err:
            if state.timeout_fired:
                await _call(on_error, Exception(timeout_msg))
                return
            if is_aborted(signal) or isinstance(err, AbortError) or is_request_cancelled_error(err):
                await _call(on_done)
                return
            if is_fetch_network_error(err):
                await _call(on_error, Exception("Connection lost while reading the complete response. Try again."))
                return
            await _call(on_error, err)
        return

    # --- streaming wire ---------------------------------------------------------------
    decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
    line_buffer = ""
    content_chars = 0
    reasoning_chars = 0
    finish_reason: Optional[str] = None

    async def process_record(line: str) -> Optional[Exception]:
        nonlocal content_chars, reasoning_chars, finish_reason
        trimmed = js_trim(line)
        if not trimmed:
            return None
        reasoning_chars += count_reasoning_chars_in_line(trimmed)
        fr = stream_finish_reason(trimmed)
        finish_reason = fr if fr is not None else finish_reason
        for part in extract_reasoning_text_from_line(trimmed):
            await _call(cb.on_reasoning_token, part)
        token = provider_config.parse_stream(trimmed)
        if token is not None:
            content_chars += utf16_len(token)
            await _call(on_token, token)
            return None
        return parse_endpoint_error_envelope(trimmed)

    async def stop_for_endpoint_error(error: Exception) -> None:
        # Close the still-open body so the transport doesn't keep the connection alive.
        try:
            await response.aclose()
        except Exception:
            pass
        await _call(on_error, error)

    try:
        async for chunk in response.aiter_bytes():
            text = line_buffer + decoder.decode(chunk, False)
            lines = text.split("\n")
            line_buffer = lines.pop()
            for line in lines:
                endpoint_error = await process_record(line)
                if endpoint_error is not None:
                    await stop_for_endpoint_error(endpoint_error)
                    return
        final_text = line_buffer + decoder.decode(b"", True)
        for line in split_final_stream_records(final_text):
            endpoint_error = await process_record(line)
            if endpoint_error is not None:
                await stop_for_endpoint_error(endpoint_error)
                return

        if content_chars == 0 and is_token_limit_finish_reason(finish_reason):
            await _call(on_done, StreamCompletion(finish_reason, True))
            return
        if content_chars == 0 and reasoning_chars >= REASONING_DIAGNOSTIC_THRESHOLD:
            await _call(on_error, Exception(
                f"Model produced {reasoning_chars:,} characters of reasoning / chain-of-thought, "
                "but no actual response content. "
                "This usually means the endpoint hit a thinking-token limit, the model didn't transition "
                "from thinking to answering, "
                "or the endpoint is misbehaving (the official Anthropic / OpenAI APIs don't have this issue). "
                "Try a shorter input, increase max_tokens, or switch to a different model in Settings."))
            return
        await _call(on_done, StreamCompletion(finish_reason, is_token_limit_finish_reason(finish_reason)))
    except asyncio.CancelledError:
        raise
    except Exception as err:
        is_abort = (is_aborted(signal) or state.timeout_fired or isinstance(err, AbortError)
                    or is_request_cancelled_error(err))
        if is_abort:
            if state.timeout_fired:
                await _call(on_error, Exception(timeout_msg))
                return
            await _call(on_done)
            return
        if is_fetch_network_error(err):
            await _call(on_error, Exception("Connection lost during streaming. Try again."))
            return
        await _call(on_error, err)


# ----------------------------------------------------------------- convenience

async def collect_stream_chat(
    config: Any,
    messages: List[ChatMessage],
    signal: Any = None,
    request_overrides: Optional[RequestOverrides] = None,
    *,
    on_token: Callback = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> str:
    """Run ``stream_chat`` and return the concatenated tokens; raise the ``on_error`` exception.

    Equivalent to the TS pattern ``await new Promise(resolve => streamChat(..., {onToken: r += t,
    onDone: resolve, onError: e => { streamError = e; resolve() }})); if (streamError) throw streamError``.
    """
    parts: List[str] = []
    errors: List[Exception] = []

    async def tok(t: str) -> None:
        parts.append(t)
        await _call(on_token, t)

    def err(e: Exception) -> None:
        errors.append(e)

    await stream_chat(config, messages, StreamCallbacks(tok, None, err), signal, request_overrides,
                      transport=transport)
    if errors:
        raise errors[0]
    return "".join(parts)
