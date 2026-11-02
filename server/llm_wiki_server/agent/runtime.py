"""Port of agent/runtime.rs: the backend Agent turn loop.

Shared by the desktop-style RPC commands (`agent_start_turn*`, commands/agent.py)
and the `/api/v1` REST chat endpoints (commands/rest_api.py), which use the
module-level adapters `run_chat` and `redact_for_external_api` at the bottom.

Representation notes (Rust -> Python):
  * `Result<T, String>`            -> return T / raise Exception(message)
  * `AgentEvent` / `AgentToolEvent` -> events.AgentEvent (dict) / types.AgentToolEvent
  * async + tokio::select! cancel  -> asyncio; blocking callables run on a daemon
    thread and are raced against the cancellation token (same contract as Rust:
    "stop the turn promptly", not "kill the OS work").
  * `.len()` on Rust strings is a UTF-8 byte count; usage numbers keep that.
"""
from __future__ import annotations

import asyncio
import dataclasses
import importlib
import inspect
import json
import os
import string
import threading
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from . import context as _context
from .cancel import CANCELLED_MESSAGE, AgentCancelled
from .events import AgentEvent
from .events import redact_for_external_api as _redact_event
from .permissions import AgentCapability, PermissionPolicy
from .router import QueryIntent, RouterDecision, route_query
from .skills import AgentSkill, load_project_skills
from .types import (
    AgentChatRequest,
    AgentChatResponse,
    AgentImage,
    AgentMode,
    AgentReference,
    AgentRetrievalMode,
    AgentSkillMode,
    AgentToolEvent,
    AgentToolOptions,
    AgentUsage,
    AgentUserInputField,
    AgentUserInputOption,
    AgentUserInputRequest,
)
from .workspace import AGENT_WORKSPACE_DIR, agent_workspace_display

# These limits are intentionally enforced in the backend Agent rather than the
# UI. API and MCP callers bypass the UI, so safety and cost boundaries must
# live here.
DEFAULT_CHAT_SEARCH_RESULTS = 5
MAX_CHAT_SEARCH_RESULTS = 10
MAX_IMAGES_PER_TURN = 5
MAX_IMAGE_BASE64_BYTES = 7 * 1024 * 1024
MAX_AGENT_TOOL_ITERATIONS = 8
AGENT_STRUCTURED_MAX_TOKENS = 8192
AGENT_SKILL_STRUCTURED_MAX_TOKENS = 16384
MAX_SKILL_REFERENCE_BYTES = 256 * 1024
MAX_USER_INPUT_FIELDS = 12
MAX_USER_INPUT_OPTIONS = 8
MAX_USER_INPUT_TEXT_CHARS = 400
SHELL_APPROVAL_REQUIRED_OBSERVATION = "shell.exec.approval_required"

AgentEventSink = Callable[[AgentEvent], None]


# --------------------------------------------------------------------------
# Lazy sibling access (provider / tools are imported at call time)
# --------------------------------------------------------------------------
def _tools():
    return importlib.import_module(f"{__package__}.tools")


def _provider():
    return importlib.import_module(f"{__package__}.provider")


def _new_llm_client(config: Any):
    """`LlmClient::new(config)`. Module-level so tests can substitute a fake provider."""
    mod = _provider()
    cls = mod.LlmClient
    factory = getattr(cls, "new", None)
    return factory(config) if factory is not None else cls(config)


def _new_tool_registry():
    """`tools::BuiltinToolRegistry::default()`."""
    return _tools().BuiltinToolRegistry()


def _call_or_attr(obj: Any, name: str) -> Any:
    v = getattr(obj, name)
    return v() if callable(v) else v


def _llm_usable(cfg: Any) -> bool:
    if cfg is None:
        return False
    fn = getattr(cfg, "is_usable_for_backend_http", None)
    if fn is not None:
        return bool(fn())
    if isinstance(cfg, dict):  # defensive: raw camelCase dict
        provider = cfg.get("provider")
        has_model = bool(str(cfg.get("model") or "").strip())
        if provider in ("openai", "anthropic", "google", "azure", "minimax"):
            return has_model and bool(str(cfg.get("apiKey") or "").strip())
        if provider == "ollama":
            return has_model and bool(str(cfg.get("ollamaUrl") or "").strip())
        if provider == "custom":
            return has_model and bool(str(cfg.get("customEndpoint") or "").strip())
    return False


def _llm_max_context_size(cfg: Any) -> Optional[int]:
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get("maxContextSize")
    return getattr(cfg, "max_context_size", None)


def _coerce_llm_config(cfg: Any) -> Any:
    """serde `from_value::<LlmConfig>` on the app-state value; None when it does not parse."""
    if cfg is None or not isinstance(cfg, dict):
        return cfg
    try:
        mod = _provider()
        return mod.LlmConfig.from_dict(cfg)
    except Exception:  # noqa: BLE001 - unparsable config == no config
        return None


# --------------------------------------------------------------------------
# Small helpers mirroring Rust std semantics
# --------------------------------------------------------------------------
def _utf8_len(s: str) -> int:
    return len(s.encode("utf-8", "surrogatepass"))


_ASCII_LOWER = {ord(c): ord(c.lower()) for c in string.ascii_uppercase}


def _ascii_lower(s: str) -> str:
    return s.translate(_ASCII_LOWER)


def _eq_ic(a: str, b: str) -> bool:
    """`eq_ignore_ascii_case`."""
    return _ascii_lower(a) == _ascii_lower(b)


def _opt_trim(v: Optional[str]) -> Optional[str]:
    return v.strip() if v is not None else None


def _non_empty(v: Optional[str]) -> Optional[str]:
    return v if v else None


def _os_err(e: BaseException) -> str:
    if isinstance(e, OSError) and e.errno:
        return f"{e.strerror} (os error {e.errno})"
    return str(e)


def _value_to_string(value: Any) -> str:
    """`serde_json::Value::to_string()` (compact, keys sorted like serde's BTreeMap)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _file_name(path: str) -> Optional[str]:
    name = Path(path).name
    return name or None


def _path_starts_with(path: PurePath, base: PurePath) -> bool:
    return path == base or base in path.parents


def _is_path_absolute(p: str) -> bool:
    return Path(p).is_absolute()


# --------------------------------------------------------------------------
# Strict JSON parsing (serde_json::from_str semantics for the derived structs)
# --------------------------------------------------------------------------
class _SerdeError(ValueError):
    pass


def _loads_serde(text: str) -> Any:
    def reject(constant: str):
        raise _SerdeError(f"invalid number: {constant}")

    try:
        return json.loads(text, parse_constant=reject)
    except _SerdeError:
        raise
    except ValueError as e:
        raise _SerdeError(str(e))


def _f_opt_str(d: dict, key: str) -> Optional[str]:
    v = d.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise _SerdeError(f"invalid type for field `{key}`, expected a string")
    return v


def _f_str_default(d: dict, key: str) -> str:
    if key not in d:
        return ""
    v = d[key]
    if not isinstance(v, str):
        raise _SerdeError(f"invalid type for field `{key}`, expected a string")
    return v


def _f_opt_uint(d: dict, key: str) -> Optional[int]:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > 0xFFFFFFFFFFFFFFFF:
        raise _SerdeError(f"invalid value for field `{key}`, expected an unsigned integer")
    return v


def _f_opt_bool(d: dict, key: str) -> Optional[bool]:
    v = d.get(key)
    if v is None:
        return None
    if not isinstance(v, bool):
        raise _SerdeError(f"invalid type for field `{key}`, expected a boolean")
    return v


@dataclass
class ModelToolCall:
    tool: str
    query: Optional[str] = None
    command: Optional[str] = None
    timeout_seconds: Optional[int] = None
    path: Optional[str] = None
    content: Optional[str] = None
    # Overwriting existing wiki pages is destructive. The planner may set this
    # only when the user explicitly asks to update/overwrite an existing page;
    # the tool defaults to create-only when the field is absent.
    allow_overwrite: Optional[bool] = None

    @classmethod
    def from_value(cls, v: Any) -> "ModelToolCall":
        if not isinstance(v, dict):
            raise _SerdeError("invalid type, expected struct ModelToolCall")
        if "tool" not in v:
            raise _SerdeError("missing field `tool`")
        if not isinstance(v["tool"], str):
            raise _SerdeError("invalid type for field `tool`, expected a string")
        return cls(
            tool=v["tool"],
            query=_f_opt_str(v, "query"),
            command=_f_opt_str(v, "command"),
            timeout_seconds=_f_opt_uint(v, "timeoutSeconds"),
            path=_f_opt_str(v, "path"),
            content=_f_opt_str(v, "content"),
            allow_overwrite=_f_opt_bool(v, "allowOverwrite"),
        )


@dataclass
class ModelToolPlan:
    tool_calls: List[ModelToolCall] = field(default_factory=list)

    @classmethod
    def from_value(cls, v: Any) -> "ModelToolPlan":
        if not isinstance(v, dict):
            raise _SerdeError("invalid type, expected struct ModelToolPlan")
        calls = v.get("toolCalls", [])
        if not isinstance(calls, list):
            raise _SerdeError("invalid type for field `toolCalls`, expected a sequence")
        return cls([ModelToolCall.from_value(c) for c in calls])


@dataclass
class AgentLoopAction:
    action: str = ""
    tool: Optional[str] = None
    answer: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    query: Optional[str] = None
    skill: Optional[str] = None
    command: Optional[str] = None
    timeout_seconds: Optional[int] = None
    path: Optional[str] = None
    content: Optional[str] = None
    allow_overwrite: Optional[bool] = None
    include_content: Optional[bool] = None
    top_k: Optional[int] = None
    fields: Any = None
    questions: Any = None

    @classmethod
    def from_value(cls, v: Any) -> "AgentLoopAction":
        if not isinstance(v, dict):
            raise _SerdeError("invalid type, expected struct AgentLoopAction")
        return cls(
            action=_f_str_default(v, "action"),
            tool=_f_opt_str(v, "tool"),
            answer=_f_opt_str(v, "answer"),
            title=_f_opt_str(v, "title"),
            description=_f_opt_str(v, "description"),
            query=_f_opt_str(v, "query"),
            skill=_f_opt_str(v, "skill"),
            command=_f_opt_str(v, "command"),
            timeout_seconds=_f_opt_uint(v, "timeoutSeconds"),
            path=_f_opt_str(v, "path"),
            content=_f_opt_str(v, "content"),
            allow_overwrite=_f_opt_bool(v, "allowOverwrite"),
            include_content=_f_opt_bool(v, "includeContent"),
            top_k=_f_opt_uint(v, "topK"),
            fields=v.get("fields"),
            questions=v.get("questions"),
        )

    @classmethod
    def from_str(cls, text: str) -> "AgentLoopAction":
        return cls.from_value(_loads_serde(text))


@dataclass
class AgentObservation:
    tool: str
    summary: str


# --------------------------------------------------------------------------
# Tool output parsing (serde Deserialize of tools::*Output)
# --------------------------------------------------------------------------
def _as_dict(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if to_dict is not None and not isinstance(value, dict):
        return to_dict()
    return value


def _pick(d: dict, *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _reference_from(value: Any) -> AgentReference:
    if isinstance(value, AgentReference):
        return value
    return AgentReference.from_dict(_as_dict(value))


def _references_from(value: Any, what: str) -> List[AgentReference]:
    if not isinstance(value, list):
        raise _SerdeError(f"Invalid {what} result: invalid type, expected a sequence")
    try:
        return [_reference_from(v) for v in value]
    except ValueError as e:
        raise _SerdeError(f"Invalid {what} result: {e}")


@dataclass
class WikiSearchToolOutput:
    mode: str
    token_hits: int
    vector_hits: int
    graph_hits: int
    references: List[AgentReference]

    @classmethod
    def from_value(cls, value: Any) -> "WikiSearchToolOutput":
        d = _as_dict(value)
        if not isinstance(d, dict):
            raise _SerdeError("Invalid wiki.search result: invalid type, expected struct WikiSearchToolOutput")
        out = []
        for key, alt in (("tokenHits", "token_hits"), ("vectorHits", "vector_hits"), ("graphHits", "graph_hits")):
            v = _pick(d, key, alt)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise _SerdeError(f"Invalid wiki.search result: missing field `{key}`")
            out.append(v)
        mode = d.get("mode")
        if not isinstance(mode, str):
            raise _SerdeError("Invalid wiki.search result: missing field `mode`")
        return cls(mode, out[0], out[1], out[2], _references_from(d.get("references"), "wiki.search"))


@dataclass
class WorkspaceWriteOutput:
    path: str
    bytes: int
    existed_before: bool = False
    previous_content: Optional[str] = None

    @classmethod
    def from_value(cls, value: Any, what: str) -> "WorkspaceWriteOutput":
        d = _as_dict(value)
        if not isinstance(d, dict):
            raise _SerdeError(f"Invalid {what} result: invalid type, expected struct WorkspaceWriteOutput")
        path = d.get("path")
        nbytes = d.get("bytes")
        if not isinstance(path, str):
            raise _SerdeError(f"Invalid {what} result: missing field `path`")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise _SerdeError(f"Invalid {what} result: missing field `bytes`")
        eb = _pick(d, "existedBefore", "existed_before")
        pc = _pick(d, "previousContent", "previous_content")
        return cls(path, nbytes, bool(eb) if eb is not None else False, pc if isinstance(pc, str) else None)


@dataclass
class WikiWriteOutput:
    reference: AgentReference
    existed_before: bool = False
    previous_content: Optional[str] = None

    @classmethod
    def from_value(cls, value: Any) -> "WikiWriteOutput":
        d = _as_dict(value)
        if not isinstance(d, dict):
            raise _SerdeError("Invalid wiki.write_page result: invalid type, expected struct WikiWriteOutput")
        try:
            ref = AgentReference.from_dict(d)
        except ValueError as e:
            raise _SerdeError(f"Invalid wiki.write_page result: {e}")
        eb = _pick(d, "existedBefore", "existed_before")
        pc = _pick(d, "previousContent", "previous_content")
        return cls(ref, bool(eb) if eb is not None else False, pc if isinstance(pc, str) else None)


@dataclass
class ShellExecToolOutput:
    command: str
    exit_code: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    generated_files: List[WorkspaceWriteOutput] = field(default_factory=list)

    @classmethod
    def from_value(cls, value: Any) -> "ShellExecToolOutput":
        d = _as_dict(value)
        if not isinstance(d, dict):
            raise _SerdeError("Invalid shell.exec result: invalid type, expected struct ShellExecToolOutput")
        command = d.get("command")
        stdout = d.get("stdout")
        stderr = d.get("stderr")
        timed_out = _pick(d, "timedOut", "timed_out")
        if not isinstance(command, str):
            raise _SerdeError("Invalid shell.exec result: missing field `command`")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise _SerdeError("Invalid shell.exec result: missing field `stdout`/`stderr`")
        if not isinstance(timed_out, bool):
            raise _SerdeError("Invalid shell.exec result: missing field `timedOut`")
        code = _pick(d, "exitCode", "exit_code")
        if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
            raise _SerdeError("Invalid shell.exec result: invalid type for field `exitCode`")
        files = _pick(d, "generatedFiles", "generated_files") or []
        if not isinstance(files, list):
            raise _SerdeError("Invalid shell.exec result: invalid type for field `generatedFiles`")
        return cls(command, code, stdout, stderr, timed_out,
                   [WorkspaceWriteOutput.from_value(f, "shell.exec") for f in files])


def _rust_option_debug(v: Optional[int]) -> str:
    """`{:?}` of Option<i32>."""
    return "None" if v is None else f"Some({v})"


def _rust_bool(v: bool) -> str:
    return "true" if v else "false"


# --------------------------------------------------------------------------
# Cancellation plumbing
# --------------------------------------------------------------------------
class EventCancellation:
    """Adapter giving a `threading.Event` the AgentCancellationToken interface."""

    def __init__(self, event: threading.Event):
        self.event = event

    def is_cancelled(self) -> bool:
        return self.event.is_set()

    def check(self) -> None:
        if self.event.is_set():
            raise AgentCancelled(CANCELLED_MESSAGE)

    async def cancelled(self) -> None:
        while not self.event.is_set():
            await asyncio.sleep(0.05)


def _as_cancellation(token: Any) -> Any:
    if token is None:
        return None
    if isinstance(token, threading.Event):
        return EventCancellation(token)
    return token


def check_cancel(cancellation: Any) -> None:
    if cancellation is not None:
        cancellation.check()


async def _invoke(fn: Callable[..., Any], *args: Any) -> Any:
    """Run `fn(*args)` to completion: coroutine functions are awaited in place;
    blocking callables run on a daemon thread (so a stuck call never delays shutdown)."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    loop = asyncio.get_running_loop()
    fut: "asyncio.Future[Any]" = loop.create_future()

    def deliver(setter: Callable[[Any], None], payload: Any) -> None:
        if not fut.done():
            setter(payload)

    def runner() -> None:
        try:
            res = fn(*args)
        except BaseException as e:  # noqa: BLE001
            try:
                loop.call_soon_threadsafe(deliver, fut.set_exception, e)
            except RuntimeError:
                pass
            return
        try:
            loop.call_soon_threadsafe(deliver, fut.set_result, res)
        except RuntimeError:
            pass

    threading.Thread(target=runner, daemon=True, name="llm-wiki-agent-call").start()
    res = await fut
    if inspect.isawaitable(res):
        res = await res
    return res


async def _race_cancellation(fn: Callable[..., Any], args: Tuple[Any, ...], cancellation: Any) -> Any:
    """`tokio::select! { biased; _ = token.cancelled() => Err(cancelled), r = future => r }`."""
    if cancellation is None:
        return await _invoke(fn, *args)
    if cancellation.is_cancelled():
        raise AgentCancelled(CANCELLED_MESSAGE)
    task = asyncio.ensure_future(_invoke(fn, *args))
    waiter = asyncio.ensure_future(cancellation.cancelled())
    try:
        await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if cancellation.is_cancelled():
            raise AgentCancelled(CANCELLED_MESSAGE)
        return task.result()
    finally:
        for t in (task, waiter):
            if not t.done():
                t.cancel()
        # Retrieve any exception of an abandoned task so asyncio does not warn.
        for t in (task, waiter):
            if t.done() and not t.cancelled():
                t.exception()


async def execute_tool_with_cancellation(fn: Callable[..., Any], args: Tuple[Any, ...],
                                         cancellation: Any) -> Any:
    return await _race_cancellation(fn, args, cancellation)


async def generate_with_cancellation(provider: Any, system: str, user: str, images: Sequence[AgentImage],
                                     cancellation: Any) -> str:
    return await _race_cancellation(provider.generate_text, (system, user, list(images)), cancellation)


async def generate_with_cancellation_stream(provider: Any, system: str, user: str, images: Sequence[AgentImage],
                                            cancellation: Any, on_delta: Callable[[str], None]) -> str:
    return await _race_cancellation(provider.generate_text_stream,
                                    (system, user, list(images), on_delta), cancellation)


# --------------------------------------------------------------------------
# Event emission
# --------------------------------------------------------------------------
def emit_event(events: List[AgentEvent], event_sink: Optional[AgentEventSink], event: AgentEvent) -> None:
    if event_sink is not None:
        try:
            event_sink(AgentEvent(event))
        except Exception:  # noqa: BLE001 - the Rust sink is infallible; never kill the turn
            pass
    events.append(event)


def tool_emit_event(tool_events: List[AgentToolEvent], _events: List[AgentEvent],
                    _event_sink: Optional[AgentEventSink], tool_event: AgentToolEvent) -> None:
    tool_events.append(tool_event)


def mode_label(mode: AgentMode) -> str:
    return AgentMode.parse(mode).value


def validate_images(images: Sequence[AgentImage]) -> None:
    if len(images) > MAX_IMAGES_PER_TURN:
        raise ValueError(f"At most {MAX_IMAGES_PER_TURN} images can be attached to one Agent turn")
    for image in images:
        media_type = image.media_type.strip()
        if media_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            raise ValueError(f"Unsupported image media type: {media_type}")
        if _utf8_len(image.data_base64) > MAX_IMAGE_BASE64_BYTES:
            raise ValueError("Attached image is too large")
        if not image.data_base64.strip():
            raise ValueError("Attached image is empty")


# --------------------------------------------------------------------------
# Context fitting
# --------------------------------------------------------------------------
def fit_context_to_model(ctx: Any, config: Any) -> Any:
    max_context_size = _llm_max_context_size(config)
    if max_context_size is None:
        return ctx
    max_chars = min(max(max_context_size, 8_000), 400_000)
    system, user = ctx.system, ctx.user
    total_chars = len(system) + len(user)
    if total_chars <= max_chars:
        return ctx
    minimum_user_budget = 4_000
    system_budget = max(0, max_chars - minimum_user_budget)
    if len(system) > system_budget:
        system = _context.trim_chars(system, system_budget)
    user_budget = max(max(0, max_chars - len(system)), minimum_user_budget)
    user = _context.trim_chars(user, user_budget)
    try:
        return dataclasses.replace(ctx, system=system, user=user)
    except TypeError:
        ctx.system, ctx.user = system, user
        return ctx


def agent_structured_max_tokens(has_skills: bool) -> int:
    return AGENT_SKILL_STRUCTURED_MAX_TOKENS if has_skills else AGENT_STRUCTURED_MAX_TOKENS


def project_context_for_retrieval_mode(project: Any, retrieval_mode: AgentRetrievalMode) -> Any:
    if retrieval_mode == AgentRetrievalMode.FAITHFUL:
        # Faithful-source mode must enforce its evidence boundary before prompt
        # construction. overview.md is generated knowledge, and schema.md can
        # contain domain descriptions; asking the model to ignore either would
        # be a soft guarantee rather than source-only context isolation.
        project = dataclasses.replace(project, overview=None, schema=None)
    return project


# --------------------------------------------------------------------------
# Budgets / retrieval bookkeeping
# --------------------------------------------------------------------------
def should_plan_tools_with_model(message: str, mode: AgentMode, tools: AgentToolOptions,
                                 skills_enabled: bool) -> bool:
    if mode == AgentMode.FAST:
        return False
    has_available_tool = tools.wiki or tools.web or tools.anytxt or skills_enabled
    return bool(message.strip()) and has_available_tool


def agent_loop_iteration_budget(mode: AgentMode, has_skills: bool) -> int:
    base = {AgentMode.FAST: 4, AgentMode.STANDARD: MAX_AGENT_TOOL_ITERATIONS,
            AgentMode.LOCAL_FIRST: MAX_AGENT_TOOL_ITERATIONS, AgentMode.DEEP: 12}[mode]
    if not has_skills:
        return base
    return {AgentMode.FAST: 8, AgentMode.STANDARD: 16, AgentMode.LOCAL_FIRST: 16, AgentMode.DEEP: 20}[mode]


def agent_loop_retrieval_budget(mode: AgentMode, retrieval_mode: AgentRetrievalMode,
                                has_explicit_skills: bool) -> int:
    if retrieval_mode == AgentRetrievalMode.FAITHFUL:
        return {AgentMode.FAST: 2, AgentMode.STANDARD: 3, AgentMode.LOCAL_FIRST: 3, AgentMode.DEEP: 5}[mode]
    if retrieval_mode == AgentRetrievalMode.SMART:
        return {AgentMode.FAST: 3, AgentMode.STANDARD: 4, AgentMode.LOCAL_FIRST: 4, AgentMode.DEEP: 6}[mode]
    base = {AgentMode.FAST: 2, AgentMode.STANDARD: 4, AgentMode.LOCAL_FIRST: 4, AgentMode.DEEP: 8}[mode]
    return base + 4 if has_explicit_skills else base


_RETRIEVAL_TOOLS = ("wiki.search", "wiki.read_page", "source.search", "graph.search", "web.search", "anytxt.search")


def is_agent_retrieval_tool(tool: str) -> bool:
    return tool in _RETRIEVAL_TOOLS


def canonical_json(value: Any) -> str:
    return _value_to_string(value)


_ASCII_PUNCT = set(string.punctuation)


def retrieval_signature(tool: str, input: Dict[str, Any], mode: AgentRetrievalMode) -> str:
    if mode == AgentRetrievalMode.STANDARD:
        return f"{tool}:{canonical_json(input)}"
    normalized = dict(input)
    query = normalized.get("query")
    if isinstance(query, str):
        parts: List[str] = []
        cur: List[str] = []
        for ch in query.lower():
            if ch.isspace() or ch in _ASCII_PUNCT:
                if cur:
                    parts.append("".join(cur))
                    cur = []
            else:
                cur.append(ch)
        if cur:
            parts.append("".join(cur))
        normalized["query"] = " ".join(parts)
    return f"{tool}:{canonical_json(normalized)}"


def smart_evidence_count(references: Sequence[AgentReference]) -> int:
    return len({f"{r.kind}:{r.path}" for r in references})


def retrieval_added_evidence(tool: str, summary: str, reference_count_before: int,
                             reference_count_after: int) -> bool:
    if reference_count_after > reference_count_before:
        return True
    if tool == "wiki.read_page":
        head, sep, content = summary.partition("\n")
        return bool(sep) and bool(content.strip())
    return False


def should_fallback_wiki_search(planner_unavailable_or_failed: bool, tools: AgentToolOptions,
                                skills_empty: bool) -> bool:
    return planner_unavailable_or_failed and tools.wiki and skills_empty


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
def build_agent_loop_system(base_system: str, retrieval_mode: AgentRetrievalMode) -> str:
    smart_retrieval = (
        "\nSmart retrieval is enabled. Treat retrieval as a bounded evidence loop: after every observation, "
        "identify only the unresolved evidence gap, then either issue one concise revised retrieval action or "
        "answer. Prefer reading/following already discovered pages before broadening the query. Do not repeat "
        "equivalent queries. Stop as soon as the available evidence supports a cited answer; optional background "
        "is not a reason to continue."
        if retrieval_mode == AgentRetrievalMode.SMART else ""
    )
    faithful_retrieval = (
        "\nFaithful-source mode is enabled by explicit user choice. Answer only from raw source excerpts returned "
        "by source.search or explicitly attached source files. Preserve quoted wording exactly and cite the source "
        "path beside every quotation or factual claim. Clearly distinguish verbatim quotations from explanation. "
        "Generated wiki pages, graph context, web results, AnyTXT results, and unsupported background knowledge "
        "are not evidence in this mode. If the available excerpts are insufficient, state that limitation instead "
        "of reconstructing or guessing."
        if retrieval_mode == AgentRetrievalMode.FAITHFUL else ""
    )
    retrieval_example = (
        '1. {"action":"tool","tool":"source.search","query":"..."}'
        if retrieval_mode == AgentRetrievalMode.FAITHFUL
        else '1. {"action":"tool","tool":"wiki.search","query":"..."}'
    )
    lines = [
        f"{base_system}\n\nAgent loop protocol:",
        "Return only compact JSON. Do not wrap it in markdown.",
        "Choose exactly one action per turn:",
        retrieval_example,
        '2. {"action":"tool","tool":"user.ask","title":"...","description":"...","fields":[{"id":"choice","type":"single","label":"...","options":[{"label":"...","value":"...","recommended":true}]}]}',
        '3. {"action":"tool","tool":"workspace.write_file","path":"cover-image/cover.svg","content":"..."}',
        '3b. {"action":"tool","tool":"workspace.append_file","path":"deck/index.html","content":"..."}',
        '4. {"action":"final","answer":"..."}',
        "Use tools when they are useful, then wait for the observation in the next turn before deciding the next step.",
        'Do not return natural-language plans such as "I need to read..." or "Let me search...". If you intend to read/search/write/run something, return the matching tool JSON action in this turn.',
        "Do not merely announce that you will use a skill or tool. If a skill references a file under its own directory, call skill.read_file with a relative path. If a skill requires an actual command-line generation step, call shell.exec.",
        "Use skill.read_file for skill Markdown/reference files. Do not use shell.exec, test, cat, or ls merely to inspect skill files or optional preference files.",
        "Use user.ask when a skill or task needs structured user choices, text input, confirmations, or multiple fields before it can continue. Do not simulate this by writing plain-text questions when a structured form is appropriate.",
        "Use workspace.write_file for generated artifacts such as SVG, HTML, Markdown drafts, JSON, or scripts. For large HTML/PPT files, first call workspace.write_file with an empty or small opening chunk, then call workspace.append_file with subsequent chunks, and finish only after the file contains closing tags such as </html>. Do not inline large heredocs or generated file bodies inside shell.exec JSON.",
        "Do not claim that a generated file exists until a workspace.write_file or shell.exec observation confirms it. In the final answer, mention only observed generated file paths.",
        "Converge quickly. Do not keep reading optional references, running optional validation, or polishing after the requested deliverable has been written. Prefer final as soon as the core user request is satisfied.",
        "Only use shell.exec when active skill instructions or the user's explicit request require command-line work after files have been written with workspace.write_file. Generated files must be written under the Agent workspace described above. Commands whose explicit file paths stay inside the Agent workspace can run without an approval prompt; commands that mention external paths, home directories, downloads, temp folders, or network URLs require approval.",
        "Use wiki.write_page only when the user explicitly asks to create or update a wiki page. Existing pages are create-only unless allowOverwrite is explicitly justified by the user's request."
        + smart_retrieval + faithful_retrieval,
    ]
    return "\n".join(lines)


def build_agent_final_system(base_system: str, retrieval_mode: AgentRetrievalMode) -> str:
    faithful_retrieval = (
        " Faithful-source mode remains in force: use only raw source excerpts or explicitly attached source files "
        "as evidence, cite their paths, preserve quotations exactly, and disclose insufficient evidence rather "
        "than guessing."
        if retrieval_mode == AgentRetrievalMode.FAITHFUL else ""
    )
    return (
        f"{base_system}\n\nThe retrieval phase is complete. No tools are available now. Answer the user's latest "
        f"request directly using the permitted context and tool observations already provided.{faithful_retrieval} "
        'Return only compact JSON in the form {"action":"final","answer":"..."}. Do not request, announce, or '
        "simulate another search or file read."
    )


def build_agent_final_user(base_user: str, observations: Sequence[AgentObservation]) -> str:
    out = base_user
    if observations:
        out += "\n\nCompleted tool observations:\n"
        out += render_observations(observations)
    out += "\n\nProvide the final answer now. No further tool action is permitted."
    return out


def looks_like_agent_tool_json(value: str) -> bool:
    trimmed = value.lstrip()
    return trimmed.startswith("{") and ('"action"' in trimmed or '"tool"' in trimmed or '"command"' in trimmed)


def forced_final_answer(raw: str, action: AgentLoopAction, references: Sequence[AgentReference]) -> str:
    if _eq_ic(action.action, "final"):
        answer = _opt_trim(action.answer)
        if answer:
            return answer
    trimmed = raw.strip()
    if trimmed and not looks_like_agent_tool_json(trimmed):
        return trimmed
    if not references:
        return "I could not find enough project context to answer this request reliably."
    answer = "I found the following relevant project context:\n"
    for reference in list(references)[:8]:
        answer += "- " + reference.title + " (" + reference.path + ")"
        if reference.snippet is not None:
            answer += ": " + _context.trim_chars(_context.collapse_whitespace(reference.snippet), 320)
        answer += "\n"
    return answer


def build_agent_loop_user(base_user: str, request: AgentChatRequest, skills: Sequence[AgentSkill],
                          observations: Sequence[AgentObservation], iteration: int, max_iterations: int,
                          has_generated_workspace_file: bool) -> str:
    out = [base_user, "\n\nAvailable Agent tools for this turn:\n"]
    faithful = request.retrieval_mode == AgentRetrievalMode.FAITHFUL
    if request.tools.wiki:
        if faithful:
            out.append("- source.search: search raw source excerpts. This is the only retrieval tool permitted in faithful-source mode.\n")
        else:
            out.append("- wiki.search: retrieve wiki pages for factual or topical questions.\n")
            out.append("- wiki.read_page: read a specific wiki markdown page by path.\n")
            out.append("- source.search: search raw source snippets.\n")
            out.append("- graph.search: retrieve relationships, neighbors, backlinks, dependencies, and connections between entities. Prefer it for relational questions and query with concise entity or concept names.\n")
            out.append("- wiki.write_page: create a wiki markdown page when explicitly requested.\n")
    if request.tools.web and not faithful:
        out.append("- web.search: search external web sources.\n")
    if request.tools.anytxt and not faithful:
        out.append("- anytxt.search: search files indexed by AnyTXT.\n")
    if skills:
        out.append("- skill.read_file: read a Markdown/reference file from an active skill directory by relative path. Prefer this over shell.exec for skill references.\n")
        out.append("- user.ask: pause and show the user a structured form with single-choice, multi-choice, text, textarea, or confirmation fields when an active skill needs user input.\n")
        out.append("- workspace.write_file: write generated artifacts under agent-workspace by relative path. For large HTML/PPT, initialize the file and then append chunks.\n")
        out.append("- workspace.append_file: append content to a generated artifact under agent-workspace. Prefer this for large HTML/PPT after workspace.write_file.\n")
        out.append("- shell.exec: run a command from the Agent workspace when a selected/available skill requires command-line work. Workspace-local commands can run directly; external paths or network access require approval.\n")
    if not observations:
        out.append("\nTool observations so far: none.\n")
    else:
        out.append("\nTool observations so far:\n")
        out.append(render_observations(observations))
        out.append("\n")
    step_number = iteration + 1
    remaining_after_this = max(0, max_iterations - step_number)
    out.append(f"\nIteration budget: step {step_number} of {max_iterations}. "
               f"You have {remaining_after_this} step(s) after this response.\n")
    if has_generated_workspace_file:
        out.append("A generated workspace file has already been observed. If it satisfies the core request, return final now with the file path. Do not spend remaining steps on optional validation or extra reference reads unless the user explicitly required them.\n")
    if remaining_after_this <= 2:
        out.append("Budget is nearly exhausted. Return final now if any useful answer or generated file exists. Use at most one more tool only when it is strictly required to complete or close an already-started file; skip optional checks, optional reads, and style polishing.\n")
    out.append('\nReturn the next JSON action now. Prefer {"action":"final","answer":"..."} whenever the core request is already satisfied.')
    return "".join(out)


def render_observations(observations: Sequence[AgentObservation]) -> str:
    if not observations:
        return ""
    return "\n\n".join(
        f"{idx + 1}. {o.tool}:\n{_context.trim_chars(o.summary, 8_000)}" for idx, o in enumerate(observations)
    )


# --------------------------------------------------------------------------
# Loop action parsing
# --------------------------------------------------------------------------
def extract_json_object(raw: str) -> Optional[str]:
    start = raw.find("{")
    if start < 0:
        return None
    end = raw.rfind("}")
    if end < 0 or end <= start:
        return None
    return raw[start:end + 1]


_LOOP_TOOL_NAMES = (
    "wiki.search", "wiki.read_page", "wiki.write_page", "source.search", "graph.search", "web.search",
    "anytxt.search", "skill.read_file", "workspace.write_file", "workspace.append_file", "shell.exec",
    "deep_research.run",
)


def is_user_ask_tool(tool: str) -> bool:
    return tool.strip() in ("user.ask", "user_input.ask", "askUserQuestion", "AskUserQuestion", "ask_user_question")


def is_agent_loop_tool_name(value: str) -> bool:
    return value in _LOOP_TOOL_NAMES or is_user_ask_tool(value)


def normalize_agent_loop_action(action: AgentLoopAction) -> AgentLoopAction:
    trimmed_action = action.action.strip()
    if action.tool is None and is_agent_loop_tool_name(trimmed_action):
        action.tool = trimmed_action
        action.action = "tool"
    if action.tool is not None and is_agent_loop_tool_name(action.tool):
        action.action = "tool"
    if not action.action.strip():
        action.action = "tool" if action.tool is not None else "final"
    if action.tool is not None and is_user_ask_tool(action.tool):
        action.tool = "user.ask"
    return action


def parse_agent_loop_action(raw: str) -> AgentLoopAction:
    trimmed = raw.strip()
    try:
        return normalize_agent_loop_action(AgentLoopAction.from_str(trimmed))
    except _SerdeError:
        pass
    js = extract_json_object(trimmed)
    if js is not None:
        try:
            return normalize_agent_loop_action(AgentLoopAction.from_str(js))
        except _SerdeError:
            pass
    if looks_like_agent_tool_json(trimmed):
        return AgentLoopAction(
            action="invalid_tool_json",
            answer="Invalid or truncated Agent tool JSON. Return one complete compact JSON object. For large "
                   "generated files, initialize with workspace.write_file and continue with "
                   "workspace.append_file chunks instead of putting the whole file or heredocs in one JSON object.",
        )
    return AgentLoopAction(
        action="invalid_tool_json",
        answer='Agent loop responses must be compact JSON. Return either a tool action like '
               '{"action":"tool","tool":"wiki.read_page","path":"..."} or a final action like '
               '{"action":"final","answer":"..."}. Do not return plain text.',
    )


def parse_model_tool_plan(raw: str) -> ModelToolPlan:
    js = extract_json_object(raw)
    json_text = (js if js is not None else raw).strip()
    try:
        return ModelToolPlan.from_value(_loads_serde(json_text))
    except _SerdeError as e:
        raise ValueError(f"Invalid Agent tool plan JSON: {e}")


def planned_tool_queries(plan: ModelToolPlan, fallback_query: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for call in plan.tool_calls:
        tool = call.tool.strip()
        if tool not in ("wiki.search", "source.search", "graph.search", "web.search", "anytxt.search",
                        "wiki.write_page"):
            continue
        query = _non_empty(_opt_trim(call.query)) or fallback_query
        out.setdefault(tool, query)
    return out


# --------------------------------------------------------------------------
# user.ask sanitising
# --------------------------------------------------------------------------
_MISSING = object()


def _first_present(obj: dict, *keys: str) -> Any:
    for k in keys:
        if k in obj:
            return obj[k]
    return _MISSING


def _as_str_or_none(v: Any) -> Optional[str]:
    return v if isinstance(v, str) else None


def clean_user_input_text(value: str) -> Optional[str]:
    cleaned = "".join(
        ch for ch in value if ch not in "<>" and unicodedata.category(ch) != "Cc"
    )[:MAX_USER_INPUT_TEXT_CHARS].strip()
    return cleaned or None


def clean_user_input_id(value: str) -> Optional[str]:
    cleaned = "".join(ch for ch in value if (ch.isascii() and ch.isalnum()) or ch in "_-")[:64]
    return cleaned or None


def normalize_user_input_field_type(value: str) -> str:
    v = value.strip()
    if v in ("single", "singleChoice", "radio", "select"):
        return "single"
    if v in ("multi", "multiChoice", "checkbox", "checkboxes"):
        return "multi"
    if v in ("text", "input"):
        return "text"
    if v in ("textarea", "longText"):
        return "textarea"
    if v in ("confirm", "boolean", "switch"):
        return "confirm"
    raise ValueError(f"Unsupported user.ask field type: {v}")


def unique_user_input_key(used: Set[str], base: str, idx: int) -> str:
    if base not in used:
        used.add(base)
        return base
    for suffix in range(2, MAX_USER_INPUT_FIELDS + MAX_USER_INPUT_OPTIONS + 2 + 1):
        candidate = f"{base}_{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    fallback = f"field_{idx + 1}"
    if fallback not in used:
        used.add(fallback)
        return fallback
    return f"field_{idx + 1}_{len(used) + 1}"


def validate_user_input_default(field_type: str, value: Any, options: Sequence[AgentUserInputOption]) -> bool:
    values = {o.value for o in options}
    if field_type == "single":
        return isinstance(value, str) and value in values
    if field_type == "multi":
        return isinstance(value, list) and all(isinstance(i, str) and i in values for i in value)
    if field_type in ("text", "textarea"):
        return isinstance(value, str)
    if field_type == "confirm":
        return isinstance(value, bool)
    return False


def sanitize_user_input_option(value: Any) -> Optional[AgentUserInputOption]:
    if not isinstance(value, dict):
        return None
    lv = _first_present(value, "label", "title")
    label = clean_user_input_text(lv) if isinstance(lv, str) else None
    if not label:
        return None
    vv = value.get("value")
    value_text = (clean_user_input_text(vv) if isinstance(vv, str) else None) or label
    dv = value.get("description")
    description = clean_user_input_text(dv) if isinstance(dv, str) else None
    rec = value.get("recommended")
    recommended = rec if isinstance(rec, bool) else None
    return AgentUserInputOption(label=label, value=value_text, description=description, recommended=recommended)


def sanitize_user_input_field(value: Any, idx: int) -> Optional[AgentUserInputField]:
    if not isinstance(value, dict):
        return None
    t = _first_present(value, "type", "kind")
    raw_type = t if isinstance(t, str) else "single"
    try:
        field_type = normalize_user_input_field_type(raw_type)
    except ValueError:
        return None
    iv = _first_present(value, "id", "name")
    field_id = (clean_user_input_id(iv) if isinstance(iv, str) else None) or f"field_{idx + 1}"
    lv = _first_present(value, "label", "question", "header")
    label = (clean_user_input_text(lv) if isinstance(lv, str) else None) or f"Question {idx + 1}"
    dv = value.get("description")
    description = clean_user_input_text(dv) if isinstance(dv, str) else None
    pv = value.get("placeholder")
    placeholder = clean_user_input_text(pv) if isinstance(pv, str) else None
    used_option_values: Set[str] = set()
    options: List[AgentUserInputOption] = []
    raw_options = value.get("options")
    if isinstance(raw_options, list):
        for option_idx, item in enumerate(raw_options[:MAX_USER_INPUT_OPTIONS]):
            option = sanitize_user_input_option(item)
            if option is None:
                continue
            option.value = unique_user_input_key(used_option_values, option.value, option_idx)
            options.append(option)
    if field_type in ("single", "multi") and not options:
        return None
    dflt = _first_present(value, "defaultValue", "default")
    default_value = None
    if dflt is not _MISSING and validate_user_input_default(field_type, dflt, options):
        default_value = dflt
    return AgentUserInputField(id=field_id, field_type=field_type, label=label, description=description,
                               placeholder=placeholder, options=options, default_value=default_value)


def sanitize_user_input_request(action: AgentLoopAction) -> AgentUserInputRequest:
    raw_fields = action.fields if action.fields is not None else action.questions
    if raw_fields is None:
        raise ValueError("user.ask requires fields or questions")
    if not isinstance(raw_fields, list):
        raise ValueError("user.ask fields must be an array")
    fields: List[AgentUserInputField] = []
    used_field_ids: Set[str] = set()
    for idx, value in enumerate(raw_fields[:MAX_USER_INPUT_FIELDS]):
        f = sanitize_user_input_field(value, idx)
        if f is not None:
            f.id = unique_user_input_key(used_field_ids, f.id, idx)
            fields.append(f)
    if not fields:
        raise ValueError("user.ask requires at least one valid field")
    title = clean_user_input_text(action.title if action.title is not None else "Input required") \
        or "Input required"
    description = clean_user_input_text(
        action.description if action.description is not None
        else "Please provide the requested information so the Agent can continue.")
    return AgentUserInputRequest(request_id=str(uuid.uuid4()), title=title, description=description,
                                 fields=fields)


# --------------------------------------------------------------------------
# Loop bookkeeping helpers
# --------------------------------------------------------------------------
def summarize_tool_input(tool: str, input: Dict[str, Any]) -> Optional[str]:
    def s(key: str) -> Optional[str]:
        v = input.get(key)
        return v if isinstance(v, str) else None

    if tool in ("wiki.search", "source.search", "graph.search", "web.search", "anytxt.search"):
        return s("query")
    if tool in ("wiki.read_page", "wiki.write_page", "workspace.write_file", "workspace.append_file",
                "skill.read_file"):
        return s("path")
    if tool == "shell.exec":
        return s("command")
    return None


def record_loop_tool_rejection(tool: str, error: str, tool_events: List[AgentToolEvent],
                               events: List[AgentEvent], event_sink: Optional[AgentEventSink]) -> AgentObservation:
    tool_emit_event(tool_events, events, event_sink, AgentToolEvent(tool, "failed", error))
    emit_event(events, event_sink, AgentEvent.tool_end(tool, f"rejected: {error}"))
    return AgentObservation(tool, f"rejected: {error}")


def push_unique_reference(references: List[AgentReference], events: List[AgentEvent],
                          event_sink: Optional[AgentEventSink], reference: AgentReference) -> bool:
    if any(e.kind == reference.kind and e.path == reference.path for e in references):
        return False
    emit_event(events, event_sink, AgentEvent.reference_added(reference))
    references.append(reference)
    return True


def agent_iteration_limit_answer(max_iterations: int, observation_count: int,
                                 references: Sequence[AgentReference]) -> str:
    seen: Set[str] = set()
    workspace_paths: List[str] = []
    for reference in references:
        if reference.kind != "workspace":
            continue
        if not reference.path.strip():
            continue
        if reference.path not in seen:
            seen.add(reference.path)
            workspace_paths.append(reference.path)
    if not workspace_paths:
        return (f"The Agent reached the tool-iteration limit after {max_iterations} step(s). It gathered "
                f"{observation_count} tool observation(s), but did not produce a final answer. Please narrow the "
                "request or ask it to continue from the latest result.")
    # Skills can successfully write deliverables and then spend the remaining
    # budget on optional validation or reference reads. Report confirmed files
    # instead of turning a completed write into a false failure.
    answer = (f"The Agent reached the tool-iteration budget after {max_iterations} step(s), but it did generate "
              "file(s).\n\nGenerated files:\n")
    for path in workspace_paths:
        answer += "- " + path + "\n"
    answer += ("\nOpen the generated file reference(s) to preview them. Some optional validation or follow-up "
               "steps may not have completed.")
    return answer


def require_tool_permission(tool: str, request: AgentChatRequest, permission_policy: PermissionPolicy) -> None:
    C = AgentCapability
    if request.retrieval_mode == AgentRetrievalMode.FAITHFUL and tool in (
            "wiki.search", "wiki.read_page", "graph.search", "web.search", "anytxt.search"):
        raise PermissionError(f"{tool} is unavailable in faithful-source mode; use source.search")
    if tool == "wiki.search":
        if not request.tools.wiki:
            raise PermissionError("wiki.search is disabled for this turn")
        return permission_policy.require(C.SEARCH_WIKI)
    if tool in ("wiki.read_page", "graph.search"):
        if not request.tools.wiki:
            raise PermissionError(f"{tool} is disabled for this turn")
        return permission_policy.require(C.READ_PROJECT)
    if tool == "source.search":
        if not request.tools.wiki:
            raise PermissionError("source.search is disabled for this turn")
        return permission_policy.require(C.READ_SOURCE)
    if tool == "wiki.write_page":
        if not request.tools.wiki:
            raise PermissionError("wiki.write_page is disabled for this turn")
        return permission_policy.require(C.WRITE_WIKI)
    if tool in ("workspace.write_file", "workspace.append_file"):
        return permission_policy.require(C.WRITE_WIKI)
    if tool == "web.search":
        if not request.tools.web:
            raise PermissionError("web.search is disabled for this turn")
        return permission_policy.require(C.NETWORK)
    if tool == "anytxt.search":
        if not request.tools.anytxt:
            raise PermissionError("anytxt.search is disabled for this turn")
        return permission_policy.require(C.NETWORK)
    if tool == "deep_research.run":
        raise PermissionError(
            "deep_research.run is not available in the loop executor; use web.search, anytxt.search, "
            "source.search, and wiki.search directly")
    if tool == "skill.read_file":
        return permission_policy.require(C.READ_PROJECT)
    if tool == "shell.exec":
        return permission_policy.require(C.PROCESS)
    raise PermissionError(f"Unknown Agent tool: {tool}")


def escape_planner_xml(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def render_skill_planner_context(skills: Sequence[AgentSkill], skill_mode: AgentSkillMode) -> str:
    if not skills:
        return "None"
    out = ""
    if skill_mode == AgentSkillMode.AUTO:
        out += ("The following enabled skills are optional. Use them only if they match the request. Inspect a "
                "SKILL.md location only after deciding the skill is relevant.\n")
        out += "<available_skills>\n"
        for skill in list(skills)[:12]:
            out += "  <skill>\n"
            out += f"    <name>{escape_planner_xml(skill.name)}</name>\n"
            out += f"    <description>{escape_planner_xml(skill.description.strip())}</description>\n"
            out += f"    <location>{escape_planner_xml(skill.location)}</location>\n"
            out += "  </skill>\n"
        out += "</available_skills>\n"
    else:
        out += "The following skills were explicitly selected by the user for this turn.\n"
        for skill in list(skills)[:8]:
            base_dir = escape_planner_xml(skill.base_dir)
            instructions = escape_planner_xml(_context.trim_chars(skill.instructions.strip(), 1_200))
            out += (f'<skill name="{escape_planner_xml(skill.name)}" location="{escape_planner_xml(skill.location)}">\n'
                    f"References are relative to {base_dir}.\n\n{instructions}\n</skill>\n")
    return _context.trim_chars(out, 8_000)


# --------------------------------------------------------------------------
# shell.exec gating
# --------------------------------------------------------------------------
def shell_command_from_call(call: ModelToolCall) -> Optional[str]:
    for candidate in (call.command, call.query, call.content):
        if candidate is not None:
            c = candidate.strip()
            return c or None
    return None


def is_shell_command_approved(command: str, approved: Sequence[str]) -> bool:
    command = command.strip()
    return bool(command) and any(item.strip() == command for item in approved)


def is_shell_command_allowed_without_prompt(command: str, approved: Sequence[str], project_path: str) -> bool:
    return (is_shell_command_approved(command, approved)
            or is_shell_command_scoped_to_agent_workspace(command, project_path))


def is_skill_preference_probe_command(command: str) -> bool:
    lower = _ascii_lower(command)
    return (("extend.md" in lower or "baoyu-skills" in lower)
            and ("test -f" in lower or "test-path" in lower))


def skipped_skill_preference_probe_summary(command: str) -> str:
    return (f"Skipped optional skill preference probe instead of running shell command:\n`{command}`\n"
            "No EXTEND.md preferences were loaded. Continue with the selected skill using its defaults, and do "
            "not retry this probe.")


def normalize_shell_path_for_compare(value: str) -> str:
    return _ascii_lower(value.strip("\"'").replace("\\", "/").rstrip("/"))


def is_shell_absolute_path(value: str) -> bool:
    return value.startswith("/") or value.startswith("\\\\") or value.encode("utf-8")[1:2] == b":"


def shell_command_tokens(command: str) -> List[str]:
    tokens: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    for ch in command:
        if quote is not None:
            if ch == quote:
                quote = None
            else:
                current.append(ch)
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch.isspace() or ch in ";|&()<>":
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


def shell_token_mentions_external_location(token: str, workspace_norm: str, project_norm: str) -> bool:
    token = token.strip("\"',;")
    if not token:
        return False
    lower = _ascii_lower(token)
    if (lower.startswith("~") or "$home" in lower or "${home" in lower or "%userprofile%" in lower
            or "%homepath%" in lower or "$xdg_" in lower or "${xdg_" in lower or "$tmp" in lower
            or "${tmp" in lower or "$temp" in lower or "${temp" in lower):
        return True
    if token == ".." or token.startswith("../") or "/../" in token or token.endswith("/.."):
        return True
    candidates = [token]
    if "=" in token:
        candidates.append(token.split("=", 1)[1])
    for candidate in candidates:
        candidate = candidate.strip("\"'")
        if not candidate:
            continue
        if is_shell_absolute_path(candidate):
            normalized = normalize_shell_path_for_compare(candidate)
            in_workspace = normalized == workspace_norm or normalized.startswith(f"{workspace_norm}/")
            project_workspace_prefix = f"{project_norm}/{AGENT_WORKSPACE_DIR}"
            in_project_workspace = (normalized == project_workspace_prefix
                                    or normalized.startswith(f"{project_workspace_prefix}/"))
            if not in_workspace and not in_project_workspace:
                return True
    return False


def is_shell_command_scoped_to_agent_workspace(command: str, project_path: str) -> bool:
    command = command.strip()
    if not command:
        return False
    lower = _ascii_lower(command)
    if ("http://" in lower or "https://" in lower or "ftp://" in lower or "sftp://" in lower
            or "curl " in lower or "wget " in lower or "scp " in lower or "ssh " in lower
            or "$(" in lower or "`" in command):
        return False
    workspace = agent_workspace_display(project_path)
    workspace_norm = normalize_shell_path_for_compare(workspace)
    project_norm = normalize_shell_path_for_compare(project_path)
    for token in shell_command_tokens(command):
        if not token:
            continue
        if shell_token_mentions_external_location(token, workspace_norm, project_norm):
            return False
    return True


# --------------------------------------------------------------------------
# skill.read_file
# --------------------------------------------------------------------------
def is_safe_relative_skill_path(path: str) -> bool:
    if path == "":
        return False
    p = Path(path)
    if p.is_absolute() or p.drive or p.root:
        return False
    return all(part != ".." for part in p.parts)


def skill_name_like(value: str) -> bool:
    value = value.strip()
    return bool(value) and "-" in value and all(
        (ch.isascii() and ch.isalnum()) or ch in "-_" for ch in value)


def skill_matches_requested_name(skill: AgentSkill, requested: str) -> bool:
    requested_lower = _ascii_lower(requested)
    if _ascii_lower(skill.name) == requested_lower:
        return True

    def folder_matches(path: str) -> bool:
        name = _file_name(path)
        if name is None:
            return False
        name_lower = _ascii_lower(name)
        return name_lower == requested_lower or name_lower.endswith(f"-{requested_lower}")

    if folder_matches(skill.base_dir):
        return True
    parent = str(Path(skill.location).parent)
    return folder_matches(parent)


def split_skill_path_prefix(path: str) -> Optional[Tuple[str, str]]:
    if ":" in path:
        prefix, rest = path.split(":", 1)
        if skill_name_like(prefix):
            return prefix, rest.lstrip("/")
    if "/" not in path:
        return None
    prefix, rest = path.split("/", 1)
    return prefix, rest


def _canonicalize(p: Path, what: str) -> Path:
    try:
        return p.resolve(strict=True)
    except OSError as e:
        raise ValueError(f"Failed to resolve {what}: {_os_err(e)}")


def _rel_posix(target: Path, base: Path) -> str:
    if target == base:
        return ""
    return target.relative_to(base).as_posix()


def select_active_skill_for_read(skills: Sequence[AgentSkill], requested: Optional[str]) -> AgentSkill:
    requested = _non_empty(_opt_trim(requested))
    if requested is None:
        if len(skills) == 1:
            return skills[0]
        raise ValueError("skill.read_file requires skill when multiple skills are active")
    for skill in skills:
        if skill_matches_requested_name(skill, requested):
            return skill
    raise ValueError(f"Active skill not found: {requested}")


def normalize_requested_path_for_skill(skill: AgentSkill, requested_path: str) -> str:
    path = Path(requested_path)
    if path.is_absolute():
        target = _canonicalize(path, "skill file")
        base = _canonicalize(Path(skill.base_dir), "skill directory")
        if not _path_starts_with(target, base):
            raise ValueError("skill.read_file absolute path does not belong to requested skill")
        return _rel_posix(target, base)
    norm = requested_path.strip().replace("\\", "/")
    if "/" in norm:
        prefix, rest = norm.split("/", 1)
        if skill_matches_requested_name(skill, prefix):
            return rest
        if skill_name_like(prefix):
            raise ValueError("skill.read_file path prefix does not match requested skill")
    if ":" in norm:
        prefix, rest = norm.split(":", 1)
        if skill_matches_requested_name(skill, prefix):
            return rest.lstrip("/")
        if skill_name_like(prefix):
            raise ValueError("skill.read_file path prefix does not match requested skill")
    return requested_path


def resolve_absolute_skill_path(skills: Sequence[AgentSkill], requested_path: str
                                ) -> Optional[Tuple[AgentSkill, str]]:
    path = Path(requested_path)
    if not path.is_absolute():
        return None
    target = _canonicalize(path, "skill file")
    for skill in skills:
        base = _canonicalize(Path(skill.base_dir), "skill directory")
        if _path_starts_with(target, base):
            rel = _rel_posix(target, base)
            if rel:
                return skill, rel
    return None


def resolve_prefixed_skill_path(skills: Sequence[AgentSkill], requested_path: str
                                ) -> Optional[Tuple[AgentSkill, str]]:
    normalized = requested_path.strip().replace("\\", "/")
    split = split_skill_path_prefix(normalized)
    if split is None:
        return None
    prefix, rest = split
    if not rest.strip():
        return None
    matched = [s for s in skills
               if skill_matches_requested_name(s, prefix) and (Path(s.base_dir) / rest).exists()]
    if len(matched) == 1:
        return matched[0], rest
    if not matched:
        return None
    raise ValueError(f"skill.read_file prefix is ambiguous: {prefix}")


def resolve_unique_existing_skill_path(skills: Sequence[AgentSkill], requested_path: str) -> Optional[AgentSkill]:
    if not is_safe_relative_skill_path(requested_path):
        return None
    matches: List[AgentSkill] = []
    for skill in skills:
        base = Path(skill.base_dir)
        candidate = base / requested_path
        if candidate.exists():
            base_canon = _canonicalize(base, "skill directory")
            candidate_canon = _canonicalize(candidate, "skill file")
            if _path_starts_with(candidate_canon, base_canon):
                matches.append(skill)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    raise ValueError(f"skill.read_file path is ambiguous: {requested_path}")


def resolve_skill_read_target(skills: Sequence[AgentSkill], requested_skill: Optional[str],
                              requested_path: str) -> Tuple[AgentSkill, str]:
    requested = _non_empty(_opt_trim(requested_skill))
    if requested is not None:
        skill = select_active_skill_for_read(skills, requested)
        rel = normalize_requested_path_for_skill(skill, requested_path)
        return skill, rel
    found = resolve_absolute_skill_path(skills, requested_path)
    if found is not None:
        return found
    skill = resolve_unique_existing_skill_path(skills, requested_path)
    if skill is not None:
        return skill, requested_path
    found = resolve_prefixed_skill_path(skills, requested_path)
    if found is not None:
        return found
    skill = select_active_skill_for_read(skills, requested_skill)
    return skill, requested_path


def read_active_skill_file(skills: Sequence[AgentSkill], input: Dict[str, Any]) -> Dict[str, Any]:
    if not skills:
        raise ValueError("skill.read_file requires an active skill")
    p = input.get("path")
    requested_path = _non_empty(_opt_trim(p if isinstance(p, str) else None))
    if requested_path is None:
        raise ValueError("skill.read_file requires path")
    sk = input.get("skill")
    skill, relative_path = resolve_skill_read_target(skills, sk if isinstance(sk, str) else None, requested_path)
    if not is_safe_relative_skill_path(relative_path):
        raise ValueError("skill.read_file path must be a safe relative path inside the skill directory")
    base = Path(skill.base_dir)
    base_canon = _canonicalize(base, "skill directory")
    target = base / relative_path
    target_canon = _canonicalize(target, "skill file")
    if not _path_starts_with(target_canon, base_canon):
        raise ValueError("skill.read_file cannot read outside the active skill directory")
    import stat as _stat

    try:
        meta = os.lstat(target_canon)
    except OSError as e:
        raise ValueError(f"Failed to inspect skill file: {_os_err(e)}")
    if _stat.S_ISLNK(meta.st_mode) or not _stat.S_ISREG(meta.st_mode):
        raise ValueError("skill.read_file target is not a regular file")
    if meta.st_size > MAX_SKILL_REFERENCE_BYTES:
        raise ValueError(f"skill.read_file target is too large (max {MAX_SKILL_REFERENCE_BYTES} bytes)")
    try:
        content = target_canon.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to read skill file as UTF-8 text: {_os_err(e)}")
    return {"skill": skill.name, "path": relative_path, "content": content}


# --------------------------------------------------------------------------
# Retrieval-only answers
# --------------------------------------------------------------------------
def build_retrieval_answer(query: str, references: Sequence[AgentReference]) -> str:
    if not references:
        return (f'I searched the current LLM Wiki project for "{query}" but did not find matching wiki pages.')
    out = (f'I searched the current LLM Wiki project for "{query}" and found {len(references)} '
           "relevant page(s):")
    for idx, reference in enumerate(list(references)[:MAX_CHAT_SEARCH_RESULTS]):
        out += f"\n{idx + 1}. {reference.title} ({reference.path})"
        if reference.snippet is not None:
            out += f"\n   {_context.collapse_whitespace(reference.snippet)}"
    return out


def answer_without_backend_generator(query: str, references: Sequence[AgentReference],
                                     allow_empty_retrieval: bool) -> str:
    if not references:
        if allow_empty_retrieval:
            return ""
        raise ValueError("Backend Agent LLM is not configured or the selected Chat model is unavailable. "
                         "Check Settings > Models and try again.")
    # Preserve the retrieval-only API behavior, but never expose router
    # diagnostics as assistant prose when no generator exists.
    return build_retrieval_answer(query, references)


# --------------------------------------------------------------------------
# AgentRuntime
# --------------------------------------------------------------------------
class AgentRuntime:
    def __init__(self, project_id: str, project_path: str, embedding_config: Any = None,
                 llm_config: Any = None, web_search_config: Any = None, anytxt_config: Any = None):
        self.project_id = project_id
        self.project_path = project_path
        self.embedding_config = embedding_config
        self.llm_config = _coerce_llm_config(llm_config)
        self.web_search_config = web_search_config
        self.anytxt_config = anytxt_config

    # ---- public entry points --------------------------------------------
    async def run_once(self, request: AgentChatRequest) -> AgentChatResponse:
        return await self.run_once_with_cancel(request, None)

    async def run_once_with_cancel(self, request: AgentChatRequest,
                                   cancellation: Any = None) -> AgentChatResponse:
        return await self.run_once_with_cancel_and_events(request, cancellation, None)

    async def run_once_with_cancel_and_events(self, request: AgentChatRequest, cancellation: Any = None,
                                              event_sink: Optional[AgentEventSink] = None
                                              ) -> AgentChatResponse:
        cancellation = _as_cancellation(cancellation)
        message = request.message.strip()
        if not message:
            raise ValueError("message is required")
        validate_images(request.images)
        check_cancel(cancellation)

        session_id = (request.session_id if request.session_id and request.session_id.strip()
                      else f"api_{uuid.uuid4()}")
        tool_events: List[AgentToolEvent] = []
        events: List[AgentEvent] = []

        def te(tool: str, status: str, detail: Optional[str] = None) -> None:
            tool_emit_event(tool_events, events, event_sink, AgentToolEvent(tool, status, detail))

        def ev(event: AgentEvent) -> None:
            emit_event(events, event_sink, event)

        ev(AgentEvent.agent_start(session_id))
        ev(AgentEvent.turn_start(mode_label(request.mode)))
        references: List[AgentReference] = []
        permission_policy = PermissionPolicy.api_default()
        router = route_query(message, request.mode, request.tools)
        skills = load_project_skills(self.project_path, request.skills)
        check_cancel(cancellation)
        if request.skills:
            permission_policy.require(AgentCapability.READ_PROJECT)
            if request.skill_mode == AgentSkillMode.AUTO:
                skill_detail = f"{len(skills)} skill(s) available"
            else:
                skill_detail = f"{len(skills)} skill(s) selected"
            te("skills.load", "completed", skill_detail)
            ev(AgentEvent.tool_end("skills.load", skill_detail))

        if request.tools.web and request.retrieval_mode != AgentRetrievalMode.FAITHFUL:
            permission_policy.require(AgentCapability.SEARCH_WEB)
            te("web.search", "available",
               "Web search is enabled for this turn. Router decides whether to execute it immediately.")
        if request.tools.anytxt and request.retrieval_mode != AgentRetrievalMode.FAITHFUL:
            permission_policy.require(AgentCapability.SEARCH_ANY_TXT)
            te("anytxt.search", "available",
               "AnyTXT search is enabled for this turn. Router decides whether to execute it immediately.")

        retrieval_parts: List[str] = []
        tool_registry = _new_tool_registry()
        if _llm_usable(self.llm_config):
            return await self._run_agent_loop(
                request, message, session_id, router, skills, permission_policy, tool_registry,
                references, tool_events, events, event_sink, cancellation)

        should_use_model_planner = should_plan_tools_with_model(
            message, request.mode, request.tools, bool(skills))
        planner_has_config = _llm_usable(self.llm_config)
        model_plan_error: Optional[BaseException] = None
        try:
            model_plan = await self._plan_tools_with_model(
                message, request.mode, request.tools, skills, request.skill_mode, cancellation)
        except Exception as e:  # noqa: BLE001 - `Result` is inspected, not propagated
            model_plan_error = e
            model_plan = ModelToolPlan()
        planner_unavailable_or_failed = should_use_model_planner and (
            not planner_has_config or model_plan_error is not None)
        if model_plan.tool_calls:
            te("agent.plan_tools", "completed", f"{len(model_plan.tool_calls)} planned tool call(s)")
        planned_queries = planned_tool_queries(model_plan, message)

        def planned_has(tool: str) -> bool:
            return tool in planned_queries

        # If the planner cannot run, preserve the API/MCP offline contract for
        # plain wiki questions without bringing back message-shape heuristics.
        # Skill turns intentionally do not use this fallback: the skill index is
        # already in the prompt, and falling back to wiki.search would recreate
        # the "what skills do you have?" retrieval bug.
        fallback_wiki_search = should_fallback_wiki_search(
            planner_unavailable_or_failed, request.tools, not skills)
        should_search_wiki = router.should_search_wiki or planned_has("wiki.search") or fallback_wiki_search
        should_include_sources = router.should_include_sources or planned_has("source.search")
        should_search_graph = router.intent == QueryIntent.NEEDS_GRAPH or planned_has("graph.search")
        should_run_web = request.tools.web and (
            router.intent == QueryIntent.NEEDS_EXTERNAL_SEARCH or planned_has("web.search")
            or request.mode == AgentMode.DEEP)
        should_run_anytxt = request.tools.anytxt and (
            should_include_sources or planned_has("anytxt.search") or request.mode == AgentMode.DEEP)
        deep_research = request.mode == AgentMode.DEEP and (
            should_run_web or should_run_anytxt or should_include_sources)

        shell_call: Optional[Tuple[str, Optional[int]]] = None
        if skills:
            req_cmd = _non_empty(_opt_trim(request.shell_command))
            if req_cmd is not None:
                shell_call = (req_cmd, None)
            else:
                for call in model_plan.tool_calls:
                    if call.tool.strip() == "shell.exec":
                        command = shell_command_from_call(call)
                        if command is not None:
                            shell_call = (command, call.timeout_seconds)
                        break

        write_call = next((c for c in model_plan.tool_calls if c.tool.strip() == "wiki.write_page"), None)
        if write_call is not None:
            check_cancel(cancellation)
            permission_policy.require(AgentCapability.WRITE_WIKI)
            path = _non_empty(_opt_trim(write_call.path))
            if path is None:
                raise ValueError("wiki.write_page requires path")
            content = _non_empty(_opt_trim(write_call.content))
            if content is None:
                raise ValueError("wiki.write_page requires content")
            te("wiki.write_page", "started", path)
            ev(AgentEvent.tool_start("wiki.write_page", path))
            try:
                value = await execute_tool_with_cancellation(
                    tool_registry.execute,
                    ("wiki.write_page",
                     {"path": path, "content": content,
                      "allowOverwrite": bool(write_call.allow_overwrite)},
                     self._tool_context()),
                    cancellation)
                output = WikiWriteOutput.from_value(value)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                te("wiki.write_page", "failed", err)
                ev(AgentEvent.tool_end("wiki.write_page", f"failed: {err}"))
            else:
                ev(AgentEvent.file_changed(output.reference.path, "wiki.write_page",
                                           output.existed_before, output.previous_content))
                reference = output.reference
                ev(AgentEvent.reference_added(reference))
                references.append(reference)
                te("wiki.write_page", "completed", path)
                ev(AgentEvent.tool_end("wiki.write_page", path))
                retrieval_parts.append(f"wiki.write_page wrote {path}.")

        if shell_call is not None:
            command, timeout_seconds = shell_call
            check_cancel(cancellation)
            if is_skill_preference_probe_command(command):
                detail = skipped_skill_preference_probe_summary(command)
                te("shell.exec", "completed", "skipped optional skill preference probe")
                ev(AgentEvent.tool_start("shell.exec", command))
                ev(AgentEvent.tool_end("shell.exec", detail))
                retrieval_parts.append(detail)
            else:
                permission_policy.require(AgentCapability.PROCESS)
                if not is_shell_command_approved(command, request.approved_shell_commands):
                    detail = f"approval required: {command}"
                    ev(AgentEvent.tool_start("shell.exec", command))
                    te("shell.exec", "available", detail)
                    ev(AgentEvent.tool_end("shell.exec", detail))
                    retrieval_parts.append(
                        "shell.exec was requested by an active skill but was not run because the command has "
                        f"not been approved: `{command}`.")
                else:
                    te("shell.exec", "started", command)
                    ev(AgentEvent.tool_start("shell.exec", command))
                    try:
                        value = await execute_tool_with_cancellation(
                            tool_registry.execute,
                            ("shell.exec", {"command": command, "timeoutSeconds": timeout_seconds},
                             self._tool_context()),
                            cancellation)
                        output = ShellExecToolOutput.from_value(value)
                    except Exception as e:  # noqa: BLE001
                        err = str(e)
                        te("shell.exec", "failed", err)
                        ev(AgentEvent.tool_end("shell.exec", f"failed: {err}"))
                    else:
                        check_cancel(cancellation)
                        summary = (f"shell.exec `{output.command}` exit={_rust_option_debug(output.exit_code)} "
                                   f"timedOut={_rust_bool(output.timed_out)}\nstdout:\n{output.stdout}\n"
                                   f"stderr:\n{output.stderr}")
                        retrieval_parts.append(summary)
                        te("shell.exec", "completed", f"exit={_rust_option_debug(output.exit_code)}")
                        ev(AgentEvent.tool_end("shell.exec", f"exit={_rust_option_debug(output.exit_code)}"))

        if deep_research:
            te("deep_research.run", "started", message)
            ev(AgentEvent.tool_start("deep_research.run", message))

        top_k_for_search = min(max(request.top_k if request.top_k is not None else DEFAULT_CHAT_SEARCH_RESULTS, 1),
                               MAX_CHAT_SEARCH_RESULTS)

        if should_search_wiki:
            check_cancel(cancellation)
            permission_policy.require(AgentCapability.SEARCH_WIKI)
            wiki_query = planned_queries.get("wiki.search", message)
            te("wiki.search", "started", wiki_query)
            ev(AgentEvent.tool_start("wiki.search", wiki_query))
            try:
                value = await execute_tool_with_cancellation(
                    tool_registry.execute,
                    ("wiki.search",
                     {"query": wiki_query, "topK": top_k_for_search,
                      "includeContent": bool(request.include_content)},
                     self._tool_context()),
                    cancellation)
                search = WikiSearchToolOutput.from_value(value)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                te("wiki.search", "failed", err)
                ev(AgentEvent.tool_end("wiki.search", f"failed: {err}"))
            else:
                check_cancel(cancellation)
                search_refs = search.references
                for reference in search_refs:
                    ev(AgentEvent.reference_added(reference))
                search_count = len(search_refs)
                references.extend(search_refs)
                te("wiki.search", "completed",
                   f"{search_count} result(s), mode={search.mode}, tokenHits={search.token_hits}, "
                   f"vectorHits={search.vector_hits}, graphHits={search.graph_hits}")
                ev(AgentEvent.tool_end("wiki.search", f"{search_count} result(s)"))
                retrieval_parts.append(build_retrieval_answer(message, references))
            if request.mode == AgentMode.DEEP and references:
                permission_policy.require(AgentCapability.READ_PROJECT)
                excerpts: List[str] = []
                for reference in [r for r in references if r.kind == "wiki"][:2]:
                    try:
                        page = _tools().read_wiki_page(self.project_path, reference.path)
                    except Exception:  # noqa: BLE001
                        continue
                    excerpts.append(f"Excerpt from {reference.path}:\n"
                                    f"{_context.collapse_whitespace(page)[:2_000]}")
                if excerpts:
                    te("wiki.read_page", "completed", f"{len(excerpts)} excerpt(s)")
                    ev(AgentEvent.tool_end("wiki.read_page", f"{len(excerpts)} excerpt(s)"))
                    retrieval_parts.append("\n\n".join(excerpts))
        elif request.tools.wiki:
            retrieval_parts.append(
                f"Router intent={_context.intent_label(router.intent)} did not require immediate wiki.search "
                "for this turn.")

        async def reference_search(tool: str, capability: AgentCapability, query: str,
                                   cancel_after: bool) -> None:
            check_cancel(cancellation)
            permission_policy.require(capability)
            te(tool, "started", query)
            ev(AgentEvent.tool_start(tool, query))
            try:
                value = await execute_tool_with_cancellation(
                    tool_registry.execute,
                    (tool, {"query": query, "topK": top_k_for_search}, self._tool_context()),
                    cancellation)
                found = _references_from(value, tool)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                te(tool, "failed", err)
                ev(AgentEvent.tool_end(tool, f"failed: {err}"))
                return
            if cancel_after:
                check_cancel(cancellation)
            for reference in found:
                ev(AgentEvent.reference_added(reference))
            count = len(found)
            references.extend(found)
            te(tool, "completed", f"{count} result(s)")
            ev(AgentEvent.tool_end(tool, f"{count} result(s)"))

        if should_include_sources:
            await reference_search("source.search", AgentCapability.READ_SOURCE,
                                   planned_queries.get("source.search", message), False)
        if should_search_graph:
            await reference_search("graph.search", AgentCapability.READ_PROJECT,
                                   planned_queries.get("graph.search", message), False)
        if should_run_web:
            await reference_search("web.search", AgentCapability.NETWORK,
                                   planned_queries.get("web.search", message), True)
        if should_run_anytxt:
            await reference_search("anytxt.search", AgentCapability.NETWORK,
                                   planned_queries.get("anytxt.search", message), True)

        if deep_research:
            te("deep_research.run", "completed", f"{len(references)} reference(s)")
            ev(AgentEvent.tool_end("deep_research.run", f"{len(references)} reference(s)"))

        if not retrieval_parts:
            if not request.tools.wiki and not request.tools.web and not request.tools.anytxt:
                retrieval_parts.append(
                    "No Agent tools were enabled for this request. Enable wiki, web, or AnyTXT tools to let the "
                    "backend Agent retrieve supporting context.")
            else:
                retrieval_parts.append(
                    "No Agent tools ran before generation. Available tools were exposed as model hints.")
        retrieval_summary = "\n\n".join(retrieval_parts)
        project_context = project_context_for_retrieval_mode(
            _context.load_project_context(self.project_path), request.retrieval_mode)
        explicit_files = await self._load_explicit_files(request.context_files)
        built_context = fit_context_to_model(
            _context.build_agent_context(_context.AgentContextInput(
                query=message, project=project_context, router=router, history=request.history,
                skills=skills, skill_mode=request.skill_mode, references=references,
                retrieval_summary=retrieval_summary, explicit_files=explicit_files)),
            self.llm_config)

        if _llm_usable(self.llm_config):
            check_cancel(cancellation)
            client = _new_llm_client(self.llm_config)
            te("llm.generate", "started",
               f"{_call_or_attr(client, 'provider_name')}:{_call_or_attr(client, 'model_name')}")
            try:
                if event_sink is not None:
                    answer = await generate_with_cancellation_stream(
                        client, built_context.system, built_context.user, request.images, cancellation,
                        lambda delta: ev(AgentEvent.message_delta(delta)))
                else:
                    answer = await generate_with_cancellation(
                        client, built_context.system, built_context.user, request.images, cancellation)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                te("llm.generate", "failed", err)
                ev(AgentEvent.error(err))
                raise
            te("llm.generate", "completed", None)
        else:
            answer = answer_without_backend_generator(message, references, request.allow_empty_retrieval)
        ev(AgentEvent.done(session_id))
        usage = AgentUsage(
            prompt_chars=_utf8_len(built_context.system) + _utf8_len(built_context.user),
            completion_chars=_utf8_len(answer),
            reference_count=len(references),
            tool_event_count=len(tool_events),
        )
        return self._response(request, session_id, answer, references, tool_events, events, usage)

    # ---- helpers --------------------------------------------------------
    def _response(self, request: AgentChatRequest, session_id: str, message: str,
                  references: List[AgentReference], tool_events: List[AgentToolEvent],
                  events: List[AgentEvent], usage: AgentUsage,
                  user_input_request: Optional[AgentUserInputRequest] = None) -> AgentChatResponse:
        return AgentChatResponse(
            ok=True, project_id=self.project_id, session_id=session_id, mode=request.mode, message=message,
            references=references, tool_events=tool_events, events=events,
            user_input_request=user_input_request, usage=usage)

    async def _load_explicit_files(self, requested: Sequence[str]) -> List[Tuple[str, str]]:
        # The Rust loader is async (blocking readers on a pool); ours is sync.
        res = await _invoke(_context.load_explicit_context_files, self.project_path, list(requested))
        return res

    def _tool_context(self) -> Any:
        return _tools().ToolContext(
            project_path=self.project_path,
            embedding_config=self.embedding_config,
            web_search_config=self.web_search_config,
            anytxt_config=self.anytxt_config,
        )

    # ---- the agent loop -------------------------------------------------
    async def _run_agent_loop(
        self,
        request: AgentChatRequest,
        message: str,
        session_id: str,
        router: RouterDecision,
        skills: List[AgentSkill],
        permission_policy: PermissionPolicy,
        tool_registry: Any,
        references: List[AgentReference],
        tool_events: List[AgentToolEvent],
        events: List[AgentEvent],
        event_sink: Optional[AgentEventSink],
        cancellation: Any,
    ) -> AgentChatResponse:
        config = self.llm_config if _llm_usable(self.llm_config) else None
        if config is None:
            raise ValueError("Backend Agent LLM is not configured")
        client = _new_llm_client(config)
        client = client.structured_task_config(agent_structured_max_tokens(bool(skills)))
        project_context = project_context_for_retrieval_mode(
            _context.load_project_context(self.project_path), request.retrieval_mode)
        explicit_files = await self._load_explicit_files(request.context_files)
        if request.context_files:
            detail = (f"{len(explicit_files)} of {min(len(request.context_files), 8)} selected file(s) attached")
            tool_emit_event(tool_events, events, event_sink, AgentToolEvent(
                "context.attach", "failed" if not explicit_files else "completed", detail))
            emit_event(events, event_sink, AgentEvent.tool_end("context.attach", detail))
        observations: List[AgentObservation] = []
        executed_retrievals: Set[str] = set()
        retrieval_steps = 0
        consecutive_no_gain_retrievals = 0
        force_final_next = False
        last_prompt_chars = 0
        has_explicit_skills = request.skill_mode == AgentSkillMode.EXPLICIT and bool(skills)
        max_iterations = agent_loop_iteration_budget(request.mode, has_explicit_skills)
        retrieval_budget = agent_loop_retrieval_budget(request.mode, request.retrieval_mode, has_explicit_skills)

        def ev(event: AgentEvent) -> None:
            emit_event(events, event_sink, event)

        def te(tool: str, status: str, detail: Optional[str] = None) -> None:
            tool_emit_event(tool_events, events, event_sink, AgentToolEvent(tool, status, detail))

        command = _non_empty(_opt_trim(request.shell_command))
        if command is not None:
            if is_shell_command_allowed_without_prompt(command, request.approved_shell_commands,
                                                       self.project_path):
                approved_action = AgentLoopAction(action="tool", tool="shell.exec", command=command,
                                                  timeout_seconds=None)
                observation = await self._execute_agent_loop_tool(
                    request, approved_action, permission_policy, tool_registry, skills, references,
                    tool_events, events, event_sink, cancellation)
                observations.append(observation)

        if request.retrieval_mode == AgentRetrievalMode.FAITHFUL and request.tools.wiki:
            source_action = AgentLoopAction(action="tool", tool="source.search", query=message)
            input = self._agent_loop_tool_input(request, "source.search", source_action)
            executed_retrievals.add(retrieval_signature("source.search", input, request.retrieval_mode))
            retrieval_steps += 1
            observation = await self._execute_agent_loop_tool(
                request, source_action, permission_policy, tool_registry, skills, references,
                tool_events, events, event_sink, cancellation)
            observations.append(observation)

        for iteration in range(max_iterations):
            check_cancel(cancellation)
            must_finalize = force_final_next or retrieval_steps >= retrieval_budget
            built_context = fit_context_to_model(
                _context.build_agent_context(_context.AgentContextInput(
                    query=message, project=project_context, router=router, history=request.history,
                    skills=skills, skill_mode=request.skill_mode, references=references,
                    # Loop observations are appended below in the loop-specific
                    # user block. Passing them through the generic retrieval
                    # summary as well would duplicate large tool outputs on
                    # every iteration.
                    retrieval_summary="", explicit_files=explicit_files)),
                self.llm_config)
            if must_finalize:
                system = build_agent_final_system(built_context.system, request.retrieval_mode)
                user = build_agent_final_user(built_context.user, observations)
            else:
                system = build_agent_loop_system(built_context.system, request.retrieval_mode)
                user = build_agent_loop_user(
                    built_context.user, request, skills, observations, iteration, max_iterations,
                    any(r.kind == "workspace" for r in references))
            last_prompt_chars = len(system) + len(user)
            te("llm.generate", "started",
               f"{_call_or_attr(client, 'provider_name')}:{_call_or_attr(client, 'model_name')}")
            # Images are only sent on the first model turn. Tool observations
            # after that contain text results; resending large base64 images
            # on each loop iteration would multiply cost and latency.
            images = request.images if iteration == 0 else []
            try:
                raw = await generate_with_cancellation(client, system, user, images, cancellation)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                te("llm.generate", "failed", err)
                ev(AgentEvent.error(err))
                raise
            te("llm.generate", "completed", None)
            action = parse_agent_loop_action(raw)

            if must_finalize:
                answer = forced_final_answer(raw, action, references)
                if event_sink is not None:
                    ev(AgentEvent.message_delta(answer))
                ev(AgentEvent.done(session_id))
                return self._response(request, session_id, answer, references, tool_events, events,
                                      AgentUsage(last_prompt_chars, _utf8_len(answer), len(references),
                                                 len(tool_events)))

            if _eq_ic(action.action, "invalid_tool_json"):
                observations.append(record_loop_tool_rejection(
                    "agent.action",
                    action.answer if action.answer is not None
                    else "Invalid tool JSON. Return a corrected compact JSON action.",
                    tool_events, events, event_sink))
                continue

            if _eq_ic(action.action, "final"):
                answer = action.answer if action.answer is not None and action.answer.strip() \
                    else raw.strip()
                if event_sink is not None:
                    ev(AgentEvent.message_delta(answer))
                ev(AgentEvent.done(session_id))
                return self._response(request, session_id, answer, references, tool_events, events,
                                      AgentUsage(last_prompt_chars, _utf8_len(answer), len(references),
                                                 len(tool_events)))

            if not _eq_ic(action.action, "tool"):
                # Some smaller/local models ignore the JSON envelope and answer
                # directly. Treat non-JSON or unknown actions as final text so
                # chat does not get stuck in a repair loop.
                answer = raw.strip()
                if event_sink is not None and answer:
                    ev(AgentEvent.message_delta(answer))
                ev(AgentEvent.done(session_id))
                return self._response(request, session_id, answer, references, tool_events, events,
                                      AgentUsage(last_prompt_chars, _utf8_len(answer), len(references),
                                                 len(tool_events)))

            if action.tool is not None and is_user_ask_tool(action.tool):
                try:
                    request_form = sanitize_user_input_request(action)
                except ValueError as e:
                    observations.append(record_loop_tool_rejection(
                        "user.ask",
                        f"{e}. Return a corrected user.ask schema or answer without asking.",
                        tool_events, events, event_sink))
                    continue
                ev(AgentEvent.user_input_required(request_form))
                ev(AgentEvent.done(session_id))
                answer = request_form.description if request_form.description is not None \
                    else "Please provide the requested information to continue."
                return self._response(
                    request, session_id, answer, references, tool_events, events,
                    AgentUsage(last_prompt_chars, _utf8_len(answer), len(references), len(tool_events)),
                    user_input_request=request_form)

            retrieval_tool = action.tool is not None and is_agent_retrieval_tool(action.tool)
            evidence_count_before = smart_evidence_count(references)
            if retrieval_tool:
                tool = action.tool or ""
                try:
                    tool_input = self._agent_loop_tool_input(request, tool, action)
                except ValueError:
                    tool_input = None
                if tool_input is not None:
                    signature = retrieval_signature(tool, tool_input, request.retrieval_mode)
                    if signature in executed_retrievals:
                        observations.append(record_loop_tool_rejection(
                            tool,
                            "duplicate retrieval skipped; use the existing observation and answer the user",
                            tool_events, events, event_sink))
                        force_final_next = True
                        continue
                    executed_retrievals.add(signature)
                retrieval_steps += 1

            observation = await self._execute_agent_loop_tool(
                request, action, permission_policy, tool_registry, skills, references, tool_events, events,
                event_sink, cancellation)
            if observation.tool == SHELL_APPROVAL_REQUIRED_OBSERVATION:
                answer = observation.summary
                if event_sink is not None:
                    ev(AgentEvent.message_delta(answer))
                ev(AgentEvent.done(session_id))
                return self._response(
                    request, session_id, answer, references, tool_events, events,
                    AgentUsage(last_prompt_chars, _utf8_len(observation.summary), len(references),
                               len(tool_events)))
            observations.append(observation)
            if retrieval_tool and request.retrieval_mode == AgentRetrievalMode.SMART:
                evidence_count_after = smart_evidence_count(references)
                latest = observations[-1]
                if retrieval_added_evidence(latest.tool, latest.summary, evidence_count_before,
                                            evidence_count_after):
                    consecutive_no_gain_retrievals = 0
                else:
                    consecutive_no_gain_retrievals += 1
                    if consecutive_no_gain_retrievals >= 2:
                        force_final_next = True

        answer = agent_iteration_limit_answer(max_iterations, len(observations), references)
        if event_sink is not None:
            ev(AgentEvent.message_delta(answer))
        ev(AgentEvent.done(session_id))
        return self._response(request, session_id, answer, references, tool_events, events,
                              AgentUsage(last_prompt_chars, _utf8_len(answer), len(references),
                                         len(tool_events)))

    async def _execute_agent_loop_tool(
        self,
        request: AgentChatRequest,
        action: AgentLoopAction,
        permission_policy: PermissionPolicy,
        tool_registry: Any,
        skills: Sequence[AgentSkill],
        references: List[AgentReference],
        tool_events: List[AgentToolEvent],
        events: List[AgentEvent],
        event_sink: Optional[AgentEventSink],
        cancellation: Any,
    ) -> AgentObservation:
        def ev(event: AgentEvent) -> None:
            emit_event(events, event_sink, event)

        def te(tool_name: str, status: str, detail: Optional[str] = None) -> None:
            tool_emit_event(tool_events, events, event_sink, AgentToolEvent(tool_name, status, detail))

        tool = _non_empty(_opt_trim(action.tool))
        if tool is None:
            return AgentObservation("agent.action", "invalid tool action: missing tool name")

        try:
            input = self._agent_loop_tool_input(request, tool, action)
        except ValueError as e:
            return record_loop_tool_rejection(tool, str(e), tool_events, events, event_sink)
        input_detail = summarize_tool_input(tool, input)

        if tool == "shell.exec":
            if not skills:
                return record_loop_tool_rejection(
                    tool, "shell.exec is only available when at least one skill is active for this turn",
                    tool_events, events, event_sink)
            cmd = input.get("command")
            command = (cmd if isinstance(cmd, str) else "").strip()
            if is_skill_preference_probe_command(command):
                summary = skipped_skill_preference_probe_summary(command)
                te("shell.exec", "completed", "skipped optional skill preference probe")
                ev(AgentEvent.tool_start("shell.exec", command))
                ev(AgentEvent.tool_end("shell.exec", summary))
                return AgentObservation("shell.exec.preference_probe_skipped", summary)
            try:
                permission_policy.require(AgentCapability.PROCESS)
            except Exception as e:  # noqa: BLE001
                return record_loop_tool_rejection(tool, str(e), tool_events, events, event_sink)
            if not is_shell_command_allowed_without_prompt(command, request.approved_shell_commands,
                                                           self.project_path):
                detail = f"approval required: {command}"
                ev(AgentEvent.tool_start("shell.exec", command))
                te("shell.exec", "available", detail)
                ev(AgentEvent.tool_end("shell.exec", detail))
                return AgentObservation(
                    SHELL_APPROVAL_REQUIRED_OBSERVATION,
                    f"The Agent needs approval before it can run this command:\n\n`{command}`\n\n"
                    "Approve the command if you want the Agent to continue with this skill.")

        if tool == "skill.read_file":
            try:
                permission_policy.require(AgentCapability.READ_PROJECT)
            except Exception as e:  # noqa: BLE001
                return record_loop_tool_rejection(tool, str(e), tool_events, events, event_sink)
            te(tool, "started", input_detail)
            ev(AgentEvent.tool_start(tool, input_detail))
            try:
                value = read_active_skill_file(skills, input)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                te(tool, "failed", err)
                ev(AgentEvent.tool_end(tool, f"failed: {err}"))
                return AgentObservation(tool, f"failed: {err}")
            summary = self._record_loop_tool_success(tool, value, references, events, event_sink)
            te(tool, "completed", summary)
            ev(AgentEvent.tool_end(tool, summary))
            return AgentObservation(tool, summary)

        try:
            require_tool_permission(tool, request, permission_policy)
        except Exception as e:  # noqa: BLE001
            return record_loop_tool_rejection(tool, str(e), tool_events, events, event_sink)
        te(tool, "started", input_detail)
        ev(AgentEvent.tool_start(tool, input_detail))

        try:
            value = await execute_tool_with_cancellation(
                tool_registry.execute, (tool, input, self._tool_context()), cancellation)
        except Exception as e:  # noqa: BLE001
            err = str(e)
            te(tool, "failed", err)
            ev(AgentEvent.tool_end(tool, f"failed: {err}"))
            return AgentObservation(tool, f"failed: {err}")
        summary = self._record_loop_tool_success(tool, value, references, events, event_sink)
        te(tool, "completed", summary)
        ev(AgentEvent.tool_end(tool, summary))
        return AgentObservation(tool, summary)

    def _agent_loop_tool_input(self, request: AgentChatRequest, tool: str,
                               action: AgentLoopAction) -> Dict[str, Any]:
        top_k = action.top_k if action.top_k is not None else request.top_k
        top_k = min(max(top_k if top_k is not None else DEFAULT_CHAT_SEARCH_RESULTS, 1), MAX_CHAT_SEARCH_RESULTS)
        if tool in ("wiki.search", "source.search", "graph.search", "web.search", "anytxt.search"):
            query = _non_empty(_opt_trim(action.query))
            if query is None:
                raise ValueError(f"{tool} requires query")
            include = action.include_content if action.include_content is not None else request.include_content
            return {"query": query, "topK": top_k, "includeContent": bool(include)}
        if tool == "wiki.read_page":
            path = _non_empty(_opt_trim(action.path))
            if path is None:
                raise ValueError("wiki.read_page requires path")
            return {"path": path}
        if tool == "skill.read_file":
            path = _non_empty(_opt_trim(action.path))
            if path is None:
                raise ValueError("skill.read_file requires path")
            return {"skill": _non_empty(_opt_trim(action.skill)), "path": path}
        if tool == "wiki.write_page":
            path = _non_empty(_opt_trim(action.path))
            if path is None:
                raise ValueError("wiki.write_page requires path")
            content = _non_empty(_opt_trim(action.content))
            if content is None:
                raise ValueError("wiki.write_page requires content")
            return {"path": path, "content": content, "allowOverwrite": bool(action.allow_overwrite)}
        if tool in ("workspace.write_file", "workspace.append_file"):
            tool_name = action.tool if action.tool is not None else "workspace.write_file"
            path = _non_empty(_opt_trim(action.path))
            if path is None:
                raise ValueError(f"{tool_name} requires path")
            if action.content is None:
                raise ValueError(f"{tool_name} requires content")
            return {"path": path, "content": action.content}
        if tool == "shell.exec":
            raw_cmd = action.command if action.command is not None else action.query
            command = _non_empty(_opt_trim(raw_cmd))
            if command is None:
                raise ValueError("shell.exec requires command")
            return {"command": command, "timeoutSeconds": action.timeout_seconds}
        raise ValueError(f"Unknown Agent tool: {tool}")

    def _record_loop_tool_success(self, tool: str, value: Any, references: List[AgentReference],
                                  events: List[AgentEvent], event_sink: Optional[AgentEventSink]) -> str:
        if tool == "wiki.search":
            search = WikiSearchToolOutput.from_value(value)
            count = len(search.references)
            added = sum(1 for r in search.references
                        if push_unique_reference(references, events, event_sink, r))
            return (f"{count} result(s), {added} new, mode={search.mode}, tokenHits={search.token_hits}, "
                    f"vectorHits={search.vector_hits}, graphHits={search.graph_hits}")
        if tool in ("source.search", "graph.search", "web.search", "anytxt.search"):
            found = _references_from(value, tool)
            count = len(found)
            added = sum(1 for r in found if push_unique_reference(references, events, event_sink, r))
            return f"{count} result(s), {added} new"
        if tool == "wiki.read_page":
            d = _as_dict(value)
            d = d if isinstance(d, dict) else {}
            path = d.get("path") if isinstance(d.get("path"), str) else "wiki page"
            content = d.get("content") if isinstance(d.get("content"), str) else ""
            summary = f"read {path}\n{_context.trim_chars(_context.collapse_whitespace(content), 4_000)}"
            kc = d.get("knowledgeContext")
            if kc is not None:
                summary += "\nKnowledge context: "
                summary += _context.trim_chars(_value_to_string(_as_dict(kc)), 2_000)
            return summary
        if tool == "skill.read_file":
            d = _as_dict(value)
            d = d if isinstance(d, dict) else {}
            skill = d.get("skill") if isinstance(d.get("skill"), str) else "skill"
            path = d.get("path") if isinstance(d.get("path"), str) else "file"
            content = d.get("content") if isinstance(d.get("content"), str) else ""
            return f"read {skill}:{path}\n{_context.trim_chars(_context.collapse_whitespace(content), 4_000)}"
        if tool == "wiki.write_page":
            output = WikiWriteOutput.from_value(value)
            emit_event(events, event_sink, AgentEvent.file_changed(
                output.reference.path, "wiki.write_page", output.existed_before, output.previous_content))
            reference = output.reference
            path = reference.path
            push_unique_reference(references, events, event_sink, reference)
            return f"wrote {path}"
        if tool in ("workspace.write_file", "workspace.append_file"):
            wout = WorkspaceWriteOutput.from_value(value, tool)
            path = wout.path
            nbytes = wout.bytes
            verb = "updated" if tool == "workspace.append_file" else "written"
            emit_event(events, event_sink, AgentEvent.file_changed(
                wout.path, tool, wout.existed_before, wout.previous_content))
            reference = AgentReference(
                title=_file_name(path) or path, path=path, kind="workspace",
                snippet=f"Generated file {verb} by Agent ({nbytes} bytes).", score=None, knowledge_context=None)
            push_unique_reference(references, events, event_sink, reference)
            return f"{'appended' if tool == 'workspace.append_file' else 'wrote'} {path} ({nbytes} bytes)"
        if tool == "shell.exec":
            sout = ShellExecToolOutput.from_value(value)
            for generated in sout.generated_files:
                reference = AgentReference(
                    title=_file_name(generated.path) or generated.path, path=generated.path, kind="workspace",
                    snippet=f"Generated file written by shell.exec ({generated.bytes} bytes).",
                    score=None, knowledge_context=None)
                push_unique_reference(references, events, event_sink, reference)
            generated_summary = "" if not sout.generated_files else (
                "\nGenerated files:\n" + "\n".join(f"- {f.path}" for f in sout.generated_files))
            return (f"`{sout.command}` exit={_rust_option_debug(sout.exit_code)} "
                    f"timedOut={_rust_bool(sout.timed_out)}\nstdout:\n{_context.trim_chars(sout.stdout, 8_000)}\n"
                    f"stderr:\n{_context.trim_chars(sout.stderr, 4_000)}") + generated_summary
        return _value_to_string(value)

    async def _plan_tools_with_model(self, message: str, mode: AgentMode, tools: AgentToolOptions,
                                     skills: Sequence[AgentSkill], skill_mode: AgentSkillMode,
                                     cancellation: Any) -> ModelToolPlan:
        skills_enabled = bool(skills)
        if not should_plan_tools_with_model(message, mode, tools, skills_enabled):
            return ModelToolPlan()
        if not _llm_usable(self.llm_config):
            return ModelToolPlan()
        config = self.llm_config
        check_cancel(cancellation)
        available = ["wiki.search", "source.search", "graph.search", "wiki.write_page"]
        if tools.web:
            available.append("web.search")
        if tools.anytxt:
            available.append("anytxt.search")
        if skills_enabled:
            available.extend(["skill.read_file", "workspace.write_file", "shell.exec"])
        system = "You are an agent tool planner. Return only compact JSON. Do not explain."
        skill_context = render_skill_planner_context(skills, skill_mode)
        workspace = agent_workspace_display(self.project_path)
        user = (
            f"User request:\n{message}\n\nSkill context:\n{skill_context}\n\n"
            f"Available tools: {', '.join(available)}\n\n"
            f"Agent workspace for generated files: {workspace}\n\n"
            'Return JSON exactly like {"toolCalls":[{"tool":"wiki.search","query":"short query"}]}. Use an empty '
            "array when no tool is needed. The skill context and tool list above are already available to the "
            "assistant; do not call wiki.search, source.search, graph.search, web.search, anytxt.search, "
            "skill.read_file, workspace.write_file, or shell.exec merely to list, explain, or summarize the "
            "currently available skills, tools, modes, or agent capabilities. Use wiki.search for factual or "
            "topical retrieval. Prefer graph.search for relationships, dependencies, neighborhoods, backlinks, or "
            "connections between entities; pass concise entity or concept names instead of the full question. The "
            "planner may select both when the answer needs page content and graph structure. Prefer web.search "
            "only for current/external information. Prefer anytxt.search only for user files outside the wiki. "
            "Use wiki.write_page only when the user explicitly asks to create a wiki page; include path under "
            "wiki/ ending in .md and full Markdown content. Existing pages are create-only by default; include "
            "allowOverwrite:true only when the user explicitly asks to overwrite or update an existing wiki page. "
            "Use skill.read_file for Markdown/reference files inside an active skill directory. Use "
            "workspace.write_file for generated artifacts under agent-workspace; do not inline large heredocs or "
            "generated file bodies inside shell.exec. Use shell.exec only when a relevant active skill requires a "
            "command-line operation after any large files have been written. shell.exec runs from the Agent "
            "workspace; commands that generate files must write them under that workspace and must not write to "
            "home, Desktop, Downloads, system temp folders, hidden app metadata folders, or skill installation "
            "folders."
        )
        client = _new_llm_client(config).structured_task_config(agent_structured_max_tokens(bool(skills)))
        raw = await generate_with_cancellation(client, system, user, [], cancellation)
        return parse_model_tool_plan(raw)


# --------------------------------------------------------------------------
# Adapters used by commands/rest_api.py and commands/agent.py
# --------------------------------------------------------------------------
def redact_for_external_api(event: Dict[str, Any]) -> Dict[str, Any]:
    """`AgentEvent::redact_for_external_api` (functional form): drops rollback snapshots."""
    return _redact_event(event)


def _run_sync(coro_factory: Callable[[], Any]) -> Any:
    """Run a coroutine to completion from sync code (worker threads, tests)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = asyncio.run(coro_factory())
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def build_runtime(project: Dict[str, Any], config: Optional[Dict[str, Any]]) -> AgentRuntime:
    """`AgentRuntime::new(project.id, project.path, ...)` from the REST-style config dicts."""
    config = config or {}
    return AgentRuntime(
        project["id"], project["path"],
        embedding_config=config.get("embedding"),
        llm_config=config.get("llm"),
        web_search_config=config.get("webSearch"),
        anytxt_config=config.get("anytxt"),
    )


def run_chat(project: Dict[str, Any], request: Dict[str, Any], config: Optional[Dict[str, Any]],
             cancel_event: Optional[threading.Event], on_event: Optional[Callable[[Dict[str, Any]], Any]]
             ) -> Dict[str, Any]:
    """REST contract (see commands/rest_api.py): run one turn, return the response dict.

    Raises Exception(message) on failure ("Agent turn cancelled" for cancellation).
    """
    try:
        req = AgentChatRequest.from_dict(request)
    except ValueError as e:
        raise ValueError(f"Invalid request: {e}")
    runtime = build_runtime(project, config)
    sink: Optional[AgentEventSink] = None
    if on_event is not None:
        def sink(event: AgentEvent) -> None:  # noqa: F811
            on_event(dict(event))

    cancellation = _as_cancellation(cancel_event)
    response = _run_sync(lambda: runtime.run_once_with_cancel_and_events(req, cancellation, sink))
    return response.to_dict()
