"""Port of the orchestration half of src/lib/ingest.ts: ``auto_ingest`` (the multi-stage
read -> analyse -> generate -> write -> index/log -> review -> embed pipeline),
``start_ingest`` / ``execute_ingest_writes`` (the interactive two-step flow), plus the
config plumbing that replaces the browser stores (``app_state()`` keys exactly as
``src/lib/project-store.ts`` names them) and the activity reporting that replaces
``useActivityStore`` (``events.publish`` topic ``ingest://activity`` + a persisted
``.llm-wiki/ingest-activity.json``).

Pure helpers (prompt builders, FILE/REVIEW parsing, chunking, budgets, path rewriting,
log builders) live in :mod:`.ingest_core`; this module owns everything with I/O.

Deviations from the TS version (all intentional):

* UI stores are gone. Activity -> events + persisted JSON; review items -> the same
  ``.llm-wiki/review.json`` (see :mod:`.review_store`); the chat store backing
  ``start_ingest`` / ``execute_ingest_writes`` is an in-memory per-project session.
* Config is resolved per project from ``app_state()`` (``llmConfig``, ``providerConfigs``,
  ``taskModelRouting``, ``projectLlmOverrides``, ``customLlmPresets``,
  ``embeddingConfig``, ``multimodalConfig``, ``mineruConfig``, ``outputLanguage`` /
  ``projectOutputLanguages``, ``sourceWatchConfig``).
* ``refreshProjectFileTree`` has no server equivalent (a ``ingest://files-changed`` event is
  published instead).
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .. import events
from ..state import app_state
from . import ingest_core as core
from .ingest_core import NonRetryableIngestError  # noqa: F401  (re-exported for the queue)
from .path_utils import get_file_name, normalize_path
from .source_identity import (
    source_identity_for_path,
    source_reference_identity,
    source_summary_slug_candidates_from_identity,
    source_summary_slug_from_identity,
)
from .sources_merge import parse_sources

ACTIVITY_TOPIC = "ingest://activity"
FILES_CHANGED_TOPIC = "ingest://files-changed"
ACTIVITY_LIMIT = 100


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------

class IngestCancelled(Exception):
    """Raised when the run's AbortSignal fired ("Ingest cancelled")."""


class AbortSignal:
    """Minimal AbortSignal: thread-safe flag plus callbacks (used by the LLM client)."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def abort(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            cbs, self._callbacks = self._callbacks, []
        for cb in cbs:
            try:
                cb()
            except Exception:
                pass

    set = abort

    def add_callback(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(cb)
                return
        cb()


def _aborted(signal: Any) -> bool:
    if signal is None:
        return False
    v = getattr(signal, "aborted", None)
    if v is None:
        fn = getattr(signal, "is_set", None)
        return bool(fn()) if callable(fn) else False
    return bool(v)


# --------------------------------------------------------------------------
# Small async helpers
# --------------------------------------------------------------------------

async def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a sibling/IO function that may be sync (-> worker thread) or async."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    result = await asyncio.to_thread(fn, *args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def _mod(name: str):
    return importlib.import_module("%s.%s" % (__package__, name))


def _mod_opt(name: str):
    try:
        return _mod(name)
    except ImportError:
        return None


def _fs():
    from ..commands import fs

    return fs


async def _try_read(path: str) -> str:
    try:
        return await asyncio.to_thread(_fs().read_file, path)
    except Exception:
        return ""


async def _try_read_source(path: str) -> str:
    try:
        return await asyncio.to_thread(_fs().read_file, path, False)
    except Exception:
        return ""


async def _write(path: str, content: str) -> None:
    """All wiki/aggregate writes: atomic, app-write-marked (file watcher) and file-history recorded."""
    await asyncio.to_thread(_fs().write_file_atomic, path, content)


async def _file_exists(path: str) -> bool:
    return bool(await asyncio.to_thread(_fs().file_exists, path))


async def _delete(path: str) -> None:
    await asyncio.to_thread(_fs().delete_file, path)


async def _mkdir(path: str) -> None:
    await asyncio.to_thread(_fs().create_directory, path)


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Config (stores -> app_state)
# --------------------------------------------------------------------------

PROVIDERS_WITHOUT_KEY = frozenset({"ollama", "custom", "claude-code", "codex-cli"})

_DEFAULT_MULTIMODAL = {
    "enabled": False, "useMainLlm": True, "provider": "custom", "apiKey": "", "model": "",
    "ollamaUrl": "http://localhost:11434", "customEndpoint": "", "azureApiVersion": "2024-10-21",
    "apiMode": "chat_completions", "concurrency": 4,
}


def has_usable_llm(cfg: Optional[Dict[str, Any]]) -> bool:
    """has-usable-llm.ts."""
    if not isinstance(cfg, dict):
        return False
    if cfg.get("provider") in PROVIDERS_WITHOUT_KEY:
        return True
    return bool(str(cfg.get("apiKey") or "").strip())


def _dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _find_preset(preset_id: str, customs: Sequence[Any]) -> Optional[Dict[str, Any]]:
    for p in customs or []:
        if isinstance(p, dict) and p.get("id") == preset_id:
            return p
    # Built-in vendor presets are a UI catalogue (llm-presets.ts) that is not ported; ids that
    # are plain provider names are understood, anything else falls back to the global config.
    if preset_id in ("openai", "anthropic", "google", "minimax", "ollama", "claude-code", "codex-cli", "azure"):
        return {"id": preset_id, "provider": preset_id}
    return None


def _resolve_preset_config(preset: Dict[str, Any], override: Optional[Dict[str, Any]], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Subset of preset-resolver.ts resolveConfig (the fields the server's LLM client reads)."""
    ov = _dict(override)
    provider = preset.get("provider")
    cfg: Dict[str, Any] = dict(fallback)
    cfg["provider"] = provider
    cfg["apiKey"] = "" if provider in ("ollama", "claude-code", "codex-cli") else (ov.get("apiKey") or "")
    cfg["model"] = ov.get("model") or preset.get("defaultModel") or ""
    cfg["maxContextSize"] = ov.get("maxContextSize") or preset.get("suggestedContextSize") or fallback.get("maxContextSize")
    cfg["reasoning"] = ov.get("reasoning") or {"mode": "auto"}
    cfg["ingestReasoning"] = ov.get("ingestReasoning") or {"mode": "off"}
    base = ov.get("baseUrl") or preset.get("baseUrl")
    if provider == "custom":
        cfg["customEndpoint"] = base or ""
        cfg["apiMode"] = ov.get("apiMode") or preset.get("apiMode") or "chat_completions"
    elif provider == "ollama":
        cfg["ollamaUrl"] = base or "http://localhost:11434"
    elif provider == "azure":
        cfg["customEndpoint"] = base or ""
        cfg["azureApiVersion"] = ov.get("azureApiVersion") or preset.get("azureApiVersion") or "2024-10-21"
        cfg["azureModelFamily"] = ov.get("azureModelFamily") or preset.get("azureModelFamily") or "auto"
    for k in ("customHeaders", "streamingEnabled", "requestTimeoutMinutes", "codexCliTimeoutMinutes", "localCliIsolation"):
        if k in ov:
            cfg[k] = ov[k]
    return cfg


def get_task_llm_config(project_id: Optional[str] = None, task: str = "ingest",
                        fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """llm-task-routing.ts getTaskLlmConfig, resolved from ``app_state()`` for a project."""
    st = app_state()
    base = dict(fallback or _dict(st.get("llmConfig")))
    if not fallback and project_id:
        try:
            from ..commands import rest_api

            parsed = rest_api.load_app_state() or {}
            resolved = rest_api.project_llm_config(parsed, project_id)
            if isinstance(resolved, dict):
                base = dict(resolved)
        except Exception:
            pass
    overrides = _dict(st.get("projectLlmOverrides"))
    if project_id and _dict(overrides.get(project_id)).get("enabled"):
        return base
    routing = _dict(st.get("taskModelRouting"))
    preset_id = routing.get("chatPresetId" if task == "chat" else "ingestPresetId")
    if not preset_id:
        return base
    preset = _find_preset(preset_id, st.get("customLlmPresets") or [])
    if not preset:
        return base
    return _resolve_preset_config(preset, _dict(st.get("providerConfigs")).get(preset_id), base)


def task_llm_is_usable(project_id: Optional[str] = None) -> bool:
    return has_usable_llm(get_task_llm_config(project_id))


@dataclass
class IngestConfig:
    """Snapshot of the stores' values the pipeline reads (``useWikiStore.getState()``)."""

    output_language: str = "auto"
    multimodal: Dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_MULTIMODAL))
    mineru: Dict[str, Any] = field(default_factory=dict)
    embedding: Dict[str, Any] = field(default_factory=dict)
    persist_extracted_markdown: bool = False

    def lang_text_name(self, text: str) -> str:
        from .language_metadata import get_language_prompt_name
        from .output_language import get_output_language

        return get_language_prompt_name(get_output_language(text, self.output_language))


def load_ingest_config(project_id: Optional[str] = None) -> IngestConfig:
    st = app_state()
    lang = None
    if project_id:
        lang = _dict(st.get("projectOutputLanguages")).get(project_id)
    lang = lang or st.get("outputLanguage") or "auto"
    mm = dict(_DEFAULT_MULTIMODAL)
    mm.update(_dict(st.get("multimodalConfig")))
    watch = _dict(st.get("sourceWatchConfig"))
    wcfg = _dict(watch.get(project_id)) if project_id and project_id in watch else _dict(watch.get("default"))
    return IngestConfig(
        output_language=str(lang),
        multimodal=mm,
        mineru=_dict(st.get("mineruConfig")),
        embedding=_dict(st.get("embeddingConfig")),
        persist_extracted_markdown=bool(wcfg.get("persistExtractedMarkdown", False)),
    )


# --------------------------------------------------------------------------
# Activity (useActivityStore replacement)
# --------------------------------------------------------------------------

_activity_lock = threading.RLock()
_activity: Dict[str, List[Dict[str, Any]]] = {}
_activity_counter = 0


def activity_path(project_path: str) -> str:
    return "%s/.llm-wiki/ingest-activity.json" % normalize_path(project_path)


def list_activity(project_path: str) -> List[Dict[str, Any]]:
    pp = normalize_path(project_path)
    with _activity_lock:
        if pp not in _activity:
            try:
                with open(activity_path(pp), "r", encoding="utf-8") as f:
                    data = json.load(f)
                _activity[pp] = [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []
            except (OSError, ValueError):
                _activity[pp] = []
        return [dict(i) for i in _activity[pp]]


class Activity:
    """One activity item (a running ingest). ``update`` publishes ``ingest://activity``."""

    def __init__(self, project_path: str, title: str, project_id: Optional[str] = None,
                 task_id: Optional[str] = None, kind: str = "ingest") -> None:
        global _activity_counter
        self.project_path = normalize_path(project_path)
        list_activity(self.project_path)  # load persisted items first
        with _activity_lock:
            _activity_counter += 1
            self.item: Dict[str, Any] = {
                "id": "activity-%d-%d" % (int(time.time() * 1000), _activity_counter), "type": kind,
                "title": title, "status": "running", "detail": "Reading source...", "filesWritten": [],
                "createdAt": int(time.time() * 1000), "projectId": project_id, "taskId": task_id,
            }
            _activity[self.project_path].insert(0, self.item)
            del _activity[self.project_path][ACTIVITY_LIMIT:]
        self._publish(persist=True)

    @property
    def status(self) -> str:
        return self.item["status"]

    @property
    def detail(self) -> str:
        return self.item["detail"]

    def update(self, status: Optional[str] = None, detail: Optional[str] = None,
               files_written: Optional[Sequence[str]] = None) -> None:
        with _activity_lock:
            if status is not None:
                self.item["status"] = status
            if detail is not None:
                self.item["detail"] = detail
            if files_written is not None:
                self.item["filesWritten"] = list(files_written)
        self._publish(persist=status is not None)

    def _publish(self, persist: bool) -> None:
        with _activity_lock:
            snap = dict(self.item)
        try:
            events.publish(ACTIVITY_TOPIC, snap)
        except Exception:
            pass
        if persist:
            try:
                with _activity_lock:
                    text = json.dumps(_activity.get(self.project_path, []), indent=2, ensure_ascii=False)
                path = activity_path(self.project_path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(text)
                os.replace(tmp, path)
            except OSError:
                pass


class _NullActivity:
    status = "running"
    detail = ""

    def update(self, *a: Any, **k: Any) -> None:
        return None


def _throw_if_aborted(signal: Any, act: Any = None) -> None:
    if not _aborted(signal):
        return
    if act is not None:
        act.update(status="error", detail="Ingest cancelled")
    raise IngestCancelled("Ingest cancelled")


# --------------------------------------------------------------------------
# LLM streaming adapter
# --------------------------------------------------------------------------

@dataclass
class StreamResult:
    text: str = ""
    truncated: bool = False
    error: Optional[str] = None


async def stream_text(llm_config: Dict[str, Any], messages: List[Dict[str, Any]], signal: Any = None,
                      *, temperature: Optional[float] = None, reasoning: Any = None,
                      max_tokens: Optional[int] = None) -> StreamResult:
    """Run ``llm_client.stream_chat`` to completion and collect the text.

    Mirrors the TS callback trio: tokens are accumulated, ``onDone`` supplies the
    ``truncated`` flag and ``onError`` is captured in ``.error`` instead of raised.
    """
    client = _mod("llm_client")
    res = StreamResult()
    overrides: Dict[str, Any] = {}
    if temperature is not None:
        overrides["temperature"] = temperature
    if reasoning is not None:
        overrides["reasoning"] = reasoning
    if max_tokens is not None:
        overrides["max_tokens"] = max_tokens

    def on_token(tok: str) -> None:
        res.text += tok

    def on_done(completion: Any = None) -> None:
        if completion is not None:
            t = completion.get("truncated") if isinstance(completion, dict) else getattr(completion, "truncated", None)
            res.truncated = t is True

    def on_error(err: Any) -> None:
        res.error = getattr(err, "message", None) or str(err)

    callbacks = {"on_token": on_token, "on_done": on_done, "on_error": on_error,
                 "onToken": on_token, "onDone": on_done, "onError": on_error}
    await _stream_chat(client, llm_config, messages, callbacks, signal, overrides)
    return res


async def _stream_chat(client: Any, llm_config: Dict[str, Any], messages: List[Dict[str, Any]],
                       callbacks: Dict[str, Any], signal: Any, overrides: Dict[str, Any]) -> None:
    fn = client.stream_chat
    r = fn(llm_config, messages, callbacks, signal, overrides)
    if inspect.isawaitable(r):
        await r


# --------------------------------------------------------------------------
# Image extraction de-duplication (ingestImageExtractionPromises)
# --------------------------------------------------------------------------

_image_extractions: Dict[str, "asyncio.Future[List[Any]]"] = {}


def _img(img: Any, camel: str, snake: Optional[str] = None, default: Any = None) -> Any:
    if isinstance(img, dict):
        if camel in img:
            return img[camel]
        return img.get(snake or camel, default)
    for n in (camel, snake or camel):
        if hasattr(img, n):
            return getattr(img, n)
    return default


async def _image_extraction_key(project_path: str, source_path: str, slug: str) -> str:
    sp = normalize_path(source_path)
    try:
        fs = _fs()
        size = await asyncio.to_thread(fs.get_file_size, sp)
        mtime = await asyncio.to_thread(fs.get_file_modified_time, sp)
        fingerprint = "%s:%s" % (size, mtime)
    except Exception:
        fingerprint = "unstable:%d" % int(time.time() * 1000)
    return "%s\n%s\n%s\n%s" % (normalize_path(project_path), sp, slug, fingerprint)


def _remember_extraction(key: str, fut: "asyncio.Future[List[Any]]") -> "asyncio.Future[List[Any]]":
    _image_extractions[key] = fut
    if len(_image_extractions) > 32:
        _image_extractions.pop(next(iter(_image_extractions)), None)

    def _drop(f: "asyncio.Future[List[Any]]") -> None:
        if f.cancelled() or f.exception() is not None:
            if _image_extractions.get(key) is f:
                _image_extractions.pop(key, None)

    fut.add_done_callback(_drop)
    return fut


async def _extract_source_images(project_path: str, source_path: str, slug: str) -> List[Any]:
    try:
        mod = _mod("extract_source_images")
        return list(await _call(mod.extract_and_save_source_images, project_path, source_path, slug) or [])
    except Exception:
        return []  # never fatal (matches extractAndSaveSourceImages)


async def _extract_source_images_once_by_key(key: str, project_path: str, source_path: str, slug: str):
    existing = _image_extractions.get(key)
    if existing is not None:
        return await existing
    fut = asyncio.ensure_future(_extract_source_images(project_path, source_path, slug))
    _remember_extraction(key, fut)
    return await fut


async def _extract_source_images_once(project_path: str, source_path: str, slug: str) -> List[Any]:
    key = await _image_extraction_key(project_path, source_path, slug)
    return await _extract_source_images_once_by_key(key, project_path, source_path, slug)


async def _saved_images_from_mineru_markdown(project_path: str, slug: str, markdown: str) -> List[Dict[str, Any]]:
    import base64
    import hashlib

    pp = normalize_path(project_path)
    images: List[Dict[str, Any]] = []
    for rel in core.extract_mineru_image_rel_paths(markdown, slug):
        abs_path = "%s/wiki/%s" % (pp, rel)
        try:
            data = await asyncio.to_thread(_fs().read_file_as_base64, abs_path)
            raw = base64.b64decode(data["base64"])
            images.append({
                "index": len(images) + 1, "mimeType": core.image_mime_type_from_path(rel), "page": None,
                "width": 0, "height": 0, "relPath": rel, "absPath": abs_path,
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        except Exception:
            continue
    return images


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

async def _source_generation_fingerprint(source_path: str, source_content: str) -> str:
    fs = _fs()
    try:
        modified = await asyncio.to_thread(fs.get_file_modified_time, source_path)
    except Exception:
        modified = 0
    try:
        size = await asyncio.to_thread(fs.get_file_size, source_path)
    except Exception:
        size = 0
    return core.source_generation_fingerprint_from_stat(source_content, modified, size)


async def _load_generation_checkpoint(path: str, source_identity: str, fingerprint: str,
                                      project_path: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(await asyncio.to_thread(_fs().read_file, path))
        if (not isinstance(parsed, dict) or parsed.get("version") != 1
                or parsed.get("sourceIdentity") != source_identity
                or parsed.get("sourceFingerprint") != fingerprint
                or not isinstance(parsed.get("completedPaths"), list)):
            return None
        completed: List[str] = []
        for p in core.unique_normalized_paths([x for x in parsed["completedPaths"] if isinstance(x, str)]):
            if core.is_safe_ingest_path(p) and await _file_exists("%s/%s" % (project_path, p)):
                completed.append(p)
        parsed["completedPaths"] = completed
        return parsed
    except Exception:
        return None


async def _save_json(path: str, data: Dict[str, Any]) -> None:
    await _mkdir(os.path.dirname(path))
    await _write(path, json.dumps(data, indent=2, ensure_ascii=False))


async def _clear_file(path: str) -> None:
    try:
        if await _file_exists(path):
            await _delete(path)
    except Exception:
        pass


async def _load_long_source_checkpoint(path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(await asyncio.to_thread(_fs().read_file, path))
        return parsed if core.is_compatible_long_source_checkpoint(parsed, params) else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Long-source chunked analysis
# --------------------------------------------------------------------------

async def _analyze_long_source_in_chunks(
    project_path: str, llm_config: Dict[str, Any], purpose: str, schema: str, index: str,
    source_identity: str, slug: str, folder_context: Optional[str], source_content: str,
    source_budget: int, act: Any, signal: Any, output_language: Optional[str],
) -> core.LongSourcePlan:
    target_chars, overlap_chars = core.long_source_chunk_sizes(source_budget)
    chunks = core.split_source_into_semantic_chunks(source_content, target_chars, overlap_chars)
    if len(chunks) <= 1:
        return core.LongSourcePlan(chunked=False, analysis="", source_context=source_content)

    system_prompt = core.build_chunk_analysis_system_prompt(
        purpose, schema, index, source_content, output_language=output_language)
    source_hash = core.hash_text_hex(source_content)
    checkpoint_path = core.long_source_checkpoint_path(project_path, slug, source_hash)
    params = {
        "sourceIdentity": source_identity, "sourceHash": source_hash, "sourceLength": core.len16(source_content),
        "sourceBudget": source_budget, "targetChars": target_chars, "overlapChars": overlap_chars,
        "chunkTotal": len(chunks),
    }
    checkpoint = await _load_long_source_checkpoint(checkpoint_path, params)
    global_digest = (checkpoint or {}).get("globalDigest", "") or ""
    analyses: List[str] = list((checkpoint or {}).get("analyses") or [])
    completed_through = int((checkpoint or {}).get("completedThrough") or 0)
    if completed_through > 0:
        act.update(detail="Resuming long source analysis from chunk %d/%d..." % (completed_through + 1, len(chunks)))

    request = core.ingest_analysis_request(llm_config)
    for chunk in chunks:
        if chunk.index <= completed_through:
            continue
        _throw_if_aborted(signal, act)
        act.update(detail="Analyzing long source chunk %d/%d..." % (chunk.index, chunk.total))
        res = await stream_text(
            llm_config,
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": core.build_chunk_analysis_user_prompt(
                 source_identity, folder_context, chunk,
                 core.trim_long_text(global_digest, core.LONG_SOURCE_DIGEST_MAX))}],
            signal, temperature=0.1, reasoning=request["reasoning"], max_tokens=request["maxTokens"])
        if res.error:
            act.update(status="error", detail="Chunk analysis failed: %s" % res.error)
        _throw_if_aborted(signal, act)
        if res.error:
            raise RuntimeError("Chunk analysis stream failed")
        if res.truncated:
            message = core.chunk_truncated_message(chunk, request["maxTokens"])
            act.update(status="error", detail=message)
            raise RuntimeError(message)
        raw = res.text
        chunk_analysis = core.extract_marked_section(raw, "Chunk Analysis") or core.js_trim(raw)
        next_digest = core.extract_marked_section(raw, "Updated Global Digest")
        analyses.append(core.format_chunk_analysis_entry(chunk, chunk_analysis))
        global_digest = core.next_global_digest(next_digest, global_digest, chunk_analysis)
        completed_through = chunk.index
        await _save_json(checkpoint_path, {
            "version": 1, **params, "completedThrough": completed_through, "globalDigest": global_digest,
            "analyses": analyses, "updatedAt": int(time.time() * 1000),
        })

    return core.LongSourcePlan(
        chunked=True,
        analysis=core.build_consolidated_long_analysis(global_digest, analyses),
        source_context=core.build_long_source_context(source_identity, len(chunks), global_digest, analyses, source_budget),
        checkpoint_path=checkpoint_path,
    )


# --------------------------------------------------------------------------
# Page merge + writing FILE blocks
# --------------------------------------------------------------------------

def _build_page_merger(llm_config: Dict[str, Any]):
    async def merger(existing_content: str, incoming_content: str, source_file_name: str, signal: Any = None) -> str:
        res = await stream_text(
            llm_config,
            [{"role": "system", "content": core.build_page_merge_system_prompt()},
             {"role": "user", "content": core.build_page_merge_user_message(existing_content, incoming_content, source_file_name)}],
            signal, temperature=0.1)
        if res.error:
            raise RuntimeError(res.error)
        return res.text

    return merger


async def _backup_existing_page(project_path: str, relative_path: str, existing_content: str) -> None:
    stamp = re.sub(r"[:.]", "-", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
    sanitized = re.sub(r"[/\\]", "_", relative_path)
    await _write("%s/.llm-wiki/page-history/%s-%s" % (project_path, sanitized, stamp), existing_content)


async def _merge_page(content: str, existing: Optional[str], llm_config: Dict[str, Any], source_identity: str,
                      relative_path: str, signal: Any, project_path: str, replace_existing_body: bool) -> str:
    mod = _mod("page_merge")
    merger = _build_page_merger(llm_config)

    async def backup(old: str) -> None:
        await _backup_existing_page(project_path, relative_path, old)

    opts = mod.MergePageOptions(sourceFileName=source_identity, pagePath=relative_path, signal=signal,
                                backup=backup, replaceExistingBody=replace_existing_body)
    r = mod.merge_page_content(content, existing, merger, opts)
    if inspect.isawaitable(r):
        r = await r
    return r


@dataclass
class WriteResult:
    written_paths: List[str] = field(default_factory=list)
    completed_input_paths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    hard_failures: List[str] = field(default_factory=list)
    truncated_paths: List[str] = field(default_factory=list)


async def _write_file_blocks(
    project_path: str, text: str, llm_config: Dict[str, Any], source_file_name: str,
    source_summary_path: Optional[str], signal: Any, act: Any,
    on_file_written: Optional[Callable[[str], None]], target_lang: Optional[str],
    today: Optional[str] = None,
) -> WriteResult:
    parsed = core.parse_file_blocks(text)
    result = WriteResult(warnings=list(parsed.warnings), truncated_paths=list(parsed.truncated_paths))
    sanitize = _mod("ingest_sanitize").sanitize_ingested_file_content
    wiki_schema = _mod("wiki_schema")
    routing = await asyncio.to_thread(wiki_schema.load_project_wiki_schema_routing, project_path)
    today = today or core.current_wiki_date()

    for block in parsed.blocks:
        _throw_if_aborted(signal, act)
        raw_path, raw_content = block.path, block.content
        relative_path = raw_path
        if source_summary_path and relative_path.startswith("wiki/sources/"):
            relative_path = source_summary_path
        if core.is_app_managed_aggregate_path(relative_path):
            result.warnings.append(
                'Ignored model-generated "%s"; aggregate navigation is maintained by the application.' % relative_path)
            continue

        content = sanitize(raw_content)
        if core.is_log_path(relative_path):
            content = core.stamp_generated_log_date(content, today)
        elif not core.is_listing_path(relative_path):
            content = core.stamp_generated_frontmatter_dates(content, today)
        if not core.is_log_path(relative_path) and not core.is_listing_path(relative_path):
            content = core.canonicalize_sources_field(content, source_file_name)
        if source_summary_path and relative_path == source_summary_path:
            content = core.source_summary_media_refs_for_external_markdown(content)
        relative_path = core.rewrite_ingest_path_from_title_for_target_language(relative_path, content, target_lang)

        if routing is not None and not core.is_log_path(relative_path) and not core.is_listing_path(relative_path):
            correction = wiki_schema.correct_wiki_page_routing(relative_path, content, routing)
            if correction.message:
                if not core.is_safe_ingest_path(correction.path):
                    result.warnings.append('Dropped "%s" — corrected schema path was unsafe.' % relative_path)
                    continue
                result.warnings.append(correction.message)
                relative_path = correction.path
            issue = wiki_schema.validate_wiki_page_routing(relative_path, content, routing)
            if issue:
                result.warnings.append('Dropped "%s" — %s' % (relative_path, issue.message))
                continue

        is_log = core.is_log_path(relative_path)
        is_entity_or_source = (
            relative_path.startswith("wiki/entities/") or "/entities/" in relative_path
            or relative_path.startswith("wiki/sources/") or "/sources/" in relative_path)
        if (target_lang and target_lang != "auto" and not is_log and not is_entity_or_source
                and not core.content_matches_target_language(content, target_lang)):
            result.warnings.append('Dropped "%s" — body language doesn\'t match target %s.' % (relative_path, target_lang))
            continue

        full_path = "%s/%s" % (project_path, relative_path)
        try:
            if core.is_log_path(relative_path):
                existing = await _try_read(full_path)
                appended = "%s\n\n%s" % (existing, core.js_trim(content)) if existing else core.js_trim(content)
                await _write(full_path, appended)
            elif core.is_listing_path(relative_path):
                await _write(full_path, content)
            else:
                existing = await _try_read(full_path)
                replace_body = bool(existing and core.is_owned_only_by_source(existing, source_file_name))
                merged = await _merge_page(content, existing or None, llm_config, source_file_name,
                                           relative_path, signal, project_path, replace_body)
                await _write(full_path, core.canonicalize_sources_field(merged, source_file_name))
            result.written_paths.append(relative_path)
            result.completed_input_paths.append(raw_path)
            if on_file_written:
                on_file_written(relative_path)
        except IngestCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            msg = 'Failed to write "%s": %s' % (relative_path, e)
            result.warnings.append(msg)
            result.hard_failures.append(relative_path)
    return result


# --------------------------------------------------------------------------
# Legacy source-summary migration, index, images
# --------------------------------------------------------------------------

async def _matching_raw_source_identities_for_basename(project_path: str, basename: str) -> List[str]:
    root = "%s/raw/sources" % project_path
    prefix = normalize_path(root).rstrip("/") + "/"
    key = basename.lower()

    def scan() -> List[str]:
        out: List[str] = []
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                full = normalize_path(os.path.join(dirpath, name))
                if name.lower() == key and full.lower().startswith(prefix.lower()):
                    out.append(full[len(prefix):])
        return out

    return await asyncio.to_thread(scan)


async def _migrate_exact_legacy(project_path: str, identity: str, summary_path: str) -> bool:
    pp = normalize_path(project_path)
    canonical = "%s/%s" % (pp, summary_path)
    try:
        if await _file_exists(canonical):
            return False
    except Exception:
        return False
    source_key = normalize_path(identity).lower()
    for slug in source_summary_slug_candidates_from_identity(identity):
        legacy_path = "wiki/sources/%s.md" % slug
        if legacy_path == summary_path:
            continue
        legacy_full = "%s/%s" % (pp, legacy_path)
        try:
            if not await _file_exists(legacy_full):
                continue
            legacy = await asyncio.to_thread(_fs().read_file, legacy_full)
        except Exception:
            continue
        if not any(normalize_path(s).lower() == source_key for s in parse_sources(legacy)):
            continue
        try:
            await _write(canonical, core.canonicalize_sources_field(legacy, identity))
            await _delete(legacy_full)
            return True
        except Exception:
            return False
    return False


async def _migrate_legacy_source_summary(project_path: str, identity: str, summary_path: str) -> None:
    norm = normalize_path(identity)
    if "/" not in norm:
        return
    if await _migrate_exact_legacy(project_path, norm, summary_path):
        return
    basename = get_file_name(norm)
    legacy_slug = re.sub(r"\.[^.]+\Z", "", basename)
    legacy_path = "wiki/sources/%s.md" % legacy_slug
    if legacy_path == summary_path:
        return
    pp = normalize_path(project_path)
    matches = await _matching_raw_source_identities_for_basename(pp, basename)
    if len(matches) != 1 or normalize_path(matches[0]).lower() != norm.lower():
        return
    try:
        if await _file_exists("%s/%s" % (pp, summary_path)) or await _file_exists("%s/raw/sources/%s" % (pp, basename)):
            return
    except Exception:
        return
    legacy = await _try_read("%s/%s" % (pp, legacy_path))
    if not legacy:
        return
    sources = parse_sources(legacy)
    key = basename.lower()
    only_basename = bool(sources) and all(
        "/" not in normalize_path(s) and get_file_name(s).lower() == key for s in sources)
    if not only_basename:
        return
    try:
        await _write("%s/%s" % (pp, summary_path), core.canonicalize_sources_field(legacy, identity))
        await _delete("%s/%s" % (pp, legacy_path))
    except Exception:
        pass


async def _update_wiki_index_deterministically(project_path: str, written_paths: Sequence[str]) -> bool:
    candidates = core.index_candidate_paths(written_paths)
    if not candidates:
        return False
    index_path = "%s/wiki/index.md" % project_path
    try:
        index = await asyncio.to_thread(_fs().read_file, index_path)
    except Exception:
        index = "# Wiki Index\n"
    known = core.known_index_targets(index)
    additions: List[str] = []
    for path in candidates:
        target = re.sub(r"\.md\Z", "", re.sub(r"^wiki/", "", path), flags=re.I)
        if core.normalize_index_target(target) in known:
            continue
        content = await _try_read("%s/%s" % (project_path, path))
        additions.append(core.index_addition_line(path, content))
    if not additions:
        return False
    await _write(index_path, core.update_bounded_recent_index_section(index, additions))
    return True


async def _append_ingest_warning_log(project_path: str, identity: str, warnings: Sequence[str]) -> None:
    if not warnings:
        return
    path = "%s/.llm-wiki/ingest-warnings.log" % project_path
    try:
        await _mkdir("%s/.llm-wiki" % project_path)
        existing = await _try_read(path)
        await _write(path, core.build_ingest_warning_log_text(existing, identity, warnings))
    except Exception:
        pass


async def _inject_images_into_source_summary(
    pp: str, identity: str, slug: str, saved_images: Sequence[Any], output_language: Optional[str] = None,
) -> None:
    if not saved_images:
        return
    summary_full = "%s/wiki/sources/%s.md" % (pp, slug)
    try:
        existing = await _try_read(summary_full)
        cap = _mod("image_caption_pipeline")
        captions = await _call(cap.load_caption_cache, pp, output_language)
        ext = _mod("extract_source_images")
        items = []
        for img in saved_images:
            d = dict(img) if isinstance(img, dict) else {
                "relPath": _img(img, "relPath", "rel_path"), "page": _img(img, "page"),
                "sha256": _img(img, "sha256")}
            d["relPath"] = core.to_source_summary_image_ref(str(_img(d, "relPath", "rel_path")))
            items.append(d)
        section = await _call(ext.build_image_markdown_section, items, captions)
        wrapped = core.wrap_embedded_images_section(section)
        if existing:
            stripped = core.strip_embedded_images_section(existing)
            await _write(summary_full, core.js_trim_end(stripped) + wrapped)
        else:
            await _write(summary_full, core.build_image_stub_source_summary(identity, wrapped, _today_utc()))
    except Exception:
        pass  # non-fatal, as in TS


async def _embed_pages(pp: str, rel_paths: Sequence[str], emb_cfg: Dict[str, Any]) -> None:
    if not (emb_cfg.get("enabled") and emb_cfg.get("model")):
        return
    from ..commands import page_embedding

    for wpath in rel_paths:
        stem = wpath.split("/")[-1]
        stem = re.sub(r"\.md\Z", "", stem)
        if stem in ("index", "log", "overview") or not wpath.startswith("wiki/") or not wpath.endswith(".md"):
            continue
        try:
            await page_embedding.embed_wiki_page(pp, wpath, emb_cfg)
        except Exception:
            continue  # non-critical


async def _reembed_source_summary(pp: str, slug: str, emb_cfg: Dict[str, Any]) -> None:
    await _embed_pages(pp, ["wiki/sources/%s.md" % slug], emb_cfg)


# --------------------------------------------------------------------------
# auto_ingest
# --------------------------------------------------------------------------

async def _with_project_lock(key: str, fn: Callable[[], Awaitable[Any]]) -> Any:
    mod = _mod("project_mutex")
    r = mod.with_project_lock(key, fn)
    if inspect.isawaitable(r):
        return await r
    return r


async def auto_ingest(
    project_path: str,
    source_path: str,
    llm_config: Dict[str, Any],
    signal: Any = None,
    folder_context: Optional[str] = None,
    on_file_written: Optional[Callable[[str], None]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Read source -> LLM analyses -> LLM writes wiki pages. Returns the written relative paths.

    ``options``: ``runCommit`` (queue ordering around the commit section), ``projectId``,
    ``taskId`` (activity metadata), ``config`` (an :class:`IngestConfig` override).
    Concurrency: preparation for different sources overlaps; page/aggregate mutations run
    inside the per-project commit lock.
    """
    pp = normalize_path(project_path)
    sp = normalize_path(source_path)
    return await _with_project_lock(
        "ingest-source\0%s\0%s" % (pp, sp),
        lambda: _auto_ingest_impl(project_path, source_path, llm_config, signal, folder_context,
                                  on_file_written, options or {}),
    )


async def _auto_ingest_impl(
    project_path: str, source_path: str, llm_config: Dict[str, Any], signal: Any,
    folder_context: Optional[str], on_file_written: Optional[Callable[[str], None]],
    options: Dict[str, Any],
) -> List[str]:
    pp = normalize_path(project_path)
    sp = normalize_path(source_path)
    cfg: IngestConfig = options.get("config") or load_ingest_config(options.get("projectId"))
    lang_cfg = cfg.output_language
    file_name = get_file_name(sp)
    identity = source_identity_for_path(pp, sp)
    slug = source_summary_slug_from_identity(identity)
    summary_path = "wiki/sources/%s.md" % slug
    queue_commit = options.get("runCommit")
    wiki_today = core.current_wiki_date()

    async def run_commit(operation: Callable[[], Awaitable[Any]]) -> Any:
        async def locked():
            return await _with_project_lock(pp, operation)

        if queue_commit:
            return await queue_commit(locked)
        return await locked()

    act = Activity(pp, file_name, options.get("projectId"), options.get("taskId"))

    # ── MinerU preprocessing for PDF files ──
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    is_pdf = ext == "pdf"
    mineru_cfg = cfg.mineru
    mineru_saved_images: List[Any] = []
    mineru_configured = mineru_cfg.get("backend") == "local" or bool(mineru_cfg.get("token"))
    if is_pdf and mineru_cfg.get("enabled") and mineru_configured:
        mineru_ok = False
        try:
            cache_dir = sp[: sp.rfind("/")]
            cache_path = "%s/.cache/%s.txt" % (cache_dir, file_name)
            act.update(detail="MinerU: parsing PDF...")
            mineru = _mod("mineru")
            progress = lambda msg: act.update(detail="MinerU: %s" % msg)  # noqa: E731
            result = await _call_mineru(mineru, mineru_cfg, sp, progress, signal, pp, slug)
            await _mkdir("%s/.cache" % cache_dir)
            await _write(cache_path, _img(result, "markdown"))
            mineru_saved_images = list(_img(result, "savedImages", "saved_images", []) or [])
            if mineru_saved_images:
                key = await _image_extraction_key(pp, sp, slug)
                f = asyncio.get_running_loop().create_future()
                f.set_result(mineru_saved_images)
                _remember_extraction(key, f)
            mineru_ok = True
        except IngestCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            _throw_if_aborted(signal, act)
            msg = core.trim_inline_status(str(e))
            act.update(detail="MinerU failed, falling back to built-in PDF extraction: %s" % msg)
        if mineru_ok and not _aborted(signal):
            act.update(detail="Reading source...")

    source_content, schema, purpose, index, overview = await asyncio.gather(
        _try_read_source(sp), _try_read("%s/schema.md" % pp), _try_read("%s/purpose.md" % pp),
        _try_read("%s/wiki/index.md" % pp), _try_read("%s/wiki/overview.md" % pp))
    if cfg.persist_extracted_markdown:
        try:
            await _call(_mod("parsed_source_output").persist_parsed_markdown, pp, sp, source_content)
        except Exception:
            pass
    if is_pdf and not mineru_saved_images and core.has_mineru_image_refs(source_content, slug):
        mineru_saved_images = await _saved_images_from_mineru_markdown(pp, slug, source_content)
        if mineru_saved_images:
            key = await _image_extraction_key(pp, sp, slug)
            f = asyncio.get_running_loop().create_future()
            f.set_result(mineru_saved_images)
            _remember_extraction(key, f)

    mm_cfg = cfg.multimodal
    caption_mod = _mod_opt("image_caption_pipeline")
    ext_mod = _mod_opt("extract_source_images")

    async def collect_images() -> List[Any]:
        skip_native = is_pdf and core.has_mineru_image_refs(source_content, slug)
        images = mineru_saved_images if skip_native else await _extract_source_images(pp, sp, slug)
        try:
            md_images = await _call(ext_mod.extract_and_save_markdown_images, pp, sp, source_content, slug) if ext_mod else []
        except Exception:
            md_images = []
        return list(images) + list(md_images or [])

    # ── Cache check ──
    cache_mod = _mod("ingest_cache")
    cached_files = await _call(cache_mod.check_ingest_cache, pp, identity, source_content)
    if cached_files is not None:
        async def cache_hit() -> List[str]:
            _throw_if_aborted(signal, act)
            try:
                saved = await collect_images()
                if saved:
                    if not mm_cfg.get("enabled"):
                        pass  # captioning disabled: leave images untouched on disk
                    else:
                        caption_llm = core.resolve_caption_config(mm_cfg, llm_config)
                        if caption_llm and caption_mod:
                            try:
                                await _with_project_lock("%s\0image-caption-cache" % pp, lambda: _call_caption(
                                    caption_mod, pp, core.append_saved_image_refs_for_caption(source_content, saved),
                                    caption_llm, signal, mm_cfg, cfg.lang_text_name(source_content),
                                    lambda url: core.is_saved_image_prompt_url(pp, slug, url), act))
                            except Exception:
                                pass
                        await _inject_images_into_source_summary(pp, identity, slug, saved, cfg.lang_text_name(source_content))
                        await _reembed_source_summary(pp, slug, cfg.embedding)
            except Exception:
                pass
            act.update(status="done", detail="Skipped (unchanged) — %d files from previous ingest" % len(cached_files),
                       files_written=cached_files)
            return list(cached_files)

        return await run_commit(cache_hit)

    # ── Step 0.5/0.6: images + captions ──
    act.update(detail="Extracting embedded images...")
    saved_images = await collect_images()
    enriched = core.strip_wiki_media_abs_paths(pp, core.append_saved_image_refs_for_caption(source_content, saved_images))
    caption_llm = core.resolve_caption_config(mm_cfg, llm_config)
    if not mm_cfg.get("enabled") and saved_images:
        enriched = await _call(caption_mod.strip_markdown_image_references, source_content) if caption_mod else source_content
    elif caption_llm and caption_mod and saved_images and "![](" in enriched:
        act.update(detail="Captioning images...")
        our_prefix = "%s/wiki/media/%s/" % (pp, slug)
        try:
            res = await _with_project_lock("%s\0image-caption-cache" % pp, lambda: _call_caption(
                caption_mod, pp, enriched, caption_llm, signal, mm_cfg, cfg.lang_text_name(enriched),
                lambda url: url.startswith(our_prefix) or core.is_saved_image_prompt_url(pp, slug, url), act))
            enriched = core.strip_wiki_media_abs_paths(pp, _img(res, "enrichedMarkdown", "enriched_markdown"))
        except IngestCancelled:
            raise
        except Exception:
            pass  # captioning failure must never break ingest

    stable_len = core.len16(schema) + core.len16(purpose) + core.len16(index) + core.len16(overview)
    source_budget = core.compute_ingest_source_budget(llm_config.get("maxContextSize"), stable_len)
    source_context = enriched
    precomputed_analysis = ""
    long_checkpoint_path: Optional[str] = None
    if core.len16(enriched) > source_budget:
        plan = await _analyze_long_source_in_chunks(
            pp, llm_config, purpose, schema, index, identity, slug, folder_context, enriched,
            source_budget, act, signal, lang_cfg)
        if plan.chunked:
            source_context = plan.source_context
            precomputed_analysis = plan.analysis
            long_checkpoint_path = plan.checkpoint_path

    # ── Step 1: analysis ──
    act.update(detail="Step 1/2: Consolidating long-source analysis..." if precomputed_analysis
               else "Step 1/2: Analyzing source...")
    analysis = precomputed_analysis
    if not analysis:
        request = core.ingest_analysis_request(llm_config)
        folder_line = "\n**Folder context:** %s" % folder_context if folder_context else ""
        res = await stream_text(
            llm_config,
            [{"role": "system", "content": core.build_analysis_prompt(purpose, index, source_context, schema, output_language=lang_cfg)},
             {"role": "user", "content": "Analyze this source document:\n\n**File:** %s%s\n\n---\n\n%s" % (identity, folder_line, source_context)}],
            signal, temperature=0.1, reasoning=request["reasoning"], max_tokens=request["maxTokens"])
        analysis = res.text
        if res.error:
            act.update(status="error", detail="Analysis failed: %s" % res.error)
        if res.truncated:
            message = (
                "Analysis was truncated after reaching the {:,} token output limit. "
                "Wiki pages were not generated from the incomplete analysis. "
                "Split the source or use a model with a larger output limit.").format(request["maxTokens"])
            act.update(status="error", detail=message)
            raise NonRetryableIngestError(message)
        _throw_if_aborted(signal, act)
        if res.error:
            raise RuntimeError(act.detail or "Analysis stream failed")

    # ── Step 2: generation ──
    act.update(detail="Step 2/2: Generating wiki pages...")
    fingerprint = await _source_generation_fingerprint(sp, source_content)
    gen_progress_path = core.generation_checkpoint_path(pp, slug, fingerprint)
    checkpoint = await _load_generation_checkpoint(gen_progress_path, identity, fingerprint, pp)
    previously_completed: List[str] = list((checkpoint or {}).get("completedPaths") or [])
    generation = ""
    generation_error: Optional[str] = None
    generation_truncated = False
    resume_lines = ""
    if previously_completed:
        resume_lines = "\n".join([
            "", "## Resume checkpoint",
            "The following FILE paths were already written successfully in an earlier attempt.",
            "Do not emit them again. Emit only the remaining pages required by the analysis:",
            *["- %s" % p for p in previously_completed],
            "If every required page is already complete, return exactly `---RESUME COMPLETE---`.",
        ])
    # TS: .filter(Boolean) drops empty strings (including the blank separator lines).
    user_content = "\n".join(x for x in [
        "Source document to process: **%s**" % identity,
        "The Stage 1 analysis below is CONTEXT to inform your output. Do NOT echo",
        "its tables, bullet points, or prose. Your output must be FILE/REVIEW",
        "blocks as specified in the system prompt — nothing else.",
        resume_lines,
        "## Stage 1 Analysis (context only — do not repeat)", analysis,
        "## Source Context", source_context, "---",
        "Now emit the FILE blocks for the wiki files derived from **%s**." % identity,
        "Your response MUST begin with `---FILE:` as the very first characters.",
        "No preamble. No analysis prose. Start immediately.",
    ] if x)
    try:
        res = await stream_text(
            llm_config,
            [{"role": "system", "content": core.build_generation_prompt(
                schema, purpose, index, identity, overview, source_context, summary_path, output_language=lang_cfg)},
             {"role": "user", "content": user_content}],
            signal, temperature=0.1, reasoning=_resolve_ingest_reasoning(llm_config),
            max_tokens=core.compute_ingest_generation_max_tokens(llm_config.get("maxContextSize")))
        generation, generation_truncated, generation_error = res.text, res.truncated, res.error
    except IngestCancelled:
        raise
    except Exception as e:  # noqa: BLE001
        _throw_if_aborted(signal, act)
        generation_error = str(e)

    parsed_generation = core.parse_file_blocks(generation)
    resume_confirmed = bool(previously_completed) and generation.strip() == "---RESUME COMPLETE---"
    if (generation_error or generation_truncated) and not parsed_generation.blocks:
        message = ("Generation failed before any complete wiki file was produced: %s" % generation_error
                   if generation_error else "Generation was truncated before any complete wiki file was produced.")
        act.update(status="error", detail=message)
        raise RuntimeError(message)
    if previously_completed and not resume_confirmed and not parsed_generation.blocks:
        message = "Generation resume returned no complete wiki file and did not confirm completion."
        act.update(status="error", detail=message)
        raise RuntimeError(message)
    _throw_if_aborted(signal, act)

    review_suggestion_output = ""
    if (not generation_error and not generation_truncated and not _aborted(signal)
            and core.should_run_dedicated_review_stage(generation)):
        try:
            rres = await stream_text(
                llm_config,
                [{"role": "system", "content": core.build_review_suggestion_prompt(
                    purpose, index, identity, analysis, source_context, generation,
                    llm_config.get("maxContextSize"), output_language=lang_cfg)},
                 {"role": "user", "content": "Emit only high-value REVIEW blocks for follow-up research or unresolved knowledge gaps. Output nothing if there are none."}],
                signal, temperature=0.1, reasoning=_resolve_ingest_reasoning(llm_config),
                max_tokens=core.compute_ingest_review_max_tokens(llm_config.get("maxContextSize")))
            review_suggestion_output = "" if rres.error else rres.text
        except IngestCancelled:
            raise
        except Exception:
            _throw_if_aborted(signal, act)
        _throw_if_aborted(signal, act)

    # ── Step 3: write files (commit section) ──
    async def commit() -> List[str]:
        _throw_if_aborted(signal, act)
        act.update(detail="Writing files...")
        await _migrate_legacy_source_summary(pp, identity, summary_path)
        wr = await _write_file_blocks(pp, generation, llm_config, identity, summary_path, signal, act,
                                      on_file_written, lang_cfg, wiki_today)
        _throw_if_aborted(signal, act)
        written = core.unique_normalized_paths([*previously_completed, *wr.written_paths])
        warnings = wr.warnings
        hard_failures = wr.hard_failures
        unrecovered = core.unique_normalized_paths(
            [p for p in wr.truncated_paths if not any(normalize_path(w) == normalize_path(p) for w in written)])

        if unrecovered and not _aborted(signal):
            act.update(detail="Retrying truncated wiki files: %s" % ", ".join(unrecovered))
            try:
                rr = await stream_text(
                    llm_config,
                    [{"role": "system", "content": core.build_truncated_file_repair_prompt(
                        unrecovered, identity, schema=schema, purpose=purpose, analysis=analysis,
                        source_context=source_context, max_context_size=llm_config.get("maxContextSize"),
                        output_language=lang_cfg)},
                     {"role": "user", "content": "Regenerate the requested FILE blocks now. Start immediately with `---FILE:`."}],
                    signal, temperature=0.1, reasoning=_resolve_ingest_reasoning(llm_config),
                    max_tokens=core.compute_ingest_generation_max_tokens(llm_config.get("maxContextSize")))
                _throw_if_aborted(signal, act)
                if rr.error:
                    warnings.append("Truncated FILE repair failed: %s" % rr.error)
                elif rr.text.strip():
                    filtered = core.filter_truncated_file_repair_output(rr.text, unrecovered)
                    warnings.extend(filtered.warnings)
                    rep = await _write_file_blocks(pp, filtered.text, llm_config, identity, summary_path,
                                                   signal, act, on_file_written, lang_cfg, wiki_today)
                    completed_keys = {normalize_path(p) for p in rep.completed_input_paths}
                    recovered = [p for p in filtered.paths if normalize_path(p) in completed_keys]
                    for p in rep.written_paths:
                        if not any(normalize_path(w) == normalize_path(p) for w in written):
                            written.append(p)
                    for p in recovered:
                        prefix = 'FILE block "%s" was not closed before end of stream' % p
                        warnings[:] = [w for w in warnings if not w.startswith(prefix)]
                    warnings.extend(rep.warnings)
                    hard_failures.extend(rep.hard_failures)
                    rkeys = {normalize_path(p) for p in recovered}
                    unrecovered = [p for p in unrecovered if normalize_path(p) not in rkeys]
            except IngestCancelled:
                raise
            except Exception as e:  # noqa: BLE001
                _throw_if_aborted(signal, act)
                warnings.append("Truncated FILE repair failed: %s" % e)

        try:
            if await _update_wiki_index_deterministically(pp, written):
                written.append("wiki/index.md")
                if on_file_written:
                    on_file_written("wiki/index.md")
        except Exception as e:  # noqa: BLE001
            warnings.append("Deterministic index update failed: %s" % e)

        if not any(normalize_path(p).lower() == "wiki/log.md" for p in written) and not _aborted(signal):
            try:
                log_path = "%s/wiki/log.md" % pp
                existing_log = await _try_read(log_path)
                await _write(log_path, core.build_deterministic_ingest_log(existing_log, identity, wiki_today))
                written.append("wiki/log.md")
                if on_file_written:
                    on_file_written("wiki/log.md")
            except Exception as e:  # noqa: BLE001
                warnings.append("Deterministic log update failed: %s" % e)

        warning_summary = ""
        if warnings:
            await _append_ingest_warning_log(pp, identity, warnings)
            warning_summary = warnings[0] if len(warnings) == 1 else "%d ingest warnings: %s%s" % (
                len(warnings), " · ".join(warnings[:2]),
                " … (+%d more in .llm-wiki/ingest-warnings.log)" % (len(warnings) - 2) if len(warnings) > 2 else "")
            act.update(detail="%s — saved to .llm-wiki/ingest-warnings.log" % warning_summary)

        if not any(normalize_path(p) == summary_path for p in written) and not _aborted(signal):
            fallback = core.build_fallback_source_summary(identity, analysis, _today_utc())
            try:
                await _write("%s/%s" % (pp, summary_path), fallback)
                written.append(summary_path)
                if on_file_written:
                    on_file_written(summary_path)
            except Exception:
                pass

        if mm_cfg.get("enabled") and saved_images and not _aborted(signal):
            await _inject_images_into_source_summary(pp, identity, slug, saved_images, cfg.lang_text_name(source_content))

        if written:
            try:
                events.publish(FILES_CHANGED_TOPIC, {"projectPath": pp, "files": list(written)})
            except Exception:
                pass

        resume_no_valid = bool(previously_completed) and not resume_confirmed and not wr.written_paths
        incomplete = bool(generation_error) or generation_truncated or resume_no_valid
        if incomplete and written:
            try:
                await _save_json(gen_progress_path, {
                    "version": 1, "sourceIdentity": identity, "sourceFingerprint": fingerprint,
                    "completedPaths": [p for p in written if not core.is_app_managed_aggregate_path(p)],
                    "updatedAt": int(time.time() * 1000)})
            except Exception as e:  # noqa: BLE001
                hard_failures.append("Generation checkpoint write failed: %s" % e)

        if hard_failures or unrecovered or incomplete:
            reasons = [r for r in [
                "generation stream failed: %s" % generation_error if generation_error else "",
                "generation reached its output limit" if generation_truncated else "",
                "generation resume produced no valid wiki files" if resume_no_valid else "",
                "%d wiki file write failure(s)" % len(hard_failures) if hard_failures else "",
                "%d truncated wiki file(s) could not be repaired: %s" % (len(unrecovered), ", ".join(unrecovered))
                if unrecovered else "",
            ] if r]
            message = "Ingest incomplete: %s" % "; ".join(reasons)
            act.update(status="error",
                       detail=("%s — %s (saved to .llm-wiki/ingest-warnings.log)" % (message, warning_summary))
                       if warning_summary else message,
                       files_written=written)
            raise RuntimeError(message)

        # ── Step 4: review items ──
        _throw_if_aborted(signal, act)
        review_items = [*core.parse_review_blocks(generation, sp),
                        *core.parse_review_blocks(review_suggestion_output, sp)]
        if review_items:
            from . import review_store

            await asyncio.to_thread(review_store.add_items, pp, review_items)

        # ── Step 5: cache ──
        if written and not hard_failures and not unrecovered:
            await _call(cache_mod.save_ingest_cache, pp, identity, source_content, written)
            await _clear_file(gen_progress_path)
            if long_checkpoint_path:
                await _clear_file(long_checkpoint_path)

        # ── Step 6: embeddings ──
        await _embed_pages(pp, written, cfg.embedding)

        base = ("%d files written%s" % (len(written), ", %d review item(s)" % len(review_items) if review_items else "")
                if written else "No files generated")
        detail = "%s — %s (saved to .llm-wiki/ingest-warnings.log)" % (base, warning_summary) if warning_summary else base
        act.update(status="done" if written else "error", detail=detail, files_written=written)
        return written

    return await run_commit(commit)


def _resolve_ingest_reasoning(llm_config: Dict[str, Any]) -> Any:
    from .reasoning_capabilities import resolve_ingest_reasoning

    return resolve_ingest_reasoning(llm_config)


async def _call_mineru(mineru: Any, cfg: Dict[str, Any], sp: str, progress: Callable[[str], None],
                       signal: Any, pp: str, slug: str) -> Any:
    fn = mineru.parse_with_mineru_result
    r = fn(cfg, sp, None, progress, signal, {"projectPath": pp, "sourceSummarySlug": slug})
    if inspect.isawaitable(r):
        r = await r
    return r


async def _call_caption(mod: Any, pp: str, markdown: str, llm: Dict[str, Any], signal: Any,
                        mm_cfg: Dict[str, Any], language_name: str, should_caption: Callable[[str], bool],
                        act: Any) -> Any:
    opts = {
        "signal": signal,
        "shouldCaption": should_caption,
        "urlToAbsPath": lambda url: core.prompt_image_url_to_abs(pp, url),
        "concurrency": mm_cfg.get("concurrency"),
        "outputLanguage": language_name,
        "onProgress": lambda done, total: act.update(detail="Captioning images... %d/%d" % (done, total)),
    }
    r = mod.caption_markdown_images(pp, markdown, llm, opts)
    if inspect.isawaitable(r):
        r = await r
    return r


# --------------------------------------------------------------------------
# Interactive flow: start_ingest / execute_ingest_writes (chat-store replacement)
# --------------------------------------------------------------------------

_sessions_lock = threading.RLock()
_sessions: Dict[str, Dict[str, Any]] = {}


def get_ingest_session(project_path: str) -> Dict[str, Any]:
    with _sessions_lock:
        s = _sessions.get(normalize_path(project_path))
        return json.loads(json.dumps(s)) if s else {"mode": None, "ingestSource": None, "messages": []}


def _session(project_path: str) -> Dict[str, Any]:
    with _sessions_lock:
        return _sessions.setdefault(normalize_path(project_path),
                                    {"mode": None, "ingestSource": None, "messages": []})


async def start_ingest(project_path: str, source_path: str, llm_config: Dict[str, Any],
                       signal: Any = None, project_id: Optional[str] = None) -> str:
    """startIngest: stream the model's takeaways for the source into an in-memory session.

    Returns the assistant text (the TS version streamed it into the chat store).
    """
    pp = normalize_path(project_path)
    sp = normalize_path(source_path)
    cfg = load_ingest_config(project_id)
    identity = source_identity_for_path(pp, sp)
    slug = source_summary_slug_from_identity(identity)
    sess = _session(pp)
    with _sessions_lock:
        sess.update({"mode": "ingest", "ingestSource": sp, "messages": []})

    # Eager image extraction so the images are on disk before the user reads the analysis.
    key = await _image_extraction_key(pp, sp, slug)
    if key not in _image_extractions:
        _remember_extraction(key, asyncio.ensure_future(_extract_source_images(pp, sp, slug)))

    source_content, schema, purpose, index = await asyncio.gather(
        _try_read_source(sp), _try_read("%s/wiki/schema.md" % pp), _try_read("%s/wiki/purpose.md" % pp),
        _try_read("%s/wiki/index.md" % pp))
    system_prompt = "\n\n".join(x for x in [
        "You are a knowledgeable assistant helping to build a wiki from source documents.",
        core.language_rule(source_content, cfg.output_language),
        "## Wiki Purpose\n%s" % purpose if purpose else "",
        "## Wiki Schema\n%s" % schema if schema else "",
        "## Current Wiki Index\n%s" % index if index else "",
    ] if x)
    user_message = "\n".join([
        "I'm ingesting the following source file into my wiki: **%s**" % identity, "",
        "Please read it carefully and present the key takeaways, important concepts, and information that would be valuable to capture in the wiki. Highlight anything that relates to the wiki's purpose and schema.",
        "", "---", "**File: %s**" % identity, "```", source_content or "(empty file)", "```",
    ])
    with _sessions_lock:
        sess["messages"].append({"role": "user", "content": user_message})
    res = await stream_text(llm_config, [{"role": "system", "content": system_prompt},
                                         {"role": "user", "content": user_message}], signal)
    text = "Error during ingest: %s" % res.error if res.error else res.text
    with _sessions_lock:
        sess["messages"].append({"role": "assistant", "content": text})
    return text


async def execute_ingest_writes(project_path: str, llm_config: Dict[str, Any], user_guidance: Optional[str] = None,
                                signal: Any = None, project_id: Optional[str] = None) -> List[str]:
    """executeIngestWrites: turn the start_ingest discussion into wiki files (serialised by the project lock)."""
    pp = normalize_path(project_path)
    return await _with_project_lock(pp, lambda: _execute_ingest_writes_impl(pp, llm_config, user_guidance, signal, project_id))


async def _execute_ingest_writes_impl(pp: str, llm_config: Dict[str, Any], user_guidance: Optional[str],
                                      signal: Any, project_id: Optional[str]) -> List[str]:
    cfg = load_ingest_config(project_id)
    sess = _session(pp)
    ingest_source = sess.get("ingestSource")
    identity = source_identity_for_path(pp, ingest_source) if ingest_source else None
    slug = source_summary_slug_from_identity(identity) if identity else None
    summary_path = "wiki/sources/%s.md" % slug if slug else None

    schema, index = await asyncio.gather(_try_read("%s/wiki/schema.md" % pp), _try_read("%s/wiki/index.md" % pp))
    with _sessions_lock:
        history = [{"role": m["role"], "content": m["content"]} for m in sess["messages"] if m["role"] != "system"]
    write_prompt = "\n".join([
        "Based on our discussion, please generate the wiki files that should be created or updated.", "",
        "Additional guidance: %s" % user_guidance if user_guidance else "", "",
        "## Wiki Schema\n%s" % schema if schema else "",
        "## Current Wiki Index\n%s" % index if index else "",
        "\n".join(["## Source File", "The original source file is: **%s**" % identity,
                   "If you generate a source summary page, it MUST use this exact path: **%s**." % summary_path,
                   'Every page generated from this source MUST include "%s" in its frontmatter `sources` field.' % identity])
        if identity and summary_path else "",
        "", "Output ONLY the file contents in this exact format for each file:", "```",
        "---FILE: wiki/path/to/file.md---", "(file content here)", "---END FILE---", "```", "",
        "For wiki/log.md, include a log entry to append. For all other files, output the complete file content.",
        "Do not generate wiki/index.md or wiki/overview.md. The application owns those aggregate files.",
        "Use relative paths from the project root (e.g., wiki/sources/topic.md).",
        "Do not include any other text outside the FILE blocks.",
    ])
    history.append({"role": "user", "content": write_prompt})
    with _sessions_lock:
        sess["messages"].append({"role": "user", "content": write_prompt})
    history_text = "\n".join(m["content"] for m in history)[:2000]
    system_prompt = "\n\n".join(x for x in [
        "You are a wiki generation assistant. Your task is to produce structured wiki file contents.",
        core.language_rule(history_text, cfg.output_language),
        "## Wiki Schema\n%s" % schema if schema else "",
    ] if x)
    res = await stream_text(llm_config, [{"role": "system", "content": system_prompt}, *history], signal)
    accumulated = "Error generating wiki files: %s" % res.error if res.error else res.text
    with _sessions_lock:
        sess["messages"].append({"role": "assistant", "content": accumulated})

    sanitize = _mod("ingest_sanitize").sanitize_ingested_file_content
    written_paths: List[str] = []
    for m in core.FILE_BLOCK_REGEX.finditer(accumulated):
        relative_path = m.group(1).strip()
        content = m.group(2)
        if not relative_path:
            continue
        if summary_path and relative_path.startswith("wiki/sources/"):
            relative_path = summary_path
        if not core.is_safe_ingest_path(relative_path) or core.is_app_managed_aggregate_path(relative_path):
            continue
        content = sanitize(content)
        if identity and not core.is_log_path(relative_path) and not core.is_listing_path(relative_path):
            content = core.canonicalize_sources_field(content, identity)
        full = "%s/%s" % (pp, relative_path)
        try:
            if core.is_log_path(relative_path):
                existing = await _try_read(full)
                await _write(full, "%s\n\n%s" % (existing, core.js_trim(content)) if existing else core.js_trim(content))
            else:
                await _write(full, content)
            written_paths.append(full)
        except Exception:
            continue

    with _sessions_lock:
        sess["messages"].append({
            "role": "system",
            "content": ("Files written to wiki:\n" + "\n".join("- %s" % p for p in written_paths)) if written_paths
            else "No files were written. The LLM response did not contain valid FILE blocks."})

    if ingest_source and cfg.multimodal.get("enabled"):
        key = None
        try:
            key = await _image_extraction_key(pp, ingest_source, slug)
            images = await _extract_source_images_once_by_key(key, pp, ingest_source, slug)
            if images:
                await _inject_images_into_source_summary(pp, identity, slug, images)
        except Exception:
            pass
        finally:
            if key:
                _image_extractions.pop(key, None)
    return written_paths
