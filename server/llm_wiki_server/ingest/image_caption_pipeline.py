"""Port of src/lib/image-caption-pipeline.ts: caption markdown images + persistent hash-keyed cache.

Public API (all ``async`` except the pure helpers)::

    caption_markdown_images(project_path, markdown, llm_config, options=None) -> CaptionPipelineResult
    load_caption_cache(project_path, output_language=None) -> Dict[str, str]     # image sha256 -> caption
    strip_markdown_image_references(markdown) -> str
    find_image_references(markdown) -> List[ImageRef]                            # TS __test.findImageReferences
    caption_cache_key(image_hash, output_language=None) -> str
    normalize_caption_language(output_language=None) -> str
    sha256_of_base64(b64) -> str

``options`` mirrors ``CaptionPipelineOptions`` (dict with the TS keys, snake_case accepted too):
``urlToAbsPath(url) -> str | None``, ``signal``, ``shouldCaption(url) -> bool``, ``concurrency``,
``outputLanguage``, ``onProgress(done, total)``.  Callables may be sync or async.

The result is a :class:`CaptionPipelineResult` (attributes ``enrichedMarkdown``, ``freshCaptions``,
``cachedCaptions``, ``failed``; also subscriptable like the TS object).

All string offsets are UTF-16 code units, exactly as in JS (``ImageRef.index`` / ``length``, the
150-unit context windows), by working on a surrogate-pair view of the text internally.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import inspect
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

from ._js import Record
from .concurrency_limits import clamp_user_concurrency
from .llm_jscompat import (
    JS_WS,
    decode_uri,
    dumps_js_pretty,
    encode_uri_component,
    from_utf16,
    js_json_loads,
    js_trim,
    to_utf16,
)
from .path_utils import normalize_path
from .vision_caption import caption_image

log = logging.getLogger("llm_wiki_server.ingest.image_caption_pipeline")

CACHE_REL_PATH = ".llm-wiki/image-caption-cache.json"
CONTEXT_CHARS = 150
_TRAILING_IMAGE_TITLE_RE = re.compile(JS_WS + r"+(?:\"[^\"]*\"|'[^']*')" + JS_WS + r"*\Z")
_SHA256_KEY_RE = re.compile(r"[a-f0-9]{64}", re.IGNORECASE | re.ASCII)


# ----------------------------------------------------------------- fs boundary (patchable in tests)

async def read_file_as_base64(path: str) -> Dict[str, str]:
    from ..commands import fs
    return await asyncio.to_thread(fs.read_file_as_base64, path)


async def read_file(path: str) -> str:
    def _read() -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return await asyncio.to_thread(_read)


async def write_file(path: str, contents: str) -> None:
    from ..commands import fs
    await asyncio.to_thread(fs.write_file_atomic, path, contents)


async def file_exists(path: str) -> bool:
    from ..commands import fs
    return await asyncio.to_thread(fs.file_exists, path)


async def create_directory(path: str) -> None:
    from ..commands import fs
    await asyncio.to_thread(fs.create_directory, path)


async def _maybe(r: Any) -> Any:
    return await r if inspect.isawaitable(r) else r


# ----------------------------------------------------------------- types

@dataclass
class ImageRef(Record):
    full: str          # the entire `![alt](url)` substring
    alt: str
    url: str           # normalized destination (filesystem-ish path)
    destination: str   # exactly as written between the parentheses, incl. a title
    index: int         # UTF-16 offset of the match
    length: int        # UTF-16 length of the match


@dataclass
class CaptionPipelineResult(Record):
    enrichedMarkdown: str
    freshCaptions: int
    cachedCaptions: int
    failed: int


# ----------------------------------------------------------------- hashing / keys

def _atob(b64: str) -> bytes:
    """Forgiving-base64 decode like `atob` (raises ValueError on invalid input)."""
    s = re.sub(r"[\t\n\f\r ]", "", b64)
    if len(s) % 4 == 0:
        s = re.sub(r"={1,2}\Z", "", s)
    if len(s) % 4 == 1 or re.search(r"[^A-Za-z0-9+/]", s):
        raise ValueError("The string to be decoded is not correctly encoded.")
    s += "=" * (-len(s) % 4)
    try:
        return base64.b64decode(s)
    except (binascii.Error, ValueError) as e:
        raise ValueError(str(e))


def sha256_of_base64(b64: str) -> str:
    """SHA-256 hex digest of the image BYTES (not the base64 text)."""
    return hashlib.sha256(_atob(b64)).hexdigest()


def normalize_caption_language(output_language: Optional[str] = None) -> str:
    language = js_trim(output_language) if output_language else ""
    if not language or language.lower() == "auto":
        return ""
    return unicodedata.normalize("NFKC", language).lower()


def caption_cache_key(image_hash: str, output_language: Optional[str] = None) -> str:
    language = normalize_caption_language(output_language)
    return f"{image_hash}::language:{encode_uri_component(language)}" if language else image_hash


# ----------------------------------------------------------------- cache io

async def _read_cache(project_path: str) -> Dict[str, Any]:
    cache_path = f"{normalize_path(project_path)}/{CACHE_REL_PATH}"
    if not await file_exists(cache_path):
        return {}
    try:
        raw = await read_file(cache_path)
        parsed = js_json_loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception as err:
        log.warning("[caption-cache] corrupt cache at %s, starting empty: %s", cache_path, err)
    return {}


async def _write_cache(project_path: str, cache: Dict[str, Any]) -> None:
    pp = normalize_path(project_path)
    cache_path = f"{pp}/{CACHE_REL_PATH}"
    await create_directory(f"{pp}/.llm-wiki")
    await write_file(cache_path, dumps_js_pretty(cache, 2))


def _parse_date_ms(value: Any) -> float:
    """`Date.parse` for the ISO strings this cache writes; NaN-equivalent -> 0."""
    if not isinstance(value, str):
        return 0
    s = js_trim(value)
    try:
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() * 1000
    except ValueError:
        return 0


async def load_caption_cache(project_path: str, output_language: Optional[str] = None) -> Dict[str, str]:
    """image sha256 -> caption from the on-disk cache (empty when absent/corrupt)."""
    cache = await _read_cache(project_path)
    out: Dict[str, str] = {}
    selected_at: Dict[str, float] = {}
    language = normalize_caption_language(output_language)
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        h = entry.get("imageHash")
        if h is None:
            h = key if _SHA256_KEY_RE.fullmatch(key) else None
        if not h:
            continue
        entry_language = normalize_caption_language(entry.get("outputLanguage"))
        if language and entry_language != language:
            continue
        if not language:
            ts = _parse_date_ms(entry.get("capturedAt"))
            if selected_at.get(h, -1) > ts:
                continue
            selected_at[h] = ts
        out[h] = entry.get("caption")
    return out


# ----------------------------------------------------------------- markdown image references

def _normalize_image_destination(destination: str) -> str:
    value = js_trim(destination)
    if value.startswith("<"):
        close = value.find(">")
        if close < 0:
            return ""
        value = js_trim(value[1:close])
    else:
        value = js_trim(_TRAILING_IMAGE_TITLE_RE.sub("", value, count=1))
    if not value:
        return ""
    value = re.sub(r"\\([\\() ])", r"\1", value)
    try:
        return decode_uri(value)
    except ValueError:
        return value


def _find_image_references_u16(md: str) -> List[ImageRef]:
    """`findImageReferences` over a UTF-16 view (see ``to_utf16``); strings stay in that view."""
    out: List[ImageRef] = []
    n = len(md)
    cursor = 0

    def at(i: int) -> str:
        return md[i] if 0 <= i < n else ""

    while cursor < n:
        start = md.find("![", cursor)
        if start < 0:
            break
        alt_end = start + 2
        escaped_alt = False
        while alt_end < n:
            ch = md[alt_end]
            if ch == "\r" or ch == "\n":
                break
            if escaped_alt:
                escaped_alt = False
                alt_end += 1
                continue
            if ch == "\\":
                escaped_alt = True
                alt_end += 1
                continue
            if ch == "]":
                break
            alt_end += 1
        if not md.startswith("](", alt_end):
            cursor = max(start + 2, alt_end + 1)
            continue
        destination_start = alt_end + 2
        end = destination_start
        if at(destination_start) == "<":
            angle_end = destination_start + 1
            while angle_end < n and md[angle_end] not in (">", "\r", "\n"):
                angle_end += 1
            if at(angle_end) != ">":
                cursor = max(destination_start, angle_end + 1)
                continue
            end = angle_end + 1
            while end < n and md[end] != ")" and md[end] not in ("\r", "\n"):
                end += 1
        else:
            depth = 0
            escaped = False
            while end < n:
                ch = md[end]
                if ch == "\r" or ch == "\n":
                    break
                if escaped:
                    escaped = False
                    end += 1
                    continue
                if ch == "\\":
                    escaped = True
                    end += 1
                    continue
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth == 0:
                        break
                    depth -= 1
                end += 1
        if at(end) != ")":
            cursor = max(destination_start, end + 1)
            continue
        destination = md[destination_start:end]
        url = _normalize_image_destination(destination)
        if not url:
            cursor = end + 1
            continue
        full = md[start:end + 1]
        out.append(ImageRef(full=full, alt=md[start + 2:alt_end], url=url, destination=destination,
                            index=start, length=len(full)))
        cursor = end + 1
    return out


def find_image_references(markdown: str) -> List[ImageRef]:
    """Every ``![alt](dest)`` in the markdown; ``index``/``length`` are UTF-16 offsets."""
    return [ImageRef(from_utf16(r.full), from_utf16(r.alt), from_utf16(r.url), from_utf16(r.destination),
                     r.index, r.length) for r in _find_image_references_u16(to_utf16(markdown))]


def strip_markdown_image_references(markdown: str) -> str:
    md = to_utf16(markdown)
    refs = _find_image_references_u16(md)
    if not refs:
        return markdown
    pieces: List[str] = []
    cursor = 0
    for ref in refs:
        pieces.append(md[cursor:ref.index])
        pieces.append(" ")
        cursor = ref.index + ref.length
    pieces.append(md[cursor:])
    return from_utf16("".join(pieces))


def _slice_context(md: str, ref: ImageRef, window: int) -> Dict[str, str]:
    before_start = max(0, ref.index - window)
    before = md[before_start:ref.index]
    after_start = ref.index + ref.length
    after = md[after_start:after_start + window]
    return {"before": from_utf16(before), "after": from_utf16(after)}


# ----------------------------------------------------------------- main entry

def _opt(options: Any, camel: str, snake: str) -> Any:
    if options is None:
        return None
    if isinstance(options, Mapping):
        v = options.get(camel)
        return v if v is not None else options.get(snake)
    v = getattr(options, camel, None)
    return v if v is not None else getattr(options, snake, None)


def _is_aborted(signal: Any) -> bool:
    from .llm_client import is_aborted
    return is_aborted(signal)


def _now_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + "%03dZ" % (now.microsecond // 1000)


async def caption_markdown_images(
    project_path: str,
    markdown: str,
    llm_config: Mapping[str, Any],
    options: Any = None,
) -> CaptionPipelineResult:
    """Caption every distinct image in ``markdown`` (cache by SHA-256 + language) and rewrite alt text."""
    u16 = to_utf16(markdown)
    refs = _find_image_references_u16(u16)
    unchanged = CaptionPipelineResult(markdown, 0, 0, 0)
    if not refs:
        return unchanged

    flt = _opt(options, "shouldCaption", "should_caption") or (lambda url: True)
    target_refs: List[ImageRef] = []
    for r in refs:
        if await _maybe(flt(from_utf16(r.url))):
            target_refs.append(r)
    if not target_refs:
        return unchanged

    if llm_config.get("provider") == "codex-cli":
        log.warning("[caption-pipeline] skipped image captioning: Codex CLI transport does not support "
                    "image input yet.")
        return CaptionPipelineResult(markdown, 0, 0, len(target_refs))

    cache = await _read_cache(project_path)
    counters = {"fresh": 0, "cached": 0, "failed": 0}
    caption_by_url: Dict[str, str] = {}

    unique_refs: List[ImageRef] = []
    seen_urls = set()
    for ref in target_refs:
        if ref.url in seen_urls:
            continue
        seen_urls.add(ref.url)
        unique_refs.append(ref)

    signal = _opt(options, "signal", "signal")
    output_language = _opt(options, "outputLanguage", "output_language")
    url_to_abs_path = _opt(options, "urlToAbsPath", "url_to_abs_path")
    on_progress = _opt(options, "onProgress", "on_progress")
    concurrency = clamp_user_concurrency(_opt(options, "concurrency", "concurrency") or 1)
    total = len(unique_refs)
    pp = normalize_path(project_path)
    state = {"completed": 0, "next": 0}

    async def process_one(ref: ImageRef) -> None:
        url = from_utf16(ref.url)
        if url_to_abs_path:
            abs_path = await _maybe(url_to_abs_path(url))
        else:
            abs_path = url if url.startswith("/") else f"{pp}/wiki/{url}"
        if not abs_path:
            counters["failed"] += 1
            return

        try:
            data = await read_file_as_base64(abs_path)
        except Exception as err:
            log.warning("[caption-pipeline] failed to read %s: %s", abs_path, err)
            counters["failed"] += 1
            return

        h = sha256_of_base64(data["base64"])
        cache_key = caption_cache_key(h, output_language)
        hit = cache.get(cache_key)
        if hit:
            caption_by_url[ref.url] = hit.get("caption") if isinstance(hit, dict) else None
            counters["cached"] += 1
            return

        ctx = _slice_context(u16, ref, CONTEXT_CHARS)
        try:
            caption = await caption_image(
                data["base64"], data["mimeType"], llm_config, signal,
                {"contextBefore": ctx["before"], "contextAfter": ctx["after"],
                 "outputLanguage": output_language})
            entry: Dict[str, Any] = {
                "caption": caption,
                "mimeType": data["mimeType"],
                "model": llm_config.get("model"),
                "capturedAt": _now_iso(),
                "imageHash": h,
            }
            if normalize_caption_language(output_language):
                entry["outputLanguage"] = js_trim(output_language)
            cache[cache_key] = entry
            caption_by_url[ref.url] = caption
            counters["fresh"] += 1
        except Exception as err:
            log.warning("[caption-pipeline] caption failed for %s: %s", abs_path, err)
            counters["failed"] += 1

    async def worker() -> None:
        while True:
            if _is_aborted(signal):
                return
            i = state["next"]
            state["next"] += 1
            if i >= len(unique_refs):
                return
            await process_one(unique_refs[i])
            state["completed"] += 1
            if on_progress:
                await _maybe(on_progress(state["completed"], total))

    await asyncio.gather(*[worker() for _ in range(min(concurrency, len(unique_refs)))])

    if counters["fresh"] > 0:
        try:
            await _write_cache(project_path, cache)
        except Exception as err:
            log.warning("[caption-pipeline] failed to persist cache: %s", err)

    pieces: List[str] = []
    cursor = 0
    for ref in refs:
        caption = caption_by_url.get(ref.url)
        pieces.append(u16[cursor:ref.index])
        cursor = ref.index + ref.length
        if not caption:
            pieces.append(ref.full)
            continue
        safe = re.sub(r"[\r\n]+", " ", caption)
        safe = safe.replace("]", ")")
        safe = re.sub(r"\\+\Z", "", safe)
        safe = js_trim(safe)
        pieces.append(f"![{safe}]({ref.destination})")
    pieces.append(u16[cursor:])
    return CaptionPipelineResult(from_utf16("".join(pieces)), counters["fresh"], counters["cached"],
                                 counters["failed"])
