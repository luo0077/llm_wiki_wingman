"""Pure (no I/O, no store, no LLM) helpers of the LLM Wiki ingest pipeline.

Faithful Python port of the *pure* parts of ``src/lib/ingest.ts``.

Boundary
--------
This module owns everything that is a deterministic function of its
arguments:

* module constants / budgets (``compute_ingest_*``)
* prompt builders (``build_analysis_prompt``, ``build_generation_prompt``,
  ``build_review_suggestion_prompt``, ``build_truncated_file_repair_prompt``,
  ``build_chunk_analysis_system_prompt``, ``build_chunk_analysis_user_prompt``,
  ``build_page_merge_system_prompt``, ``build_page_merge_user_message``,
  ``language_rule``)
* FILE / REVIEW block parsing (``parse_file_blocks``, ``parse_review_blocks``,
  ``is_safe_ingest_path``, ``filter_truncated_file_repair_output``)
* path / title rewriting, sources canonicalisation, bounded index sections,
  date stamping, fallback source summary, deterministic ingest / warning log
  text, media-ref helpers, semantic chunk splitting, FNV hash, checkpoint
  path / compatibility helpers and the pure pieces of long-source planning.

It deliberately does NOT contain: ``auto_ingest`` / ``start_ingest`` /
``execute_ingest_writes`` and anything that reads or writes files, touches the
activity / review / chat stores, calls the LLM (``stream_chat``), extracts or
captions images, runs MinerU, migrates legacy source summaries, embeds pages,
or loads / saves checkpoints.  Those are the pipeline author's job and import
from here.

JavaScript fidelity notes
-------------------------
* JS ``String.length`` / ``slice`` count UTF-16 code units; this module uses
  :func:`len16` / :func:`slice16` wherever TS compared or sliced lengths.
* JS ``\\s`` / ``trim()`` use a different whitespace set from Python; the
  :data:`JS_WS` class and :func:`js_trim` reproduce the JS one.
* JS ``.`` excludes ``\\n \\r \\u2028 \\u2029``; ``\\w`` / ``\\d`` are ASCII;
  ``$`` without ``/m`` is end-of-string only.  Patterns below compile with
  ``re.ASCII`` and explicit classes accordingly.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote, unquote

from ._js import Record
from .context_budget import compute_context_budget
from .frontmatter import parse_frontmatter
from .output_language import build_language_directive
from .path_utils import get_file_name, normalize_path
from .research_query_grounding import PROJECT_LOCAL_TERM_QUERY_RULES
from .source_identity import source_reference_identity
from .sources_merge import parse_sources, write_sources
from .wiki_filename import make_query_slug
from .wiki_page_types import GENERATION_WIKI_TYPES
from .detect_language import detect_language
from .language_metadata import same_script_family

logger = logging.getLogger("llm_wiki.ingest")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LONG_SOURCE_MIN_BUDGET = 8_000
LONG_SOURCE_MAX_SINGLE_PASS_BUDGET = 300_000
LONG_SOURCE_CHUNK_MIN = 12_000
LONG_SOURCE_CHUNK_MAX = 60_000
LONG_SOURCE_DIGEST_MAX = 15_000
LONG_SOURCE_CHUNK_ANALYSIS_MAX = 40_000
INGEST_GENERATION_TOKENS_DEFAULT = 8_192
INGEST_GENERATION_TOKENS_128K = 16_384
INGEST_GENERATION_TOKENS_256K = 24_576
INGEST_GENERATION_TOKENS_512K = 32_768
INGEST_ANALYSIS_TOKENS_MIN = 4_096
INGEST_ANALYSIS_TOKENS_MAX = 8_192
CONSERVATIVE_CHARS_PER_OUTPUT_TOKEN = 4
REVIEW_STAGE_MIN_SIGNAL_CHARS = 10_000
REVIEW_STAGE_MIN_FILE_BLOCKS = 4
AGGREGATE_WIKI_PATHS: Tuple[str, ...] = ("wiki/index.md", "wiki/overview.md", "wiki/log.md")

CJK_OUTPUT_LANGUAGES = frozenset(["Chinese", "Traditional Chinese", "Japanese", "Korean"])

EMBEDDED_IMAGES_MARKER = "<!-- llm-wiki:embedded-images -->"


class NonRetryableIngestError(Exception):
    """Ingest failure that the queue must not retry."""

    non_retryable = True
    nonRetryable = True  # TS-compatible alias

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.name = "NonRetryableIngestError"


# ---------------------------------------------------------------------------
# JS-compat string helpers
# ---------------------------------------------------------------------------

_WS_CHARS = (
    "\t\n\v\f\r          "
    "        　﻿"
)
# Regex class equal to JS ``\s``.
JS_WS = "[\t\n\v\f\r    -     　﻿]"
_JS_DOT = "[^\n\r  ]"
_JS_LINE_START = "(?:\\A|(?<=[\n\r  ]))"  # JS ``^`` under the m flag
_JS_LINE_END = "(?=[\n\r  ]|\\Z)"  # JS ``$`` under the m flag
_ASTRAL = re.compile("[\U00010000-\U0010ffff]")
_RE = re.ASCII


def js_trim(s: str) -> str:
    """``String.prototype.trim``."""
    return s.strip(_WS_CHARS)


def js_trim_end(s: str) -> str:
    """``String.prototype.trimEnd``."""
    return s.rstrip(_WS_CHARS)


def len16(s: str) -> int:
    """JS ``s.length`` (UTF-16 code units)."""
    if not _ASTRAL.search(s):
        return len(s)
    return len(s.encode("utf-16-le", "surrogatepass")) // 2


def slice16(s: str, start: int = 0, end: Optional[int] = None) -> str:
    """JS ``s.slice(start, end)`` on UTF-16 code units.

    A slice that cuts a surrogate pair yields U+FFFD (what any UTF-8 write of
    the JS result would produce) instead of an unencodable lone surrogate.
    """
    if not _ASTRAL.search(s):
        return s[start:end]
    raw = s.encode("utf-16-le", "surrogatepass")
    n = len(raw) // 2
    a = max(n + start, 0) if start < 0 else min(start, n)
    if end is None:
        b = n
    else:
        b = max(n + end, 0) if end < 0 else min(end, n)
    if a >= b:
        return ""
    return raw[2 * a: 2 * b].decode("utf-16-le", "replace")


def _js_num_str(value: Any) -> str:
    """Render a number the way ``${n}`` does in JS."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a field from a mapping or object trying several spellings."""
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return default


def _frontmatter_dict(content: str) -> Optional[Dict[str, Any]]:
    result = parse_frontmatter(content)
    fm = _get(result, "frontmatter")
    return fm if fm else None


def _frontmatter_title(content: str) -> Any:
    fm = _frontmatter_dict(content)
    return fm.get("title") if fm else None


def _clamp_number(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _compute_budget(max_context_size: Optional[int]) -> Tuple[int, int]:
    """Return ``(max_ctx, response_reserve)`` from ``compute_context_budget``."""
    budget = compute_context_budget(max_context_size)
    max_ctx = _get(budget, "maxCtx", "max_ctx")
    reserve = _get(budget, "responseReserve", "response_reserve")
    return max_ctx, reserve


# ---------------------------------------------------------------------------
# Small image / media helpers
# ---------------------------------------------------------------------------


def append_saved_image_refs_for_caption(content: str, images: Sequence[Any]) -> str:
    """``images`` items expose ``relPath`` (dict key or attr ``rel_path``)."""
    if len(images) == 0:
        return content
    refs = [
        f"![]({rel})"
        for rel in (_get(img, "relPath", "rel_path") for img in images)
        if rel
    ]
    if len(refs) == 0:
        return content
    return f"{content}\n\n## Referenced Local Images\n\n" + "\n".join(refs) + "\n"


def ingest_analysis_request(config: Any) -> Dict[str, Any]:
    """Return ``{"maxTokens": int, "reasoning": {...}}`` for analysis calls.

    ``config`` is an LlmConfig (dict with TS camelCase keys).  Imports the
    reasoning resolver lazily so this module stays importable on its own.
    """
    from .reasoning_capabilities import resolve_ingest_reasoning

    content_tokens = compute_ingest_analysis_max_tokens(_get(config, "maxContextSize", "max_context_size"))
    requested = resolve_ingest_reasoning(config)
    requested = dict(requested) if isinstance(requested, Mapping) else {
        k: getattr(requested, k) for k in ("mode", "budgetTokens") if hasattr(requested, k)
    }
    if requested.get("mode") == "custom":
        reasoning = dict(requested)
        reasoning["budgetTokens"] = max(
            1_024, min(requested.get("budgetTokens") or 0, INGEST_ANALYSIS_TOKENS_MAX)
        )
    else:
        reasoning = requested
    mode = reasoning.get("mode")
    if mode == "low":
        reasoning_tokens = 1_024
    elif mode == "medium":
        reasoning_tokens = 4_096
    elif mode in ("high", "max"):
        reasoning_tokens = 8_192
    elif mode == "custom":
        reasoning_tokens = reasoning.get("budgetTokens") or 0
    else:
        reasoning_tokens = 0
    return {"maxTokens": content_tokens + reasoning_tokens, "reasoning": reasoning}


def is_saved_image_prompt_url(project_path: str, source_summary_slug: str, url: str) -> bool:
    return url.startswith(f"{project_path}/wiki/media/{source_summary_slug}/") or url.startswith(
        f"media/{source_summary_slug}/"
    )


def prompt_image_url_to_abs(project_path: str, url: str) -> str:
    return f"{project_path}/wiki/{url}" if url.startswith("media/") else url


def image_mime_type_from_path(path: str) -> str:
    ext = get_file_name(path).split(".")[-1].lower()
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "svg": "image/svg+xml",
        "tif": "image/tiff",
        "tiff": "image/tiff",
    }.get(ext, "application/octet-stream")


def strip_wiki_media_abs_paths(project_path: str, content: str) -> str:
    return content.replace(f"{project_path}/wiki/media/", "media/")


_MEDIA_LINK_RE = re.compile(r"(\]\()\.?/?media/", _RE)
_MEDIA_SRC_RE = re.compile(r"""(\bsrc=["'])\.?/?media/""", _RE | re.IGNORECASE)


def source_summary_media_refs_for_external_markdown(content: str) -> str:
    content = _MEDIA_LINK_RE.sub(lambda m: m.group(1) + "../media/", content)
    return _MEDIA_SRC_RE.sub(lambda m: m.group(1) + "../media/", content)


def to_source_summary_image_ref(rel_path: str) -> str:
    normalized = re.sub(r"^\./", "", rel_path)
    return f"../{normalized}" if normalized.startswith("media/") else rel_path


def encode_markdown_path_segment(segment: str) -> str:
    """``encodeURIComponent`` plus percent-encoding of ``!'()*``."""
    return quote(segment, safe="")


def has_mineru_image_refs(content: str, source_summary_slug: str) -> bool:
    return (
        f"media/{source_summary_slug}/mineru/" in content
        or f"media/{encode_markdown_path_segment(source_summary_slug)}/mineru/" in content
    )


def _decode_uri_component(value: str) -> Optional[str]:
    """``decodeURIComponent``; ``None`` where JS would throw URIError."""
    if re.search(r"%(?![0-9A-Fa-f]{2})", value):
        return None
    try:
        return unquote(value, errors="strict")
    except UnicodeDecodeError:
        return None


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(((?:[^()]|\([^()]*\))*)\)", _RE)
_TITLE_SPLIT_RE = re.compile(f"{JS_WS}+[\"']", _RE)


def extract_mineru_image_rel_paths(markdown: str, source_summary_slug: str) -> List[str]:
    """Pure part of ``savedImagesFromMineruMarkdown``: ordered, de-duplicated
    ``media/<slug>/mineru/...`` rel paths referenced by Markdown images."""
    prefix = f"media/{source_summary_slug}/mineru/"
    encoded_prefix = f"media/{encode_markdown_path_segment(source_summary_slug)}/mineru/"
    refs: List[str] = []
    seen = set()
    for match in _MD_IMAGE_RE.finditer(markdown):
        raw_target = js_trim(match.group(1) or "")
        if raw_target.startswith("<") and ">" in raw_target:
            url = raw_target[1: raw_target.index(">")]
        else:
            url = _TITLE_SPLIT_RE.split(raw_target)[0]
        if not url:
            continue
        decoded = _decode_uri_component(url)
        if decoded is None:
            decoded = url
        normalized = normalize_path(re.sub(r"^\./", "", decoded))
        if not normalized.startswith(prefix) and not normalized.startswith(encoded_prefix):
            continue
        if normalized.startswith(encoded_prefix):
            rel_path = f"media/{source_summary_slug}/mineru/{normalized[len(encoded_prefix):]}"
        else:
            rel_path = normalized
        if rel_path in seen:
            continue
        seen.add(rel_path)
        refs.append(rel_path)
    return refs


_EMBEDDED_IMAGES_RE = re.compile(
    "\n*" + re.escape(EMBEDDED_IMAGES_MARKER) + r"[\s\S]*?" + re.escape(EMBEDDED_IMAGES_MARKER) + "\n*"
)


def wrap_embedded_images_section(section: str) -> str:
    """``\\n\\n<marker>\\n<section.trim()>\\n<marker>\\n``."""
    return f"\n\n{EMBEDDED_IMAGES_MARKER}\n{js_trim(section)}\n{EMBEDDED_IMAGES_MARKER}\n"


def strip_embedded_images_section(existing: str) -> str:
    """Remove any prior paired-marker image injection."""
    return _EMBEDDED_IMAGES_RE.sub("", existing)


def build_image_stub_source_summary(source_identity: str, wrapped_section: str, date: str) -> str:
    """Minimal source-summary stub used when the page is missing (``date`` is
    ``YYYY-MM-DD`` in UTC, i.e. ``new Date().toISOString().slice(0, 10)``)."""
    stub = "\n".join(
        [
            "---",
            "type: source",
            f'title: "Source: {source_identity}"',
            f"created: {date}",
            f"updated: {date}",
            f'sources: ["{source_identity}"]',
            "tags: []",
            "related: []",
            "---",
            "",
            f"# Source: {source_identity}",
            "",
        ]
    )
    return stub + wrapped_section


def resolve_caption_config(mm: Mapping[str, Any], main_llm: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve the LLM config the caption pipeline should use (``None`` = off)."""
    if not mm.get("enabled"):
        return None
    if mm.get("useMainLlm"):
        return dict(main_llm)
    return {
        "provider": mm.get("provider"),
        "apiKey": mm.get("apiKey"),
        "model": mm.get("model"),
        "ollamaUrl": mm.get("ollamaUrl"),
        "customEndpoint": mm.get("customEndpoint"),
        "azureApiVersion": mm.get("azureApiVersion"),
        "azureModelFamily": mm.get("azureModelFamily"),
        "apiMode": mm.get("apiMode"),
        "ingestReasoning": main_llm.get("ingestReasoning"),
        "maxContextSize": main_llm.get("maxContextSize"),
    }


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class SourceChunk(Record):
    """Field names are the TS names (camelCase), like the sibling modules."""

    id: str
    index: int
    total: int
    headingPath: str
    overlapBefore: str
    main: str

    @property
    def heading_path(self) -> str:
        return self.headingPath

    @property
    def overlap_before(self) -> str:
        return self.overlapBefore


@dataclass
class LongSourcePlan(Record):
    chunked: bool
    analysis: str
    sourceContext: str
    checkpointPath: Optional[str] = None


# ---------------------------------------------------------------------------
# FILE block parsing
# ---------------------------------------------------------------------------

# Legacy export kept for diagnostics; the live pipeline uses parse_file_blocks().
FILE_BLOCK_REGEX = re.compile(
    f"---FILE:{JS_WS}*([^\n]+?){JS_WS}*---\n([\\s\\S]*?)---END FILE---", _RE
)


@dataclass
class ParsedFileBlock(Record):
    path: str
    content: str


@dataclass
class ParseFileBlocksResult(Record):
    blocks: List[ParsedFileBlock]
    warnings: List[str]
    truncatedPaths: List[str]

    @property
    def truncated_paths(self) -> List[str]:
        return self.truncatedPaths


_OPENER_LINE = re.compile(
    f"^---{JS_WS}*FILE:{JS_WS}*({_JS_DOT}+?){JS_WS}*---{JS_WS}*\\Z", _RE | re.IGNORECASE
)
_CLOSER_LINE = re.compile(f"^---{JS_WS}*END{JS_WS}+FILE{JS_WS}*---{JS_WS}*\\Z", _RE | re.IGNORECASE)
_FENCE_LINE = re.compile(f"^{JS_WS}{{0,3}}(```+|~~~+)", _RE)
_CONTROL_RE = re.compile("[\x00-\x1f]")
_WIN_INVALID_RE = re.compile('[<>:"|?*]')
_WIN_RESERVED_COM = re.compile(r"^COM[1-9]\Z")
_WIN_RESERVED_LPT = re.compile(r"^LPT[1-9]\Z")


def _is_windows_safe_path_segment(segment: str) -> bool:
    if len(segment) == 0:
        return False
    if _WIN_INVALID_RE.search(segment):
        return False
    if segment.endswith(" ") or segment.endswith("."):
        return False
    stem = segment.split(".")[0].upper()
    if not stem:
        return False
    if stem in ("CON", "PRN", "AUX", "NUL") or _WIN_RESERVED_COM.match(stem) or _WIN_RESERVED_LPT.match(stem):
        return False
    return True


def is_safe_ingest_path(p: Any) -> bool:
    """Reject FILE block paths that escape ``wiki/`` (see ingest.ts for the
    threat model)."""
    if not isinstance(p, str) or len(js_trim(p)) == 0:
        return False
    if _CONTROL_RE.search(p):
        return False
    if p.startswith("/") or p.startswith("\\"):
        return False
    if re.match(r"^[a-zA-Z]:", p):
        return False
    normalized = p.replace("\\", "/")
    segments = normalized.split("/")
    if any(seg == ".." for seg in segments):
        return False
    if any(not _is_windows_safe_path_segment(seg) for seg in segments):
        return False
    if not normalized.startswith("wiki/"):
        return False
    return True


def parse_file_blocks(text: str) -> ParseFileBlocksResult:
    """Parse an LLM stage-2 generation into FILE blocks (CRLF-safe,
    marker-tolerant, fence-aware, truncation- and traversal-guarded)."""
    normalized = text.replace("\r\n", "\n")
    lines = normalized.split("\n")

    blocks: List[ParsedFileBlock] = []
    warnings: List[str] = []
    truncated_paths: List[str] = []

    i = 0
    while i < len(lines):
        opener = _OPENER_LINE.match(lines[i])
        if not opener:
            i += 1
            continue
        path = js_trim(opener.group(1))
        i += 1

        content_lines: List[str] = []
        fence_marker: Optional[str] = None
        fence_len = 0
        closed = False

        while i < len(lines):
            line = lines[i]
            fence = _FENCE_LINE.match(line)
            if fence:
                run = fence.group(1)
                ch = run[0]
                n = len(run)
                if fence_marker is None:
                    fence_marker = ch
                    fence_len = n
                elif ch == fence_marker and n >= fence_len:
                    fence_marker = None
                    fence_len = 0
                content_lines.append(line)
                i += 1
                continue
            if fence_marker is None and _CLOSER_LINE.match(line):
                closed = True
                i += 1
                break
            content_lines.append(line)
            i += 1

        if not closed:
            path_label = path or "(unnamed)"
            msg = (
                f'FILE block "{path_label}" was not closed before end of stream — likely truncation '
                "(model hit max_tokens, timeout, or connection dropped). Block dropped."
            )
            logger.warning("[ingest] %s", msg)
            warnings.append(msg)
            if is_safe_ingest_path(path):
                truncated_paths.append(path)
            continue

        if not path:
            msg = "FILE block with empty path skipped (LLM omitted the path after `---FILE:`)."
            logger.warning("[ingest] %s", msg)
            warnings.append(msg)
            continue

        if not is_safe_ingest_path(path):
            msg = (
                f'FILE block with unsafe path "{path}" rejected (must be under wiki/, no .., '
                "no absolute paths, and Windows-safe file names)."
            )
            logger.warning("[ingest] %s", msg)
            warnings.append(msg)
            continue

        blocks.append(ParsedFileBlock(path=path, content="\n".join(content_lines)))

    return ParseFileBlocksResult(blocks=blocks, warnings=warnings, truncatedPaths=truncated_paths)


def language_rule(source_content: str = "", output_language: Optional[str] = None) -> str:
    """Language directive for ingest prompts.

    ``output_language`` is the project's configured output language (the TS
    store's ``outputLanguage``); ``None`` / ``""`` / ``"auto"`` means detect
    from ``source_content``.
    """
    return build_language_directive(source_content, output_language)


# ---------------------------------------------------------------------------
# Deterministic logs
# ---------------------------------------------------------------------------


def _iso_string(at: Optional[datetime]) -> str:
    """``Date.prototype.toISOString`` (UTC, millisecond precision)."""
    if at is None:
        at = datetime.now(timezone.utc)
    if at.tzinfo is None:
        at = at.astimezone()
    at = at.astimezone(timezone.utc)
    return at.strftime("%Y-%m-%dT%H:%M:%S.") + f"{at.microsecond // 1000:03d}Z"


def format_ingest_warning_log_entry(
    source_identity: str, warnings: Sequence[str], at: Optional[datetime] = None
) -> str:
    return "\n".join(
        [f"## {_iso_string(at)} | {source_identity}", ""]
        + [f"{index + 1}. {warning}" for index, warning in enumerate(warnings)]
        + [""]
    )


def build_ingest_warning_log_text(
    existing: str, source_identity: str, warnings: Sequence[str], at: Optional[datetime] = None
) -> str:
    """New content of ``.llm-wiki/ingest-warnings.log`` after appending an
    entry (pure part of ``appendIngestWarningLog``)."""
    entry = js_trim_end(format_ingest_warning_log_entry(source_identity, warnings, at))
    sep = "\n\n" if js_trim(existing) else ""
    return f"{js_trim_end(existing)}{sep}{entry}\n"


def current_wiki_date(now: Optional[datetime] = None) -> str:
    """Local calendar date ``YYYY-MM-DD`` (JS ``getFullYear/Month/Date``)."""
    if now is None:
        now = datetime.now()
    elif now.tzinfo is not None:
        now = now.astimezone()
    return f"{now.year}-{now.month:02d}-{now.day:02d}"


def build_deterministic_ingest_log(existing: str, source_identity: str, date: Optional[str] = None) -> str:
    if date is None:
        date = current_wiki_date()
    entry = f"## [{date}] ingest | {source_identity}"
    if js_trim(existing):
        return f"{js_trim_end(existing)}\n\n{entry}\n"
    return f"# Wiki Log\n\n{entry}\n"


def build_fallback_source_summary(source_identity: str, analysis: str, date: str) -> str:
    return "\n".join(
        [
            "---",
            "type: source",
            f'title: "Source: {source_identity}"',
            f"created: {date}",
            f"updated: {date}",
            f'sources: ["{source_identity}"]',
            "tags: []",
            "related: []",
            "---",
            "",
            f"# Source: {source_identity}",
            "",
            analysis or "(Analysis not available)",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Date stamping
# ---------------------------------------------------------------------------

_FM_RE = re.compile(
    f"^(---{JS_WS}*\r?\n)([\\s\\S]*?)(\r?\n---{JS_WS}*(?:\r?\n|\\Z))", _RE
)


def _set_or_append_frontmatter_date(payload: str, key: str, date: str) -> str:
    line_re = re.compile(f"(\\A|\n)({key}{JS_WS}*:{JS_WS}*)[^\n\r]*", _RE | re.IGNORECASE)
    if line_re.search(payload):
        return line_re.sub(lambda m: f"{m.group(1)}{m.group(2)}{date}", payload, count=1)
    return f"{js_trim_end(payload)}\n{key}: {date}"


def stamp_generated_frontmatter_dates(content: str, date: str) -> str:
    match = _FM_RE.match(content)
    if not match:
        return content
    payload = match.group(2)
    payload = _set_or_append_frontmatter_date(payload, "created", date)
    payload = _set_or_append_frontmatter_date(payload, "updated", date)
    return f"{match.group(1)}{payload}{match.group(3)}{content[match.end():]}"


_YYYY_RE = re.compile(r"\bYYYY-MM-DD\b", _RE)
_LOG_HEADING_RE = re.compile(
    f"{_JS_LINE_START}({JS_WS}*##{JS_WS}*\\[?)\\d{{4}}-\\d{{2}}-\\d{{2}}(\\]?)", _RE
)


def stamp_generated_log_date(content: str, date: str) -> str:
    normalized = _YYYY_RE.sub(lambda _m: date, content)
    if _LOG_HEADING_RE.search(normalized):
        return _LOG_HEADING_RE.sub(
            lambda m: f"{m.group(1)}{date}{m.group(2)}", normalized, count=1
        )
    return normalized


# ---------------------------------------------------------------------------
# Language guard, path rewriting, index sections, sources
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", _RE)
_MATH_BLOCK_RE = re.compile(r"\$\$[\s\S]*?\$\$", _RE)
_MATH_INLINE_RE = re.compile(r"\$[^$\n]*\$", _RE)
_DISTINCT_NON_LATIN = frozenset(["Arabic", "Persian", "Hindi", "Thai", "Hebrew"])


def content_matches_target_language(content: str, target: str) -> bool:
    """Per-file language guard (CJK vs. non-CJK families)."""
    fm_end = content.find("\n---\n", 3)
    body = content[fm_end + 5:] if fm_end > 0 else content
    body = _CODE_FENCE_RE.sub("", body)
    body = _MATH_BLOCK_RE.sub("", body)
    body = _MATH_INLINE_RE.sub("", body)
    sample = slice16(body, 0, 1500)
    if len16(js_trim(sample)) < 20:
        return True

    detected = detect_language(sample)
    target_is_cjk = target in CJK_OUTPUT_LANGUAGES
    detected_is_cjk = detected in CJK_OUTPUT_LANGUAGES
    if target_is_cjk:
        return detected_is_cjk
    if target in _DISTINCT_NON_LATIN:
        return detected == target
    if detected in _DISTINCT_NON_LATIN:
        return bool(same_script_family(target, detected))
    return not detected_is_cjk


def is_log_path(relative_path: str) -> bool:
    return relative_path == "wiki/log.md" or relative_path.endswith("/log.md")


def is_listing_path(relative_path: str) -> bool:
    return (
        relative_path == "wiki/index.md"
        or relative_path.endswith("/index.md")
        or relative_path == "wiki/overview.md"
        or relative_path.endswith("/overview.md")
    )


def is_app_managed_aggregate_path(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/").lower()
    return normalized == "wiki/index.md" or normalized == "wiki/overview.md"


_CJK_RE = re.compile("[㐀-鿿぀-ヿ가-힯]")
_H1_RE = re.compile(f"{_JS_LINE_START}#{JS_WS}+({_JS_DOT}+){_JS_LINE_END}", _RE)


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def extract_generated_page_title(content: str) -> Optional[str]:
    title = _frontmatter_title(content)
    if isinstance(title, str) and js_trim(title):
        return js_trim(title)
    m = _H1_RE.search(content)
    heading = js_trim(m.group(1)) if m else None
    return heading or None


def rewrite_ingest_path_from_title_for_target_language(
    relative_path: str, content: str, target_lang: Optional[str]
) -> str:
    title = extract_generated_page_title(content)
    if not target_lang or target_lang == "auto":
        should_use_cjk_filename = bool(title and contains_cjk(title))
    else:
        should_use_cjk_filename = target_lang in CJK_OUTPUT_LANGUAGES
    if not should_use_cjk_filename:
        return relative_path
    if is_log_path(relative_path) or is_listing_path(relative_path) or relative_path.startswith("wiki/sources/"):
        return relative_path
    if not title or not contains_cjk(title):
        return relative_path

    slash = relative_path.rfind("/")
    directory = relative_path[: slash + 1] if slash >= 0 else ""
    file_name = relative_path[slash + 1:] if slash >= 0 else relative_path
    if contains_cjk(file_name):
        return relative_path

    slug = make_query_slug(title)
    if not contains_cjk(slug):
        return relative_path
    next_path = f"{directory}{slug}.md"
    return next_path if is_safe_ingest_path(next_path) else relative_path


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]", _RE)


def normalize_index_target(target: str) -> str:
    out = normalize_path(target)
    out = re.sub(r"^wiki/", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\.md\Z", "", out, flags=re.IGNORECASE)
    return out.lower()


def known_index_targets(index: str) -> set:
    """Normalised ``[[target]]`` set already present in ``wiki/index.md``."""
    return {normalize_index_target(m.group(1)) for m in _WIKILINK_RE.finditer(index)}


def index_addition_line(path: str, content: str) -> str:
    """``- [[target]] — title`` line for a written page (``path`` is
    ``wiki/...md``; ``content`` its text, ``""`` if unreadable)."""
    target = re.sub(r"\.md\Z", "", re.sub(r"^wiki/", "", path), flags=re.IGNORECASE)
    fm_title = _frontmatter_title(content)
    if isinstance(fm_title, str):
        title = js_trim(fm_title)
    else:
        title = re.sub(r"\.md\Z", "", get_file_name(path), flags=re.IGNORECASE)
    return f"- [[{target}]] — {title}"


def index_candidate_paths(written_paths: Iterable[str]) -> List[str]:
    """Pure filter of ``updateWikiIndexDeterministically``: unique normalised
    ``wiki/**.md`` paths excluding aggregate pages (order preserved)."""
    seen: Dict[str, None] = {}
    for p in written_paths:
        seen.setdefault(normalize_path(p), None)
    return [
        p for p in seen
        if p.startswith("wiki/") and p.endswith(".md") and p not in AGGREGATE_WIKI_PATHS
    ]


_H2_START_RE = re.compile(f"^##{JS_WS}+", _RE)
_BULLET_RE = re.compile(f"^-{JS_WS}+", _RE)


def update_bounded_recent_index_section(index: str, additions: Sequence[str]) -> str:
    section = "## Recently Updated"
    lines = js_trim_end(index).split("\n")
    start = next((i for i, line in enumerate(lines) if js_trim(line) == section), -1)
    prefix = lines[:start] if start >= 0 else lines
    section_end = -1
    if start >= 0:
        section_end = next(
            (i for i, line in enumerate(lines) if i > start and _H2_START_RE.match(line)), -1
        )
    if start >= 0:
        stop = section_end if section_end >= 0 else None
        existing = [line for line in lines[start + 1: stop] if _BULLET_RE.match(line)]
    else:
        existing = []
    suffix = lines[section_end:] if section_end >= 0 else []
    recent = list(dict.fromkeys([*additions, *existing]))[:200]
    return "\n".join([*prefix, "", section, *recent, *(["", *suffix] if suffix else []), ""])


def _is_valid_source_reference(source: str, active_source_identity: str) -> bool:
    normalized = re.sub(r"^(?:\./)+", "", normalize_path(source))
    key = normalized.lower()
    identity_key = normalize_path(active_source_identity).lower()
    if not normalized or normalized.startswith("/") or re.match(r"^[a-z]:/", normalized, re.IGNORECASE | _RE):
        return False
    if any(part == ".." for part in normalized.split("/")):
        return False
    if source_reference_identity(normalized).lower() == identity_key:
        return True
    if key in ("wiki/index.md", "wiki/overview.md", "wiki/log.md"):
        return False
    if key == ".llm-wiki" or key.startswith(".llm-wiki/"):
        return False
    return True


def canonicalize_sources_field(content: str, source_identity: str) -> str:
    if not content.startswith("---\n"):
        return content

    identity_key = normalize_path(source_identity).lower()
    identity_base_name = get_file_name(source_identity).lower()
    source_values = parse_sources(content)
    canonical_values: List[str] = []
    for source in source_values:
        if not _is_valid_source_reference(source, source_identity):
            continue
        normalized = source_reference_identity(source)
        key = normalized.lower()
        if key == identity_key:
            canonical_values.append(source_identity)
        elif "/" not in normalized and key == identity_base_name:
            canonical_values.append(source_identity)
        else:
            canonical_values.append(normalized)
    if not any(normalize_path(s).lower() == identity_key for s in canonical_values):
        canonical_values.append(source_identity)

    seen = set()
    deduped: List[str] = []
    for source in canonical_values:
        key = normalize_path(source).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return write_sources(content, deduped)


def is_owned_only_by_source(content: str, source_identity: str) -> bool:
    sources = parse_sources(content)
    if len(sources) == 0:
        return False
    expected = source_reference_identity(source_identity).lower()
    return all(source_reference_identity(s).lower() == expected for s in sources)


def unique_normalized_paths(paths: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in paths:
        key = normalize_path(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# REVIEW blocks
# ---------------------------------------------------------------------------

REVIEW_BLOCK_REGEX = re.compile(
    f"---REVIEW:{JS_WS}*(\\w[\\w-]*){JS_WS}*\\|{JS_WS}*({_JS_DOT}+?){JS_WS}*---\n([\\s\\S]*?)---END REVIEW---",
    _RE,
)
_OPTIONS_LINE = re.compile(f"{_JS_LINE_START}OPTIONS:{JS_WS}*({_JS_DOT}+){_JS_LINE_END}", _RE)
_PAGES_LINE = re.compile(f"{_JS_LINE_START}PAGES:{JS_WS}*({_JS_DOT}+){_JS_LINE_END}", _RE)
_SEARCH_LINE = re.compile(f"{_JS_LINE_START}SEARCH:{JS_WS}*({_JS_DOT}+){_JS_LINE_END}", _RE)
_OPTIONS_STRIP = re.compile(f"{_JS_LINE_START}OPTIONS:{_JS_DOT}*{_JS_LINE_END}", _RE)
_PAGES_STRIP = re.compile(f"{_JS_LINE_START}PAGES:{_JS_DOT}*{_JS_LINE_END}", _RE)
_SEARCH_STRIP = re.compile(f"{_JS_LINE_START}SEARCH:{_JS_DOT}*{_JS_LINE_END}", _RE)


def parse_review_blocks(text: str, source_path: str) -> List[Dict[str, Any]]:
    """Return review items (camelCase keys: type, title, description,
    sourcePath, affectedPages?, searchQueries?, options)."""
    items: List[Dict[str, Any]] = []
    for match in REVIEW_BLOCK_REGEX.finditer(text):
        raw_type = js_trim(match.group(1)).lower()
        title = js_trim(match.group(2))
        body = js_trim(match.group(3))

        rtype = (
            raw_type
            if raw_type in ("contradiction", "duplicate", "missing-page", "suggestion")
            else "confirm"
        )

        options_match = _OPTIONS_LINE.search(body)
        if options_match:
            options = []
            for o in options_match.group(1).split("|"):
                label = js_trim(o)
                options.append({"label": label, "action": label})
        else:
            options = [
                {"label": "Approve", "action": "Approve"},
                {"label": "Skip", "action": "Skip"},
            ]

        pages_match = _PAGES_LINE.search(body)
        affected_pages = (
            [js_trim(p) for p in pages_match.group(1).split(",")] if pages_match else None
        )

        search_match = _SEARCH_LINE.search(body)
        search_queries = (
            [q for q in (js_trim(q) for q in search_match.group(1).split("|")) if len(q) > 0]
            if search_match
            else None
        )

        description = body
        description = _OPTIONS_STRIP.sub("", description, count=1)
        description = _PAGES_STRIP.sub("", description, count=1)
        description = _SEARCH_STRIP.sub("", description, count=1)
        description = js_trim(description)

        item: Dict[str, Any] = {
            "type": rtype,
            "title": title,
            "description": description,
            "sourcePath": source_path,
        }
        if affected_pages is not None:
            item["affectedPages"] = affected_pages
        if search_queries is not None:
            item["searchQueries"] = search_queries
        item["options"] = options
        items.append(item)
    return items


_FILE_OPENER_COUNT_RE = re.compile(r"---FILE:" + JS_WS + r"*[^-]+---", _RE)
_REVIEW_OPENER_RE = re.compile(r"---REVIEW:" + JS_WS + r"*[\w-]+" + JS_WS + r"*\|", _RE | re.IGNORECASE)


def count_file_blocks(text: str) -> int:
    return len(_FILE_OPENER_COUNT_RE.findall(text))


def should_run_dedicated_review_stage(generation: str) -> bool:
    return (
        len16(generation) >= REVIEW_STAGE_MIN_SIGNAL_CHARS
        or count_file_blocks(generation) >= REVIEW_STAGE_MIN_FILE_BLOCKS
        or bool(_REVIEW_OPENER_RE.search(generation))
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ENTITY_ANALYSIS_RULES: Tuple[str, ...] = (
    "Apply a standalone-page significance threshold: page-worthy subjects are central to the source, recur meaningfully, carry substantive claims, or are reusable across the wiki.",
    "Do not mark passing mentions, ordinary story props, unnamed or generic-role people, chapter/part labels, or internal workflow artifacts as page-worthy unless the source makes them independently important knowledge subjects.",
    "For every candidate, output `Page-worthy: yes` or `Page-worthy: no` plus one short justification. Keep non-page-worthy candidates only as context for the source summary.",
    "Canonical naming priority: reuse the exact title of an existing wiki page first; otherwise preserve the source spelling. Widely used localized names may be recorded as aliases, but never invent or phonetically guess a name.",
)

ENTITY_PAGE_RULES: Tuple[str, ...] = (
    "Create a standalone entity page only for a candidate explicitly marked `Page-worthy: yes`. Keep other mentions as plain prose in the source summary, or as wikilinks only when the target already exists in the current wiki index.",
    "Canonical naming priority: reuse the exact title of an existing wiki page first; otherwise preserve the source spelling. A widely used localized name may be an alias, but never invent or phonetically guess a name.",
    "The opening paragraph of every entity page must answer what the subject is, not merely describe its role in this source.",
    "Prefer an identity supported by the source or existing wiki. If one short identity sentence must use stable, high-confidence public knowledge to identify a well-known subject, clearly label it as independently verifiable background in the mandatory output language so it is not attributed to the imported source. Never add uncertain details; when neither the source nor existing wiki establishes an unambiguous identity, do not create the standalone page.",
)

LONG_SOURCE_ENTITY_RULES: Tuple[str, ...] = (
    "In each Chunk Analysis, record named-subject mentions and identity evidence without making a final standalone-page decision from one chunk alone.",
    "In Updated Global Digest > Entities, accumulate recurrence across chunks and output for each candidate: canonical name, identity, supporting chunks, significance evidence, and `Page-worthy: yes` or `Page-worthy: no`.",
    *ENTITY_ANALYSIS_RULES,
)


def _join_nonempty(parts: Iterable[str]) -> str:
    """``[...].filter(Boolean).join("\\n")``."""
    return "\n".join(p for p in parts if p)


def build_analysis_prompt(
    purpose: str,
    index: str,
    source_content: str = "",
    schema: str = "",
    *,
    output_language: Optional[str] = None,
) -> str:
    """Step 1 prompt: AI reads the source and produces a structured analysis."""
    return _join_nonempty([
        "You are an expert research analyst. Read the source document and produce a structured analysis.",
        "Do not output chain-of-thought, hidden reasoning, or a thinking transcript. Reason internally and write only the concise final analysis.",
        "",
        language_rule(source_content, output_language),
        "",
        "Your analysis should cover:",
        "",
        "## Key Entities",
        "List candidate named people, organizations, products, datasets, and tools. For each:",
        "- Name and type",
        "- Role and significance in the source",
        "- Whether it likely already exists in the wiki (check the index)",
        *ENTITY_ANALYSIS_RULES,
        "",
        "## Key Concepts",
        "List theories, methods, techniques, phenomena. For each:",
        "- Name and brief definition",
        "- Why it matters in this source",
        "- Whether it likely already exists in the wiki",
        "",
        "## Main Arguments & Findings",
        "- What are the core claims or results?",
        "- What evidence supports them?",
        "- How strong is the evidence?",
        "- Which named subject is each claim about? Do not transfer claims, limits, or evaluations from one entity/model/product/method to another just because they share keywords.",
        "- Preserve structured source data verbatim in the analysis when present: include SQL DDL / CREATE TABLE statements, schema definitions, API signatures, configuration, and tables in fenced code blocks or Markdown tables. Do not reduce exact field names, types, constraints, keys, or indexes to prose.",
        "",
        "## Connections to Existing Wiki",
        "- What existing pages does this source relate to?",
        "- Does it strengthen, challenge, or extend existing knowledge?",
        "",
        "## Contradictions & Tensions",
        "- Does anything in this source conflict with existing wiki content?",
        "- Are there internal tensions or caveats?",
        "",
        "## Recommendations",
        "- What wiki pages should be created or updated?",
        "- If the project schema (below) defines page types beyond entity/concept (e.g. goal, habit, reflection, finding, decision, meeting), and the source genuinely contains matching content, recommend pages of those types — name the type explicitly. Only when the source actually supports it; never invent goals/habits/journal entries that aren't in the source.",
        "- What should be emphasized vs. de-emphasized?",
        "- Any open questions worth flagging for the user?",
        "",
        "Be thorough but concise. Focus on what's genuinely important.",
        "",
        "If a folder context is provided, use it as a hint for categorization — the folder structure often reflects the user's organizational intent (e.g., 'papers/energy' suggests the file is an energy-related paper).",
        "",
        f"## Project Schema (page types available — map source content to schema-defined types when it fits)\n{schema}"
        if schema
        else "",
        f"## Wiki Purpose (for context)\n{purpose}" if purpose else "",
        f"## Current Wiki Index (for checking existing content)\n{index}" if index else "",
    ])


def build_generation_prompt(
    schema: str,
    purpose: str,
    index: str,
    source_file_name: str,
    overview: Optional[str] = None,
    source_content: str = "",
    source_summary_path: Optional[str] = None,
    *,
    output_language: Optional[str] = None,
) -> str:
    """Step 2 prompt: AI takes its own analysis and generates wiki files + review items."""
    source_base_name = re.sub(r"\.[^.]+\Z", "", source_file_name)
    summary_path = source_summary_path if source_summary_path is not None else f"wiki/sources/{source_base_name}.md"
    today = current_wiki_date()
    types_joined = " | ".join(GENERATION_WIKI_TYPES)

    return _join_nonempty([
        "You are a wiki maintainer. Based on the analysis provided, generate wiki files.",
        "Do not output chain-of-thought, hidden reasoning, or explanatory preamble. Reason internally and output only the requested FILE/REVIEW blocks.",
        "",
        language_rule(source_content, output_language),
        "",
        f"## IMPORTANT: Source File",
        f"The original source file is: **{source_file_name}**",
        f"All wiki pages generated from this source MUST include this filename in their frontmatter `sources` field.",
        f"Today's date is **{today}**. Use this exact date for all new `created`, `updated`, and wiki/log.md ingest dates.",
        "",
        "\n".join([
            "## Project Schema and Routing (AUTHORITATIVE)",
            schema,
            "",
            "Use this schema as the primary routing rule for page types and directories.",
            "If it defines custom folders or distinctions (for example people, technologies, organizations, methods, or cases), write pages into those schema-defined folders instead of forcing them into wiki/entities/ or wiki/concepts/.",
            "Use wiki/entities/ and wiki/concepts/ only when the schema does not provide a more specific destination.",
            "Every generated page's frontmatter type must match the schema directory used in its FILE path.",
        ])
        if schema
        else "",
        "",
        "## What to generate",
        "",
        f"1. A source summary page at **{summary_path}** (MUST use this exact path)",
        "2. Entity or schema-defined typed pages for key named things identified in the analysis. Prefer schema-defined directories when present; otherwise use wiki/entities/.",
        "3. Concept or schema-defined typed pages for key ideas, methods, techniques, and abstractions. Prefer schema-defined directories when present; otherwise use wiki/concepts/.",
        "4. A log entry for wiki/log.md (just the new entry to append, format: ## [YYYY-MM-DD] ingest | Title)",
        *ENTITY_PAGE_RULES,
        "For consolidated long-document analyses, the Final Global Digest is authoritative for `Page-worthy` decisions; per-chunk mention notes are evidence, not page-creation instructions.",
        "Do not generate wiki/index.md or wiki/overview.md. The application maintains aggregate navigation separately so large wikis are never rewritten through model output.",
        "",
        "## Frontmatter Rules (CRITICAL — parser is strict)",
        "",
        "Every page begins with a YAML frontmatter block. Format rules, in order of importance:",
        "",
        "1. The VERY FIRST line of the file MUST be exactly `---` (three hyphens, nothing else).",
        "   Do NOT wrap the file in a ```yaml ... ``` code fence.",
        "   Do NOT prefix it with a `frontmatter:` key or any other line.",
        "2. Each frontmatter line is a `key: value` pair on its own line.",
        "3. The frontmatter ends with another `---` line on its own.",
        "4. The next line after the closing `---` is the start of the page body.",
        "5. Arrays use the standard YAML inline form `[a, b, c]` (no outer brackets around each item).",
        "   Wikilinks belong in the BODY only — never write `related: [[a]], [[b]]` (invalid YAML);",
        "   write `related: [a, b]` with bare slugs.",
        "",
        "Required fields and types:",
        f"  • type     — one of the known types ({types_joined}), or a custom type explicitly defined by the project schema",
        "  • title    — string (quote it if it contains a colon, e.g. `title: \"Foo: Bar\"`)",
        f"  • created  — {today} for new pages (YYYY-MM-DD, no quotes)",
        f"  • updated  — {today} for new pages (same as created)",
        "  • tags     — array of bare strings: `tags: [microbiology, ai]`",
        "  • related  — array of bare wiki page slugs: `related: [foo, bar-baz]`. Do NOT include",
        "               `wiki/`, `.md`, or `[[…]]` here — slugs only.",
        f"  • sources  — array of source filenames; MUST include \"{source_file_name}\".",
        "",
        "Concrete example of a complete, parseable page (everything between the two `---` lines",
        "is the frontmatter; the heading and prose below are the body):",
        "",
        "    ---",
        "    type: entity",
        "    title: Example Entity",
        f"    created: {today}",
        f"    updated: {today}",
        "    tags: [example, demo]",
        "    related: [related-slug-1, related-slug-2]",
        f"    sources: [\"{source_file_name}\"]",
        "    ---",
        "",
        "    # Example Entity",
        "",
        "    Body content goes here. Use [[wikilink]] syntax in the body for cross-references.",
        "",
        "Other rules:",
        "- Use [[wikilink]] syntax in the BODY for cross-references between pages",
        "- If you include images, use wiki-root-relative paths such as `media/source-slug/image.png`; never output absolute filesystem paths.",
        "- Preserve subject boundaries: when a source discusses multiple entities/models/products/methods, keep claims, evaluations, limitations, benchmark results, and recommendations attached to the exact subject they describe.",
        "- Do not merge or generalize a claim about one subject into another subject's page solely because they share terms (for example context window size, benchmark name, dataset, architecture, or feature name).",
        "- If a page needs to mention another subject for comparison, write it explicitly as a comparison and cite which source/frontmatter `sources` entry supports that statement.",
        "- Use kebab-case for Latin-script filenames; for Chinese/Japanese/Korean titles keep the CJK characters (do NOT romanize to pinyin/romaji or translate to English)",
        "- Derive filenames from the page title in the mandatory output language, but short proper nouns and technical identifiers take precedence: preserve names such as OpenAI, GPT-5, Transformer, CLIP, ImageNet, PyTorch, CUDA, GitHub, arXiv, React, LanceDB, AnyTXT, MinerU, model names, dataset names, tool names, and code identifiers in their standard original form. Do not put raw URLs, citation strings, or full paper titles directly into file paths; convert surrounding descriptive prose to a safe readable title. For Chinese/Japanese/Korean prose titles, keep readable CJK characters in the filename instead of translating the slug to English.",
        "- Preserve structured source data verbatim: copy SQL DDL / CREATE TABLE statements, schema definitions, API signatures, configuration, and tabular data into fenced code blocks (or Markdown tables) in the source summary page instead of paraphrasing them. Exact column names, types, constraints, primary/foreign keys, and indexes must survive ingest — a prose-only summary that drops them loses the structure the user imported the source to keep.",
        "- Follow the analysis recommendations on what to emphasize",
        "- If the analysis found connections to existing pages, add cross-references",
        "",
        "## Review block types",
        "",
        "After all FILE blocks, optionally emit REVIEW blocks for anything that needs human judgment:",
        "",
        "- contradiction: the analysis found conflicts with existing wiki content",
        "- duplicate: an entity/concept might already exist under a different name in the index",
        "- missing-page: an important concept is referenced but has no dedicated page",
        "- suggestion: ideas for further research, related sources to look for, or connections worth exploring",
        "",
        "Only create reviews for things that genuinely need human input. Don't create trivial reviews.",
        "",
        "## OPTIONS allowed values (only these predefined labels):",
        "",
        "- contradiction: OPTIONS: Create Page | Skip",
        "- duplicate: OPTIONS: Create Page | Skip",
        "- missing-page: OPTIONS: Create Page | Skip",
        "- suggestion: OPTIONS: Create Page | Skip",
        "",
        "The user also has a 'Deep Research' button (auto-added by the system) that triggers web search.",
        "Do NOT invent custom option labels. Only use 'Create Page' and 'Skip'.",
        "",
        "For suggestion and missing-page reviews, the SEARCH field must contain 2-3 web search queries",
        "(concise, context-grounded, suitable for a search engine — NOT titles or sentences).",
        *PROJECT_LOCAL_TERM_QUERY_RULES,
        "Example:",
        "  SEARCH: automated technical debt detection AI generated code | software quality metrics LLM code generation | static analysis tools agentic software development",
        "",
        f"## Wiki Purpose\n{purpose}" if purpose else "",
        f"## Current Wiki Index (preserve all existing entries, add new ones)\n{index}" if index else "",
        f"## Current Overview (update this to reflect the new source)\n{overview}" if overview else "",
        "",
        # ── OUTPUT FORMAT MUST BE THE LAST SECTION — models weight recent instructions highest ──
        "## Output Format (MUST FOLLOW EXACTLY — this is how the parser reads your response)",
        "",
        "Your ENTIRE response consists of FILE blocks followed by optional REVIEW blocks. Nothing else.",
        "",
        "FILE block template:",
        "```",
        "---FILE: wiki/path/to/page.md---",
        "(complete file content with YAML frontmatter)",
        "---END FILE---",
        "```",
        "",
        "REVIEW block template (optional, after all FILE blocks):",
        "```",
        "---REVIEW: type | Title---",
        "Description of what needs the user's attention.",
        "OPTIONS: Create Page | Skip",
        "PAGES: wiki/page1.md, wiki/page2.md",
        "SEARCH: query 1 | query 2 | query 3",
        "---END REVIEW---",
        "```",
        "",
        "## Output Requirements (STRICT — deviations will cause parse failure)",
        "",
        "1. The FIRST character of your response MUST be `-` (the opening of `---FILE:`).",
        "2. DO NOT output any preamble such as \"Here are the files:\", \"Based on the analysis...\", or any introductory prose.",
        "3. DO NOT echo or restate the analysis — that was stage 1's job. Your job is to emit FILE blocks.",
        "4. DO NOT output markdown tables, bullet lists, or headings outside of FILE/REVIEW blocks.",
        "5. DO NOT output any trailing commentary after the last `---END FILE---` or `---END REVIEW---`.",
        "6. Between blocks, use only blank lines — no prose.",
        "7. FILE block prose (body, explanations, descriptions, section text) must use the mandatory output language specified below. Preserve proper nouns, acronyms, model names, dataset names, tool/library names, code identifiers, URLs, file names, citation strings, paper titles, and technical terms with no widely-used localized equivalent in their standard original form, including in page names and section headings.",
        "",
        "If you start with anything other than `---FILE:`, the entire response will be discarded.",
        "",
        # Repeat the language directive at the very end so it wins the "most
        # recent instruction" tie-breaker. Small-to-medium models otherwise
        # drift back to their training-data language for individual pages.
        "---",
        "",
        language_rule(source_content, output_language),
    ])


def build_review_suggestion_prompt(
    purpose: str,
    index: str,
    source_identity: str,
    analysis: str,
    source_context: str,
    generation: str,
    max_context_size: Optional[int],
    *,
    output_language: Optional[str] = None,
) -> str:
    max_ctx, _reserve = _compute_budget(max_context_size)
    section_cap = max(4_000, math.floor(max_ctx * 0.15))
    index_cap = max(3_000, math.floor(section_cap * 0.8))
    return _join_nonempty([
        "You are identifying high-value follow-up research items for a personal wiki.",
        "Do not output chain-of-thought, hidden reasoning, or explanatory preamble.",
        "",
        language_rule(source_context, output_language),
        "",
        "Your job is NOT to generate wiki pages. The wiki page generation already happened.",
        "Output only REVIEW blocks for unresolved knowledge gaps that deserve human attention or Deep Research.",
        "",
        "Create REVIEW blocks only for genuinely useful follow-up work:",
        "- missing-page: an important entity/concept is referenced but still lacks a dedicated page",
        "- suggestion: a research question, source type, or comparison that would materially improve the wiki",
        "- contradiction: a conflict or tension that requires user judgment",
        "- duplicate: likely duplicate pages/names that need user review",
        "",
        "Prefer 1-5 high-signal reviews. If there is nothing worth reviewing, output nothing.",
        "For suggestion and missing-page reviews, include a SEARCH line with 2-3 context-grounded web search queries separated by ` | `.",
        *PROJECT_LOCAL_TERM_QUERY_RULES,
        "Use only these options: OPTIONS: Create Page | Skip",
        "",
        "REVIEW block template:",
        "```",
        "---REVIEW: suggestion | Precise title---",
        "Concise description of the gap and why it matters.",
        "OPTIONS: Create Page | Skip",
        "PAGES: wiki/page1.md, wiki/page2.md",
        "SEARCH: query 1 | query 2 | query 3",
        "---END REVIEW---",
        "```",
        "",
        "Return REVIEW blocks only. Do not output FILE blocks. Do not wrap the response in markdown fences.",
        "",
        f"## Wiki Purpose\n{purpose}" if purpose else "",
        f"## Current Wiki Index\n{trim_long_text(index, index_cap)}" if index else "",
        "",
        f"## Source\n{source_identity}",
        "",
        "## Stage 1 Analysis",
        trim_long_text(analysis, section_cap),
        "",
        "## Source Context",
        trim_long_text(source_context, section_cap),
        "",
        "## Generated Wiki Output",
        trim_long_text(generation, section_cap),
    ])


def build_truncated_file_repair_prompt(
    paths: Sequence[str],
    source_identity: str,
    *,
    schema: str,
    purpose: str,
    analysis: str,
    source_context: str,
    max_context_size: Optional[int],
    output_language: Optional[str] = None,
) -> str:
    max_ctx, _reserve = _compute_budget(max_context_size)
    section_cap = max(4_000, math.floor(max_ctx * 0.12))
    return _join_nonempty([
        "You are repairing truncated wiki FILE blocks from an earlier generation.",
        "Return exactly one complete FILE block for each requested path and no other files.",
        "Every block must end with `---END FILE---`. Do not output a preamble, REVIEW blocks, or trailing commentary.",
        "Preserve the requested paths exactly and include the source identity in each page's frontmatter `sources` field.",
        "",
        language_rule(source_context, output_language),
        "",
        "## Requested paths",
        *[f"- {path}" for path in paths],
        "",
        f"## Source identity\n{source_identity}",
        f"## Project schema\n{trim_long_text(schema, section_cap)}" if schema else "",
        f"## Wiki purpose\n{trim_long_text(purpose, section_cap)}" if purpose else "",
        f"## Stage 1 analysis\n{trim_long_text(analysis, section_cap)}",
        f"## Source context\n{trim_long_text(source_context, section_cap)}",
    ])


@dataclass
class FileRepairOutput(Record):
    text: str
    paths: List[str]
    warnings: List[str]


def filter_truncated_file_repair_output(text: str, allowed_paths: Sequence[str]) -> FileRepairOutput:
    allowed = {normalize_path(p) for p in allowed_paths}
    parsed = parse_file_blocks(text)
    warnings = parsed.warnings
    seen = set()
    kept: List[ParsedFileBlock] = []
    dropped: List[ParsedFileBlock] = []
    duplicates: List[ParsedFileBlock] = []
    for block in parsed.blocks:
        path_key = normalize_path(block.path)
        if path_key not in allowed:
            dropped.append(block)
            continue
        if path_key in seen:
            duplicates.append(block)
            continue
        seen.add(path_key)
        kept.append(block)
    if dropped:
        warnings.append(
            f"Dropped {len(dropped)} unrequested FILE block(s) from truncated repair output: "
            + ", ".join(b.path for b in dropped)
        )
    if duplicates:
        warnings.append(
            f"Dropped {len(duplicates)} duplicate FILE block(s) from truncated repair output: "
            + ", ".join(b.path for b in duplicates)
        )
    return FileRepairOutput(
        text="\n\n".join(
            f"---FILE: {b.path}---\n{js_trim_end(b.content)}\n---END FILE---" for b in kept
        ),
        paths=[b.path for b in kept],
        warnings=warnings,
    )


def build_chunk_analysis_system_prompt(
    purpose: str,
    schema: str,
    index: str,
    source_content: str,
    *,
    output_language: Optional[str] = None,
) -> str:
    return _join_nonempty([
        "You are analyzing a long source document for a personal wiki.",
        "Do not output chain-of-thought, hidden reasoning, or a thinking transcript.",
        "Analyze only the current MAIN CHUNK. Use overlap and digest for context only.",
        "Keep stable names consistent with the existing wiki and prior digest.",
        "",
        language_rule(source_content, output_language),
        "",
        "Output exactly two markdown sections:",
        "",
        "## Chunk Analysis",
        "- Concise summary of the main chunk",
        "- New or updated entities",
        "- New or updated concepts",
        "- Any schema-defined page types beyond entity/concept that the main chunk genuinely supports",
        "- Claims, findings, evidence, contradictions",
        "- Exact structured data from this chunk, when present: preserve SQL DDL / CREATE TABLE statements, schema definitions, API signatures, configuration, and tables verbatim in fenced code blocks or Markdown tables; retain field names, types, constraints, keys, and indexes",
        "- Open questions or research gaps",
        "",
        "## Updated Global Digest",
        "A compact document-level digest that incorporates this chunk and preserves prior cross-chunk context.",
        "Keep this digest structured under: Summary, Entities, Concepts, Schema-Typed Candidates, Claims, Evidence, Contradictions, Open Questions, Cross-Chunk Relations.",
        "Entity handling rules:",
        *[f"- {rule}" for rule in LONG_SOURCE_ENTITY_RULES],
        "Use schema-defined types only when the source actually supports them; never invent goals, habits, journal entries, decisions, or similar user-authored records that are not present in the source.",
        "",
        "Stable project context follows. It changes rarely and should be treated as background:",
        f"## Wiki Purpose\n{purpose}" if purpose else "",
        f"## Wiki Schema\n{schema}" if schema else "",
        f"## Current Wiki Index\n{trim_long_text(index, 40_000)}" if index else "",
    ])


def build_chunk_analysis_user_prompt(
    source_identity: str,
    folder_context: Optional[str],
    chunk: SourceChunk,
    global_digest: str,
) -> str:
    return _join_nonempty([
        f"Source file: {source_identity}",
        f"Folder context: {folder_context}" if folder_context else "",
        f"Chunk: {chunk.index}/{chunk.total}",
        f"Heading path: {chunk.heading_path}" if chunk.heading_path else "",
        "",
        "## Current Global Digest",
        global_digest or "(No prior digest yet.)",
        "",
        "## Previous Overlap Context\n" + chunk.overlap_before if chunk.overlap_before else "",
        "",
        "",
        "## MAIN CHUNK TO ANALYZE",
        chunk.main,
        "",
        "",
        "Return only the two requested sections. Do not repeat overlap-only facts unless the main chunk supports them.",
    ])


def build_page_merge_system_prompt() -> str:
    return "\n".join([
        "You are merging source-backed material for the same wiki page into one coherent document.",
        "The inputs are internal containers, not revisions or competing versions.",
        "Either input may mention additional subjects for comparison or context.",
        "",
        "Output ONE merged version that:",
        "- Preserves every factual claim from both inputs (do not drop content)",
        "- Eliminates redundancy when both inputs state the same fact",
        "- Preserves subject/source boundaries: if either input mentions other entities/models/products/methods for comparison, keep those comparisons attribution-exact and do not fold them into claims about the main page subject",
        "- When claims conflict or apply to different subjects, keep them separated; attribute a claim to a real source filename only when the supporting file is unambiguous, otherwise do not guess",
        "- When in doubt whether two similar-looking claims describe the same fact, prefer keeping them separate",
        "- Reorganizes sections so the structure is logical for the merged topic,",
        "  not just a concatenation of the two inputs",
        "- Uses consistent markdown structure (headings, tables, lists, callouts)",
        "- Keeps `[[wikilink]]` references intact",
        "- Never describe the merge inputs as existing/new/original/old versions in headings, prose, notes, or table columns",
        "- Never create comparison sections/tables about the merge inputs themselves",
        "- Never invent URLs, citations, source names, or placeholder references",
        "",
        "Output requirements:",
        "- The FIRST character of your response MUST be `-` (the opening of `---`)",
        "- Output the COMPLETE file: YAML frontmatter + body",
        "- No preamble (no \"Here is the merged version:\"), no analysis prose",
        "- The caller will overwrite `sources`/`tags`/`related`/`updated` with",
        "  deterministic values — your job is the body and any other fields",
    ])


def _sanitize_provenance(value: str) -> str:
    value = re.sub(r"[\r\n#`]+", " ", value)
    value = re.sub(r"---+", " ", value)
    value = re.sub(f"{JS_WS}+", " ", value)
    return slice16(js_trim(value), 0, 240)


def build_page_merge_user_message(existing_content: str, incoming_content: str, source_file_name: str) -> str:
    """User message of the LLM page merger (pure part of ``buildPageMerger``)."""
    safe_source_file_name = _sanitize_provenance(source_file_name)
    fm = _frontmatter_dict(existing_content)
    existing_sources = fm.get("sources") if fm else None
    if isinstance(existing_sources, list):
        existing_provenance = slice16(
            ", ".join(
                s for s in (_sanitize_provenance(x) for x in existing_sources if isinstance(x, str)) if s
            ),
            0,
            1_000,
        )
    else:
        existing_provenance = "sources listed in the existing page frontmatter"
    return "\n".join([
        f"## Previously collected material (provenance: {existing_provenance or 'existing page sources'})",
        "",
        existing_content,
        "",
        "---",
        "",
        f"## Additional material (provenance: {safe_source_file_name or 'additional source'})",
        "",
        incoming_content,
        "",
        "---",
        "",
        "Now output the merged file. Start with `---` on the first line.",
    ])


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def compute_ingest_source_budget(max_context_size: Optional[int], stable_context_length: int) -> int:
    max_ctx, response_reserve = _compute_budget(max_context_size)
    stable_reserve = min(math.floor(max_ctx * 0.25), max(12_000, stable_context_length))
    instruction_reserve = max(12_000, math.floor(max_ctx * 0.08))
    available = max_ctx - response_reserve - stable_reserve - instruction_reserve
    upper = min(LONG_SOURCE_MAX_SINGLE_PASS_BUDGET, max(LONG_SOURCE_MIN_BUDGET, math.floor(max_ctx * 0.6)))
    return _clamp_number(math.floor(available), LONG_SOURCE_MIN_BUDGET, upper)


def compute_ingest_generation_max_tokens(max_context_size: Optional[int]) -> int:
    max_ctx, _ = _compute_budget(max_context_size)
    if max_ctx >= 512_000:
        return INGEST_GENERATION_TOKENS_512K
    if max_ctx >= 256_000:
        return INGEST_GENERATION_TOKENS_256K
    if max_ctx >= 128_000:
        return INGEST_GENERATION_TOKENS_128K
    return INGEST_GENERATION_TOKENS_DEFAULT


def compute_ingest_analysis_max_tokens(max_context_size: Optional[int]) -> int:
    _, response_reserve = _compute_budget(max_context_size)
    reserved_output_tokens = math.floor(response_reserve / CONSERVATIVE_CHARS_PER_OUTPUT_TOKEN)
    return _clamp_number(reserved_output_tokens, INGEST_ANALYSIS_TOKENS_MIN, INGEST_ANALYSIS_TOKENS_MAX)


def compute_ingest_review_max_tokens(max_context_size: Optional[int]) -> int:
    return min(8_192, max(4_096, math.floor(compute_ingest_generation_max_tokens(max_context_size) / 2)))


# ---------------------------------------------------------------------------
# Semantic chunk splitting
# ---------------------------------------------------------------------------

_SENTENCE_PIECES_RE = re.compile("[^.!?。！？\n]+[.!?。！？]?|\n+", _RE)
_HEADING_LINE_RE = re.compile(f"^(#{{1,6}}){JS_WS}+({_JS_DOT}+?){JS_WS}*\\Z", _RE)
_PARAGRAPH_BREAK_RE = re.compile(f"\n{JS_WS}*\n", _RE)
_SENTENCE_BREAK_RE = re.compile(f"[.!?。！？]{JS_WS}+", _RE)


def split_oversized_block(block: str, target_chars: int) -> List[str]:
    if len16(block) <= target_chars * 1.25:
        return [block]

    pieces = _SENTENCE_PIECES_RE.findall(block) or [block]
    out: List[str] = []
    current = ""
    for piece in pieces:
        piece_len = len16(piece)
        if current and len16(current) + piece_len > target_chars:
            out.append(js_trim(current))
            current = ""
        if piece_len > target_chars:
            for i in range(0, piece_len, target_chars):
                sl = js_trim(slice16(piece, i, i + target_chars))
                if sl:
                    out.append(sl)
        else:
            current += piece
    if js_trim(current):
        out.append(js_trim(current))
    return out


def semantic_blocks(content: str, target_chars: int) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    heading_stack: List[str] = []
    paragraph: List[str] = []
    paragraph_heading = ""

    def current_heading_path() -> str:
        return " > ".join(h for h in heading_stack if h)

    def flush_paragraph() -> None:
        nonlocal paragraph
        text = js_trim("\n".join(paragraph))
        if text:
            for piece in split_oversized_block(text, target_chars):
                blocks.append({"text": piece, "headingPath": paragraph_heading})
        paragraph = []

    for line in content.replace("\r\n", "\n").split("\n"):
        heading = _HEADING_LINE_RE.match(line)
        if heading:
            flush_paragraph()
            depth = len(heading.group(1))
            del heading_stack[depth - 1:]
            while len(heading_stack) < depth - 1:
                heading_stack.append("")
            heading_stack.append(js_trim(heading.group(2)))
            blocks.append({"text": js_trim(line), "headingPath": current_heading_path()})
            paragraph_heading = current_heading_path()
            continue

        if js_trim(line) == "":
            flush_paragraph()
            paragraph_heading = current_heading_path()
            continue

        if len(paragraph) == 0:
            paragraph_heading = current_heading_path()
        paragraph.append(line)
    flush_paragraph()
    return blocks


def overlap_suffix(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    if len16(text) <= max_chars:
        return text
    raw = slice16(text, -max_chars)
    raw_len = len16(raw)
    m = _PARAGRAPH_BREAK_RE.search(raw)
    if m:
        pb = len16(raw[: m.start()])
        if pb > 0 and raw_len - pb > max_chars * 0.4:
            return js_trim(raw[m.start():])
    m = _SENTENCE_BREAK_RE.search(raw)
    if m:
        sb = len16(raw[: m.start()])
        if sb > 0 and raw_len - sb > max_chars * 0.4:
            return js_trim(raw[m.start() + 1:])
    return js_trim(raw)


def split_source_into_semantic_chunks(content: str, target_chars: int, overlap_chars: int) -> List[SourceChunk]:
    target = max(1_000, target_chars)
    blocks = semantic_blocks(content, target)
    if len(blocks) == 0:
        return []

    raw_chunks: List[Dict[str, str]] = []
    current: List[str] = []
    current_length = 0
    current_heading = blocks[0]["headingPath"]

    def flush() -> None:
        nonlocal current, current_length
        main = js_trim("\n\n".join(current))
        if main:
            raw_chunks.append({"main": main, "headingPath": current_heading})
        current = []
        current_length = 0

    for block in blocks:
        block_len = len16(block["text"])
        next_length = current_length + block_len + (2 if len(current) > 0 else 0)
        if len(current) > 0 and next_length > target:
            flush()
        if len(current) == 0:
            current_heading = block["headingPath"]
        current.append(block["text"])
        current_length += block_len + (2 if len(current) > 1 else 0)
    flush()

    return [
        SourceChunk(
            id=f"chunk-{idx + 1}",
            index=idx + 1,
            total=len(raw_chunks),
            headingPath=chunk["headingPath"],
            overlapBefore=overlap_suffix(raw_chunks[idx - 1]["main"], overlap_chars) if idx > 0 else "",
            main=chunk["main"],
        )
        for idx, chunk in enumerate(raw_chunks)
    ]


def trim_long_text(text: str, max_chars: int) -> str:
    if len16(text) <= max_chars:
        return text
    return f"{js_trim_end(slice16(text, 0, max_chars))}\n\n[...trimmed for prompt budget...]"


def trim_inline_status(text: str, max_chars: int = 240) -> str:
    return text if len16(text) <= max_chars else f"{js_trim_end(slice16(text, 0, max_chars))}..."


# ---------------------------------------------------------------------------
# Hashing, checkpoints, long-source planning (pure parts)
# ---------------------------------------------------------------------------

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1


def hash_text_hex(text: str) -> str:
    """64-bit FNV-1a over UTF-16 code units, 16 hex chars."""
    h = _FNV_OFFSET
    if _ASTRAL.search(text) or any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        raw = text.encode("utf-16-le", "surrogatepass")
        units = [raw[i] | (raw[i + 1] << 8) for i in range(0, len(raw), 2)]
    else:
        units = [ord(c) for c in text]
    for u in units:
        h ^= u
        h = (h * _FNV_PRIME) & _MASK64
    return format(h, "x").rjust(16, "0")


def long_source_checkpoint_path(project_path: str, source_summary_slug: str, source_hash: str) -> str:
    return f"{normalize_path(project_path)}/.llm-wiki/ingest-progress/{source_summary_slug}-{source_hash}.json"


def generation_checkpoint_path(project_path: str, source_summary_slug: str, source_fingerprint: str) -> str:
    return f"{normalize_path(project_path)}/.llm-wiki/ingest-generation/{source_summary_slug}-{source_fingerprint}.json"


def source_generation_fingerprint_from_stat(source_content: str, modified: Any, size: Any) -> str:
    """``hashTextHex(`${content}\\0${modified}\\0${size}`)``; pass ``0`` for
    ``modified`` / ``size`` when the stat failed (as TS does)."""
    return hash_text_hex(f"{source_content}\0{_js_num_str(modified)}\0{_js_num_str(size)}")


def is_compatible_long_source_checkpoint(checkpoint: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """Both mappings use TS camelCase keys (``sourceIdentity`` ...)."""
    try:
        analyses = checkpoint.get("analyses")
        completed = checkpoint.get("completedThrough")
        return bool(
            checkpoint.get("version") == 1
            and checkpoint.get("sourceIdentity") == params["sourceIdentity"]
            and checkpoint.get("sourceHash") == params["sourceHash"]
            and checkpoint.get("sourceLength") == params["sourceLength"]
            and checkpoint.get("sourceBudget") == params["sourceBudget"]
            and checkpoint.get("targetChars") == params["targetChars"]
            and checkpoint.get("overlapChars") == params["overlapChars"]
            and checkpoint.get("chunkTotal") == params["chunkTotal"]
            and isinstance(completed, (int, float))
            and not isinstance(completed, bool)
            and completed >= 0
            and completed <= params["chunkTotal"]
            and isinstance(analyses, list)
            and len(analyses) == completed
        )
    except (AttributeError, KeyError, TypeError):
        return False


def extract_marked_section(raw: str, heading: str) -> str:
    escaped = re.sub(r"[.*+?^${}()|\[\]\\]", lambda m: "\\" + m.group(0), heading)
    pattern = re.compile(
        f"(?:\\A|\n)##{JS_WS}+{escaped}{JS_WS}*\n([\\s\\S]*?)(?=\n##{JS_WS}|\\Z)", _RE | re.IGNORECASE
    )
    m = pattern.search(raw)
    return js_trim(m.group(1)) if m else ""


def long_source_chunk_sizes(source_budget: int) -> Tuple[int, int]:
    """``(target_chars, overlap_chars)`` used by ``analyzeLongSourceInChunks``."""
    target_chars = int(_clamp_number(math.floor(source_budget * 0.55), LONG_SOURCE_CHUNK_MIN, LONG_SOURCE_CHUNK_MAX))
    overlap_chars = int(_clamp_number(math.floor(target_chars * 0.08), 800, 3_000))
    return target_chars, overlap_chars


def format_chunk_analysis_entry(chunk: SourceChunk, chunk_analysis: str) -> str:
    heading = f" — {chunk.heading_path}" if chunk.heading_path else ""
    return "\n".join([
        f"## Chunk {chunk.index}/{chunk.total}{heading}",
        trim_long_text(chunk_analysis, LONG_SOURCE_CHUNK_ANALYSIS_MAX),
    ])


def next_global_digest(next_digest: str, global_digest: str, chunk_analysis: str) -> str:
    fallback = "\n\n".join(p for p in (global_digest, chunk_analysis) if p)
    return trim_long_text(next_digest or fallback, LONG_SOURCE_DIGEST_MAX)


def build_consolidated_long_analysis(global_digest: str, analyses: Sequence[str]) -> str:
    return "\n".join([
        "# Consolidated Long-Document Analysis",
        "",
        "## Final Global Digest",
        global_digest or "(No digest produced.)",
        "",
        "## Per-Chunk Analyses",
        "\n\n".join(analyses),
    ])


def build_long_source_context(
    source_identity: str, chunk_count: int, global_digest: str, analyses: Sequence[str], source_budget: int
) -> str:
    return "\n".join([
        f"# Long Source Context: {source_identity}",
        "",
        f"The original source was analyzed in {chunk_count} semantic chunks with paragraph/section boundaries and overlap. Use this consolidated context instead of assuming the raw document ended early.",
        "",
        "## Final Global Digest",
        global_digest or "(No digest produced.)",
        "",
        "## Chunk Analysis Notes",
        trim_long_text("\n\n".join(analyses), max(source_budget, LONG_SOURCE_CHUNK_ANALYSIS_MAX)),
    ])


def chunk_truncated_message(chunk: SourceChunk, max_tokens: int) -> str:
    return (
        f"Chunk {chunk.index}/{chunk.total} analysis was truncated after reaching the "
        f"{max_tokens:,} token output limit; the saved checkpoint will be retried"
    )
