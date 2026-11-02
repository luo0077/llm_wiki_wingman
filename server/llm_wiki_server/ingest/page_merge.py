"""Port of src/lib/page-merge.ts.

Merge a wiki page the LLM just generated with whatever is already on disk.
Pure logic; the LLM call is injected as ``merger``.

``merger(existing_content, incoming_content, source_file_name, signal)`` returns
the LLM's full proposed merged file (frontmatter + body). It may be an ``async
def`` or a plain function. ``signal`` is whatever abort token the caller put in
``MergePageOptions.signal`` (opaque here; None by default).
"""
from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlsplit

from ._js import (
    BOL_M,
    EOL_M,
    JS_DOT,
    Record,
    js_expand_replacement,
    js_replace_first,
    js_to_fixed0,
    js_trim,
    rx,
    u16len,
)
from .frontmatter import parse_frontmatter
from .ingest_sanitize import normalize_malformed_wikilinks
from .sources_merge import merge_array_fields_into_content

log = logging.getLogger(__name__)

UNION_FIELDS = ("sources", "tags", "related")
LOCKED_FIELDS = ("type", "title", "created")
BODY_SHRINK_THRESHOLD = 0.7

HTTP_URL_RE = rx(r"https?://[^\s<>\"'\])]+", re.I)

MergeFn = Callable[..., Union[str, Awaitable[str]]]


@dataclass
class MergePageOptions(Record):
    sourceFileName: str
    pagePath: str
    signal: Any = None
    backup: Optional[Callable[[str], Any]] = None  # sync or async; errors swallowed
    today: Optional[Callable[[], str]] = None  # defaults to today's UTC date (YYYY-MM-DD)
    replaceExistingBody: bool = False


async def _maybe_await(v: Any) -> Any:
    if inspect.isawaitable(v):
        return await v
    return v


_FM_BLOCK_RE = rx(r"\A---\r?\n[\s\S]*?\r?\n---(?:\r?\n|\Z)")


async def merge_page_content(
    new_content: str,
    existing_content: Optional[str],
    merger: MergeFn,
    opts: MergePageOptions,
) -> str:
    # Fast path 1: brand-new page.
    if not existing_content:
        return normalize_malformed_wikilinks(new_content)

    # Fast path 2: byte-identical re-ingest.
    if new_content == existing_content:
        return normalize_malformed_wikilinks(existing_content)

    # Step 1 - always-on: union the array frontmatter fields.
    array_merged = merge_array_fields_into_content(new_content, existing_content, list(UNION_FIELDS))

    old_parsed = parse_frontmatter(existing_content)
    array_merged_parsed = parse_frontmatter(array_merged)
    if opts.replaceExistingBody:
        await _try_backup(opts, existing_content)
        replacement = array_merged
        for field_name in LOCKED_FIELDS:
            existing_value = old_parsed.frontmatter.get(field_name) if old_parsed.frontmatter else None
            if isinstance(existing_value, str) and existing_value != "":
                replacement = _set_frontmatter_scalar(replacement, field_name, existing_value)
        return normalize_malformed_wikilinks(
            _set_frontmatter_scalar(replacement, "updated", (opts.today or _default_today)())
        )
    if js_trim(old_parsed.body) == js_trim(array_merged_parsed.body):
        return normalize_malformed_wikilinks(array_merged)

    # Step 2 - ask the merger. Failures fall back to the array-merged-only path.
    try:
        llm_output = await _maybe_await(
            merger(existing_content, array_merged, opts.sourceFileName, opts.signal)
        )
    except Exception as err:  # noqa: BLE001 - mirrors catch-all in TS
        log.warning(
            "[page-merge] LLM merge failed for %s, falling back to incoming + array-field union: %s",
            opts.pagePath,
            err,
        )
        await _try_backup(opts, existing_content)
        return normalize_malformed_wikilinks(array_merged)

    # Sanity 1: LLM output must parse as a frontmatter-bearing wiki page.
    llm_parsed = parse_frontmatter(llm_output)
    if llm_parsed.frontmatter is None:
        log.warning("[page-merge] LLM output for %s has no frontmatter - rejecting, falling back", opts.pagePath)
        await _try_backup(opts, existing_content)
        return normalize_malformed_wikilinks(array_merged)

    # Sanity 2: body length (JS String.length = UTF-16 code units).
    old_body_len = u16len(old_parsed.body)
    new_body_len = u16len(array_merged_parsed.body)
    llm_body_len = u16len(llm_parsed.body)
    min_threshold = max(old_body_len, new_body_len) * BODY_SHRINK_THRESHOLD
    if llm_body_len < min_threshold:
        log.warning(
            "[page-merge] LLM merge for %s produced body %d chars, below threshold %s (max input was %d) - rejecting, falling back",
            opts.pagePath,
            llm_body_len,
            js_to_fixed0(min_threshold),
            max(old_body_len, new_body_len),
        )
        await _try_backup(opts, existing_content)
        return normalize_malformed_wikilinks(array_merged)

    llm_output = _remove_introduced_reserved_example_urls(llm_output, existing_content, array_merged)

    # Step 3 - deterministic post-processing.
    final = llm_output
    for field_name in LOCKED_FIELDS:
        existing_value = old_parsed.frontmatter.get(field_name) if old_parsed.frontmatter else None
        if isinstance(existing_value, str) and existing_value != "":
            final = _set_frontmatter_scalar(final, field_name, existing_value)
    final = merge_array_fields_into_content(final, array_merged, list(UNION_FIELDS))
    today_fn = opts.today or _default_today
    final = _set_frontmatter_scalar(final, "updated", today_fn())

    return normalize_malformed_wikilinks(
        strip_body_wikilink_path_prefixes(strip_merge_scaffolding(final, [existing_content, array_merged]))
    )


# ── reserved example-domain URLs ──────────────────────────────────

_RESERVED_DOMAINS = ("example.com", "example.org", "example.net")
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?]+\Z")


def _parse_url(raw: str) -> Optional[Tuple[str, str]]:
    """Approximation of `new URL(raw)` -> (hostname, canonical string); None if invalid."""
    try:
        parts = urlsplit(raw)
        host = parts.hostname
        if not parts.scheme or not host:
            return None
        port = parts.port  # may raise ValueError
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    default_port = {"http": 80, "https": 443}.get(scheme)
    netloc = host
    if port is not None and port != default_port:
        netloc += f":{port}"
    userinfo = ""
    if "@" in parts.netloc:
        userinfo = parts.netloc.rsplit("@", 1)[0] + "@"
    path = parts.path or "/"
    canonical = f"{scheme}://{userinfo}{netloc}{path}"
    if parts.query or raw.split("#")[0].endswith("?"):
        canonical += "?" + parts.query
    if parts.fragment or raw.endswith("#"):
        canonical += "#" + parts.fragment
    return host, canonical


@dataclass
class _ReservedUrlMatch:
    canonical: str
    raw: str
    start: int
    end: int


def _reserved_example_urls(content: str) -> List[_ReservedUrlMatch]:
    urls: List[_ReservedUrlMatch] = []
    for match in HTTP_URL_RE.finditer(content):
        raw = _TRAILING_PUNCT_RE.sub("", match.group(0))
        parsed = _parse_url(raw)
        if parsed is None:
            continue
        host = re.sub(r"\.\Z", "", parsed[0]).lower()
        if any(host == d or host.endswith("." + d) for d in _RESERVED_DOMAINS):
            urls.append(_ReservedUrlMatch(parsed[1], raw, match.start(), match.start() + len(raw)))
    return urls


_MD_LINK_RE = rx(r"!?\[([^\]\r\n]+)\]\((https?://[^\s<>\")]+)(?:\s+[\"'][^\"']*[\"'])?\)", re.I)
_ANGLE_URL_RE = rx(r"<(https?://[^\s<>]+)>", re.I)
_EMPTY_REF_RE = rx(r"\A\s*\[[^\]]+\]:\s*\Z")
_EMPTY_BULLET_RE = rx(r"\A\s*[-*+]\s*\Z")
_TRAILING_BLANKS_RE = re.compile(r"[ \t]+\Z")
_NEWLINE_PART_RE = re.compile(r"\r?\n")
_NEWLINE_SPLIT_CAPTURE_RE = re.compile(r"(\r?\n)")
_FENCE_MARKER_RE = re.compile(r"\A {0,3}(`{3,}|~{3,})")
_INDENT_RE = re.compile(r"\A(?: {4}|\t)")


def _remove_introduced_reserved_example_urls(output: str, existing: str, incoming: str) -> str:
    input_urls: Set[str] = {m.canonical for m in _reserved_example_urls(existing)} | {
        m.canonical for m in _reserved_example_urls(incoming)
    }
    introduced = [m for m in _reserved_example_urls(output) if m.canonical not in input_urls]
    if not introduced:
        return output
    log.warning("[page-merge] removed invented reserved example-domain URL(s) from merge output")
    introduced_urls = {m.canonical for m in introduced}
    fm = _FM_BLOCK_RE.search(output)
    prefix = fm.group(0) if fm else ""
    body = output[len(prefix):]
    fence: Optional[Tuple[str, int]] = None

    def should_remove(raw: str) -> bool:
        parsed = _parse_url(_TRAILING_PUNCT_RE.sub("", raw))
        return parsed is not None and parsed[1] in introduced_urls

    out_parts: List[str] = []
    for part in _NEWLINE_SPLIT_CAPTURE_RE.split(body):
        if _NEWLINE_PART_RE.fullmatch(part):
            out_parts.append(part)
            continue
        marker = _FENCE_MARKER_RE.search(part)
        if marker:
            if not fence:
                fence = (marker.group(1)[0], len(marker.group(1)))
            elif (
                marker.group(1)[0] == fence[0]
                and len(marker.group(1)) >= fence[1]
                and js_trim(part[len(marker.group(0)):]) == ""
            ):
                fence = None
            out_parts.append(part)
            continue
        if fence or _INDENT_RE.search(part):
            out_parts.append(part)
            continue

        line = _MD_LINK_RE.sub(lambda m: m.group(1) if should_remove(m.group(2)) else m.group(0), part)
        line = _ANGLE_URL_RE.sub(lambda m: "" if should_remove(m.group(1)) else m.group(0), line)
        line = HTTP_URL_RE.sub(lambda m: "" if should_remove(m.group(0)) else m.group(0), line)
        if _EMPTY_REF_RE.search(line) or _EMPTY_BULLET_RE.search(line):
            out_parts.append("")
            continue
        out_parts.append(_TRAILING_BLANKS_RE.sub("", line, count=1))
    return prefix + "".join(out_parts)


# ── scaffolding removal ───────────────────────────────────────────

_LABEL = (
    "(?:现有|新生成|新|原始|旧|已有)(?:内容|版本|材料)(?:的?(?:定义|说明|描述))?"
    "|(?:补充材料|此前收集的材料)"
    "|(?:existing|newly generated|new|original|old|incoming|additional|previously collected) "
    "(?:content|version|material)(?: (?:definition|description))?"
)

_SOURCE_NOTE_RE = rx(
    r"\A[ \t]*(?:[-*][ \t]*)?\*?[（(]?[ \t]*(?:来源|source)[ \t]*[:：][ \t]*(?:" + _LABEL + r")[ \t]*[）)]?\*?[ \t]*\Z",
    re.I,
)
_BRACKETED_SUFFIX_RE = rx(r"[ \t]*[（(【][ \t]*(?:" + _LABEL + r")[ \t]*[）)】][ \t]*\Z", re.I)
_BARE_SUFFIX_RE = rx(r"\A(#{1,6}\s+" + JS_DOT + r"+?)[ \t]+[-—:][ \t]*(?:" + _LABEL + r")[ \t]*\Z", re.I)
_LINE_SPLIT_RE = re.compile(r"\r?\n")
_HEADING_TEXT_RE = rx(r"\A#{1,6}\s+(" + JS_DOT + r"+?)\s*\Z")
_HEADING_PREFIX_RE = rx(r"\A#{1,6}\s+")
_LINE_RE = re.compile(JS_DOT + r"*(?:\r?\n|\Z)")
_TRAIL_NL_RE = re.compile(r"\r?\n\Z")


def strip_merge_scaffolding(content: str, protected_inputs: Optional[List[str]] = None) -> str:
    protected_inputs = protected_inputs or []
    fm = _FM_BLOCK_RE.search(content)
    prefix = fm.group(0) if fm else ""
    body = content[len(prefix):]
    state: Dict[str, Any] = {"fence": None}

    protected_lines: Set[str] = set()
    protected_headings: Set[str] = set()
    for inp in protected_inputs:
        for line in _LINE_SPLIT_RE.split(inp):
            protected_lines.add(js_trim(line))
            m = _HEADING_TEXT_RE.search(line)
            if m and m.group(1):
                protected_headings.add(m.group(1))

    def handle(m: "re.Match[str]") -> str:
        line = m.group(0)
        line_content = _TRAIL_NL_RE.sub("", line, count=1)
        newline = line[len(line_content):]
        marker = _FENCE_MARKER_RE.search(line_content)
        if marker:
            fence = state["fence"]
            if not fence:
                state["fence"] = (marker.group(1)[0], len(marker.group(1)))
            elif (
                marker.group(1)[0] == fence[0]
                and len(marker.group(1)) >= fence[1]
                and js_trim(line_content[len(marker.group(0)):]) == ""
            ):
                state["fence"] = None
            return line
        if state["fence"] or _INDENT_RE.search(line_content):
            return line
        if js_trim(line_content) in protected_lines:
            return line
        if _SOURCE_NOTE_RE.search(line_content):
            return ""
        if not _HEADING_PREFIX_RE.search(line_content):
            return line
        heading_text = js_trim(_HEADING_PREFIX_RE.sub("", line_content, count=1))
        if heading_text in protected_headings:
            return line
        without_bracket = _BRACKETED_SUFFIX_RE.sub("", line_content, count=1)
        without_bare = _BARE_SUFFIX_RE.sub(lambda mm: mm.group(1), without_bracket, count=1)
        return f"{without_bare}{newline}"

    cleaned_body = _LINE_RE.sub(handle, body)
    return f"{prefix}{cleaned_body}"


# ── wikilink path prefixes ────────────────────────────────────────

def strip_body_wikilink_path_prefixes(content: str) -> str:
    """Rewrite body `[[dir/slug]]` targets to bare slugs (frontmatter, code and embeds untouched)."""
    frontmatter = _FM_BLOCK_RE.search(content)
    if not frontmatter:
        return content
    body = content[len(frontmatter.group(0)):]
    if "[[" not in body:
        return content
    return f"{frontmatter.group(0)}{_normalize_wikilinks_outside_code(body)}"


def _normalize_wikilinks_outside_code(body: str) -> str:
    state: Dict[str, Any] = {"fence": None}

    def handle(m: "re.Match[str]") -> str:
        line = m.group(0)
        content = _TRAIL_NL_RE.sub("", line, count=1)
        marker = _FENCE_MARKER_RE.search(content)
        if marker:
            mk = marker.group(1)[0]
            length = len(marker.group(1))
            fence = state["fence"]
            if not fence:
                state["fence"] = (mk, length)
            elif mk == fence[0] and length >= fence[1] and js_trim(content[len(marker.group(0)):]) == "":
                state["fence"] = None
            return line
        if state["fence"] or _INDENT_RE.search(content):
            return line
        return _replace_outside_inline_code(line)

    return _LINE_RE.sub(handle, body)


def _replace_outside_inline_code(text: str) -> str:
    output = ""
    cursor = 0
    while cursor < len(text):
        opening = text.find("`", cursor)
        if opening < 0:
            return output + _replace_wikilink_prefixes(text[cursor:])
        output += _replace_wikilink_prefixes(text[cursor:opening])
        run_end = opening + 1
        while run_end < len(text) and text[run_end] == "`":
            run_end += 1
        delimiter = text[opening:run_end]
        closing = text.find(delimiter, run_end)
        if closing < 0:
            output += _replace_wikilink_prefixes(text[opening:run_end])
            cursor = run_end
            continue
        output += text[opening:closing + len(delimiter)]
        cursor = closing + len(delimiter)
    return output


_PAGE_WIKILINK_RE = rx(r"\[\[([^\]|\n]+)(?:\|([^\]\n]*))?\]\]")
_TRAILING_BACKSLASHES_RE = re.compile(r"\\+\Z")


def _replace_wikilink_prefixes(text: str) -> str:
    def repl(m: "re.Match[str]") -> str:
        match = m.group(0)
        offset = m.start()
        raw_target = m.group(1)
        raw_alias = m.group(2)
        if offset > 0 and text[offset - 1] == "!":
            return match
        bs = _TRAILING_BACKSLASHES_RE.search(text[:offset])
        preceding = len(bs.group(0)) if bs else 0
        if preceding % 2 == 1:
            return match

        target = js_trim(raw_target)
        normalized_target = _bare_wikilink_target(target)
        if normalized_target == target:
            return match
        alias = "" if raw_alias is None else f"|{raw_alias}"
        return f"[[{normalized_target}{alias}]]"

    return _PAGE_WIKILINK_RE.sub(repl, text)


_URI_LIKE_RE = rx(r"\A[a-z][a-z0-9+.-]*:", re.I)


def _bare_wikilink_target(target: str) -> str:
    if not target or target.startswith("#"):
        return target
    if _URI_LIKE_RE.search(target):
        return target

    fragment_index = target.find("#")
    page_target = target[:fragment_index] if fragment_index >= 0 else target
    fragment = target[fragment_index:] if fragment_index >= 0 else ""
    normalized_path = page_target.replace("\\", "/")
    if "/" not in normalized_path:
        return target

    leaf = normalized_path.split("/")[-1]
    if not leaf:
        return target
    extension_index = leaf.rfind(".")
    if extension_index > 0 and leaf[extension_index:].lower() != ".md":
        return target
    return f"{leaf}{fragment}"


async def _try_backup(opts: MergePageOptions, existing_content: str) -> None:
    if not opts.backup:
        return
    try:
        await _maybe_await(opts.backup(existing_content))
    except Exception as err:  # noqa: BLE001
        log.warning("[page-merge] backup failed for %s: %s", opts.pagePath, err)


def _default_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


_SET_FM_RE = rx(r"\A(---\n)([\s\S]*?)(\n---)")


def _set_frontmatter_scalar(content: str, field_name: str, value: str) -> str:
    fm_match = _SET_FM_RE.search(content)
    if not fm_match:
        return content
    open_delim, fm_body, close_delim = fm_match.group(1), fm_match.group(2), fm_match.group(3)
    escaped = re.sub(r"[.*+?^${}()|\[\]\\]", lambda m: "\\" + m.group(0), field_name)
    new_line = f"{field_name}: {value}"
    tail = content[len(fm_match.group(0)):]

    line_re = rx(BOL_M + escaped + r":\s*(?!\[)([^\n]*)", re.M)
    if line_re.search(fm_body):
        rewritten = js_replace_first(line_re, fm_body, new_line)
        return f"{open_delim}{rewritten}{close_delim}{tail}"
    rewritten = f"{fm_body}\n{new_line}"
    return f"{open_delim}{rewritten}{close_delim}{tail}"
