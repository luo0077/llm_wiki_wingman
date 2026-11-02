"""Port of src/lib/ingest-sanitize.ts: clean an LLM-generated wiki page before it hits disk."""
from __future__ import annotations

import re
from typing import List, Optional

from ._js import BOL_M, EOL_M, JS_DOT, js_trim, rx

_FM_PREFIX_RE = rx(
    r"\A---[ \t]*\r?\n(?:---[ \t]*(?:\r?\n|\Z)|[\s\S]*?\r?\n---[ \t]*(?:\r?\n|\Z))"
)
_LINE_RE = re.compile(JS_DOT + r"*(?:\r?\n|\Z)")
_TRAIL_NL_RE = re.compile(r"\r?\n\Z")
_MARKER_RE = rx(r"\A {0,3}(?:>\s*)*(`{3,}|~{3,})")
_HTML_CODE_OPEN_RE = rx(r"<(?:pre|code)\b", re.I)
_HTML_CODE_CLOSE_RE = rx(r"</(?:pre|code)>", re.I)
_INDENTED_RE = re.compile(r"\A(?: {4}|\t)")
_NESTED_LIST_RE = rx(r"\A(?: {4}|\t)(?:[-+*]|\d+[.)])\s+")
_CITATION_RE = rx(r"\[\[(\d+\](?:\s*,\s*\[\d+\])*)\]")
_DIGITS_RE = rx(r"\d+")
_ALIAS_LINK_RE = rx(r"\[\[([^\]|\n]+)\|([^\]\n]+)\](?!\])")


def sanitize_ingested_file_content(content: str) -> str:
    cleaned = content
    cleaned = _strip_outer_code_fence(cleaned)
    cleaned = _strip_frontmatter_key_prefix(cleaned)
    cleaned = _add_missing_opening_frontmatter_fence(cleaned)
    cleaned = _repair_wikilink_lists_in_frontmatter(cleaned)
    cleaned = normalize_malformed_wikilinks(cleaned)
    return cleaned


def normalize_malformed_wikilinks(content: str) -> str:
    bom = "﻿" if content.startswith("﻿") else ""
    without_bom = content[len(bom):]
    frontmatter = _FM_PREFIX_RE.search(without_bom)
    fm_text = frontmatter.group(0) if frontmatter else ""
    prefix = f"{bom}{fm_text}"
    body = without_bom[len(fm_text):]
    if "[[" not in body:
        return content

    state = {"fence": None, "math": False, "html": False}

    def handle(m: "re.Match[str]") -> str:
        line = m.group(0)
        line_content = _TRAIL_NL_RE.sub("", line, count=1)
        marker_match = _MARKER_RE.search(line_content)
        if marker_match:
            marker = marker_match.group(1)[0]
            length = len(marker_match.group(1))
            fence = state["fence"]
            if not fence:
                state["fence"] = (marker, length)
            elif (
                marker == fence[0]
                and length >= fence[1]
                and js_trim(line_content[len(marker_match.group(0)):]) == ""
            ):
                state["fence"] = None
            return line
        if state["fence"]:
            return line
        trimmed = js_trim(line_content)
        if trimmed == "$$":
            state["math"] = not state["math"]
            return line
        if state["math"]:
            return line
        if _HTML_CODE_OPEN_RE.search(line_content):
            state["html"] = True
        if state["html"]:
            if _HTML_CODE_CLOSE_RE.search(line_content):
                state["html"] = False
            return line
        indented = bool(_INDENTED_RE.search(line_content))
        nested_list = bool(_NESTED_LIST_RE.search(line_content))
        if indented and not nested_list:
            return line
        return _normalize_malformed_links_outside_inline_code(line)

    normalized = _LINE_RE.sub(handle, body)
    return f"{prefix}{normalized}"


def _normalize_malformed_links_outside_inline_code(text: str) -> str:
    output = ""
    cursor = 0
    while cursor < len(text):
        opening = text.find("`", cursor)
        if opening < 0:
            return output + _normalize_malformed_link_segment(text[cursor:])
        output += _normalize_malformed_link_segment(text[cursor:opening])
        run_end = opening + 1
        while run_end < len(text) and text[run_end] == "`":
            run_end += 1
        delimiter = text[opening:run_end]
        closing = text.find(delimiter, run_end)
        if closing < 0:
            output += _normalize_malformed_link_segment(text[opening:run_end])
            cursor = run_end
            continue
        output += text[opening:closing + len(delimiter)]
        cursor = closing + len(delimiter)
    return output


def _normalize_malformed_link_segment(segment: str) -> str:
    def citation(m: "re.Match[str]") -> str:
        match = m.group(0)
        citations = m.group(1)
        offset = m.start()
        if offset > 0 and (segment[offset - 1] == "!" or segment[offset - 1] == "["):
            return match
        end = offset + len(match)
        if end < len(segment) and segment[end] == "(":
            return match
        numbers = _DIGITS_RE.findall(citations)
        if any(len(v) > 3 for v in numbers):
            return match
        return f"[{citations}"

    normalized = _CITATION_RE.sub(citation, segment)
    if "](" in normalized:
        return normalized

    def alias(m: "re.Match[str]") -> str:
        offset = m.start()
        if offset > 0 and normalized[offset - 1] == "!":
            return m.group(0)
        return f"[[{m.group(1)}|{m.group(2)}]]"

    return _ALIAS_LINK_RE.sub(alias, normalized)


_OPEN_FENCE_RE = rx(r"\A(?:﻿)?(?:[ \t]*\r?\n)*[ \t]*```(?:yaml|md|markdown)?[ \t]*\r?\n", re.I)
_CLOSE_FENCE_RE = rx(r"\r?\n[ \t]*```[ \t]*\r?\n?\s*\Z")
_FM_ONLY_RE = rx(
    BOL_M + r"(---[ \t]*\r?\n[\s\S]*?" + BOL_M + r"---[ \t]*\r?\n)[ \t]*```[ \t]*(?:\r?\n|" + EOL_M + ")",
    re.M,
)
# The leading `^` of the TS pattern is a line-start anchor (m flag); the match may start on any line.


def _strip_outer_code_fence(content: str) -> str:
    open_ = _OPEN_FENCE_RE.search(content)
    if not open_:
        return content
    after_open = content[len(open_.group(0)):]

    close = _CLOSE_FENCE_RE.search(after_open)
    if close:
        return after_open[: close.start()]

    fm_only = _FM_ONLY_RE.search(after_open)
    if not fm_only:
        return content
    return fm_only.group(1) + after_open[len(fm_only.group(0)):]


_KEY_PREFIX_RE = rx(r"\A[ \t]*frontmatter\s*:\s*\r?\n(?=[ \t]*---\s*\r?\n)")


def _strip_frontmatter_key_prefix(content: str) -> str:
    m = _KEY_PREFIX_RE.search(content)
    if not m:
        return content
    return content[len(m.group(0)):]


_OPEN_DELIM_RE = rx(r"\A[ \t]*---\s*(\r?\n|\Z)")
_NEWLINE_SPLIT_RE = re.compile(r"\r?\n")
_FM_KEY_RE = rx(r"\A(type|title|created|updated|tags|related|sources)\s*:", re.I)
_H_RE = rx(r"\A#{1,6}\s+")


def _add_missing_opening_frontmatter_fence(content: str) -> str:
    if _OPEN_DELIM_RE.search(content):
        return content

    lines = _NEWLINE_SPLIT_RE.split(content)
    first_content_idx = -1
    for i, line in enumerate(lines):
        if len(js_trim(line)) > 0:
            first_content_idx = i
            break
    if first_content_idx < 0:
        return content

    first = js_trim(lines[first_content_idx])
    if not _FM_KEY_RE.search(first):
        return content

    search_end = min(len(lines), first_content_idx + 30)
    for i in range(first_content_idx + 1, search_end):
        trimmed = js_trim(lines[i])
        if trimmed == "---":
            return "---\n" + "\n".join(lines[first_content_idx:])
        if _H_RE.search(trimmed):
            break
    return content


_FM_REPAIR_RE = rx(r"\A(---[ \t]*(\r?\n))([\s\S]*?)(\r?\n---[ \t]*(?:\r?\n|\Z))")
_WIKILINK_LIST_RE = rx(
    r"\A(\s*[A-Za-z_][\w-]*\s*:\s*)(\[\[[^\]]+\]\](?:\s*,\s*\[\[[^\]]+\]\])+)\s*\Z"
)


def _repair_wikilink_lists_in_frontmatter(content: str) -> str:
    m = _FM_REPAIR_RE.search(content)
    if not m:
        return content

    repaired: List[str] = []
    for line in _NEWLINE_SPLIT_RE.split(m.group(3)):
        lm = _WIKILINK_LIST_RE.search(line)
        if not lm:
            repaired.append(line)
            continue
        items = ", ".join(
            '"%s"' % t for t in (js_trim(s) for s in lm.group(2).split(",")) if t
        )
        repaired.append(f"{lm.group(1)}[{items}]")
    repaired_payload = m.group(2).join(repaired)
    return m.group(1) + repaired_payload + m.group(4) + content[len(m.group(0)):]
