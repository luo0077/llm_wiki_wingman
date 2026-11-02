"""Port of src/lib/sources-merge.ts (frontmatter array-field union merging)."""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from ._js import BOL_M, EOL_M, JS_DOT, js_replace_first, js_trim, rx

_FM_PARSE_RE = rx(r"\A---\r?\n([\s\S]*?)\r?\n---")
_FM_WRITE_RE = rx(r"\A(---\r?\n)([\s\S]*?)(\r?\n---)")
_LEADING_FM_RE = rx(r"\A---\r?\n")
_ITEM_RE = rx(r"\A\s+-\s+[\"']?(" + JS_DOT + r"+?)[\"']?\s*\Z")
_TRAILING_NL_RE = rx(r"\r?\n\Z")
_SPLIT_LINES_RE = re.compile(r"\r?\n")


def _escape(name: str) -> str:
    # Same effect as the JS /[.*+?^${}()|[\]\\]/g escape.
    return re.sub(r"[.*+?^${}()|\[\]\\]", lambda m: "\\" + m.group(0), name)


def _block_re(escaped_name: str) -> "re.Pattern[str]":
    return rx(
        BOL_M + escaped_name + r":\s*\r?\n((?:[ \t]+-\s+" + JS_DOT + r"+(?:\r?\n|" + EOL_M + r"))+)",
        re.M,
    )


def _inline_re(escaped_name: str, capture: bool) -> "re.Pattern[str]":
    inner = r"([^\]]*)" if capture else r"[^\]]*"
    return rx(BOL_M + escaped_name + r":\s*\[" + inner + r"\]", re.M)


def parse_frontmatter_array(content: str, field_name: str) -> List[str]:
    fm_match = _FM_PARSE_RE.search(content)
    if not fm_match:
        return []
    fm = fm_match.group(1)
    escaped = _escape(field_name)
    block = _block_re(escaped).search(fm)
    if block:
        out: List[str] = []
        for line in _SPLIT_LINES_RE.split(block.group(1)):
            m = _ITEM_RE.search(line)
            if m and m.group(1):
                out.append(js_trim(m.group(1)))
        return out

    inline = _inline_re(escaped, True).search(fm)
    if not inline:
        return []
    body = js_trim(inline.group(1))
    if body == "":
        return []
    return _split_inline_array(body)


def _split_inline_array(body: str) -> List[str]:
    out: List[str] = []
    current = ""
    quote: Optional[str] = None
    escaped = False

    for ch in body:
        if escaped:
            current += ch
            escaped = False
            continue
        if quote == '"' and ch == "\\":
            escaped = True
            continue
        if (ch == '"' or ch == "'") and quote is None:
            quote = ch
            continue
        if quote == ch:
            quote = None
            continue
        if ch == "," and quote is None:
            value = js_trim(current)
            if value:
                out.append(value)
            current = ""
            continue
        current += ch

    value = js_trim(current)
    if value:
        out.append(value)
    return out


def write_frontmatter_array(content: str, field_name: str, values: Sequence[str]) -> str:
    fm_match = _FM_WRITE_RE.search(content)
    if not fm_match:
        return content

    open_delim, fm_body, close_delim = fm_match.group(1), fm_match.group(2), fm_match.group(3)
    newline = "\r\n" if open_delim.endswith("\r\n") else "\n"
    escaped = _escape(field_name)
    serialized = ", ".join(_quote_inline_array_value(v) for v in values)
    new_line = f"{field_name}: [{serialized}]"
    tail = content[len(fm_match.group(0)):]

    inline_re = _inline_re(escaped, False)
    if inline_re.search(fm_body):
        rewritten = js_replace_first(inline_re, fm_body, new_line)
        return f"{open_delim}{rewritten}{close_delim}{tail}"

    block_re = _block_re(escaped)
    if block_re.search(fm_body):
        def repl(m: "re.Match[str]") -> str:
            return new_line + (newline if _TRAILING_NL_RE.search(m.group(0)) else "")

        rewritten = block_re.sub(repl, fm_body, count=1)
        return f"{open_delim}{rewritten}{close_delim}{tail}"

    rewritten = f"{fm_body}{newline}{new_line}"
    return f"{open_delim}{rewritten}{close_delim}{tail}"


def _quote_inline_array_value(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _merge_lists(existing: Sequence[str], incoming: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in [*existing, *incoming]:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def merge_array_fields_into_content(
    new_content: str, existing_content: Optional[str], fields: Sequence[str]
) -> str:
    if not existing_content:
        return new_content
    if not _LEADING_FM_RE.search(existing_content):
        return new_content

    result = new_content
    changed = False
    for field_name in fields:
        old_values = parse_frontmatter_array(existing_content, field_name)
        if len(old_values) == 0:
            continue
        new_values = parse_frontmatter_array(result, field_name)
        merged = _merge_lists(old_values, new_values)
        if merged == new_values:
            continue
        result = write_frontmatter_array(result, field_name, merged)
        changed = True
    return result if changed else new_content


def parse_sources(content: str) -> List[str]:
    return parse_frontmatter_array(content, "sources")


def write_sources(content: str, sources: Sequence[str]) -> str:
    return write_frontmatter_array(content, "sources", sources)


def merge_sources_lists(existing: Sequence[str], incoming: Sequence[str]) -> List[str]:
    return _merge_lists(existing, incoming)


def merge_sources_into_content(new_content: str, existing_content: Optional[str]) -> str:
    return merge_array_fields_into_content(new_content, existing_content, ["sources"])
