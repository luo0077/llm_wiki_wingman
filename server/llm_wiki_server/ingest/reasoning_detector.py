"""Port of src/lib/reasoning-detector.ts: diagnostics for thinking-only streams."""
from __future__ import annotations

import re
from typing import List

from .llm_jscompat import JS_DOT, JS_WS, js_json_loads, js_trim, utf16_len

# /"reasoning(?:_content)?"\s*:\s*"((?:[^"\\]|\\.)*)"/g   (`.` excludes line terminators in JS)
_REASONING_FIELD_RE = re.compile(
    r'"reasoning(?:_content)?"' + JS_WS + r'*:' + JS_WS + r'*"((?:[^"\\]|\\' + JS_DOT + r')*)"')


def count_reasoning_chars_in_line(raw_line: str) -> int:
    """Sum of the (JSON-escaped) lengths, in UTF-16 code units like JS `.length`."""
    total = 0
    for m in _REASONING_FIELD_RE.finditer(raw_line):
        total += utf16_len(m.group(1))
    return total


def _g(obj, key):
    return obj.get(key) if isinstance(obj, dict) else None


def extract_reasoning_text_from_line(raw_line: str) -> List[str]:
    line = js_trim(raw_line)
    if not line.startswith("data: "):
        return []
    data = js_trim(line[6:])
    if not data or data == "[DONE]":
        return []
    try:
        parsed = js_json_loads(data)
    except Exception:
        return []
    try:
        out: List[str] = []
        choices = _g(parsed, "choices")
        if choices is not None and not isinstance(choices, list):
            return []
        for choice in (choices or []):
            if choice is None:
                return []  # JS: `choice.delta` on null throws -> caught
            delta = _g(choice, "delta")
            rc = _g(delta, "reasoning_content")
            if isinstance(rc, str):
                out.append(rc)
            r = _g(delta, "reasoning")
            if isinstance(r, str):
                out.append(r)

        pdelta = _g(parsed, "delta")
        if _g(pdelta, "type") == "thinking_delta":
            if isinstance(_g(pdelta, "thinking"), str):
                out.append(pdelta["thinking"])
            if isinstance(_g(pdelta, "text"), str):
                out.append(pdelta["text"])

        candidates = _g(parsed, "candidates")
        if candidates is not None and not isinstance(candidates, list):
            return []
        for cand in (candidates or []):
            if cand is None:
                return []
            parts = _g(_g(cand, "content"), "parts")
            for part in (parts if isinstance(parts, list) else []):
                if part is None:
                    # JS: `part.thought` on null throws -> caught -> []
                    return []
                if _g(part, "thought") and isinstance(_g(part, "text"), str):
                    out.append(part["text"])
        return out
    except Exception:
        return []
