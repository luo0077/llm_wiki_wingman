"""Helpers that reproduce JavaScript string / number semantics needed for faithful ports."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, fields
from typing import Any, Callable, List, Optional

# Characters matched by JS `\s` (also the set trimmed by String.prototype.trim).
JS_WS = "\\t\\n\\v\\f\\r \\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff"
# JS `.` (without the s flag): anything but line terminators.
JS_DOT = "[^\\n\\r\\u2028\\u2029]"

_WS_CHARS = "\t\n\v\f\r                  　﻿"


def S(pattern: str) -> str:
    """Expand the JS escapes `\\s` / `\\S` inside a pattern to explicit classes.

    Works for both bare use and use inside a character class: inside a class
    `\\s` becomes the class body, outside it becomes `[...]`. `\\S` outside a
    class becomes `[^...]` (not supported inside a class).
    """
    out: List[str] = []
    i = 0
    in_class = False
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\" and i + 1 < n:
            nxt = pattern[i + 1]
            if nxt == "s":
                out.append(JS_WS if in_class else "[" + JS_WS + "]")
                i += 2
                continue
            if nxt == "S" and not in_class:
                out.append("[^" + JS_WS + "]")
                i += 2
                continue
            out.append(c + nxt)
            i += 2
            continue
        if c == "[" and not in_class:
            in_class = True
        elif c == "]" and in_class:
            in_class = False
        out.append(c)
        i += 1
    return "".join(out)


def rx(pattern: str, flags: int = 0) -> "re.Pattern[str]":
    """Compile a JS-flavoured pattern: ASCII `\\w \\b \\d`, JS `\\s`, JS `.`-free (caller uses JS_DOT)."""
    return re.compile(S(pattern), flags | re.ASCII)


def js_trim(s: str) -> str:
    return s.strip(_WS_CHARS)


def js_trim_start(s: str) -> str:
    return s.lstrip(_WS_CHARS)


def js_trim_end(s: str) -> str:
    return s.rstrip(_WS_CHARS)


def js_is_blank(s: str) -> bool:
    return js_trim(s) == ""


# ── UTF-16 helpers ────────────────────────────────────────────────

def u16len(s: str) -> int:
    n = len(s)
    for ch in s:
        if ord(ch) > 0xFFFF:
            n += 1
    return n


def u16units(s: str) -> List[int]:
    b = s.encode("utf-16-le", "surrogatepass")
    return [b[i] | (b[i + 1] << 8) for i in range(0, len(b), 2)]


def u16slice(s: str, start: int, end: Optional[int] = None) -> str:
    """JS String.prototype.slice semantics on UTF-16 code units (non-negative or negative indices)."""
    b = s.encode("utf-16-le", "surrogatepass")
    n = len(b) // 2
    if start < 0:
        start = max(n + start, 0)
    start = min(start, n)
    if end is None:
        end = n
    elif end < 0:
        end = max(n + end, 0)
    end = min(end, n)
    if end <= start:
        return ""
    return b[start * 2:end * 2].decode("utf-16-le", "surrogatepass")


# ── Number formatting ─────────────────────────────────────────────

def js_number_to_string(x: float) -> str:
    """Number.prototype.toString() for base 10."""
    if isinstance(x, bool):
        x = int(x)
    if isinstance(x, int):
        try:
            x = float(x)
        except OverflowError:
            return "Infinity" if x > 0 else "-Infinity"
    if math.isnan(x):
        return "NaN"
    if x == 0:
        return "0"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    sign = "-" if x < 0 else ""
    x = abs(x)
    r = repr(x)
    # Extract shortest digits and exponent from repr.
    if "e" in r or "E" in r:
        mant, exp = r.lower().split("e")
        e10 = int(exp)
    else:
        mant, e10 = r, 0
    if "." in mant:
        ip, fp = mant.split(".")
    else:
        ip, fp = mant, ""
    digits = (ip + fp)
    point = len(ip) + e10  # position of decimal point relative to digits start
    stripped = digits.lstrip("0")
    point -= len(digits) - len(stripped)
    digits = stripped.rstrip("0") or "0"
    k = len(digits)
    n = point
    if k <= n <= 21:
        s = digits + "0" * (n - k)
    elif 0 < n <= 21:
        s = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        s = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        es = ("+" if e >= 0 else "-") + str(abs(e))
        if k == 1:
            s = digits + "e" + es
        else:
            s = digits[0] + "." + digits[1:] + "e" + es
    return sign + s


def js_to_fixed0(x: float) -> str:
    """(x).toFixed(0) for finite non-negative values (rounds half up)."""
    return str(int(math.floor(x + 0.5)))


def js_json_string(s: str) -> str:
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20 or 0xD800 <= o <= 0xDFFF:
            out.append("\\u%04x" % o)
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def js_json_stringify(v: Any) -> str:
    """JSON.stringify(v) with no indentation."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "null"
        return js_number_to_string(v)
    if isinstance(v, str):
        return js_json_string(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(js_json_stringify(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(js_json_string(str(k)) + ":" + js_json_stringify(x) for k, x in v.items()) + "}"
    return js_json_string(str(v))


# ── String.prototype.replace replacement-pattern expansion ────────

def js_expand_replacement(repl: str, m: "re.Match[str]") -> str:
    """Expand `$$ $& $` $' $n` like JS String.prototype.replace does for a string replacement."""
    if "$" not in repl:
        return repl
    s = m.string
    ngroups = m.re.groups
    out: List[str] = []
    i = 0
    n = len(repl)
    while i < n:
        c = repl[i]
        if c != "$" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        nx = repl[i + 1]
        if nx == "$":
            out.append("$")
            i += 2
        elif nx == "&":
            out.append(m.group(0))
            i += 2
        elif nx == "`":
            out.append(s[:m.start()])
            i += 2
        elif nx == "'":
            out.append(s[m.end():])
            i += 2
        elif nx.isdigit() and nx in "0123456789":
            two = repl[i + 1:i + 3]
            if len(two) == 2 and two.isdigit() and 1 <= int(two) <= ngroups:
                out.append(m.group(int(two)) or "")
                i += 3
            elif 1 <= int(nx) <= ngroups:
                out.append(m.group(int(nx)) or "")
                i += 2
            else:
                out.append("$")
                i += 1
        else:
            out.append("$")
            i += 1
    return "".join(out)


def js_replace_first(pattern: "re.Pattern[str]", s: str, repl: str) -> str:
    """`s.replace(regex_without_g, "string")` including `$` replacement patterns."""
    m = pattern.search(s)
    if not m:
        return s
    return s[:m.start()] + js_expand_replacement(repl, m) + s[m.end():]


# ── encodeURIComponent / decodeURIComponent ───────────────────────

def encode_uri_component(s: str) -> str:
    from urllib.parse import quote

    return quote(s, safe="-_.!~*'()")


_PCT_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def decode_uri_component(s: str) -> str:
    """Raises ValueError (JS URIError) on malformed sequences."""
    from urllib.parse import unquote

    if _PCT_RE.search(s):
        raise ValueError("URI malformed")
    return unquote(s, encoding="utf-8", errors="strict")


# ── tiny record base ──────────────────────────────────────────────

class Record:
    """Mixin for dataclasses whose field names are the TS (camelCase) names.

    Supports both attribute access and `rec["field"]` / `rec.get("field")`.
    """

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return asdict(self)  # type: ignore[call-overload]

    def keys(self):
        return [f.name for f in fields(self)]  # type: ignore[arg-type]


# JS `^` under the `m` flag also matches after \r,  ,   (Python's only after \n).
BOL_M = "(?:^|(?<=[\\r\\u2028\\u2029]))"

# JS `$` under the `m` flag also matches before \r,  ,   (use with re.M).
EOL_M = "(?:(?=[\\r\\u2028\\u2029])|$)"
