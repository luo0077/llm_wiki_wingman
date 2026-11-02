"""JavaScript-semantics helpers used by the ingest ports (UTF-16 offsets, trim, URL, JSON, URI coding).

The TS sources lean on JS behaviours that differ from Python: `String.prototype.trim` /
regex `\\s`, UTF-16 string indices, WHATWG `URL`, `JSON.parse`/`JSON.stringify` and
`encodeURIComponent`/`decodeURIComponent`. These helpers reproduce the pieces we need.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional
from urllib.parse import quote

# ECMAScript WhiteSpace + LineTerminator code points (what `\s` and `trim()` use).
JS_WS_CHARS = "\t\n\v\f\r              " \
              "    　﻿"
# Regex class source for JS `\s` (use inside a character class or on its own).
JS_WS = "[" + JS_WS_CHARS.replace("\\", "\\\\") + "]"
JS_WS_IN_CLASS = JS_WS_CHARS  # for embedding inside a larger [...] class
# JS `.` (no dotAll) does not match these line terminators.
JS_DOT = "[^\\n\\r\\u2028\\u2029]"

# `i` flag without `u` in JS only case-folds simple mappings that stay within ASCII for ASCII
# pattern chars; Python needs re.ASCII to avoid e.g. matching U+017F against `s`.
RE_I = re.IGNORECASE | re.ASCII


def js_trim(s: str) -> str:
    return s.strip(JS_WS_CHARS)


# ----------------------------------------------------------------- UTF-16

def to_utf16(s: str) -> str:
    """Return a str whose code points are the UTF-16 code units of `s` (astral chars -> surrogate pair)."""
    if all(ord(c) <= 0xFFFF for c in s):
        return s
    out = []
    for c in s:
        o = ord(c)
        if o > 0xFFFF:
            o -= 0x10000
            out.append(chr(0xD800 + (o >> 10)))
            out.append(chr(0xDC00 + (o & 0x3FF)))
        else:
            out.append(c)
    return "".join(out)


def from_utf16(s: str) -> str:
    """Inverse of `to_utf16`; lone surrogates become U+FFFD (as TextEncoder would)."""
    if not any(0xD800 <= ord(c) <= 0xDFFF for c in s):
        return s
    return s.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def utf16_len(s: str) -> int:
    return sum(2 if ord(c) > 0xFFFF else 1 for c in s)


# ----------------------------------------------------------------- number / string coercion

def js_number_str(v: float) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    if v == int(v) and abs(v) < 1e21:
        return str(int(v))
    return repr(v)


def js_str(v: Any) -> str:
    """Template-literal coercion (`${v}`) for JSON-derived values."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return js_number_str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return ",".join("" if x is None else js_str(x) for x in v)
    return "[object Object]"


def is_js_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ----------------------------------------------------------------- JSON

def _reject_constant(name: str) -> Any:
    raise ValueError(f"Unexpected token {name} in JSON")


def js_json_loads(text: str) -> Any:
    """`JSON.parse`: like json.loads but rejects NaN/Infinity/-Infinity literals."""
    return json.loads(text, parse_constant=_reject_constant)


def _normalize_for_json(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _normalize_for_json(x) for k, x in v.items() if x is not _UNDEFINED}
    if isinstance(v, (list, tuple)):
        return [None if x is _UNDEFINED else _normalize_for_json(x) for x in v]
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        if v == int(v) and abs(v) < 1e21:
            return int(v)
    return v


class _Undefined:
    def __repr__(self) -> str:  # pragma: no cover
        return "UNDEFINED"


_UNDEFINED = _Undefined()
_LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def dumps_js(obj: Any) -> str:
    """`JSON.stringify(obj)` (compact). Integral floats print as ints; NaN/Infinity -> null;
    lone surrogates are \\u-escaped."""
    s = json.dumps(_normalize_for_json(obj), ensure_ascii=False, separators=(",", ":"))
    if _LONE_SURROGATE_RE.search(s):
        s = _LONE_SURROGATE_RE.sub(lambda m: "\\u%04x" % ord(m.group(0)), s)
    return s


def dumps_js_pretty(obj: Any, indent: int = 2) -> str:
    """`JSON.stringify(obj, null, indent)`."""
    s = json.dumps(_normalize_for_json(obj), ensure_ascii=False, indent=indent)
    if _LONE_SURROGATE_RE.search(s):
        s = _LONE_SURROGATE_RE.sub(lambda m: "\\u%04x" % ord(m.group(0)), s)
    return s


# ----------------------------------------------------------------- URI coding

_URI_COMPONENT_SAFE = "-_.!~*'()"
_RESERVED = ";/?:@&=+$,#"


def encode_uri_component(s: str) -> str:
    return quote(s, safe=_URI_COMPONENT_SAFE)


_PCT_RUN = re.compile(r"(?:%[0-9A-Fa-f]{2})+")
_BAD_PCT = re.compile(r"%(?![0-9A-Fa-f]{2})")


def _decode_uri(s: str, reserved: str) -> str:
    if _BAD_PCT.search(s):
        raise ValueError("URI malformed")

    def repl(m: "re.Match[str]") -> str:
        raw = m.group(0)
        bs = bytes(int(raw[i + 1:i + 3], 16) for i in range(0, len(raw), 3))
        out = []
        i = 0
        while i < len(bs):
            b = bs[i]
            if b < 0x80:
                ch = chr(b)
                out.append(raw[i * 3:i * 3 + 3] if ch in reserved else ch)
                i += 1
                continue
            n = 2 if (b >> 5) == 0b110 else 3 if (b >> 4) == 0b1110 else 4 if (b >> 3) == 0b11110 else 0
            if not n or i + n > len(bs):
                raise ValueError("URI malformed")
            try:
                out.append(bs[i:i + n].decode("utf-8"))
            except UnicodeDecodeError:
                raise ValueError("URI malformed")
            i += n
        return "".join(out)

    return _PCT_RUN.sub(repl, s)


def decode_uri_component(s: str) -> str:
    """`decodeURIComponent`; raises ValueError (URIError) on malformed input."""
    return _decode_uri(s, "")


def decode_uri(s: str) -> str:
    """`decodeURI`: like decodeURIComponent but keeps escapes of reserved characters."""
    return _decode_uri(s, _RESERVED)


# ----------------------------------------------------------------- URL (WHATWG subset)

_SPECIAL = {"http": "80", "https": "443", "ftp": "21", "ws": "80", "wss": "443", "file": ""}
_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")


class JsUrl:
    def __init__(self, protocol: str, username: str, password: str, hostname: str, port: str, pathname: str,
                 search: str = "", hash_: str = ""):
        self.protocol = protocol
        self.username = username
        self.password = password
        self.hostname = hostname
        self.port = port
        self.pathname = pathname
        self.search = search
        self.hash = hash_


def parse_url(value: str) -> Optional[JsUrl]:
    """Approximation of `new URL(value)` (absolute URLs only). Returns None where JS would throw."""
    s = value.strip("".join(chr(c) for c in range(0x21)))
    s = s.replace("\t", "").replace("\n", "").replace("\r", "")
    m = _SCHEME_RE.match(s)
    if not m:
        return None
    scheme = m.group(1).lower()
    rest = s[m.end():]
    special = scheme in _SPECIAL
    hash_ = ""
    search = ""
    if "#" in rest:
        rest, h = rest.split("#", 1)
        hash_ = "#" + h if h else ""
    if "?" in rest:
        rest, q = rest.split("?", 1)
        search = "?" + q if q else ""
    if special:
        if scheme != "file":
            rest = rest.replace("\\", "/")
            rest = rest.lstrip("/")
        else:
            rest = rest.replace("\\", "/")
            if rest.startswith("//"):
                rest = rest[2:]
            else:
                return JsUrl(scheme + ":", "", "", "", "", rest or "/", search, hash_)
        slash = rest.find("/")
        authority, path = (rest, "") if slash < 0 else (rest[:slash], rest[slash:])
        path = path or "/"
    else:
        if not rest.startswith("//"):
            return JsUrl(scheme + ":", "", "", "", "", rest, search, hash_)
        rest = rest[2:]
        slash = rest.find("/")
        authority, path = (rest, "") if slash < 0 else (rest[:slash], rest[slash:])
    username = password = ""
    if "@" in authority:
        userinfo, authority = authority.rsplit("@", 1)
        if ":" in userinfo:
            username, password = userinfo.split(":", 1)
        else:
            username = userinfo
    host = authority
    port = ""
    if host.startswith("["):
        end = host.find("]")
        if end < 0:
            return None
        port_part = host[end + 1:]
        host = host[:end + 1]
        if port_part:
            if not port_part.startswith(":"):
                return None
            port = port_part[1:]
    elif ":" in host:
        host, port = host.rsplit(":", 1)
    if port:
        if not port.isascii() or not port.isdigit() or int(port) > 65535:
            return None
        port = str(int(port))
        if _SPECIAL.get(scheme) == port:
            port = ""
    host = host.lower()
    if special and scheme != "file" and not host:
        return None
    for ch in host:
        if ch in " #%/:<>?@\\^|" and not host.startswith("["):
            return None
    return JsUrl(scheme + ":", username, password, host, port, path, search, hash_)
