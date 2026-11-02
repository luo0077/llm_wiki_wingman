"""Port of src/lib/frontmatter.ts (YAML frontmatter parsing with LLM-corruption recovery).

YAML is parsed with PyYAML restricted to the js-yaml ``JSON_SCHEMA`` (no
timestamps, no YAML 1.1 booleans/sexagesimal ints, no merge keys, duplicate
keys are an error).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from ._js import Record, js_json_stringify, js_number_to_string, js_trim, rx

FrontmatterValue = Union[str, List[str]]


@dataclass
class FrontmatterParseResult(Record):
    frontmatter: Optional[Dict[str, FrontmatterValue]]
    body: str
    rawBlock: str


# ── js-yaml JSON_SCHEMA emulation ─────────────────────────────────

_NULL_RE = re.compile(r"^(?:~|null|Null|NULL|)\Z")
_BOOL_RE = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)\Z")
_INT_RE = re.compile(
    r"^[-+]?(?:0b[01_]*[01]|0x[0-9a-fA-F_]*[0-9a-fA-F]|0o[0-7_]*[0-7]|0|0(?:[0-9]|[0-9][0-9_]*[0-9])|[1-9](?:[0-9_]*[0-9])?)\Z"
)
_FLOAT_RE = re.compile(
    r"^(?:[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?(?:[eE][-+]?[0-9]+)?"
    r"|\.[0-9_]+(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF)"
    r"|\.(?:nan|NaN|NAN))(?<!_)\Z"
)

_T_NULL = "tag:yaml.org,2002:null"
_T_BOOL = "tag:yaml.org,2002:bool"
_T_INT = "tag:yaml.org,2002:int"
_T_FLOAT = "tag:yaml.org,2002:float"


class DuplicateKeyError(yaml.YAMLError):
    pass


class _JsonSchemaLoader(yaml.SafeLoader):
    # Start from an empty implicit-resolver table (do not inherit YAML 1.1 ones).
    yaml_implicit_resolvers: Dict[Any, Any] = {}


_JsonSchemaLoader.add_implicit_resolver(_T_NULL, _NULL_RE, ["~", "n", "N", ""])
_JsonSchemaLoader.add_implicit_resolver(_T_BOOL, _BOOL_RE, list("tTfF"))
_JsonSchemaLoader.add_implicit_resolver(_T_INT, _INT_RE, list("-+0123456789"))
_JsonSchemaLoader.add_implicit_resolver(_T_FLOAT, _FLOAT_RE, list("-+0123456789."))


def _construct_null(loader, node):
    return None


def _construct_bool(loader, node):
    return loader.construct_scalar(node).lower() == "true"


def _construct_int(loader, node):
    value = loader.construct_scalar(node).replace("_", "")
    sign = 1
    if value[0] in "+-":
        if value[0] == "-":
            sign = -1
        value = value[1:]
    if value == "0":
        return 0
    if value[:2] == "0b":
        return sign * int(value[2:], 2)
    if value[:2] == "0x":
        return sign * int(value[2:], 16)
    if value[:2] == "0o":
        return sign * int(value[2:], 8)
    return sign * int(value, 10)


def _construct_float(loader, node):
    value = loader.construct_scalar(node).replace("_", "").lower()
    sign = 1.0
    if value[0] in "+-":
        if value[0] == "-":
            sign = -1.0
        value = value[1:]
    if value == ".inf":
        return sign * math.inf
    if value == ".nan":
        return math.nan
    return sign * float(value)


def _key_string(key: Any) -> str:
    if isinstance(key, str):
        return key
    if key is None:
        return "null"
    if key is True:
        return "true"
    if key is False:
        return "false"
    if isinstance(key, (int, float)):
        return js_number_to_string(key)
    if isinstance(key, list):
        return ",".join("" if x is None else _key_string(x) for x in key)
    return "[object Object]"


def _construct_mapping(loader, node, deep=False):
    if not isinstance(node, yaml.MappingNode):
        raise yaml.constructor.ConstructorError(None, None, "expected a mapping node", node.start_mark)
    mapping: Dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = _key_string(loader.construct_object(key_node, deep=True))
        value = loader.construct_object(value_node, deep=deep)
        if key in mapping:
            raise DuplicateKeyError("duplicated mapping key")
        mapping[key] = value
    return _js_property_order(mapping)


def _js_property_order(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """JS objects enumerate array-index-like keys first (ascending), then the rest in insertion order."""
    index_keys = [k for k in mapping if _is_array_index(k)]
    if not index_keys:
        return mapping
    index_keys.sort(key=int)
    ordered = {k: mapping[k] for k in index_keys}
    for k, v in mapping.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _is_array_index(k: str) -> bool:
    return k.isascii() and k.isdigit() and (k == "0" or k[0] != "0") and int(k) < 4294967295


_JsonSchemaLoader.add_constructor(_T_NULL, _construct_null)
_JsonSchemaLoader.add_constructor(_T_BOOL, _construct_bool)
_JsonSchemaLoader.add_constructor(_T_INT, _construct_int)
_JsonSchemaLoader.add_constructor(_T_FLOAT, _construct_float)
_JsonSchemaLoader.construct_mapping = _construct_mapping  # type: ignore[assignment]


def _yaml_load(payload: str) -> Any:
    # js-yaml treats end-of-input as a line break for block scalars ("lit" -> "lit\n").
    return yaml.load(payload + "\n", Loader=_JsonSchemaLoader)


# ── block location ────────────────────────────────────────────────

FM_BLOCK_STRICT_RE = rx(r"\A---\s*\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|\Z)")
FM_BLOCK_ANYWHERE_RE = rx(r"\n---\s*\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|\Z)")
MAX_PREFIX_LINES_BEFORE_FRONTMATTER = 6

_YAML_FENCE_PREFIX_RE = rx(r"\A\s*```(?:yaml|yml)?\s*\r?\n\Z", re.I)
_CLOSING_FENCE_RE = rx(r"\A\s*```\s*(?:\r?\n|\Z)")
_WIKILINK_LIST_RE = rx(
    r"\A(\s*[A-Za-z_][\w-]*\s*:\s*)(\[\[[^\]]+\]\](?:\s*,\s*\[\[[^\]]+\]\])+)\s*\Z"
)


def parse_frontmatter(content: str) -> FrontmatterParseResult:
    located = _locate_frontmatter_block(content)
    if not located:
        return FrontmatterParseResult(frontmatter=None, body=content, rawBlock="")

    yaml_payload, raw_block, body = located

    try:
        parsed = _yaml_load(yaml_payload)
    except Exception:
        try:
            parsed = _yaml_load(_repair_wikilink_lists(yaml_payload))
        except Exception:
            return FrontmatterParseResult(frontmatter=None, body=body, rawBlock=raw_block)

    return FrontmatterParseResult(frontmatter=_normalize(parsed), body=body, rawBlock=raw_block)


def _locate_frontmatter_block(content: str) -> Optional[Tuple[str, str, str]]:
    strict = FM_BLOCK_STRICT_RE.search(content)
    if strict:
        return strict.group(1), strict.group(0), content[len(strict.group(0)):]

    fallback = FM_BLOCK_ANYWHERE_RE.search(content)
    if not fallback:
        return None

    open_idx = fallback.start() + 1  # skip the leading "\n"
    if _line_number_at(content, open_idx) > MAX_PREFIX_LINES_BEFORE_FRONTMATTER:
        return None

    raw_block = content[open_idx:open_idx + len(fallback.group(0)) - 1]
    body_after_fm = content[open_idx + len(raw_block):]

    prefix = content[:open_idx]
    if _YAML_FENCE_PREFIX_RE.search(prefix):
        stripped = _CLOSING_FENCE_RE.sub("", body_after_fm, count=1)
        return fallback.group(1), raw_block, stripped

    return fallback.group(1), raw_block, body_after_fm


def _line_number_at(s: str, index: int) -> int:
    return 1 + s.count("\n", 0, index)


def _repair_wikilink_lists(payload: str) -> str:
    out = []
    for line in payload.split("\n"):
        m = _WIKILINK_LIST_RE.search(line)
        if not m:
            out.append(line)
            continue
        prefix = m.group(1)
        items = ", ".join(
            '"%s"' % t for t in (js_trim(s) for s in m.group(2).split(",")) if t
        )
        out.append("%s[%s]" % (prefix, items))
    return "\n".join(out)


def _normalize(parsed: Any) -> Optional[Dict[str, FrontmatterValue]]:
    if not isinstance(parsed, dict):
        return None
    out: Dict[str, FrontmatterValue] = {}
    for key, value in parsed.items():
        if isinstance(value, list):
            out[key] = [_stringify_scalar(v) for v in value]
            continue
        out[key] = _stringify_scalar(value)
    return out


def _stringify_scalar(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return js_number_to_string(v)
    return js_json_stringify(v)
