"""Port of src/lib/wiki-schema.ts (schema.md "Page Types" routing)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ._js import JS_DOT, Record, js_trim, rx
from .frontmatter import parse_frontmatter


@dataclass
class WikiSchemaRouting(Record):
    typeDirs: Dict[str, str] = field(default_factory=dict)


@dataclass
class WikiSchemaRoutingIssue(Record):
    message: str


@dataclass
class WikiSchemaRouteCorrection(Record):
    path: str
    message: Optional[str]


def load_project_wiki_schema_routing(project_path: str) -> Optional[WikiSchemaRouting]:
    """Sync (file IO). Returns None when schema.md is missing/empty/has no routing."""
    try:
        from ..commands import fs as _fs

        raw = _fs.read_file(re.sub(r"/+\Z", "", project_path) + "/schema.md")
    except Exception:
        return None
    if not js_trim(raw):
        return None
    routing = parse_wiki_schema_routing(raw)
    return routing if len(routing.typeDirs) > 0 else None


_HEADING_RE = rx(r"\A(#{1,6})\s+(" + JS_DOT + r"+?)\s*#*\Z")
_PAGE_TYPES_RE = rx(r"\Apage\s+types\Z", re.I)
_TYPE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*\Z", re.I | re.A)
_HEADING_PREFIX_RE = rx(r"\A(#{1,6})\s+")
_HEADING_LEVEL_RE = re.compile(r"\A(#{1,6})")


def parse_wiki_schema_routing(markdown: str) -> WikiSchemaRouting:
    type_dirs: Dict[str, str] = {}
    for line in _page_types_section_lines(markdown):
        if not js_trim(line).startswith("|"):
            continue
        cells = [js_trim(cell) for cell in line.split("|")[1:-1]]
        if len(cells) < 2:
            continue
        type_, dir_ = cells[0], cells[1]
        if not _TYPE_NAME_RE.search(type_):
            continue
        if not _is_safe_wiki_directory(dir_):
            continue
        type_dirs[type_] = _strip_trailing_slash(dir_)
    return WikiSchemaRouting(typeDirs=type_dirs)


def _page_types_section_lines(markdown: str) -> List[str]:
    lines = markdown.split("\n")
    start = -1
    for i, line in enumerate(lines):
        match = _HEADING_RE.search(js_trim(line))
        if match and _PAGE_TYPES_RE.search(js_trim(match.group(2))):
            start = i
            break
    if start < 0:
        return []
    lm = _HEADING_LEVEL_RE.search(js_trim(lines[start]))
    heading_level = len(lm.group(1)) if lm else 6
    out: List[str] = []
    for line in lines[start + 1:]:
        heading = _HEADING_PREFIX_RE.search(js_trim(line))
        if heading and len(heading.group(1)) <= heading_level:
            break
        out.append(line)
    return out


def validate_wiki_page_routing(
    relative_path: str, content: str, routing: WikiSchemaRouting
) -> Optional[WikiSchemaRoutingIssue]:
    parsed = parse_frontmatter(content)
    type_ = parsed.frontmatter.get("type") if parsed.frontmatter else None
    if not isinstance(type_, str) or not js_trim(type_):
        return None

    normalized_path = _normalize_relative_path(relative_path)
    actual_dir = _dirname(normalized_path)
    expected_dir = routing.typeDirs.get(type_)
    if expected_dir and actual_dir != expected_dir:
        return WikiSchemaRoutingIssue(
            message=f'Page type "{type_}" must be under "{expected_dir}/". Current directory: "{actual_dir}".'
        )

    type_from_path = _infer_type_from_schema_path(normalized_path, routing)
    if type_from_path and type_from_path != type_:
        return WikiSchemaRoutingIssue(
            message=f'Pages under "{actual_dir}/" must use type "{type_from_path}", but found "{type_}".'
        )
    return None


def correct_wiki_page_routing(
    relative_path: str, content: str, routing: WikiSchemaRouting
) -> WikiSchemaRouteCorrection:
    parsed = parse_frontmatter(content)
    type_ = parsed.frontmatter.get("type") if parsed.frontmatter else None
    normalized_path = _normalize_relative_path(relative_path)
    if not isinstance(type_, str) or not js_trim(type_):
        return WikiSchemaRouteCorrection(path=normalized_path, message=None)

    expected_dir = routing.typeDirs.get(type_)
    actual_dir = _dirname(normalized_path)
    if not expected_dir or actual_dir == expected_dir:
        return WikiSchemaRouteCorrection(path=normalized_path, message=None)

    file_name = normalized_path[normalized_path.rfind("/") + 1:]
    return WikiSchemaRouteCorrection(
        path=f"{expected_dir}/{file_name}",
        message=f'Auto-routed page type "{type_}" from "{actual_dir}/" to "{expected_dir}/".',
    )


def _infer_type_from_schema_path(relative_path: str, routing: WikiSchemaRouting) -> Optional[str]:
    actual_dir = _dirname(relative_path)
    for type_, dir_ in routing.typeDirs.items():
        if actual_dir == dir_:
            return type_
    return None


def _normalize_relative_path(relative_path: str) -> str:
    return re.sub(r"\A/+", "", relative_path.replace("\\", "/"))


def _dirname(relative_path: str) -> str:
    normalized = _normalize_relative_path(relative_path)
    index = normalized.rfind("/")
    return normalized[:index] if index >= 0 else "."


def _strip_trailing_slash(value: str) -> str:
    return re.sub(r"/+\Z", "", value)


_BAD_SEGMENT_CHARS = re.compile(r'[<>:"|?*\\]')


def _is_safe_wiki_directory(value: str) -> bool:
    without_trailing_slash = _strip_trailing_slash(value)
    normalized = _normalize_relative_path(without_trailing_slash)
    if normalized != without_trailing_slash:
        return False
    if normalized != "wiki" and not normalized.startswith("wiki/"):
        return False
    return all(
        len(seg) > 0 and seg != "." and seg != ".." and not _BAD_SEGMENT_CHARS.search(seg)
        for seg in normalized.split("/")
    )
