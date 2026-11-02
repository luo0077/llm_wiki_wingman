"""Port of src/lib/source-identity.ts (source identity + summary-page slugs)."""
from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

from ._js import (
    encode_uri_component,
    js_trim,
    rx,
    u16len,
    u16slice,
    u16units,
)
from .path_utils import get_file_name, normalize_path

RAW_SOURCES_PREFIX = "raw/sources/"
RAW_SOURCES_MARKER = "/raw/sources/"
MAX_SOURCE_SUMMARY_SLUG_LENGTH = 120
FALLBACK_SOURCE_PART = "source"

_EXT_RE = re.compile(r"\.[^/.]+\Z")
_WS_RUN_RE = rx(r"\s+")
_TRAILING_HYPHENS_RE = re.compile(r"-+\Z")


def _keep_letters_numbers_hyphen(s: str) -> str:
    """JS .replace(/[^\\p{L}\\p{N}-]/gu, "")."""
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch == "-" or cat[0] == "L" or cat[0] == "N":
            out.append(ch)
    return "".join(out)


def source_identity_for_path(project_path: str, source_path: str) -> str:
    pp = re.sub(r"/+\Z", "", normalize_path(project_path))
    sp = normalize_path(source_path)
    project_raw_sources_prefix = f"{pp}/{RAW_SOURCES_PREFIX}"
    sp_key = sp.lower()
    if sp_key.startswith(project_raw_sources_prefix.lower()):
        return u16slice(sp, u16len(project_raw_sources_prefix))
    if sp_key.startswith(RAW_SOURCES_PREFIX):
        return u16slice(sp, len(RAW_SOURCES_PREFIX))
    marker_index = sp_key.find(RAW_SOURCES_MARKER)
    if marker_index >= 0:
        # Index is into the lowercased string; JS slices the original by the same
        # UTF-16 offset (lowercasing can change length for exotic characters).
        return u16slice(sp, u16len(sp_key[:marker_index]) + len(RAW_SOURCES_MARKER))
    return get_file_name(sp)


def source_reference_identity(source_reference: str) -> str:
    ref = normalize_path(source_reference)
    ref_key = ref.lower()
    if ref_key.startswith(RAW_SOURCES_PREFIX):
        return u16slice(ref, len(RAW_SOURCES_PREFIX))
    marker_index = ref_key.find(RAW_SOURCES_MARKER)
    if marker_index >= 0:
        return u16slice(ref, u16len(ref_key[:marker_index]) + len(RAW_SOURCES_MARKER))
    return ref


def _parts(source_identity: str) -> List[str]:
    without_ext = _EXT_RE.sub("", source_identity, count=1)
    return [p for p in (js_trim(x) for x in without_ext.split("/")) if p]


def source_summary_slug_from_identity(source_identity: str) -> str:
    parts = _parts(source_identity)
    if len(parts) <= 1:
        return parts[0] if parts and parts[0] else "source"

    hash_ = _stable_slug_hash(source_identity)
    segs = []
    for part in parts:
        readable, structural_length = _readable_slug_part(part)
        segs.append(f"{structural_length}-{readable}")
    slug = "--".join(segs)
    full_slug = f"{slug}--{hash_}"
    if u16len(full_slug) <= MAX_SOURCE_SUMMARY_SLUG_LENGTH:
        return full_slug

    readable_limit = MAX_SOURCE_SUMMARY_SLUG_LENGTH - u16len(hash_) - 2
    readable_prefix = _TRAILING_HYPHENS_RE.sub("", u16slice(slug, 0, readable_limit))
    return f"{readable_prefix or 'source'}--{hash_}"


def legacy_source_summary_slug_from_identity(source_identity: str) -> str:
    parts = _parts(source_identity)
    if len(parts) <= 1:
        return parts[0] if parts and parts[0] else "source"

    hash_ = _stable_slug_hash(source_identity)
    segs = []
    for part in parts:
        encoded = encode_uri_component(part)
        segs.append(f"{len(encoded)}-{encoded}")
    return "--".join(segs) + f"--{hash_}"


def source_summary_slug_candidates_from_identity(source_identity: str) -> List[str]:
    canonical = source_summary_slug_from_identity(source_identity)
    previous_readable = _previous_readable_source_summary_slug_from_identity(source_identity)
    legacy = legacy_source_summary_slug_from_identity(source_identity)
    return list(dict.fromkeys([canonical, previous_readable, legacy]))


def _previous_readable_source_summary_slug_from_identity(source_identity: str) -> str:
    parts = _parts(source_identity)
    if len(parts) <= 1:
        return parts[0] if parts and parts[0] else "source"

    hash_ = _stable_slug_hash(source_identity)
    segs = []
    for part in parts:
        readable, _ = _readable_slug_part(part)
        segs.append(f"{len(readable)}-{readable}")
    slug = "--".join(segs)
    full_slug = f"{slug}--{hash_}"
    if u16len(full_slug) <= MAX_SOURCE_SUMMARY_SLUG_LENGTH:
        return full_slug

    readable_limit = MAX_SOURCE_SUMMARY_SLUG_LENGTH - u16len(hash_) - 2
    readable_prefix = _TRAILING_HYPHENS_RE.sub("", u16slice(slug, 0, readable_limit))
    return f"{readable_prefix or 'source'}--{hash_}"


_LEAD_TRAIL_HYPHEN_RE = re.compile(r"\A-|-\Z")
_HYPHEN_RUN_RE = re.compile(r"-+")


def _readable_slug_part(part: str) -> Tuple[str, int]:
    structural = js_trim(unicodedata.normalize("NFKC", part))
    structural = _WS_RUN_RE.sub("-", structural)
    structural = _keep_letters_numbers_hyphen(structural)
    structural = _LEAD_TRAIL_HYPHEN_RE.sub("", structural)
    structural = structural.lower()
    readable = _HYPHEN_RUN_RE.sub("-", structural) or FALLBACK_SOURCE_PART
    return readable, max(1, len(structural or FALLBACK_SOURCE_PART))


def _to_base36(n: int) -> str:
    if n == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = []
    while n:
        n, r = divmod(n, 36)
        out.append(digits[r])
    return "".join(reversed(out))


def _stable_slug_hash(value: str) -> str:
    h = 0x811C9DC5
    for unit in u16units(value):
        h ^= unit
        h = (h * 0x01000193) & 0xFFFFFFFF
    return _to_base36(h)
