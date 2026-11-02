"""Port of src/lib/path-utils.ts."""
from __future__ import annotations

import re
from typing import Optional

from ._js import rx


def normalize_path(p: str) -> str:
    return p.replace("\\", "/")


def join_path(*segments: str) -> str:
    return re.sub(r"/+", "/", "/".join(s.replace("\\", "/") for s in segments))


def get_file_name(p: str) -> str:
    normalized = p.replace("\\", "/")
    return normalized.split("/")[-1]


def get_file_stem(p: str) -> str:
    name = get_file_name(p)
    last_dot = name.rfind(".")
    return name[:last_dot] if last_dot > 0 else name


_DRIVE_RE = re.compile(r"^[A-Za-z]:/")


def _case_fold_path(normalized: str) -> str:
    return normalized.lower() if (_DRIVE_RE.match(normalized) or normalized.startswith("//")) else normalized


def get_relative_path(full_path: str, base_path: str) -> str:
    normal_full = normalize_path(full_path)
    normal_base = re.sub(r"/\Z", "", normalize_path(base_path))
    full_key = _case_fold_path(normal_full)
    base_key = _case_fold_path(normal_base)
    if full_key.startswith(base_key + "/"):
        return "/".join(normal_full.split("/")[len(normal_base.split("/")):])
    return normal_full


_ABS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def is_absolute_path(p: Optional[str]) -> bool:
    if not p:
        return False
    if p.startswith("/"):
        return True
    if _ABS_DRIVE_RE.match(p):
        return True
    if p.startswith("\\\\") or p.startswith("//"):
        return True
    return False
