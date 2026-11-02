"""Port of src/lib/ingest-cache.ts.

SHA-256 based ingest cache stored at ``<project>/.llm-wiki/ingest-cache.json``
(format identical to the TS one: ``{"entries": {<source>: {hash, timestamp, filesWritten}}}``,
``timestamp`` in epoch milliseconds, 2-space indented JSON).

All functions are synchronous (file IO); call via ``asyncio.to_thread`` from async code.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

from .path_utils import is_absolute_path, normalize_path

log = logging.getLogger(__name__)

_cache_lock = threading.Lock()
_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def _fs():
    from ..commands import fs

    return fs


def sha256(content: str) -> str:
    """Hex SHA-256 of the UTF-8 encoding (lone surrogates become U+FFFD like TextEncoder)."""
    data = _SURROGATE_RE.sub("�", content).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def cache_path(project_path: str) -> str:
    return f"{normalize_path(project_path)}/.llm-wiki/ingest-cache.json"


def _load_cache(project_path: str) -> Dict[str, Any]:
    try:
        raw = _fs().read_file(cache_path(project_path))
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
            return {"entries": {}}
        return data
    except Exception:
        return {"entries": {}}


def _save_cache(project_path: str, cache: Dict[str, Any]) -> None:
    try:
        _fs().write_file(cache_path(project_path), json.dumps(cache, indent=2, ensure_ascii=False))
    except Exception:
        pass  # non-critical


def list_ingested_source_identities(project_path: str) -> List[str]:
    """Source identities with a recorded successful ingest result."""
    return list(_load_cache(project_path)["entries"].keys())


def check_ingest_cache(project_path: str, source_file_name: str, source_content: str) -> Optional[List[str]]:
    """Previously written files if the cache entry is current and every file still exists, else None."""
    cache = _load_cache(project_path)
    entry = cache["entries"].get(source_file_name)
    if not entry:
        return None

    if entry.get("hash") != sha256(source_content):
        return None

    pp = normalize_path(project_path)
    for file_path in entry.get("filesWritten", []):
        full_path = normalize_path(file_path) if is_absolute_path(file_path) else f"{pp}/{file_path}"
        try:
            if not _fs().file_exists(full_path):
                log.info("[ingest-cache] cache miss for %s: %s no longer on disk", source_file_name, file_path)
                return None
        except Exception:
            return None
    return entry["filesWritten"]


def save_ingest_cache(
    project_path: str, source_file_name: str, source_content: str, files_written: List[str]
) -> None:
    with _cache_lock:
        cache = _load_cache(project_path)
        new_entries = dict(cache["entries"])
        new_entries[source_file_name] = {
            "hash": sha256(source_content),
            "timestamp": int(time.time() * 1000),
            "filesWritten": list(files_written),
        }
        _save_cache(project_path, {"entries": new_entries})


def remove_from_ingest_cache(project_path: str, source_file_name: str) -> None:
    with _cache_lock:
        cache = _load_cache(project_path)
        new_entries = dict(cache["entries"])
        new_entries.pop(source_file_name, None)
        _save_cache(project_path, {"entries": new_entries})


def move_ingest_cache_entry(
    project_path: str,
    old_source_identity: str,
    new_source_identity: str,
    moved_files: Optional[Mapping[str, str]] = None,
) -> None:
    """Move a cache entry when an unchanged source is renamed inside raw/sources."""
    moved_files = moved_files or {}
    with _cache_lock:
        cache = _load_cache(project_path)
        entry = cache["entries"].get(old_source_identity)
        if not entry or old_source_identity == new_source_identity:
            return
        migrated = {**entry, "filesWritten": [moved_files.get(p, p) for p in entry.get("filesWritten", [])]}
        new_entries = {**cache["entries"], new_source_identity: migrated}
        del new_entries[old_source_identity]
        _save_cache(project_path, {"entries": new_entries})
