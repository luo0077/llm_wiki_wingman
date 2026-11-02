"""Port of commands/file_history.rs.

On-disk format is identical to the Rust version: `<project>/.llm-wiki/history/<fnv1a>.json`
holds a compact JSON array of `{id, path, timestamp, author, tool, content}` and
`<project>/.llm-wiki/history-settings.json` holds `{enabled, maxVersionsPerFile}`.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..errors import CommandError
from ..registry import command

MAX_HISTORY_CONTENT_BYTES = 512 * 1024
DEFAULT_ENTRIES_PER_FILE = 10
MAX_ENTRIES_PER_FILE = 30
MAX_HISTORY_STORE_BYTES = 128 * 1024 * 1024
MAX_HISTORY_STORE_FILES = 2048
_HISTORY_LOCK = threading.Lock()

_ENTRY_FIELDS = (("id", str), ("path", str), ("timestamp", int), ("author", str), ("tool", str), ("content", str))


class _Err(Exception):
    """Internal error carrying the Rust Err(String) text."""


def _io_err(e: BaseException) -> str:
    if isinstance(e, FileNotFoundError):
        return "No such file or directory (os error 2)"
    if isinstance(e, PermissionError):
        return "Permission denied (os error 13)"
    return str(e)


# ---------------------------------------------------------------- settings

def _usize(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _parse_settings(raw: Any) -> Optional[Dict[str, Any]]:
    """serde `FileHistorySettings` (camelCase, default) deserialisation; None on failure."""
    if not isinstance(raw, dict):
        return None
    out = {"enabled": False, "maxVersionsPerFile": DEFAULT_ENTRIES_PER_FILE}
    if "enabled" in raw:
        if not isinstance(raw["enabled"], bool):
            return None
        out["enabled"] = raw["enabled"]
    if "maxVersionsPerFile" in raw:
        if not _usize(raw["maxVersionsPerFile"]):
            return None
        out["maxVersionsPerFile"] = raw["maxVersionsPerFile"]
    return out


def _normalized(s: Dict[str, Any]) -> Dict[str, Any]:
    return {"enabled": s["enabled"], "maxVersionsPerFile": min(s["maxVersionsPerFile"], MAX_ENTRIES_PER_FILE)}


def _settings_path(root: Path) -> Path:
    return root / ".llm-wiki" / "history-settings.json"


def _validate_optional_regular_file(path: Path, label: str) -> None:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as e:
        raise _Err(_io_err(e))
    import stat

    if not stat.S_ISREG(st.st_mode):
        raise _Err(f"{label} must be a regular file, not a symlink")


def _read_settings(root: Path) -> Dict[str, Any]:
    path = _settings_path(root)
    default = {"enabled": False, "maxVersionsPerFile": DEFAULT_ENTRIES_PER_FILE}
    try:
        _validate_optional_regular_file(path, "History settings")
    except _Err:
        return default
    try:
        parsed = _parse_settings(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        parsed = None
    return _normalized(parsed or default)


# ---------------------------------------------------------------- helpers

def _dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _parse_entries(raw: str) -> Optional[List[Dict[str, Any]]]:
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, list):
        return None
    for e in data:
        if not isinstance(e, dict):
            return None
        for k, t in _ENTRY_FIELDS:
            v = e.get(k)
            if not isinstance(v, t) or (t is int and isinstance(v, bool)):
                return None
    return data


def _is_dir_nolink(p: Path) -> bool:
    return p.is_dir() and not p.is_symlink()


def _project_root_for(path: Path) -> Optional[Path]:
    for d in path.parents:
        if _is_dir_nolink(d / ".llm-wiki"):
            return d
    return None


def _within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _history_path(root: Path, path: Path) -> Path:
    rel = str(path.relative_to(root)) if _within(path, root) else str(path)
    h = 0xCBF29CE484222325
    for b in rel.encode("utf-8", "surrogateescape"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return root / ".llm-wiki" / "history" / f"{h:016x}.json"


def _history_dir(root: Path) -> Path:
    return root / ".llm-wiki" / "history"


def _checked_history_file(root: Path, path: Path) -> Path:
    p = _history_path(root, path)
    _validate_optional_regular_file(p, "History file")
    return p


def _checked_history_dir(root: Path, create: bool) -> Path:
    try:
        canonical_root = Path(os.path.realpath(root))
        if not canonical_root.exists():
            raise FileNotFoundError()
        meta_dir = canonical_root / ".llm-wiki"
        if not os.path.lexists(meta_dir):
            raise FileNotFoundError()
        if not _is_dir_nolink(meta_dir):
            raise _Err("History metadata directory must be a real project directory")
        d = meta_dir / "history"
        if create:
            d.mkdir(parents=True, exist_ok=True)
        elif not d.exists():
            return d
        if not _is_dir_nolink(d):
            raise _Err("History directory must not be a symlink")
        cd = Path(os.path.realpath(d))
        if not _within(cd, meta_dir):
            raise _Err("History directory must stay inside the project")
        return cd
    except _Err:
        raise
    except OSError as e:
        raise _Err(_io_err(e))


def _canonicalize(p: str) -> Path:
    if not os.path.exists(p):
        raise _Err("No such file or directory (os error 2)")
    return Path(os.path.realpath(p))


def _files_oldest_first(root: Path) -> List[Tuple[Path, int, float]]:
    d = _history_dir(root)
    out: List[Tuple[Path, int, float]] = []
    try:
        it = list(os.scandir(d))
    except OSError:
        return out
    for e in it:
        try:
            if e.is_symlink() or not e.is_file(follow_symlinks=False):
                continue
            if not e.name.endswith(".json") or e.name == ".json":
                continue
            st = e.stat(follow_symlinks=False)
            out.append((Path(e.path), st.st_size, st.st_mtime))
        except OSError:
            continue
    out.sort(key=lambda t: t[2])
    return out


def _prune_store(root: Path, protected: Path) -> None:
    files = _files_oldest_first(root)
    total_bytes = sum(f[1] for f in files)
    total_files = len(files)
    for path, size, _ in files:
        if total_bytes <= MAX_HISTORY_STORE_BYTES and total_files <= MAX_HISTORY_STORE_FILES:
            break
        if path == protected:
            continue
        try:
            os.remove(path)
        except OSError:
            continue
        total_bytes = max(0, total_bytes - size)
        total_files = max(0, total_files - 1)


def _prune_entries_per_file(root: Path, max_versions: int) -> None:
    d = _checked_history_dir(root, False)
    if not d.exists():
        return
    if max_versions == 0:
        try:
            shutil.rmtree(d)
        except OSError as e:
            raise _Err(_io_err(e))
        return
    for path, _, _ in _files_oldest_first(root):
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        entries = _parse_entries(raw)
        if entries is None:
            continue
        if len(entries) > max_versions:
            entries = entries[len(entries) - max_versions:]
            try:
                path.write_text(_dumps(entries), encoding="utf-8")
            except OSError as e:
                raise _Err(_io_err(e))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _norm_path(p: Path) -> str:
    return str(p).replace("\\", "/")


# ---------------------------------------------------------------- public helpers

def latest_file_version(path: Any) -> Optional[Tuple[int, str, str]]:
    """(timestamp, author, tool) of the newest entry; provenance only."""
    try:
        path = Path(path)
        root = _project_root_for(path)
        if root is None:
            return None
        with _HISTORY_LOCK:
            _checked_history_dir(root, False)
            store = _checked_history_file(root, path)
            entries = _parse_entries(store.read_text(encoding="utf-8"))
        if not entries:
            return None
        last = entries[-1]
        return (last["timestamp"], last["author"], last["tool"])
    except Exception:
        return None


def record_file_version(path: Any, kind: str, source: str) -> None:
    """Snapshot `path` into the project's history (Rust `record_file_version(path, author, tool)`).

    `kind` is the author ("agent", "human", "baseline"), `source` the tool name
    ("wiki.write_page"). Never raises.
    """
    try:
        _record(Path(path), kind, source)
    except Exception:
        return


def _record(path: Path, author: str, tool: str) -> None:
    root = _project_root_for(path)
    if root is None:
        return
    if _within(path, root / ".llm-wiki"):
        return
    with _HISTORY_LOCK:
        settings = _read_settings(root)
        if not settings["enabled"] or settings["maxVersionsPerFile"] == 0:
            return
        try:
            st = os.stat(path)
        except OSError:
            return
        import stat

        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_HISTORY_CONTENT_BYTES:
            return
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        try:
            d = _checked_history_dir(root, True)
            checked = _checked_history_file(root, path)
        except _Err:
            return
        store_path = d / checked.name
        entries: List[Dict[str, Any]] = []
        try:
            entries = _parse_entries(store_path.read_text(encoding="utf-8")) or []
        except (OSError, UnicodeDecodeError):
            entries = []
        if entries and entries[-1]["content"] == content:
            return
        entries.append({
            "id": str(uuid.uuid4()),
            "path": _norm_path(path),
            "timestamp": _now_ms(),
            "author": author,
            "tool": tool,
            "content": content,
        })
        mx = settings["maxVersionsPerFile"]
        if len(entries) > mx:
            entries = entries[len(entries) - mx:]
        try:
            store_path.write_text(_dumps(entries), encoding="utf-8")
        except OSError:
            return
        _prune_store(root, store_path)


# ---------------------------------------------------------------- commands

def _checked_project_root(project_path: str) -> Path:
    root = _canonicalize(project_path)
    meta = root / ".llm-wiki"
    if not os.path.lexists(meta):
        raise _Err("No such file or directory (os error 2)")
    if not _is_dir_nolink(meta):
        raise _Err("History project must contain .llm-wiki")
    return root


def _checked_file(project_path: str, file_path: str) -> Tuple[Path, Path]:
    root = _canonicalize(project_path)
    f = _canonicalize(file_path)
    if not _within(f, root) or _within(f, root / ".llm-wiki"):
        raise _Err("History path must stay inside the project")
    return root, f


def _wrap(fn):
    try:
        return fn()
    except _Err as e:
        raise CommandError(str(e))
    except OSError as e:
        raise CommandError(_io_err(e))


@command()
def get_file_history_settings(project_path: str):
    def run():
        return _read_settings(_checked_project_root(project_path))

    return _wrap(run)


@command()
def set_file_history_settings(project_path: str, settings: Dict[str, Any]):
    def run():
        root = _checked_project_root(project_path)
        parsed = _parse_settings(settings)
        if parsed is None:
            raise _Err("invalid args `settings` for command `set_file_history_settings`")
        s = _normalized(parsed)
        if s["enabled"] and s["maxVersionsPerFile"] == 0:
            s["maxVersionsPerFile"] = DEFAULT_ENTRIES_PER_FILE
        with _HISTORY_LOCK:
            previous = _read_settings(root)
            sp = _settings_path(root)
            had = False
            try:
                _validate_optional_regular_file(sp, "History settings")
                had = _parse_settings(json.loads(sp.read_text(encoding="utf-8"))) is not None
            except (_Err, OSError, ValueError):
                had = False
            raw = json.dumps(s, ensure_ascii=False, indent=2)
            _validate_optional_regular_file(sp, "History settings")
            sp.write_text(raw, encoding="utf-8")
            if not had or s["maxVersionsPerFile"] < previous["maxVersionsPerFile"]:
                _prune_entries_per_file(root, s["maxVersionsPerFile"])
        return s

    return _wrap(run)


@command()
def list_file_history(project_path: str, file_path: str):
    def run():
        root, f = _checked_file(project_path, file_path)
        with _HISTORY_LOCK:
            _checked_history_dir(root, False)
            store = _checked_history_file(root, f)
            try:
                raw = store.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                raw = "[]"
        entries = _parse_entries(raw) or []
        entries.reverse()
        return entries

    return _wrap(run)


@command()
def restore_file_history(project_path: str, file_path: str, entry_id: str):
    def run():
        root, f = _checked_file(project_path, file_path)
        with _HISTORY_LOCK:
            _checked_history_dir(root, False)
            store = _checked_history_file(root, f)
            try:
                raw = store.read_text(encoding="utf-8")
            except OSError as e:
                raise _Err(_io_err(e))
            except UnicodeDecodeError:
                raise _Err("stream did not contain valid UTF-8")
            entries = _parse_entries(raw)
            if entries is None:
                raise _Err("invalid history file")
            entry = next((e for e in entries if e["id"] == entry_id), None)
            if entry is None:
                raise _Err("History entry not found")
            try:
                current = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise _Err("stream did not contain valid UTF-8")
            if len(current.encode("utf-8")) > MAX_HISTORY_CONTENT_BYTES:
                raise _Err(
                    f"Current file is too large to back up safely before restore (maximum {MAX_HISTORY_CONTENT_BYTES} bytes)"
                )
            if current != entry["content"] and not (entries and entries[-1]["content"] == current):
                entries.append({
                    "id": str(uuid.uuid4()),
                    "path": _norm_path(f),
                    "timestamp": _now_ms(),
                    "author": "human",
                    "tool": "before.history.restore",
                    "content": current,
                })
                retention = max(_read_settings(root)["maxVersionsPerFile"], 1)
                if len(entries) > retention:
                    entries = entries[len(entries) - retention:]
                store.write_text(_dumps(entries), encoding="utf-8")
                _prune_store(root, store)
            f.write_text(entry["content"], encoding="utf-8")
            return entry["content"]

    return _wrap(run)


@command()
def get_file_history_stats(project_path: str):
    def run():
        root = _checked_project_root(project_path)
        with _HISTORY_LOCK:
            _checked_history_dir(root, False)
            files = _files_oldest_first(root)
            n = 0
            for p, _, _ in files:
                try:
                    ents = _parse_entries(p.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError):
                    continue
                if ents is not None:
                    n += len(ents)
            return {"bytes": sum(f[1] for f in files), "files": len(files), "entries": n}

    return _wrap(run)


@command()
def clear_file_history(project_path: str):
    def run():
        root = _checked_project_root(project_path)
        with _HISTORY_LOCK:
            d = _checked_history_dir(root, False)
            if d.exists():
                shutil.rmtree(d)
        return None

    return _wrap(run)
