"""Port of commands/file_sync.rs (project file watcher, snapshot + change queue).

Events (via ..events.publish, identical names/payloads to the Rust `emit`):
  * ``file-sync://queue-updated``  {"projectId": str, "tasks": [FileChangeTask]}
  * ``file-sync://changed``        {"projectId": str, "tasks": [FileChangeTask]}

The `notify` crate is replaced by `watchdog` when installed; otherwise a
polling watcher feeds the same debounce/worker pipeline. The low-frequency
snapshot reconciliation of the Rust code runs in both cases.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue as _queue
import sys
import threading
import time
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from .. import events
from ..errors import CommandError
from ..registry import command

SNAPSHOT_FILE = ".llm-wiki/file-snapshot.json"
QUEUE_FILE = ".llm-wiki/file-change-queue.json"
EVENT_QUEUE_UPDATED = "file-sync://queue-updated"
EVENT_CHANGED = "file-sync://changed"
MAX_HASH_BYTES = 32 * 1024 * 1024
MAX_RETRY_COUNT = 3
APP_WRITE_IGNORE_MS = 4_000
QUEUE_EMIT_EVERY = 25
LINUX_WATCHER_RESCAN_INTERVAL_MS = 10_000
OTHER_WATCHER_RESCAN_INTERVAL_MS = 60_000
POLL_INTERVAL_SECS = 2.0

# src/lib/source-watch-defaults.json (embedded so the server is self-contained)
DEFAULT_SOURCE_WATCH_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "autoIngest": True,
    "includeExtensions": [
        "md", "mdx", "txt", "org", "pdf", "doc", "docx", "docm", "ppt", "pps", "pot",
        "pptx", "pptm", "ppsx", "ppsm", "xls", "xlsx", "xlsm", "xlsb", "odt", "odp",
        "ods", "rtf", "html", "htm", "csv", "png", "jpg", "jpeg",
    ],
    "excludeExtensions": [
        "tmp", "temp", "bak", "swp", "part", "partial", "crdownload", "exe", "dll",
        "so", "dylib", "bin", "iso", "dmg",
    ],
    "excludeDirs": [
        ".git", ".svn", ".hg", ".obsidian", ".idea", ".vscode", "node_modules",
        ".cache", "__pycache__",
    ],
    "excludeGlobs": ["~$*", ".~lock.*#", "*.draft.*", "draft-*", "*.private.*"],
    "maxFileSizeMb": 100,
}

_IS_LINUX = sys.platform.startswith("linux")
_IS_WINDOWS = sys.platform.startswith("win")

_queue_locks: Dict[str, threading.RLock] = {}
_queue_locks_guard = threading.Lock()
_app_write_ignores: Dict[str, int] = {}
_ignores_guard = threading.Lock()
_generation = 0
_generation_guard = threading.Lock()
_state_lock = threading.Lock()
_current: Optional["_ProjectWatcher"] = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def _bump_generation() -> int:
    global _generation
    with _generation_guard:
        _generation += 1
        return _generation


def _is_active_generation(gen: int) -> bool:
    return _generation == gen


def normalize_key(path: str) -> str:
    return path.lower() if _IS_WINDOWS else path


def path_key(path: Any) -> str:
    """Canonical key for a path, even when it (partly) does not exist yet."""
    try:
        real = os.path.realpath(str(path))
    except (OSError, ValueError):
        real = str(path)
    return normalize_key(real.replace("\\", "/"))


def queue_lock_for(root: Path) -> threading.RLock:
    key = path_key(root)
    with _queue_locks_guard:
        lock = _queue_locks.get(key)
        if lock is None:
            lock = _queue_locks[key] = threading.RLock()
        return lock


class _QueueLock:
    def __init__(self, root: Path):
        self.lock = queue_lock_for(root)

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, *exc):
        self.lock.release()


def with_queue_lock(root: Path, fn: Callable[[], Any]) -> Any:
    with _QueueLock(root):
        return fn()


def mark_app_write_path(path: Any) -> None:
    """Called by fs.py for every app-originated write so the watcher ignores it (4s)."""
    key = path_key(path)
    now = now_ms()
    with _ignores_guard:
        for k in [k for k, exp in _app_write_ignores.items() if exp <= now]:
            del _app_write_ignores[k]
        _app_write_ignores[key] = now + APP_WRITE_IGNORE_MS


def is_app_write_ignored(path: Any) -> bool:
    key = path_key(path)
    now = now_ms()
    with _ignores_guard:
        for k in [k for k, exp in _app_write_ignores.items() if exp <= now]:
            del _app_write_ignores[k]
        return any(key == ig or key.startswith(ig + "/") for ig in _app_write_ignores)


def normalize_rel_path(path: Any) -> Optional[str]:
    """Port of normalize_rel_path: only Normal / CurDir components allowed."""
    pp = PurePath(str(path))
    if pp.anchor:
        return None
    parts = []
    for p in pp.parts:
        if p == "..":
            return None
        if p == ".":
            continue
        parts.append(p)
    return "/".join(parts)


def strip_prefix(root: Path, path: Path) -> Optional[Path]:
    try:
        return path.relative_to(root)
    except ValueError:
        pass
    try:
        return Path(os.path.realpath(path)).relative_to(Path(os.path.realpath(root)))
    except (ValueError, OSError):
        return None


def stable_path_hash(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()[:12]


def wildcard_match(pattern: str, value: str) -> bool:
    pat = pattern.lower()
    val = value.lower()
    p = v = 0
    star: Optional[int] = None
    match_after_star = 0
    while v < len(val):
        if p < len(pat) and (pat[p] == "?" or pat[p] == val[v]):
            p += 1
            v += 1
        elif p < len(pat) and pat[p] == "*":
            star = p
            match_after_star = v
            p += 1
        elif star is not None:
            p = star + 1
            match_after_star += 1
            v = match_after_star
        else:
            return False
    while p < len(pat) and pat[p] == "*":
        p += 1
    return p == len(pat)


def _normalize_ext_list(values: List[str]) -> List[str]:
    return sorted({v.strip().lstrip(".").lower() for v in values if v.strip().lstrip(".")})


def _normalize_string_list(values: List[str]) -> List[str]:
    return sorted({v.strip() for v in values if v.strip()})


def normalize_source_watch_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_SOURCE_WATCH_CONFIG))
    if isinstance(config, dict):
        for k in ("enabled", "autoIngest", "includeExtensions", "excludeExtensions",
                  "excludeDirs", "excludeGlobs", "maxFileSizeMb"):
            if config.get(k) is not None:
                cfg[k] = config[k]
    cfg["includeExtensions"] = _normalize_ext_list(cfg["includeExtensions"])
    cfg["excludeExtensions"] = _normalize_ext_list(cfg["excludeExtensions"])
    cfg["excludeDirs"] = _normalize_string_list(cfg["excludeDirs"])
    cfg["excludeGlobs"] = _normalize_string_list(cfg["excludeGlobs"])
    try:
        mb = int(cfg["maxFileSizeMb"])
    except (TypeError, ValueError):
        mb = 100
    cfg["maxFileSizeMb"] = min(max(mb, 1), 4096)
    return cfg


class SourceWatchRules:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.include_extensions: Set[str] = set(config["includeExtensions"])
        self.exclude_extensions: Set[str] = set(config["excludeExtensions"])
        dirs = (d.replace("\\", "/").strip("/").lower() for d in config["excludeDirs"])
        self.exclude_dirs: Set[str] = {d for d in dirs if d}
        self.exclude_globs: List[str] = list(config["excludeGlobs"])

    def matches_excluded_dir(self, rel_lower: str) -> bool:
        for d in self.exclude_dirs:
            if "/" in d:
                if rel_lower == d or rel_lower.startswith(d + "/") or ("/" + d + "/") in rel_lower:
                    return True
            elif d in rel_lower.split("/"):
                return True
        return False


def should_watch_rel(rel: str, rules: SourceWatchRules) -> bool:
    if not rel:
        return False
    lower = rel.lower()
    if ("/.llm-wiki/" in lower or lower.startswith(".llm-wiki/")
            or lower.startswith("wiki/media/") or lower.endswith(".ds_store")):
        return False
    name = lower.rsplit("/", 1)[-1]
    if name in ("thumbs.db", "desktop.ini"):
        return False
    if rules.matches_excluded_dir(lower):
        return False
    if any(wildcard_match(p, rel) or wildcard_match(p, name) for p in rules.exclude_globs):
        return False
    if rel.startswith("raw/sources/"):
        ext = name.rsplit(".", 1)[1] if "." in name else ""
        if ext and ext in rules.exclude_extensions:
            return False
        if rules.include_extensions and (not ext or ext not in rules.include_extensions):
            return False
        return True
    return rel == "purpose.md" or rel == "schema.md" or (rel.startswith("wiki/") and rel.endswith(".md"))


def relative_watch_path(root: Path, path: Path, rules: SourceWatchRules,
                        size: Optional[int]) -> Optional[str]:
    rel_p = strip_prefix(root, path)
    if rel_p is None:
        return None
    rel = normalize_rel_path(rel_p)
    if rel is None or not should_watch_rel(rel, rules):
        return None
    if rel.startswith("raw/sources/") and path.exists():
        max_bytes = rules.config["maxFileSizeMb"] * 1024 * 1024
        if size is None:
            try:
                size = path.stat().st_size
            except OSError:
                return None
        if size > max_bytes:
            return None
    return rel


def _walk_files(top: Path, rules: Optional[SourceWatchRules] = None,
                root: Optional[Path] = None) -> Iterator[Tuple[Path, int]]:
    """WalkDir equivalent: regular (non-symlink) files only. Prunes directories
    that `should_watch_rel` would reject anyway (pure optimisation)."""
    for dirpath, dirnames, filenames in os.walk(top, onerror=lambda e: None):
        if rules is not None and root is not None:
            keep = []
            for d in dirnames:
                rp = strip_prefix(root, Path(dirpath) / d)
                rel = normalize_rel_path(rp) if rp is not None else None
                if rel is not None:
                    low = rel.lower()
                    if (low == ".llm-wiki" or low.endswith("/.llm-wiki") or low == "wiki/media"
                            or rules.matches_excluded_dir(low)):
                        continue
                keep.append(d)
            dirnames[:] = keep
        for f in filenames:
            p = Path(dirpath) / f
            try:
                if os.path.islink(p):
                    continue
                st = p.stat()
            except OSError:
                continue
            yield p, st.st_size


# --------------------------------------------------------------------------
# JSON persistence
# --------------------------------------------------------------------------

def _read_json(path: Path, default: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    if not path.exists():
        return default()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise CommandError(f"Failed to read '{path}': {e}")
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data
    except ValueError as e:
        raise CommandError(f"Failed to parse '{path}': {e}")


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise CommandError(f"Failed to create '{path.parent}': {e}")
    text = json.dumps(value, indent=2, ensure_ascii=False)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
    except OSError as e:
        raise CommandError(f"Failed to write '{tmp}': {e}")
    try:
        os.replace(tmp, path)
    except OSError as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise CommandError(f"Failed to move '{tmp}' to '{path}': {e}")


def ensure_sync_dir(root: Path) -> None:
    try:
        (root / ".llm-wiki").mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise CommandError(f"Failed to create .llm-wiki: {e}")


def read_snapshot(root: Path) -> Dict[str, Any]:
    s = _read_json(root / SNAPSHOT_FILE, lambda: {"version": 0, "updatedAt": 0, "files": {}})
    if not s.get("version"):
        s["version"] = 1
    s.setdefault("updatedAt", 0)
    if not isinstance(s.get("files"), dict):
        s["files"] = {}
    return s


def write_snapshot(root: Path, snapshot: Dict[str, Any]) -> None:
    out = {
        "version": snapshot.get("version", 1),
        "updatedAt": snapshot.get("updatedAt", 0),
        "files": {k: snapshot["files"][k] for k in sorted(snapshot["files"])},
    }
    _write_json(root / SNAPSHOT_FILE, out)


def read_queue(root: Path) -> Dict[str, Any]:
    q = _read_json(root / QUEUE_FILE, lambda: {"version": 0, "tasks": []})
    if not q.get("version"):
        q["version"] = 1
    if not isinstance(q.get("tasks"), list):
        q["tasks"] = []
    return q


def write_queue(root: Path, queue: Dict[str, Any]) -> None:
    _write_json(root / QUEUE_FILE, {"version": queue.get("version", 1), "tasks": queue["tasks"]})


# --------------------------------------------------------------------------
# file meta
# --------------------------------------------------------------------------

def _meta_of(root: Path, rel: str, with_hash: bool) -> Optional[Dict[str, Any]]:
    path = root / rel
    if not path.exists():
        return None
    try:
        st = path.stat()
    except OSError as e:
        raise CommandError(f"metadata failed for {rel}: {e}")
    import stat as _stat
    if not _stat.S_ISREG(st.st_mode):
        return None
    size = st.st_size
    mtime_ms = st.st_mtime_ns // 1_000_000
    h = None
    if with_hash and size <= MAX_HASH_BYTES:
        h = md5_file(path)
    return {"hash": h, "size": size, "mtimeMs": mtime_ms}


def read_meta(root: Path, rel: str) -> Optional[Dict[str, Any]]:
    return _meta_of(root, rel, True)


def read_meta_fast(root: Path, rel: str) -> Optional[Dict[str, Any]]:
    return _meta_of(root, rel, False)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                buf = f.read(64 * 1024)
                if not buf:
                    break
                h.update(buf)
    except OSError as e:
        raise CommandError(f"open failed for '{path}': {e}")
    return h.hexdigest()


# --------------------------------------------------------------------------
# queue logic
# --------------------------------------------------------------------------

def merge_kind(existing: str, incoming: str) -> str:
    if (existing, incoming) in (("deleted", "created"), ("created", "deleted")) or incoming == "modified":
        return "modified"
    return incoming


def upsert_task(queue: Dict[str, Any], project_id: str, rel: str, kind: str,
                old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]], now: int) -> None:
    for t in queue["tasks"]:
        if (t["projectId"] == project_id and normalize_key(t["path"]) == normalize_key(rel)
                and t["status"] in ("pending", "processing", "failed")):
            t["kind"] = merge_kind(t["kind"], kind)
            t["hashAfter"] = new.get("hash") if new else None
            src = new or old
            t["size"] = src["size"] if src else None
            t["mtimeMs"] = new["mtimeMs"] if new else None
            t["updatedAt"] = now
            if t["status"] == "failed":
                if t["retryCount"] < MAX_RETRY_COUNT:
                    t["status"] = "pending"
                    t["error"] = None
                else:
                    t["error"] = f"Retry limit reached ({MAX_RETRY_COUNT})"
            elif t["status"] == "processing":
                t["needsRerun"] = True
                t["error"] = None
            else:
                t["error"] = None
            return
    src = new or old
    queue["tasks"].append({
        "id": f"change_{now}_{stable_path_hash(rel)}",
        "projectId": project_id,
        "path": rel,
        "kind": kind,
        "status": "pending",
        "hashBefore": old.get("hash") if old else None,
        "hashAfter": new.get("hash") if new else None,
        "size": src["size"] if src else None,
        "mtimeMs": new["mtimeMs"] if new else None,
        "createdAt": now,
        "updatedAt": now,
        "retryCount": 0,
        "error": None,
        "needsRerun": False,
    })


def enqueue_paths(root: Path, project_id: str, rels: Set[str]) -> None:
    snapshot = with_queue_lock(root, lambda: read_snapshot(root))
    now = now_ms()
    changes = []
    for rel in sorted(rels):
        old = snapshot["files"].get(rel)
        new = read_meta(root, rel)
        old_key = (old.get("hash"), old.get("size")) if old else None
        new_key = (new.get("hash"), new.get("size")) if new else None
        if old_key == new_key:
            continue
        if old is None and new is not None:
            kind = "created"
        elif old is not None and new is None:
            kind = "deleted"
        elif old is not None and new is not None:
            kind = "modified"
        else:
            continue
        changes.append((rel, kind, old, new))
    if not changes:
        return

    def apply():
        queue = read_queue(root)
        for rel, kind, old, new in changes:
            upsert_task(queue, project_id, rel, kind, old, new, now)
        write_queue(root, queue)

    with_queue_lock(root, apply)


def sync_snapshot_paths(root: Path, rels: Set[str]) -> None:
    metas = [(rel, read_meta(root, rel)) for rel in sorted(rels)]

    def apply():
        snapshot = read_snapshot(root)
        for rel, meta in metas:
            if meta is not None:
                snapshot["files"][rel] = meta
            else:
                snapshot["files"].pop(rel, None)
        snapshot["version"] = 1
        snapshot["updatedAt"] = now_ms()
        write_snapshot(root, snapshot)

    with_queue_lock(root, apply)


def enqueue_rescan_changes(root: Path, project_id: str, cfg: Dict[str, Any]) -> None:
    rules = SourceWatchRules(cfg)
    rels: Set[str] = set()
    for p, size in _walk_files(root, rules, root):
        rel = relative_watch_path(root, p, rules, size)
        if rel:
            rels.add(rel)
    snapshot = with_queue_lock(root, lambda: read_snapshot(root))
    for rel in snapshot["files"]:
        if not (root / rel).exists():
            rels.add(rel)
    enqueue_paths(root, project_id, rels)


STARTUP_PREFIXES = ["raw/sources", "wiki", "purpose.md", "schema.md"]


def enqueue_startup_rescan_changes(root: Path, project_id: str, cfg: Dict[str, Any]) -> None:
    enqueue_rescan_changes_for_prefixes(root, project_id, STARTUP_PREFIXES, cfg)


def enqueue_rescan_changes_for_prefixes(root: Path, project_id: str, prefixes: List[str],
                                        cfg: Dict[str, Any]) -> None:
    rules = SourceWatchRules(cfg)
    rels: Set[str] = set()
    app_written: Set[str] = set()
    snapshot = with_queue_lock(root, lambda: read_snapshot(root))

    def consider(path: Path, size: Optional[int]) -> None:
        rel = relative_watch_path(root, path, rules, size)
        if not rel:
            return
        if is_app_write_ignored(path):
            app_written.add(rel)
            return
        old = snapshot["files"].get(rel)
        fast = read_meta_fast(root, rel)
        if (old and (old.get("size"), old.get("mtimeMs"))) != (fast and (fast["size"], fast["mtimeMs"])):
            rels.add(rel)

    for prefix in prefixes:
        path = root / prefix
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            consider(path, size)
        elif path.exists():
            for p, size in _walk_files(path, rules, root):
                consider(p, size)

    for rel in snapshot["files"]:
        if any(rel == pf or rel.startswith(pf + "/") for pf in prefixes) and not (root / rel).exists():
            rels.add(rel)

    if app_written:
        sync_snapshot_paths(root, app_written)
    enqueue_paths(root, project_id, rels)


def write_task_meta_to_snapshot(root: Path, task: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> None:
    snapshot = read_snapshot(root)
    if meta is not None:
        snapshot["files"][task["path"]] = meta
    else:
        snapshot["files"].pop(task["path"], None)
    snapshot["version"] = 1
    snapshot["updatedAt"] = now_ms()
    write_snapshot(root, snapshot)


def reset_processing_tasks(root: Path, project_id: str) -> None:
    queue = read_queue(root)
    changed = False
    queue["tasks"] = [t for t in queue["tasks"] if t["projectId"] == project_id]
    for t in queue["tasks"]:
        if t["status"] == "processing":
            t["status"] = "pending"
            t["needsRerun"] = False
            t["error"] = None
            t["updatedAt"] = now_ms()
            changed = True
    if changed:
        write_queue(root, queue)


def process_queue_inner(root: Path, project_id: str,
                        on_queue: Callable[[Dict[str, Any]], None],
                        on_changed: Callable[[List[Dict[str, Any]]], None]) -> List[Dict[str, Any]]:
    changed_tasks: List[Dict[str, Any]] = []
    all_changed: List[Dict[str, Any]] = []
    processed_since_emit = 0
    emitted_processing = False
    while True:
        def pick():
            queue = read_queue(root)
            for t in queue["tasks"]:
                if t["projectId"] == project_id and t["status"] == "pending":
                    t["status"] = "processing"
                    t["updatedAt"] = now_ms()
                    task = dict(t)
                    write_queue(root, queue)
                    return task, queue
            return None

        try:
            picked = with_queue_lock(root, pick)
        except CommandError:
            on_changed(changed_tasks)
            raise
        if picked is None:
            try:
                queue = with_queue_lock(root, lambda: read_queue(root))
            except CommandError:
                on_changed(changed_tasks)
                raise
            on_changed(changed_tasks)
            on_queue(queue)
            return all_changed
        task, queue = picked
        if not emitted_processing:
            emitted_processing = True
            on_queue(queue)

        try:
            meta_result: Tuple[bool, Any] = (True, read_meta(root, task["path"]))
        except CommandError as e:
            meta_result = (False, e.message)
        emit_after_update = False

        def update():
            nonlocal processed_since_emit, emit_after_update
            q = read_queue(root)
            current = next((t for t in q["tasks"] if t["id"] == task["id"]), None)
            if current is not None:
                if current["status"] != "processing" or current["updatedAt"] != task["updatedAt"]:
                    if current["status"] == "processing" and current["needsRerun"]:
                        current["status"] = "pending"
                        current["needsRerun"] = False
                        current["updatedAt"] = now_ms()
                else:
                    ok, val = meta_result
                    if ok:
                        write_task_meta_to_snapshot(root, task, val)
                        if current["needsRerun"]:
                            current["status"] = "pending"
                            current["needsRerun"] = False
                        else:
                            current["status"] = "done"
                        current["error"] = None
                    else:
                        current["status"] = "failed"
                        current["error"] = val
                        current["retryCount"] += 1
                    current["updatedAt"] = now_ms()
                    all_changed.append(dict(task))
                    changed_tasks.append(dict(task))
                    processed_since_emit += 1
                    if processed_since_emit >= QUEUE_EMIT_EVERY:
                        processed_since_emit = 0
                        emit_after_update = True
            q["tasks"] = [t for t in q["tasks"] if t["status"] != "done"]
            write_queue(root, q)
            return read_queue(root)

        try:
            queue = with_queue_lock(root, update)
        except CommandError:
            on_changed(changed_tasks)
            raise
        if emit_after_update:
            on_queue(queue)


def emit_queue(project_id: str, queue: Dict[str, Any]) -> None:
    events.publish(EVENT_QUEUE_UPDATED, {"projectId": project_id, "tasks": list(queue["tasks"])})


def emit_changed_batch(project_id: str, tasks: List[Dict[str, Any]]) -> None:
    if not tasks:
        return
    events.publish(EVENT_CHANGED, {"projectId": project_id, "tasks": tasks})


def process_queue(root: Path, project_id: str) -> List[Dict[str, Any]]:
    return process_queue_inner(
        root, project_id,
        lambda q: emit_queue(project_id, q),
        lambda tasks: emit_changed_batch(project_id, tasks),
    )


# --------------------------------------------------------------------------
# watcher
# --------------------------------------------------------------------------

def collect_known_paths(root: Path, path: Path, snapshot: Dict[str, Any], rels: Set[str],
                        rules: SourceWatchRules) -> None:
    if path.is_dir():
        for p, size in _walk_files(path):
            rel = relative_watch_path(root, p, rules, size)
            if rel:
                rels.add(rel)
        return
    rp = strip_prefix(root, path)
    if rp is None:
        return
    rel = normalize_rel_path(rp)
    if rel is None:
        return
    if not path.exists():
        for known in snapshot["files"]:
            if known == rel or known.startswith(rel + "/"):
                rels.add(known)
        return
    if should_watch_rel(rel, rules):
        rels.add(rel)


def handle_changed_paths(root: Path, project_id: str, cfg: Dict[str, Any], gen: int,
                         paths: List[Path]) -> None:
    if not _is_active_generation(gen):
        return
    rules = SourceWatchRules(cfg)
    rels: Set[str] = set()
    app_written: Set[str] = set()
    snapshot = with_queue_lock(root, lambda: read_snapshot(root))
    for path in paths:
        if is_app_write_ignored(path):
            collect_known_paths(root, path, snapshot, app_written, rules)
            continue
        if path.is_dir():
            for p, size in _walk_files(path):
                if not is_app_write_ignored(p):
                    rel = relative_watch_path(root, p, rules, size)
                    if rel:
                        rels.add(rel)
        else:
            rel = relative_watch_path(root, path, rules, None)
            if rel:
                rels.add(rel)
            elif not path.exists():
                collect_known_paths(root, path, snapshot, rels, rules)
    if app_written:
        sync_snapshot_paths(root, app_written)
    if not rels:
        return
    if not _is_active_generation(gen):
        return
    enqueue_paths(root, project_id, rels)
    if not _is_active_generation(gen):
        return
    process_queue(root, project_id)
    queue = with_queue_lock(root, lambda: read_queue(root))
    if not _is_active_generation(gen):
        return
    emit_queue(project_id, queue)


def rescan_watch_roots(root: Path, project_id: str, cfg: Dict[str, Any], gen: int) -> None:
    if not _is_active_generation(gen):
        return
    prefixes = ["raw/sources", "wiki", "purpose.md", "schema.md"] if _IS_LINUX else ["raw/sources"]
    enqueue_rescan_changes_for_prefixes(root, project_id, prefixes, cfg)
    if not _is_active_generation(gen):
        return
    process_queue(root, project_id)
    queue = with_queue_lock(root, lambda: read_queue(root))
    if not _is_active_generation(gen):
        return
    emit_queue(project_id, queue)


class _ProjectWatcher:
    def __init__(self, root: Path, project_id: str, cfg: Dict[str, Any], gen: int):
        self.root = root
        self.project_id = project_id
        self.cfg = cfg
        self.gen = gen
        self.q: "_queue.Queue[Path]" = _queue.Queue(maxsize=8192)
        self.stop_evt = threading.Event()
        self.observer = None
        self.poll_thread: Optional[threading.Thread] = None
        self.worker = threading.Thread(target=self._run, name="file-sync-worker", daemon=True)

    # -- producer side --
    def send(self, path: Path) -> None:
        try:
            self.q.put_nowait(path)
        except _queue.Full:
            try:
                self.q.put_nowait(self.root)
            except _queue.Full:
                pass

    def start(self) -> None:
        self.worker.start()
        try:
            self._start_watchdog()
        except Exception as e:  # watchdog missing or backend failure -> polling
            if not isinstance(e, ImportError):
                print(f"[file-sync] watchdog unavailable ({e}); falling back to polling", file=sys.stderr)
            self._start_polling()

    def _start_watchdog(self) -> None:
        from watchdog.events import FileSystemEventHandler  # type: ignore
        from watchdog.observers import Observer  # type: ignore

        outer = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                et = getattr(event, "event_type", "")
                if et not in ("created", "modified", "deleted", "moved", "closed"):
                    return
                outer.send(Path(os.fsdecode(event.src_path)))
                dest = getattr(event, "dest_path", "")
                if dest:
                    outer.send(Path(os.fsdecode(dest)))

        obs = Observer()
        obs.schedule(Handler(), str(self.root), recursive=True)
        obs.daemon = True
        obs.start()
        self.observer = obs

    def _start_polling(self) -> None:
        self.poll_thread = threading.Thread(target=self._poll, name="file-sync-poll", daemon=True)
        self.poll_thread.start()

    def _scan(self) -> Dict[str, Tuple[int, int]]:
        out: Dict[str, Tuple[int, int]] = {}
        rules = SourceWatchRules(self.cfg)
        for prefix in STARTUP_PREFIXES:
            p = self.root / prefix
            if p.is_file():
                try:
                    st = p.stat()
                    out[str(p)] = (st.st_size, st.st_mtime_ns)
                except OSError:
                    pass
            elif p.is_dir():
                for fp, _ in _walk_files(p, rules, self.root):
                    try:
                        st = fp.stat()
                        out[str(fp)] = (st.st_size, st.st_mtime_ns)
                    except OSError:
                        pass
        return out

    def _poll(self) -> None:
        prev = self._scan()
        while not self.stop_evt.wait(POLL_INTERVAL_SECS):
            try:
                cur = self._scan()
            except Exception:
                continue
            for k in set(prev) | set(cur):
                if prev.get(k) != cur.get(k):
                    self.send(Path(k))
            prev = cur

    # -- consumer side --
    def _periodic(self, last: List[int]) -> None:
        interval = LINUX_WATCHER_RESCAN_INTERVAL_MS if _IS_LINUX else OTHER_WATCHER_RESCAN_INTERVAL_MS
        if now_ms() - last[0] < interval:
            return
        try:
            rescan_watch_roots(self.root, self.project_id, self.cfg, self.gen)
        except Exception as e:
            print(f"[file-sync] periodic rescan failed: {e}", file=sys.stderr)
        last[0] = now_ms()

    def _run(self) -> None:
        pending: Set[Path] = set()
        last = [now_ms()]
        while not self.stop_evt.is_set():
            try:
                pending.add(self.q.get(timeout=0.7))
                while True:
                    try:
                        pending.add(self.q.get_nowait())
                    except _queue.Empty:
                        break
            except _queue.Empty:
                if not pending:
                    self._periodic(last)
                    continue
                paths = sorted(pending)
                pending.clear()
                try:
                    handle_changed_paths(self.root, self.project_id, self.cfg, self.gen, paths)
                except Exception as e:
                    print(f"[file-sync] change handling failed: {e}", file=sys.stderr)
                self._periodic(last)

    def stop(self) -> None:
        self.stop_evt.set()
        if self.observer is not None:
            try:
                self.observer.stop()
            except Exception:
                pass
            self.observer = None


def _stop_current() -> None:
    global _current
    with _state_lock:
        w, _current = _current, None
    if w is not None:
        w.stop()


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def _result(root: Path, project_id: str, changed: List[Dict[str, Any]]) -> Dict[str, Any]:
    queue = with_queue_lock(root, lambda: read_queue(root))
    emit_queue(project_id, queue)
    return {"queue": queue, "changedTasks": changed}


@command()
def start_project_file_watcher(project_id: str, project_path: str,
                               source_watch_config: Optional[dict] = None):
    global _current
    root = Path(project_path)
    cfg = normalize_source_watch_config(source_watch_config)
    gen = _bump_generation()
    _stop_current()
    ensure_sync_dir(root)
    with_queue_lock(root, lambda: reset_processing_tasks(root, project_id))
    enqueue_startup_rescan_changes(root, project_id, cfg)
    changed = process_queue(root, project_id)

    watcher = _ProjectWatcher(root, project_id, cfg, gen)
    if not root.is_dir():
        raise CommandError(f"Failed to watch '{root}': No such directory")
    watcher.start()
    with _state_lock:
        _current = watcher
    return _result(root, project_id, changed)


@command()
def stop_project_file_watcher():
    _bump_generation()
    _stop_current()
    return None


@command()
def rescan_project_files(project_id: str, project_path: str,
                         source_watch_config: Optional[dict] = None,
                         watch_roots_only: Optional[bool] = None):
    root = Path(project_path)
    cfg = normalize_source_watch_config(source_watch_config)
    ensure_sync_dir(root)
    if watch_roots_only:
        enqueue_startup_rescan_changes(root, project_id, cfg)
    else:
        enqueue_rescan_changes(root, project_id, cfg)
    changed = process_queue(root, project_id)
    return _result(root, project_id, changed)


@command()
def invalidate_project_file_snapshot_paths(project_path: str, paths: List[str]):
    root = Path(project_path)
    rels = []
    for p in paths:
        rel = normalize_rel_path(p)
        if rel is None:
            raise CommandError(f"invalid project-relative snapshot path: {p}")
        rels.append(rel)

    def apply():
        snapshot = read_snapshot(root)
        for rel in rels:
            snapshot["files"].pop(rel, None)
        snapshot["updatedAt"] = now_ms()
        write_snapshot(root, snapshot)

    with_queue_lock(root, apply)
    return None


@command()
def get_file_change_queue(project_path: str):
    root = Path(project_path)
    return with_queue_lock(root, lambda: read_queue(root))


@command()
def retry_file_change_task(project_id: str, project_path: str, task_id: str):
    root = Path(project_path)

    def apply():
        queue = read_queue(root)
        now = now_ms()
        for t in queue["tasks"]:
            if t["id"] == task_id and t["projectId"] == project_id:
                t["status"] = "pending"
                t["error"] = None
                t["retryCount"] = 0
                t["needsRerun"] = False
                t["updatedAt"] = now
        write_queue(root, queue)

    with_queue_lock(root, apply)
    process_queue(root, project_id)
    queue = with_queue_lock(root, lambda: read_queue(root))
    emit_queue(project_id, queue)
    return queue


@command()
def ignore_file_change_task(project_id: str, project_path: str, task_id: str):
    root = Path(project_path)

    def apply():
        queue = read_queue(root)
        queue["tasks"] = [t for t in queue["tasks"]
                          if not (t["id"] == task_id and t["projectId"] == project_id)]
        write_queue(root, queue)
        return read_queue(root)

    queue = with_queue_lock(root, apply)
    emit_queue(project_id, queue)
    return queue
