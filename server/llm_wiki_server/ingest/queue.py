"""Port of src/lib/ingest-queue.ts: persistent, per-project ingest queue.

Differences from the desktop version (which tracks a single *active* project):

* The server hosts many projects at once, so state is kept per project
  (:class:`ProjectQueue`); there is no active-project handshake. Queues load
  lazily from disk on first use and every project that has restorable work is
  restored by :func:`restore_all_queues` (registered as a startup hook in
  ``commands/ingest.py``).
* The on-disk format is byte-compatible with the desktop app:
  ``<project>/.llm-wiki/ingest-queue.json`` is a JSON array of
  ``{id, projectId, sourcePath, folderContext, status, addedAt, error,
  retryCount}`` (``done`` tasks are never persisted).
* All ingest work runs on one dedicated asyncio event loop in a daemon thread
  (``_Runtime``), so the queue can be driven from sync RPC handlers, FastAPI
  routes and startup hooks alike.
* Failed tasks are retried with a small exponential backoff
  (``RETRY_BACKOFF_BASE_SECONDS``); the desktop app retries immediately.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import string
import threading
import time
import uuid
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, Callable, Dict, List, Optional, Sequence

from .. import events

QUEUE_UPDATED_TOPIC = "ingest://queue-updated"
TASK_DONE_TOPIC = "ingest://task-done"

MAX_RETRIES = 3
USAGE_LIMIT_AUTO_RESUME_SECONDS = 15 * 60
RETRY_BACKOFF_BASE_SECONDS = 2.0
RETRY_BACKOFF_MAX_SECONDS = 60.0
MIN_USER_CONCURRENCY = 1
MAX_USER_CONCURRENCY = 64

STOPPED = ("failed", "cancelled")


def clamp_user_concurrency(value: Any, fallback: int = MIN_USER_CONCURRENCY) -> int:
    try:
        resolved = int(float(value))
    except (TypeError, ValueError, OverflowError):
        resolved = int(fallback)
    return max(MIN_USER_CONCURRENCY, min(MAX_USER_CONCURRENCY, resolved))


def normalize_path(p: str) -> str:
    return str(p).replace("\\", "/").rstrip("/") if len(str(p)) > 1 else str(p)


def is_absolute_path(p: str) -> bool:
    return bool(p) and (p.startswith("/") or bool(re.match(r"^[A-Za-z]:[\\/]", p)) or p.startswith("\\\\"))


def _generate_id() -> str:
    rnd = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return "ingest-%d-%s" % (int(time.time() * 1000), rnd)


def is_usage_limit_error(message: str) -> bool:
    return bool(re.search(r"\b429\b|rate[_\s-]*limit|usage\s+limit|quota|too many requests", message, re.I))


def is_structural_wiki_path(file_path: str) -> bool:
    n = normalize_path(file_path)
    return n.endswith(("/wiki/index.md", "/wiki/log.md", "/wiki/overview.md")) or n in (
        "wiki/index.md", "wiki/log.md", "wiki/overview.md")


# --------------------------------------------------------------------------
# Runtime: one background asyncio loop for all ingest work
# --------------------------------------------------------------------------

class _Runtime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None

    def ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self.loop is not None and self.loop.is_running() and self.thread and self.thread.is_alive():
                return self.loop
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def run() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                loop.run_forever()

            t = threading.Thread(target=run, name="llm-wiki-ingest", daemon=True)
            t.start()
            ready.wait(5)
            self.loop, self.thread = loop, t
            return loop

    def submit(self, coro) -> "ConcurrentFuture":
        return asyncio.run_coroutine_threadsafe(coro, self.ensure())

    def call_later(self, delay: float, fn: Callable[[], None]) -> "threading.Timer":
        t = threading.Timer(max(0.0, delay), fn)
        t.daemon = True
        t.start()
        return t


_runtime = _Runtime()


def runtime() -> _Runtime:
    return _runtime


class CommitCoordinator:
    """Reserves commit turns in task-start order (port of IngestCommitCoordinator).

    Must be used from the ingest event loop only.
    """

    def __init__(self) -> None:
        self._tail: Optional[asyncio.Future] = None

    def reserve(self):
        loop = asyncio.get_running_loop()
        predecessor = self._tail
        turn: asyncio.Future = loop.create_future()
        self._tail = turn
        released = {"v": False}

        def release() -> None:
            if released["v"]:
                return
            released["v"] = True
            if not turn.done():
                turn.set_result(None)

        async def run_commit(operation):
            if predecessor is not None:
                try:
                    await asyncio.shield(predecessor)
                except BaseException:  # noqa: BLE001 - a failed predecessor never blocks us
                    pass
            try:
                return await operation()
            finally:
                release()

        return run_commit, release


class ActiveRun:
    def __init__(self, task_id: str, epoch: int, signal) -> None:
        self.task_id = task_id
        self.epoch = epoch
        self.signal = signal
        self.written_files: List[str] = []
        self.release_commit_turn: Callable[[], None] = lambda: None
        self.commit_active = False
        self.future: Optional[ConcurrentFuture] = None
        self.async_task: Optional[asyncio.Task] = None

    def abort(self) -> None:
        try:
            self.signal.abort()
        except Exception:
            pass
        task, fut = self.async_task, self.future
        if task is not None and fut is not None and not fut.done() and not self.commit_active:
            try:
                task.get_loop().call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass


def _pipeline():
    from . import pipeline  # late: pipeline imports sibling modules

    return pipeline


# --------------------------------------------------------------------------
# Per-project queue
# --------------------------------------------------------------------------

def queue_file_path(project_path: str) -> str:
    return "%s/.llm-wiki/ingest-queue.json" % normalize_path(project_path)


class ProjectQueue:
    def __init__(self, project_id: str, project_path: str) -> None:
        self.project_id = project_id
        self.project_path = normalize_path(project_path)
        self.lock = threading.RLock()
        self.tasks: List[Dict[str, Any]] = []
        self.active: Dict[str, ActiveRun] = {}
        self.cancelled_in_flight: set = set()
        self.not_before: Dict[str, float] = {}
        self.worker_limit = 1
        self.paused = False
        self.blocked_on_llm_config = False
        self.epoch = 0
        self.completed_since_idle = 0
        self.processed_since_drain = False
        self.coordinator: Optional[CommitCoordinator] = None
        self._usage_timer: Optional[threading.Timer] = None
        self._retry_timer: Optional[threading.Timer] = None
        self._write_lock = threading.Lock()
        self.loaded = False

    # ---- persistence -----------------------------------------------------
    def save(self) -> None:
        with self.lock:
            snapshot = json.dumps([t for t in self.tasks if t["status"] != "done"], indent=2, ensure_ascii=False)
        self._write(snapshot)

    def _write(self, snapshot: str) -> None:
        path = queue_file_path(self.project_path)
        with self._write_lock:  # serialises writers; last snapshot written wins
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = "%s.%s.tmp" % (path, uuid.uuid4().hex[:8])
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(snapshot)
                os.replace(tmp, path)
            except OSError:
                pass  # best-effort; in-memory queue stays authoritative

    def load_from_disk(self) -> List[Dict[str, Any]]:
        try:
            with open(queue_file_path(self.project_path), "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        out = []
        for t in raw:
            if not isinstance(t, dict) or not isinstance(t.get("id"), str):
                continue
            task = dict(t)
            task["projectId"] = t.get("projectId") or self.project_id
            task.setdefault("sourcePath", "")
            task.setdefault("folderContext", "")
            task.setdefault("status", "pending")
            task.setdefault("addedAt", int(time.time() * 1000))
            task.setdefault("error", None)
            task.setdefault("retryCount", 0)
            out.append(task)
        return out

    # ---- helpers ---------------------------------------------------------
    def norm_source(self, source_path: str) -> str:
        n = normalize_path(source_path)
        if self.project_path and n.startswith(self.project_path + "/"):
            return n[len(self.project_path) + 1:]
        return n

    def same_source(self, a: str, b: str) -> bool:
        return self.norm_source(a) == self.norm_source(b)

    def find(self, task_id: str) -> Optional[Dict[str, Any]]:
        for t in self.tasks:
            if t["id"] == task_id:
                return t
        return None

    def summary(self) -> Dict[str, Any]:
        with self.lock:
            cnt = lambda s: sum(1 for t in self.tasks if t["status"] == s)  # noqa: E731
            return {
                "pending": cnt("pending"), "processing": cnt("processing"),
                "failed": cnt("failed"), "cancelled": cnt("cancelled"),
                "completed": self.completed_since_idle,
                "total": len(self.tasks) + self.completed_since_idle,
                "paused": self.paused, "userPaused": self.paused,
                "blockedOnLlmConfig": self.blocked_on_llm_config,
            }

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {"projectId": self.project_id, "summary": self.summary(),
                    "tasks": [dict(t) for t in self.tasks]}

    def emit(self) -> None:
        try:
            events.publish(QUEUE_UPDATED_TOPIC, self.snapshot())
        except Exception:
            pass

    def _has_pending_or_processing(self) -> bool:
        return any(t["status"] in ("pending", "processing") for t in self.tasks)

    def _clear_usage_timer(self) -> None:
        if self._usage_timer is not None:
            self._usage_timer.cancel()
            self._usage_timer = None

    # ---- queue operations --------------------------------------------------
    def upsert(self, source_path: str, folder_context: str) -> str:
        if not self.tasks and not self.active:
            self.completed_since_idle = 0
        norm = self.norm_source(source_path)
        for t in self.tasks:
            if t["status"] in ("pending", "failed", "cancelled") and self.same_source(t["sourcePath"], norm):
                t["sourcePath"] = norm
                t["folderContext"] = folder_context or t.get("folderContext", "")
                t["status"] = "pending"
                t["error"] = None
                t["retryCount"] = 0
                self.not_before.pop(t["id"], None)
                return t["id"]
        processing = next((t for t in self.tasks if t["status"] == "processing" and self.same_source(t["sourcePath"], norm)), None)
        if processing is not None:
            rerun = next((t for t in self.tasks if t["status"] == "pending" and self.same_source(t["sourcePath"], norm)), None)
            if rerun is not None:
                return rerun["id"]
        task = {"id": _generate_id(), "projectId": self.project_id, "sourcePath": norm,
                "folderContext": folder_context, "status": "pending",
                "addedAt": int(time.time() * 1000), "error": None, "retryCount": 0}
        self.tasks.append(task)
        return task["id"]

    def enqueue_batch(self, files: Sequence[Dict[str, str]]) -> List[str]:
        with self.lock:
            ids = [self.upsert(f["sourcePath"], f.get("folderContext", "") or "") for f in files]
        self.save()
        self.emit()
        self.process_next()
        return ids

    def retry(self, task_ids: Sequence[str]) -> int:
        selected = set(task_ids)
        n = 0
        with self.lock:
            for t in self.tasks:
                if t["id"] in selected and t["status"] in STOPPED:
                    t["status"], t["error"], t["retryCount"] = "pending", None, 0
                    self.not_before.pop(t["id"], None)
                    n += 1
        if n:
            self.save()
            self.emit()
            self.process_next()
        return n

    def retry_all_stopped(self, failed_only: bool = False) -> int:
        with self.lock:
            ids = [t["id"] for t in self.tasks if (t["status"] == "failed" if failed_only else t["status"] in STOPPED)]
        return self.retry(ids)

    def move_pending(self, task_id: str, direction: str) -> bool:
        with self.lock:
            pending = [t for t in self.tasks if t["status"] == "pending"]
            pos = next((i for i, t in enumerate(pending) if t["id"] == task_id), -1)
            swap = pos - 1 if direction == "up" else pos + 1
            if pos < 0 or swap < 0 or swap >= len(pending):
                return False
            a = self.tasks.index(pending[pos])
            b = self.tasks.index(pending[swap])
            self.tasks[a], self.tasks[b] = self.tasks[b], self.tasks[a]
        self.save()
        self.emit()
        return True

    def cancel(self, task_ids: Sequence[str]) -> int:
        selected = set(task_ids)
        with self.lock:
            targets = [t for t in self.tasks if t["id"] in selected and t["status"] in ("pending", "processing")]
            if not targets:
                return 0
            had_processing = any(t["status"] == "processing" for t in targets)
            for t in targets:
                if t["status"] == "processing":
                    self.cancelled_in_flight.add(t["id"])
            for t in targets:
                run = self.active.get(t["id"])
                if run:
                    run.abort()
            for t in targets:
                t["status"], t["error"] = "cancelled", None
            if not self._has_pending_or_processing():
                self.paused = False
                self._clear_usage_timer()
        self.save()
        self.emit()
        if not had_processing:
            self.process_next()
        return len(targets)

    def cancel_all(self) -> int:
        with self.lock:
            had_processing = any(t["status"] == "processing" for t in self.tasks)
            for t in self.tasks:
                if t["status"] == "processing":
                    self.cancelled_in_flight.add(t["id"])
            for run in list(self.active.values()):
                run.abort()
            n = 0
            for t in self.tasks:
                if t["status"] in ("pending", "processing"):
                    t["status"], t["error"] = "cancelled", None
                    n += 1
            self.paused = False
            self._clear_usage_timer()
        self.save()
        self.emit()
        if not had_processing:
            self.process_next()
        return n

    def discard_for_sources(self, source_paths: Sequence[str]) -> int:
        def matches(task: Dict[str, Any]) -> bool:
            queued = self.norm_source(task["sourcePath"])
            for sp in source_paths:
                s = self.norm_source(sp)
                if s == queued:
                    return True
                if is_absolute_path(s) and not is_absolute_path(queued) and s.endswith("/" + queued):
                    return True
                if is_absolute_path(queued) and not is_absolute_path(s) and queued.endswith("/" + s):
                    return True
            return False

        with self.lock:
            targets = [t for t in self.tasks if matches(t)]
            if not targets:
                return 0
            ids = {t["id"] for t in targets}
            had_processing = any(t["status"] == "processing" for t in targets)
            for t in targets:
                if t["status"] == "processing":
                    self.cancelled_in_flight.add(t["id"])
                run = self.active.get(t["id"])
                if run:
                    run.abort()
            self.tasks = [t for t in self.tasks if t["id"] not in ids]
            if not self._has_pending_or_processing():
                self.paused = False
                self._clear_usage_timer()
        self.save()
        self.emit()
        if not had_processing:
            self.process_next()
        return len(targets)

    def clear_completed(self) -> None:
        with self.lock:
            self.tasks = [t for t in self.tasks if t["status"] in ("pending", "processing")]
        self.save()
        self.emit()

    def pause(self) -> None:
        with self.lock:
            self._clear_usage_timer()
            self.paused = True
            processing = [t for t in self.tasks if t["status"] == "processing"]
            for t in processing:
                t["status"] = "pending"
                run = self.active.get(t["id"])
                if run:
                    run.abort()
        if processing:
            self.save()
        self.emit()

    def resume(self) -> None:
        with self.lock:
            self._clear_usage_timer()
            self.paused = False
            self.blocked_on_llm_config = False
        self.emit()
        self.process_next()

    def set_worker_limit(self, limit: Any) -> None:
        with self.lock:
            self.worker_limit = clamp_user_concurrency(limit)
        if not self.paused:
            self.process_next()

    # ---- restore -----------------------------------------------------------
    def restore(self) -> None:
        """Load from disk, reset interrupted tasks to pending and resume work."""
        with self.lock:
            self.epoch += 1
            self.tasks = []
            self.cancelled_in_flight.clear()
            self.active.clear()
            self.paused = False
            self.blocked_on_llm_config = False
            self.completed_since_idle = 0
            self._clear_usage_timer()
            saved = self.load_from_disk()
            self.loaded = True
            mine = [t for t in saved if t.get("projectId") == self.project_id]
            restored = 0
            for t in mine:
                if t["status"] == "processing":
                    t["status"] = "pending"
                    restored += 1
            self.tasks = mine
            pending = sum(1 for t in self.tasks if t["status"] == "pending")
        if not mine:
            return
        self.save()
        if pending > 0 or restored > 0:
            if not _pipeline().task_llm_is_usable(self.project_id):
                with self.lock:
                    self.blocked_on_llm_config = True
                    for t in self.tasks:
                        if t["status"] == "pending" and not t.get("error"):
                            t["error"] = "LLM not configured — set an ingest model in Settings, then resume"
                self.save()
                self.emit()
                return
            self.emit()
            self.process_next()

    # ---- scheduling --------------------------------------------------------
    def process_next(self) -> None:
        pl = None
        with self.lock:
            if self.paused:
                return
            if self.blocked_on_llm_config:
                pl = _pipeline()
                if not pl.task_llm_is_usable(self.project_id):
                    return
                self.blocked_on_llm_config = False
            now = time.monotonic()
            while len(self.active) < self.worker_limit and not self.paused and not self.blocked_on_llm_config:
                nxt = next((t for t in self.tasks if t["status"] == "pending"
                            and t["id"] not in self.active
                            and self.not_before.get(t["id"], 0) <= now), None)
                if nxt is None:
                    break
                if not self._start_task(nxt):
                    break
            self._schedule_backoff_wakeup()
            drained = (not self.active and not any(t["status"] == "pending" for t in self.tasks))
            need_drain = drained and self.processed_since_drain
            epoch = self.epoch
        if need_drain:
            self.processed_since_drain = False
            runtime().submit(self._on_drained(epoch))

    def _schedule_backoff_wakeup(self) -> None:
        waits = [self.not_before[t["id"]] - time.monotonic() for t in self.tasks
                 if t["status"] == "pending" and t["id"] in self.not_before and t["id"] not in self.active]
        waits = [w for w in waits if w > 0]
        if not waits or self.paused:
            return
        if self._retry_timer is not None:
            self._retry_timer.cancel()
        self._retry_timer = runtime().call_later(min(waits), self.process_next)

    def _start_task(self, task: Dict[str, Any]) -> bool:
        """Caller holds ``self.lock``."""
        pl = _pipeline()
        project_path = self.project_path
        if not project_path or not os.path.isdir(project_path):
            task["status"] = "failed"
            task["error"] = "Project not found in registry (was it deleted?)"
            self.save()
            return True
        if not pl.task_llm_is_usable(self.project_id):
            self.blocked_on_llm_config = True
            task["status"] = "pending"
            if not task.get("error"):
                task["error"] = "LLM not configured — set an ingest model in Settings, then resume"
            self.save()
            return False
        task["error"] = None
        run = ActiveRun(task["id"], self.epoch, pl.AbortSignal())
        task["status"] = "processing"
        self.active[task["id"]] = run
        self.save()
        run.future = runtime().submit(self._run_task(task, run))
        self.emit()
        return True

    async def _run_task(self, task: Dict[str, Any], run: ActiveRun) -> None:
        pl = _pipeline()
        run.async_task = asyncio.current_task()
        if self.coordinator is None:
            self.coordinator = CommitCoordinator()
        run_commit, release = self.coordinator.reserve()
        run.release_commit_turn = release

        async def ordered_commit(operation):
            async def wrapped():
                run.commit_active = True
                try:
                    return await operation()
                finally:
                    run.commit_active = False
            return await run_commit(wrapped)

        project_path = self.project_path
        source = task["sourcePath"]
        full_source = normalize_path(source) if is_absolute_path(source) else "%s/%s" % (project_path, source)

        def track(rel: str) -> None:
            if rel not in run.written_files:
                run.written_files.append(rel)

        err: Optional[BaseException] = None
        written: List[str] = []
        try:
            written = await pl.auto_ingest(
                project_path, full_source, pl.get_task_llm_config(self.project_id), run.signal,
                task.get("folderContext", ""), track, {"runCommit": ordered_commit, "projectId": self.project_id,
                                                       "taskId": task["id"]})
        except asyncio.CancelledError as e:
            err = e
        except BaseException as e:  # noqa: BLE001
            err = e
        try:
            self._finish_run(task, run, written, err)
        finally:
            release()
            with self.lock:
                if self.active.get(task["id"]) is run:
                    del self.active[task["id"]]
                stale = run.epoch != self.epoch
                paused = self.paused
            if not stale and not paused:
                self.process_next()

    def _finish_run(self, task: Dict[str, Any], run: ActiveRun, written: List[str], err: Optional[BaseException]) -> None:
        pl = _pipeline()
        with self.lock:
            if run.epoch != self.epoch:
                return
            current = self.find(task["id"])
            cancelled = task["id"] in self.cancelled_in_flight or (current is not None and current["status"] == "cancelled")
            if err is None:
                run.written_files = written
                if cancelled:
                    # Written pages may be merges into pre-existing user content: keep them.
                    self.cancelled_in_flight.discard(task["id"])
                    result = "cancelled"
                elif not written:
                    err = RuntimeError("Ingest produced no output files")
                    result = "error"
                else:
                    self.cancelled_in_flight.discard(task["id"])
                    self.tasks = [t for t in self.tasks if t["id"] != task["id"]]
                    self.completed_since_idle += 1
                    self.processed_since_drain = True
                    result = "done"
            else:
                result = "error"
            if result == "error":
                self.cancelled_in_flight.discard(task["id"])
                if cancelled:
                    result = "cancelled"
                elif current is not None and current["status"] == "pending":
                    result = "reverted"  # paused / usage-limit sibling: already reset to pending
                elif isinstance(err, asyncio.CancelledError):
                    task["status"] = "pending"
                    result = "reverted"
                else:
                    message = str(err) or err.__class__.__name__
                    if isinstance(err, pl.NonRetryableIngestError) or getattr(err, "nonRetryable", False) is True:
                        task["status"], task["error"] = "failed", message
                        result = "failed"
                    elif is_usage_limit_error(message):
                        task["status"] = "pending"
                        task["error"] = "Paused after provider usage limit: " + message
                        self.paused = True
                        for tid, other in list(self.active.items()):
                            if tid == task["id"]:
                                continue
                            sib = self.find(tid)
                            if sib is None or sib["status"] != "processing":
                                continue
                            sib["status"] = "pending"
                            sib["error"] = "Paused after provider usage limit: " + message
                            other.abort()
                        self._schedule_usage_resume()
                        result = "paused"
                    else:
                        task["retryCount"] = int(task.get("retryCount", 0)) + 1
                        task["error"] = message
                        if task["retryCount"] >= MAX_RETRIES:
                            task["status"] = "failed"
                            result = "failed"
                        else:
                            task["status"] = "pending"
                            delay = min(RETRY_BACKOFF_MAX_SECONDS,
                                        RETRY_BACKOFF_BASE_SECONDS * (2 ** (task["retryCount"] - 1)))
                            if delay > 0:
                                self.not_before[task["id"]] = time.monotonic() + delay
                            result = "retry"
            snapshot_status = task["status"]
        self.save()
        self.emit()
        try:
            events.publish(TASK_DONE_TOPIC, {"projectId": self.project_id, "taskId": task["id"],
                                             "sourcePath": task["sourcePath"], "result": result,
                                             "status": snapshot_status, "error": task.get("error"),
                                             "filesWritten": list(written) if result == "done" else []})
        except Exception:
            pass

    def _schedule_usage_resume(self) -> None:
        if self._usage_timer is not None:
            return

        def resume() -> None:
            self._usage_timer = None
            with self.lock:
                self.paused = False
            self.process_next()

        self._usage_timer = runtime().call_later(USAGE_LIMIT_AUTO_RESUME_SECONDS, resume)

    async def _on_drained(self, epoch: int) -> None:
        if self.epoch != epoch:
            return
        try:
            from . import review_store

            await review_store.sweep_resolved_reviews(self.project_path, self.project_id)
        except Exception:
            pass

    def shutdown(self) -> None:
        """Abort everything and persist a restartable snapshot (tests / server stop)."""
        with self.lock:
            self.epoch += 1
            self._clear_usage_timer()
            if self._retry_timer is not None:
                self._retry_timer.cancel()
            for run in list(self.active.values()):
                run.abort()
            self.active.clear()
            for t in self.tasks:
                if t["status"] == "processing":
                    t["status"] = "pending"
        self.save()


# --------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------

class IngestQueueManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._queues: Dict[str, ProjectQueue] = {}

    # ---- project lookup -----------------------------------------------------
    @staticmethod
    def resolve_project(project: str) -> Dict[str, str]:
        """Resolve an id / path / "current" to ``{"id","path"}`` using the shared project registry."""
        from ..commands import rest_api
        from ..errors import CommandError

        try:
            p = rest_api.resolve_project(project)
        except rest_api.ApiError as e:
            if os.path.isdir(project) and os.path.isabs(project):
                pid = rest_api.read_project_id(project) or normalize_path(project)
                return {"id": pid, "path": normalize_path(project)}
            raise CommandError(e.message, e.status)
        return {"id": p["id"], "path": normalize_path(p["path"])}

    def get(self, project: str, *, restore: bool = True) -> ProjectQueue:
        info = self.resolve_project(project)
        return self.get_for(info["id"], info["path"], restore=restore)

    def get_for(self, project_id: str, project_path: str, *, restore: bool = True) -> ProjectQueue:
        with self._lock:
            pq = self._queues.get(project_id)
            if pq is not None and pq.project_path != normalize_path(project_path):
                pq.shutdown()  # project moved: rebuild against the new location
                pq = None
            if pq is None:
                pq = ProjectQueue(project_id, project_path)
                pq.worker_limit = _configured_worker_limit(project_id)
                self._queues[project_id] = pq
                fresh = True
            else:
                fresh = False
        if fresh and restore:
            pq.restore()
        return pq

    def all_queues(self) -> List[ProjectQueue]:
        with self._lock:
            return list(self._queues.values())

    def restore_all(self) -> int:
        """Startup hook: resume interrupted work for every known project."""
        from ..commands import rest_api

        n = 0
        try:
            projects = rest_api.load_projects()
        except Exception:
            projects = []
        for p in projects:
            path = p.get("path") or ""
            if not path or not os.path.isfile(queue_file_path(path)):
                continue
            try:
                self.get_for(p["id"], path, restore=True)
                n += 1
            except Exception:
                continue
        return n

    def reset(self) -> None:
        """Drop all in-memory state (tests)."""
        with self._lock:
            qs = list(self._queues.values())
            self._queues.clear()
        for q in qs:
            q.shutdown()

    # ---- waiting (tests / sync callers) -------------------------------------
    @staticmethod
    def wait_idle(pq: ProjectQueue, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with pq.lock:
                busy = bool(pq.active) or any(
                    t["status"] == "pending" and not pq.paused and not pq.blocked_on_llm_config for t in pq.tasks)
            if not busy:
                return True
            time.sleep(0.02)
        return False


def _configured_worker_limit(project_id: str) -> int:
    try:
        from ..state import app_state

        cfgs = app_state().get("sourceWatchConfig")
        if isinstance(cfgs, dict):
            cfg = cfgs.get(project_id) or cfgs.get("default")
            if isinstance(cfg, dict) and "ingestConcurrency" in cfg:
                return clamp_user_concurrency(cfg["ingestConcurrency"])
    except Exception:
        pass
    return 1


_manager = IngestQueueManager()


def manager() -> IngestQueueManager:
    return _manager


# --------------------------------------------------------------------------
# Module-level API (mirrors the TS exports, keyed by project)
# --------------------------------------------------------------------------

def enqueue_ingest(project: str, source_path: str, folder_context: str = "") -> str:
    return manager().get(project).enqueue_batch([{"sourcePath": source_path, "folderContext": folder_context}])[0]


def enqueue_batch(project: str, files: Sequence[Dict[str, str]]) -> List[str]:
    return manager().get(project).enqueue_batch(list(files))


def retry_task(project: str, task_id: str) -> int:
    return manager().get(project).retry([task_id])


def retry_tasks(project: str, task_ids: Sequence[str]) -> int:
    return manager().get(project).retry(task_ids)


def retry_all_failed_tasks(project: str) -> int:
    return manager().get(project).retry_all_stopped(failed_only=True)


def retry_all_stopped_tasks(project: str) -> int:
    return manager().get(project).retry_all_stopped()


def cancel_task(project: str, task_id: str) -> int:
    return manager().get(project).cancel([task_id])


def cancel_tasks(project: str, task_ids: Sequence[str]) -> int:
    return manager().get(project).cancel(task_ids)


def cancel_all_tasks(project: str) -> int:
    return manager().get(project).cancel_all()


def clear_completed_tasks(project: str) -> None:
    manager().get(project).clear_completed()


def discard_tasks_for_sources(project: str, source_paths: Sequence[str]) -> int:
    return manager().get(project).discard_for_sources(source_paths)


def move_pending_task(project: str, task_id: str, direction: str) -> bool:
    return manager().get(project).move_pending(task_id, direction)


def pause_processing(project: str) -> None:
    manager().get(project).pause()


def resume_processing(project: str) -> None:
    manager().get(project).resume()


def is_queue_paused(project: str) -> bool:
    return manager().get(project).paused


def get_queue(project: str) -> List[Dict[str, Any]]:
    pq = manager().get(project)
    with pq.lock:
        return [dict(t) for t in pq.tasks]


def get_queue_summary(project: str) -> Dict[str, Any]:
    return manager().get(project).summary()


def set_ingest_worker_limit(project: str, limit: Any) -> None:
    manager().get(project).set_worker_limit(limit)


def restore_queue(project_id: str, project_path: str) -> None:
    manager().get_for(project_id, project_path, restore=False).restore()


def restore_all_queues() -> int:
    return manager().restore_all()


def clear_queue_state() -> None:
    manager().reset()


async def cleanup_written_files(project_path: str, file_paths: Sequence[str]) -> None:
    """Explicitly delete wiki files (and their vectors). Cancellation never calls this."""
    from ..commands import fs

    pp = normalize_path(project_path)
    for fp in file_paths:
        if is_structural_wiki_path(fp):
            continue
        full = normalize_path(fp) if is_absolute_path(fp) else "%s/%s" % (pp, fp)
        try:
            await asyncio.to_thread(fs.delete_file, full)
        except Exception:
            pass
