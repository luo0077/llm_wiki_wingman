"""Port of src/lib/project-mutex.ts: per-project mutual exclusion.

Multiple ingest / Save-to-Wiki / deep-research entry points mutate shared
project files (``wiki/index.md``, ``wiki/log.md``, page bodies, ingest cache).
Those read-modify-write sections must not overlap.

Two entry points share ONE underlying per-project lock, so coroutines on the
event loop and worker threads exclude each other:

* ``with_project_lock(project_path, fn)``        -- async (``fn`` is an ``async def`` callable,
                                                  or a plain callable returning a value/awaitable)
* ``with_project_lock_sync(project_path, fn)``   -- blocking, for worker threads
* ``project_lock_sync(project_path)``            -- ``with`` context manager (threads)
* ``async with project_lock(project_path):``     -- async context manager

Same as the TS version: no timeouts, no re-entrancy (re-acquiring from inside
the critical section deadlocks), strict FIFO among coroutines of one loop.
Blocking the event loop thread with the sync variants while a coroutine holds
the lock will deadlock -- use the sync variants only from worker threads.
"""
from __future__ import annotations

import asyncio
import inspect
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Awaitable, Callable, Dict, Tuple, TypeVar, Union

T = TypeVar("T")

_meta = threading.Lock()
_thread_locks: Dict[str, "list"] = {}  # path -> [threading.Lock, refcount]
_async_locks: Dict[Tuple[int, str], "list"] = {}  # (loop id, path) -> [asyncio.Lock, refcount]


def _acquire_thread_lock_ref(project_path: str) -> threading.Lock:
    with _meta:
        entry = _thread_locks.get(project_path)
        if entry is None:
            entry = [threading.Lock(), 0]
            _thread_locks[project_path] = entry
        entry[1] += 1
        return entry[0]


def _release_thread_lock_ref(project_path: str) -> None:
    with _meta:
        entry = _thread_locks.get(project_path)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            del _thread_locks[project_path]


def _acquire_async_lock_ref(key: Tuple[int, str]) -> asyncio.Lock:
    with _meta:
        entry = _async_locks.get(key)
        if entry is None:
            entry = [asyncio.Lock(), 0]
            _async_locks[key] = entry
        entry[1] += 1
        return entry[0]


def _release_async_lock_ref(key: Tuple[int, str]) -> None:
    with _meta:
        entry = _async_locks.get(key)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            del _async_locks[key]


@asynccontextmanager
async def project_lock(project_path: str):
    """Async context manager holding the per-project lock."""
    loop = asyncio.get_running_loop()
    key = (id(loop), project_path)
    alock = _acquire_async_lock_ref(key)
    tlock = _acquire_thread_lock_ref(project_path)
    try:
        async with alock:  # FIFO among coroutines of this loop
            if not tlock.acquire(blocking=False):
                fut = loop.run_in_executor(None, tlock.acquire)
                try:
                    await asyncio.shield(fut)
                except asyncio.CancelledError:
                    # The executor thread may still acquire later; release it then.
                    def _late_release(f: "asyncio.Future[Any]") -> None:
                        if not f.cancelled() and f.exception() is None:
                            tlock.release()

                    fut.add_done_callback(_late_release)
                    raise
            try:
                yield
            finally:
                tlock.release()
    finally:
        _release_async_lock_ref(key)
        _release_thread_lock_ref(project_path)


async def with_project_lock(project_path: str, fn: Callable[[], Union[T, Awaitable[T]]]) -> T:
    """Run ``fn`` holding the per-project lock; returns its result, propagates its exception."""
    async with project_lock(project_path):
        result = fn()
        if inspect.isawaitable(result):
            result = await result
        return result  # type: ignore[return-value]


@contextmanager
def project_lock_sync(project_path: str):
    """Blocking context manager (worker threads only)."""
    tlock = _acquire_thread_lock_ref(project_path)
    try:
        tlock.acquire()
        try:
            yield
        finally:
            tlock.release()
    finally:
        _release_thread_lock_ref(project_path)


def with_project_lock_sync(project_path: str, fn: Callable[[], T]) -> T:
    with project_lock_sync(project_path):
        return fn()


def reset_project_locks_for_testing() -> None:
    """Test-only: drop all lock bookkeeping."""
    with _meta:
        _thread_locks.clear()
        _async_locks.clear()
