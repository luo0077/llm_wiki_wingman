"""In-process event bus: replacement for Tauri's `app.emit(name, payload)`.

`publish(topic, payload)` is thread-safe and may be called from any thread
(file watcher threads, subprocess readers, ...). Consumers are SSE clients
(`GET /events?topics=a,b`, see commands/runtime_config.py) or in-process
listeners registered with `add_listener`.

Topic names are exactly the Rust `emit` event names, e.g.
`file-sync://queue-updated`, `file-sync://changed`, `claude-cli:<stream_id>`,
`codex-cli:<stream_id>`, `claude-cli:<stream_id>:done`.
"""
from __future__ import annotations

import asyncio
import fnmatch
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional

_lock = threading.RLock()
_subs: List["Subscription"] = []
_listeners: List[Callable[[str, Any], None]] = []

_MAX_QUEUE = 10_000


class Subscription:
    """An asyncio queue fed from any thread. Items are (topic, payload) tuples."""

    def __init__(self, topics: Optional[Iterable[str]], loop: asyncio.AbstractEventLoop):
        # None / empty => all topics. Patterns may use fnmatch wildcards (`*`).
        self.topics = [t for t in (topics or []) if t]
        self.loop = loop
        self.queue: "asyncio.Queue[tuple]" = asyncio.Queue(maxsize=_MAX_QUEUE)

    def matches(self, topic: str) -> bool:
        if not self.topics:
            return True
        return any(t == topic or fnmatch.fnmatchcase(topic, t) for t in self.topics)

    def _put(self, item: tuple) -> None:
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            pass  # slow consumer: drop (UI events are idempotent snapshots)

    def deliver(self, topic: str, payload: Any) -> None:
        try:
            self.loop.call_soon_threadsafe(self._put, (topic, payload))
        except RuntimeError:
            unsubscribe(self)  # loop closed


def subscribe(topics: Optional[Iterable[str]] = None,
              loop: Optional[asyncio.AbstractEventLoop] = None) -> Subscription:
    loop = loop or asyncio.get_running_loop()
    sub = Subscription(topics, loop)
    with _lock:
        _subs.append(sub)
    return sub


def unsubscribe(sub: Subscription) -> None:
    with _lock:
        if sub in _subs:
            _subs.remove(sub)


def add_listener(fn: Callable[[str, Any], None]) -> Callable[[], None]:
    """Register a synchronous in-process listener; returns a remover."""
    with _lock:
        _listeners.append(fn)

    def remove() -> None:
        with _lock:
            if fn in _listeners:
                _listeners.remove(fn)

    return remove


def publish(topic: str, payload: Any = None) -> None:
    with _lock:
        subs = [s for s in _subs if s.matches(topic)]
        listeners = list(_listeners)
    for s in subs:
        s.deliver(topic, payload)
    for fn in listeners:
        try:
            fn(topic, payload)
        except Exception:
            pass


def subscriber_count() -> int:
    with _lock:
        return len(_subs)
