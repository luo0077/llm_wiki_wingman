"""Port of agent/cancel.rs. Thread-safe cancellation registry for agent runs."""
from __future__ import annotations

import asyncio
import threading
from typing import Dict, Optional

CANCELLED_MESSAGE = "Agent turn cancelled"


def normalize_key(value: str) -> str:
    return value.replace("\\", "_").replace("/", "_")


def cancel_key(project_id: str, session_id: str, run_id: str) -> str:
    return f"{normalize_key(project_id)}::{normalize_key(session_id)}::{normalize_key(run_id)}"


class AgentCancellationToken:
    """Holds the shared flag. `event` is a threading.Event (usable as the
    `cancel_event` that rest_api passes around). Removing from the registry on
    `close()`/context exit/GC mirrors Rust's Drop."""

    def __init__(self, event: threading.Event, key: str, registry: "AgentCancellationRegistry"):
        self.event = event
        self.key = key
        self._registry = registry

    def is_cancelled(self) -> bool:
        return self.event.is_set()

    def check(self) -> None:
        """Rust `check() -> Result<(), String>`: raises Exception("Agent turn cancelled")."""
        if self.is_cancelled():
            raise AgentCancelled(CANCELLED_MESSAGE)

    async def cancelled(self) -> None:
        while not self.is_cancelled():
            await asyncio.sleep(0.05)

    def close(self) -> None:
        self._registry._remove_token(self.key, self.event)

    def __enter__(self) -> "AgentCancellationToken":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):  # safety net like Drop
        try:
            self.close()
        except Exception:
            pass


class AgentCancelled(Exception):
    pass


class AgentCancellationRegistry:
    def __init__(self) -> None:
        self._tokens: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start(self, project_id: str, session_id: str, run_id: str) -> AgentCancellationToken:
        event = threading.Event()
        key = cancel_key(project_id, session_id, run_id)
        with self._lock:
            self._tokens[key] = event
        return AgentCancellationToken(event, key, self)

    def cancel(self, project_id: str, session_id: str, run_id: Optional[str] = None) -> bool:
        prefix = f"{normalize_key(project_id)}::{normalize_key(session_id)}::"
        with self._lock:
            if run_id is not None:
                token = self._tokens.get(cancel_key(project_id, session_id, run_id))
            else:
                token = next((t for k, t in self._tokens.items() if k.startswith(prefix)), None)
        if token is None:
            return False
        token.set()
        return True

    def finish(self, project_id: str, session_id: str, run_id: str) -> None:
        with self._lock:
            self._tokens.pop(cancel_key(project_id, session_id, run_id), None)

    def _remove_token(self, key: str, event: threading.Event) -> None:
        # Only remove if the key still maps to *this* token (a restarted run with the same key keeps its own).
        with self._lock:
            if self._tokens.get(key) is event:
                del self._tokens[key]
