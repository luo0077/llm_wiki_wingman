"""Port of agent/session.rs. On-disk: <project>/.llm-wiki/agent-sessions/<id>.json (pretty JSON, camelCase)."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_SESSION_MESSAGES = 40
MAX_CACHED_SESSIONS = 128


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class AgentSessionMessage:
    role: str
    content: str
    timestamp: int

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: Any) -> "AgentSessionMessage":
        if not isinstance(d, dict) or not isinstance(d.get("role"), str) or not isinstance(d.get("content"), str):
            raise ValueError("invalid session message")
        ts = d.get("timestamp")
        if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
            raise ValueError("invalid session message timestamp")
        return cls(d["role"], d["content"], ts)


@dataclass
class AgentSession:
    session_id: str = ""
    project_id: str = ""
    messages: List[AgentSessionMessage] = field(default_factory=list)
    updated_at: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "projectId": self.project_id,
            "messages": [m.to_dict() for m in self.messages],
            "updatedAt": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "AgentSession":
        if not isinstance(d, dict) or not isinstance(d.get("messages"), list):
            raise ValueError("invalid session")
        ua = d.get("updatedAt")
        if isinstance(ua, bool) or not isinstance(ua, int) or ua < 0:
            raise ValueError("invalid session updatedAt")
        sid = d.get("sessionId", "")
        pid = d.get("projectId", "")
        if not isinstance(sid, str) or not isinstance(pid, str):
            raise ValueError("invalid session ids")
        return cls(sid, pid, [AgentSessionMessage.from_dict(m) for m in d["messages"]], ua)


def _normalize_project_path(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def session_cache_key(project_path: str, session_id: str) -> str:
    return f"{_normalize_project_path(project_path)}::{session_id}"


def sanitize_session_id(session_id: str) -> Optional[str]:
    t = session_id.strip()
    if not t or "/" in t or "\\" in t or ".." in t or len(t.encode("utf-8")) > 128:
        return None
    return "".join(
        ch if (ch.isascii() and ch.isalnum()) or ch in "-_." else "_" for ch in t
    )


def session_file(project_path: str, session_id: str) -> Optional[Path]:
    sid = sanitize_session_id(session_id)
    if sid is None:
        return None
    return Path(project_path) / ".llm-wiki" / "agent-sessions" / f"{sid}.json"


def load_session(project_path: str, session_id: str) -> Optional[AgentSession]:
    path = session_file(project_path, session_id)
    if path is None:
        return None
    try:
        return AgentSession.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def save_session(project_path: str, session: AgentSession) -> None:
    """Raises RuntimeError with the Rust error text on failure."""
    path = session_file(project_path, session.session_id)
    if path is None:
        raise RuntimeError("Invalid Agent session id")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create session dir: {e}")
    try:
        path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"Failed to write session: {e}")


class AgentSessionStore:
    def __init__(self) -> None:
        self._inner: Dict[str, AgentSession] = {}
        self._lock = threading.Lock()

    def _trim(self) -> None:
        if len(self._inner) <= MAX_CACHED_SESSIONS:
            return
        entries = sorted(((s.updated_at, k) for k, s in self._inner.items()))
        for _, key in entries[: len(self._inner) - MAX_CACHED_SESSIONS]:
            self._inner.pop(key, None)

    def append_turn(self, project_path: str, project_id: str, session_id: str, user: str, assistant: str) -> None:
        now = _now_ms()
        with self._lock:
            key = session_cache_key(project_path, session_id)
            session = self._inner.get(key)
            if session is None:
                session = load_session(project_path, session_id) or AgentSession()
                self._inner[key] = session
            session.session_id = session_id
            session.project_id = project_id
            session.messages.append(AgentSessionMessage("user", user, now))
            session.messages.append(AgentSessionMessage("assistant", assistant, now))
            if len(session.messages) > MAX_SESSION_MESSAGES:
                del session.messages[: len(session.messages) - MAX_SESSION_MESSAGES]
            session.updated_at = now
            try:
                save_session(project_path, session)
            except RuntimeError:
                pass
            self._trim()

    def recent_messages(self, project_path: str, session_id: str, limit: int) -> List[AgentSessionMessage]:
        with self._lock:
            key = session_cache_key(project_path, session_id)
            session = self._inner.get(key)
            if session is None:
                loaded = load_session(project_path, session_id)
                if loaded is not None:
                    self._inner[key] = loaded
                    self._trim()
                    session = self._inner.get(key)
            msgs = list(session.messages) if session is not None else []
        if limit <= 0:
            return []
        return msgs[max(0, len(msgs) - limit):]

    def list_sessions(self, project_path: str) -> List[AgentSession]:
        d = Path(project_path) / ".llm-wiki" / "agent-sessions"
        out: List[AgentSession] = []
        try:
            entries = list(os.scandir(d))
        except OSError:
            return []
        for e in entries:
            if not e.name.endswith(".json") or e.name == ".json":
                continue
            try:
                out.append(AgentSession.from_dict(json.loads(Path(e.path).read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        out.sort(key=lambda s: (s.updated_at, s.session_id), reverse=True)
        return out


# Process-wide default store shared by the REST chat layer and the /rpc agent
# commands, so both paths see the same cached sessions.
DEFAULT_STORE = AgentSessionStore()


def append_turn(project_path: str, project_id: str, session_id: str, user: str, assistant: str) -> None:
    DEFAULT_STORE.append_turn(project_path, project_id, session_id, user, assistant)


def recent_messages(project_path: str, session_id: str, limit: int) -> List[AgentSessionMessage]:
    return DEFAULT_STORE.recent_messages(project_path, session_id, limit)


def list_sessions(project_path: str) -> List[AgentSession]:
    return DEFAULT_STORE.list_sessions(project_path)
