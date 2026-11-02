"""Persistent app state (replaces tauri-plugin-store's `app-state.json`).

Location: $LLM_WIKI_DATA_DIR or ~/.llm-wiki-server. Same key names as the
frontend store (`recentProjects`, `llmConfig`, `apiConfig`, ...), so configs
exported from the desktop app can be dropped in as `app-state.json`.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    d = Path(os.environ.get("LLM_WIKI_DATA_DIR") or (Path.home() / ".llm-wiki-server"))
    d.mkdir(parents=True, exist_ok=True)
    return d


class AppState:
    def __init__(self, path: Path | None = None):
        self.path = path or (data_dir() / "app-state.json")
        self._lock = threading.RLock()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._load().get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            data = self._load()
            data[key] = value
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)


_state: AppState | None = None


def app_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state
