"""Command registry: Python equivalent of Tauri's `generate_handler!` + `invoke`.

Every ported `#[tauri::command]` registers here under its original snake_case
name and becomes reachable as `POST /rpc/<name>` with a JSON object of
arguments. Argument keys may be camelCase (what the frontend sends through
`invoke`) or snake_case; both are normalised to snake_case.
"""
from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict

from .errors import CommandError

_COMMANDS: Dict[str, Callable[..., Any]] = {}
_CAMEL = re.compile(r"(?<=[a-z0-9])([A-Z])")


def to_snake(key: str) -> str:
    return _CAMEL.sub(lambda m: "_" + m.group(1).lower(), key).lower()


def command(name: str | None = None):
    def deco(fn: Callable[..., Any]):
        _COMMANDS[name or fn.__name__] = fn
        return fn

    return deco


def get_command(name: str) -> Callable[..., Any] | None:
    return _COMMANDS.get(name)


def command_names() -> list[str]:
    return sorted(_COMMANDS)


def bind_args(fn: Callable[..., Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise keys and keep only parameters the function accepts (Tauri ignores extras)."""
    args = {to_snake(k): v for k, v in (raw or {}).items()}
    sig = inspect.signature(fn)
    out: Dict[str, Any] = {}
    for pname, p in sig.parameters.items():
        if pname in args:
            out[pname] = args[pname]
        elif p.default is inspect.Parameter.empty and p.kind in (
            p.POSITIONAL_OR_KEYWORD,
            p.KEYWORD_ONLY,
        ):
            raise CommandError(f"missing required argument `{pname}`")
    return out


_STARTUP: list = []


def on_startup(fn: Callable[[], Any]):
    """Register a zero-arg hook run once when the server starts (e.g. start file watchers)."""
    _STARTUP.append(fn)
    return fn


def startup_hooks() -> list:
    return list(_STARTUP)
