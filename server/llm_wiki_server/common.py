"""Helpers shared by command modules (ports of small Rust utilities)."""
from __future__ import annotations

import ntpath
import os
from pathlib import Path

from .errors import CommandError


def is_absolute_path_cross_platform(path: str) -> bool:
    """Port of fs.rs `is_absolute_path_cross_platform`: accepts POSIX, `C:\\`, `C:/` and UNC paths."""
    if not path:
        return False
    return os.path.isabs(path) or ntpath.isabs(path) or path.startswith(("\\\\", "//"))


def require_absolute_path(operation: str, path: str) -> None:
    if not is_absolute_path_cross_platform(path):
        raise CommandError(f"{operation} requires an absolute path; got relative path '{path}'")


def fwd(path: str | os.PathLike) -> str:
    """Forward slashes for cross-platform consistency in the TS layer."""
    return str(path).replace("\\", "/")


def validate_wiki_project_root(root: str | os.PathLike) -> Path:
    r = Path(root)
    if not r.exists():
        raise CommandError(f"Path does not exist: '{r}'")
    if not r.is_dir():
        raise CommandError(f"Path is not a directory: '{r}'")
    if not (r / "schema.md").exists():
        raise CommandError(f"Not a valid wiki project (missing schema.md): '{r}'")
    if not (r / "wiki").is_dir():
        raise CommandError(f"Not a valid wiki project (missing wiki/ directory): '{r}'")
    return r
