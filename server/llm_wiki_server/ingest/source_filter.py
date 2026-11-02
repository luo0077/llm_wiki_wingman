"""Port of src/lib/source-filter.ts."""
from __future__ import annotations

from typing import Any, Dict, List

from .path_utils import normalize_path

_HIDDEN_SOURCE_ENTRY_NAMES = {".cache", ".DS_Store"}
_SENSITIVE_CONFIG_EXTENSIONS = {"env", "json", "toml", "yaml", "yml", "xml"}
_SENSITIVE_CONFIG_DIR_NAMES = {".claude", ".codex", ".cursor", ".gemini", ".mcp"}


def is_hidden_raw_source_entry_name(name: str) -> bool:
    return name in _HIDDEN_SOURCE_ENTRY_NAMES


def filter_raw_source_tree(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`nodes` are FileNode dicts: {name, path, is_dir, children?}."""
    kept = [
        node
        for node in nodes
        if not is_hidden_raw_source_entry_name(node["name"])
        and (node.get("is_dir") or not is_sensitive_config_source_file(node["path"]))
    ]
    mapped = []
    for node in kept:
        if not node.get("is_dir"):
            mapped.append(node)
            continue
        mapped.append({**node, "children": filter_raw_source_tree(node.get("children") or [])})
    return [n for n in mapped if not n.get("is_dir") or (n.get("children") and len(n["children"]) > 0)]


def is_sensitive_config_source_file(path: str) -> bool:
    parts = [p for p in normalize_path(path).split("/") if p]
    name = parts[-1] if parts else ""
    ext = name.split(".")[-1].lower() if "." in name else ""
    return bool(
        ext
        and ext in _SENSITIVE_CONFIG_EXTENSIONS
        and any(part.lower() in _SENSITIVE_CONFIG_DIR_NAMES for part in parts)
    )
