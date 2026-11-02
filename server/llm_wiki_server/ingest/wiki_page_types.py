"""Port of src/lib/wiki-page-types.ts."""
from __future__ import annotations

import re
from typing import Optional

from ._js import rx

GENERATION_WIKI_TYPES = (
    "source",
    "entity",
    "concept",
    "comparison",
    "query",
    "synthesis",
    "thesis",
    "methodology",
    "finding",
)

_WIKI_TYPE_DIRS = [
    ("entities", "entity"),
    ("concepts", "concept"),
    ("sources", "source"),
    ("queries", "query"),
    ("comparisons", "comparison"),
    ("synthesis", "synthesis"),
    ("findings", "finding"),
    ("standards", "standard"),
    ("playbooks", "playbook"),
    ("thesis", "thesis"),
    ("methodology", "methodology"),
]

_CUSTOM_DIR_RE = re.compile(r"(?:^|/)wiki/([^/.][^/]*)/[^/]+\.md\Z")
_SPLIT_RE = rx(r"[-_\s]+")


def infer_wiki_type_from_path(path: str, file_name: Optional[str] = None) -> Optional[str]:
    normalized = path.replace("\\", "/").lower()
    for dir_, type_ in _WIKI_TYPE_DIRS:
        if f"/wiki/{dir_}/" in normalized or f"/{dir_}/" in normalized or normalized.startswith(f"wiki/{dir_}/"):
            return type_
    name = (file_name if file_name is not None else normalized.split("/")[-1]).lower()
    if name == "overview.md" or "/overview.md" in normalized:
        return "overview"
    m = _CUSTOM_DIR_RE.search(normalized)
    custom_dir = m.group(1) if m else None
    if custom_dir:
        return custom_dir
    return None


def wiki_type_label(type_: str) -> str:
    if type_ == "thesis":
        return "Thesis"
    if type_ == "methodology":
        return "Methodology"
    if type_ == "finding":
        return "Finding"
    return " ".join(part[:1].upper() + part[1:] for part in _SPLIT_RE.split(type_) if part)
