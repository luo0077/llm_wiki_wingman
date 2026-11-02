"""Port of src/lib/wiki-filename.ts."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ._js import js_trim, rx
from .source_identity import _keep_letters_numbers_hyphen

_WS_RUN_RE = rx(r"\s+")
_HYPHEN_RUN_RE = re.compile(r"-+")
_EDGE_HYPHEN_RE = re.compile(r"\A-|-\Z")


def make_query_slug(title: str) -> str:
    slug = js_trim(unicodedata.normalize("NFKC", title))
    slug = _WS_RUN_RE.sub("-", slug)
    slug = _keep_letters_numbers_hyphen(slug)
    slug = _HYPHEN_RUN_RE.sub("-", slug)
    slug = _EDGE_HYPHEN_RE.sub("", slug)
    slug = slug.lower()
    truncated = slug[:50]  # code points, like Array.from(slug).slice(0, 50)
    return truncated if len(truncated) > 0 else "query"


def make_query_file_name(title: str, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Returns {"slug","fileName","date","time"}; `now` defaults to the current UTC time."""
    slug = make_query_slug(title)
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is not None:
        now = now.astimezone(timezone.utc)
    # (naive datetimes are taken as already UTC)
    date = now.strftime("%Y-%m-%d")
    time_ = now.strftime("%H%M%S")
    return {"slug": slug, "date": date, "time": time_, "fileName": f"{slug}-{date}-{time_}.md"}
