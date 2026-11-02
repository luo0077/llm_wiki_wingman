"""Port of src/lib/concurrency-limits.ts."""
from __future__ import annotations

import math

MIN_USER_CONCURRENCY = 1
MAX_USER_CONCURRENCY = 64


def clamp_user_concurrency(value: float, fallback: float = MIN_USER_CONCURRENCY) -> int:
    try:
        finite = math.isfinite(value)
    except TypeError:
        finite = False
    if finite:
        resolved = math.floor(value)
    else:
        try:
            resolved = math.floor(fallback)
        except (TypeError, ValueError, OverflowError):
            # Math.floor(NaN) -> NaN; Math.max/min(NaN) -> NaN. Closest sane result:
            return MIN_USER_CONCURRENCY
    return max(MIN_USER_CONCURRENCY, min(MAX_USER_CONCURRENCY, resolved))
