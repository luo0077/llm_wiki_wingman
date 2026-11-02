"""Port of src/lib/context-budget.ts (character budgets for context assembly)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ._js import Record

DEFAULT_MAX_CTX = 204_800
RESPONSE_RESERVE_FRAC = 0.15
INDEX_BUDGET_FRAC = 0.05
PAGE_BUDGET_FRAC = 0.5
PER_PAGE_FRAC = 0.3
PER_PAGE_FLOOR = 5_000


@dataclass
class ContextBudget(Record):
    maxCtx: int
    responseReserve: int
    indexBudget: int
    pageBudget: int
    maxPageSize: int


def compute_context_budget(max_context_size: Optional[float] = None) -> ContextBudget:
    valid = (
        isinstance(max_context_size, (int, float))
        and not isinstance(max_context_size, bool)
        and not math.isnan(max_context_size)
        and max_context_size > 0
    )
    max_ctx = max_context_size if valid else DEFAULT_MAX_CTX
    if isinstance(max_ctx, float) and max_ctx.is_integer():
        max_ctx = int(max_ctx)

    response_reserve = math.floor(max_ctx * RESPONSE_RESERVE_FRAC)
    index_budget = math.floor(max_ctx * INDEX_BUDGET_FRAC)
    page_budget = math.floor(max_ctx * PAGE_BUDGET_FRAC)
    max_page_size = min(page_budget, max(PER_PAGE_FLOOR, math.floor(page_budget * PER_PAGE_FRAC)))
    return ContextBudget(
        maxCtx=max_ctx,
        responseReserve=response_reserve,
        indexBudget=index_budget,
        pageBudget=page_budget,
        maxPageSize=max_page_size,
    )
