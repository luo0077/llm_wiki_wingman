"""Port of src/lib/research-query-grounding.ts."""
from __future__ import annotations

import re
from typing import List, Sequence

from ._js import BOL_M, js_trim, rx

PROJECT_LOCAL_TERM_QUERY_RULES = (
    "Treat ambiguous names, acronyms, character names, module names, and ordinary-looking labels as project-local vocabulary unless the supplied context clearly identifies a public meaning.",
    "Never emit a bare ambiguous term as a web query. Preserve its exact spelling and add domain, project, entity, or mechanism keywords supported by the supplied context.",
    "Do not substitute a popular public meaning for a project-local meaning.",
    "If an internal term has no likely public documentation, search for the underlying real-world concepts or comparisons needed to fill the gap.",
)

TERM_MAX_LENGTH = 120
CONTEXT_MAX_LENGTH = 180
QUERY_MAX_LENGTH = 280

_TRAILING_WORD_RE = rx(r"\s+\S*\Z")
_WS_RUN_RE = rx(r"\s+")
_QUOTE_EDGE_RE = re.compile(r"\A[\"']|[\"']\Z")
_WIKI_EDGE_RE = re.compile(r"\A\[\[|\]\]\Z")
_FM_RE = rx(r"\A---\r?\n[\s\S]*?\r?\n---(?:\r?\n|\Z)")
_MD_CHARS_RE = re.compile(r"[`*_>\[\](){}]")
_HEADING_RE = rx(BOL_M + r"#{1,6}\s+", re.M)
_URL_RE = rx(r"https?://\S+", re.I)


def _truncate_at_word(value: str, max_length: int) -> str:
    if len(value) <= max_length:  # code points == Array.from(value).length
        return value
    truncated = value[:max_length]
    word_boundary = js_trim(_TRAILING_WORD_RE.sub("", truncated, count=1))
    return word_boundary if len(word_boundary) >= max_length * 0.6 else js_trim(truncated)


def _compact_term(value: str) -> str:
    v = _QUOTE_EDGE_RE.sub("", value)
    v = _WIKI_EDGE_RE.sub("", v)
    v = _WS_RUN_RE.sub(" ", v)
    return _truncate_at_word(js_trim(v), TERM_MAX_LENGTH)


def _compact_context(value: str) -> str:
    v = _FM_RE.sub("", value, count=1)
    v = _MD_CHARS_RE.sub(" ", v)
    v = _HEADING_RE.sub("", v)
    v = _URL_RE.sub(" ", v)
    v = _WS_RUN_RE.sub(" ", v)
    return _truncate_at_word(js_trim(v), CONTEXT_MAX_LENGTH)


def grounded_fallback_research_query(term: str, context: str) -> str:
    term_part = _compact_term(term)
    context_part = _compact_context(context)
    if not context_part:
        return term_part
    if not term_part:
        return context_part
    return _truncate_at_word(f"{term_part} {context_part}", QUERY_MAX_LENGTH)


def ground_bare_research_queries(
    queries: Sequence[str], term: str, context: str, bare_aliases: Sequence[str] = ()
) -> List[str]:
    bare_terms = {
        v.lower() for v in (_compact_term(x) for x in [term, *bare_aliases]) if v
    }
    fallback = grounded_fallback_research_query(term, context)
    grounded: List[str] = []
    for query in queries:
        normalized = js_trim(_WS_RUN_RE.sub(" ", query))
        comparison_key = _compact_term(normalized).lower()
        chosen = normalized if (normalized and comparison_key not in bare_terms) else fallback
        if chosen:
            grounded.append(chosen)
    seen = set()
    out: List[str] = []
    for query in grounded:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(query)
    return out
