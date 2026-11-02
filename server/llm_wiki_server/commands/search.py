"""Port of commands/search.rs (keyword + vector + graph project search, embedding fetchers, page links).

Public API (also used by the REST layer):
    async search_project_inner(project_path, query, top_k, include_content, query_embedding) -> dict
    async resolve_query_embedding(query, explicit_embedding, embedding_config) -> Optional[list[float]]
    tokenize_query(query) -> list[str]
    extract_title(content, file_name) -> str
    extract_image_refs(content) -> list[{"url","alt"}]
    build_snippet(content, query) -> str
    extract_graph_links(content) -> list[str]
    async fetch_embedding_with_retry(text, cfg, max_retries) / fetch_embedding_batch(texts, cfg)
    supports_embedding_batch(cfg)
    SearchEmbeddingConfig.from_dict(d)
"""
from __future__ import annotations

import asyncio
import bisect
import copy
import inspect
import json
import math
import os
import string
import struct
import sys
from dataclasses import dataclass, field
from itertools import accumulate
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, quote_plus, urlsplit, urlunsplit

import httpx

from ..errors import CommandError
from ..registry import command

DEFAULT_RESULTS = 20
MAX_RESULTS = 50
RRF_K = 60.0
FILENAME_EXACT_BONUS = 200.0
PHRASE_IN_TITLE_BONUS = 50.0
PHRASE_IN_CONTENT_PER_OCC = 20.0
MAX_PHRASE_OCC_COUNTED = 10
TITLE_TOKEN_WEIGHT = 5.0
CONTENT_TOKEN_WEIGHT = 1.0
SNIPPET_CONTEXT = 80
SEARCH_EMBEDDING_TIMEOUT_SECS = 8.0
MAX_SEARCH_FILES = 10_000
MIN_GRAPH_RESULT_RATIO = 0.15
MAX_GRAPH_RESULT_RATIO = 0.30
MAX_GRAPH_SEEDS = 20


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


# --------------------------------------------------------------------------- small utils


def _f32(x: float) -> float:
    """Round to float32 and return the float whose repr is the shortest f32 repr (matches serde_json f32)."""
    try:
        v = struct.unpack("f", struct.pack("f", x))[0]
    except (OverflowError, struct.error):
        return x
    if math.isinf(v) or math.isnan(v):
        return v
    for p in range(1, 10):
        cand = float("%.*g" % (p, v))
        try:
            if struct.unpack("f", struct.pack("f", cand))[0] == v:
                return cand
        except OverflowError:
            break
    return v


def _lines(s: str) -> List[str]:
    """Rust str::lines(): split on \\n, strip one trailing \\r per line, no trailing empty line."""
    if not s:
        return []
    parts = s.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return [p[:-1] if p.endswith("\r") else p for p in parts]


def _trim_end_matches(s: str, suffix: str) -> str:
    while suffix and s.endswith(suffix):
        s = s[: -len(suffix)]
    return s


def _trim_start_matches(s: str, prefix: str) -> str:
    while prefix and s.startswith(prefix):
        s = s[len(prefix):]
    return s


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def file_stem(path: str) -> str:
    name = path.rstrip("/").rsplit("/", 1)[-1]
    idx = name.rfind(".")
    return name if idx <= 0 else name[:idx]


def _relative_to_project(project_path: str, path: str) -> str:
    try:
        rel = Path(path).relative_to(Path(project_path))
        return str(rel).replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def vector_page_id(path: str) -> Optional[str]:
    normalized = normalize_path(path)
    wiki_relative = normalized[5:] if normalized.startswith("wiki/") else normalized
    if len(wiki_relative) < 3 or wiki_relative[-3:].lower() != ".md":
        return None
    pid = wiki_relative[:-3].strip("/")
    return pid or None


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _walk_files(root: str):
    """WalkDir-equivalent: yields regular files (symlinks are not followed / not files)."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            if os.path.islink(full):
                continue
            yield full


# --------------------------------------------------------------------------- tokenizer


_ASCII_PUNCT = set(string.punctuation)
_CJK_SEPARATORS = set("，。！？、；：“”‘’（）·～…")


def _is_query_separator(c: str) -> bool:
    return c.isspace() or c in _ASCII_PUNCT or c in _CJK_SEPARATORS


_STOP_WORDS = {
    "的", "是", "了", "什么", "在", "有", "和", "与", "对", "从",
    "the", "is", "a", "an", "what", "how", "are", "was", "were", "do", "does", "did",
    "be", "been", "being", "have", "has", "had", "it", "its", "in", "on", "at", "to",
    "for", "of", "with", "by", "this", "that", "these", "those",
}


def _split_query(s: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    for c in s:
        if _is_query_separator(c):
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    out.append("".join(cur))
    return out


def tokenize_query(query: str) -> List[str]:
    raw = [t for t in _split_query(query.lower()) if len(t) > 1 and t not in _STOP_WORDS]
    out: List[str] = []
    for token in raw:
        has_cjk = any("㐀" <= c <= "鿿" for c in token)
        if has_cjk and len(token) > 2:
            for i in range(len(token) - 1):
                out.append(token[i : i + 2])
            for ch in token:
                if ch not in _STOP_WORDS:
                    out.append(ch)
            out.append(token)
        else:
            out.append(token)
    return sorted(set(out))


def trim_query_punctuation(value: str) -> str:
    start, end = 0, len(value)
    while start < end and _is_query_separator(value[start]):
        start += 1
    while end > start and _is_query_separator(value[end - 1]):
        end -= 1
    return value[start:end]


def _token_match_score(text: str, tokens: List[str]) -> int:
    lower = text.lower()
    return sum(1 for t in tokens if t in lower)


def _count_occurrences(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    return haystack.count(needle)


# --------------------------------------------------------------------------- content helpers


def extract_title(content: str, file_name: str) -> str:
    has_frontmatter = content.startswith("---")
    in_frontmatter = has_frontmatter
    frontmatter_closed = False
    lines = _lines(content)
    for line in lines[1:] if has_frontmatter else lines:
        trimmed = line.strip()
        if in_frontmatter and trimmed == "---":
            in_frontmatter = False
            frontmatter_closed = True
            continue
        if in_frontmatter and trimmed.startswith("title:"):
            return _trim_start_matches(trimmed, "title:").strip().strip('"').strip("'")
        if has_frontmatter and not frontmatter_closed:
            continue
        if trimmed.startswith("# "):
            return trimmed[2:].strip()
    return _trim_end_matches(file_name, ".md").replace("-", " ")


def extract_image_refs(content: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    rest = content
    while True:
        start = rest.find("![")
        if start < 0:
            break
        rest = rest[start + 2:]
        alt_end = rest.find("](")
        if alt_end < 0:
            break
        alt = rest[:alt_end]
        rest = rest[alt_end + 2:]
        url_end = rest.find(")")
        if url_end < 0:
            break
        url = rest[:url_end]
        if url.strip() and not any(c.isspace() for c in url) and url not in seen:
            seen.add(url)
            out.append({"url": url, "alt": alt})
        rest = rest[url_end + 1:]
    return out


def build_snippet(content: str, query: str) -> str:
    lower = content.lower()
    q = query.lower()
    idx = lower.encode("utf-8").find(q.encode("utf-8"))
    if idx < 0:
        idx = 0
    n = len(content)
    if n == 0:
        return ""
    # byte offset of each char (as in Rust char_indices)
    offsets = [0] + list(accumulate(len(c.encode("utf-8")) for c in content))[:-1]
    pos = bisect.bisect_left(offsets, idx)
    match_char = pos if pos < n else n - 1
    query_chars = max(len(query), 1)
    start_char = max(match_char - SNIPPET_CONTEXT, 0)
    end_char = min(match_char + query_chars + SNIPPET_CONTEXT, n)
    snippet = content[start_char:end_char].replace("\n", " ")
    if start_char > 0:
        snippet = f"...{snippet}"
    if end_char < n:
        snippet += "..."
    return snippet


def _normalize_related_value(value: str) -> Optional[str]:
    value = value.strip().strip('"').strip("'").strip()
    if value.startswith("[[") and value.endswith("]]") and len(value) >= 4:
        value = value[2:-2]
    value = value.strip()
    return value or None


def _extract_related_frontmatter(content: str) -> List[str]:
    normalized = content.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return []
    rest = normalized[4:]
    end = rest.find("\n---")
    if end < 0:
        return []
    lines = _lines(rest[:end])
    out: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        t = line.strip()
        if not t.startswith("related:"):
            index += 1
            continue
        value = t[len("related:"):].strip()
        if value.startswith("[") and value.endswith("]") and len(value) >= 2:
            for item in value[1:-1].split(","):
                v = _normalize_related_value(item)
                if v is not None:
                    out.append(v)
        elif value == "":
            index += 1
            while index < len(lines):
                trimmed = lines[index].strip()
                if not trimmed.startswith("-"):
                    break
                v = _normalize_related_value(trimmed[1:])
                if v is not None:
                    out.append(v)
                index += 1
            continue
        index += 1
    return out


def extract_graph_links(content: str) -> List[str]:
    links: List[str] = []
    rest = content
    while True:
        start = rest.find("[[")
        if start < 0:
            break
        rest = rest[start + 2:]
        end = rest.find("]]")
        if end < 0:
            break
        target = rest[:end].split("|")[0].strip()
        if target:
            links.append(target)
        rest = rest[end + 2:]
    for related in _extract_related_frontmatter(content):
        if related not in links:
            links.append(related)
    return links


# --------------------------------------------------------------------------- result model


@dataclass
class _Result:
    path: str
    title: str
    snippet: str
    title_match: bool
    score: float
    vector_score: Optional[float] = None
    images: List[Dict[str, str]] = field(default_factory=list)
    content: Optional[str] = None
    graph_related_to: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "path": self.path,
            "title": self.title,
            "snippet": self.snippet,
            "titleMatch": self.title_match,
            "score": self.score,
        }
        if self.vector_score is not None:
            d["vectorScore"] = self.vector_score
        d["images"] = self.images
        if self.content is not None:
            d["content"] = self.content
        if self.graph_related_to:
            d["graphRelatedTo"] = self.graph_related_to
        return d


@dataclass
class _GraphPage:
    path: str
    title: str
    content: str
    links: List[str]


@dataclass
class _PageVectorResult:
    id: str
    score: float
    chunk_text: str
    heading_path: str


# --------------------------------------------------------------------------- scoring


def _score_file(project_path: str, full_path: str, content: str, tokens: List[str],
                query_phrase: str, query: str, include_content: bool) -> Optional[_Result]:
    file_name = os.path.basename(full_path)
    title = extract_title(content, file_name)
    title_text = f"{title} {file_name}"
    title_lower = title_text.lower()
    content_lower = content.lower()
    stem = _trim_end_matches(file_name, ".md").lower()

    filename_exact = bool(query_phrase) and stem == query_phrase
    title_has_phrase = bool(query_phrase) and query_phrase in title_lower
    content_phrase_occ = min(_count_occurrences(content_lower, query_phrase), MAX_PHRASE_OCC_COUNTED)
    title_token_score = _token_match_score(title_text, tokens)
    content_token_score = _token_match_score(content, tokens)

    if (not filename_exact and not title_has_phrase and content_phrase_occ == 0
            and title_token_score == 0 and content_token_score == 0):
        return None

    score = (
        (FILENAME_EXACT_BONUS if filename_exact else 0.0)
        + (PHRASE_IN_TITLE_BONUS if title_has_phrase else 0.0)
        + content_phrase_occ * PHRASE_IN_CONTENT_PER_OCC
        + title_token_score * TITLE_TOKEN_WEIGHT
        + content_token_score * CONTENT_TOKEN_WEIGHT
    )
    if content_phrase_occ > 0:
        anchor = query_phrase
    else:
        anchor = next((t for t in tokens if t in content_lower), query)
    return _Result(
        path=_relative_to_project(project_path, full_path),
        title=title,
        snippet=build_snippet(content, anchor),
        title_match=title_token_score > 0 or title_has_phrase,
        score=float(score),
        images=extract_image_refs(content),
        content=content if include_content else None,
    )


def _search_mode(token_rank_empty: bool, vector_hits: int, graph_hits: int) -> str:
    if graph_hits > 0:
        return "hybrid"
    if vector_hits == 0:
        return "keyword"
    if token_rank_empty:
        return "vector"
    return "hybrid"


def _apply_rrf_scores(results: List[_Result], token_rank: Dict[str, int], vector_rank: Dict[str, int],
                      vector_score: Dict[str, float], legacy_page_paths: Dict[str, str]) -> None:
    for result in results:
        token = token_rank.get(normalize_path(result.path))
        vector_id = vector_page_id(result.path) or file_stem(result.path)
        legacy_stem = file_stem(result.path)
        lp = legacy_page_paths.get(legacy_stem)
        legacy_matches = lp is not None and normalize_path(lp) == normalize_path(result.path)

        def lookup(m: Dict[str, Any]):
            v = m.get(vector_id)
            if v is None and legacy_matches:
                v = m.get(legacy_stem)
            return v

        vector = lookup(vector_rank)
        rrf = 0.0
        if token is not None:
            rrf += 1.0 / (RRF_K + token)
        if vector is not None:
            rrf += 1.0 / (RRF_K + vector)
        vs = lookup(vector_score)
        if vs is not None:
            result.vector_score = vs
        result.score = rrf


def _graph_result_quota(limit: int, vector_hits: int) -> int:
    if limit < 2:
        return 0
    coverage = min(vector_hits, limit) / limit
    ratio = MAX_GRAPH_RESULT_RATIO - (MAX_GRAPH_RESULT_RATIO - MIN_GRAPH_RESULT_RATIO) * coverage
    q = int(math.ceil(limit * ratio))
    return max(1, min(q, limit - 1))


def _normalize_graph_alias(value: str) -> str:
    v = value.split("#")[0].strip()
    v = _trim_end_matches(v, ".md")
    return v.replace("\\", "/").replace(" ", "-").lower()


def _blend_graph_results(ranked: List[_Result], pages: Dict[str, _GraphPage], limit: int,
                         vector_hits: int, include_content: bool) -> Tuple[List[_Result], int]:
    if not ranked or not pages:
        return ranked[:limit], 0

    aliases: Dict[str, str] = {}
    for normalized_path in sorted(pages):
        page = pages[normalized_path]
        wiki_relative = page.path[5:] if page.path.startswith("wiki/") else page.path
        stem = file_stem(page.path)
        for alias in (page.path, wiki_relative, stem, page.title):
            aliases[_normalize_graph_alias(alias)] = normalized_path

    adjacency: Dict[str, set] = {}
    for source in sorted(pages):
        for link in pages[source].links:
            target = aliases.get(_normalize_graph_alias(link))
            if target is None or source == target:
                continue
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)

    seed_paths = [normalize_path(r.path) for r in ranked[: min(limit, MAX_GRAPH_SEEDS)]]
    seed_set = set(seed_paths)
    candidate_scores: Dict[str, float] = {}
    candidate_seeds: Dict[str, set] = {}
    for rank, seed in enumerate(seed_paths):
        neighbors = adjacency.get(seed)
        if not neighbors:
            continue
        for neighbor in sorted(neighbors):
            if neighbor in seed_set:
                continue
            candidate_scores[neighbor] = candidate_scores.get(neighbor, 0.0) + 1.0 / (rank + 1)
            seed_page = pages.get(seed)
            if seed_page is not None:
                candidate_seeds.setdefault(neighbor, set()).add(seed_page.title)

    candidates = sorted(candidate_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    candidates = candidates[: _graph_result_quota(limit, vector_hits)]
    if not candidates:
        return ranked[:limit], 0

    selected_paths = {p for p, _ in candidates}
    existing: Dict[str, _Result] = {}
    ranked_paths: List[str] = []
    for r in ranked:
        p = normalize_path(r.path)
        ranked_paths.append(p)
        existing[p] = r

    graph_count = len(candidates)
    base_limit = max(limit - graph_count, 0)
    base_results: List[_Result] = []
    for p in ranked_paths:
        if len(base_results) >= base_limit:
            break
        if p in selected_paths:
            continue
        r = existing.get(p)
        if r is not None:
            base_results.append(copy.copy(r))

    for path, graph_score in candidates:
        res = existing.pop(path, None)
        if res is not None:
            res.graph_related_to = sorted(candidate_seeds.pop(path, set()))
            base_results.append(res)
            continue
        page = pages.get(path)
        if page is None:
            continue
        related_titles = sorted(candidate_seeds.pop(path, set()))
        base_results.append(_Result(
            path=page.path,
            title=page.title,
            snippet=f"Graph neighbor of {', '.join(related_titles)}",
            title_match=False,
            score=graph_score / (RRF_K + 1.0),
            images=extract_image_refs(page.content),
            content=page.content if include_content else None,
            graph_related_to=related_titles,
        ))
    return base_results, graph_count


# --------------------------------------------------------------------------- vector side


async def _call_maybe_async(fn, *args):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    res = await asyncio.to_thread(fn, *args)
    if inspect.isawaitable(res):
        res = await res
    return res


def _chunk_get(chunk: Any, snake: str, camel: str, default: Any = None) -> Any:
    if isinstance(chunk, dict):
        if camel in chunk:
            return chunk[camel]
        return chunk.get(snake, default)
    if hasattr(chunk, snake):
        return getattr(chunk, snake)
    return getattr(chunk, camel, default)


async def _search_by_embedding(project_path: str, query_embedding: List[float], top_k: int
                               ) -> List[_PageVectorResult]:
    try:
        from . import vectorstore  # type: ignore
        fn = getattr(vectorstore, "vector_search_chunks")
    except (ImportError, AttributeError):
        raise CommandError("Vector search is unavailable")
    raw_chunks = await _call_maybe_async(fn, project_path, query_embedding, max(top_k * 3, 30))
    if not raw_chunks:
        return []
    by_page: Dict[str, list] = {}
    for chunk in raw_chunks:
        by_page.setdefault(str(_chunk_get(chunk, "page_id", "pageId", "")), []).append(chunk)
    ranked: List[_PageVectorResult] = []
    for pid in sorted(by_page):
        chunks = by_page[pid]
        chunks.sort(key=lambda c: (-float(_chunk_get(c, "score", "score", 0.0)),
                                   int(_chunk_get(c, "chunk_index", "chunkIndex", 0))))
        top_chunk = chunks[0]
        top = float(_chunk_get(top_chunk, "score", "score", 0.0))
        tail = sum(float(_chunk_get(c, "score", "score", 0.0)) for c in chunks[1:])
        blended = top + min(tail * 0.3, max(1.0 - top, 0.0))
        ranked.append(_PageVectorResult(
            id=pid, score=_f32(blended),
            chunk_text=_chunk_get(top_chunk, "chunk_text", "chunkText", "") or "",
            heading_path=_chunk_get(top_chunk, "heading_path", "headingPath", "") or "",
        ))
    ranked.sort(key=lambda r: (-r.score, r.id))
    return ranked[:top_k]


def _build_vector_snippet(r: _PageVectorResult) -> str:
    text = r.chunk_text.strip().replace("\n", " ")
    if not text:
        return ""
    if len(text) > SNIPPET_CONTEXT * 2:
        text = text[: SNIPPET_CONTEXT * 2] + "..."
    heading = r.heading_path.strip()
    return text if not heading else f"{heading}: {text}"


def _materialize_vector_only_results(vector_results: List[_PageVectorResult],
                                     page_paths_by_vector_id: Dict[str, str],
                                     legacy_page_paths: Dict[str, str], project_path: str,
                                     results: List[_Result], include_content: bool) -> None:
    known = {normalize_path(r.path) for r in results}
    for qualified_only in (True, False):
        for vr in vector_results:
            if qualified_only:
                rel = page_paths_by_vector_id.get(vr.id)
            elif vr.id in page_paths_by_vector_id:
                rel = None
            else:
                rel = legacy_page_paths.get(vr.id)
            if rel is None:
                continue
            nrel = normalize_path(rel)
            if nrel in known:
                continue
            content = _read_text(os.path.join(project_path, rel))
            if content is None:
                continue
            title = extract_title(content, rel.rsplit("/", 1)[-1])
            results.append(_Result(
                path=rel, title=title, snippet=_build_vector_snippet(vr), title_match=False,
                score=0.0, vector_score=vr.score, images=extract_image_refs(content),
                content=content if include_content else None,
            ))
            known.add(nrel)


# --------------------------------------------------------------------------- main search


def _scan_project(project_path: str, query: str, effective_tokens: List[str], query_phrase: str,
                  include_content: bool):
    results: List[_Result] = []
    page_paths_by_vector_id: Dict[str, str] = {}
    legacy_stem_candidates: Dict[str, List[str]] = {}
    graph_pages: Dict[str, _GraphPage] = {}
    wiki_root = os.path.join(project_path, "wiki")
    if os.path.exists(wiki_root):
        searched = 0
        for full in _walk_files(wiki_root):
            if not full.lower().endswith(".md"):
                continue
            searched += 1
            if searched > MAX_SEARCH_FILES:
                _eprint(f"[Search] stopped scanning wiki after {MAX_SEARCH_FILES} markdown files in {project_path}")
                break
            content = _read_text(full)
            if content is None:
                continue
            rel = _relative_to_project(project_path, full)
            vid = vector_page_id(rel)
            if vid is not None:
                page_paths_by_vector_id[vid] = rel
            stem = file_stem(full)
            legacy_stem_candidates.setdefault(stem, []).append(rel)
            title = extract_title(content, os.path.basename(full))
            hit = _score_file(project_path, full, content, effective_tokens, query_phrase, query, include_content)
            graph_pages[normalize_path(rel)] = _GraphPage(
                path=rel, title=title, content=content, links=extract_graph_links(content))
            if hit is not None:
                results.append(hit)
    legacy_page_paths = {s: ps[0] for s, ps in legacy_stem_candidates.items() if len(ps) == 1}
    return results, page_paths_by_vector_id, legacy_page_paths, graph_pages


async def search_project_inner(project_path: str, query: str, top_k: int, include_content: bool,
                               query_embedding: Optional[List[float]] = None) -> Dict[str, Any]:
    if not query.strip():
        raise CommandError("query is required")
    limit = max(1, min(top_k, MAX_RESULTS))
    tokens = tokenize_query(query)
    effective_tokens = tokens if tokens else [query.strip().lower()]
    query_phrase = trim_query_punctuation(query.lower())

    results, page_paths_by_vector_id, legacy_page_paths, graph_pages = await asyncio.to_thread(
        _scan_project, project_path, query, effective_tokens, query_phrase, include_content)

    token_sorted = sorted(range(len(results)), key=lambda i: (-results[i].score, results[i].path))
    token_rank: Dict[str, int] = {}
    for idx, ri in enumerate(token_sorted):
        token_rank[normalize_path(results[ri].path)] = idx + 1

    vector_rank: Dict[str, int] = {}
    vector_score: Dict[str, float] = {}
    vector_hits = 0
    if query_embedding:
        try:
            vector_results = await _search_by_embedding(project_path, query_embedding, max(limit, 10))
            vector_hits = len(vector_results)
            for idx, vr in enumerate(vector_results):
                vector_rank[vr.id] = idx + 1
                vector_score[vr.id] = vr.score
            await asyncio.to_thread(
                _materialize_vector_only_results, vector_results, page_paths_by_vector_id,
                legacy_page_paths, project_path, results, include_content)
        except Exception as err:  # noqa: BLE001 - mirrors Rust: fall back to keyword results
            _eprint(f"[Search] vector search failed; falling back to keyword results: "
                    f"{getattr(err, 'message', err)}")

    if vector_hits > 0:
        _apply_rrf_scores(results, token_rank, vector_rank, vector_score, legacy_page_paths)

    results.sort(key=lambda r: (-r.score, r.path))
    results, graph_hits = _blend_graph_results(results, graph_pages, limit, vector_hits, include_content)

    return {
        "mode": _search_mode(not token_rank, vector_hits, graph_hits),
        "results": [r.to_dict() for r in results],
        "tokenHits": len(token_rank),
        "vectorHits": vector_hits,
        "graphHits": graph_hits,
    }


# --------------------------------------------------------------------------- embeddings


class SearchEmbeddingConfig:
    def __init__(self, enabled: bool, endpoint: str, api_key: str, model: str,
                 output_dimensionality: Optional[float] = None,
                 extra_headers: Optional[Dict[str, str]] = None,
                 max_chunk_chars: Optional[int] = None, overlap_chunk_chars: Optional[int] = None):
        self.enabled = enabled
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.extra_headers = extra_headers
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chunk_chars = overlap_chunk_chars

    @classmethod
    def from_dict(cls, d: Any, arg: str = "embeddingConfig", cmd: str = "search_project"
                  ) -> "SearchEmbeddingConfig":
        if isinstance(d, cls):
            return d
        pre = f"invalid args `{arg}` for command `{cmd}`: "
        if not isinstance(d, dict):
            raise CommandError(pre + "invalid type, expected struct SearchEmbeddingConfig")
        for key, typ, name in (("enabled", bool, "bool"), ("endpoint", str, "string"),
                               ("apiKey", str, "string"), ("model", str, "string")):
            if key not in d:
                raise CommandError(pre + f"missing field `{key}`")
            if not isinstance(d[key], typ):
                raise CommandError(pre + f"invalid type for `{key}`, expected a {name}")
        eh = d.get("extraHeaders")
        if eh is not None and not isinstance(eh, dict):
            raise CommandError(pre + "invalid type for `extraHeaders`, expected a map")
        od = d.get("outputDimensionality")
        return cls(d["enabled"], d["endpoint"], d["apiKey"], d["model"],
                   float(od) if isinstance(od, (int, float)) and not isinstance(od, bool) else None,
                   {str(k): str(v) for k, v in eh.items()} if eh else None,
                   d.get("maxChunkChars"), d.get("overlapChunkChars"))


def _is_safe_extra_header_name(name: str) -> bool:
    if not name:
        return False
    allowed = set("!#$%&'*+-.^_`|~") | set(string.digits) | set(string.ascii_letters)
    return all(c in allowed for c in name)


def _is_reserved_extra_header_name(name: str) -> bool:
    return name.strip().lower() in (
        "authorization", "content-type", "host", "content-length", "origin", "x-goog-api-key")


def _is_google_embedding_config(cfg: SearchEmbeddingConfig) -> bool:
    e = cfg.endpoint.lower()
    return "generativelanguage.googleapis.com" in e or ":embedcontent" in e


def _is_doubao_multimodal_embedding_config(cfg: SearchEmbeddingConfig) -> bool:
    return "doubao-embedding-vision" in cfg.model.strip().lower()


def _parse_url(endpoint: str):
    if "://" not in endpoint:
        return None
    try:
        sp = urlsplit(endpoint)
        sp.port  # noqa: B018 - validates port
    except ValueError:
        return None
    if not sp.scheme or not sp.netloc:
        return None
    return sp


def _lower_host_netloc(netloc: str) -> str:
    if "@" in netloc:
        userinfo, hostport = netloc.rsplit("@", 1)
        return userinfo + "@" + hostport.lower()
    return netloc.lower()


def _unsplit(sp, path: str, query: Optional[str] = None) -> str:
    return urlunsplit((sp.scheme.lower(), _lower_host_netloc(sp.netloc), path or "/",
                       sp.query if query is None else query, sp.fragment))


def _is_local_or_private_http_endpoint(endpoint: str) -> bool:
    sp = _parse_url(endpoint)
    if sp is None or sp.scheme.lower() not in ("http", "https"):
        return False
    host = sp.hostname
    if not host:
        return False
    host = host.strip("[]").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    parts = host.split(".")
    if len(parts) != 4 or not all(p.isascii() and p.isdigit() and int(p) <= 255 for p in parts):
        return False
    o = [int(p) for p in parts]
    return (o[0] == 10 or (o[0] == 172 and 16 <= o[1] <= 31) or (o[0] == 192 and o[1] == 168)
            or o[0] == 127)


def _is_volcengine_embedding_endpoint(endpoint: str) -> bool:
    sp = _parse_url(endpoint)
    if sp is not None and sp.hostname:
        host = sp.hostname.lower()
    else:
        trimmed = endpoint.strip()
        rest = trimmed.split("://", 1)[1] if "://" in trimmed else trimmed
        for sep in "/?#":
            rest = rest.split(sep)[0]
        host = rest.lower()
    return host == "volces.com" or host.endswith(".volces.com") or "volcengine" in host


def _append_endpoint_path(endpoint: str, target_suffix: str) -> str:
    suffix = target_suffix.lstrip("/")
    lower_suffix = "/" + suffix.lower()
    sp = _parse_url(endpoint)
    if sp is not None:
        path = sp.path.rstrip("/")
        lower_path = path.lower()
        if lower_path.endswith(lower_suffix):
            return _unsplit(sp, path or "/")
        if lower_path.endswith("/embeddings/multimodal") and lower_suffix == "/embeddings":
            base = _trim_end_matches(path, "/multimodal")
            return _unsplit(sp, base or "/")
        if lower_path.endswith("/embeddings") and lower_suffix == "/embeddings/multimodal":
            return _unsplit(sp, f"{path}/multimodal")
        return _unsplit(sp, f"{path}/{suffix}".replace("//", "/"))
    base, _, query = endpoint.partition("?")
    trimmed = base.rstrip("/")
    lower = trimmed.lower()
    if lower.endswith(lower_suffix):
        nxt = trimmed
    elif lower.endswith("/embeddings/multimodal") and lower_suffix == "/embeddings":
        nxt = _trim_end_matches(trimmed, "/multimodal")
    elif lower.endswith("/embeddings") and lower_suffix == "/embeddings/multimodal":
        nxt = f"{trimmed}/multimodal"
    else:
        nxt = f"{trimmed}/{suffix}"
    return nxt if not query else f"{nxt}?{query}"


def _volcengine_embedding_endpoint(cfg: SearchEmbeddingConfig) -> str:
    raw = cfg.endpoint.strip()
    if not _is_volcengine_embedding_endpoint(raw):
        return raw
    suffix = "/embeddings/multimodal" if _is_doubao_multimodal_embedding_config(cfg) else "/embeddings"
    return _append_endpoint_path(raw, suffix)


def _strip_google_api_key_query(endpoint: str) -> str:
    if "?" not in endpoint:
        return endpoint
    sp = _parse_url(endpoint)
    if sp is not None:
        kept = [(k, v) for k, v in parse_qsl(sp.query, keep_blank_values=True) if k.lower() != "key"]
        query = "&".join(f"{quote_plus(k)}={quote_plus(v)}" for k, v in kept)
        return _unsplit(sp, sp.path, query).rstrip("?")
    base, _, query = endpoint.partition("?")
    kept_pairs = [p for p in query.split("&") if p.split("=", 1)[0].lower() != "key"]
    return base if not kept_pairs else f"{base}?{'&'.join(kept_pairs)}"


def _google_embedding_endpoint(cfg: SearchEmbeddingConfig) -> str:
    raw = _strip_google_api_key_query(cfg.endpoint.strip()).rstrip("/")
    low = raw.lower()
    if ":batchembedcontents" in low:
        return raw.replace(":batchEmbedContents", ":embedContent").replace(":batchembedcontents", ":embedContent")
    if ":embedcontent" in low:
        return raw
    model = _trim_start_matches(cfg.model.strip(), "models/")
    if "/models/" in low:
        return f"{raw}:embedContent"
    return f"{raw}/models/{model}:embedContent"


def _google_embedding_body(model: str, text: str, output_dimensionality: Optional[float]) -> Dict[str, Any]:
    m = model.strip()
    model_path = m if m.startswith("models/") else f"models/{m}"
    body: Dict[str, Any] = {"model": model_path, "content": {"parts": [{"text": text}]}}
    if output_dimensionality is not None and math.isfinite(output_dimensionality) and output_dimensionality >= 1.0:
        body["output_dimensionality"] = int(math.floor(output_dimensionality))
    return body


def _doubao_multimodal_embedding_body(model: str, text: str) -> Dict[str, Any]:
    return {"model": model, "encoding_format": "float", "input": [{"type": "text", "text": text}]}


def _looks_like_oversize_error(status: int, body: str) -> bool:
    if status == 413:
        return True
    lower = body.lower()
    return any(s in lower for s in (
        "too long", "maximum context", "max_tokens", "max tokens", "context length",
        "token limit", "exceeds", "input length"))


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _parse_embedding_values(data: Any, is_google: bool, is_doubao_multimodal: bool) -> List[float]:
    def get(o: Any, k: str) -> Any:
        return o.get(k) if isinstance(o, dict) else None

    if is_google:
        values = get(get(data, "embedding"), "values")
    elif is_doubao_multimodal:
        values = get(get(data, "data"), "embedding")
    else:
        arr = get(data, "data")
        values = get(arr[0], "embedding") if isinstance(arr, list) and arr else None
    if not isinstance(values, list):
        raise CommandError("Embedding response missing vector")
    out: List[float] = []
    for v in values:
        n = _num(v)
        if n is None:
            raise CommandError("Embedding response contains non-number values")
        if not math.isfinite(n):
            raise CommandError("Embedding response contains non-finite values")
        out.append(_f32(n))
    if not out:
        raise CommandError("Embedding response vector is empty")
    return out


def _parse_embedding_batch_values(data: Any, expected: int) -> List[List[float]]:
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise CommandError("Embedding batch response missing data array")
    if len(entries) != expected:
        raise CommandError(f"Embedding batch returned {len(entries)} vectors for {expected} inputs")
    indexed: List[Tuple[int, List[float]]] = []
    for position, entry in enumerate(entries):
        e = entry if isinstance(entry, dict) else {}
        idx = e.get("index")
        index = idx if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0 else position
        if index >= expected:
            raise CommandError("Embedding batch response contains an out-of-range index")
        values = e.get("embedding")
        if not isinstance(values, list):
            raise CommandError("Embedding batch response missing vector")
        vec: List[float] = []
        for v in values:
            n = _num(v)
            if n is None:
                raise CommandError("Embedding batch response contains non-number values")
            if not math.isfinite(n):
                raise CommandError("Embedding batch response contains non-finite values")
            vec.append(_f32(n))
        if not vec:
            raise CommandError("Embedding batch response vector is empty")
        indexed.append((index, vec))
    indexed.sort(key=lambda p: p[0])
    if any(indexed[i][0] == indexed[i + 1][0] for i in range(len(indexed) - 1)):
        raise CommandError("Embedding batch response contains duplicate indexes")
    dim = len(indexed[0][1]) if indexed else 0
    if any(len(v) != dim for _, v in indexed):
        raise CommandError("Embedding batch response contains inconsistent vector dimensions")
    return [v for _, v in indexed]


def _build_headers(cfg: SearchEmbeddingConfig, endpoint: str, google: bool) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if _is_local_or_private_http_endpoint(endpoint):
        headers["Origin"] = "http://localhost"
    key = cfg.api_key.strip()
    if key:
        if google:
            headers["x-goog-api-key"] = key
        else:
            headers["Authorization"] = f"Bearer {key}"
    for name, value in (cfg.extra_headers or {}).items():
        name, value = name.strip(), value.strip()
        if not name or not value or not _is_safe_extra_header_name(name):
            continue
        if _is_reserved_extra_header_name(name):
            continue
        headers[name] = value
    return headers


class _Oversize(Exception):
    pass


async def _fetch_embedding_once(text: str, cfg: SearchEmbeddingConfig) -> List[float]:
    is_google = _is_google_embedding_config(cfg)
    is_doubao = _is_doubao_multimodal_embedding_config(cfg)
    endpoint = _google_embedding_endpoint(cfg) if is_google else _volcengine_embedding_endpoint(cfg)
    headers = _build_headers(cfg, endpoint, is_google)
    if is_google:
        body = _google_embedding_body(cfg.model, text, cfg.output_dimensionality)
    elif is_doubao:
        body = _doubao_multimodal_embedding_body(cfg.model, text)
    else:
        body = {"model": cfg.model, "input": text}
    try:
        async with httpx.AsyncClient(timeout=SEARCH_EMBEDDING_TIMEOUT_SECS) as client:
            resp = await client.post(endpoint, headers=headers, json=body)
    except httpx.InvalidURL as e:
        raise CommandError(f"Embedding request failed: {e}")
    except httpx.HTTPError as e:
        raise CommandError(f"Embedding request failed: {e}")
    resp_text = resp.text
    if not resp.is_success:
        preview = resp_text[:200]
        msg = f"Embedding API HTTP {resp.status_code} {resp.reason_phrase}: {preview}"
        if _looks_like_oversize_error(resp.status_code, resp_text):
            raise _Oversize(msg)
        raise CommandError(msg)
    try:
        data = json.loads(resp_text)
    except ValueError as e:
        raise CommandError(f"Embedding response parse failed: {e}: {resp_text[:200]}")
    return _parse_embedding_values(data, is_google, is_doubao)


def _halve_text(text: str) -> Optional[str]:
    n = len(text)
    if n <= 1:
        return None
    return text[: max(n // 2, 1)]


async def fetch_embedding_with_retry(text: str, cfg: Any, max_retries: int = 3) -> List[float]:
    cfg = SearchEmbeddingConfig.from_dict(cfg, "cfg", "embedding_fetch")
    current = text
    attempts = 0
    while True:
        attempts += 1
        try:
            return await _fetch_embedding_once(current, cfg)
        except _Oversize as e:
            if attempts <= max_retries and len(current.encode("utf-8")) > 64:
                halved = _halve_text(current)
                if halved is not None:
                    _eprint(f"[Embedding] auto-halving after oversize error at {len(text)} chars; "
                            f"retrying at {len(halved)} chars ({attempts}/{max_retries + 1})")
                    current = halved
                    continue
            raise CommandError(
                f"Endpoint rejected input even at {len(current.encode('utf-8'))} chars. "
                f"Lower Settings -> Embedding -> Max Chunk Chars. {e}")


def supports_embedding_batch(cfg: Any) -> bool:
    cfg = SearchEmbeddingConfig.from_dict(cfg, "cfg", "embedding_fetch_batch")
    return not _is_google_embedding_config(cfg) and not _is_doubao_multimodal_embedding_config(cfg)


async def fetch_embedding_batch(texts: List[str], cfg: Any) -> List[List[float]]:
    cfg = SearchEmbeddingConfig.from_dict(cfg, "cfg", "embedding_fetch_batch")
    if not texts or len(texts) > 64:
        raise CommandError("Embedding batch must contain between 1 and 64 inputs")
    if not supports_embedding_batch(cfg):
        raise CommandError("This embedding provider does not use the OpenAI-compatible batch format")
    endpoint = _volcengine_embedding_endpoint(cfg)
    headers = _build_headers(cfg, endpoint, False)
    try:
        async with httpx.AsyncClient(timeout=SEARCH_EMBEDDING_TIMEOUT_SECS) as client:
            resp = await client.post(endpoint, headers=headers, json={"model": cfg.model, "input": texts})
    except httpx.HTTPError as e:
        raise CommandError(f"Embedding batch request failed: {e}")
    text = resp.text
    if not resp.is_success:
        raise CommandError(f"Embedding batch API HTTP {resp.status_code} {resp.reason_phrase}: {text[:200]}")
    try:
        data = json.loads(text)
    except ValueError as e:
        raise CommandError(f"Embedding batch response parse failed: {e}: {text[:200]}")
    return _parse_embedding_batch_values(data, len(texts))


def _validate_query_embedding(embedding: List[float]) -> List[float]:
    if not embedding:
        raise CommandError("queryEmbedding must not be empty")
    for v in embedding:
        n = _num(v)
        if n is None:
            raise CommandError("invalid args `queryEmbedding` for command `search_project`: "
                               "invalid type, expected f32")
        if not math.isfinite(n):
            raise CommandError("queryEmbedding must contain only finite numbers")
    return [float(v) for v in embedding]


async def resolve_query_embedding(query: str, explicit_embedding: Optional[List[float]] = None,
                                  embedding_config: Any = None) -> Optional[List[float]]:
    if explicit_embedding is not None:
        return _validate_query_embedding(explicit_embedding)
    if embedding_config is None:
        return None
    cfg = SearchEmbeddingConfig.from_dict(embedding_config)
    if not cfg.enabled or not cfg.endpoint.strip() or not cfg.model.strip():
        return None
    try:
        emb = await fetch_embedding_with_retry(query, cfg, 0)
    except CommandError as err:
        _eprint(f"[Search] embedding disabled for this request: {err.message}")
        return None
    return _validate_query_embedding(emb)


# --------------------------------------------------------------------------- page links


def _resolve_reader_wikilink(pages: Dict[str, _GraphPage], sorted_keys: List[str],
                             by_filename: Dict[str, str], link: str) -> Optional[str]:
    link = link.strip().replace("\\", "/")
    if "/" in link:
        return link if link in pages else None
    filename = link if link.endswith(".md") else f"{link}.md"
    return by_filename.get(filename)


def _dedup_consecutive(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        if out and out[-1][key] == it[key]:
            continue
        out.append(it)
    return out


def _canonicalize(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise CommandError(f"Failed to resolve {what}: No such file or directory (os error 2)")
    return os.path.realpath(path)


def get_page_links_inner(project_path: str, file_path: str) -> Dict[str, Any]:
    project = _canonicalize(project_path, "project path")
    file = _canonicalize(file_path, "page path")
    wiki_root = os.path.join(project, "wiki")
    try:
        Path(file).relative_to(Path(wiki_root))
        under_wiki = True
    except ValueError:
        under_wiki = False
    if not under_wiki or not os.path.isfile(file) or not file.endswith(".md") \
            or os.path.splitext(file)[1] != ".md":
        raise CommandError("Page links target must be an existing Markdown file under wiki/")

    pages: Dict[str, _GraphPage] = {}
    for full in _walk_files(wiki_root):
        if len(pages) >= MAX_SEARCH_FILES:
            break
        if os.path.splitext(full)[1] != ".md":
            continue
        content = _read_text(full)
        if content is None:
            continue
        path = _relative_to_project(project, full)
        title = extract_title(content, os.path.basename(full))
        pages[normalize_path(path)] = _GraphPage(path=path, title=title, content=content,
                                                 links=extract_graph_links(content))

    sorted_keys = sorted(pages)
    by_filename: Dict[str, str] = {}
    for k in sorted_keys:
        by_filename.setdefault(k.rsplit("/", 1)[-1], k)

    current_path = normalize_path(_relative_to_project(project, file))
    current = pages.get(current_path)
    if current is None:
        raise CommandError("Page is not available in the current wiki index")

    outgoing: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for link in current.links:
        target_path = _resolve_reader_wikilink(pages, sorted_keys, by_filename, link)
        if target_path is not None:
            if target_path == current_path:
                continue
            target = pages.get(target_path)
            if target is not None:
                outgoing.append({"title": target.title, "path": target.path, "snippet": None})
        else:
            missing.append({"title": link, "path": None, "snippet": None})

    backlinks: List[Dict[str, Any]] = []
    for path in sorted_keys:
        if path == current_path:
            continue
        page = pages[path]
        if any(_resolve_reader_wikilink(pages, sorted_keys, by_filename, l) == current_path
               for l in page.links):
            backlinks.append({"title": page.title, "path": page.path,
                              "snippet": build_snippet(page.content, current.title)})

    outgoing.sort(key=lambda e: e["title"])
    outgoing = _dedup_consecutive(outgoing, "path")
    backlinks.sort(key=lambda e: e["title"])
    missing.sort(key=lambda e: e["title"])
    missing = _dedup_consecutive(missing, "title")
    return {"outgoing": outgoing, "backlinks": backlinks, "missing": missing}


# --------------------------------------------------------------------------- commands


@command()
async def search_project(project_path: str, query: str, top_k: Optional[int] = None,
                         include_content: Optional[bool] = None,
                         query_embedding: Optional[List[float]] = None,
                         embedding_config: Optional[Dict[str, Any]] = None):
    emb = await resolve_query_embedding(query, query_embedding, embedding_config)
    return await search_project_inner(
        project_path, query,
        DEFAULT_RESULTS if top_k is None else int(top_k),
        bool(include_content) if include_content is not None else False,
        emb,
    )


@command()
async def embedding_fetch(text: str, cfg: Dict[str, Any], max_retries: Optional[int] = None):
    c = SearchEmbeddingConfig.from_dict(cfg, "cfg", "embedding_fetch")
    return await fetch_embedding_with_retry(text, c, 3 if max_retries is None else int(max_retries))


@command()
async def embedding_fetch_batch(texts: List[str], cfg: Dict[str, Any]):
    c = SearchEmbeddingConfig.from_dict(cfg, "cfg", "embedding_fetch_batch")
    return await fetch_embedding_batch(texts, c)


@command()
def get_page_links(project_path: str, file_path: str):
    return get_page_links_inner(project_path, file_path)
