"""Port of commands/page_embedding.rs (no #[tauri::command]s; used by the REST layer).

Public API:
    embed_wiki_page(project_path, relative_path, config, force=False) -> dict   (async)
    PageEmbeddingError(kind, message)  with .status (HTTP status used by api_server.rs)
    load_revision / save_revision / invalidate_page_revision
    chunk_markdown / extract_title / vector_page_id / embedding_fingerprint (pure helpers)
"""
from __future__ import annotations

import asyncio
import errno as _errno
import hashlib
import inspect
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..errors import CommandError
from . import vectorstore

DEFAULT_CHUNK_CHARS = 1_000
DEFAULT_OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 64
MAX_CHUNK_CHARS = 32_000
MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_PAGE_CHUNKS = 512
EMBEDDING_BATCH_SIZE = 64
PROVIDER_PHASE_TIMEOUT = 300.0
REVISION_DIR = ".llm-wiki/embedding-revisions"

INVALID_REQUEST = "InvalidRequest"
NOT_FOUND = "NotFound"
PROVIDER = "Provider"
STORAGE = "Storage"
TIMEOUT = "Timeout"

_STATUS = {INVALID_REQUEST: 400, NOT_FOUND: 404, PROVIDER: 502, STORAGE: 500, TIMEOUT: 504}


class PageEmbeddingError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message

    @property
    def status(self) -> int:
        return _STATUS[self.kind]


def _os_err(e: OSError) -> str:
    return "%s (os error %s)" % (e.strerror or str(e), e.errno)


# --------------------------------------------------------------------------
# config helpers
# --------------------------------------------------------------------------

def _cfg(config: Dict[str, Any], camel: str, snake: str, default: Any = None) -> Any:
    if camel in config:
        return config[camel]
    return config.get(snake, default)


def embedding_fingerprint(revision: str, config: Dict[str, Any]) -> str:
    od = _cfg(config, "outputDimensionality", "output_dimensionality")
    mc = _cfg(config, "maxChunkChars", "max_chunk_chars")
    oc = _cfg(config, "overlapChunkChars", "overlap_chunk_chars")
    signature = {
        "revision": revision,
        "endpoint": str(config.get("endpoint") or "").strip(),
        "model": str(config.get("model") or "").strip(),
        "outputDimensionality": float(od) if od is not None else None,
        "extraHeaders": _cfg(config, "extraHeaders", "extra_headers"),
        "maxChunkChars": mc if mc is not None else DEFAULT_CHUNK_CHARS,
        "overlapChunkChars": oc if oc is not None else DEFAULT_OVERLAP_CHARS,
    }
    # serde_json (no preserve_order) emits keys sorted, compact, non-ASCII raw.
    text = json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# revision metadata
# --------------------------------------------------------------------------

def revision_path(project_path: str, page_id: str) -> Path:
    key = hashlib.sha256(page_id.encode("utf-8")).hexdigest()
    return Path(project_path) / REVISION_DIR / ("%s.revision" % key)


def load_revision(project_path: str, page_id: str) -> Optional[str]:
    try:
        value = revision_path(project_path, page_id).read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    return value or None


def save_revision(project_path: str, page_id: str, revision: str) -> None:
    path = revision_path(project_path, page_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise CommandError("Failed to create revision directory: %s" % _os_err(e))
    temp = path.with_suffix(".tmp-%s" % uuid.uuid4())
    try:
        temp.write_text(revision, encoding="utf-8")
    except OSError as e:
        raise CommandError("Failed to write embedding revision temp file: %s" % _os_err(e))
    try:
        os.replace(temp, path)  # atomic, replaces on Windows too
    except OSError as e:
        try:
            temp.unlink()
        except OSError:
            pass
        raise CommandError("Failed to save embedding revisions: %s" % _os_err(e))


def invalidate_page_revision(project_path: str, page_id: str) -> None:
    try:
        revision_path(project_path, page_id).unlink()
    except FileNotFoundError:
        return
    except OSError as e:
        raise CommandError("Failed to invalidate embedding revision: %s" % _os_err(e))


# --------------------------------------------------------------------------
# path / id helpers
# --------------------------------------------------------------------------

def vector_page_id(normalized_path: str) -> Optional[str]:
    normalized = normalized_path.replace("\\", "/")
    if not normalized.startswith("wiki/"):
        return None
    relative = normalized[len("wiki/"):]
    if len(relative) < 3 or relative[-3:].lower() != ".md":
        return None
    relative = relative[:-3].strip("/")
    return relative or None


def resolve_wiki_markdown_path(project_path: str, relative_path: str) -> Tuple[Path, str]:
    raw = relative_path.strip().replace("\\", "/")
    if not raw:
        raise PageEmbeddingError(INVALID_REQUEST, "path is required")
    parts = raw.split("/")
    comps = [p for p in parts if p not in ("", ".")]
    bad = (
        raw.startswith("/")
        or raw.startswith("./")
        or re.match(r"^[A-Za-z]:", raw) is not None
        or ".." in comps
        or not comps
        or comps[0] != "wiki"
        or os.path.splitext(comps[-1])[1].lower() != ".md"
    )
    if bad:
        raise PageEmbeddingError(INVALID_REQUEST, "path must be a project-relative Markdown file under wiki/")

    try:
        project = Path(os.path.realpath(project_path))
        if not project.exists():
            raise FileNotFoundError(_errno.ENOENT, os.strerror(_errno.ENOENT))
    except OSError as e:
        raise PageEmbeddingError(NOT_FOUND, "Failed to resolve project path: %s" % _os_err(e))
    wiki = Path(os.path.realpath(project / "wiki"))
    if not wiki.exists():
        raise PageEmbeddingError(
            NOT_FOUND,
            "Failed to resolve project wiki directory: %s (os error %d)"
            % (os.strerror(_errno.ENOENT), _errno.ENOENT),
        )
    page = Path(os.path.realpath(project.joinpath(*comps)))
    if not page.exists():
        raise PageEmbeddingError(
            NOT_FOUND,
            "Wiki page not found: %s (os error %d)" % (os.strerror(_errno.ENOENT), _errno.ENOENT),
        )
    inside = page == wiki or wiki in page.parents
    if not inside or not page.is_file() or page.suffix.lower() != ".md":
        raise PageEmbeddingError(INVALID_REQUEST, "path must resolve to a file inside the project wiki directory")
    try:
        normalized = page.relative_to(project).as_posix()
    except ValueError:
        raise PageEmbeddingError(INVALID_REQUEST, "Invalid page path")
    return page, normalized


# --------------------------------------------------------------------------
# markdown helpers (Rust str semantics)
# --------------------------------------------------------------------------

def _lines(text: str) -> List[str]:
    """Rust `str::lines`: split on \\n, drop one trailing \\r per line, no trailing empty line."""
    if text == "":
        return []
    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()
    return [p[:-1] if p.endswith("\r") else p for p in parts]


def _find_frontmatter_close(rest: str) -> Optional[Tuple[int, int]]:
    offset = 0
    for line in _split_inclusive(rest):
        if line.rstrip("\r\n").strip() == "---":
            return offset, len(line)
        offset += len(line)
    if rest.strip() == "---":
        return 0, len(rest)
    return None


def _split_inclusive(s: str) -> List[str]:
    out = []
    start = 0
    while start < len(s):
        i = s.find("\n", start)
        if i == -1:
            out.append(s[start:])
            break
        out.append(s[start:i + 1])
        start = i + 1
    return out


def _leading_frontmatter(content: str) -> Optional[str]:
    if not content.startswith("---\n"):
        return None
    rest = content[4:]
    found = _find_frontmatter_close(rest)
    if found is None:
        return None
    return rest[:found[0]]


def strip_frontmatter(content: str) -> str:
    if not content.startswith("---\n"):
        return content
    rest = content[4:]
    found = _find_frontmatter_close(rest)
    if found is None:
        return content
    end, close_len = found
    after = rest[end + close_len:]
    return after[1:] if after.startswith("\n") else after


def extract_title(content: str, fallback: str) -> str:
    normalized = content.replace("\r\n", "\n")
    fm = _leading_frontmatter(normalized)
    if fm is not None:
        for line in _lines(fm):
            t = line.strip()
            if t.startswith("title:"):
                title = t[len("title:"):].strip().strip("\"'")
                if title:
                    return title
    for line in _lines(normalized):
        t = line.strip()
        if t.startswith("# "):
            title = t[2:].strip()
            if title:
                return title
            break  # find_map yields Some("") -> filtered -> fallback
    return fallback


def _fence_marker(line: str) -> Optional[Tuple[str, int]]:
    t = line.lstrip()
    if not t or t[0] not in ("`", "~"):
        return None
    marker = t[0]
    width = len(t) - len(t.lstrip(marker))
    return (marker, width) if width >= 3 else None


def _parse_heading(line: str) -> Optional[Tuple[int, str]]:
    hashes = len(line) - len(line.lstrip("#"))
    if not 1 <= hashes <= 6 or line[hashes:hashes + 1] != " ":
        return None
    title = line[hashes + 1:].strip()
    return (hashes, title) if title else None


def _split_with_overlap(text: str, target: int, overlap: int) -> List[str]:
    chars = text
    n = len(chars)
    if n <= target:
        return [text]
    out: List[str] = []
    start = 0
    breakers = "。！？.!?;；"
    while start < n:
        hard_end = min(start + target, n)
        end = hard_end
        if hard_end < n:
            for i in range(hard_end - 1, start + target // 2 - 1, -1):
                ch = chars[i]
                if ch.isspace() or ch in breakers:
                    end = i + 1
                    break
        piece = chars[start:end]
        if piece.strip():
            out.append(piece.strip())
        if end == n:
            break
        start = max(max(end - overlap, 0), start + 1)
    return out


def _push_atomic(chunks: List[str], chunk: str) -> None:
    if len(chunk) <= MAX_CHUNK_CHARS:
        chunks.append(chunk)
    else:
        chunks.extend(_split_with_overlap(chunk, MAX_CHUNK_CHARS, DEFAULT_OVERLAP_CHARS))


def _split_preserving_atomic_blocks(text: str, target: int, overlap: int) -> List[str]:
    lines = _lines(text)
    chunks: List[str] = []
    normal: List[str] = []
    i = 0

    def flush_normal():
        if normal:
            chunks.extend(_split_with_overlap("\n".join(normal), target, overlap))
            normal.clear()

    while i < len(lines):
        fm = _fence_marker(lines[i])
        if fm is not None:
            flush_normal()
            marker, width = fm
            start = i
            i += 1
            while i < len(lines):
                cand = _fence_marker(lines[i])
                is_close = cand is not None and cand[0] == marker and cand[1] >= width
                i += 1
                if is_close:
                    break
            _push_atomic(chunks, "\n".join(lines[start:i]))
            continue
        if lines[i].lstrip().startswith("|"):
            flush_normal()
            start = i
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
            _push_atomic(chunks, "\n".join(lines[start:i]))
            continue
        normal.append(lines[i])
        i += 1
    flush_normal()
    return [c for c in chunks if c.strip()]


def chunk_markdown(content: str, target_chars: int, overlap_chars: int) -> List[Dict[str, str]]:
    """Returns [{'text':..., 'heading_path':...}]."""
    normalized = content.replace("\r\n", "\n")
    body = strip_frontmatter(normalized)
    chunks: List[Dict[str, str]] = []
    stack: List[Tuple[int, str]] = []
    heading_path = ""
    section = ""

    def flush():
        nonlocal section
        text = section.strip()
        if text:
            for piece in _split_preserving_atomic_blocks(text, target_chars, overlap_chars):
                chunks.append({"text": piece, "heading_path": heading_path})
        section = ""

    open_fence: Optional[Tuple[str, int]] = None
    for line in _lines(body):
        fm = _fence_marker(line.lstrip())
        if fm is not None:
            if open_fence is None:
                open_fence = fm
            elif open_fence[0] == fm[0] and fm[1] >= open_fence[1]:
                open_fence = None
        heading = _parse_heading(line) if open_fence is None else None
        if heading is not None:
            level, title = heading
            flush()
            stack = [(lv, t) for lv, t in stack if lv < level]
            stack.append((level, title))
            heading_path = " > ".join("%s %s" % ("#" * lv, t) for lv, t in stack)
        if section:
            section += "\n"
        section += line
    flush()
    return chunks


def enrich_chunk(title: str, chunk: Dict[str, str]) -> str:
    parts = [title.strip(), chunk["heading_path"].strip(), chunk["text"].strip()]
    return "\n\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# provider phase
# --------------------------------------------------------------------------

def _chunk_row(index: int, chunk: Dict[str, str], embedding: List[float]) -> Dict[str, Any]:
    return {
        "chunk_index": index,
        "chunk_text": chunk["text"],
        "heading_path": chunk["heading_path"],
        "embedding": embedding,
    }


def validate_embedding_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise PageEmbeddingError(PROVIDER, "Embedding provider returned no vectors")
    expected = len(rows[0]["embedding"])
    if expected == 0 or any(len(r["embedding"]) != expected for r in rows):
        raise PageEmbeddingError(PROVIDER, "Embedding provider returned empty or inconsistent vector dimensions")


async def _call(fn, *args):
    res = fn(*args)
    if inspect.isawaitable(res):
        res = await res
    return res


def _provider_error(e: BaseException) -> PageEmbeddingError:
    return PageEmbeddingError(PROVIDER, getattr(e, "message", None) or str(e))


async def _prepare_embedding_rows(title: str, chunks: List[Dict[str, str]], config: Dict[str, Any]):
    try:
        from . import search  # lazy: written concurrently by another module
        supports = search.supports_embedding_batch
    except (ImportError, AttributeError) as e:
        raise PageEmbeddingError(PROVIDER, "Embedding provider client is unavailable: %s" % e)

    rows: List[Dict[str, Any]] = []
    if await _call(supports, config):
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
            texts = [enrich_chunk(title, c) for c in batch]
            fetch_batch = search.fetch_embedding_batch
            try:
                embeddings = await _call(fetch_batch, texts, config)
            except PageEmbeddingError:
                raise
            except Exception as e:
                raise _provider_error(e)
            if len(embeddings) != len(batch):
                raise PageEmbeddingError(PROVIDER, "Embedding provider returned an incomplete batch")
            for chunk, emb in zip(batch, embeddings):
                rows.append(_chunk_row(len(rows), chunk, list(emb)))
    else:
        fetch_one = search.fetch_embedding_with_retry
        for index, chunk in enumerate(chunks):
            try:
                emb = await _call(fetch_one, enrich_chunk(title, chunk), config, 3)
            except PageEmbeddingError:
                raise
            except Exception as e:
                raise _provider_error(e)
            rows.append(_chunk_row(index, chunk, list(emb)))
    return rows


def _remove_legacy_vector_id_if_safe(project_path: str, page_id: str) -> None:
    if "/" not in page_id:
        return
    legacy_id = page_id.rsplit("/", 1)[-1]
    if not legacy_id:
        return
    owners = 0
    for _root, _dirs, files in os.walk(os.path.join(project_path, "wiki"), followlinks=False):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() == ".md" and _ascii_lower(stem) == _ascii_lower(legacy_id):
                owners += 1
                if owners >= 2:
                    break
        if owners >= 2:
            break
    if owners != 1:
        return
    try:
        vectorstore.vector_delete_page(project_path, legacy_id)
    except Exception as e:
        print("[Embedding] failed to remove legacy vector id '%s': %s" % (legacy_id, getattr(e, "message", e)),
              file=sys.stderr)


def _ascii_lower(s: str) -> str:
    return "".join(c.lower() if "A" <= c <= "Z" else c for c in s)


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------

async def embed_wiki_page(
    project_path: str, relative_path: str, config: Dict[str, Any], force: bool = False
) -> Dict[str, Any]:
    """Embed one wiki page. `config` is a SearchEmbeddingConfig dict (camelCase keys:
    enabled, endpoint, apiKey, model, outputDimensionality, extraHeaders,
    maxChunkChars, overlapChunkChars). Returns the camelCase PageEmbeddingResult dict.
    Raises PageEmbeddingError (kind/message/status)."""
    if not config.get("enabled") or not str(config.get("model") or "").strip():
        raise PageEmbeddingError(INVALID_REQUEST, "Embedding is not enabled or no embedding model is configured")

    page_path, normalized_path = resolve_wiki_markdown_path(project_path, relative_path)
    try:
        size = page_path.stat().st_size
    except OSError as e:
        raise PageEmbeddingError(NOT_FOUND, "Failed to inspect wiki page: %s" % _os_err(e))
    if size > MAX_PAGE_BYTES:
        raise PageEmbeddingError(
            INVALID_REQUEST, "Wiki page exceeds the %d MiB indexing limit" % (MAX_PAGE_BYTES // 1024 // 1024)
        )
    try:
        raw = page_path.read_bytes()
    except FileNotFoundError as e:
        raise PageEmbeddingError(NOT_FOUND, "Failed to read wiki page as UTF-8 text: %s" % _os_err(e))
    except OSError as e:
        raise PageEmbeddingError(INVALID_REQUEST, "Failed to read wiki page as UTF-8 text: %s" % _os_err(e))
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise PageEmbeddingError(
            INVALID_REQUEST,
            "Failed to read wiki page as UTF-8 text: stream did not contain valid UTF-8",
        )

    page_id = vector_page_id(normalized_path)
    if page_id is None:
        raise PageEmbeddingError(INVALID_REQUEST, "Invalid wiki page name")
    try:
        vectorstore.validate_page_id_for_v2(page_id)
    except CommandError as e:
        raise PageEmbeddingError(INVALID_REQUEST, e.message)
    stem = page_id.rsplit("/", 1)[-1]
    if _ascii_lower(stem) in ("index", "log", "overview"):
        raise PageEmbeddingError(
            INVALID_REQUEST,
            "Aggregate wiki pages index.md, log.md, and overview.md are maintained by the app and are not vector-indexed",
        )
    revision = "sha256:" + hashlib.sha256(raw).hexdigest()
    fingerprint = embedding_fingerprint(revision, config)
    if not force:
        try:
            existing = await asyncio.to_thread(
                vectorstore.vector_page_revision_match, project_path, page_id, fingerprint
            )
        except CommandError as e:
            raise PageEmbeddingError(STORAGE, e.message)
        if existing is not None:
            return {
                "path": normalized_path, "pageId": page_id, "revision": revision,
                "chunks": existing, "vectorsWritten": 0, "status": "unchanged",
            }

    title = extract_title(content, page_id)
    mc = _cfg(config, "maxChunkChars", "max_chunk_chars")
    oc = _cfg(config, "overlapChunkChars", "overlap_chunk_chars")
    chunk_chars = max(MIN_CHUNK_CHARS, min(MAX_CHUNK_CHARS, mc if mc is not None else DEFAULT_CHUNK_CHARS))
    overlap_chars = min(oc if oc is not None else DEFAULT_OVERLAP_CHARS, chunk_chars // 2)
    chunks = chunk_markdown(content, chunk_chars, overlap_chars)
    if not chunks:
        raise PageEmbeddingError(INVALID_REQUEST, "Wiki page has no indexable content")
    if len(chunks) > MAX_PAGE_CHUNKS:
        raise PageEmbeddingError(
            INVALID_REQUEST,
            "Wiki page produces %d chunks, exceeding the %d chunk limit; increase maxChunkChars or split the page"
            % (len(chunks), MAX_PAGE_CHUNKS),
        )

    try:
        rows = await asyncio.wait_for(_prepare_embedding_rows(title, chunks, config), PROVIDER_PHASE_TIMEOUT)
    except asyncio.TimeoutError:
        raise PageEmbeddingError(
            TIMEOUT, "Embedding provider timed out after %d seconds" % int(PROVIDER_PHASE_TIMEOUT)
        )
    validate_embedding_rows(rows)

    try:
        await asyncio.to_thread(
            vectorstore.vector_upsert_chunks_with_revision, project_path, page_id, rows, fingerprint
        )
    except CommandError as e:
        raise PageEmbeddingError(STORAGE, e.message)
    await asyncio.to_thread(_remove_legacy_vector_id_if_safe, project_path, page_id)
    return {
        "path": normalized_path, "pageId": page_id, "revision": revision,
        "chunks": len(chunks), "vectorsWritten": len(chunks), "status": "indexed",
    }
