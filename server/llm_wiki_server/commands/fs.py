"""Port of commands/fs.rs (file-system commands + preprocessing cache).

Text extraction lives in ..extractors. GUI/OS-opener behaviour does not exist
in fs.rs, so nothing is dropped here. Hook modules (file_history, file_sync) are
imported lazily and ignored when absent.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import os
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import extractors as ex
from ..common import require_absolute_path
from ..errors import CommandError
from ..extractors import (
    EBOOK_EXTS, IMAGE_EXTS, LEGACY_DOC_EXTS, MEDIA_EXTS, OFFICE_EXTS, os_err,
)
from ..registry import command

# Marker for office caches. Differs from the Rust "anydoc-0.1.6-v1" because the
# python extractors produce different text; stale Rust caches are rebuilt.
OFFICE_CACHE_FORMAT = "pyextract-0.1-v1"


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------

def _mark(path) -> None:
    try:
        from . import file_sync  # type: ignore
        file_sync.mark_app_write_path(str(path))
    except (ImportError, AttributeError):
        pass


def _record(path, kind: str, source: str) -> None:
    try:
        from . import file_history  # type: ignore
        file_history.record_file_version(str(path), kind, source)
    except (ImportError, AttributeError):
        pass


def _ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[1].lstrip(".").lower()


def _mkparent(path: str, msg: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            raise CommandError(msg.format(e=os_err(e)))


# ---------------------------------------------------------------------------
# preprocessing cache
# ---------------------------------------------------------------------------

def cache_path_for(original: str) -> str:
    parent = os.path.dirname(original) or "."
    return os.path.join(parent, ".cache", os.path.basename(original) + ".txt")


def cache_format_path_for(original: str) -> str:
    return cache_path_for(original) + ".parser"


def uses_office_cache(original: str) -> bool:
    return _ext(original) in OFFICE_EXTS


def _read_text_strict(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def read_cache(original: str) -> Optional[str]:
    """Return the cached extraction iff it is at least as new as the original."""
    original = str(original)
    cache_path = cache_path_for(original)
    if uses_office_cache(original):
        marker = _read_text_strict(cache_format_path_for(original))
        if marker is None or marker.strip() != OFFICE_CACHE_FORMAT:
            return None
    try:
        original_mtime = os.stat(original).st_mtime_ns
        cache_mtime = os.stat(cache_path).st_mtime_ns
    except OSError:
        return None
    if cache_mtime >= original_mtime:
        return _read_text_strict(cache_path)
    return None


def read_preprocessed_cache(original) -> Optional[str]:
    """Fresh preprocessing cache for Agent source retrieval (never parses or writes)."""
    return read_cache(str(original))


def write_cache(original: str, text: str) -> None:
    cache_path = cache_path_for(original)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    except OSError as e:
        raise CommandError(f"Failed to create preprocessing cache directory: {os_err(e)}")
    _mark(cache_path)
    format_path = cache_format_path_for(original)
    office = uses_office_cache(original)
    if office:
        # Drop the validity marker first: a crash then costs a rebuild, never stale text.
        try:
            os.remove(format_path)
        except OSError:
            pass
    try:
        with open(cache_path, "wb") as f:
            f.write(text.encode("utf-8"))
    except OSError as e:
        raise CommandError(f"Failed to write cache: {os_err(e)}")
    if office:
        _mark(format_path)
        try:
            with open(format_path, "wb") as f:
                f.write(OFFICE_CACHE_FORMAT.encode("utf-8"))
        except OSError as e:
            raise CommandError(f"Failed to write preprocessing cache format marker: {os_err(e)}")


# ---------------------------------------------------------------------------
# read / preprocess
# ---------------------------------------------------------------------------

@command()
def read_file(path: str, extract_images: Optional[bool] = None) -> str:
    ext = _ext(path)
    cached = read_cache(path)
    if cached is not None:
        return cached
    include_images = True if extract_images is None else bool(extract_images)

    if ext == "pdf":
        return ex.extract_pdf_text(path, include_images)
    if ext == "org":
        return ex.extract_org_text(path)
    if ext in OFFICE_EXTS:
        return ex.extract_office_text(path, ext)
    if ext in EBOOK_EXTS:
        return ex.extract_ebook_text(path, ext)
    name = os.path.basename(path)
    if ext in IMAGE_EXTS:
        size = _size_or_zero(path)
        return f"[Image: {name} ({size / 1024.0:.1f} KB)]"
    if ext in MEDIA_EXTS:
        size = _size_or_zero(path)
        return f"[Media: {name} ({size / 1048576.0:.1f} MB)]"
    if ext in LEGACY_DOC_EXTS:
        return f"[Document: {name} — text extraction not supported for .{ext} format]"
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        if not os.path.exists(path):
            raise CommandError(f"File does not exist: '{path}'")
        raise CommandError(
            f"Failed to read file '{path}' as text: {os_err(e)} (likely binary, locked, or non-UTF-8)"
        )


def _size_or_zero(path: str) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


@command()
def preprocess_file(path: str) -> str:
    ext = _ext(path)
    if ext == "pdf":
        text = ex.extract_pdf_text(path, False)
    elif ext == "org":
        text = ex.extract_org_text(path)
    elif ext in OFFICE_EXTS:
        text = ex.extract_office_text(path, ext)
    elif ext in EBOOK_EXTS:
        text = ex.extract_ebook_text(path, ext)
    else:
        return "no preprocessing needed"
    write_cache(path, text)
    return text


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------

@command()
def write_file(path: str, contents: str) -> None:
    require_absolute_path("write_file", path)
    _mkparent(path, "Failed to create parent dirs for '" + path + "': {e}")
    _mark(path)
    _record(path, "baseline", "before.ui.write_file")
    try:
        with open(path, "wb") as f:
            f.write(contents.encode("utf-8"))
    except (OSError, UnicodeEncodeError) as e:
        raise CommandError(f"Failed to write file '{path}': {os_err(e)}")
    _record(path, "human", "ui.write_file")
    _mark(path)
    return None


@command()
def write_file_base64(path: str, base64: str) -> None:  # noqa: A002 - name mirrors the Rust arg
    import base64 as b64mod

    require_absolute_path("write_file_base64", path)
    try:
        data = b64mod.b64decode(base64.encode("ascii"), validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as e:
        raise CommandError(f"Invalid base64 for '{path}': {e}")
    _mkparent(path, "Failed to create parent dirs for '" + path + "': {e}")
    _mark(path)
    try:
        with open(path, "wb") as f:
            f.write(data)
    except OSError as e:
        raise CommandError(f"Failed to write binary file '{path}': {os_err(e)}")
    _mark(path)
    return None


@command()
def write_file_atomic(path: str, contents: str) -> None:
    require_absolute_path("write_file_atomic", path)
    _mkparent(path, "Failed to create parent dirs for '" + path + "': {e}")
    file_name = os.path.basename(path) or "llm-wiki-file"
    tmp_path = os.path.join(os.path.dirname(path), f".{file_name}.{time.time_ns()}.tmp")
    _mark(tmp_path)
    _mark(path)
    _record(path, "baseline", "before.ui.write_file_atomic")
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents.encode("utf-8"))
    except (OSError, UnicodeEncodeError) as e:
        raise CommandError(f"Failed to write temp file '{tmp_path}': {os_err(e)}")
    try:
        os.replace(tmp_path, path)
    except OSError as e:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise CommandError(f"Failed to move temp file '{tmp_path}' to '{path}': {os_err(e)}")
    _record(path, "human", "ui.write_file_atomic")
    _mark(path)
    return None


def _canonicalize(path: str, what: str) -> str:
    """Rust fs::canonicalize: resolves symlinks and fails when the path is missing."""
    try:
        os.stat(path)
        return os.path.realpath(path)
    except OSError as e:
        raise CommandError(f"{what}: {os_err(e)}")


def _is_within(child: str, parent: str) -> bool:
    """Component-wise Path::starts_with."""
    c, p = Path(child).parts, Path(parent).parts
    return len(c) >= len(p) and c[:len(p)] == p


@command()
def apply_text_selection_edit(
    project_path: str,
    file_path: str,
    prefix: str,
    selected_text: str,
    suffix: str,
    replacement: str,
) -> str:
    project = _canonicalize(project_path, "Failed to resolve project path")
    file = _canonicalize(file_path, "Failed to resolve selected file")
    if not _is_within(file, project) or not os.path.isfile(file):
        raise CommandError("Selection edit target must be an existing file inside the project")
    try:
        with open(file, "rb") as f:
            current = f.read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise CommandError(f"Failed to read selection edit target: {os_err(e)}")
    expected = f"{prefix}{selected_text}{suffix}"
    if current != expected:
        raise CommandError(
            "The file changed after the selection was captured. "
            "Re-select the text before applying the Agent suggestion."
        )
    updated = f"{prefix}{replacement}{suffix}"
    _mark(file)
    _record(file, "baseline", "before.agent.selection_edit")
    try:
        with open(file, "wb") as f:
            f.write(updated.encode("utf-8"))
    except (OSError, UnicodeEncodeError) as e:
        raise CommandError(f"Failed to apply Agent selection edit: {os_err(e)}")
    _record(file, "agent", "agent.selection_edit")
    _mark(file)
    return updated


def _is_control(c: str) -> bool:
    return unicodedata.category(c) == "Cc"


def safe_missing_page_stem(title: str) -> str:
    chars = []
    for c in title:
        if _is_control(c) or c in '<>:"/\\|?*':
            chars.append("-")
        elif c.isspace():
            chars.append(" ")
        else:
            chars.append(c)
    stem = "".join(chars)
    while "--" in stem:
        stem = stem.replace("--", "-")
    stem = stem.strip(" .-")
    if not stem:
        stem = "untitled"
    device = stem.split(".")[0]
    device = "".join(c.upper() if "a" <= c <= "z" else c for c in device)
    if device in ("CON", "PRN", "AUX", "NUL") or (
        len(device) == 4
        and (device.startswith("COM") or device.startswith("LPT"))
        and "1" <= device[3] <= "9"
    ):
        stem = f"page-{stem}"
    return stem[:120]


@command()
def create_missing_wiki_page(project_path: str, title: str, content: Optional[str] = None) -> str:
    project = _canonicalize(project_path, "Failed to resolve project path")
    title = title.strip()
    if not title or len(title) > 200:
        raise CommandError("Missing-link page title must contain 1 to 200 characters")
    title = "".join(" " if _is_control(c) else c for c in title).strip()
    if content is not None and len(content.encode("utf-8")) > 2 * 1024 * 1024:
        raise CommandError("Missing-link page content exceeds the 2 MB limit")
    canonical_wiki = _canonicalize(os.path.join(project, "wiki"), "Failed to resolve wiki directory")
    if not _is_within(canonical_wiki, project):
        raise CommandError("Wiki directory escapes the project boundary")
    directory = os.path.join(canonical_wiki, "concepts")
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        raise CommandError(f"Failed to create wiki concepts directory: {os_err(e)}")
    canonical_directory = _canonicalize(directory, "Failed to resolve wiki concepts directory")
    if not _is_within(canonical_directory, project):
        raise CommandError("Wiki concepts directory escapes the project boundary")
    base = safe_missing_page_stem(title)
    target = os.path.join(canonical_directory, f"{base}.md")
    for suffix in range(2, 10000):
        if not os.path.lexists(target):
            break
        target = os.path.join(canonical_directory, f"{base}-{suffix}.md")
    if os.path.lexists(target):
        raise CommandError("Could not allocate a unique wiki page filename")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    escaped_title = title.replace('"', '\\"')
    default_content = (
        f'---\ntype: concept\ntitle: "{escaped_title}"\ncreated: {today}\nupdated: {today}\n'
        f"tags: []\nrelated: []\n---\n\n# {title}\n"
    )
    body = content if (content is not None and content.strip()) else default_content
    _mark(target)
    try:
        out = open(target, "xb")
    except OSError as e:
        raise CommandError(f"Failed to reserve missing-link page: {os_err(e)}")
    try:
        with out:
            out.write(body.encode("utf-8"))
    except (OSError, UnicodeEncodeError) as e:
        try:
            os.remove(target)
        except OSError:
            pass
        raise CommandError(f"Failed to create missing-link page: {os_err(e)}")
    _record(target, "agent", "wiki.missing_link.create")
    _mark(target)
    return os.path.relpath(target, project).replace("\\", "/")


# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------

def entry_is_visible(name: str, include_hidden: bool) -> bool:
    return include_hidden or not name.startswith(".")


def _valid_utf8_name(name: str) -> bool:
    try:
        name.encode("utf-8")
        return True
    except UnicodeEncodeError:  # surrogateescape'd (non UTF-8) filename
        return False


def build_tree(directory: str, depth: int, max_depth: int, include_hidden: bool) -> List[Dict[str, Any]]:
    if depth >= max_depth:
        return []
    try:
        names = os.listdir(directory)
    except OSError as e:
        raise CommandError(f"Failed to read directory '{directory}': {os_err(e)}")
    entries = []
    for name in names:
        if not _valid_utf8_name(name) or not entry_is_visible(name, include_hidden):
            continue
        p = os.path.join(directory, name)
        entries.append((name, p, os.path.isdir(p)))
    entries.sort(key=lambda e: (not e[2], e[0].encode("utf-8")))
    nodes: List[Dict[str, Any]] = []
    for name, p, is_dir in entries:
        node: Dict[str, Any] = {"name": name, "path": p.replace("\\", "/"), "is_dir": is_dir}
        if is_dir:
            kids = build_tree(p, depth + 1, max_depth, include_hidden)
            if kids:
                node["children"] = kids
        nodes.append(node)
    return nodes


@command()
def list_directory(
    path: str, include_hidden: Optional[bool] = None, max_depth: Optional[int] = None
) -> List[Dict[str, Any]]:
    include = bool(include_hidden) if include_hidden is not None else False
    depth = 30 if max_depth is None else max(1, min(30, int(max_depth)))
    if not os.path.exists(path):
        raise CommandError(f"Path does not exist: '{path}'")
    if not os.path.isdir(path):
        raise CommandError(f"Path is not a directory: '{path}'")
    return build_tree(path, 0, depth, include)


# ---------------------------------------------------------------------------
# copy / delete
# ---------------------------------------------------------------------------

@command()
def copy_file(source: str, destination: str) -> None:
    _mkparent(destination, "Failed to create parent dirs: {e}")
    _mark(destination)
    try:
        shutil.copyfile(source, destination)
        try:
            shutil.copymode(source, destination)
        except OSError:
            pass
    except OSError as e:
        raise CommandError(f"Failed to copy '{source}' to '{destination}': {os_err(e)}")
    _mark(destination)
    return None


@command()
def copy_directory(source: str, destination: str) -> List[str]:
    _mark(destination)
    if not os.path.isdir(source):
        raise CommandError(f"'{source}' is not a directory")
    copied: List[str] = []

    def rec(src: str, dest: str) -> None:
        try:
            os.makedirs(dest, exist_ok=True)
        except OSError as e:
            raise CommandError(f"Failed to create dir '{dest}': {os_err(e)}")
        try:
            names = sorted(os.listdir(src))
        except OSError as e:
            raise CommandError(f"Failed to read dir '{src}': {os_err(e)}")
        for name in names:
            if name.startswith("."):
                continue
            p = os.path.join(src, name)
            dp = os.path.join(dest, name)
            if os.path.isdir(p):
                rec(p, dp)
            else:
                try:
                    shutil.copyfile(p, dp)
                    try:
                        shutil.copymode(p, dp)
                    except OSError:
                        pass
                except OSError as e:
                    raise CommandError(f"Failed to copy '{p}': {os_err(e)}")
                _mark(dp)
                copied.append(dp.replace("\\", "/"))

    rec(source, destination)
    return copied


def _remove_path_with_retry(path: str, is_dir: bool) -> None:
    # The Rust retry loop only retries Windows sharing violations; on other
    # platforms it is a single attempt. Keep the retry for Windows.
    last: Optional[OSError] = None
    for attempt in range(4):
        try:
            if is_dir:
                if os.path.islink(path):
                    os.unlink(path)
                else:
                    shutil.rmtree(path)
            else:
                os.remove(path)
            return
        except OSError as e:
            transient = os.name == "nt" and (
                getattr(e, "winerror", None) in (32, 33) or isinstance(e, PermissionError)
            )
            if attempt < 3 and transient:
                last = e
                time.sleep(0.25 * (1 << attempt))
                continue
            raise
    if last:
        raise last


@command()
def delete_file(path: str) -> None:
    _mark(path)
    if os.path.isdir(path):
        try:
            _remove_path_with_retry(path, True)
        except OSError as e:
            raise CommandError(f"Failed to delete directory '{path}': {os_err(e)}")
    else:
        try:
            _remove_path_with_retry(path, False)
        except OSError as e:
            raise CommandError(f"Failed to delete file '{path}': {os_err(e)}")
    _mark(path)
    return None


# ---------------------------------------------------------------------------
# related pages
# ---------------------------------------------------------------------------

@command()
def find_related_wiki_pages(project_path: str, source_name: str) -> List[str]:
    wiki_dir = os.path.join(project_path, "wiki")
    if not os.path.isdir(wiki_dir):
        return []
    results: List[str] = []
    _collect_related_pages(wiki_dir, source_name, results)
    return results


def _rust_file_name(p: str) -> Optional[str]:
    stripped = p.rstrip("/")
    if not stripped:
        return None
    name = stripped.rsplit("/", 1)[-1]
    if name in ("..",):
        return None
    if name == ".":
        # "a/." normalises away the trailing "."
        return _rust_file_name(stripped[:-1]) if "/" in stripped else None
    return name


def _collect_related_pages(directory: str, source_name: str, results: List[str]) -> None:
    try:
        names = sorted(os.listdir(directory))
    except OSError as e:
        raise CommandError(os_err(e))

    file_name = _rust_file_name(source_name) or source_name
    file_name_lower = file_name.lower()
    file_stem = ".".join(file_name.split(".")[:-1])
    file_stem_lower = file_name_lower if not file_stem else file_stem.lower()

    for name in names:
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            _collect_related_pages(path, source_name, results)
        elif os.path.splitext(name)[1] == ".md" and name != ".md":
            if name in ("index.md", "log.md", "overview.md"):
                continue
            content = _read_text_strict(path)
            if content is None:
                continue
            content_lower = content.lower()
            sources_match = (
                f'"{file_name_lower}"' in content_lower or f"'{file_name_lower}'" in content_lower
            )
            is_in_sources_dir = "sources" in Path(path).parts
            is_source_summary = is_in_sources_dir and name.lower().startswith(file_stem_lower)

            frontmatter_match = False
            if content.startswith("---\n"):
                # Rust slices bytes; operate on bytes to keep offsets identical.
                raw = content.encode("utf-8")
                end = raw[4:].find(b"\n---")
                if end >= 0:
                    frontmatter = raw[4:4 + end].decode("utf-8", "replace").lower()
                    in_sources_block = False
                    for line in frontmatter.split("\n"):
                        if line.startswith("sources:"):
                            if file_name_lower in line:
                                frontmatter_match = True
                                break
                            in_sources_block = True
                            continue
                        if in_sources_block:
                            if line == "" or line.startswith(" ") or line.startswith("\t"):
                                if file_name_lower in line:
                                    frontmatter_match = True
                                    break
                            else:
                                in_sources_block = False
            if sources_match or is_source_summary or frontmatter_match:
                results.append(path.replace("\\", "/"))


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------

@command()
def create_directory(path: str) -> None:
    require_absolute_path("create_directory", path)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise CommandError(f"Failed to create directory '{path}': {os_err(e)}")
    return None


_MIME = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
    "webp": "image/webp", "bmp": "image/bmp", "tiff": "image/tiff", "tif": "image/tiff",
    "svg": "image/svg+xml", "pdf": "application/pdf",
}


@command()
def read_file_as_base64(path: str) -> Dict[str, str]:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise CommandError(f"Failed to read '{path}': {os_err(e)}")
    return {
        "base64": base64.b64encode(data).decode("ascii"),
        "mimeType": _MIME.get(_ext(path), "application/octet-stream"),
    }


@command()
def file_exists(path: str) -> bool:
    return os.path.exists(path)


def _stat(path: str):
    try:
        return os.stat(path)
    except OSError as e:
        raise CommandError(f"Failed to get metadata for '{path}': {os_err(e)}")


@command()
def get_file_modified_time(path: str) -> int:
    st = _stat(path)
    ms = st.st_mtime_ns // 1_000_000
    if st.st_mtime_ns < 0:
        raise CommandError(f"Time error for '{path}': second time provided was later than self")
    return int(ms)


@command()
def get_file_size(path: str) -> int:
    return int(_stat(path).st_size)


@command()
def get_file_md5(path: str) -> str:
    try:
        f = open(path, "rb")
    except OSError as e:
        raise CommandError(f"Failed to open file '{path}': {os_err(e)}")
    h = hashlib.md5()
    with f:
        try:
            while True:
                buf = f.read(64 * 1024)
                if not buf:
                    break
                h.update(buf)
        except OSError as e:
            raise CommandError(f"Failed to read file '{path}': {os_err(e)}")
    return h.hexdigest()
