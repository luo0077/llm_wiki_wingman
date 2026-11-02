"""Port of src/lib/extract-source-images.ts: image extraction orchestration for ingest.

Dispatches to the Python ports of ``extract_and_save_pdf_images_cmd`` /
``extract_and_save_office_images_cmd`` (``llm_wiki_server.commands.extract_images``) and the
fs commands (``llm_wiki_server.commands.fs``).  Blocking work runs in a worker thread so the API is
``async`` like the TS one.

Public API::

    SavedImage                      # Record dataclass: index, mimeType, page, width, height, relPath, absPath, sha256
    find_local_markdown_image_refs(markdown) -> List[str]
    async extract_and_save_source_images(project_path, source_path, slug_override=None) -> List[SavedImage]
    async extract_and_save_markdown_images(project_path, source_path, markdown, slug_override=None) -> List[SavedImage]
    build_image_markdown_section(images, captions_by_sha=None) -> str
"""
from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ._js import Record
from .llm_jscompat import JS_DOT, JS_WS_CHARS, decode_uri_component, js_number_str, js_trim
from .path_utils import get_file_name, normalize_path

log = logging.getLogger("llm_wiki_server.ingest.extract_source_images")

SUPPORTED_PDF_EXTS = ("pdf",)
SUPPORTED_OFFICE_EXTS = ("pptx", "docx")
SUPPORTED_SOURCE_IMAGE_EXTS = frozenset(("png", "jpg", "jpeg"))
MARKDOWN_IMAGE_EXTS = frozenset(("png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "svg"))


@dataclass
class SavedImage(Record):
    """Mirrors ``commands::extract_images::SavedImage`` (camelCase field names)."""
    index: int
    mimeType: str
    page: Optional[int]
    width: int
    height: int
    relPath: str    # relative to the wiki/ root, e.g. ``media/rope-paper/img-1.png``
    absPath: str
    sha256: str


# ----------------------------------------------------------------- fs boundary (patchable in tests)

async def _copy_file(source: str, destination: str) -> None:
    from ..commands import fs
    await asyncio.to_thread(fs.copy_file, source, destination)


async def _create_directory(path: str) -> None:
    from ..commands import fs
    await asyncio.to_thread(fs.create_directory, path)


async def _file_exists(path: str) -> bool:
    from ..commands import fs
    return await asyncio.to_thread(fs.file_exists, path)


async def _read_file_as_base64(path: str) -> Dict[str, str]:
    from ..commands import fs
    return await asyncio.to_thread(fs.read_file_as_base64, path)


async def _invoke_extract(is_pdf: bool, source_path: str, dest_dir: str, rel_to: str) -> List[Any]:
    from ..commands import extract_images
    fn = (extract_images.extract_and_save_pdf_images_cmd if is_pdf
          else extract_images.extract_and_save_office_images_cmd)
    return await asyncio.to_thread(fn, source_path, dest_dir, rel_to)


# ----------------------------------------------------------------- helpers

def _dirname(path: str) -> str:
    n = normalize_path(path)
    idx = n.rfind("/")
    return n[:idx] if idx >= 0 else ""


def _ext_of(name: str) -> str:
    return name.split(".")[-1].lower()


def _is_remote_or_data_image_ref(raw: str) -> bool:
    return re.match(r"(https?:|data:|blob:|file:|tauri:)", raw, re.IGNORECASE | re.ASCII) is not None


def _clean_markdown_image_ref(raw: str) -> str:
    stripped = re.sub(r"^<(" + JS_DOT + r"+)>\Z", r"\1", js_trim(raw))
    try:
        return decode_uri_component(stripped)
    except ValueError:
        return stripped


def _image_mime_type(path: str) -> str:
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif",
        "webp": "image/webp", "bmp": "image/bmp", "svg": "image/svg+xml",
        "tif": "image/tiff", "tiff": "image/tiff",
    }.get(_ext_of(get_file_name(path)), "application/octet-stream")


def _unique_dest_name(index: int, source_path: str) -> str:
    name = re.sub(r"[<>:\"|?*\x00-\x1f]", "_", get_file_name(source_path))
    return f"{index:03d}-{name}"


async def _sha256_of_file(path: str) -> str:
    data = await _read_file_as_base64(path)
    return hashlib.sha256(base64.b64decode(data["base64"])).hexdigest()


_OBSIDIAN_RE = re.compile(r"!\[\[([^\]]+)\]\]")
_MD_IMAGE_RE = re.compile(
    r"!\[[^\]]*]\(([^)" + JS_WS_CHARS + r"]+)(?:[" + JS_WS_CHARS + r"]+[\"'][^\"']*[\"'])?\)")


def find_local_markdown_image_refs(markdown: str) -> List[str]:
    refs: List[str] = []
    seen = set()

    def add(raw: str) -> None:
        ref = _clean_markdown_image_ref(raw.split("#")[0].split("|")[0])
        if not ref or _is_remote_or_data_image_ref(ref):
            return
        if _ext_of(get_file_name(ref)) not in MARKDOWN_IMAGE_EXTS:
            return
        key = normalize_path(ref)
        if key in seen:
            return
        seen.add(key)
        refs.append(ref)

    for m in _OBSIDIAN_RE.finditer(markdown):
        add(m.group(1) or "")
    for m in _MD_IMAGE_RE.finditer(markdown):
        add(m.group(1) or "")
    return refs


def _to_saved_image(obj: Mapping[str, Any]) -> SavedImage:
    return SavedImage(
        index=obj["index"],
        mimeType=obj.get("mimeType") if isinstance(obj.get("mimeType"), str) else "",
        page=obj.get("page"),
        width=obj.get("width") or 0,
        height=obj.get("height") or 0,
        relPath=obj["relPath"],
        absPath=obj["absPath"],
        sha256=obj.get("sha256") if isinstance(obj.get("sha256"), str) else "",
    )


# ----------------------------------------------------------------- extraction

async def extract_and_save_source_images(
    project_path: str, source_path: str, slug_override: Optional[str] = None,
) -> List[SavedImage]:
    """Extract every embedded image of ``source_path`` into ``<project>/wiki/media/<slug>/``.

    Returns ``[]`` for unsupported types / no images / any extraction failure (never raises).
    """
    pp = normalize_path(project_path)
    sp = normalize_path(source_path)
    file_name = get_file_name(sp)
    ext = _ext_of(file_name)

    is_pdf = ext in SUPPORTED_PDF_EXTS
    is_office = ext in SUPPORTED_OFFICE_EXTS
    is_standalone = ext in SUPPORTED_SOURCE_IMAGE_EXTS
    if not is_pdf and not is_office and not is_standalone:
        return []

    slug = slug_override if slug_override is not None else re.sub(r"\.[^.]+\Z", "", file_name)
    dest_dir = f"{pp}/wiki/media/{slug}"
    rel_to = f"{pp}/wiki"

    try:
        if is_standalone:
            await _create_directory(dest_dir)
            dest_name = _unique_dest_name(1, sp)
            dest = f"{dest_dir}/{dest_name}"
            await _copy_file(sp, dest)
            return [SavedImage(
                index=1, mimeType=_image_mime_type(dest), page=None, width=0, height=0,
                relPath=f"media/{slug}/{dest_name}", absPath=dest, sha256=await _sha256_of_file(dest))]
        images = await _invoke_extract(is_pdf, sp, dest_dir, rel_to)
        out: List[SavedImage] = []
        for it in images:
            if (isinstance(it, Mapping) and isinstance(it.get("index"), (int, float))
                    and not isinstance(it.get("index"), bool)
                    and isinstance(it.get("relPath"), str) and isinstance(it.get("absPath"), str)):
                out.append(_to_saved_image(it))
        return out
    except Exception as err:
        log.warning('[ingest:images] extraction failed for "%s": %s', file_name, err)
        return []


async def extract_and_save_markdown_images(
    project_path: str, source_path: str, markdown: str, slug_override: Optional[str] = None,
) -> List[SavedImage]:
    """Import local images referenced by a markdown source into ``wiki/media/<slug>/``."""
    refs = find_local_markdown_image_refs(markdown)
    if not refs:
        return []

    pp = normalize_path(project_path)
    sp = normalize_path(source_path)
    source_dir = _dirname(sp)
    slug = slug_override if slug_override is not None else re.sub(r"\.[^.]+\Z", "", get_file_name(sp))
    dest_dir = f"{pp}/wiki/media/{slug}"
    images: List[SavedImage] = []

    try:
        await _create_directory(dest_dir)
    except Exception as err:
        log.warning("[ingest:images] failed to create markdown image directory: %s", err)
        return []

    for ref in refs:
        absolute = ref.startswith("/") or re.match(r"[a-zA-Z]:", ref) or ref.startswith("\\\\")
        abs_path = normalize_path(ref if absolute else f"{source_dir}/{ref}")
        try:
            if not await _file_exists(abs_path):
                continue
            dest_name = _unique_dest_name(len(images) + 1, abs_path)
            dest = f"{dest_dir}/{dest_name}"
            await _copy_file(abs_path, dest)
            sha256 = await _sha256_of_file(dest)
            images.append(SavedImage(
                index=len(images) + 1, mimeType=_image_mime_type(dest), page=None, width=0, height=0,
                relPath=f"media/{slug}/{dest_name}", absPath=dest, sha256=sha256))
        except Exception as err:
            log.warning('[ingest:images] markdown image import failed for "%s": %s', ref, err)
    return images


# ----------------------------------------------------------------- markdown section

def _page_order(a: str, b: str) -> int:
    if a == "Document":
        return 1
    if b == "Document":
        return -1

    def num(s: str) -> int:
        digits = re.sub(r"[^0-9]", "", s)
        return int(digits) if digits else 0

    return num(a) - num(b)


def build_image_markdown_section(
    images: List[SavedImage], captions_by_sha: Optional[Mapping[str, str]] = None,
) -> str:
    """Markdown ``## Embedded Images`` section (empty string when there are no images)."""
    if not images:
        return ""

    lines: List[str] = ["", "", "## Embedded Images", ""]
    def field(img: Any, key: str) -> Any:
        # SavedImage records and plain dicts (camelCase keys) are both accepted.
        if isinstance(img, Mapping):
            return img.get(key)
        return getattr(img, key, None)

    by_page: Dict[str, List[SavedImage]] = {}
    for img in images:
        page = field(img, "page")
        key = "Document" if page is None else f"Page {js_number_str(page)}"
        by_page.setdefault(key, []).append(img)

    ordered = sorted(by_page.keys(), key=functools.cmp_to_key(_page_order))

    def sanitize(s: str) -> str:
        return js_trim(re.sub(r"[\r\n]+", " ", s).replace("]", ")"))

    for key in ordered:
        lines.extend([f"### {key}", ""])
        for img in by_page.get(key, []):
            sha = field(img, "sha256")
            caption = captions_by_sha.get(sha) if captions_by_sha and sha is not None else None
            alt = sanitize(caption) if caption else ""
            lines.append(f"![{alt}]({field(img, 'relPath')})")
        lines.append("")
    return "\n".join(lines)
