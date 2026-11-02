"""Port of commands/extract_images.rs (image extraction from PDF / PPTX / DOCX).

pdfium -> PyMuPDF (`pymupdf` / `fitz`), PNG re-encode -> Pillow. Both are
imported lazily. Output JSON shapes (camelCase keys) match the Rust serde output.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..errors import CommandError
from ..registry import command

MIN_FULL_PAGE_SKIP_RATIO = 0.5
MAX_FULL_PAGE_SKIP_RATIO = 1.0

# The Rust version serialises PDFium access with a global lock; keep the same
# discipline (MuPDF is also not thread safe across shared state).
_PDF_LOCK = threading.RLock()


class ExtractOptions:
    def __init__(
        self,
        min_width: int = 100,
        min_height: int = 100,
        max_images: int = 500,
        skip_full_page_images: bool = True,
        full_page_coverage_ratio: float = 0.90,
        full_page_min_text_chars: int = 80,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.max_images = max_images
        self.skip_full_page_images = skip_full_page_images
        self.full_page_coverage_ratio = full_page_coverage_ratio
        self.full_page_min_text_chars = full_page_min_text_chars


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------- pure helpers

def is_full_page_image_bounds(
    object_width: float, object_height: float, page_width: float, page_height: float, coverage_ratio: float
) -> bool:
    if object_width <= 0.0 or object_height <= 0.0 or page_width <= 0.0 or page_height <= 0.0:
        return False
    ratio = min(max(coverage_ratio, MIN_FULL_PAGE_SKIP_RATIO), MAX_FULL_PAGE_SKIP_RATIO)
    return object_width / page_width >= ratio and object_height / page_height >= ratio


def has_meaningful_page_text(text: str, min_chars: int) -> bool:
    return sum(1 for c in text if not c.isspace()) >= min_chars


def should_skip_full_page_image_decision(
    skip_enabled: bool, page_has_meaningful_text: bool, image_is_full_page: bool
) -> bool:
    return skip_enabled and page_has_meaningful_text and image_is_full_page


def is_media_path(name: str) -> bool:
    lower = name.lower()
    return lower.startswith(("ppt/media/", "word/media/", "xl/media/"))


def guess_mime_from_name(name: str) -> Optional[str]:
    base = name.rsplit("/", 1)[-1]
    if "." not in base:
        return None
    ext = base.rsplit(".", 1)[-1].lower()
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }.get(ext)


def ext_for_mime(mime: str) -> str:
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/bmp": "bmp",
    }.get(mime, "bin")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------- lazy deps

def _import_pdf():
    try:
        import pymupdf as fitz  # type: ignore
        return fitz
    except ImportError:
        pass
    try:
        import fitz  # type: ignore
        return fitz
    except ImportError:
        raise CommandError("PDF image extraction requires the 'pymupdf' package (pip install pymupdf)")


def _import_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        raise CommandError("Image extraction requires the 'pillow' package (pip install pillow)")


# ---------------------------------------------------------------- PDF

def _open_pdf(fitz, path: str, password_msg: bool = False):
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise CommandError(f"Failed to open PDF '{path}': {e}")
    if getattr(doc, "needs_pass", False):
        if password_msg:
            doc.close()
            raise CommandError(f"PDF is password-protected and cannot be read: '{path}'")
        doc.close()
        raise CommandError(f"Failed to open PDF '{path}': password required")
    return doc


def _page_text(page) -> str:
    return page.get_text("text") or ""


def _page_image_objects(page) -> List[Dict[str, Any]]:
    """Image objects in page order: dicts with xref and bbox (like pdfium image page objects)."""
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        infos = []
    return infos


def _pixmap_to_png(fitz, Image, doc, xref: int) -> Tuple[bytes, int, int]:
    """Decode an embedded image into PNG bytes (raises on failure)."""
    pix = fitz.Pixmap(doc, xref)
    try:
        # Apply soft mask when present.
        try:
            smask = doc.xref_get_key(xref, "SMask")
            if smask and smask[0] == "xref":
                mask_xref = int(smask[1].split()[0])
                mask = fitz.Pixmap(doc, mask_xref)
                if not pix.alpha and (mask.width, mask.height) == (pix.width, pix.height):
                    if pix.colorspace is None or pix.colorspace.n != 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix = fitz.Pixmap(pix, mask)
        except Exception:
            pass
        if pix.colorspace is not None and pix.colorspace.n not in (1, 3):
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if pix.colorspace is None:
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        elif pix.colorspace.n == 1:
            img = Image.frombytes("LA" if pix.alpha else "L", (pix.width, pix.height), pix.samples)
        else:
            img = Image.frombytes("RGBA" if pix.alpha else "RGB", (pix.width, pix.height), pix.samples)
        width, height = img.size
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), width, height
    finally:
        pix = None


def _iter_pdf_images(doc, path: str, options: ExtractOptions, tag: str, counters: Optional[Dict[str, int]] = None):
    """Yield (page_number, png_bytes, width, height) for each passing image, honouring max_images."""
    fitz = _import_pdf()
    Image = _import_pil()
    produced = 0
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        has_text: Optional[bool] = None
        pw, ph = page.rect.width, page.rect.height
        for obj in _page_image_objects(page):
            if counters is not None:
                counters["total_objects"] += 1
                counters["image_objects"] += 1
            xref = obj.get("xref", 0)
            bbox = obj.get("bbox") or (0, 0, 0, 0)
            if options.skip_full_page_images:
                if has_text is None:
                    try:
                        has_text = has_meaningful_page_text(_page_text(page), options.full_page_min_text_chars)
                    except Exception:
                        has_text = False
                full = is_full_page_image_bounds(
                    bbox[2] - bbox[0], bbox[3] - bbox[1], pw, ph, options.full_page_coverage_ratio
                )
                if should_skip_full_page_image_decision(True, has_text, full):
                    if counters is not None:
                        counters["full_page"] += 1
                    _log(f"[{tag}] page {page_idx + 1} image covers most of page — skipped")
                    continue
            if not xref:
                if counters is not None:
                    counters["decode_err"] += 1
                continue
            try:
                png, width, height = _pixmap_to_png(fitz, Image, doc, xref)
            except Exception as e:
                if counters is not None:
                    counters["decode_err"] += 1
                _log(f"[{tag}] page {page_idx + 1} image read failed: {e}")
                continue
            if width < options.min_width or height < options.min_height:
                if counters is not None:
                    counters["too_small"] += 1
                continue
            yield page_idx + 1, png, width, height
            produced += 1
            if produced >= options.max_images:
                _log(f"[{tag}] reached max_images={options.max_images} cap; remaining images skipped")
                return


def extract_pdf_images(path: str, options: ExtractOptions) -> List[Dict[str, Any]]:
    with _PDF_LOCK:
        fitz = _import_pdf()
        doc = _open_pdf(fitz, path)
        try:
            out: List[Dict[str, Any]] = []
            for idx, (page, png, w, h) in enumerate(_iter_pdf_images(doc, path, options, "extract_pdf_images"), 1):
                out.append({
                    "index": idx,
                    "mimeType": "image/png",
                    "page": page,
                    "width": w,
                    "height": h,
                    "dataBase64": _b64(png),
                    "sha256": sha256_hex(png),
                })
            return out
        finally:
            doc.close()


def save_one_image(data: bytes, dest_dir: Path, rel_to: Path, file_name: str) -> Tuple[str, str]:
    dest_dir = Path(dest_dir)
    if not dest_dir.exists():
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise CommandError(f"create_dir_all '{dest_dir}': {e}")
    ab = dest_dir / file_name
    try:
        ab.write_bytes(data)
    except OSError as e:
        raise CommandError(f"write '{ab}': {e}")
    try:
        rel = str(ab.relative_to(rel_to)).replace("\\", "/")
    except ValueError:
        rel = file_name
    return rel, str(ab)


def extract_and_save_pdf_images(path: str, dest_dir: Path, rel_to: Path, options: ExtractOptions) -> List[Dict[str, Any]]:
    with _PDF_LOCK:
        fitz = _import_pdf()
        doc = _open_pdf(fitz, path)
        try:
            _log(
                f"[extract_and_save_pdf_images] '{path}': {doc.page_count} page(s), "
                f"filter=({options.min_width}x{options.min_height}) min, max={options.max_images}"
            )
            counters = {"total_objects": 0, "image_objects": 0, "too_small": 0, "full_page": 0, "decode_err": 0}
            out: List[Dict[str, Any]] = []
            for idx, (page, png, w, h) in enumerate(
                _iter_pdf_images(doc, path, options, "extract_and_save_pdf_images", counters), 1
            ):
                rel_path, abs_path = save_one_image(png, dest_dir, rel_to, f"img-{idx}.png")
                out.append({
                    "index": idx,
                    "mimeType": "image/png",
                    "page": page,
                    "width": w,
                    "height": h,
                    "relPath": rel_path,
                    "absPath": abs_path,
                    "sha256": sha256_hex(png),
                })
            _log(f"[extract_and_save_pdf_images] '{path}' DONE — saved={len(out)}, {counters}")
            return out
        finally:
            doc.close()


def extract_pdf_markdown(path: str, media_dest_dir: Optional[Path], media_url_prefix: str, options: ExtractOptions) -> str:
    """Text + (optionally) saved images as markdown with `## Page N` headers (used by fs.py)."""
    with _PDF_LOCK:
        fitz = _import_pdf()
        Image = _import_pil() if media_dest_dir is not None else None
        doc = _open_pdf(fitz, path, password_msg=True)
        try:
            out = ""
            idx = 0
            total_saved = 0
            prefix = media_url_prefix.rstrip("/")
            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                page_num = page_idx + 1
                if out:
                    out += "\n\n"
                out += f"## Page {page_num}\n\n"
                try:
                    text = _page_text(page)
                except Exception as e:
                    raise CommandError(f"Page {page_num} text extraction failed in '{path}': {e}")
                has_text = has_meaningful_page_text(text, options.full_page_min_text_chars)
                out += text
                out += "\n"
                if media_dest_dir is None:
                    continue
                md: List[str] = []
                pw, ph = page.rect.width, page.rect.height
                for obj in _page_image_objects(page):
                    bbox = obj.get("bbox") or (0, 0, 0, 0)
                    full = is_full_page_image_bounds(
                        bbox[2] - bbox[0], bbox[3] - bbox[1], pw, ph, options.full_page_coverage_ratio
                    )
                    if should_skip_full_page_image_decision(options.skip_full_page_images, has_text, full):
                        continue
                    xref = obj.get("xref", 0)
                    if not xref:
                        continue
                    try:
                        png, width, height = _pixmap_to_png(fitz, Image, doc, xref)
                    except Exception as e:
                        _log(f"[extract_pdf_markdown] page {page_num} image read failed: {e}")
                        continue
                    if width < options.min_width or height < options.min_height:
                        continue
                    idx += 1
                    name = f"img-{idx}.png"
                    try:
                        save_one_image(png, Path(media_dest_dir), Path(media_dest_dir), name)
                    except CommandError as e:
                        _log(f"[extract_pdf_markdown] page {page_num} save failed: {e.message}")
                        continue
                    total_saved += 1
                    md.append(f"![]({prefix}/{name})")
                    if total_saved >= options.max_images:
                        _log(f"[extract_pdf_markdown] reached max_images={options.max_images} cap; skipped rest")
                        break
                if md:
                    out += "\n"
                    for m in md:
                        out += m + "\n"
                if total_saved >= options.max_images:
                    break
            return out
        finally:
            doc.close()


# ---------------------------------------------------------------- Office (zip)

def _build_pptx_media_slide_map(zf: zipfile.ZipFile) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {}
    pre, suf = "ppt/slides/_rels/slide", ".xml.rels"
    rels = [n for n in zf.namelist() if n.startswith(pre) and n.endswith(suf)]
    for rp in rels:
        mid = rp[len(pre):-len(suf)]
        slide_num: Optional[int] = int(mid) if mid.isascii() and mid.isdigit() else None
        try:
            xml = zf.read(rp).decode("utf-8")
        except Exception:
            continue
        pos = 0
        key = 'Target="'
        while True:
            i = xml.find(key, pos)
            if i < 0:
                break
            start = i + len(key)
            end = xml.find('"', start)
            if end < 0:
                break
            target = xml[start:end]
            pos = end + 1
            if target.startswith("../"):
                canonical = "ppt/" + target[3:]
                if is_media_path(canonical):
                    out[canonical] = slide_num
    return out


def _open_zip(path: str) -> zipfile.ZipFile:
    try:
        f = open(path, "rb")
    except OSError as e:
        raise CommandError(f"Failed to open '{path}': {e}")
    try:
        return zipfile.ZipFile(f)
    except Exception as e:
        f.close()
        raise CommandError(f"Failed to read zip '{path}': {e}")


def _iter_office_images(path: str, options: ExtractOptions, tag: str):
    """Yield (idx, mime, page, width, height, bytes)."""
    Image = _import_pil()
    zf = _open_zip(path)
    try:
        names = zf.namelist()
        is_pptx = any(n == "ppt/presentation.xml" or n.startswith("ppt/slides/slide") for n in names)
        media_to_slide = _build_pptx_media_slide_map(zf) if is_pptx else {}
        idx = 0
        produced = 0
        for info in zf.infolist():
            if not is_media_path(info.filename):
                continue
            name = info.filename
            mime = guess_mime_from_name(name)
            if mime is None:
                continue
            try:
                data = zf.read(info)
            except Exception as e:
                _log(f"[{tag}] read '{name}' failed: {e}")
                continue
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
                width, height = img.size
            except Exception as e:
                _log(f"[{tag}] decode '{name}' failed: {e}")
                continue
            if width < options.min_width or height < options.min_height:
                continue
            idx += 1
            yield idx, mime, media_to_slide.get(name), width, height, data
            produced += 1
            if produced >= options.max_images:
                _log(f"[{tag}] reached max_images={options.max_images} cap; remaining skipped")
                return
    finally:
        zf.close()


def extract_office_images(path: str, options: ExtractOptions) -> List[Dict[str, Any]]:
    return [
        {
            "index": idx,
            "mimeType": mime,
            "page": page,
            "width": w,
            "height": h,
            "dataBase64": _b64(data),
            "sha256": sha256_hex(data),
        }
        for idx, mime, page, w, h, data in _iter_office_images(path, options, "extract_office_images")
    ]


def extract_and_save_office_images(path: str, dest_dir: Path, rel_to: Path, options: ExtractOptions) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, mime, page, w, h, data in _iter_office_images(path, options, "extract_and_save_office_images"):
        rel_path, abs_path = save_one_image(data, dest_dir, rel_to, f"img-{idx}.{ext_for_mime(mime)}")
        out.append({
            "index": idx,
            "mimeType": mime,
            "page": page,
            "width": w,
            "height": h,
            "relPath": rel_path,
            "absPath": abs_path,
            "sha256": sha256_hex(data),
        })
    return out


# ---------------------------------------------------------------- commands

@command()
def extract_pdf_images_cmd(path: str):
    return extract_pdf_images(path, ExtractOptions())


@command()
def extract_office_images_cmd(path: str):
    return extract_office_images(path, ExtractOptions())


@command()
def extract_and_save_pdf_images_cmd(source_path: str, dest_dir: str, rel_to: str):
    return extract_and_save_pdf_images(source_path, Path(dest_dir), Path(rel_to), ExtractOptions())


@command()
def extract_and_save_office_images_cmd(source_path: str, dest_dir: str, rel_to: str):
    return extract_and_save_office_images(source_path, Path(dest_dir), Path(rel_to), ExtractOptions())
