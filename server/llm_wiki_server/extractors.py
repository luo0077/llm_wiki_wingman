"""Text extraction for binary/structured sources (port of the extraction half of
commands/fs.rs plus commands/ebook.rs).

pdfium / anydoc / calamine are replaced by pymupdf|pypdf, python-docx,
openpyxl|xlrd, striprtf and stdlib zip/XML parsers. Third-party libraries are
imported lazily and produce a clear CommandError when missing. Output follows
the Rust conventions (``## Page N``, markdown tables for sheets, ``## Slide N``).
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import io
import math
import os
import posixpath
import re
import struct
import unicodedata
import zipfile
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from .errors import CommandError

OFFICE_EXTS = [
    "doc", "docx", "docm", "ppt", "pps", "pot", "pptx", "pptm", "ppsx", "ppsm", "xls", "xlsx",
    "xlsm", "xlsb", "odt", "ods", "odp", "rtf",
]
IMAGE_EXTS = [
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "tiff", "tif", "avif", "heic", "heif", "svg",
]
MEDIA_EXTS = [
    "mp4", "webm", "mov", "avi", "mkv", "flv", "wmv", "m4v", "mp3", "wav", "ogg", "flac", "aac",
    "m4a", "wma",
]
EBOOK_EXTS = ["epub", "mobi"]
LEGACY_DOC_EXTS = ["pages", "numbers", "key"]

MAX_EBOOK_BYTES = 100 * 1024 * 1024
MAX_EPUB_ENTRIES = 10_000
MAX_EPUB_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_EPUB_COMPRESSION_RATIO = 200
MAX_EPUB_CHAPTER_BYTES = 16 * 1024 * 1024
MAX_EXTRACTED_TEXT_BYTES = 32 * 1024 * 1024
MAX_CHAPTERS = 10_000


def os_err(e: BaseException) -> str:
    """Render an OSError like Rust's io::Error Display: 'No such file or directory (os error 2)'."""
    if isinstance(e, OSError) and e.errno is not None:
        return f"{e.strerror or e} (os error {e.errno})"
    if isinstance(e, UnicodeDecodeError):
        return "stream did not contain valid UTF-8"
    return str(e)


# ---------------------------------------------------------------------------
# Org -> Markdown
# ---------------------------------------------------------------------------

def _ascii_upper(s: str) -> str:
    return "".join(c.upper() if "a" <= c <= "z" else c for c in s)


def _ascii_lower(s: str) -> str:
    return "".join(c.lower() if "A" <= c <= "Z" else c for c in s)


def _is_ascii_alnum(c: str) -> bool:
    return c.isascii() and c.isalnum()


def _parse_org_keyword(line: str) -> Optional[Tuple[str, str]]:
    if not line.startswith("#+"):
        return None
    rest = line[2:]
    if ":" not in rest:
        return None
    key, value = rest.split(":", 1)
    if not key or not all(_is_ascii_alnum(c) or c == "_" for c in key):
        return None
    return _ascii_upper(key), value.strip()


def _parse_org_heading(line: str) -> Optional[Tuple[int, str]]:
    stars = 0
    for c in line:
        if c == "*":
            stars += 1
        else:
            break
    if stars == 0 or len(line) <= stars or line[stars] != " ":
        return None
    return stars, line[stars + 1:].strip()


def _title_case_ascii(value: str) -> str:
    lower = _ascii_lower(value.replace("_", " "))
    if not lower:
        return ""
    return _ascii_upper(lower[0]) + lower[1:]


def convert_org_links(line: str) -> str:
    out: List[str] = []
    rest = line
    while True:
        start = rest.find("[[")
        if start < 0:
            break
        out.append(rest[:start])
        after = rest[start + 2:]
        end = after.find("]]")
        if end < 0:
            out.append(rest[start:])
            return "".join(out)
        inner = after[:end]
        if "][" in inner:
            target, description = inner.split("][", 1)
            out.append(f"[{description}]({target})")
        else:
            out.append(f"<{inner}>")
        rest = after[end + 2:]
    out.append(rest)
    return "".join(out)


def org_to_markdown(content: str) -> str:
    """Convert common Org syntax into Markdown-shaped text (source blocks verbatim in fences)."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # Rust str::lines() drops the final empty segment
    output: List[str] = []
    block_end: Optional[str] = None

    for line in lines:
        trimmed = line.strip()
        upper = _ascii_upper(trimmed)
        if block_end is not None:
            if upper == block_end:
                output.append("```")
                block_end = None
            else:
                output.append(line)
            continue

        if upper.startswith("#+BEGIN_SRC"):
            parts = trimmed[11:].split()
            language = parts[0] if parts else ""
            output.append(f"```{language}")
            block_end = "#+END_SRC"
            continue
        if upper == "#+BEGIN_EXAMPLE":
            output.append("```text")
            block_end = "#+END_EXAMPLE"
            continue
        if upper == "#+BEGIN_QUOTE":
            output.append("```text")
            block_end = "#+END_QUOTE"
            continue

        kw = _parse_org_keyword(trimmed)
        if kw is not None:
            key, value = kw
            if key == "TITLE":
                output.append(f"# {value}")
            elif key not in ("OPTIONS", "PROPERTY", "SETUPFILE"):
                output.append(f"**{_title_case_ascii(key)}:** {value}")
            continue

        heading = _parse_org_heading(line)
        if heading is not None:
            level, text = heading
            output.append(f"{'#' * min(level, 6)} {convert_org_links(text)}")
            continue

        if (
            trimmed.startswith("|")
            and trimmed.endswith("|")
            and "+" in trimmed
            and all(c in "|+-: " for c in trimmed)
        ):
            output.append(trimmed.replace("+", "|"))
            continue

        output.append(convert_org_links(line))
    if block_end is not None:
        output.append("```")
    return "\n".join(output)


def extract_org_text(path: str) -> str:
    try:
        with open(path, "rb") as f:
            content = f.read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise CommandError(f"Failed to read Org file '{path}': {os_err(e)}")
    return org_to_markdown(content)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _pdf_project_media_dir(path: str) -> Optional[Tuple[str, str]]:
    """If path is <root>/raw/sources/<file>.pdf return (media_dir, url_prefix)."""
    parent = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    if os.path.basename(parent) != "sources":
        return None
    raw_dir = os.path.dirname(parent)
    if os.path.basename(raw_dir) != "raw":
        return None
    root = os.path.dirname(raw_dir)
    if not stem or not root:
        return None
    media_dir = os.path.join(root, "wiki", "media", stem)
    return media_dir, media_dir.replace("\\", "/")


def extract_pdf_text(path: str, include_images: bool) -> str:
    """Extract a PDF as markdown (`## Page N` sections).

    Image extraction is owned by the extract_images module; when available and the
    PDF lives in <project>/raw/sources it is used (as in the Rust implementation).
    """
    if include_images:
        loc = _pdf_project_media_dir(path)
        if loc is not None:
            fn = None
            try:
                from .commands import extract_images as _ei  # type: ignore
                fn = getattr(_ei, "extract_pdf_markdown", None)
            except ImportError:
                try:
                    from . import extract_images as _ei  # type: ignore
                    fn = getattr(_ei, "extract_pdf_markdown", None)
                except ImportError:
                    fn = None
            if fn is not None:
                return fn(path, loc[0], loc[1], _ei.ExtractOptions())
    return pdf_text_only(path)


def pdf_text_only(path: str) -> str:
    pages: List[str]
    fitz = None
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:
        try:
            import fitz  # type: ignore
        except ImportError:
            fitz = None

    if fitz is not None:
        try:
            doc = fitz.open(path)
        except Exception as e:  # noqa: BLE001
            raise CommandError(f"Failed to open PDF '{path}': {e}")
        try:
            if getattr(doc, "needs_pass", False):
                raise CommandError(f"PDF is password-protected and cannot be read: '{path}'")
            pages = []
            for i in range(doc.page_count):
                try:
                    pages.append(doc.load_page(i).get_text("text"))
                except Exception as e:  # noqa: BLE001
                    raise CommandError(f"Page {i + 1} text extraction failed in '{path}': {e}")
        finally:
            doc.close()
    else:
        try:
            import pypdf  # type: ignore
        except ImportError:
            raise CommandError(
                "PDF extraction requires the 'pymupdf' or 'pypdf' package (pip install pymupdf)"
            )
        try:
            reader = pypdf.PdfReader(path)
            if reader.is_encrypted:
                try:
                    ok = reader.decrypt("")
                except Exception:  # noqa: BLE001
                    ok = 0
                if not ok:
                    raise CommandError(f"PDF is password-protected and cannot be read: '{path}'")
            page_list = list(reader.pages)
        except CommandError:
            raise
        except Exception as e:  # noqa: BLE001
            raise CommandError(f"Failed to open PDF '{path}': {e}")
        pages = []
        for i, page in enumerate(page_list):
            try:
                pages.append(page.extract_text() or "")
            except Exception as e:  # noqa: BLE001
                raise CommandError(f"Page {i + 1} text extraction failed in '{path}': {e}")

    out = ""
    for i, text in enumerate(pages):
        if out:
            out += "\n\n"
        out += f"## Page {i + 1}\n\n"
        out += text
        out += "\n"
    return out


# ---------------------------------------------------------------------------
# Office
# ---------------------------------------------------------------------------

def decode_xml_entities(text: str) -> str:
    return (
        text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&quot;", '"').replace("&apos;", "'").replace("&#10;", "\n").replace("&#13;", "")
    )


def _read_zip_text(z: zipfile.ZipFile, name: str) -> Optional[str]:
    try:
        return z.read(name).decode("utf-8")
    except (KeyError, UnicodeDecodeError, OSError, zipfile.BadZipFile):
        return None


def _open_zip(path: str) -> zipfile.ZipFile:
    try:
        with open(path, "rb"):
            pass
    except OSError as e:
        raise CommandError(f"Failed to open '{path}': {os_err(e)}")
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise CommandError(f"Failed to read ZIP archive '{path}': {e}")


def extract_office_text(path: str, ext: str) -> str:
    if ext in ("xlsx", "xlsm", "xls", "ods"):
        return extract_spreadsheet(path, ext)
    if ext == "rtf":
        return extract_rtf(path)
    if ext in ("docx", "docm"):
        return extract_docx(path)
    if ext in ("pptx", "pptm", "ppsx", "ppsm"):
        with _open_zip(path) as z:
            return extract_pptx_markdown(z)
    if ext in ("odt", "odp"):
        with _open_zip(path) as z:
            return extract_odf_text(z)
    raise CommandError(f"no compatibility parser is available for .{ext}")


def extract_rtf(path: str) -> str:
    try:
        from striprtf.striprtf import rtf_to_text  # type: ignore
    except ImportError:
        raise CommandError("RTF extraction requires the 'striprtf' package (pip install striprtf)")
    try:
        with open(path, "rb") as f:
            raw = f.read().decode("latin-1")
        text = rtf_to_text(raw, errors="ignore")
    except Exception as e:  # noqa: BLE001
        raise CommandError(f"Failed to parse RTF '{path}': {e}")
    return text if text.strip() else "[Could not extract text from this file]"


def extract_docx(path: str) -> str:
    try:
        import docx  # type: ignore  # python-docx
    except ImportError:
        docx = None
    if docx is not None:
        try:
            result = _docx_with_library(docx, path)
        except Exception as e:  # noqa: BLE001
            raise CommandError(f"Failed to parse DOCX '{path}': {e}")
        if result.strip():
            return result
    with _open_zip(path) as z:
        return extract_docx_markdown(z)


def _docx_with_library(docx, path: str) -> str:
    """Port of extract_docx_with_library (docx-rs) on top of python-docx."""
    document = docx.Document(path)
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    result: List[str] = []

    def run_text(r) -> str:
        return "".join(t.text or "" for t in r.findall(f"{W}t"))

    def flag(rpr, tag) -> bool:
        if rpr is None:
            return False
        return rpr.find(f"{W}{tag}") is not None

    for child in document.element.body:
        if child.tag == f"{W}p":
            ppr = child.find(f"{W}pPr")
            style_val = ""
            is_list = False
            if ppr is not None:
                ps = ppr.find(f"{W}pStyle")
                if ps is not None:
                    style_val = ps.get(f"{W}val") or ""
                is_list = ppr.find(f"{W}numPr") is not None
            is_heading = "Heading" in style_val or "heading" in style_val
            level = 1
            if is_heading:
                for ch in style_val:
                    if ch.isascii() and ch.isdigit():
                        level = int(ch)
                        break
            para = ""
            for r in child.findall(f"{W}r"):
                rpr = r.find(f"{W}rPr")
                bold, italic = flag(rpr, "b"), flag(rpr, "i")
                for t in r.findall(f"{W}t"):
                    s = t.text or ""
                    if bold and italic:
                        para += f"***{s}***"
                    elif bold:
                        para += f"**{s}**"
                    elif italic:
                        para += f"*{s}*"
                    else:
                        para += s
            text = para.strip()
            if not text:
                continue
            if is_heading:
                result.append(f"{'#' * level} {text}\n\n")
            elif is_list:
                result.append(f"- {text}\n")
            else:
                result.append(text + "\n\n")
        elif child.tag == f"{W}tbl":
            rows: List[List[str]] = []
            for tr in child.findall(f"{W}tr"):
                cells = []
                for tc in tr.findall(f"{W}tc"):
                    cell_text = ""
                    for p in tc.findall(f"{W}p"):
                        for r in p.findall(f"{W}r"):
                            cell_text += run_text(r)
                    cells.append(cell_text.strip().replace("|", "\\|"))
                rows.append(cells)
            if rows:
                max_cols = max(len(r) for r in rows)
                for i, row in enumerate(rows):
                    padded = row + [""] * (max_cols - len(row))
                    result.append("| " + " | ".join(padded) + " |\n")
                    if i == 0:
                        result.append("|" + " --- |" * max_cols + "\n")
                result.append("\n")
    return "".join(result)


def extract_docx_markdown(z: zipfile.ZipFile) -> str:
    """Stdlib fallback: port of extract_docx_markdown (raw XML scanner)."""
    xml = _read_zip_text(z, "word/document.xml")
    if xml is None:
        raise CommandError("No document.xml found")

    result: List[str] = []
    i, n = 0, len(xml)
    paragraph_text = ""
    is_heading = False
    heading_level = 1
    is_bold = False
    is_italic = False
    in_table = False
    table_row: List[str] = []
    table_cell_text = ""
    in_cell = False
    is_first_table_row = True
    in_list_item = False

    while i < n:
        if xml[i] != "<":
            i += 1
            continue
        i += 1
        closing = i < n and xml[i] == "/"
        if closing:
            i += 1
        start = i
        while i < n and xml[i] not in ">/ ":
            i += 1
        tag = xml[start:i]
        start = i
        while i < n and xml[i] != ">":
            i += 1
        content = xml[start:i]
        if i < n:
            i += 1

        if tag == "w:p" and not closing:
            paragraph_text = ""
            is_heading = False
            in_list_item = False
        elif tag == "w:p" and closing:
            text = paragraph_text.strip()
            if text:
                if in_table and in_cell:
                    table_cell_text = text
                elif is_heading:
                    result.append(f"{'#' * heading_level} {text}\n\n")
                elif in_list_item:
                    result.append(f"- {text}\n")
                else:
                    result.append(text + "\n\n")
            paragraph_text = ""
        elif tag == "w:pStyle" and not closing:
            if "Heading" in content or "heading" in content:
                is_heading = True
                pos = content.find("Heading")
                if pos >= 0:
                    after = content[pos + 7:pos + 8]
                    if after and after.isascii() and after.isdigit():
                        heading_level = int(after)
            if "ListParagraph" in content or "listParagraph" in content:
                in_list_item = True
        elif tag == "w:b" and not closing and 'w:val="0"' not in content and 'w:val="false"' not in content:
            is_bold = True
        elif tag == "w:i" and not closing and 'w:val="0"' not in content and 'w:val="false"' not in content:
            is_italic = True
        elif tag == "w:r" and closing:
            is_bold = False
            is_italic = False
        elif tag == "w:t" and not closing:
            end = xml.find("<", i)
            if end < 0:
                end = n
            decoded = decode_xml_entities(xml[i:end])
            i = end
            if is_bold and is_italic:
                paragraph_text += f"***{decoded}***"
            elif is_bold:
                paragraph_text += f"**{decoded}**"
            elif is_italic:
                paragraph_text += f"*{decoded}*"
            else:
                paragraph_text += decoded
        elif tag == "w:tbl" and not closing:
            in_table = True
            is_first_table_row = True
        elif tag == "w:tbl" and closing:
            in_table = False
            result.append("\n")
        elif tag == "w:tr" and not closing:
            table_row = []
        elif tag == "w:tr" and closing:
            if table_row:
                result.append("| " + " | ".join(table_row) + " |\n")
                if is_first_table_row:
                    result.append("|" + " --- |" * len(table_row) + "\n")
                    is_first_table_row = False
        elif tag == "w:tc" and not closing:
            in_cell = True
            table_cell_text = ""
        elif tag == "w:tc" and closing:
            table_row.append(table_cell_text.strip())
            in_cell = False
            table_cell_text = ""

    out = "".join(result)
    if not out.strip():
        return "[Could not extract structured text from DOCX]"
    return out


def extract_pptx_markdown(z: zipfile.ZipFile) -> str:
    names = [
        i.filename for i in z.infolist()
        if i.filename.startswith("ppt/slides/slide") and i.filename.endswith(".xml")
    ]

    def num(name: str) -> int:
        s = name[len("ppt/slides/slide"):]
        s = s[:-4] if s.endswith(".xml") else s
        try:
            return int(s)
        except ValueError:
            return 0

    names.sort(key=num)
    result: List[str] = []
    for idx, name in enumerate(names):
        xml = _read_zip_text(z, name)
        if xml is None:
            continue
        result.append(f"## Slide {idx + 1}\n\n")
        paragraphs: List[str] = []
        for para_part in xml.split("<a:p"):
            para_text = ""
            for t_part in para_part.split("<a:t"):
                close = t_part.find("</a:t>")
                if close >= 0:
                    gt = t_part.find(">")
                    if 0 <= gt < close:
                        para_text += decode_xml_entities(t_part[gt + 1:close])
            t = para_text.strip()
            if t:
                paragraphs.append(t)
        if paragraphs:
            result.append(f"**{paragraphs[0]}**\n\n")
            for p in paragraphs[1:]:
                result.append(f"- {p}\n")
        result.append("\n")
    out = "".join(result)
    if not out.strip():
        return "[Could not extract text from PPTX]"
    return out


def extract_odf_text(z: zipfile.ZipFile) -> str:
    xml = _read_zip_text(z, "content.xml")
    if xml is None:
        raise CommandError("No content.xml found")
    buf: List[str] = []
    in_tag = False
    for ch in xml:
        if ch == "<":
            in_tag = True
        elif ch == ">":
            in_tag = False
            buf.append(" ")
        elif not in_tag:
            buf.append(ch)
    cleaned = decode_xml_entities("".join(buf))
    lines = [ln.strip() for ln in cleaned.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return "[Could not extract text from this file]"
    return "\n\n".join(lines)


# ---- spreadsheets -------------------------------------------------------

_XLS_ERRORS = {
    "#DIV/0!": "Div0", "#N/A": "NA", "#NAME?": "Name", "#NULL!": "Null",
    "#NUM!": "Num", "#REF!": "Ref", "#VALUE!": "Value", "#GETTING_DATA": "GettingData",
}
_EXCEL_EPOCH_DATE = _dt.date(1899, 12, 30)


def _fmt_float(f: float) -> str:
    if math.isfinite(f) and f == float(int(f)):
        return str(int(f))
    return f"{f:.2f}"


def _fmt_time(t: _dt.time) -> str:
    base = f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"
    ms = t.microsecond // 1000
    return f"{base}.{ms:03d}" if ms else base


def _fmt_duration(td: _dt.timedelta) -> str:
    total_ms = int(round(td.total_seconds() * 1000))
    sign = "-" if total_ms < 0 else ""
    total_ms = abs(total_ms)
    h, m = total_ms // 3_600_000, (total_ms % 3_600_000) // 60_000
    s, ms = (total_ms % 60_000) // 1000, total_ms % 1000
    base = f"{sign}{h:02d}:{m:02d}:{s:02d}"
    return f"{base}.{ms:03d}" if ms else base


def _fmt_datetime(v: _dt.datetime) -> str:
    ms = v.microsecond // 1000
    if v.date() == _EXCEL_EPOCH_DATE:
        return _fmt_time(v.time())
    if v.hour == 0 and v.minute == 0 and v.second == 0 and ms == 0:
        return f"{v.year:04d}-{v.month:02d}-{v.day:02d}"
    base = f"{v.year:04d}-{v.month:02d}-{v.day:02d} {v.hour:02d}:{v.minute:02d}:{v.second:02d}"
    return f"{base}.{ms:03d}" if ms else base


def _fmt_cell(v) -> str:
    """Format a python cell value like the Rust calamine mapping. None -> ''."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _fmt_float(v)
    if isinstance(v, _dt.datetime):
        return _fmt_datetime(v)
    if isinstance(v, _dt.date):
        return f"{v.year:04d}-{v.month:02d}-{v.day:02d}"
    if isinstance(v, _dt.time):
        return _fmt_time(v)
    if isinstance(v, _dt.timedelta):
        return _fmt_duration(v)
    return str(v)


Sheet = Tuple[str, Dict[Tuple[int, int], str]]


def _read_xlsx(path: str) -> List[Sheet]:
    try:
        import openpyxl  # type: ignore
    except ImportError:
        raise CommandError("Spreadsheet extraction requires the 'openpyxl' package (pip install openpyxl)")
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as e:  # noqa: BLE001
        raise CommandError(f"Failed to open spreadsheet '{path}': {e}")
    sheets: List[Sheet] = []
    try:
        for ws in wb.worksheets:
            cells: Dict[Tuple[int, int], str] = {}
            try:
                for r, row in enumerate(ws.iter_rows()):
                    for c, cell in enumerate(row):
                        v = getattr(cell, "value", None)
                        if v is None or v == "":
                            continue
                        if getattr(cell, "data_type", "") == "e" and isinstance(v, str):
                            s = f"ERR:{_XLS_ERRORS.get(v, v)}"
                        else:
                            s = _fmt_cell(v)
                        cells[(r, c)] = s
            except Exception:  # noqa: BLE001 - unreadable sheet == skipped (calamine Err)
                cells = {}
            sheets.append((ws.title, cells))
    finally:
        wb.close()
    return sheets


def _read_xls(path: str) -> List[Sheet]:
    try:
        import xlrd  # type: ignore
    except ImportError:
        raise CommandError("Legacy .xls extraction requires the 'xlrd' package (pip install xlrd)")
    try:
        book = xlrd.open_workbook(path)
    except Exception as e:  # noqa: BLE001
        raise CommandError(f"Failed to open spreadsheet '{path}': {e}")
    sheets: List[Sheet] = []
    for sh in book.sheets():
        cells: Dict[Tuple[int, int], str] = {}
        for r in range(sh.nrows):
            for c in range(sh.row_len(r)):
                t, v = sh.cell_type(r, c), sh.cell_value(r, c)
                if t in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
                    continue
                if t == xlrd.XL_CELL_TEXT:
                    if v == "":
                        continue
                    s = v
                elif t == xlrd.XL_CELL_NUMBER:
                    s = _fmt_float(float(v))
                elif t == xlrd.XL_CELL_DATE:
                    try:
                        s = _fmt_cell(xlrd.xldate.xldate_as_datetime(v, book.datemode))
                    except Exception:  # noqa: BLE001
                        s = _fmt_float(float(v))
                elif t == xlrd.XL_CELL_BOOLEAN:
                    s = "true" if v else "false"
                elif t == xlrd.XL_CELL_ERROR:
                    name = xlrd.error_text_from_code.get(v, str(v))
                    s = f"ERR:{_XLS_ERRORS.get(name, name)}"
                else:
                    s = str(v)
                cells[(r, c)] = s
        sheets.append((sh.name, cells))
    return sheets


def _read_ods(path: str) -> List[Sheet]:
    with _open_zip(path) as z:
        try:
            data = z.read("content.xml")
        except KeyError:
            raise CommandError(f"Failed to open spreadsheet '{path}': missing content.xml")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise CommandError(f"Failed to open spreadsheet '{path}': {e}")
    T = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
    X = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
    O = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
    sheets: List[Sheet] = []
    for table in root.iter(f"{T}table"):
        name = table.get(f"{T}name") or ""
        cells: Dict[Tuple[int, int], str] = {}
        r = 0
        for row in table.iter(f"{T}table-row"):
            rrep = int(row.get(f"{T}number-rows-repeated") or 1)
            c = 0
            row_cells: Dict[int, str] = {}
            for cell in row:
                if cell.tag not in (f"{T}table-cell", f"{T}covered-table-cell"):
                    continue
                crep = int(cell.get(f"{T}number-columns-repeated") or 1)
                vt = cell.get(f"{O}value-type")
                s = ""
                if vt in ("float", "percentage", "currency"):
                    try:
                        s = _fmt_float(float(cell.get(f"{O}value") or ""))
                    except ValueError:
                        s = ""
                elif vt == "boolean":
                    s = cell.get(f"{O}boolean-value") or ""
                elif vt == "date":
                    s = (cell.get(f"{O}date-value") or "").replace("T", " ")
                    if s.endswith(" 00:00:00"):
                        s = s[:-9]
                elif vt == "time":
                    s = cell.get(f"{O}time-value") or ""
                elif vt == "string":
                    s = cell.get(f"{O}string-value") or "\n".join(
                        "".join(p.itertext()) for p in cell.findall(f"{X}p")
                    )
                if s:
                    for k in range(min(crep, 1024)):
                        row_cells[c + k] = s
                c += crep
            if row_cells:
                for k in range(min(rrep, 1024)):
                    for cc, s in row_cells.items():
                        cells[(r + k, cc)] = s
            r += rrep
        sheets.append((name, cells))
    return sheets


def extract_spreadsheet(path: str, ext: str) -> str:
    if ext in ("xlsx", "xlsm"):
        sheets = _read_xlsx(path)
    elif ext == "xls":
        sheets = _read_xls(path)
    elif ext == "ods":
        sheets = _read_ods(path)
    else:
        raise CommandError(f"no compatibility parser is available for .{ext}")
    return sheets_to_markdown(sheets)


def sheets_to_markdown(sheets: List[Sheet]) -> str:
    result: List[str] = []
    multi = len(sheets) > 1
    for name, cells in sheets:
        if not cells:
            continue
        if multi:
            result.append(f"## {name}\n\n")
        r0 = min(r for r, _ in cells)
        r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        c1 = max(c for _, c in cells)
        max_cols = c1 - c0 + 1
        for i, r in enumerate(range(r0, r1 + 1)):
            row = [cells.get((r, c), "").replace("|", "\\|") for c in range(c0, c1 + 1)]
            result.append("| " + " | ".join(row) + " |\n")
            if i == 0:
                result.append("|" + " --- |" * max_cols + "\n")
        result.append("\n")
    out = "".join(result)
    if not out.strip():
        return "[Could not extract data from spreadsheet]"
    return out


# ---------------------------------------------------------------------------
# Ebooks (port of commands/ebook.rs)
# ---------------------------------------------------------------------------

def extract_ebook_text(path: str, extension: str) -> str:
    _validate_source_file(path)
    if extension == "epub":
        return _extract_epub(path)
    if extension == "mobi":
        return _extract_mobi(path)
    raise CommandError(f"Unsupported ebook format: .{extension}")


def _validate_source_file(path: str) -> None:
    try:
        st = os.stat(path)
    except OSError as e:
        raise CommandError(f"Failed to inspect ebook '{path}': {os_err(e)}")
    import stat as _stat
    if not _stat.S_ISREG(st.st_mode):
        raise CommandError(f"Ebook path is not a file: '{path}'")
    if st.st_size > MAX_EBOOK_BYTES:
        raise CommandError(f"Ebook exceeds the {MAX_EBOOK_BYTES // 1024 // 1024} MB extraction limit")


def _enclosed_name_ok(name: str) -> bool:
    if not name or "\0" in name:
        return False
    n = name.replace("\\", "/")
    if n.startswith("/") or re.match(r"^[A-Za-z]:", n):
        return False
    depth = 0
    for part in n.split("/"):
        if part == "..":
            depth -= 1
            if depth < 0:
                return False
        elif part not in ("", "."):
            depth += 1
    return True


def is_epub_text_entry(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    return ext in ("html", "htm", "xhtml", "xml")


def validate_epub_archive(path: str) -> None:
    try:
        f = open(path, "rb")
    except OSError as e:
        raise CommandError(f"Failed to open EPUB '{path}': {os_err(e)}")
    with f:
        try:
            z = zipfile.ZipFile(f)
        except zipfile.BadZipFile as e:
            raise CommandError(f"Invalid EPUB ZIP container: {e}")
        infos = z.infolist()
        if len(infos) > MAX_EPUB_ENTRIES:
            raise CommandError(
                f"EPUB contains too many archive entries ({len(infos)} > {MAX_EPUB_ENTRIES})"
            )
        expanded = 0
        for info in infos:
            if not _enclosed_name_ok(info.filename):
                raise CommandError(f"EPUB contains an unsafe archive path: {info.filename}")
            if is_epub_text_entry(info.filename) and info.file_size > MAX_EPUB_CHAPTER_BYTES:
                raise CommandError(
                    f"EPUB text entry '{info.filename}' exceeds the "
                    f"{MAX_EPUB_CHAPTER_BYTES // 1024 // 1024} MB safety limit"
                )
            expanded += info.file_size
            if expanded > MAX_EPUB_EXPANDED_BYTES:
                raise CommandError(
                    f"EPUB expanded content exceeds the {MAX_EPUB_EXPANDED_BYTES // 1024 // 1024} MB safety limit"
                )
            comp = info.compress_size
            if info.file_size > 1024 * 1024 and comp > 0 and info.file_size // comp > MAX_EPUB_COMPRESSION_RATIO:
                raise CommandError(f"EPUB entry has an unsafe compression ratio: {info.filename}")


def safe_inline_text(value: str, max_chars: int) -> str:
    chars = [" " if unicodedata.category(c) == "Cc" else c for c in value][:max_chars]
    return " ".join("".join(chars).split()).strip()


def _non_empty(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    return v or None


def ebook_header(title, author, language, publisher, fmt) -> str:
    out = f"# {safe_inline_text(title, 500)}\n\n"
    out += "## Book metadata\n\n"
    out += f"- Format: {fmt.upper()}\n"
    if _non_empty(author):
        out += f"- Author: {safe_inline_text(_non_empty(author), 1000)}\n"
    if _non_empty(language):
        out += f"- Language: {safe_inline_text(_non_empty(language), 100)}\n"
    if _non_empty(publisher):
        out += f"- Publisher: {safe_inline_text(_non_empty(publisher), 1000)}\n"
    out += "\n## Contents"
    return out


class _TextHTML(HTMLParser):
    _SKIP = {"script", "style", "head", "noscript", "template"}
    _BLOCK = {
        "p", "div", "section", "article", "header", "footer", "nav", "aside", "ul", "ol", "table",
        "tr", "pre", "blockquote", "figure", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "dl", "dt", "dd", "hr", "br", "body", "html", "title",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if tag in self._BLOCK:
            self.parts.append("\n\n" if tag not in ("br", "li", "tr") else "\n")
        if len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
            self.parts.append("#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("* ")
        elif tag == "blockquote":
            self.parts.append("> ")

    def handle_startendtag(self, tag, attrs):
        if tag in self._SKIP:
            return
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            if self.skip:
                self.skip -= 1
            return
        if self.skip:
            return
        if tag in self._BLOCK and tag != "br":
            self.parts.append("\n\n" if tag not in ("li", "tr") else "\n")

    def handle_data(self, data):
        if self.skip:
            return
        self.parts.append(re.sub(r"[ \t\r\n\f]+", " ", data))


def _decode_html(data: bytes) -> str:
    head = data[:2048]
    m = re.search(rb"""(?:encoding|charset)\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""", head)
    encodings = []
    if data.startswith(b"\xef\xbb\xbf"):
        encodings.append("utf-8-sig")
    if m:
        encodings.append(m.group(1).decode("ascii", "ignore"))
    encodings.append("utf-8")
    for enc in encodings:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("cp1252", errors="replace")


def html_to_text(data) -> str:
    text = _decode_html(data) if isinstance(data, (bytes, bytearray)) else data
    p = _TextHTML()
    try:
        p.feed(text)
        p.close()
    except Exception as e:  # noqa: BLE001
        raise CommandError(f"Failed to convert ebook HTML to text: {e}")
    raw = "".join(p.parts)
    lines = [ln.strip() for ln in raw.replace("\r\n", "\n").split("\n")]
    out: List[str] = []
    blank = 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank <= 1 and out:
                out.append("")
        else:
            blank = 0
            out.append(ln)
    return "\n".join(out).strip("\n")


def push_bounded(output: List[str], size: List[int], value: str) -> None:
    add = len(value.encode("utf-8", "surrogatepass"))
    if size[0] + add > MAX_EXTRACTED_TEXT_BYTES:
        raise CommandError(
            f"Extracted ebook text exceeds the {MAX_EXTRACTED_TEXT_BYTES // 1024 // 1024} MB safety limit"
        )
    output.append(value)
    size[0] += add


def _file_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0] or "Untitled ebook"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _extract_epub(path: str) -> str:
    validate_epub_archive(path)
    try:
        z = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as e:
        raise CommandError(f"Failed to parse EPUB '{path}': {os_err(e)}")
    with z:
        try:
            container = ET.fromstring(z.read("META-INF/container.xml"))
            opf_path = None
            for el in container.iter():
                if _local(el.tag) == "rootfile" and el.get("full-path"):
                    opf_path = el.get("full-path")
                    break
            if not opf_path:
                raise ValueError("no rootfile in container.xml")
            opf = ET.fromstring(z.read(opf_path))
        except Exception as e:  # noqa: BLE001
            raise CommandError(f"Failed to parse EPUB '{path}': {e}")
        opf_dir = posixpath.dirname(opf_path)

        meta: Dict[str, str] = {}
        manifest: Dict[str, Tuple[str, str]] = {}
        spine: List[str] = []
        for el in opf.iter():
            ln = _local(el.tag)
            if ln in ("title", "creator", "language", "publisher") and ln not in meta:
                meta[ln] = (el.text or "").strip()
            elif ln == "item" and el.get("id") and el.get("href"):
                manifest[el.get("id")] = (el.get("href"), el.get("media-type") or "")
            elif ln == "itemref" and el.get("idref"):
                spine.append(el.get("idref"))

        title = meta.get("title") or _file_stem(path)
        output: List[str] = []
        size = [0]
        push_bounded(output, size, ebook_header(title, meta.get("creator"), meta.get("language"),
                                                meta.get("publisher"), "epub"))
        extracted = 0
        names = set(z.namelist())
        for index in range(min(len(spine), MAX_CHAPTERS)):
            item = manifest.get(spine[index])
            if item is None:
                continue
            href, mime = item
            chapter_path = posixpath.normpath(posixpath.join(opf_dir, unquote(href.split("#")[0])))
            if chapter_path not in names:
                continue
            if "html" not in mime and "xml" not in mime:
                continue
            with z.open(chapter_path) as fh:
                data = fh.read(MAX_EPUB_CHAPTER_BYTES + 1)
            if len(data) > MAX_EPUB_CHAPTER_BYTES:
                raise CommandError(
                    f"EPUB chapter {index + 1} exceeds the {MAX_EPUB_CHAPTER_BYTES // 1024 // 1024} MB safety limit"
                )
            text = html_to_text(data)
            if not text.strip():
                continue
            push_bounded(
                output, size,
                f"\n\n## Chapter {index + 1} · {safe_inline_text(chapter_path, 240)}\n\n{text.strip()}",
            )
            extracted += 1
    if extracted == 0:
        raise CommandError("EPUB contains no extractable chapter text")
    return "".join(output)


_MOBI_LANG = {
    1: "Arabic", 4: "Chinese", 5: "Czech", 6: "Danish", 7: "German", 8: "Greek", 9: "English",
    10: "Spanish", 11: "Finnish", 12: "French", 13: "Hebrew", 14: "Hungarian", 15: "Icelandic",
    16: "Italian", 17: "Japanese", 18: "Korean", 19: "Dutch", 20: "Norwegian", 21: "Polish",
    22: "Portuguese", 24: "Romanian", 25: "Russian", 29: "Swedish", 30: "Thai", 31: "Turkish",
}


def _palmdoc_decompress(data: bytes) -> bytes:
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        c = data[i]
        i += 1
        if c == 0 or 9 <= c <= 0x7F:
            out.append(c)
        elif 1 <= c <= 8:
            out += data[i:i + c]
            i += c
        elif c >= 0xC0:
            out.append(0x20)
            out.append(c ^ 0x80)
        else:  # 0x80..0xBF: back reference
            if i >= n:
                break
            c = (c << 8) | data[i]
            i += 1
            dist = (c >> 3) & 0x7FF
            length = (c & 7) + 3
            if dist == 0 or dist > len(out):
                continue
            for _ in range(length):
                out.append(out[-dist])
    return bytes(out)


def _trailing_entry_size(data: bytes, psize: int) -> int:
    bitpos = 0
    result = 0
    while True:
        v = data[psize - 1]
        result |= (v & 0x7F) << bitpos
        bitpos += 7
        psize -= 1
        if (v & 0x80) != 0 or bitpos >= 28 or psize == 0:
            return result


def _extract_mobi(path: str) -> str:
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except OSError as e:
        raise CommandError(f"Failed to parse MOBI '{path}': {os_err(e)}")

    def bad(msg: str) -> CommandError:
        return CommandError(f"Failed to parse MOBI '{path}': {msg}")

    if len(blob) < 78 + 8 or blob[60:68] not in (b"BOOKMOBI", b"TEXtREAd"):
        raise bad("not a MOBI/PalmDOC database")
    nrec = struct.unpack(">H", blob[76:78])[0]
    if nrec < 2 or len(blob) < 78 + 8 * nrec:
        raise bad("invalid record table")
    offs = [struct.unpack(">I", blob[78 + 8 * i:82 + 8 * i])[0] for i in range(nrec)]
    offs.append(len(blob))
    rec0 = blob[offs[0]:offs[1]]
    if len(rec0) < 16:
        raise bad("invalid header record")
    compression, _, text_length, text_records, _rs, encryption = struct.unpack(">HHIHHH", rec0[:14])
    if encryption != 0:
        raise CommandError("Encrypted/DRM-protected MOBI files are not supported")
    if text_length > MAX_EXTRACTED_TEXT_BYTES:
        raise CommandError(f"MOBI declares more than {MAX_EXTRACTED_TEXT_BYTES // 1024 // 1024} MB of text")
    if compression not in (1, 2):
        raise bad("unsupported compression (only uncompressed and PalmDOC are supported)")

    has_mobi = rec0[16:20] == b"MOBI"
    enc = 1252
    full_name = ""
    author = publisher = None
    language = "Neutral"
    extra_flags = 0
    if has_mobi:
        hlen = struct.unpack(">I", rec0[20:24])[0]
        enc = struct.unpack(">I", rec0[28:32])[0]
        codec = "utf-8" if enc == 65001 else "cp1252"
        if len(rec0) >= 96:
            name_off, name_len = struct.unpack(">II", rec0[84:92])
            full_name = rec0[name_off:name_off + name_len].decode(codec, "replace")
            locale = struct.unpack(">I", rec0[92:96])[0]
            language = _MOBI_LANG.get(locale & 0xFF, "Neutral")
        if hlen >= 0xE4 and len(rec0) >= 0xF4:
            extra_flags = struct.unpack(">H", rec0[0xF2:0xF4])[0]
        exth_flags = struct.unpack(">I", rec0[128:132])[0] if len(rec0) >= 132 else 0
        if exth_flags & 0x40:
            p = 16 + hlen
            if rec0[p:p + 4] == b"EXTH":
                count = struct.unpack(">I", rec0[p + 8:p + 12])[0]
                q = p + 12
                for _ in range(count):
                    if q + 8 > len(rec0):
                        break
                    rtype, rlen = struct.unpack(">II", rec0[q:q + 8])
                    val = rec0[q + 8:q + rlen].decode(codec, "replace")
                    if rtype == 100 and author is None:
                        author = val
                    elif rtype == 101 and publisher is None:
                        publisher = val
                    if rlen < 8:
                        break
                    q += rlen
    else:
        codec = "cp1252"

    chunks = bytearray()
    for i in range(1, min(text_records, nrec - 1) + 1):
        data = blob[offs[i]:offs[i + 1]]
        if extra_flags:
            size = len(data)
            num = 0
            flags = extra_flags >> 1
            while flags:
                if flags & 1 and size - num > 0:
                    num += _trailing_entry_size(data, size - num)
                flags >>= 1
            if extra_flags & 1 and size - num - 1 >= 0:
                num += (data[size - num - 1] & 3) + 1
            data = data[:size - num]
        chunks += _palmdoc_decompress(data) if compression == 2 else data
        if len(chunks) > MAX_EXTRACTED_TEXT_BYTES * 2:
            raise CommandError(f"MOBI declares more than {MAX_EXTRACTED_TEXT_BYTES // 1024 // 1024} MB of text")
    raw = bytes(chunks[:text_length] if text_length else chunks).decode(codec, "replace")

    title = full_name.strip() or _file_stem(path)
    text = html_to_text(raw) if "<" in raw else raw
    if not text.strip():
        raise CommandError("MOBI contains no extractable text")
    output: List[str] = []
    size_ = [0]
    push_bounded(output, size_, ebook_header(title, author, language, publisher, "mobi"))
    push_bounded(output, size_, f"\n\n{text.strip()}")
    return "".join(output)
