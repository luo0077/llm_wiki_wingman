"""Port of src/lib/mineru.ts: MinerU cloud (v4 API) and self-hosted (``mineru-api``) PDF parsing.

Public API (all coroutines unless noted)::

    parse_with_mineru(config, source_path, source_url=None, on_progress=None, signal=None,
                      asset_options=None) -> str
    parse_with_mineru_result(...same...) -> MineruExtractedMarkdown   # .markdown, .savedImages
    test_mineru_connection(token, config=None) -> None                # raises on failure
    DEFAULT_LOCAL_MINERU_ENDPOINT

* ``config``: dict with the TS ``MineruConfig`` keys (``enabled``, ``backend``, ``localEndpoint``,
  ``localToken``, ``localBackend``, ``localEffort``, ``localParseMethod``, ``localLanguage``,
  ``localFormulaEnabled``, ``localTableEnabled``, ``localImageAnalysis``, ``localServerUrl``,
  ``token``, ``modelVersion``).
* ``asset_options``: ``{"projectPath": ..., "sourceSummarySlug": ...}`` (snake_case accepted) -
  when given, images from the result are written under ``<project>/wiki/media/<slug>/mineru/`` and
  the Markdown links are rewritten.
* ``on_progress(msg)`` may be sync or async; ``signal`` is abort-like (see ``llm_client.AbortSignal``).

ZIP safety (additions over the JSZip original): the downloaded archive is size-capped, the entry
count and the total / per-entry uncompressed size are bounded (zip-bomb guard, enforced while
reading, not from the declared sizes), encrypted entries are ignored, and entry names are reduced
to a safe relative path (``..``/``.`` segments dropped) before anything touches the disk.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import re
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

import httpx

from ._js import Record
from .extract_source_images import SavedImage
from .http_fetch import make_async_client
from .llm_jscompat import (
    JS_WS,
    JS_WS_CHARS,
    RE_I,
    dumps_js,
    decode_uri_component,
    encode_uri_component,
    from_utf16,
    js_json_loads,
    js_str,
    js_trim,
    parse_url,
    to_utf16,
)
from .path_utils import get_file_name, normalize_path

log = logging.getLogger("llm_wiki_server.ingest.mineru")

API_BASE = "https://mineru.net/api/v4"
DEFAULT_LOCAL_MINERU_ENDPOINT = "http://127.0.0.1:8000"
POLL_INTERVAL_S = 3.0
POLL_TIMEOUT_S = 300.0                 # 5 minutes
LOCAL_POLL_TIMEOUT_S = 3600.0          # official mineru-api tasks can include model cold starts
MAX_ACCURATE_PARSE_BYTES = 200 * 1024 * 1024
MINERU_IMAGE_EXTS = frozenset(("png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "tif", "tiff"))

# zip safety limits
MAX_ZIP_DOWNLOAD_BYTES = 512 * 1024 * 1024
MAX_ZIP_ENTRIES = 20_000
MAX_ZIP_ENTRY_BYTES = 256 * 1024 * 1024
MAX_ZIP_MARKDOWN_BYTES = 64 * 1024 * 1024
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024

_HTTP_TIMEOUT = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0)


class MineruError(Exception):
    pass


@dataclass
class MineruExtractedMarkdown(Record):
    markdown: str
    savedImages: List[SavedImage]


# ----------------------------------------------------------------- fs boundary (patchable in tests)

async def create_directory(path: str) -> None:
    from ..commands import fs
    await asyncio.to_thread(fs.create_directory, path)


async def get_file_size(path: str) -> int:
    from ..commands import fs
    return await asyncio.to_thread(fs.get_file_size, path)


async def read_file_as_base64(path: str) -> Dict[str, str]:
    from ..commands import fs
    return await asyncio.to_thread(fs.read_file_as_base64, path)


async def write_file_base64(path: str, b64: str) -> None:
    from ..commands import fs
    await asyncio.to_thread(fs.write_file_base64, path, b64)


async def _maybe(r: Any) -> Any:
    import inspect
    return await r if inspect.isawaitable(r) else r


# ----------------------------------------------------------------- small helpers

def _opt(d: Any, camel: str, snake: Optional[str] = None) -> Any:
    if d is None:
        return None
    if isinstance(d, Mapping):
        v = d.get(camel)
        if v is None and snake:
            v = d.get(snake)
        return v
    v = getattr(d, camel, None)
    if v is None and snake:
        v = getattr(d, snake, None)
    return v


def _is_aborted(signal: Any) -> bool:
    from .llm_client import is_aborted
    return is_aborted(signal)


def throw_if_aborted(signal: Any = None) -> None:
    if _is_aborted(signal):
        raise MineruError("MinerU parsing cancelled")


async def _abortable(coro: Awaitable[Any], signal: Any) -> Any:
    """Await `coro`, cancelling it (and raising the 'cancelled' error) if the signal fires."""
    if signal is None:
        return await coro
    task = asyncio.ensure_future(coro)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.02)
            if done:
                return task.result()
            if _is_aborted(signal):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise MineruError("MinerU parsing cancelled")
    except asyncio.CancelledError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


async def _wait_for_poll_interval(signal: Any = None) -> None:
    throw_if_aborted(signal)
    end = time.monotonic() + POLL_INTERVAL_S
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(0.02, remaining))
        throw_if_aborted(signal)


def _g(obj: Any, key: Any) -> Any:
    if isinstance(key, int):
        return obj[key] if isinstance(obj, list) and -1 < key < len(obj) else None
    return obj.get(key) if isinstance(obj, dict) else None


async def _json(res: httpx.Response) -> Any:
    return js_json_loads(res.content.decode("utf-8-sig", errors="replace"))


def _ok(res: httpx.Response) -> bool:
    return 200 <= res.status_code < 300


def _text(res: httpx.Response) -> str:
    try:
        return res.content.decode("utf-8", errors="replace")
    except Exception:
        return ""


# ----------------------------------------------------------------- API error mapping

_KNOWN_ERRORS = {
    "A0202": "MinerU token is invalid. Check the API token in Settings.",
    "A0211": "MinerU token has expired. Create a new API token and update Settings.",
    "-60005": "MinerU rejected the file because it is larger than 200 MB.",
    "-60006": "MinerU rejected the file because it exceeds the 200 page limit.",
    "-60018": "MinerU daily parsing quota has been reached.",
}


def mineru_api_error_message(code: Any, msg: Optional[str] = None) -> str:
    key = "" if code is None else js_str(code)
    known = _KNOWN_ERRORS.get(key)
    if known:
        return f"{known} ({msg})" if msg else known
    shown = key or "unknown"
    return f"MinerU API error {shown}: {msg}" if msg else f"MinerU API error {shown}"


def _code_is_success(code: Any) -> bool:
    return (isinstance(code, (int, float)) and not isinstance(code, bool) and code == 0) or code == "0"


def assert_mineru_success(json_value: Any) -> None:
    code = _g(json_value, "code")
    if not _code_is_success(code):
        msg = _g(json_value, "msg")
        raise MineruError(mineru_api_error_message(code, msg if isinstance(msg, str) else None))


# ----------------------------------------------------------------- URL / headers

def local_mineru_api_base(endpoint: Optional[str]) -> str:
    candidate = js_trim(endpoint or "") or DEFAULT_LOCAL_MINERU_ENDPOINT
    parsed = parse_url(candidate)
    if parsed is None:
        raise MineruError("Local MinerU endpoint must be a valid HTTP(S) URL")
    if parsed.protocol not in ("http:", "https:") or parsed.username or parsed.password:
        raise MineruError("Local MinerU endpoint must be an HTTP(S) URL without credentials")
    return re.sub(r"/+\Z", "", candidate)


def _mineru_headers(token: str) -> Dict[str, str]:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def _local_headers(token: Optional[str]) -> Dict[str, str]:
    value = js_trim(token or "")
    return {"Authorization": f"Bearer {value}"} if value else {}


# ----------------------------------------------------------------- bytes / paths

def decode_base64_to_bytes(b64: str) -> bytes:
    """`atob`-style decode (forgiving of missing padding/whitespace; ValueError on junk)."""
    s = re.sub(r"[\t\n\f\r ]", "", b64)
    if len(s) % 4 == 0:
        s = re.sub(r"={1,2}\Z", "", s)
    if len(s) % 4 == 1 or re.search(r"[^A-Za-z0-9+/]", s):
        raise ValueError("The string to be decoded is not correctly encoded.")
    return base64.b64decode(s + "=" * (-len(s) % 4))


def _image_mime_type(path: str) -> str:
    ext = get_file_name(path).split(".")[-1].lower()
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif",
        "webp": "image/webp", "bmp": "image/bmp", "svg": "image/svg+xml",
        "tif": "image/tiff", "tiff": "image/tiff",
    }.get(ext, "application/octet-stream")


def _extension_for_mime_type(mime_type: str) -> Optional[str]:
    return {
        "image/jpeg": "jpg", "image/png": "png", "image/gif": "gif", "image/webp": "webp",
        "image/bmp": "bmp", "image/svg+xml": "svg", "image/tiff": "tiff",
    }.get(js_trim(mime_type).lower())


def _safe_asset_segment(segment: str) -> str:
    try:
        decoded = decode_uri_component(segment)
    except ValueError:
        decoded = segment
    cleaned = re.sub(r"[<>:\"|?*\x00-\x1f]", "_", decoded)
    cleaned = re.sub(r"[\\/]+", "_", cleaned)
    cleaned = re.sub(r"^\.+\Z", "_", cleaned)
    cleaned = re.sub(r"[. ]+\Z", "_", cleaned)
    if not cleaned:
        return "asset"
    u = to_utf16(cleaned)
    if len(u) <= 80:
        return cleaned
    dot = u.rfind(".")
    if 0 < dot < len(u) - 1:
        ext = u[dot:][:16]
        return from_utf16(u[:max(1, 80 - len(ext))] + ext)
    return from_utf16(u[:80])


def normalize_zip_path(path: str) -> str:
    p = re.sub(r"^\./+", "", normalize_path(path))
    return "/".join(part for part in p.split("/") if part and part != "." and part != "..")


def _decode_path(path: str) -> str:
    try:
        return decode_uri_component(path)
    except ValueError:
        return path


def _is_image_path(path: str) -> bool:
    return get_file_name(path).split(".")[-1].lower() in MINERU_IMAGE_EXTS


def _asset_rel_path(slug: str, zip_path: str) -> str:
    parts = [_safe_asset_segment(p) for p in normalize_zip_path(zip_path).split("/")]
    parts = [p for p in parts if p]
    safe_path = "/".join(parts) if parts else "image.png"
    return f"media/{slug}/mineru/{safe_path}"


_EXTERNAL_URL_RE = re.compile(r"(https?:|data:|blob:|file:|tauri:|asset:)", RE_I)


def _is_external_or_data_url(url: str) -> bool:
    return _EXTERNAL_URL_RE.match(url) is not None


# ----------------------------------------------------------------- HTML -> Markdown

def _safe_code_point(raw: str, radix: int) -> str:
    original = f"&#x{raw};" if radix == 16 else f"&#{raw};"
    try:
        n = int(raw, radix)
    except ValueError:
        return original
    if n < 0 or n > 0x10FFFF:
        return original
    if 0xD800 <= n <= 0xDFFF:
        return "�"
    return chr(n)


def decode_html_entities(text: str) -> str:
    text = re.sub(r"&nbsp;", " ", text, flags=RE_I)
    text = re.sub(r"&amp;", "&", text, flags=RE_I)
    text = re.sub(r"&lt;", "<", text, flags=RE_I)
    text = re.sub(r"&gt;", ">", text, flags=RE_I)
    text = re.sub(r"&quot;", '"', text, flags=RE_I)
    text = text.replace("&#39;", "'")
    text = re.sub(r"&#([0-9]+);", lambda m: _safe_code_point(m.group(1), 10), text)
    text = re.sub(r"&#x([0-9a-f]+);", lambda m: _safe_code_point(m.group(1), 16), text, flags=RE_I)
    return text


_IMG_TAG_RE = re.compile(r"<img\b[^>]*\bsrc=([\"'])([^\"']+)\1[^>]*>", RE_I)
_ALT_RE = re.compile(r"\balt=([\"'])([^\"']*)\1", RE_I)


def _alt_of(tag: str) -> str:
    m = _ALT_RE.search(tag)
    return m.group(2) if m else ""


def _html_img_tags_to_markdown(html: str) -> str:
    return _IMG_TAG_RE.sub(lambda m: f"![{_alt_of(m.group(0))}]({m.group(2)})", html)


def _html_cell_to_markdown(cell: str) -> str:
    s = _html_img_tags_to_markdown(cell)
    s = re.sub(r"<br" + JS_WS + r"*/?>", "<br>", s, flags=RE_I)
    s = re.sub(r"</p" + JS_WS + r"*>", "<br>", s, flags=RE_I)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(JS_WS + r"*<br>" + JS_WS + r"*", "<br>", s, flags=RE_I)
    s = re.sub(JS_WS + r"+", " ", s)
    return decode_html_entities(js_trim(s)).replace("|", "\\|")


_TABLE_RE = re.compile(r"<table\b[\s\S]*?</table>", RE_I)
_ROW_RE = re.compile(r"<tr\b[^>]*>([\s\S]*?)</tr>", RE_I)
_CELL_RE = re.compile(r"<t[dh]\b[^>]*>([\s\S]*?)</t[dh]>", RE_I)


def _convert_html_tables_in_segment(segment: str) -> str:
    def convert(m: "re.Match[str]") -> str:
        table_html = m.group(0)
        rows: List[List[str]] = []
        for row_match in _ROW_RE.finditer(table_html):
            cells = [_html_cell_to_markdown(c.group(1) or "") for c in _CELL_RE.finditer(row_match.group(1) or "")]
            if cells:
                rows.append(cells)
        if not rows:
            return table_html
        width = max(len(r) for r in rows)
        padded = [r + [""] * (width - len(r)) for r in rows]
        header = padded[0]
        separator = ["---"] * width
        body = padded[1:]
        return "\n".join(["", f"| {' | '.join(header)} |", f"| {' | '.join(separator)} |",
                          *[f"| {' | '.join(r)} |" for r in body], ""])
    return _TABLE_RE.sub(convert, segment)


_CODE_SPLIT_RE = re.compile(r"(```[\s\S]*?```|~~~[\s\S]*?~~~)")


def convert_html_tables_to_markdown(markdown: str) -> str:
    out = []
    for segment in _CODE_SPLIT_RE.split(markdown):
        out.append(segment if segment.startswith("```") or segment.startswith("~~~")
                   else _convert_html_tables_in_segment(segment))
    return "".join(out)


def _encode_markdown_image_url(rel_path: str) -> str:
    def enc(part: str) -> str:
        return re.sub(r"[!'()*]", lambda m: "%" + format(ord(m.group(0)), "X"), encode_uri_component(part))
    return "/".join(enc(p) for p in rel_path.split("/"))


_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)]\(((?:[^()]|\([^()]*\))*)\)")
_TITLE_RE = re.compile(r"^([\s\S]+?)(" + JS_WS + r"+[\"'][^\"']*[\"']" + JS_WS + r"*)\Z")
_TOKEN_RE = re.compile(r"^([^" + JS_WS_CHARS + r"]+)([\s\S]*)\Z")


def rewrite_mineru_markdown_images(markdown: str, path_map: Mapping[str, str]) -> str:
    def lookup(raw_url: str) -> Optional[str]:
        if not raw_url or _is_external_or_data_url(raw_url):
            return None
        cleaned = normalize_zip_path(raw_url.split("#")[0])
        if not cleaned:
            return None
        decoded = _decode_path(cleaned)
        for key in (cleaned, decoded, get_file_name(decoded)):
            v = path_map.get(key)
            if v is not None:
                return v
        return None

    def md_repl(m: "re.Match[str]") -> str:
        full, alt, target = m.group(0), m.group(1), m.group(2)
        trimmed = js_trim(target)
        candidates: List[Tuple[str, str]] = []
        if trimmed.startswith("<") and ">" in trimmed:
            end = trimmed.index(">")
            candidates.append((trimmed[1:end], trimmed[end + 1:]))
        else:
            tm = _TITLE_RE.match(trimmed)
            if tm:
                candidates.append((js_trim(tm.group(1)), tm.group(2)))
            candidates.append((trimmed, ""))
            km = _TOKEN_RE.match(trimmed)
            if km:
                candidates.append((km.group(1), km.group(2)))
        for url, suffix in candidates:
            rel = lookup(url)
            if not rel:
                continue
            return f"![{alt}]({_encode_markdown_image_url(rel)}{suffix})"
        return full

    with_md = _MD_IMAGE_RE.sub(md_repl, markdown)

    def img_repl(m: "re.Match[str]") -> str:
        rel = lookup(m.group(2))
        if not rel:
            return m.group(0)
        return f"![{_alt_of(m.group(0))}]({_encode_markdown_image_url(rel)})"

    return _IMG_TAG_RE.sub(img_repl, with_md)


# ----------------------------------------------------------------- cloud API calls

async def _submit_url_task(client: httpx.AsyncClient, token: str, url: str, model_version: str,
                           signal: Any) -> str:
    throw_if_aborted(signal)
    res = await _abortable(client.post(
        f"{API_BASE}/extract/task", headers=_mineru_headers(token),
        content=dumps_js({"url": url, "model_version": model_version}).encode("utf-8")), signal)
    if not _ok(res):
        raise MineruError(f"MinerU submit failed: HTTP {res.status_code}")
    data = await _json(res)
    assert_mineru_success(data)
    return _g(_g(data, "data"), "task_id")


async def _upload_file_for_task(client: httpx.AsyncClient, token: str, file_name: str, file_base64: str,
                                model_version: str, signal: Any) -> Dict[str, str]:
    headers = _mineru_headers(token)
    throw_if_aborted(signal)

    res = await _abortable(client.post(
        f"{API_BASE}/file-urls/batch", headers=headers,
        content=dumps_js({"files": [{"name": file_name, "data_id": file_name}],
                          "model_version": model_version}).encode("utf-8")), signal)
    if not _ok(res):
        raise MineruError(f"MinerU batch submit failed: HTTP {res.status_code}")
    data = await _json(res)
    assert_mineru_success(data)

    batch_id = _g(_g(data, "data"), "batch_id")
    upload_url = _g(_g(_g(data, "data"), "file_urls"), 0)
    if not batch_id or not upload_url:
        raise MineruError("MinerU did not return a file upload URL")

    body = decode_base64_to_bytes(file_base64)
    throw_if_aborted(signal)
    upload = await _abortable(client.put(upload_url, content=body), signal)
    if not _ok(upload) and upload.status_code not in (200, 201):
        raise MineruError(f"MinerU file upload failed: HTTP {upload.status_code}")
    return {"batchId": batch_id, "uploadUrl": upload_url}


async def _poll_task(client: httpx.AsyncClient, token: str, task_id: str, signal: Any) -> str:
    headers = _mineru_headers(token)
    start = time.monotonic()
    while time.monotonic() - start < POLL_TIMEOUT_S:
        throw_if_aborted(signal)
        res = await _abortable(client.get(f"{API_BASE}/extract/task/{task_id}", headers=headers), signal)
        if not _ok(res):
            raise MineruError(f"MinerU poll failed: HTTP {res.status_code}")
        data = await _json(res)
        assert_mineru_success(data)
        d = _g(data, "data")
        state = _g(d, "state")
        if state == "done" and _g(d, "full_zip_url"):
            return _g(d, "full_zip_url")
        if state == "failed":
            err = _g(d, "err_msg")
            raise MineruError(f"MinerU parsing failed: {js_str(err) if err is not None else 'unknown error'}")
        await _wait_for_poll_interval(signal)
    raise MineruError("MinerU parsing timed out after 5 minutes")


async def _poll_batch_task(client: httpx.AsyncClient, token: str, batch_id: str, signal: Any) -> str:
    headers = _mineru_headers(token)
    start = time.monotonic()
    while time.monotonic() - start < POLL_TIMEOUT_S:
        throw_if_aborted(signal)
        res = await _abortable(client.get(f"{API_BASE}/extract-results/batch/{batch_id}", headers=headers), signal)
        if not _ok(res):
            raise MineruError(f"MinerU batch poll failed: HTTP {res.status_code}")
        data = await _json(res)
        assert_mineru_success(data)
        result = _g(_g(_g(data, "data"), "extract_result"), 0)
        state = _g(result, "state")
        if state == "done" and _g(result, "full_zip_url"):
            return _g(result, "full_zip_url")
        if state == "failed":
            err = _g(result, "err_msg")
            raise MineruError(f"MinerU parsing failed: {js_str(err) if err is not None else 'unknown error'}")
        await _wait_for_poll_interval(signal)
    raise MineruError("MinerU parsing timed out after 5 minutes")


# ----------------------------------------------------------------- zip handling (safe)

class _ZipBudget:
    """Tracks total uncompressed bytes read from one archive."""

    def __init__(self) -> None:
        self.total = 0

    def read(self, zf: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
        remaining_total = MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES - self.total
        cap = min(limit, remaining_total)
        with zf.open(info) as f:
            data = f.read(cap + 1)
        if len(data) > cap:
            raise MineruError("MinerU result zip is too large when extracted (possible zip bomb)")
        self.total += len(data)
        return data


def _open_zip(content: bytes) -> zipfile.ZipFile:
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except (zipfile.BadZipFile, OSError, ValueError, NotImplementedError) as e:
        raise MineruError(f"MinerU result is not a valid zip file: {e}")
    if len(zf.infolist()) > MAX_ZIP_ENTRIES:
        zf.close()
        raise MineruError("MinerU result zip contains too many files")
    return zf


def _is_dir(info: zipfile.ZipInfo) -> bool:
    return info.is_dir()


def _readable(info: zipfile.ZipInfo) -> bool:
    return not (info.flag_bits & 0x1)  # skip encrypted entries


async def _save_zip_images(zf: zipfile.ZipFile, budget: _ZipBudget, opts: Any, signal: Any
                           ) -> Tuple[Dict[str, str], List[SavedImage]]:
    project_path = _opt(opts, "projectPath", "project_path")
    slug = _opt(opts, "sourceSummarySlug", "source_summary_slug")
    pp = normalize_path(project_path)
    root_dir = f"{pp}/wiki/media/{slug}/mineru"
    path_map: Dict[str, str] = {}
    saved: List[SavedImage] = []
    image_entries: List[Tuple[str, zipfile.ZipInfo]] = []
    basename_counts: Dict[str, int] = {}

    for info in zf.infolist():
        normalized = normalize_zip_path(info.filename)
        if not _is_dir(info) and _readable(info) and normalized and _is_image_path(normalized):
            image_entries.append((normalized, info))
            base = get_file_name(normalized)
            basename_counts[base] = basename_counts.get(base, 0) + 1

    if not image_entries:
        return path_map, saved

    await create_directory(root_dir)
    for zip_path, info in image_entries:
        throw_if_aborted(signal)
        rel_path = _asset_rel_path(slug, zip_path)
        abs_path = f"{pp}/wiki/{rel_path}"
        data = await asyncio.to_thread(budget.read, zf, info, MAX_ZIP_ENTRY_BYTES)
        await write_file_base64(abs_path, base64.b64encode(data).decode("ascii"))
        path_map[zip_path] = rel_path
        path_map[_decode_path(zip_path)] = rel_path
        base = get_file_name(zip_path)
        if basename_counts.get(base) == 1:
            path_map[base] = rel_path
            path_map[_decode_path(base)] = rel_path
        saved.append(SavedImage(
            index=len(saved) + 1, mimeType=_image_mime_type(rel_path), page=None, width=0, height=0,
            relPath=rel_path, absPath=abs_path, sha256=hashlib.sha256(data).hexdigest()))
    return path_map, saved


async def _download_zip(client: httpx.AsyncClient, zip_url: str, signal: Any) -> bytes:
    parsed = parse_url(zip_url)
    if parsed is None or parsed.protocol not in ("http:", "https:"):
        raise MineruError("MinerU returned an invalid result URL")

    async def fetch() -> bytes:
        async with client.stream("GET", zip_url) as res:
            if not _ok(res):
                raise MineruError(f"MinerU zip download failed: HTTP {res.status_code}")
            buf = bytearray()
            async for chunk in res.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > MAX_ZIP_DOWNLOAD_BYTES:
                    raise MineruError("MinerU result zip exceeds the maximum allowed download size")
            return bytes(buf)

    return await _abortable(fetch(), signal)


async def _download_and_extract_markdown(client: httpx.AsyncClient, zip_url: str, signal: Any = None,
                                         asset_options: Any = None) -> MineruExtractedMarkdown:
    throw_if_aborted(signal)
    content = await _download_zip(client, zip_url, signal)
    zf = _open_zip(content)
    try:
        # Official MinerU result archives contain full.md. Prefer it; fall back to another Markdown
        # file only for compatibility with older or unusual archives.
        md_entries = [i for i in zf.infolist() if not _is_dir(i) and _readable(i) and i.filename.endswith(".md")]
        if not md_entries:
            raise MineruError("No Markdown file found in MinerU result zip")
        full_md = next((i for i in md_entries if i.filename.split("/")[-1].lower() == "full.md"), None)
        budget = _ZipBudget()
        raw = await asyncio.to_thread(budget.read, zf, full_md or md_entries[0], MAX_ZIP_MARKDOWN_BYTES)
        markdown = raw.decode("utf-8", errors="replace")
        with_tables = convert_html_tables_to_markdown(markdown)
        if not asset_options:
            return MineruExtractedMarkdown(with_tables, [])

        try:
            path_map, saved = await _save_zip_images(zf, budget, asset_options, signal)
            return MineruExtractedMarkdown(
                rewrite_mineru_markdown_images(with_tables, path_map) if path_map else with_tables, saved)
        except Exception as err:
            if _is_aborted(signal):
                raise
            log.warning("[MinerU] failed to save extracted images; keeping parsed Markdown text: %s", err)
            return MineruExtractedMarkdown(with_tables, [])
    finally:
        zf.close()


async def download_and_extract_markdown_result(zip_url: str, signal: Any = None,
                                               asset_options: Any = None) -> MineruExtractedMarkdown:
    async with make_async_client(timeout=_HTTP_TIMEOUT) as client:
        return await _download_and_extract_markdown(client, zip_url, signal, asset_options)


async def download_and_extract_markdown(zip_url: str, signal: Any = None, asset_options: Any = None) -> str:
    return (await download_and_extract_markdown_result(zip_url, signal, asset_options)).markdown


# ----------------------------------------------------------------- local backend

_LOCAL_BACKENDS_3_0_TO_3_2 = {"vlm-engine": "vlm-auto-engine", "hybrid-engine": "hybrid-auto-engine"}


def local_mineru_backend_for_version(backend: str, version: Any) -> str:
    if isinstance(version, str) and re.match(r"3\.[0-2](?:\.|\Z)", js_trim(version)):
        return _LOCAL_BACKENDS_3_0_TO_3_2.get(backend, backend)
    return backend


async def _parse_with_local_mineru(
    config: Mapping[str, Any], source_path: str, file_name: str,
    on_progress: Optional[Callable[[str], Any]], signal: Any, asset_options: Any,
) -> MineruExtractedMarkdown:
    api_base = local_mineru_api_base(config.get("localEndpoint"))
    configured_backend = config.get("localBackend") or "hybrid-engine"
    local_server_url = js_trim(config.get("localServerUrl") or "")
    if configured_backend.endswith("http-client") and not local_server_url:
        raise MineruError("MinerU HTTP client backends require a model server URL")
    file_size = await get_file_size(source_path)
    if file_size > MAX_ACCURATE_PARSE_BYTES:
        raise MineruError("MinerU accurate parsing supports files up to 200 MB")
    throw_if_aborted(signal)

    token = config.get("localToken")
    headers = _local_headers(token)
    async with make_async_client(timeout=_HTTP_TIMEOUT, follow_redirects=False) as client:
        server_version: Any = None
        if configured_backend.endswith("-engine"):
            try:
                health = await _abortable(client.get(f"{api_base}/health", headers=headers), signal)
                if _ok(health):
                    server_version = _g(await _json(health), "version")
            except Exception:
                if _is_aborted(signal):
                    raise
        local_backend = local_mineru_backend_for_version(configured_backend, server_version)
        b64 = (await read_file_as_base64(source_path))["base64"]
        file_bytes = decode_base64_to_bytes(b64)

        def flag(key: str) -> str:
            return "true" if config.get(key) is not False else "false"

        form: Dict[str, str] = {
            "lang_list": config.get("localLanguage") or "ch",
            "backend": local_backend,
            "effort": config.get("localEffort") or "medium",
            "parse_method": config.get("localParseMethod") or "auto",
            "formula_enable": flag("localFormulaEnabled"),
            "table_enable": flag("localTableEnabled"),
            "image_analysis": flag("localImageAnalysis"),
            "return_md": "true",
            "return_images": "true" if asset_options else "false",
            "response_format_zip": "false",
        }
        if local_server_url:
            form["server_url"] = local_server_url

        if on_progress:
            await _maybe(on_progress("Uploading to local MinerU..."))
        submit = await _abortable(client.post(
            f"{api_base}/tasks", headers=headers, data=form,
            files={"files": (file_name, file_bytes, "application/pdf")}), signal)
        if not _ok(submit):
            raise MineruError(f"Local MinerU submit failed: HTTP {submit.status_code}: {_text(submit)}")

        submit_data = await _json(submit)
        raw_id = _g(submit_data, "task_id")
        task_id = js_trim(raw_id) if isinstance(raw_id, str) else ""
        if not task_id:
            raise MineruError("Local MinerU returned no task ID")
        encoded = encode_uri_component(task_id)
        # Keep follow-up requests pinned to the user-configured service (no SSRF via response URLs).
        status_url = f"{api_base}/tasks/{encoded}"
        result_url = f"{api_base}/tasks/{encoded}/result"

        if on_progress:
            await _maybe(on_progress("Waiting for local MinerU to finish..."))
        start = time.monotonic()
        while time.monotonic() - start < LOCAL_POLL_TIMEOUT_S:
            throw_if_aborted(signal)
            status_res = await _abortable(client.get(status_url, headers=headers), signal)
            if not _ok(status_res):
                raise MineruError(f"Local MinerU status check failed: HTTP {status_res.status_code}")
            status = await _json(status_res)

            if _g(status, "status") == "completed":
                if on_progress:
                    await _maybe(on_progress("Downloading parsed result..."))
                result_res = await _abortable(client.get(result_url, headers=headers), signal)
                if not _ok(result_res):
                    raise MineruError(f"Local MinerU download failed: HTTP {result_res.status_code}")
                result = await _json(result_res)
                results = _g(result, "results")
                first = next(iter(results.values()), None) if isinstance(results, dict) else None
                md_content = _g(first, "md_content")
                if not first or not isinstance(md_content, str) or not js_trim(md_content):
                    raise MineruError("Local MinerU returned an empty parsing result")
                markdown = convert_html_tables_to_markdown(md_content)
                saved: List[SavedImage] = []
                path_map: Dict[str, str] = {}
                images = _g(first, "images")
                if asset_options and isinstance(images, dict):
                    pp = normalize_path(_opt(asset_options, "projectPath", "project_path"))
                    slug = _opt(asset_options, "sourceSummarySlug", "source_summary_slug")
                    media_dir = f"{pp}/wiki/media/{slug}/mineru/images"
                    await create_directory(media_dir)
                    for raw_name, raw_data in images.items():
                        throw_if_aborted(signal)
                        if not isinstance(raw_data, str):
                            continue
                        m = re.match(r"data:(image/[^;]+);base64,(.+)\Z", raw_data, re.S)
                        if not m:
                            continue
                        source_name = get_file_name(normalize_zip_path(raw_name))
                        if not source_name or not _is_image_path(source_name):
                            continue
                        # The data URI describes the bytes actually written; the server's file name is
                        # untrusted lookup metadata (may carry a mismatched extension).
                        mime_type = m.group(1).lower()
                        extension = _extension_for_mime_type(mime_type)
                        if not extension:
                            continue
                        # Deterministic local names; the original Markdown target keeps a lookup entry.
                        safe_name = f"image-{len(saved) + 1}.{extension}"
                        abs_path = f"{media_dir}/{safe_name}"
                        rel_path = f"media/{slug}/mineru/images/{safe_name}"
                        await write_file_base64(abs_path, m.group(2))
                        saved.append(SavedImage(
                            index=len(saved), mimeType=mime_type, page=None, width=0, height=0,
                            relPath=rel_path, absPath=abs_path, sha256=""))
                        path_map[raw_name] = rel_path
                        path_map[source_name] = rel_path
                if path_map:
                    markdown = rewrite_mineru_markdown_images(markdown, path_map)
                return MineruExtractedMarkdown(markdown, saved)
            if _g(status, "status") == "failed":
                err = _g(status, "error")
                raise MineruError(f"Local MinerU parsing failed: {js_str(err) if err else 'unknown error'}")

            await _wait_for_poll_interval(signal)

    raise MineruError("Local MinerU parsing timed out")


# ----------------------------------------------------------------- public API

async def parse_with_mineru(
    config: Mapping[str, Any], source_path: str, source_url: Optional[str] = None,
    on_progress: Optional[Callable[[str], Any]] = None, signal: Any = None, asset_options: Any = None,
) -> str:
    """Parse a PDF with MinerU and return the Markdown."""
    return (await parse_with_mineru_result(config, source_path, source_url, on_progress, signal,
                                           asset_options)).markdown


async def parse_with_mineru_result(
    config: Mapping[str, Any], source_path: str, source_url: Optional[str] = None,
    on_progress: Optional[Callable[[str], Any]] = None, signal: Any = None, asset_options: Any = None,
) -> MineruExtractedMarkdown:
    throw_if_aborted(signal)

    async def progress(msg: str) -> None:
        if on_progress:
            await _maybe(on_progress(msg))

    if config.get("backend") == "local":
        file_name = get_file_name(source_path) or "document.pdf"
        result = await _parse_with_local_mineru(config, source_path, file_name, on_progress, signal,
                                                asset_options)
        await progress("Done")
        return result

    token = config.get("token")
    if not token:
        raise MineruError("MinerU API token not configured")
    model_version = config.get("modelVersion")
    if model_version != "pipeline" and model_version != "vlm":
        raise MineruError("MinerU PDF parsing supports only pipeline or vlm model versions")

    async with make_async_client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        if source_url:
            await progress("Submitting URL to MinerU...")
            task_id = await _submit_url_task(client, token, source_url, model_version, signal)
            await progress("Waiting for MinerU to finish...")
            zip_url = await _poll_task(client, token, task_id, signal)
        else:
            await progress("Uploading file to MinerU...")
            throw_if_aborted(signal)
            file_size = await get_file_size(source_path)
            if file_size > MAX_ACCURATE_PARSE_BYTES:
                raise MineruError("MinerU accurate parsing supports files up to 200 MB")

            file_name = source_path.split("/")[-1]
            throw_if_aborted(signal)
            b64 = (await read_file_as_base64(source_path))["base64"]
            uploaded = await _upload_file_for_task(client, token, file_name, b64, model_version, signal)
            await progress("Waiting for MinerU to finish...")
            zip_url = await _poll_batch_task(client, token, uploaded["batchId"], signal)

        await progress("Downloading parsed result...")
        result = await _download_and_extract_markdown(client, zip_url, signal, asset_options)
    await progress("Done")
    return result


async def test_mineru_connection(token: str, config: Optional[Mapping[str, Any]] = None) -> None:
    """Check connectivity: cloud submits a minimal task; local checks ``/health``. Raises on failure."""
    if config and config.get("backend") == "local":
        base = local_mineru_api_base(config.get("localEndpoint"))
        async with make_async_client(timeout=_HTTP_TIMEOUT, follow_redirects=False) as client:
            res = await client.get(f"{base}/health", headers=_local_headers(config.get("localToken")))
        if not _ok(res):
            raise MineruError(f"Local MinerU service unavailable: HTTP {res.status_code}: {_text(res)}")
        try:
            health = await _json(res)
        except Exception:
            health = None
        if _g(health, "status") != "healthy":
            raise MineruError("Local MinerU service returned an invalid or unhealthy status")
        return

    async with make_async_client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        res = await client.post(
            f"{API_BASE}/extract/task", headers=_mineru_headers(token),
            content=dumps_js({"url": "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
                              "model_version": "pipeline"}).encode("utf-8"))
    if not _ok(res):
        raise MineruError(f"HTTP {res.status_code}: {_text(res)}")
    assert_mineru_success(await _json(res))


test_mineru_connection.__test__ = False  # type: ignore[attr-defined]  # not a pytest test
