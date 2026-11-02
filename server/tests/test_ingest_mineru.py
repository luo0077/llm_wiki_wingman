"""Port of src/lib/mineru.test.ts onto httpx.MockTransport + patched fs boundary, plus zip-safety tests."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import zipfile
from typing import Any, Callable, Dict, List

import httpx
import pytest

from llm_wiki_server.ingest import mineru as mn
from llm_wiki_server.ingest.http_fetch import use_transport
from llm_wiki_server.ingest.llm_client import AbortSignal


def run(coro):
    return asyncio.run(coro)


def b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def zip_bytes(files: Dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


def jr(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=body)


class Net:
    """Sequential mock responder (like mockResolvedValueOnce) recording every request."""

    def __init__(self):
        self.queue: List[Any] = []
        self.calls: List[httpx.Request] = []

    def add(self, *responses):
        self.queue.extend(responses)
        return self

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req)
        if not self.queue:
            raise AssertionError(f"unexpected request {req.method} {req.url}")
        r = self.queue.pop(0)
        return r(req) if callable(r) else r


@pytest.fixture
def net():
    n = Net()
    with use_transport(httpx.MockTransport(n)):
        yield n


class Fs:
    def __init__(self):
        self.created: List[str] = []
        self.written: List[tuple] = []
        self.size = 1024
        self.read_calls: List[str] = []
        self.pdf_b64 = b64("pdf bytes")
        self.write_error = None


@pytest.fixture(autouse=True)
def fs(monkeypatch):
    f = Fs()

    async def create_directory(p):
        f.created.append(p)

    async def get_file_size(p):
        return f.size

    async def read_b64(p):
        f.read_calls.append(p)
        return {"base64": f.pdf_b64, "mimeType": "application/pdf"}

    async def write_b64(p, data):
        if f.write_error:
            e, f.write_error = f.write_error, None
            raise e
        f.written.append((p, data))

    monkeypatch.setattr(mn, "create_directory", create_directory)
    monkeypatch.setattr(mn, "get_file_size", get_file_size)
    monkeypatch.setattr(mn, "read_file_as_base64", read_b64)
    monkeypatch.setattr(mn, "write_file_base64", write_b64)
    monkeypatch.setattr(mn, "POLL_INTERVAL_S", 0.01)
    return f


ASSET = {"projectPath": "/project", "sourceSummarySlug": "paper"}
CLOUD = {"enabled": True, "token": "token", "modelVersion": "vlm"}


def zip_net(net, files):
    net.add(httpx.Response(200, content=zip_bytes(files)))


# ------------------------------------------------------------------ helpers

def test_api_error_messages():
    f = mn.mineru_api_error_message
    assert "invalid" in f("A0202", "bad token")
    assert "expired" in f("A0211", "expired")
    assert "200 MB" in f(-60005, "too large")
    assert "200 page" in f(-60006, "too many pages")
    assert "quota" in f(-60018, "quota")
    assert f(123, "other") == "MinerU API error 123: other"
    assert f(None) == "MinerU API error unknown"
    assert f("A0202") == "MinerU token is invalid. Check the API token in Settings."


def test_prefers_full_md(net):
    zip_net(net, {"result/other.md": "other markdown", "result/full.md": "full markdown"})
    assert run(mn.download_and_extract_markdown("https://cdn/result.zip")) == "full markdown"


def test_falls_back_to_other_md(net):
    zip_net(net, {"result/page.md": "fallback markdown"})
    assert run(mn.download_and_extract_markdown("https://cdn/result.zip")) == "fallback markdown"


def test_rejects_zip_without_markdown(net):
    zip_net(net, {"result/layout.json": "{}"})
    with pytest.raises(Exception, match="No Markdown file"):
        run(mn.download_and_extract_markdown("https://cdn/result.zip"))


def test_html_tables_to_markdown(net):
    zip_net(net, {"full.md": "\n".join(["# Parsed", "<table>", "<tr><th>Name</th><th>Value</th></tr>",
                                        "<tr><td>A&amp;B</td><td>1|2</td></tr>", "</table>"])})
    out = run(mn.download_and_extract_markdown("https://cdn/result.zip"))
    assert "| Name | Value |\n| --- | --- |\n| A&B | 1\\|2 |" in out


def test_malformed_numeric_entities(net):
    zip_net(net, {"full.md": "\n".join([
        "<table>", "<tr><td>&#65;</td><td>&#9999999999;</td><td>&#x41;</td><td>&#xFFFFFFF;</td></tr>", "</table>"])})
    assert "| A | &#9999999999; | A | &#xFFFFFFF; |" in run(mn.download_and_extract_markdown("https://cdn/result.zip"))


def test_tables_in_code_fences_are_untouched(net):
    code = "\n".join(["```html", "<table><tr><td>Keep raw</td></tr></table>", "```"])
    zip_net(net, {"full.md": f"{code}\n\n<table><tr><td>Convert me</td></tr></table>"})
    md = run(mn.download_and_extract_markdown("https://cdn/result.zip"))
    assert code in md and "| Convert me |" in md


def test_images_inside_table_cells(net):
    zip_net(net, {"full.md": "\n".join(["<table>", "<tr><th>Figure</th><th>Note</th></tr>",
                                        '<tr><td><img src="images/chart.png" alt="Chart"></td><td>A</td></tr>',
                                        "</table>"]),
                  "images/chart.png": "chart-bytes"})
    md = run(mn.download_and_extract_markdown("https://cdn/result.zip", None, ASSET))
    assert "| ![Chart](media/paper/mineru/images/chart.png) | A |" in md


def test_extracts_images_and_rewrites_refs(net, fs):
    zip_net(net, {"full.md": "\n".join(["# Parsed", "![Chart](images/chart.png)",
                                        '<img src="figures/table 1.jpg" alt="Table">',
                                        "![Remote](https://example.test/x.png)"]),
                  "images/chart.png": "chart-bytes", "figures/table 1.jpg": "table-bytes"})
    md = run(mn.download_and_extract_markdown("https://cdn/result.zip", None, ASSET))
    assert "/project/wiki/media/paper/mineru" in fs.created
    assert ("/project/wiki/media/paper/mineru/images/chart.png", b64("chart-bytes")) in fs.written
    assert ("/project/wiki/media/paper/mineru/figures/table 1.jpg", b64("table-bytes")) in fs.written
    assert "![Chart](media/paper/mineru/images/chart.png)" in md
    assert "![Table](media/paper/mineru/figures/table%201.jpg)" in md
    assert "![Remote](https://example.test/x.png)" in md


def test_saved_image_metadata(net):
    zip_net(net, {"full.md": "![Chart](images/chart.png)", "images/chart.png": "chart-bytes"})
    r = run(mn.download_and_extract_markdown_result("https://cdn/result.zip", None, ASSET))
    assert r.markdown == "![Chart](media/paper/mineru/images/chart.png)"
    assert len(r.savedImages) == 1
    img = r.savedImages[0]
    assert (img.index, img.mimeType, img.page, img.width, img.height) == (1, "image/png", None, 0, 0)
    assert img.relPath == "media/paper/mineru/images/chart.png"
    assert img.absPath == "/project/wiki/media/paper/mineru/images/chart.png"
    assert re.fullmatch(r"[0-9a-f]{64}", img.sha256)


def test_paths_with_spaces_and_parentheses(net):
    zip_net(net, {"full.md": "![Wide chart](images/wide chart.png)", "images/wide chart.png": "x"})
    assert run(mn.download_and_extract_markdown("https://cdn/r.zip", None, ASSET)) == \
        "![Wide chart](media/paper/mineru/images/wide%20chart.png)"
    zip_net(net, {"full.md": "![Chart](images/chart(1).png)", "images/chart(1).png": "x"})
    assert run(mn.download_and_extract_markdown("https://cdn/r.zip", None, ASSET)) == \
        "![Chart](media/paper/mineru/images/chart%281%29.png)"


def test_large_image_exact_base64(net, fs):
    data = "x" * 40_000
    zip_net(net, {"full.md": "![Large](images/large.png)", "images/large.png": data})
    run(mn.download_and_extract_markdown("https://cdn/r.zip", None, ASSET))
    assert ("/project/wiki/media/paper/mineru/images/large.png", b64(data)) in fs.written


def test_basename_fallback(net):
    zip_net(net, {"result/full.md": "![Chart](chart.png)", "result/images/chart.png": "x"})
    assert run(mn.download_and_extract_markdown("https://cdn/r.zip", None, ASSET)) == \
        "![Chart](media/paper/mineru/result/images/chart.png)"


def test_zip_slip_paths_stay_inside_media_dir(net, fs):
    zip_net(net, {"full.md": "![Evil](evil.png)", "../../evil.png": "evil-bytes"})
    md = run(mn.download_and_extract_markdown("https://cdn/r.zip", None, ASSET))
    assert fs.written == [("/project/wiki/media/paper/mineru/evil.png", b64("evil-bytes"))]
    assert md == "![Evil](media/paper/mineru/evil.png)"


def test_no_basename_fallback_on_collision(net):
    zip_net(net, {"full.md": "![Ambiguous](chart.png)\n![A](a/chart.png)", "a/chart.png": "a", "b/chart.png": "b"})
    md = run(mn.download_and_extract_markdown("https://cdn/r.zip", None, ASSET))
    assert "![Ambiguous](chart.png)" in md and "![A](media/paper/mineru/a/chart.png)" in md


def test_image_save_failure_keeps_markdown(net, fs):
    fs.write_error = OSError("disk full")
    zip_net(net, {"full.md": "![Chart](images/chart.png)\nBody", "images/chart.png": "x"})
    assert run(mn.download_and_extract_markdown("https://cdn/r.zip", None, ASSET)) == \
        "![Chart](images/chart.png)\nBody"


def test_external_img_tags_untouched(net):
    zip_net(net, {"full.md": '<img src="https://example.test/x.png" alt="Remote">', "images/local.png": "x"})
    assert run(mn.download_and_extract_markdown("https://cdn/r.zip", None, ASSET)) == \
        '<img src="https://example.test/x.png" alt="Remote">'


# ------------------------------------------------------------------ zip safety

def test_zip_bomb_entry_is_rejected(net, monkeypatch):
    monkeypatch.setattr(mn, "MAX_ZIP_MARKDOWN_BYTES", 1000)
    zip_net(net, {"full.md": "a" * 5000})
    with pytest.raises(Exception, match="zip bomb"):
        run(mn.download_and_extract_markdown("https://cdn/r.zip"))


def test_total_uncompressed_budget(net, monkeypatch):
    monkeypatch.setattr(mn, "MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES", 150)
    zip_net(net, {"full.md": "![a](a.png) ![b](b.png)", "a.png": "x" * 100, "b.png": "y" * 100})
    # budget exceeded while saving images -> image saving is abandoned, Markdown is kept unchanged
    assert run(mn.download_and_extract_markdown("https://cdn/r.zip", None, ASSET)) == "![a](a.png) ![b](b.png)"


def test_too_many_entries(net, monkeypatch):
    monkeypatch.setattr(mn, "MAX_ZIP_ENTRIES", 2)
    zip_net(net, {"full.md": "x", "a.png": "1", "b.png": "2"})
    with pytest.raises(Exception, match="too many files"):
        run(mn.download_and_extract_markdown("https://cdn/r.zip"))


def test_download_size_cap(net, monkeypatch):
    monkeypatch.setattr(mn, "MAX_ZIP_DOWNLOAD_BYTES", 10)
    net.add(httpx.Response(200, content=b"x" * 100))
    with pytest.raises(Exception, match="maximum allowed download size"):
        run(mn.download_and_extract_markdown("https://cdn/r.zip"))


def test_invalid_zip_and_bad_scheme(net):
    net.add(httpx.Response(200, content=b"not a zip"))
    with pytest.raises(Exception, match="not a valid zip"):
        run(mn.download_and_extract_markdown("https://cdn/r.zip"))
    with pytest.raises(Exception, match="invalid result URL"):
        run(mn.download_and_extract_markdown("file:///etc/passwd"))
    assert len(net.calls) == 1


def test_http_error_on_zip_download(net):
    net.add(httpx.Response(404))
    with pytest.raises(Exception, match="zip download failed: HTTP 404"):
        run(mn.download_and_extract_markdown("https://cdn/r.zip"))


def test_asset_segment_sanitizing():
    f = mn._safe_asset_segment
    assert f("a:b?.png") == "a_b_.png"
    assert f("...") == "_"
    assert f("name. ") == "name_"
    assert f("") == "asset"
    long = "x" * 100 + ".png"
    assert len(f(long)) == 80 and f(long).endswith(".png")
    assert mn.normalize_zip_path("./a/../b/./c.png") == "a/b/c.png"


# ------------------------------------------------------------------ parse_with_mineru

def form_field(req: httpx.Request, name: str) -> str:
    m = re.search(rb'name="' + name.encode() + rb'"\r\n\r\n(.*?)\r\n--', req.content, re.S)
    assert m, f"field {name} missing"
    return m.group(1).decode()


def test_local_windows_path_custom_endpoint_pipeline(net, fs):
    net.add(jr({"task_id": "task/1", "status_url": "x", "result_url": "y"}, 202),
            jr({"status": "completed"}),
            jr({"results": {"report": {"md_content": "# Parsed locally"}}}))
    cfg = {"enabled": True, "backend": "local", "localEndpoint": "http://localhost:9000/custom/",
           "localToken": "local-secret", "localBackend": "pipeline", "token": "", "modelVersion": "pipeline"}
    assert run(mn.parse_with_mineru(cfg, "C:\\documents\\report.pdf")) == "# Parsed locally"
    c = net.calls
    assert str(c[0].url) == "http://localhost:9000/custom/tasks" and c[0].method == "POST"
    assert form_field(c[0], "backend") == "pipeline" and form_field(c[0], "return_md") == "true"
    assert b'filename="report.pdf"' in c[0].content and b"pdf bytes" in c[0].content
    assert "/tasks/task%2F1" in str(c[1].url) and str(c[2].url).endswith("/tasks/task%2F1/result")
    for r in c:
        assert r.headers["authorization"] == "Bearer local-secret"


@pytest.mark.parametrize("version,expected", [("2.1.11", "hybrid-engine"), ("3.1.15", "hybrid-auto-engine"),
                                              ("3.4.4", "hybrid-engine")])
def test_local_backend_name_by_version(net, version, expected):
    net.add(jr({"status": "healthy", "version": version}), jr({"task_id": "task-1"}, 202), jr({"status": "completed"}),
            jr({"results": {"report": {"md_content": "# Parsed locally"}}}))
    cfg = {"enabled": True, "backend": "local", "localBackend": "hybrid-engine", "token": "", "modelVersion": "vlm"}
    assert run(mn.parse_with_mineru(cfg, "/tmp/report.pdf")) == "# Parsed locally"
    assert str(net.calls[0].url) == "http://127.0.0.1:8000/health"
    assert form_field(net.calls[1], "backend") == expected


def test_local_oversized_file_rejected_before_network(net, fs):
    fs.size = mn.MAX_ACCURATE_PARSE_BYTES + 1
    with pytest.raises(Exception, match="200 MB"):
        run(mn.parse_with_mineru({"enabled": True, "backend": "local", "token": "", "modelVersion": "vlm"}, "/tmp/l.pdf"))
    assert fs.read_calls == [] and net.calls == []


def test_local_empty_result_rejected(net):
    net.add(jr({"status": "healthy", "version": "3.1.15"}), jr({"task_id": "task-1"}, 202), jr({"status": "completed"}),
            jr({"results": {"doc": {"md_content": "  "}}}))
    with pytest.raises(Exception, match="empty parsing result"):
        run(mn.parse_with_mineru({"enabled": True, "backend": "local", "token": "", "modelVersion": "vlm"}, "/tmp/doc.pdf"))


def _local_image_net(net, name, mime):
    net.add(jr({"status": "healthy", "version": "3.1.15"}), jr({"task_id": "task-1"}, 202), jr({"status": "completed"}),
            jr({"results": {"doc": {"md_content": f"![Chart](images/{name})",
                                    "images": {name: f"data:{mime};base64,{b64('image bytes')}"}}}}))


def test_local_images_saved_and_rewritten(net, fs):
    _local_image_net(net, "chart.png", "image/png")
    cfg = {"enabled": True, "backend": "local", "token": "", "modelVersion": "vlm", "localBackend": "hybrid-engine"}
    r = run(mn.parse_with_mineru_result(cfg, "/tmp/doc.pdf", None, None, None,
                                        {"projectPath": "/project", "sourceSummarySlug": "doc"}))
    assert r.markdown == "![Chart](media/doc/mineru/images/image-1.png)"
    assert r.savedImages[0].relPath == "media/doc/mineru/images/image-1.png"
    assert ("/project/wiki/media/doc/mineru/images/image-1.png", b64("image bytes")) in fs.written
    assert form_field(net.calls[1], "return_images") == "true"


def test_local_uses_data_uri_mime_when_extension_disagrees(net):
    _local_image_net(net, "chart.jpg", "image/png")
    cfg = {"enabled": True, "backend": "local", "token": "", "modelVersion": "vlm", "localBackend": "hybrid-engine"}
    r = run(mn.parse_with_mineru_result(cfg, "/tmp/doc.pdf", None, None, None,
                                        {"projectPath": "/project", "sourceSummarySlug": "doc"}))
    assert r.markdown == "![Chart](media/doc/mineru/images/image-1.png)"
    assert r.savedImages[0].mimeType == "image/png"


def test_http_client_backend_needs_server_url(net):
    with pytest.raises(Exception, match="require a model server URL"):
        run(mn.parse_with_mineru({"enabled": True, "backend": "local", "token": "", "modelVersion": "vlm",
                                  "localBackend": "vlm-http-client", "localServerUrl": ""}, "/tmp/doc.pdf"))
    assert net.calls == []


def test_endpoint_validation(net):
    with pytest.raises(Exception, match=r"HTTP\(S\)"):
        run(mn.parse_with_mineru({"enabled": True, "backend": "local", "localEndpoint": "file:///tmp/mineru",
                                  "token": "", "modelVersion": "pipeline"}, "/tmp/doc.pdf"))
    with pytest.raises(Exception, match="without credentials"):
        run(mn.test_mineru_connection("", {"backend": "local", "localEndpoint": "http://user:pass@localhost:8000"}))
    assert net.calls == []


def test_unsupported_model_version(net, fs):
    with pytest.raises(Exception, match="pipeline or vlm"):
        run(mn.parse_with_mineru({"enabled": True, "token": "token", "modelVersion": "mineru-html"}, "/tmp/doc.pdf"))
    assert net.calls == []


def test_cloud_oversized_and_missing_token(net, fs):
    fs.size = mn.MAX_ACCURATE_PARSE_BYTES + 1
    with pytest.raises(Exception, match="200 MB"):
        run(mn.parse_with_mineru(CLOUD, "/tmp/large.pdf"))
    assert net.calls == [] and fs.read_calls == []
    with pytest.raises(Exception, match="token not configured"):
        run(mn.parse_with_mineru({"enabled": True, "token": "", "modelVersion": "vlm"}, "/tmp/doc.pdf"))


def test_pre_aborted_signal(net):
    sig = AbortSignal()
    sig.abort()
    with pytest.raises(Exception, match="cancelled"):
        run(mn.parse_with_mineru(CLOUD, "/tmp/doc.pdf", None, None, sig))
    assert net.calls == []


def batch_status(state, **extra):
    return jr({"code": 0, "msg": "ok", "data": {"batch_id": "batch-1",
                                                "extract_result": [{"file_name": "doc.pdf", "state": state, **extra}]}})


UPLOAD_OK = jr({"code": 0, "msg": "ok", "data": {"batch_id": "batch-1", "file_urls": ["https://upload"]}})


def test_batch_without_upload_url(net):
    net.add(jr({"code": 0, "msg": "ok", "data": {"batch_id": "batch-1", "file_urls": []}}))
    with pytest.raises(Exception, match="upload URL"):
        run(mn.parse_with_mineru(CLOUD, "/tmp/doc.pdf"))


def test_uploads_decoded_pdf_bytes(net, fs):
    fs.pdf_b64 = b64("custom pdf bytes")
    net.add(UPLOAD_OK, httpx.Response(200), batch_status("done", full_zip_url="https://zip"),
            httpx.Response(200, content=zip_bytes({"full.md": "parsed markdown"})))
    assert run(mn.parse_with_mineru(CLOUD, "/tmp/doc.pdf")) == "parsed markdown"
    put = net.calls[1]
    assert put.method == "PUT" and str(put.url) == "https://upload" and put.content == b"custom pdf bytes"
    first = net.calls[0]
    assert str(first.url) == "https://mineru.net/api/v4/file-urls/batch"
    assert first.headers["authorization"] == "Bearer token"
    assert json.loads(first.content) == {"files": [{"name": "doc.pdf", "data_id": "doc.pdf"}], "model_version": "vlm"}


def test_asset_options_pass_through_cloud(net, fs):
    net.add(UPLOAD_OK, httpx.Response(200), batch_status("done", full_zip_url="https://zip"),
            httpx.Response(200, content=zip_bytes({"full.md": "![Chart](images/chart.png)", "images/chart.png": "chart-bytes"})))
    r = run(mn.parse_with_mineru_result(CLOUD, "/tmp/doc.pdf", None, None, None,
                                        {"projectPath": "/project", "sourceSummarySlug": "doc"}))
    assert r.markdown == "![Chart](media/doc/mineru/images/chart.png)"
    assert [i.relPath for i in r.savedImages] == ["media/doc/mineru/images/chart.png"]
    assert ("/project/wiki/media/doc/mineru/images/chart.png", b64("chart-bytes")) in fs.written


def test_url_task_flow(net, fs):
    net.add(jr({"code": "0", "msg": "ok", "data": {"task_id": "task-1"}}),
            jr({"code": 0, "msg": "ok", "data": {"task_id": "task-1", "state": "done", "full_zip_url": "https://zip"}}),
            httpx.Response(200, content=zip_bytes({"full.md": "url markdown"})))
    cfg = {"enabled": True, "token": "token", "modelVersion": "pipeline"}
    assert run(mn.parse_with_mineru(cfg, "/tmp/doc.pdf", "https://example.test/doc.pdf")) == "url markdown"
    assert fs.read_calls == []
    assert json.loads(net.calls[0].content) == {"url": "https://example.test/doc.pdf", "model_version": "pipeline"}
    assert str(net.calls[1].url) == "https://mineru.net/api/v4/extract/task/task-1"


def test_failed_state_reports_service_message(net):
    net.add(UPLOAD_OK, httpx.Response(200), batch_status("failed", err_msg="parse exploded"))
    with pytest.raises(Exception, match="parse exploded"):
        run(mn.parse_with_mineru(CLOUD, "/tmp/doc.pdf"))


def test_api_error_code_is_mapped(net):
    net.add(jr({"code": "A0202", "msg": "bad", "data": {}}))
    with pytest.raises(Exception, match="token is invalid"):
        run(mn.parse_with_mineru(CLOUD, "/tmp/doc.pdf"))


def test_http_failures_name_the_step(net):
    net.add(httpx.Response(500))
    with pytest.raises(Exception, match="batch submit failed: HTTP 500"):
        run(mn.parse_with_mineru(CLOUD, "/tmp/doc.pdf"))
    net.add(UPLOAD_OK, httpx.Response(403))
    with pytest.raises(Exception, match="file upload failed: HTTP 403"):
        run(mn.parse_with_mineru(CLOUD, "/tmp/doc.pdf"))


def test_abort_during_poll_interval(net, monkeypatch):
    monkeypatch.setattr(mn, "POLL_INTERVAL_S", 30.0)
    net.add(UPLOAD_OK, httpx.Response(200), batch_status("running"))
    sig = AbortSignal()

    async def main():
        task = asyncio.ensure_future(mn.parse_with_mineru(CLOUD, "/tmp/doc.pdf", None, None, sig))
        await asyncio.sleep(0.2)
        sig.abort()
        with pytest.raises(Exception, match="cancelled"):
            await task

    run(main())
    assert len(net.calls) == 3


def test_all_official_states_before_completion(net):
    net.add(UPLOAD_OK, httpx.Response(200), batch_status("pending"), batch_status("waiting-file"),
            batch_status("converting"), batch_status("running"), batch_status("done", full_zip_url="https://zip"),
            httpx.Response(200, content=zip_bytes({"full.md": "parsed markdown"})))
    progress: List[str] = []
    assert run(mn.parse_with_mineru(CLOUD, "/tmp/doc.pdf", None, progress.append)) == "parsed markdown"
    assert "Waiting for MinerU to finish..." in progress and progress[-1] == "Done"
    assert len(net.calls) == 8


# ------------------------------------------------------------------ test_mineru_connection

def test_connection_local_health(net):
    net.add(jr({"status": "healthy"}))
    assert run(mn.test_mineru_connection("", {"backend": "local", "localToken": "health-secret"})) is None
    req = net.calls[0]
    assert str(req.url) == "http://127.0.0.1:8000/health" and req.headers["authorization"] == "Bearer health-secret"


def test_connection_local_unhealthy(net):
    net.add(jr({"status": "unhealthy"}))
    with pytest.raises(Exception, match="invalid or unhealthy"):
        run(mn.test_mineru_connection("", {"backend": "local"}))
    net.add(httpx.Response(503, text="down"))
    with pytest.raises(Exception, match="unavailable: HTTP 503: down"):
        run(mn.test_mineru_connection("", {"backend": "local"}))


def test_connection_cloud(net):
    net.add(jr({"code": "0", "msg": "ok", "data": {"task_id": "task-1"}}))
    assert run(mn.test_mineru_connection("token")) is None
    body = json.loads(net.calls[0].content)
    assert body["model_version"] == "pipeline" and net.calls[0].headers["authorization"] == "Bearer token"
    net.add(httpx.Response(502, text="bad gateway"))
    with pytest.raises(Exception, match="HTTP 502: bad gateway"):
        run(mn.test_mineru_connection("token"))
    net.add(jr({"code": "A0202", "msg": "token invalid", "data": {}}))
    with pytest.raises(Exception, match="invalid"):
        run(mn.test_mineru_connection("bad-token"))
