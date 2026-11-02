"""Port of src/lib/extract-source-images.test.ts (+ section builder and real extraction wiring)."""
from __future__ import annotations

import asyncio
import hashlib

import pytest

from llm_wiki_server.ingest import extract_source_images as esi
from llm_wiki_server.ingest.extract_source_images import (
    SavedImage,
    build_image_markdown_section,
    extract_and_save_markdown_images,
    extract_and_save_source_images,
    find_local_markdown_image_refs,
)


def run(coro):
    return asyncio.run(coro)


def test_extracts_obsidian_and_markdown_refs():
    refs = find_local_markdown_image_refs("""
![[attachments/chart.png]]
![Figure](images/plot%201.jpg "title")
![Remote](https://example.com/a.png)
![[attachments/chart.png|400]]
""")
    assert refs == ["attachments/chart.png", "images/plot 1.jpg"]


def test_ignores_non_image_and_remote_refs():
    assert find_local_markdown_image_refs("""
![Doc](notes/page.md)
![Data](data:image/png;base64,abc)
![[draft.txt]]
""") == []


def test_standalone_image_import(tmp_path):
    proj = tmp_path / "project"
    src = proj / "raw" / "sources"
    src.mkdir(parents=True)
    (src / "process flow.JPG").write_bytes(b"image")
    images = run(extract_and_save_source_images(str(proj), str(src / "process flow.JPG"), "process-flow"))
    assert len(images) == 1
    img = images[0]
    dest = proj / "wiki" / "media" / "process-flow" / "001-process flow.JPG"
    assert dest.read_bytes() == b"image"
    assert img.index == 1 and img.mimeType == "image/jpeg" and img.page is None
    assert img.relPath == "media/process-flow/001-process flow.JPG" and img.absPath == str(dest)
    assert img.sha256 == hashlib.sha256(b"image").hexdigest()
    assert img["sha256"] == img.sha256 and img.to_dict()["relPath"] == img.relPath


def test_unsupported_extension_returns_empty(tmp_path):
    assert run(extract_and_save_source_images(str(tmp_path), str(tmp_path / "a.txt"))) == []


def test_pdf_dispatch_and_filtering(monkeypatch, tmp_path):
    seen = {}

    async def fake_invoke(is_pdf, sp, dest_dir, rel_to):
        seen.update(is_pdf=is_pdf, sp=sp, dest=dest_dir, rel=rel_to)
        return [{"index": 1, "mimeType": "image/png", "page": 2, "width": 3, "height": 4,
                 "relPath": "media/p/img-1.png", "absPath": "/x/img-1.png", "sha256": "ab"},
                {"index": "bad"}, None, {"index": 2, "relPath": "r"}]

    monkeypatch.setattr(esi, "_invoke_extract", fake_invoke)
    out = run(extract_and_save_source_images("/p", "/p/raw/paper.pdf"))
    assert seen == {"is_pdf": True, "sp": "/p/raw/paper.pdf", "dest": "/p/wiki/media/paper", "rel": "/p/wiki"}
    assert [i.relPath for i in out] == ["media/p/img-1.png"] and out[0].page == 2


def test_office_dispatch_and_failure_is_swallowed(monkeypatch):
    seen = {}

    async def boom(is_pdf, *a):
        seen["is_pdf"] = is_pdf
        raise RuntimeError("explode")

    monkeypatch.setattr(esi, "_invoke_extract", boom)
    assert run(extract_and_save_source_images("/p", "/p/raw/deck.pptx", "deck")) == []
    assert seen["is_pdf"] is False


def test_real_office_extraction_end_to_end(tmp_path):
    """docx with an embedded PNG goes through the real extract_images command."""
    import io
    import zipfile
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (150, 150), (255, 0, 0)).save(buf, "PNG")
    doc = tmp_path / "raw" / "doc.docx"
    doc.parent.mkdir(parents=True)
    with zipfile.ZipFile(doc, "w") as z:
        z.writestr("word/document.xml", "<w:document/>")
        z.writestr("word/media/image1.png", buf.getvalue())
    images = run(extract_and_save_source_images(str(tmp_path), str(doc)))
    assert len(images) == 1 and images[0].relPath.startswith("media/doc/")
    assert images[0].mimeType == "image/png"
    assert (tmp_path / "wiki" / images[0].relPath).exists()


def test_markdown_image_import(tmp_path):
    src = tmp_path / "raw" / "notes"
    (src / "attachments").mkdir(parents=True)
    (src / "attachments" / "chart.png").write_bytes(b"png")
    md = "![[attachments/chart.png]] ![x](missing.png)"
    images = run(extract_and_save_markdown_images(str(tmp_path), str(src / "n.md"), md))
    assert [i.relPath for i in images] == ["media/n/001-chart.png"]
    assert (tmp_path / "wiki" / "media" / "n" / "001-chart.png").read_bytes() == b"png"
    assert run(extract_and_save_markdown_images(str(tmp_path), str(src / "n.md"), "no images")) == []


def _img(i, page, sha="s", rel=None):
    return SavedImage(i, "image/png", page, 0, 0, rel or f"media/x/{i}.png", f"/abs/{i}", sha)


def test_build_section_empty_and_ordering():
    assert build_image_markdown_section([]) == ""
    imgs = [_img(1, None), _img(2, 10), _img(3, 2, "s3"), _img(4, 2, "s4")]
    out = build_image_markdown_section(imgs, {"s3": "a ] cap\nline"})
    assert out == "\n".join([
        "", "", "## Embedded Images", "",
        "### Page 2", "", "![a ) cap line](media/x/3.png)", "![](media/x/4.png)", "",
        "### Page 10", "", "![](media/x/2.png)", "",
        "### Document", "", "![](media/x/1.png)", "",
    ])


def test_build_section_accepts_plain_dicts():
    out = build_image_markdown_section([{"relPath": "media/x/1.png", "page": 3}, {"relPath": "media/x/2.png", "page": None,
                                                                                  "sha256": "s"}], {"s": "cap"})
    assert "### Page 3\n\n![](media/x/1.png)" in out and "### Document\n\n![cap](media/x/2.png)" in out
