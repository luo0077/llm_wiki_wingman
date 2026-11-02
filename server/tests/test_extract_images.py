import base64
import hashlib
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from llm_wiki_server.app import create_app
from llm_wiki_server.commands import extract_images as ei

PIL = pytest.importorskip("PIL.Image")
c = TestClient(create_app())


def png(w, h):
    b = io.BytesIO()
    PIL.new("RGB", (w, h), (200, 10, 10)).save(b, format="PNG")
    return b.getvalue()


def make_pptx(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ppt/presentation.xml", "<x/>")
        z.writestr("ppt/slides/_rels/slide3.xml.rels", '<R Target="../media/image1.png"/><R Target="../slideLayouts/a.xml"/>')
        z.writestr("ppt/media/image1.png", png(120, 130))
        z.writestr("ppt/media/small.png", png(50, 50))
        z.writestr("ppt/media/v.svg", "<svg/>")
        z.writestr("ppt/media/bad.png", b"notanimage")
        z.writestr("ppt/media/image2.png", png(100, 100))


def test_helpers():
    assert ei.is_media_path("ppt/media/a.png") and not ei.is_media_path("docProps/thumbnail.jpeg")
    assert ei.guess_mime_from_name("ppt/media/image1.PNG") == "image/png"
    assert ei.guess_mime_from_name("a/foo.svg") is None
    assert ei.sha256_hex(b"hello world") == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert ei.is_full_page_image_bounds(900, 900, 1000, 1000, 0.9)
    assert not ei.is_full_page_image_bounds(899, 1000, 1000, 1000, 0.9)
    assert ei.is_full_page_image_bounds(500, 500, 1000, 1000, -1.0)
    assert not ei.is_full_page_image_bounds(999, 1000, 1000, 1000, 1.5)
    assert not ei.has_meaningful_page_text(" 12 \n", 80)


def test_office_images(tmp_path):
    p = tmp_path / "d.pptx"
    make_pptx(p)
    r = c.post("/rpc/extract_office_images_cmd", json={"path": str(p)}).json()
    assert [(i["index"], i["page"], i["width"]) for i in r] == [(1, 3, 120), (2, None, 100)]
    assert r[0]["mimeType"] == "image/png"
    raw = base64.b64decode(r[0]["dataBase64"])
    assert hashlib.sha256(raw).hexdigest() == r[0]["sha256"]
    assert set(r[0]) == {"index", "mimeType", "page", "width", "height", "dataBase64", "sha256"}


def test_office_save(tmp_path):
    p = tmp_path / "d.pptx"
    make_pptx(p)
    dest = tmp_path / "wiki/media/d"
    r = c.post("/rpc/extract_and_save_office_images_cmd",
               json={"sourcePath": str(p), "destDir": str(dest), "relTo": str(tmp_path / "wiki")}).json()
    assert r[0]["relPath"] == "media/d/img-1.png" and r[0]["absPath"] == str(dest / "img-1.png")
    assert (dest / "img-2.png").exists()


def test_bad_inputs(tmp_path):
    assert c.post("/rpc/extract_office_images_cmd", json={"path": str(tmp_path / "no")}).status_code == 400
    assert c.post("/rpc/extract_pdf_images_cmd", json={"path": str(tmp_path / "no.pdf")}).status_code == 400


def test_pdf_images(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((72, 72), "hello " * 40)
    page.insert_image(fitz.Rect(100, 200, 300, 400), stream=png(200, 200))
    page.insert_image(fitz.Rect(10, 10, 30, 30), stream=png(20, 20))
    p = tmp_path / "a.pdf"
    doc.save(str(p))
    r = c.post("/rpc/extract_pdf_images_cmd", json={"path": str(p)}).json()
    assert len(r) == 1 and r[0]["page"] == 1 and r[0]["width"] == 200
    assert base64.b64decode(r[0]["dataBase64"])[:4] == b"\x89PNG"
    s = c.post("/rpc/extract_and_save_pdf_images_cmd",
               json={"sourcePath": str(p), "destDir": str(tmp_path / "m"), "relTo": str(tmp_path)}).json()
    assert s[0]["relPath"] == "m/img-1.png" and (tmp_path / "m/img-1.png").exists()
    md = ei.extract_pdf_markdown(str(p), tmp_path / "m2", "/x/", ei.ExtractOptions())
    assert "## Page 1" in md and "![](/x/img-1.png)" in md
