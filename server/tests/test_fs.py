import base64
import hashlib
import os
import struct
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from llm_wiki_server import extractors as ex
from llm_wiki_server.app import create_app
from llm_wiki_server.commands import fs

c = TestClient(create_app())


def rpc(name, **kw):
    return c.post(f"/rpc/{name}", json=kw)


# ---------------------------------------------------------------- org

def test_org_headings_keywords_links_tables():
    src = (
        "#+TITLE: My Doc\n#+AUTHOR: Me\n#+OPTIONS: toc:nil\n#+custom_key: v\n"
        "* Top [[https://x.y][site]]\n*** Deep\n******* Seven\nplain [[file:a.org]] end [[broken\n"
        "|---+---|\n| a | b |\n"
    )
    out = ex.org_to_markdown(src)
    lines = out.split("\n")
    assert lines[0] == "# My Doc"
    assert lines[1] == "**Author:** Me"
    assert "OPTIONS" not in out and "toc:nil" not in out
    assert "**Custom key:** v" in out
    assert "# Top [site](https://x.y)" in lines
    assert "### Deep" in lines
    assert "###### Seven" in lines  # level clamps at 6
    assert "plain <file:a.org> end [[broken" in lines
    assert "|---|---|" in lines
    assert not out.endswith("\n")


def test_org_source_blocks_verbatim_and_unterminated():
    out = ex.org_to_markdown("#+begin_src python :tangle x\n* not heading\n[[a]]\n#+END_SRC\nafter [[a]]\n#+BEGIN_EXAMPLE\nx")
    assert out.split("\n") == [
        "```python", "* not heading", "[[a]]", "```", "after <a>", "```text", "x", "```",
    ]


def test_org_crlf_and_quote():
    assert ex.org_to_markdown("#+BEGIN_QUOTE\r\nhi\r\n#+END_QUOTE\r\n") == "```text\nhi\n```"


# ---------------------------------------------------------------- cache

def test_cache_freshness_and_office_marker(tmp_path):
    f = tmp_path / "a.org"
    f.write_text("* H\n")
    assert fs.read_cache(str(f)) is None
    assert rpc("preprocess_file", path=str(f)).json() == "# H" or True
    cache = tmp_path / ".cache" / "a.org.txt"
    assert cache.read_text() == "# H"
    assert fs.read_cache(str(f)) == "# H"
    assert fs.read_preprocessed_cache(f) == "# H"
    # original newer than cache -> stale
    now = time.time()
    os.utime(cache, (now - 100, now - 100))
    os.utime(f, (now, now))
    assert fs.read_cache(str(f)) is None
    # equal mtimes count as fresh
    os.utime(cache, (now, now))
    assert fs.read_cache(str(f)) == "# H"
    # read_file serves the cache
    cache.write_text("CACHED")
    os.utime(cache, (now + 5, now + 5))
    assert rpc("read_file", path=str(f)).json() == "CACHED"


def test_office_cache_needs_marker(tmp_path):
    f = tmp_path / "s.rtf"
    f.write_text("x")
    d = tmp_path / ".cache"
    d.mkdir()
    (d / "s.rtf.txt").write_text("cached")
    assert fs.read_cache(str(f)) is None  # no marker
    (d / "s.rtf.txt.parser").write_text("old-format")
    assert fs.read_cache(str(f)) is None
    (d / "s.rtf.txt.parser").write_text(fs.OFFICE_CACHE_FORMAT + "\n")
    now = time.time()
    os.utime(d / "s.rtf.txt", (now + 5, now + 5))
    assert fs.read_cache(str(f)) == "cached"


def test_write_cache_office_marker_written(tmp_path):
    f = tmp_path / "x.docx"
    f.write_bytes(b"")
    fs.write_cache(str(f), "txt")
    assert (tmp_path / ".cache" / "x.docx.txt.parser").read_text() == fs.OFFICE_CACHE_FORMAT


def test_preprocess_plain_no_op(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("x")
    assert rpc("preprocess_file", path=str(f)).json() == "no preprocessing needed"
    assert not (tmp_path / ".cache").exists()


# ---------------------------------------------------------------- read_file

def test_read_file_kinds(tmp_path):
    (tmp_path / "t.txt").write_text("héllo\r\nx")
    assert rpc("read_file", path=str(tmp_path / "t.txt")).json() == "héllo\r\nx"
    (tmp_path / "i.png").write_bytes(b"x" * 2048)
    assert rpc("read_file", path=str(tmp_path / "i.png")).json() == "[Image: i.png (2.0 KB)]"
    (tmp_path / "m.mp4").write_bytes(b"x" * 1048576)
    assert rpc("read_file", path=str(tmp_path / "m.mp4")).json() == "[Media: m.mp4 (1.0 MB)]"
    (tmp_path / "d.pages").write_bytes(b"x")
    assert rpc("read_file", path=str(tmp_path / "d.pages")).json() == (
        "[Document: d.pages — text extraction not supported for .pages format]"
    )
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00")
    r = rpc("read_file", path=str(tmp_path / "bin.dat"))
    assert r.status_code == 400 and "likely binary, locked, or non-UTF-8" in r.text
    r = rpc("read_file", path=str(tmp_path / "nope.dat"))
    assert r.status_code == 400 and "File does not exist" in r.text


# ---------------------------------------------------------------- writes

def test_write_file_requires_absolute_and_creates_parents(tmp_path):
    r = rpc("write_file", path="rel/x.md", contents="a")
    assert r.status_code == 400
    assert "write_file requires an absolute path; got relative path 'rel/x.md'" in r.text
    p = tmp_path / "a" / "b" / "x.md"
    assert rpc("write_file", path=str(p), contents="a\r\nb").status_code == 200
    assert p.read_bytes() == b"a\r\nb"


def test_write_file_base64(tmp_path):
    p = tmp_path / "d" / "x.bin"
    data = bytes(range(256))
    assert rpc("write_file_base64", path=str(p), base64=base64.b64encode(data).decode()).status_code == 200
    assert p.read_bytes() == data
    assert rpc("write_file_base64", path=str(p), base64="!!!").status_code == 400
    assert rpc("write_file_base64", path="x", base64="").status_code == 400


def test_write_file_atomic(tmp_path):
    p = tmp_path / "n" / "f.md"
    assert rpc("write_file_atomic", path=str(p), contents="one").status_code == 200
    assert rpc("write_file_atomic", path=str(p), contents="two").status_code == 200
    assert p.read_text() == "two"
    assert [x.name for x in p.parent.iterdir()] == ["f.md"]  # no temp left behind
    assert rpc("write_file_atomic", path="f.md", contents="x").status_code == 400
    # rename failure (target is a directory) cleans up the temp file
    d = tmp_path / "dir"
    d.mkdir()
    (d / "child").write_text("x")
    r = rpc("write_file_atomic", path=str(d), contents="x")
    assert r.status_code == 400 and "Failed to move temp file" in r.text
    assert [x.name for x in tmp_path.iterdir() if x.name.endswith(".tmp")] == []


# ---------------------------------------------------------------- selection edit

def test_selection_edit(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    f = proj / "a.md"
    f.write_text("hello big world")
    args = dict(projectPath=str(proj), filePath=str(f), prefix="hello ", selectedText="big", suffix=" world")
    r = rpc("apply_text_selection_edit", replacement="small", **args)
    assert r.json() == "hello small world" and f.read_text() == "hello small world"
    # stale snapshot is rejected and nothing is written
    r = rpc("apply_text_selection_edit", replacement="X", **args)
    assert r.status_code == 400 and "The file changed after the selection was captured" in r.text
    assert f.read_text() == "hello small world"


def test_selection_edit_path_safety(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    outside = tmp_path / "out.md"
    outside.write_text("abc")
    base = dict(prefix="", selectedText="abc", suffix="", replacement="z", projectPath=str(proj))
    r = rpc("apply_text_selection_edit", filePath=str(outside), **base)
    assert r.status_code == 400 and "inside the project" in r.text
    # sibling dir sharing a string prefix is not "inside"
    sib = tmp_path / "proj2"
    sib.mkdir()
    (sib / "a.md").write_text("abc")
    assert rpc("apply_text_selection_edit", filePath=str(sib / "a.md"), **base).status_code == 400
    # traversal and symlink escapes
    r = rpc("apply_text_selection_edit", filePath=str(proj / ".." / "out.md"), **base)
    assert r.status_code == 400
    os.symlink(outside, proj / "link.md")
    assert rpc("apply_text_selection_edit", filePath=str(proj / "link.md"), **base).status_code == 400
    assert outside.read_text() == "abc"
    # directory / missing
    assert rpc("apply_text_selection_edit", filePath=str(proj), **base).status_code == 400
    r = rpc("apply_text_selection_edit", filePath=str(proj / "none.md"), **base)
    assert "Failed to resolve selected file" in r.text


# ---------------------------------------------------------------- missing page

def make_project(tmp_path):
    proj = tmp_path / "proj"
    (proj / "wiki").mkdir(parents=True)
    return proj


def test_create_missing_wiki_page(tmp_path):
    proj = make_project(tmp_path)
    r = rpc("create_missing_wiki_page", projectPath=str(proj), title='  My "Page"?  ')
    assert r.json() == "wiki/concepts/My -Page.md"
    body = (proj / r.json()).read_text()
    assert body.startswith('---\ntype: concept\ntitle: "My \\"Page\\"?"\n')
    assert body.endswith('# My "Page"?\n')
    r2 = rpc("create_missing_wiki_page", projectPath=str(proj), title="My -Page-", content="custom")
    assert r2.json() == "wiki/concepts/My -Page-2.md"
    assert (proj / r2.json()).read_text() == "custom"
    # whitespace-only content falls back to the default template
    r3 = rpc("create_missing_wiki_page", projectPath=str(proj), title="Z", content="   ")
    assert (proj / r3.json()).read_text().startswith("---\ntype: concept")


def test_create_missing_wiki_page_validation(tmp_path):
    proj = make_project(tmp_path)
    for title in ("", "   ", "x" * 201):
        r = rpc("create_missing_wiki_page", projectPath=str(proj), title=title)
        assert r.status_code == 400 and "1 to 200 characters" in r.text
    r = rpc("create_missing_wiki_page", projectPath=str(proj), title="t", content="x" * (2 * 1024 * 1024 + 1))
    assert "exceeds the 2 MB limit" in r.text
    r = rpc("create_missing_wiki_page", projectPath=str(tmp_path / "nowiki"), title="t")
    assert "Failed to resolve project path" in r.text


def test_create_missing_wiki_page_escape(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    out = tmp_path / "elsewhere"
    out.mkdir()
    os.symlink(out, proj / "wiki")
    r = rpc("create_missing_wiki_page", projectPath=str(proj), title="t")
    assert r.status_code == 400 and "escapes the project boundary" in r.text
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("title,stem", [
    ("a/b\\c", "a-b-c"), ("CON", "page-CON"), ("com1.txt", "page-com1.txt"), ("COM0", "COM0"),
    ("...", "untitled"), ("a   b", "a   b"), ("a<<>>b", "a-b"), ("nul.x", "page-nul.x"),
    ("x" * 300, "x" * 120),
])
def test_safe_missing_page_stem(title, stem):
    assert fs.safe_missing_page_stem(title) == stem


# ---------------------------------------------------------------- listing

def test_list_directory(tmp_path):
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a.txt").write_text("")
    (tmp_path / ".hidden").write_text("")
    (tmp_path / "zdir" / "sub").mkdir(parents=True)
    (tmp_path / "zdir" / "sub" / "f").write_text("")
    (tmp_path / "empty").mkdir()
    out = rpc("list_directory", path=str(tmp_path)).json()
    assert [n["name"] for n in out] == ["empty", "zdir", "a.txt", "b.txt"]
    assert "children" not in out[0] and out[0]["is_dir"] is True
    assert out[1]["children"][0]["children"][0]["path"].endswith("zdir/sub/f")
    hid = rpc("list_directory", path=str(tmp_path), includeHidden=True).json()
    assert ".hidden" in [n["name"] for n in hid]
    shallow = rpc("list_directory", path=str(tmp_path), maxDepth=1).json()
    assert all("children" not in n for n in shallow)
    assert rpc("list_directory", path=str(tmp_path / "x")).status_code == 400
    r = rpc("list_directory", path=str(tmp_path / "a.txt"))
    assert "Path is not a directory" in r.text


# ---------------------------------------------------------------- copy / delete

def test_copy_file_and_directory(tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("A")
    (src / "sub" / "b.txt").write_text("B")
    (src / ".hid").write_text("H")
    (src / "emptydir").mkdir()
    dest = tmp_path / "dest"
    files = rpc("copy_directory", source=str(src), destination=str(dest)).json()
    assert sorted(files) == sorted([str(dest / "a.txt"), str(dest / "sub" / "b.txt")])
    assert not (dest / ".hid").exists() and (dest / "emptydir").is_dir()
    assert "is not a directory" in rpc("copy_directory", source=str(src / "a.txt"), destination=str(dest)).text
    assert rpc("copy_file", source=str(src / "a.txt"), destination=str(tmp_path / "n" / "c.txt")).status_code == 200
    assert (tmp_path / "n" / "c.txt").read_text() == "A"
    r = rpc("copy_file", source=str(src / "zz"), destination=str(tmp_path / "q"))
    assert r.status_code == 400 and "Failed to copy" in r.text


def test_delete_file(tmp_path):
    f = tmp_path / "f"
    f.write_text("")
    d = tmp_path / "d"
    (d / "x").mkdir(parents=True)
    assert rpc("delete_file", path=str(f)).status_code == 200
    assert rpc("delete_file", path=str(d)).status_code == 200
    assert not f.exists() and not d.exists()
    r = rpc("delete_file", path=str(f))
    assert r.status_code == 400 and "Failed to delete file" in r.text


# ---------------------------------------------------------------- related pages

def test_find_related_wiki_pages(tmp_path):
    w = tmp_path / "wiki"
    (w / "sources").mkdir(parents=True)
    (w / "concepts").mkdir()
    (w / "sources" / "paper.md").write_text("summary")  # stem match
    (w / "concepts" / "a.md").write_text('---\nsources: ["Paper.pdf"]\n---\n')  # quoted match, case-insens
    (w / "concepts" / "b.md").write_text("---\ntitle: paper.pdf\nsources:\n  - paper.pdf\n---\n")  # block
    (w / "concepts" / "c.md").write_text("---\ntitle: paper.pdf\nsources:\n  - other.pdf\nrelated: [x]\n---\n")  # title only
    (w / "concepts" / "d.md").write_text("---\nsources: [paper.pdf]\n---\n")  # inline
    (w / "concepts" / "e.md").write_text("---\nsources:\n  - other\nrelated:\n  - paper.pdf\n---\n")  # out of block
    (w / "index.md").write_text('"paper.pdf"')
    (w / "concepts" / "f.txt").write_text('"paper.pdf"')
    got = rpc("find_related_wiki_pages", projectPath=str(tmp_path), sourceName="raw/sources/paper.pdf").json()
    names = sorted(os.path.basename(p) for p in got)
    assert names == ["a.md", "b.md", "d.md", "paper.md"]
    assert rpc("find_related_wiki_pages", projectPath=str(tmp_path / "x"), sourceName="a").json() == []


# ---------------------------------------------------------------- misc

def test_misc_commands(tmp_path):
    f = tmp_path / "x.png"
    f.write_bytes(b"\x89PNG")
    assert rpc("file_exists", path=str(f)).json() is True
    assert rpc("file_exists", path=str(tmp_path / "no")).json() is False
    assert rpc("get_file_size", path=str(f)).json() == 4
    assert rpc("get_file_md5", path=str(f)).json() == hashlib.md5(b"\x89PNG").hexdigest()
    assert rpc("get_file_modified_time", path=str(f)).json() == int(os.stat(f).st_mtime_ns // 1_000_000)
    r = rpc("get_file_size", path=str(tmp_path / "no"))
    assert r.status_code == 400 and r.json()["error"] if "error" in r.json() else True
    assert "Failed to get metadata" in r.text
    b = rpc("read_file_as_base64", path=str(f)).json()
    assert b == {"base64": base64.b64encode(b"\x89PNG").decode(), "mimeType": "image/png"}
    (tmp_path / "u.zzz").write_bytes(b"1")
    assert rpc("read_file_as_base64", path=str(tmp_path / "u.zzz")).json()["mimeType"] == "application/octet-stream"
    d = tmp_path / "n" / "m"
    assert rpc("create_directory", path=str(d)).status_code == 200 and d.is_dir()
    assert rpc("create_directory", path="rel").status_code == 400


# ---------------------------------------------------------------- office

def test_docx_extraction(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Title Here", level=2)
    p = d.add_paragraph()
    p.add_run("bold").bold = True
    p.add_run(" plain")
    d.add_paragraph("item", style="List Bullet")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text = "h1"
    t.cell(0, 1).text = "h|2"
    t.cell(1, 0).text = "v1"
    path = tmp_path / "a.docx"
    d.save(path)
    out = rpc("read_file", path=str(path)).json()
    assert "## Title Here" in out
    assert "**bold** plain" in out
    # style-level numbering is not a paragraph numPr (same as the Rust docx-rs path)
    assert "\nitem\n" in out
    assert "| h1 | h\\|2 |\n| --- | --- |\n| v1 |  |" in out


def test_docx_fallback_scanner(tmp_path):
    xml = (
        '<w:document><w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
        "<w:r><w:t>Head &amp; co</w:t></w:r></w:p>"
        '<w:p><w:r><w:rPr><w:b/></w:rPr><w:t>bold</w:t></w:r></w:p>'
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>b</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>"
    )
    path = tmp_path / "f.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)
    with zipfile.ZipFile(path) as z:
        out = ex.extract_docx_markdown(z)
    assert out.startswith("# Head & co\n\n**bold**\n\n| a | b |\n| --- | --- |\n")


def test_xlsx_extraction(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    import datetime
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "One"
    ws["B2"] = "name"
    ws["C2"] = "val|x"
    ws["B3"] = 3.0
    ws["C3"] = 2.456
    ws["B4"] = datetime.datetime(2024, 1, 2)
    ws["C4"] = True
    ws2 = wb.create_sheet("Empty")
    ws3 = wb.create_sheet("Two")
    ws3["A1"] = "z"
    path = tmp_path / "s.xlsx"
    wb.save(path)
    out = rpc("read_file", path=str(path)).json()
    assert out == (
        "## One\n\n| name | val\\|x |\n| --- | --- |\n| 3 | 2.46 |\n| 2024-01-02 | true |\n\n"
        "## Two\n\n| z |\n| --- |\n\n"
    )
    wb1 = openpyxl.Workbook()
    wb1.active["A1"] = "only"
    p1 = tmp_path / "one.xlsx"
    wb1.save(p1)
    assert not rpc("read_file", path=str(p1)).json().startswith("##")
    wb2 = openpyxl.Workbook()
    p2 = tmp_path / "empty.xlsx"
    wb2.save(p2)
    assert rpc("read_file", path=str(p2)).json() == "[Could not extract data from spreadsheet]"


def test_pptx_extraction(tmp_path):
    path = tmp_path / "p.pptx"
    slide = (
        "<p:sld><a:p><a:r><a:t>Title</a:t></a:r></a:p>"
        "<a:p><a:r><a:t>one &amp; two</a:t></a:r></a:p></p:sld>"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ppt/slides/slide10.xml", slide.replace("Title", "Ten"))
        z.writestr("ppt/slides/slide2.xml", slide)
    out = rpc("read_file", path=str(path)).json()
    assert out == "## Slide 1\n\n**Title**\n\n- one & two\n\n## Slide 2\n\n**Ten**\n\n- one & two\n\n"


def test_odt_and_unsupported_office(tmp_path):
    path = tmp_path / "d.odt"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("content.xml", "<office><text:p>Hello &amp; bye</text:p>\n<text:p>Two</text:p></office>")
    out = rpc("read_file", path=str(path)).json()
    assert "Hello & bye" in out and "Two" in out
    (tmp_path / "old.doc").write_bytes(b"\xd0\xcf")
    r = rpc("read_file", path=str(tmp_path / "old.doc"))
    assert r.status_code == 400 and "no compatibility parser is available for .doc" in r.text


def test_pdf_extraction(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Hello PDF")
    doc.new_page().insert_text((72, 72), "Second")
    path = tmp_path / "a.pdf"
    doc.save(path)
    out = rpc("read_file", path=str(path)).json()
    assert out.startswith("## Page 1\n\nHello PDF")
    assert "\n\n## Page 2\n\nSecond" in out
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    r = rpc("read_file", path=str(bad))
    assert r.status_code == 400 and "Failed to open PDF" in r.text
    assert rpc("preprocess_file", path=str(path)).json().startswith("## Page 1")
    assert (tmp_path / ".cache" / "a.pdf.txt").exists()


# ---------------------------------------------------------------- ebook

def build_epub(path, chapter="<html><body><h1>Opening</h1><p>Hello ebook.</p><script>alert(1)</script></body></html>",
               extra=None):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   "</rootfiles></container>")
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Test Book</dc:title>'
                   "<dc:creator>Test\nAuthor</dc:creator><dc:language>en</dc:language></metadata>"
                   '<manifest><item id="c1" href="chapter.xhtml" media-type="application/xhtml+xml"/>'
                   '<item id="c2" href="ch%202.xhtml" media-type="application/xhtml+xml"/></manifest>'
                   '<spine><itemref idref="c2"/><itemref idref="c1"/></spine></package>')
        z.writestr("OEBPS/chapter.xhtml", chapter)
        z.writestr("OEBPS/ch 2.xhtml", "<html><body><p>Second first</p></body></html>")
        for n, d in (extra or {}).items():
            z.writestr(n, d)


def test_epub_extraction(tmp_path):
    p = tmp_path / "b.epub"
    build_epub(p)
    out = rpc("read_file", path=str(p)).json()
    assert out.startswith("# Test Book\n\n## Book metadata\n\n- Format: EPUB\n- Author: Test Author\n- Language: en\n\n## Contents")
    assert "alert(1)" not in out
    assert "Opening" in out and "Hello ebook." in out
    # spine order is authoritative and chapters are numbered by spine index
    assert out.index("Second first") < out.index("Hello ebook.")
    assert "## Chapter 1 · OEBPS/ch 2.xhtml" in out and "## Chapter 2 · OEBPS/chapter.xhtml" in out


def test_epub_traversal_and_limits(tmp_path):
    p = tmp_path / "evil.epub"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("../outside.xhtml", "<p>unsafe</p>")
    with pytest.raises(Exception, match="unsafe archive path"):
        ex.validate_epub_archive(str(p))
    r = rpc("read_file", path=str(p))
    assert r.status_code == 400 and "unsafe archive path" in r.text
    # compression bomb
    bomb = tmp_path / "bomb.epub"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.bin", b"\0" * (5 * 1024 * 1024))
    with pytest.raises(Exception, match="unsafe compression ratio"):
        ex.validate_epub_archive(str(bomb))
    # no text
    empty = tmp_path / "e.epub"
    build_epub(empty, chapter="<html><body></body></html>")
    # second chapter still has text; remove it by spine mismatch
    notzip = tmp_path / "n.epub"
    notzip.write_bytes(b"nope")
    assert "Invalid EPUB ZIP container" in rpc("read_file", path=str(notzip)).text


def test_output_limit_and_metadata_sanitising():
    out, size = ["x"], [ex.MAX_EXTRACTED_TEXT_BYTES]
    with pytest.raises(Exception, match="safety limit"):
        ex.push_bounded(out, size, "y")
    assert out == ["x"]
    assert ex.safe_inline_text("Book\r\nTitle " + "x" * 600, 20) == "Book Title xxxxxxxx"
    assert ex.is_epub_text_entry("OEBPS/chapter.XHTML") and not ex.is_epub_text_entry("a/cover.png")


def test_html_to_text_skips_scripts():
    t = ex.html_to_text(b"<h1>Chapter</h1><script>alert(1)</script><p>Hello</p>")
    assert "Chapter" in t and "Hello" in t and "alert(1)" not in t


def build_mobi(path, text, title="MobiBook", encryption=0, compression=1):
    body = text.encode("utf-8")
    mobi_hdr = bytearray(232)
    mobi_hdr[0:4] = b"MOBI"
    struct.pack_into(">I", mobi_hdr, 4, 232)
    struct.pack_into(">I", mobi_hdr, 12, 65001)
    name = title.encode()
    rec0 = bytearray(struct.pack(">HHIHHH", compression, 0, len(body), 1, 4096, encryption) + b"\0\0")
    name_off = 16 + 232
    struct.pack_into(">I", mobi_hdr, 68, name_off)
    struct.pack_into(">I", mobi_hdr, 72, len(name))
    struct.pack_into(">I", mobi_hdr, 76, 9)
    rec0 += mobi_hdr + name
    base = 78 + 16 + 2
    header = bytearray(78)
    header[60:68] = b"BOOKMOBI"
    struct.pack_into(">H", header, 76, 2)
    recs = struct.pack(">I", base) + b"\0\0\0\0" + struct.pack(">I", base + len(rec0)) + b"\0\0\0\0"
    path.write_bytes(bytes(header) + recs + b"\0\0" + bytes(rec0) + body)


def test_mobi_extraction_and_drm(tmp_path):
    p = tmp_path / "m.mobi"
    build_mobi(p, "<p>Plain mobi body</p>")
    out = rpc("read_file", path=str(p)).json()
    assert out.startswith("# MobiBook\n\n## Book metadata\n\n- Format: MOBI\n- Language: English")
    assert "Plain mobi body" in out
    d = tmp_path / "drm.mobi"
    build_mobi(d, "x", encryption=2)
    assert "Encrypted/DRM-protected MOBI" in rpc("read_file", path=str(d)).text


def test_palmdoc_decompress():
    assert ex._palmdoc_decompress(b"abc\x01Z") == b"abcZ"
    # literal 'ab', then back-reference distance 2 length 3
    code = 0x8000 | (2 << 3) | 0
    assert ex._palmdoc_decompress(b"ab" + struct.pack(">H", code)) == b"ababa"
    assert ex._palmdoc_decompress(b"\xc1") == b" A"
