"""Port of src/lib/image-caption-pipeline.test.ts (caption_image + fs mocked)."""
from __future__ import annotations

import asyncio
import json

import pytest

from llm_wiki_server.ingest import image_caption_pipeline as icp
from llm_wiki_server.ingest.image_caption_pipeline import (
    caption_cache_key,
    caption_markdown_images,
    find_image_references,
    load_caption_cache,
    sha256_of_base64,
    strip_markdown_image_references,
)

CFG = {"provider": "custom", "apiKey": "", "model": "vl-test", "ollamaUrl": "", "customEndpoint": "http://example/v1",
       "apiMode": "chat_completions", "maxContextSize": 8192}


def run(coro):
    return asyncio.run(coro)


class Env:
    def __init__(self):
        self.files = {}       # path -> text (cache file)
        self.writes = []
        self.bytes_by_path = {}
        self.default_bytes = {"base64": "AAAA", "mimeType": "image/png"}
        self.reads = []
        self.caption_calls = []
        self.caption_impl = lambda b64, *a, **k: "caption"


@pytest.fixture
def env(monkeypatch):
    e = Env()

    async def read_b64(path):
        e.reads.append(path)
        v = e.bytes_by_path.get(path, e.default_bytes)
        if isinstance(v, Exception):
            raise v
        return v

    async def read_file(path):
        return e.files[path]

    async def write_file(path, contents):
        e.writes.append((path, contents))
        e.files[path] = contents

    async def file_exists(path):
        return path in e.files

    async def create_dir(path):
        return None

    async def fake_caption(b64, mime, cfg, signal=None, options=None):
        e.caption_calls.append((b64, mime, cfg, signal, options))
        r = e.caption_impl(b64, mime, cfg, signal, options)
        if asyncio.iscoroutine(r):
            r = await r
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(icp, "read_file_as_base64", read_b64)
    monkeypatch.setattr(icp, "read_file", read_file)
    monkeypatch.setattr(icp, "write_file", write_file)
    monkeypatch.setattr(icp, "file_exists", file_exists)
    monkeypatch.setattr(icp, "create_directory", create_dir)
    monkeypatch.setattr(icp, "caption_image", fake_caption)
    return e


CACHE_PATH = "/proj/.llm-wiki/image-caption-cache.json"


# ------------------------------------------------------------------ findImageReferences

def test_find_refs_with_position_info():
    refs = find_image_references("text\n![](a.png)\n![label](b.jpg) more")
    assert [r.to_dict() for r in refs] == [
        {"full": "![](a.png)", "alt": "", "url": "a.png", "destination": "a.png", "index": 5, "length": 10},
        {"full": "![label](b.jpg)", "alt": "label", "url": "b.jpg", "destination": "b.jpg", "index": 16, "length": 15},
    ]


def test_ignores_links_and_html_img():
    refs = find_image_references("[link](url) <img src=foo.png /> ![real](z.png)")
    assert len(refs) == 1 and refs[0].url == "z.png"


def test_destinations_with_spaces():
    refs = find_image_references("![](assets/figure one.png) ![two](<assets/figure two.jpg>)")
    assert [r.url for r in refs] == ["assets/figure one.png", "assets/figure two.jpg"]


def test_parentheses_titles_encoded_spaces_and_empty():
    refs = find_image_references("\n".join([
        "![](media/Shot (1).png)", "![](<media/Shot (2).png> \"Screenshot\")",
        "![](media/figure%20three.png 'Figure')", "![]( )"]))
    assert [r.url for r in refs] == ["media/Shot (1).png", "media/Shot (2).png", "media/figure three.png"]


def test_trailing_parenthesized_filename_is_not_a_title():
    assert find_image_references("![](media/Shot (1))")[0].url == "media/Shot (1)"


def test_malformed_candidates_stop_at_line():
    md = "![not an image\ntext\n![](real image.png)"
    refs = find_image_references(md)
    assert len(refs) == 1 and refs[0].url == "real image.png" and refs[0].index == md.index("![](")


def test_embeds_and_reference_images_do_not_join_later_links():
    for sample in ["See ![[a.png]] and [docs](https://x.test)", "![logo][1] then [link](https://x.test)",
                   "x![citation needed] see [here](https://x.test)"]:
        assert find_image_references(sample) == []
        assert strip_markdown_image_references(sample) == sample


def test_encoded_spaces_and_escaped_parens():
    refs = find_image_references("![](media/figure%20one.png) ![](media/figure\\)two.png)")
    assert [r.url for r in refs] == ["media/figure one.png", "media/figure)two.png"]


def test_offsets_are_utf16_units():
    refs = find_image_references("\U0001F600![](a.png)")
    assert refs[0].index == 2 and refs[0].length == 10


def test_strip_markdown_image_references():
    assert strip_markdown_image_references('a![](<x y.png> "t")![](z.png)b') == "a  b"
    assert strip_markdown_image_references("![not](image") == "![not](image"


# ------------------------------------------------------------------ captionMarkdownImages

def test_no_images_returns_input(env):
    out = run(caption_markdown_images("/proj", "no images here", CFG))
    assert out.enrichedMarkdown == "no images here" and out.freshCaptions == 0 and out.cachedCaptions == 0
    assert env.caption_calls == []


def test_fresh_caption_rewrites_alt_and_persists_cache(env):
    env.caption_impl = lambda *a: "a red square"
    out = run(caption_markdown_images("/proj", "Before\n\n![](/abs/img-1.png)\n\nAfter", CFG))
    assert out.freshCaptions == 1 and out.cachedCaptions == 0
    assert out.enrichedMarkdown == "Before\n\n![a red square](/abs/img-1.png)\n\nAfter"
    assert len(env.caption_calls) == 1
    opts = env.caption_calls[0][4]
    assert isinstance(opts["contextBefore"], str) and isinstance(opts["contextAfter"], str)
    assert len(env.writes) == 1
    path, contents = env.writes[0]
    assert path == CACHE_PATH
    entries = list(json.loads(contents).values())
    assert len(entries) == 1
    assert entries[0]["caption"] == "a red square" and entries[0]["mimeType"] == "image/png"
    assert entries[0]["model"] == "vl-test"
    assert contents.startswith('{\n  "')  # JSON.stringify(cache, null, 2) layout


def test_destination_with_spaces(env):
    env.caption_impl = lambda *a: "a process diagram"
    out = run(caption_markdown_images("/proj", "![](<assets/process flow.png>)", CFG))
    assert env.reads == ["/proj/wiki/assets/process flow.png"]
    assert out.enrichedMarkdown == "![a process diagram](<assets/process flow.png>)"


def test_dedupes_plain_and_angle_destinations(env):
    env.caption_impl = lambda *a: "same image"
    out = run(caption_markdown_images("/proj", "![](assets/figure one.png)\n![](<assets/figure one.png>)", CFG))
    assert len(env.caption_calls) == 1
    assert out.enrichedMarkdown == "![same image](assets/figure one.png)\n![same image](<assets/figure one.png>)"


def test_forwards_output_language(env):
    run(caption_markdown_images("/proj", "![](/abs/img-1.png)", CFG, {"outputLanguage": "German"}))
    assert env.caption_calls[0][4]["outputLanguage"] == "German"


def test_does_not_reuse_caption_for_other_language(env):
    env.caption_impl = lambda *a: "ein rotes Quadrat"
    h = sha256_of_base64("AAAA")
    env.files[CACHE_PATH] = json.dumps({h: {"caption": "a red square", "mimeType": "image/png", "model": "vl-old",
                                           "capturedAt": "2026-01-01T00:00:00Z"}})
    out = run(caption_markdown_images("/proj", "![](/abs/img-1.png)", CFG, {"outputLanguage": "German"}))
    assert out.freshCaptions == 1 and out.cachedCaptions == 0 and len(env.caption_calls) == 1
    written = json.loads(env.writes[0][1])
    entry = written[caption_cache_key(h, "German")]
    assert entry["caption"] == "ein rotes Quadrat" and entry["imageHash"] == h and entry["outputLanguage"] == "German"
    assert h in written  # legacy entry preserved


def test_reuses_caption_for_same_language(env):
    h = sha256_of_base64("AAAA")
    env.files[CACHE_PATH] = json.dumps({caption_cache_key(h, "German"): {
        "caption": "ein rotes Quadrat", "mimeType": "image/png", "model": "vl-old",
        "capturedAt": "2026-01-01T00:00:00Z", "imageHash": h, "outputLanguage": "German"}})
    out = run(caption_markdown_images("/proj", "![](/abs/img-1.png)", CFG, {"outputLanguage": "german"}))
    assert out.cachedCaptions == 1 and out.freshCaptions == 0 and env.caption_calls == []
    assert "![ein rotes Quadrat]" in out.enrichedMarkdown


def test_dedupes_by_sha256(env):
    env.caption_impl = lambda *a: "the logo"
    out = run(caption_markdown_images("/proj", "![](/a/logo.png) and ![](/b/logo-copy.png)", CFG))
    assert len(env.reads) == 2 and len(env.caption_calls) == 1 and out.freshCaptions == 1
    assert out.enrichedMarkdown == "![the logo](/a/logo.png) and ![the logo](/b/logo-copy.png)"


def test_uses_cached_caption_without_writing(env):
    h = sha256_of_base64("AAAA")
    env.files[CACHE_PATH] = json.dumps({h: {"caption": "previously captioned", "mimeType": "image/png",
                                           "model": "vl-old", "capturedAt": "2026-01-01T00:00:00Z"}})
    out = run(caption_markdown_images("/proj", "![](/abs/x.png)", CFG))
    assert out.cachedCaptions == 1 and out.freshCaptions == 0 and env.caption_calls == []
    assert out.enrichedMarkdown == "![previously captioned](/abs/x.png)"
    assert env.writes == []


def test_sanitizes_caption(env):
    env.caption_impl = lambda *a: "line1\nline2 with ] bracket\\"
    out = run(caption_markdown_images("/proj", "![](/abs/x.png)", CFG))
    assert out.enrichedMarkdown == "![line1 line2 with ) bracket](/abs/x.png)"


def test_batch_continues_after_a_failure(env):
    env.bytes_by_path = {"/abs/a.png": {"base64": "AAAA", "mimeType": "image/png"},
                         "/abs/b.png": {"base64": "BBBB", "mimeType": "image/png"}}
    env.caption_impl = lambda b64, *a: "first" if b64 == "AAAA" else Exception("HTTP 500")
    out = run(caption_markdown_images("/proj", "![](/abs/a.png) ![](/abs/b.png)", CFG))
    assert out.freshCaptions == 1 and out.failed == 1
    assert out.enrichedMarkdown == "![first](/abs/a.png) ![](/abs/b.png)"


def test_unreadable_image_counts_as_failed(env):
    env.bytes_by_path = {"/abs/a.png": Exception("nope")}
    out = run(caption_markdown_images("/proj", "![](/abs/a.png)", CFG))
    assert out.failed == 1 and out.enrichedMarkdown == "![](/abs/a.png)"


def test_should_caption_filter(env):
    env.caption_impl = lambda *a: "c"
    out = run(caption_markdown_images("/proj", "![](/wanted/img.png) ![](https://external.example/foo.png)", CFG,
                                      {"shouldCaption": lambda url: url.startswith("/wanted/")}))
    assert env.reads == ["/wanted/img.png"]
    assert out.enrichedMarkdown == "![c](/wanted/img.png) ![](https://external.example/foo.png)"


def test_url_to_abs_path_hook(env):
    env.caption_impl = lambda *a: "c"
    run(caption_markdown_images("/proj", "![](media/foo/img-1.png)", CFG,
                                {"urlToAbsPath": lambda url: f"/custom/anchor/{url}"}))
    assert env.reads == ["/custom/anchor/media/foo/img-1.png"]


def test_codex_cli_skipped_before_reading(env):
    out = run(caption_markdown_images("/proj", "![](/abs/a.png)\n![](/abs/b.png)", {**CFG, "provider": "codex-cli"}))
    assert out.enrichedMarkdown == "![](/abs/a.png)\n![](/abs/b.png)" and out.failed == 2
    assert env.reads == [] and env.caption_calls == []


def test_forwards_signal(env):
    sentinel = object()
    run(caption_markdown_images("/proj", "![](/abs/x.png)", CFG, {"signal": sentinel}))
    assert env.caption_calls[0][3] is sentinel


def test_corrupt_cache_starts_fresh(env):
    env.files[CACHE_PATH] = "{ this is not valid JSON"
    env.caption_impl = lambda *a: "ok"
    out = run(caption_markdown_images("/proj", "![](/abs/x.png)", CFG))
    assert out.enrichedMarkdown == "![ok](/abs/x.png)" and out.freshCaptions == 1


def test_concurrency_dispatches_in_parallel(env):
    env.bytes_by_path = {"/abs/a.png": {"base64": "AQ==", "mimeType": "image/png"},
                         "/abs/b.png": {"base64": "Ag==", "mimeType": "image/png"},
                         "/abs/c.png": {"base64": "Aw==", "mimeType": "image/png"}}
    state = {"in": 0, "peak": 0}

    async def impl(b64, *a):
        state["in"] += 1
        state["peak"] = max(state["peak"], state["in"])
        await asyncio.sleep(0.02)
        state["in"] -= 1
        return f"cap-{b64}"

    env.caption_impl = impl
    out = run(caption_markdown_images("/proj", "![](/abs/a.png) ![](/abs/b.png) ![](/abs/c.png)", CFG,
                                      {"concurrency": 3}))
    assert out.freshCaptions == 3 and state["peak"] >= 2


def test_on_progress_counts(env):
    env.bytes_by_path = {"/abs/a.png": {"base64": "AQ==", "mimeType": "image/png"},
                         "/abs/b.png": {"base64": "Ag==", "mimeType": "image/png"}}
    progress = []
    run(caption_markdown_images("/proj", "![](/abs/a.png) ![](/abs/b.png)", CFG,
                                {"onProgress": lambda d, t: progress.append((d, t))}))
    assert progress == [(1, 2), (2, 2)]


def test_context_slices(env):
    before = "Figure 3: Quarterly revenue 2024 — preceding text. " * 2
    after = " Following commentary about the chart. " * 2
    run(caption_markdown_images("/proj", f"{before}![](/abs/x.png){after}", CFG))
    opts = env.caption_calls[0][4]
    assert "Figure 3: Quarterly revenue 2024" in opts["contextBefore"]
    assert "Following commentary about the chart" in opts["contextAfter"]
    assert "![](/abs/x.png)" not in opts["contextBefore"] and "![](/abs/x.png)" not in opts["contextAfter"]
    assert len(opts["contextBefore"]) == len(before) and len(opts["contextAfter"]) == len(after)


def test_context_window_is_utf16_units(env):
    run(caption_markdown_images("/proj", "\U0001F600" * 100 + "![](/abs/x.png)", CFG))
    assert len(env.caption_calls[0][4]["contextBefore"]) == 75  # 150 UTF-16 units = 75 emoji


def test_context_clamps_at_boundaries(env):
    run(caption_markdown_images("/proj", "![](/abs/start.png) trailing text only", CFG))
    o = env.caption_calls[0][4]
    assert o["contextBefore"] == "" and o["contextAfter"] == " trailing text only"
    env.caption_calls.clear()
    env.files.clear()
    run(caption_markdown_images("/proj", "leading text only ![](/abs/end.png)", CFG))
    o = env.caption_calls[0][4]
    assert o["contextBefore"] == "leading text only " and o["contextAfter"] == ""


def test_astral_text_survives_round_trip(env):
    env.caption_impl = lambda *a: "c \U0001F600"
    out = run(caption_markdown_images("/proj", "\U0001F600 ![](/abs/x.png) \U0001F680", CFG))
    assert out.enrichedMarkdown == "\U0001F600 ![c \U0001F600](/abs/x.png) \U0001F680"


def test_load_caption_cache_picks_newest_per_image(env):
    h = sha256_of_base64("AAAA")
    german = caption_cache_key(h, "German")
    env.files[CACHE_PATH] = json.dumps({
        german: {"caption": "neue Beschreibung", "mimeType": "image/png", "model": "vl-new",
                 "capturedAt": "2026-08-18T00:00:00Z", "imageHash": h, "outputLanguage": "German"},
        h: {"caption": "old description", "mimeType": "image/png", "model": "vl-old",
            "capturedAt": "2026-01-01T00:00:00Z"}})
    assert run(load_caption_cache("/proj"))[h] == "neue Beschreibung"
    assert run(load_caption_cache("/proj", "German"))[h] == "neue Beschreibung"
    assert run(load_caption_cache("/proj", "French")) == {}


def test_real_filesystem_round_trip(tmp_path, monkeypatch):
    """Without the fs mocks: real cache file under the project is written and re-read."""
    img = tmp_path / "wiki" / "media" / "p"
    img.mkdir(parents=True)
    (img / "a.png").write_bytes(b"\x89PNG-fake")
    calls = []

    async def fake_caption(*a, **k):
        calls.append(a)
        return "cap"

    monkeypatch.setattr(icp, "caption_image", fake_caption)
    out = run(caption_markdown_images(str(tmp_path), "![](media/p/a.png)", CFG))
    assert out.enrichedMarkdown == "![cap](media/p/a.png)" and out.freshCaptions == 1
    assert (tmp_path / ".llm-wiki" / "image-caption-cache.json").exists()
    out2 = run(caption_markdown_images(str(tmp_path), "![](media/p/a.png)", CFG))
    assert out2.cachedCaptions == 1 and len(calls) == 1
