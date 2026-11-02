import pytest
from fastapi.testclient import TestClient

from llm_wiki_server.app import create_app
from llm_wiki_server.commands import search as S
from llm_wiki_server.commands import external_search as E

c = TestClient(create_app())


def mk(tmp_path):
    r = c.post("/rpc/create_project", json={"name": "w", "path": str(tmp_path)})
    root = r.json()["path"]
    return root


def page(root, rel, content):
    import os
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p


def test_tokenize_and_title_and_images():
    t = S.tokenize_query("默会知识")
    assert {"默会", "知识", "默"} <= set(t)
    assert S.tokenize_query("What is the Vector DB?") == ["db", "vector"]
    refs = S.extract_image_refs("![a](wiki/media/x.png)\n![b](wiki/media/x.png) ![c](a b.png)")
    assert refs == [{"url": "wiki/media/x.png", "alt": "a"}]
    assert S.extract_title("---\ntitle: Real Title\n---\n\ntitle: Body\n# Heading", "f.md") == "Real Title"
    assert S.extract_title("intro\ntitle: Body\n# Real Heading", "f.md") == "Real Heading"
    assert S.extract_title("plain", "vector-database.md") == "vector database"
    assert S.extract_graph_links("---\nrelated: [foo, \"[[bar]]\"]\n---\n# Page\n[[foo]]") == ["bar", "foo"]
    assert S.extract_graph_links("---\r\nrelated:\r\n  - wiki/entities/alpha.md\r\n  - beta\r\n---\r\n# P") == [
        "wiki/entities/alpha.md", "beta"]


def test_build_snippet():
    s = S.build_snippet("a" * 200 + "needle" + "b" * 200, "needle")
    assert s.startswith("...") and s.endswith("...") and "needle" in s
    assert S.build_snippet("", "x") == ""
    assert S.build_snippet("hello world", "zzz") == "hello world"


def test_search_keyword_and_graph(tmp_path):
    root = mk(tmp_path)
    page(root, "wiki/concepts/transformer.md", "---\ntitle: Transformer\n---\n# Transformer\nAttention is all you need. See [[bert]].\n![d](img/a.png)")
    page(root, "wiki/concepts/bert.md", "# BERT\nA model.")
    page(root, "wiki/concepts/other.md", "# Other\nnothing")
    r = c.post("/rpc/search_project", json={"projectPath": root, "query": "transformer", "includeContent": True})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["mode"] == "hybrid" and j["tokenHits"] == 1 and j["graphHits"] == 1
    first = j["results"][0]
    assert first["path"] == "wiki/concepts/transformer.md" and first["titleMatch"] is True
    assert first["images"] == [{"url": "img/a.png", "alt": "d"}] and "content" in first
    assert "vectorScore" not in first and "graphRelatedTo" not in first
    g = j["results"][1]
    assert g["path"] == "wiki/concepts/bert.md" and g["graphRelatedTo"] == ["Transformer"]
    assert g["snippet"] == "Graph neighbor of Transformer"
    assert c.post("/rpc/search_project", json={"projectPath": root, "query": "  "}).json()["error"] == "query is required"


def test_vector_fusion(tmp_path, monkeypatch):
    import types, sys
    root = mk(tmp_path)
    page(root, "wiki/concepts/alpha.md", "# Alpha\nkeyword zebra here")
    page(root, "wiki/concepts/beta.md", "# Beta\nunrelated")
    fake = types.ModuleType("llm_wiki_server.commands.vectorstore")

    def vector_search_chunks(project_path, emb, top_k):
        return [
            {"chunkId": "concepts/beta#0", "pageId": "concepts/beta", "chunkIndex": 0, "chunkText": "beta text", "headingPath": "H", "score": 0.9},
            {"chunkId": "concepts/beta#1", "pageId": "concepts/beta", "chunkIndex": 1, "chunkText": "x", "headingPath": "", "score": 0.5},
        ]
    fake.vector_search_chunks = vector_search_chunks
    monkeypatch.setitem(sys.modules, "llm_wiki_server.commands.vectorstore", fake)
    import llm_wiki_server.commands as cm
    monkeypatch.setattr(cm, "vectorstore", fake, raising=False)
    r = c.post("/rpc/search_project", json={"projectPath": root, "query": "zebra", "queryEmbedding": [0.1, 0.2]}).json()
    assert r["vectorHits"] == 1 and r["tokenHits"] == 1
    paths = [x["path"] for x in r["results"]]
    assert set(paths) == {"wiki/concepts/alpha.md", "wiki/concepts/beta.md"}
    beta = [x for x in r["results"] if x["path"].endswith("beta.md")][0]
    assert beta["vectorScore"] == 1.0 and beta["snippet"] == "H: beta text"
    bad = c.post("/rpc/search_project", json={"projectPath": root, "query": "zebra", "queryEmbedding": []})
    assert bad.json()["error"] == "queryEmbedding must not be empty"


def test_vector_unavailable_falls_back(tmp_path):
    root = mk(tmp_path)
    page(root, "wiki/concepts/alpha.md", "# Alpha\nzebra")
    r = c.post("/rpc/search_project", json={"projectPath": root, "query": "zebra", "queryEmbedding": [0.1]}).json()
    assert r["vectorHits"] == 0 and r["mode"] == "keyword"


def test_page_links(tmp_path):
    root = mk(tmp_path)
    a = page(root, "wiki/concepts/a.md", "# A\n[[b]] [[ghost]] [[a]]")
    page(root, "wiki/concepts/b.md", "# B\nback to [[a]]")
    r = c.post("/rpc/get_page_links", json={"projectPath": root, "filePath": a}).json()
    assert [e["title"] for e in r["outgoing"]] == ["B"]
    assert r["outgoing"][0] == {"title": "B", "path": "wiki/concepts/b.md", "snippet": None}
    assert r["missing"] == [{"title": "ghost", "path": None, "snippet": None}]
    assert len(r["backlinks"]) == 1 and "[[a]]" in r["backlinks"][0]["snippet"] or "a" in r["backlinks"][0]["snippet"]
    bad = c.post("/rpc/get_page_links", json={"projectPath": root, "filePath": root + "/schema.md"})
    assert bad.json()["error"] == "Page links target must be an existing Markdown file under wiki/"


def cfg(**kw):
    d = dict(enabled=True, endpoint="https://ark.cn-beijing.volces.com/api/v3", apiKey="K", model="m")
    d.update(kw)
    return S.SearchEmbeddingConfig.from_dict(d)


def test_embedding_endpoints():
    assert S._volcengine_embedding_endpoint(cfg(model="doubao-embedding-text")) == "https://ark.cn-beijing.volces.com/api/v3/embeddings"
    assert S._volcengine_embedding_endpoint(cfg(endpoint="https://gateway.example.com/p?upstream=volces.com")) == "https://gateway.example.com/p?upstream=volces.com"
    assert S._volcengine_embedding_endpoint(cfg(endpoint="https://ark.cn-beijing.volces.com/api/v3/embeddings?trace=1", model="doubao-embedding-vision")) == "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal?trace=1"
    assert S._volcengine_embedding_endpoint(cfg(endpoint="https://ARK.cn-beijing.volces.com/api/v3/embeddings")) == "https://ark.cn-beijing.volces.com/api/v3/embeddings"
    assert S._volcengine_embedding_endpoint(cfg(endpoint="https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal")) == "https://ark.cn-beijing.volces.com/api/v3/embeddings"
    g = cfg(endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key=URL_KEY&alt=json", model="gemini-embedding-001")
    ep = S._google_embedding_endpoint(g)
    assert ":embedContent" in ep and "URL_KEY" not in ep and "alt=json" in ep
    b = S._google_embedding_body("gemini-embedding-001", "hi", 768.0)
    assert b["model"] == "models/gemini-embedding-001" and b["output_dimensionality"] == 768
    assert S._is_local_or_private_http_endpoint("http://192.168.1.2:1234/v1")
    assert not S._is_local_or_private_http_endpoint("https://api.openai.com")
    assert S._parse_embedding_values({"data": {"embedding": [0.1, 0.2]}}, False, True) == [0.1, 0.2]
    with pytest.raises(Exception):
        S._parse_embedding_values({"data": [{"embedding": [0.1]}]}, False, True)
    assert S._is_safe_extra_header_name("X-A") and not S._is_safe_extra_header_name("Bad Header")


def test_batch_parse_and_validation():
    out = S._parse_embedding_batch_values({"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}, 2)
    assert out == [[1.0], [2.0]]
    r = c.post("/rpc/embedding_fetch_batch", json={"texts": [], "cfg": dict(enabled=True, endpoint="http://x", apiKey="", model="m")})
    assert r.json()["error"] == "Embedding batch must contain between 1 and 64 inputs"


def test_embedding_fetch_http(monkeypatch):
    import httpx
    seen = {}

    def handler(request):
        seen["h"] = dict(request.headers)
        seen["b"] = request.content
        return httpx.Response(200, json={"data": [{"embedding": [0.25, 0.5]}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(S.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    r = c.post("/rpc/embedding_fetch", json={"text": "hi", "cfg": dict(enabled=True, endpoint="http://localhost:9/v1/embeddings", apiKey="K", model="m", extraHeaders={"X-T": "1", "Authorization": "no"})})
    assert r.json() == [0.25, 0.5]
    assert seen["h"]["authorization"] == "Bearer K" and seen["h"]["origin"] == "http://localhost" and seen["h"]["x-t"] == "1"


def test_oversize_halving(monkeypatch):
    import httpx
    calls = []

    def handler(request):
        calls.append(request.content)
        return httpx.Response(413, text="too long")

    real = httpx.AsyncClient
    monkeypatch.setattr(S.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    r = c.post("/rpc/embedding_fetch", json={"text": "x" * 400, "cfg": dict(enabled=True, endpoint="http://e/v1", apiKey="", model="m"), "maxRetries": 2})
    assert len(calls) == 3 and r.status_code == 400
    assert r.json()["error"].startswith("Endpoint rejected input even at 100 chars.")


def test_external_helpers(monkeypatch):
    assert E.file_url_for_path(r"C:\docs\煤矿 安全.pdf") == "file:///C:/docs/%E7%85%A4%E7%9F%BF%20%E5%AE%89%E5%85%A8.pdf"
    assert E.file_url_for_path("/Users/me/docs/a b.txt") == "file:///Users/me/docs/a%20b.txt"
    assert E.file_url_for_path("anytxt://99") == "anytxt://99"
    assert E.hostname_label("https://www.example.com/x") == "example.com"
    items = E.extract_anytxt_items({"result": {"output": {"fields": ["fid", "path"], "items": [[7, ""]]}}})
    assert items[0]["path"] == "anytxt://7"


def test_web_search_and_anytxt_http(monkeypatch):
    import httpx

    def handler(request):
        u = str(request.url)
        if "searx" in u:
            return httpx.Response(200, json={"results": [
                {"title": "T", "url": "https://www.a.com/p", "content": "snip"}, {"title": "no url"}]})
        body = request.content.decode()
        if "GetFragment" in body:
            return httpx.Response(200, json={"result": {"text": "FRAG"}})
        return httpx.Response(200, json={"result": {"items": [{"fid": 1, "path": "/a/b c.txt", "name": "b c.txt"}]}})

    real = httpx.AsyncClient
    monkeypatch.setattr(E.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    r = c.post("/rpc/web_search", json={"query": "q", "config": {"provider": "searxng", "searXngUrl": "searx.local"}}).json()
    assert r == [{"title": "T", "url": "https://www.a.com/p", "snippet": "snip", "source": "a.com"}]
    r = c.post("/rpc/anytxt_search", json={"query": "q", "config": {}}).json()
    assert r == [{"title": "b c.txt", "url": "file:///a/b%20c.txt", "snippet": "FRAG", "source": "AnyTXT"}]
    assert c.post("/rpc/web_search", json={"query": "q", "config": {"provider": "none"}}).json()["error"] == "Web search provider is not configured."
    assert c.post("/rpc/web_search", json={"query": "q", "config": {"provider": "tavily"}}).json()["error"] == "Tavily web.search requires an API key in Settings."
