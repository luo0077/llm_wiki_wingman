import math
import os

import pytest
from fastapi.testclient import TestClient

from llm_wiki_server.app import create_app

os.environ["LLM_WIKI_VECTOR_BACKEND"] = "sqlite"
c = TestClient(create_app())


def rpc(name, **kw):
    return c.post("/rpc/" + name, json=kw)


def emb(seed, dim=16):
    return [math.sin((seed * 2654435761 ^ i) % 1000 / 1000.0) for i in range(dim)]


def chunks(page, n, dim=16):
    return [{"chunk_index": i, "chunk_text": f"{page} chunk {i}", "heading_path": f"## Heading {i}",
             "embedding": emb(i, dim)} for i in range(n)]


def test_upsert_count_replace_delete(tmp_path):
    p = str(tmp_path)
    assert rpc("vector_upsert_chunks", projectPath=p, pageId="a", chunks=chunks("a", 5)).status_code == 200
    assert rpc("vector_count_chunks", projectPath=p).json() == 5
    rpc("vector_upsert_chunks", projectPath=p, pageId="a", chunks=chunks("a", 2))
    assert rpc("vector_count_chunks", projectPath=p).json() == 2
    rpc("vector_upsert_chunks", projectPath=p, pageId="b", chunks=chunks("b", 3))
    assert rpc("vector_count_chunks", projectPath=p).json() == 5
    rpc("vector_upsert_chunks", projectPath=p, pageId="a", chunks=[])  # no-op
    assert rpc("vector_count_chunks", projectPath=p).json() == 5
    rpc("vector_delete_page", projectPath=p, pageId="a")
    rpc("vector_delete_page", projectPath=p, pageId="a")
    assert rpc("vector_count_chunks", projectPath=p).json() == 3
    rpc("vector_clear_chunks", projectPath=p)
    rpc("vector_clear_chunks", projectPath=p)
    assert rpc("vector_count_chunks", projectPath=p).json() == 0
    assert rpc("vector_optimize_chunks", projectPath=p).status_code == 200


def test_search_ordering_and_shape(tmp_path):
    p = str(tmp_path)
    assert rpc("vector_search_chunks", projectPath=p, queryEmbedding=emb(1), topK=5).json() == []
    rpc("vector_upsert_chunks", projectPath=p, pageId="page-a", chunks=chunks("page-a", 3))
    r = rpc("vector_search_chunks", projectPath=p, queryEmbedding=emb(1), topK=2).json()
    assert len(r) == 2
    assert r[0]["chunk_id"] == "page-a#1" and r[0]["score"] == 1.0
    assert r[0]["score"] >= r[1]["score"]
    assert set(r[0]) == {"chunk_id", "page_id", "chunk_index", "chunk_text", "heading_path", "score"}
    assert r[0]["heading_path"] == "## Heading 1"


def test_errors(tmp_path):
    p = str(tmp_path)
    bad = chunks("x", 2)
    bad[1]["embedding"] = emb(1, 8)
    r = rpc("vector_upsert_chunks", projectPath=p, pageId="x", chunks=bad)
    assert r.status_code == 400 and "dim" in r.json()["error"].lower()
    assert rpc("vector_upsert_chunks", projectPath=p, pageId="bad'; DROP", chunks=chunks("x", 1)).status_code == 400
    assert rpc("vector_upsert_chunks", projectPath=p, pageId="", chunks=[]).json()["error"] == \
        "Invalid page_id: empty or too long"
    for pid in ["bad\nnl", "zero​width", "a\\b", "﻿bom", "t\U000e0041g"]:
        assert rpc("vector_delete_page", projectPath=p, pageId=pid).status_code == 400
    assert rpc("vector_delete_page", projectPath=p, pageId="concepts/反硝化除磷·A2O：DPAO（测试）").status_code == 200
    assert rpc("vector_delete_page", projectPath=p, pageId="a" * 257).status_code == 400
    assert rpc("vector_delete_page", projectPath=p, pageId="a" * 256).status_code == 200


def test_v1_legacy(tmp_path):
    p = str(tmp_path)
    assert rpc("vector_legacy_row_count", projectPath=p).json() == 0
    rpc("vector_drop_legacy", projectPath=p)
    rpc("vector_upsert", projectPath=p, pageId="old", embedding=emb(0))
    rpc("vector_upsert", projectPath=p, pageId="old", embedding=emb(0))
    rpc("vector_upsert", projectPath=p, pageId="old2", embedding=emb(3))
    assert rpc("vector_count", projectPath=p).json() == 2
    assert rpc("vector_count_chunks", projectPath=p).json() == 0
    hits = rpc("vector_search", projectPath=p, queryEmbedding=emb(0), topK=5).json()
    assert hits[0] == {"page_id": "old", "score": 1.0}
    rpc("vector_upsert_chunks", projectPath=p, pageId="new", chunks=chunks("new", 2))
    rpc("vector_delete", projectPath=p, pageId="old2")
    assert rpc("vector_legacy_row_count", projectPath=p).json() == 1
    rpc("vector_drop_legacy", projectPath=p)
    assert rpc("vector_legacy_row_count", projectPath=p).json() == 0
    assert rpc("vector_count_chunks", projectPath=p).json() == 2


def test_revision_helpers(tmp_path):
    from llm_wiki_server.commands import vectorstore as vs

    p = str(tmp_path)
    assert vs.vector_page_revision_match(p, "a", "fp") is None
    vs.vector_upsert_chunks_with_revision(p, "a", chunks("a", 2), "fp")
    assert vs.vector_page_revision_match(p, "a", "fp") == 2
    assert vs.vector_page_revision_match(p, "a", "other") is None
    vs.vector_upsert_chunks(p, "a", chunks("a", 1))  # plain upsert invalidates
    assert vs.vector_page_revision_match(p, "a", "fp") is None


def test_backend_selection(monkeypatch):
    from llm_wiki_server.commands import vectorstore as vs

    monkeypatch.setenv("LLM_WIKI_VECTOR_BACKEND", "sqlite")
    assert vs.backend_name() == "sqlite"
