from __future__ import annotations

import pytest

from llm_wiki_server.agent import events as ev
from llm_wiki_server.agent import permissions, router, types, workspace
from llm_wiki_server.agent.cancel import AgentCancellationRegistry
from llm_wiki_server.agent.types import AgentChatRequest, AgentMode, AgentRetrievalMode, AgentSkillMode


def test_chat_request_defaults():
    r = AgentChatRequest.from_dict({"message": "hello", "sessionId": "s1", "topK": 7, "contextFiles": ["wiki/page.md"]})
    assert r.session_id == "s1" and r.run_id is None
    assert r.mode == AgentMode.STANDARD and r.retrieval_mode == AgentRetrievalMode.STANDARD
    assert r.top_k == 7 and r.context_files == ["wiki/page.md"]
    assert r.skill_mode == AgentSkillMode.EXPLICIT
    assert r.tools.wiki and not r.tools.web and not r.tools.anytxt
    assert not r.allow_empty_retrieval and r.persist_session


def test_chat_request_overrides():
    r = AgentChatRequest.from_dict({
        "message": "x", "mode": "local_first", "retrievalMode": "smart", "skillMode": "auto",
        "tools": {"wiki": False, "web": True, "anytxt": True}, "history": [], "historyExplicit": True,
        "allowEmptyRetrieval": True, "persistSession": False,
        "images": [{"mediaType": "image/png", "dataBase64": "AA=="}],
    })
    assert r.mode == AgentMode.LOCAL_FIRST and r.retrieval_mode == AgentRetrievalMode.SMART
    assert r.skill_mode == AgentSkillMode.AUTO
    assert not r.tools.wiki and r.tools.web and r.tools.anytxt
    assert r.history_explicit and r.allow_empty_retrieval and not r.persist_session
    assert r.images[0].media_type == "image/png"
    assert AgentChatRequest.from_dict(r.to_dict()).to_dict() == r.to_dict()


def test_chat_request_rejects_bad_input():
    with pytest.raises(ValueError):
        AgentChatRequest.from_dict({})
    with pytest.raises(ValueError):
        AgentChatRequest.from_dict({"message": "x", "mode": "bogus"})
    with pytest.raises(ValueError):
        AgentChatRequest.from_dict({"message": "x", "topK": -1})


def test_reference_serialization_skips_none():
    ref = types.AgentReference("T", "wiki/a.md", "wiki")
    assert ref.to_dict() == {"title": "T", "path": "wiki/a.md", "kind": "wiki"}
    kc = types.AgentKnowledgeContext(link_count=2, tags=["core"],
                                     latest_version=types.AgentVersionSummary(1, "a", "t"))
    ref = types.AgentReference("T", "p", "wiki", "s", 1.0, kc)
    d = ref.to_dict()
    assert d["knowledgeContext"] == {"tags": ["core"], "linkCount": 2,
                                     "latestVersion": {"timestamp": 1, "author": "a", "tool": "t"}}
    assert types.AgentReference.from_dict(d) == ref


def test_response_serialization():
    resp = types.AgentChatResponse(True, "p", "s", AgentMode.DEEP, "m")
    d = resp.to_dict()
    assert d == {"ok": True, "projectId": "p", "sessionId": "s", "mode": "deep", "message": "m",
                 "references": [], "toolEvents": []}
    resp.events = [ev.AgentEvent.done("s")]
    resp.usage = types.AgentUsage(1, 2, 3, 4)
    d = resp.to_dict()
    assert d["events"] == [{"type": "done", "sessionId": "s"}]
    assert d["usage"]["promptChars"] == 1


def test_user_input_request_roundtrip():
    d = {"requestId": "r", "title": "t", "fields": [
        {"id": "f", "type": "select", "label": "L", "options": [{"label": "a", "value": "b", "recommended": True}],
         "defaultValue": "b"}]}
    r = types.AgentUserInputRequest.from_dict(d)
    assert r.to_dict() == d


def test_events_shapes_and_redaction():
    e = ev.AgentEvent.tool_start("wiki.search", "query")
    assert e == {"type": "toolStart", "tool": "wiki.search", "input": "query"}
    assert ev.AgentEvent.tool_end("x")["output"] is None
    fc = ev.AgentEvent.file_changed("a.md", "workspace.write_file", True, "before")
    assert fc["type"] == "fileChanged" and fc["existedBefore"] is True and fc["previousContent"] == "before"
    red = ev.redact_for_external_api(fc)
    assert "previousContent" not in red and fc["previousContent"] == "before"
    fc.redact_for_external_api()
    assert "previousContent" not in fc
    assert "previousContent" not in ev.AgentEvent.file_changed("a", "t", False)
    assert ev.AgentEvent.agent_start("s") == {"type": "agentStart", "sessionId": "s"}
    assert ev.AgentEvent.turn_start(AgentMode.FAST) == {"type": "turnStart", "mode": "fast"}
    ref = ev.AgentEvent.reference_added(types.AgentReference("T", "p", "k"))
    assert ref["reference"]["title"] == "T"


def test_cancellation_registry():
    reg = AgentCancellationRegistry()
    t = reg.start("p1", "s1", "r1")
    assert not t.is_cancelled()
    assert reg.cancel("p1", "s1", "r1")
    assert t.is_cancelled() and t.event.is_set()
    with pytest.raises(Exception, match="Agent turn cancelled"):
        t.check()
    assert not reg.cancel("p1", "missing")


def test_cancellation_isolation_and_drop():
    reg = AgentCancellationRegistry()
    a = reg.start("p1", "same", "r1")
    b = reg.start("p2", "same", "r1")
    assert reg.cancel("p1", "same", "r1")
    assert a.is_cancelled() and not b.is_cancelled()
    r2 = reg.start("p2", "same", "r2")
    reg.finish("p2", "same", "r1")
    assert reg.cancel("p2", "same", "r2") and r2.is_cancelled()
    assert reg.cancel("p2", "same")  # any run
    with reg.start("p3", "s", "r") as _:
        assert reg.cancel("p3", "s", "r")
    assert not reg.cancel("p3", "s", "r")


def test_cancellation_stale_token_does_not_remove_new_run():
    reg = AgentCancellationRegistry()
    old = reg.start("p", "s", "r")
    new = reg.start("p", "s", "r")
    old.close()
    assert reg.cancel("p", "s", "r") and new.is_cancelled()


def test_permissions():
    p = permissions.PermissionPolicy.api_default()
    C = permissions.AgentCapability
    assert all(p.allows(c) for c in (C.SEARCH_WIKI, C.NETWORK, C.WRITE_WIKI, C.PROCESS))
    assert not p.allows(C.RUN_DEEP_RESEARCH)
    with pytest.raises(Exception, match="Agent capability 'RunDeepResearch' is not allowed"):
        p.require(C.RUN_DEEP_RESEARCH)
    p.require(C.NETWORK)


def test_workspace():
    assert workspace.AGENT_WORKSPACE_DIR == "agent-workspace"
    assert workspace.agent_workspace_display("/tmp/p") == "/tmp/p/agent-workspace"


def test_router():
    d = router.route_query("Search the web for latest policy updates", AgentMode.STANDARD,
                           types.AgentToolOptions(True, True, False))
    assert d.intent == router.QueryIntent.NEEDS_EXTERNAL_SEARCH
    assert not d.should_search_wiki and d.should_hint_web and not d.should_hint_anytxt
    d = router.route_query("你现在有哪些 skill 可以使用？", AgentMode.STANDARD, types.AgentToolOptions())
    assert d.intent == router.QueryIntent.AMBIGUOUS
    assert router.route_query("hello", AgentMode.STANDARD, types.AgentToolOptions()).intent == \
        router.QueryIntent.SIMPLE_CONVERSATIONAL
    d = router.route_query("show raw source", AgentMode.FAST, types.AgentToolOptions())
    assert d.should_include_sources
    assert router.route_query("x", AgentMode.DEEP, types.AgentToolOptions()).should_include_sources
    assert d.to_dict()["intent"] == "needs_raw_source_search" and "shouldHintAnytxt" in d.to_dict()
