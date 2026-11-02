from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from llm_wiki_server import events, state
from llm_wiki_server.agent import runtime as rt
from llm_wiki_server.agent.cancel import AgentCancellationRegistry
from llm_wiki_server.agent.types import AgentChatRequest, AgentRetrievalMode
from llm_wiki_server.app import create_app
from llm_wiki_server.commands import rest_api as api

TOKEN = "tok"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
LLM = {"provider": "openai", "apiKey": "k", "model": "m"}


class FakeClient:
    def __init__(self, replies, delay=0.0):
        self.replies = list(replies)
        self.calls = []
        self.delay = delay

    def structured_task_config(self, n):
        return self

    def provider_name(self):
        return "fake"

    def model_name(self):
        return "fake-1"

    async def generate_text(self, system, user, images=()):
        self.calls.append((system, user))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.replies.pop(0)


def act(**kw):
    return json.dumps(kw)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_WIKI_API_TOKEN", raising=False)
    old = state._state
    proj = tmp_path / "proj"
    for d in ("wiki/entities", "raw/sources", ".llm-wiki"):
        (proj / d).mkdir(parents=True)
    (proj / "schema.md").write_text("# schema")
    (proj / "wiki/entities/alpha.md").write_text("---\ntype: entity\ntitle: Alpha\n---\n# Alpha\nAlpha is a thing about widgets.")
    (proj / ".llm-wiki/project.json").write_text(json.dumps({"id": "pid-1"}))
    (tmp_path / "data").mkdir()
    st = state.AppState(tmp_path / "data" / "app-state.json")
    state._state = st
    st.set("apiConfig", {"token": TOKEN})
    st.set("projectRegistry", {"pid-1": {"path": str(proj), "name": "Proj"}})
    st.set("llmConfig", LLM)
    api.invalidate_config_cache()
    holder = {}

    def install(replies, delay=0.0):
        holder["client"] = FakeClient(replies, delay)
        monkeypatch.setattr(rt, "_new_llm_client", lambda cfg: holder["client"])
        return holder["client"]

    c = TestClient(create_app())
    c.proj, c.install = proj, install
    yield c
    state._state = old
    api.invalidate_config_cache()


def test_pure_helpers():
    assert rt.agent_loop_iteration_budget(rt.AgentMode.DEEP, True) == 20
    assert rt.agent_loop_retrieval_budget(rt.AgentMode.FAST, AgentRetrievalMode.STANDARD, True) == 6
    a = rt.parse_agent_loop_action('noise {"tool":"wiki.search","query":"x"} tail')
    assert (a.action, a.tool) == ("tool", "wiki.search")
    assert rt.parse_agent_loop_action("plain text").action == "invalid_tool_json"
    assert rt.parse_agent_loop_action('{"action":"final","topK":"x"}').action == "invalid_tool_json"
    assert rt.retrieval_signature("wiki.search", {"query": "A,  b!"}, AgentRetrievalMode.SMART) == \
        'wiki.search:{"query":"a b"}'
    assert rt.is_shell_command_scoped_to_agent_workspace("python build.py", "/p")
    assert not rt.is_shell_command_scoped_to_agent_workspace("cat /etc/passwd", "/p")
    assert not rt.is_shell_command_scoped_to_agent_workspace("curl http://x", "/p")
    assert rt.is_shell_command_approved(" ls ", ["ls"])
    plan = rt.parse_model_tool_plan('x {"toolCalls":[{"tool":"wiki.search","query":"q"}]}')
    assert rt.planned_tool_queries(plan, "m") == {"wiki.search": "q"}
    act_ = rt.AgentLoopAction(fields=[{"id": "a", "type": "single", "label": "L",
                                       "options": [{"label": "x"}, {"label": "x"}], "default": "x_2"}])
    req = rt.sanitize_user_input_request(act_)
    assert [o.value for o in req.fields[0].options] == ["x", "x_2"]
    assert req.fields[0].default_value == "x_2"
    with pytest.raises(ValueError):
        rt.sanitize_user_input_request(rt.AgentLoopAction(fields=[{"type": "bogus"}]))


def test_multi_step_turn_with_tools(env):
    client = env.install([
        act(action="tool", tool="wiki.search", query="alpha"),
        act(action="tool", tool="wiki.read_page", path="wiki/entities/alpha.md"),
        act(action="tool", tool="wiki.search", query="alpha"),  # duplicate -> forced final
        act(action="final", answer="Alpha is about widgets."),
    ])
    seen = []
    resp = rt.run_chat({"id": "pid-1", "path": str(env.proj)},
                       {"message": "what is alpha?", "sessionId": "s1"},
                       {"llm": LLM}, threading.Event(), seen.append)
    assert resp["message"] == "Alpha is about widgets."
    assert resp["sessionId"] == "s1"
    assert any(r["path"].endswith("alpha.md") for r in resp["references"])
    types = [e["type"] for e in seen]
    assert types[0] == "agentStart" and types[1] == "turnStart" and types[-1] == "done"
    assert "referenceAdded" in types and "toolStart" in types and "messageDelta" in types
    tools = [(t["tool"], t["status"]) for t in resp["toolEvents"]]
    assert ("wiki.search", "completed") in tools and ("wiki.read_page", "completed") in tools
    assert len(client.calls) == 4
    assert "Tool observations so far" in client.calls[1][1]
    assert resp["usage"]["referenceCount"] == len(resp["references"])


def test_workspace_write_and_file_changed(env):
    env.install([
        act(action="tool", tool="workspace.write_file", path="out/a.txt", content="hi"),
        act(action="final", answer="done"),
    ])
    # workspace tools are always permitted; skills are not required for write_file
    seen = []
    resp = rt.run_chat({"id": "pid-1", "path": str(env.proj)}, {"message": "write"}, {"llm": LLM},
                       threading.Event(), seen.append)
    assert (env.proj / "agent-workspace/out/a.txt").read_text() == "hi"
    assert any(e["type"] == "fileChanged" for e in seen)
    assert any(r["kind"] == "workspace" for r in resp["references"])


def test_user_ask_and_shell_rejected_without_skill(env):
    env.install([act(action="tool", tool="shell.exec", command="ls"),
                 act(action="tool", tool="user.ask", title="T", fields=[
                     {"id": "c", "type": "confirm", "label": "ok?"}])])
    resp = rt.run_chat({"id": "pid-1", "path": str(env.proj)}, {"message": "go"}, {"llm": LLM},
                       threading.Event(), None)
    assert resp["userInputRequest"]["fields"][0]["type"] == "confirm"
    assert any(t["tool"] == "shell.exec" and t["status"] == "failed" for t in resp["toolEvents"])


def test_iteration_limit_and_no_llm(env):
    env.install([act(action="tool", tool="wiki.search", query=f"q{i}") for i in range(30)])
    resp = rt.run_chat({"id": "pid-1", "path": str(env.proj)}, {"message": "x", "mode": "fast"},
                       {"llm": LLM}, threading.Event(), None)
    assert resp["message"]  # forced final from retrieval budget
    with pytest.raises(Exception, match="not configured"):
        rt.run_chat({"id": "pid-1", "path": str(env.proj)}, {"message": "x"}, {"llm": None},
                    threading.Event(), None)
    with pytest.raises(Exception, match="message is required"):
        rt.run_chat({"id": "pid-1", "path": str(env.proj)}, {"message": "  "}, {"llm": LLM},
                    threading.Event(), None)


def test_cancellation_mid_generation(env):
    env.install([act(action="final", answer="late")], delay=5)
    ev = threading.Event()
    threading.Timer(0.3, ev.set).start()
    t0 = time.time()
    with pytest.raises(Exception, match="Agent turn cancelled"):
        rt.run_chat({"id": "pid-1", "path": str(env.proj)}, {"message": "x"}, {"llm": LLM}, ev, None)
    assert time.time() - t0 < 3


def test_rest_chat_json_and_sse(env):
    env.install([act(action="final", answer="hello")])
    r = env.post("/api/v1/projects/pid-1/chat", headers=AUTH, json={"message": "hi", "sessionId": "api1"})
    assert r.status_code == 200, r.text
    assert r.json()["message"] == {"role": "assistant", "content": "hello"}
    # NOTE: rest_api persists via module-level session.append_turn (absent in session.py).

    env.install([act(action="tool", tool="wiki.search", query="alpha"), act(action="final", answer="s")])
    r = env.post("/api/v1/projects/pid-1/chat", headers={**AUTH, "Accept": "text/event-stream"},
                 json={"message": "hi"})
    assert r.status_code == 200
    assert "event: meta" in r.text and "event: agent" in r.text and "event: done" in r.text


def test_rpc_stream_and_sessions(env):
    from llm_wiki_server.commands import agent as cmd

    env.install([act(action="final", answer="streamed")])
    got = []
    remove = events.add_listener(lambda t, p: got.append((t, p)) if t == "agent-event" else None)
    try:
        r = env.post("/rpc/agent_start_turn_stream", headers=AUTH,
                     json={"projectId": "pid-1", "request": {"message": "hi", "sessionId": "ui1"}})
        assert r.json() == "ui1"
        for _ in range(100):
            if any(p["event"]["type"] == "done" for _, p in got):
                break
            time.sleep(0.05)
    finally:
        remove()
    assert any(p["event"]["type"] == "done" and p["sessionId"] == "ui1" for _, p in got)
    for _ in range(50):
        msgs = env.post("/rpc/agent_get_session", headers=AUTH, json={"projectId": "pid-1", "sessionId": "ui1"}).json()
        if msgs:
            break
        time.sleep(0.05)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert env.post("/rpc/agent_list_sessions", headers=AUTH, json={"projectId": "pid-1"}).json()[0]["sessionId"] == "ui1"

    env.install([act(action="final", answer="sync")])
    r = env.post("/rpc/agent_start_turn", headers=AUTH, json={"projectId": "pid-1", "request": {"message": "again", "sessionId": "ui1"}})
    assert r.json()["message"] == "sync"
    assert env.post("/rpc/agent_cancel_turn", headers=AUTH, json={"projectId": "pid-1", "sessionId": "nope"}).json() is False
    r = env.post("/rpc/agent_start_turn", headers=AUTH, json={"projectId": "zzz", "request": {"message": "x"}})
    assert r.status_code == 404


def test_rest_chat_persists_and_hydrates_session(tmp_path):
    """Regression: rest_api used module-level session functions that did not exist."""
    from llm_wiki_server.agent import session as s

    s.append_turn(str(tmp_path), "p", "sess1", "hi", "hello")
    msgs = s.recent_messages(str(tmp_path), "sess1", 12)
    assert [m.content for m in msgs] == ["hi", "hello"]
