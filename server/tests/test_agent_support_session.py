from __future__ import annotations

from llm_wiki_server.agent import session as sess
from llm_wiki_server.agent.session import AgentSessionStore


def test_append_turn_tracks_recent_messages(tmp_path):
    p = str(tmp_path)
    s = AgentSessionStore()
    s.append_turn(p, "p1", "s1", "hello", "hi")
    s.append_turn(p, "p1", "s1", "question", "answer")
    m = s.recent_messages(p, "s1", 3)
    assert len(m) == 3 and m[0].content == "hi" and m[1].role == "user" and m[2].content == "answer"


def test_missing_session_empty(tmp_path):
    assert AgentSessionStore().recent_messages(str(tmp_path), "missing", 10) == []


def test_persists_and_reloads(tmp_path):
    p = str(tmp_path)
    AgentSessionStore().append_turn(p, "p1", "s.persist", "hello", "hi")
    m = AgentSessionStore().recent_messages(p, "s.persist", 10)
    assert [x.content for x in m] == ["hello", "hi"]
    f = tmp_path / ".llm-wiki" / "agent-sessions" / "s.persist.json"
    assert f.exists()
    import json
    d = json.loads(f.read_text())
    assert set(d) == {"sessionId", "projectId", "messages", "updatedAt"}
    assert set(d["messages"][0]) == {"role", "content", "timestamp"}


def test_cache_bounded_and_trim_messages(tmp_path):
    p = str(tmp_path)
    s = AgentSessionStore()
    for i in range(sess.MAX_CACHED_SESSIONS + 5):
        s.append_turn(p, "p1", f"s{i:03}", "hello", "hi")
    assert len(s._inner) <= sess.MAX_CACHED_SESSIONS
    for i in range(30):
        s.append_turn(p, "p1", "long", f"u{i}", f"a{i}")
    m = s.recent_messages(p, "long", 1000)
    assert len(m) == sess.MAX_SESSION_MESSAGES and m[-1].content == "a29"


def test_isolated_by_project(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    s = AgentSessionStore()
    s.append_turn(str(a), "p1", "same", "hello a", "answer a")
    s.append_turn(str(b), "p2", "same", "hello b", "answer b")
    assert s.recent_messages(str(a), "same", 10)[0].content == "hello a"
    assert s.recent_messages(str(b), "same", 10)[1].content == "answer b"


def test_session_ids_reject_traversal():
    assert sess.session_file("/tmp/project", "../secret") is None
    assert sess.session_file("/tmp/project", "safe-id") is not None
    assert sess.sanitize_session_id("a b!") == "a_b_"
    assert sess.sanitize_session_id("x" * 129) is None


def test_list_sessions_newest_first(tmp_path, monkeypatch):
    p = str(tmp_path)
    t = iter(range(1000, 2000))
    monkeypatch.setattr(sess, "_now_ms", lambda: next(t))
    s = AgentSessionStore()
    s.append_turn(p, "p1", "s1", "one", "a")
    s.append_turn(p, "p1", "s2", "two", "b")
    out = s.list_sessions(p)
    assert [x.session_id for x in out] == ["s2", "s1"]
