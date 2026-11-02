"""Port of the agent_* commands in lib.rs (agent_start_turn, agent_start_turn_stream,
agent_cancel_turn, agent_get_session, agent_list_sessions).

Streaming events are published on the `agent-event` topic with the Tauri payload
`{"sessionId", "runId", "event"}` (consume via `GET /events?topics=agent-event`).
Project resolution / app-state parsing reuse the lib.rs-equivalent helpers in
rest_api.py.
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, Optional

from .. import events
from ..agent import skills  # noqa: F401  (registers agent_list_skills)
from ..agent import cancel as _cancel
from ..agent import runtime as _runtime
from ..agent import session as _session
from ..agent.types import AgentChatRequest, AgentConversationMessage
from ..errors import CommandError
from ..registry import command

SESSION_STORE = _session.DEFAULT_STORE
CANCELLATIONS = _cancel.AgentCancellationRegistry()


def _resolve_project(project_id: str) -> Dict[str, Any]:
    from . import rest_api

    try:
        return rest_api.resolve_project(project_id)
    except rest_api.ApiError as e:
        raise CommandError(e.message, e.status)


def _runtime_config(llm_override: Optional[dict]) -> Dict[str, Any]:
    """lib.rs load_agent_runtime_config (+ the llm_config argument override)."""
    from . import rest_api

    parsed = rest_api.load_app_state() or {}

    def d(v: Any) -> Optional[dict]:
        return v if isinstance(v, dict) else None

    search_cfg = d(parsed.get("searchApiConfig"))
    return {
        "embedding": d(parsed.get("embeddingConfig")),
        "llm": llm_override if llm_override is not None else d(parsed.get("llmConfig")),
        "webSearch": search_cfg,
        "anytxt": d(search_cfg.get("anyTxt")) if search_cfg else None,
    }


def _prepare(project_id: str, request: Dict[str, Any], llm_config: Optional[dict], hydrate_needs_id: bool):
    project = _resolve_project(project_id)
    try:
        req = AgentChatRequest.from_dict(request)
    except ValueError as e:
        raise CommandError(f"invalid args `request` for command `agent_start_turn`: {e}")
    if not (req.session_id or "").strip():
        req.session_id = f"ui_{uuid.uuid4()}"
    session_id = req.session_id
    if not (req.run_id or "").strip():
        req.run_id = f"run_{uuid.uuid4()}"
    run_id = req.run_id
    if not req.history and not req.history_explicit:
        req.history = [AgentConversationMessage(m.role, m.content)
                       for m in SESSION_STORE.recent_messages(project["path"], session_id, 12)]
    config = _runtime_config(llm_config)
    runtime = _runtime.build_runtime(project, config)
    token = CANCELLATIONS.start(project["id"], session_id, run_id)
    return project, req, runtime, token, session_id, run_id


@command()
async def agent_start_turn(project_id: str, request: dict, llm_config: Optional[dict] = None) -> dict:
    project, req, runtime, token, session_id, run_id = _prepare(project_id, request, llm_config, True)
    user_message = req.message
    persist = req.persist_session
    try:
        try:
            response = await runtime.run_once_with_cancel(req, token)
        except CommandError:
            raise
        except Exception as e:  # noqa: BLE001
            raise CommandError(str(e))
    finally:
        CANCELLATIONS.finish(project["id"], session_id, run_id)
    if persist:
        SESSION_STORE.append_turn(project["path"], project["id"], response.session_id,
                                  user_message, response.message)
    return response.to_dict()


@command()
async def agent_start_turn_stream(project_id: str, request: dict, llm_config: Optional[dict] = None) -> str:
    project, req, runtime, token, session_id, run_id = _prepare(project_id, request, llm_config, False)
    user_message = req.message
    persist = req.persist_session

    def emit(event: Any) -> None:
        events.publish("agent-event", {"sessionId": session_id, "runId": run_id, "event": dict(event)})

    def work() -> None:
        try:
            try:
                response = _runtime._run_sync(
                    lambda: runtime.run_once_with_cancel_and_events(req, token, emit))
            except BaseException as e:  # noqa: BLE001
                CANCELLATIONS.finish(project["id"], session_id, run_id)
                emit({"type": "error", "message": str(e)})
                return
            CANCELLATIONS.finish(project["id"], session_id, run_id)
            if persist:
                SESSION_STORE.append_turn(project["path"], project["id"], response.session_id,
                                          user_message, response.message)
        finally:
            token.close()

    threading.Thread(target=work, daemon=True, name="llm-wiki-agent-stream").start()
    return session_id


@command()
def agent_cancel_turn(project_id: str, session_id: str, run_id: Optional[str] = None) -> bool:
    project = _resolve_project(project_id)
    return CANCELLATIONS.cancel(project["id"], session_id, run_id)


@command()
def agent_get_session(project_id: str, session_id: str, limit: Optional[int] = None) -> list:
    project = _resolve_project(project_id)
    n = min(max(limit if limit is not None else 40, 1), 200)
    return [m.to_dict() for m in SESSION_STORE.recent_messages(project["path"], session_id, n)]


@command()
def agent_list_sessions(project_id: str) -> list:
    project = _resolve_project(project_id)
    return [s.to_dict() for s in SESSION_STORE.list_sessions(project["path"])]
