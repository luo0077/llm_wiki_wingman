"""Ingest RPC commands + REST routes (headless port of the desktop ingest queue UI wiring).

RPC (``POST /rpc/<name>``, args camelCase or snake_case; ``project`` is a project id, path or "current"):

* ``ingest_enqueue``    {project, sourcePath | sourcePaths | files:[{sourcePath, folderContext}], folderContext?}
* ``ingest_status``     {project, taskId?}        -> summary (+ task when taskId given)
* ``ingest_cancel``     {project, taskId? | taskIds? | all?}
* ``ingest_retry``      {project, taskId? | taskIds? | all?}
* ``ingest_list_queue`` {project}                 -> {summary, tasks, activity}
* ``ingest_pause`` / ``ingest_resume`` / ``ingest_clear_completed`` {project}
* ``ingest_start`` {project, sourcePath} / ``ingest_execute_writes`` {project, guidance?}  (interactive flow)

REST (same auth as rest_api: writes always need the API token):

* POST /api/v1/projects/{id}/ingest              body {sourcePath|sourcePaths|files|upload{name,content|contentBase64}, folderContext}
* GET  /api/v1/projects/{id}/ingest/queue
* GET  /api/v1/projects/{id}/ingest/{taskId}
* POST /api/v1/projects/{id}/ingest/{taskId}/cancel
* POST /api/v1/projects/{id}/ingest/{taskId}/retry

Progress is published on the event bus (``GET /events?topics=ingest://*``): ``ingest://queue-updated``,
``ingest://task-done``, ``ingest://activity``, ``ingest://files-changed``, ``ingest://reviews-updated``.
"""
from __future__ import annotations

import base64
import functools
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from ..errors import CommandError
from ..ingest import pipeline
from ..ingest import queue as iq
from ..registry import command, on_startup
from . import rest_api

router = APIRouter()


# --------------------------------------------------------------------------
# helpers shared by RPC + REST
# --------------------------------------------------------------------------

def _pq(project: str) -> iq.ProjectQueue:
    return iq.manager().get(project)


def _as_files(source_path: Any, source_paths: Any, files: Any, folder_context: Any) -> List[Dict[str, str]]:
    fc = folder_context if isinstance(folder_context, str) else ""
    out: List[Dict[str, str]] = []
    if isinstance(source_path, str) and source_path:
        out.append({"sourcePath": source_path, "folderContext": fc})
    if isinstance(source_paths, list):
        out += [{"sourcePath": p, "folderContext": fc} for p in source_paths if isinstance(p, str) and p]
    if isinstance(files, list):
        for f in files:
            if isinstance(f, dict) and isinstance(f.get("sourcePath"), str) and f["sourcePath"]:
                c = f.get("folderContext")
                out.append({"sourcePath": f["sourcePath"], "folderContext": c if isinstance(c, str) else fc})
    return out


def _task_view(t: Dict[str, Any]) -> Dict[str, Any]:
    return dict(t)


def _status(pq: iq.ProjectQueue, task_id: Optional[str] = None) -> Dict[str, Any]:
    with pq.lock:
        out: Dict[str, Any] = {"projectId": pq.project_id, "summary": pq.summary()}
        if task_id:
            t = pq.find(task_id)
            if t is None:
                raise CommandError("Unknown ingest task: %s" % task_id, 404)
            out["task"] = _task_view(t)
    return out


def _ids(task_id: Any, task_ids: Any) -> List[str]:
    ids: List[str] = []
    if isinstance(task_id, str) and task_id:
        ids.append(task_id)
    if isinstance(task_ids, list):
        ids += [i for i in task_ids if isinstance(i, str)]
    return ids


def _within(project_path: str, path: str) -> str:
    root = os.path.realpath(project_path)
    real = os.path.realpath(path if os.path.isabs(path) else os.path.join(project_path, path))
    if real != root and not real.startswith(root + os.sep):
        raise CommandError("Source path must be inside the project: %s" % path, 400)
    if not os.path.isfile(real):
        raise CommandError("Source file does not exist: %s" % path, 404)
    return real


def _save_upload(project_path: str, upload: Dict[str, Any]) -> str:
    name = upload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise CommandError("upload.name is required", 400)
    base = os.path.basename(name.replace("\\", "/")).strip().lstrip(".")
    base = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", base)
    if not base:
        raise CommandError("Invalid upload name", 400)
    if isinstance(upload.get("contentBase64"), str):
        try:
            data = base64.b64decode(upload["contentBase64"], validate=True)
        except Exception:
            raise CommandError("upload.contentBase64 is not valid base64", 400)
    elif isinstance(upload.get("content"), str):
        data = upload["content"].encode("utf-8")
    else:
        raise CommandError("upload.content or upload.contentBase64 is required", 400)
    dest_dir = Path(project_path) / "raw" / "sources"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / base
    tmp = dest_dir / (".%s.upload.tmp" % base)
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return "raw/sources/%s" % base


def enqueue(project: str, body: Dict[str, Any], *, restrict_to_project: bool) -> Dict[str, Any]:
    info = iq.manager().resolve_project(project)
    files = _as_files(body.get("sourcePath"), body.get("sourcePaths"), body.get("files"), body.get("folderContext"))
    upload = body.get("upload")
    if isinstance(upload, dict):
        rel = _save_upload(info["path"], upload)
        fc = body.get("folderContext")
        files.append({"sourcePath": rel, "folderContext": fc if isinstance(fc, str) else ""})
    if not files:
        raise CommandError("sourcePath, sourcePaths, files or upload is required", 400)
    if restrict_to_project:
        for f in files:
            _within(info["path"], f["sourcePath"])
    pq = iq.manager().get_for(info["id"], info["path"])
    ids = pq.enqueue_batch(files)
    return {"projectId": info["id"], "taskIds": ids, "summary": pq.summary()}


# --------------------------------------------------------------------------
# RPC commands
# --------------------------------------------------------------------------

@command()
def ingest_enqueue(project: str, source_path: Optional[str] = None, source_paths: Optional[list] = None,
                   files: Optional[list] = None, folder_context: Optional[str] = None,
                   upload: Optional[dict] = None):
    body = {"sourcePath": source_path, "sourcePaths": source_paths, "files": files,
            "folderContext": folder_context, "upload": upload}
    return {"ok": True, **enqueue(project, body, restrict_to_project=False)}


@command()
def ingest_status(project: str, task_id: Optional[str] = None):
    return {"ok": True, **_status(_pq(project), task_id)}


@command()
def ingest_list_queue(project: str):
    pq = _pq(project)
    out = pq.snapshot()
    out["activity"] = pipeline.list_activity(pq.project_path)
    return {"ok": True, **out}


@command()
def ingest_cancel(project: str, task_id: Optional[str] = None, task_ids: Optional[list] = None,
                  all: bool = False):  # noqa: A002
    pq = _pq(project)
    n = pq.cancel_all() if all else pq.cancel(_ids(task_id, task_ids))
    return {"ok": True, "cancelled": n, "summary": pq.summary()}


@command()
def ingest_retry(project: str, task_id: Optional[str] = None, task_ids: Optional[list] = None,
                 all: bool = False):  # noqa: A002
    pq = _pq(project)
    n = pq.retry_all_stopped() if all else pq.retry(_ids(task_id, task_ids))
    return {"ok": True, "requeued": n, "summary": pq.summary()}


@command()
def ingest_pause(project: str):
    pq = _pq(project)
    pq.pause()
    return {"ok": True, "summary": pq.summary()}


@command()
def ingest_resume(project: str):
    pq = _pq(project)
    pq.resume()
    return {"ok": True, "summary": pq.summary()}


@command()
def ingest_clear_completed(project: str):
    pq = _pq(project)
    pq.clear_completed()
    return {"ok": True, "summary": pq.summary()}


@command()
def ingest_start(project: str, source_path: str):
    info = iq.manager().resolve_project(project)
    llm = pipeline.get_task_llm_config(info["id"])
    if not pipeline.has_usable_llm(llm):
        raise CommandError("LLM not configured", 400)
    src = source_path if os.path.isabs(source_path) else os.path.join(info["path"], source_path)
    text = iq.runtime().submit(pipeline.start_ingest(info["path"], src, llm, None, info["id"])).result()
    return {"ok": True, "text": text}


@command()
def ingest_execute_writes(project: str, guidance: Optional[str] = None):
    info = iq.manager().resolve_project(project)
    llm = pipeline.get_task_llm_config(info["id"])
    if not pipeline.has_usable_llm(llm):
        raise CommandError("LLM not configured", 400)
    written = iq.runtime().submit(pipeline.execute_ingest_writes(info["path"], llm, guidance, None, info["id"])).result()
    return {"ok": True, "writtenPaths": written}


@on_startup
def _restore_ingest_queues() -> None:
    """Resume interrupted/pending ingest tasks of every known project after a restart."""
    if os.environ.get("LLM_WIKI_DISABLE_INGEST_RESTORE") == "1":
        return
    try:
        iq.restore_all_queues()
    except Exception as e:  # never block server start
        print("[ingest] queue restore failed: %r" % (e,), file=sys.stderr)


# --------------------------------------------------------------------------
# REST
# --------------------------------------------------------------------------

def _err(status: int, message: str) -> JSONResponse:
    return JSONResponse(rest_api._err_body(message), status_code=status)


async def _guard(request: Request, write: bool) -> Optional[JSONResponse]:
    method = request.method.upper()
    path = request.url.path
    if rest_api.should_rate_limit(method, path) and not rest_api.allow_request():
        return _err(429, "Too many requests")
    if not rest_api.api_enabled():
        return _err(503, "API server is disabled in Settings → API Server")
    headers = [(k.lower(), v) for k, v in request.headers.items()]
    query = request.scope.get("query_string", b"").decode("latin-1")
    if write:
        if not rest_api.is_token_authorized(query, headers):
            return _err(401, "Unauthorized")
    elif not rest_api.is_authorized(query, headers):
        return _err(401, "Unauthorized")
    return None


async def _body(request: Request) -> Dict[str, Any]:
    try:
        raw = await rest_api._read_body(request, rest_api.MAX_CHAT_BODY_BYTES)
    except rest_api.ApiError as e:
        raise CommandError(e.message, e.status)
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as e:
        raise CommandError("Invalid JSON body: %s" % e, 400)
    if not isinstance(parsed, dict):
        raise CommandError("Request body must be a JSON object", 400)
    return parsed


def _wrap(fn):
    @functools.wraps(fn)
    async def inner(request: Request, *args: Any, **kwargs: Any):
        write = request.method.upper() != "GET"
        denied = await _guard(request, write)
        if denied is not None:
            return denied
        try:
            return JSONResponse(await fn(request, *args, **kwargs))
        except CommandError as e:
            return _err(e.status, e.message)
        except Exception as e:  # noqa: BLE001
            print("[ingest] request failed: %r" % (e,), file=sys.stderr)
            return _err(500, "Internal API server error")

    return inner


def _resolve(project_id: str) -> iq.ProjectQueue:
    return _pq(rest_api.percent_decode(project_id))


@router.post("/api/v1/projects/{project_id}/ingest")
@_wrap
async def rest_ingest(request: Request, project_id: str):
    body = await _body(request)
    out = await run_in_threadpool(enqueue, rest_api.percent_decode(project_id), body, restrict_to_project=True)
    return {"ok": True, **out}


@router.get("/api/v1/projects/{project_id}/ingest/queue")
@_wrap
async def rest_ingest_queue(request: Request, project_id: str):
    pq = await run_in_threadpool(_resolve, project_id)
    return {"ok": True, **pq.snapshot(), "activity": pipeline.list_activity(pq.project_path)[:20]}


@router.get("/api/v1/projects/{project_id}/ingest/{task_id}")
@_wrap
async def rest_ingest_task(request: Request, project_id: str, task_id: str):
    pq = await run_in_threadpool(_resolve, project_id)
    return {"ok": True, **_status(pq, task_id)}


@router.post("/api/v1/projects/{project_id}/ingest/{task_id}/cancel")
@_wrap
async def rest_ingest_cancel(request: Request, project_id: str, task_id: str):
    pq = await run_in_threadpool(_resolve, project_id)
    if pq.find(task_id) is None:
        raise CommandError("Unknown ingest task: %s" % task_id, 404)
    n = await run_in_threadpool(pq.cancel, [task_id])
    return {"ok": True, "cancelled": n, **_status(pq, task_id)}


@router.post("/api/v1/projects/{project_id}/ingest/{task_id}/retry")
@_wrap
async def rest_ingest_retry(request: Request, project_id: str, task_id: str):
    pq = await run_in_threadpool(_resolve, project_id)
    if pq.find(task_id) is None:
        raise CommandError("Unknown ingest task: %s" % task_id, 404)
    n = await run_in_threadpool(pq.retry, [task_id])
    return {"ok": True, "requeued": n, **_status(pq, task_id)}


# rest_api registers a catch-all `/api/v1/{rest:path}` that would shadow these routes when this
# module is included after it (commands/__init__._MODULES order), so put ours first.
def _install_before_catch_all() -> None:
    existing = {id(r) for r in rest_api.router.routes}
    mine = [r for r in router.routes if id(r) not in existing]
    rest_api.router.routes[0:0] = mine


_install_before_catch_all()
