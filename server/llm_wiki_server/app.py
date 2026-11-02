from __future__ import annotations

import inspect
import traceback
from typing import Any

from fastapi import Body, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__, registry
from .errors import CommandError


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Wiki Server", version=__version__)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    # Import for side effects: each module registers its commands.
    from . import commands  # noqa: F401

    # Modules may expose a FastAPI `router` (REST / SSE endpoints, e.g. the
    # /api/v1 surface and the web-clipper server) that is mounted as-is.
    import sys

    for mod_name in commands._MODULES:
        mod = sys.modules.get(f"{commands.__name__}.{mod_name}")
        if mod is not None and getattr(mod, "router", None) is not None:
            app.include_router(mod.router)

    @app.on_event("startup")
    async def _startup():
        for hook in registry.startup_hooks():
            await run_in_threadpool(hook)

    @app.middleware("http")
    async def _guard_rpc(request: Request, call_next):
        # /rpc and /events expose arbitrary-path file I/O (they replace Tauri's
        # in-process IPC), so they are stricter than /api/v1: a configured token is
        # ALWAYS required (allowUnauthenticated does not apply); with no token only
        # loopback clients are allowed.
        path = request.url.path
        if path == "/rpc" or path.startswith("/rpc/") or path == "/events":
            from .commands import rest_api

            if rest_api.api_token() is not None:
                headers = [(k.lower(), v) for k, v in request.headers.items()]
                if not rest_api.is_token_authorized(request.url.query, headers):
                    return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)
            elif (request.client.host if request.client else "") not in ("127.0.0.1", "::1", "localhost", "testclient"):
                return JSONResponse(
                    {"ok": False, "error": "Remote access to /rpc requires a token: set LLM_WIKI_API_TOKEN"},
                    status_code=403,
                )
        return await call_next(request)

    @app.exception_handler(CommandError)
    async def _command_error(_: Request, exc: CommandError):
        return JSONResponse({"ok": False, "error": exc.message}, status_code=exc.status)

    @app.get("/health")
    def health():
        return {"ok": True, "status": "running", "version": __version__}

    @app.get("/rpc")
    def list_commands():
        return {"commands": registry.command_names()}

    @app.post("/rpc/{name}")
    async def rpc(name: str, args: dict[str, Any] = Body(default_factory=dict)):
        fn = registry.get_command(name)
        if fn is None:
            raise CommandError(f"unknown command `{name}`", 404)
        kwargs = registry.bind_args(fn, args)
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(**kwargs)
            else:
                result = await run_in_threadpool(fn, **kwargs)
        except CommandError:
            raise
        except Exception as e:  # panic_guard equivalent: never kill the server
            traceback.print_exc()
            raise CommandError(f"{name} failed: {e}", 500)
        return JSONResponse(result)

    return app
