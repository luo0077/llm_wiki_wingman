"""Self-check helpers used by examples/selftest.py (environment report + LLM/embedding pings).

Nothing here returns secrets: API keys are reported only as present/absent.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import sys
import time
from urllib.parse import urlparse

from ..registry import command
from ..state import app_state, data_dir

# (import name, pip/pixi name, what needs it)
_LIBS = [
    ("fastapi", "fastapi", "server"), ("uvicorn", "uvicorn", "server"), ("httpx", "httpx", "LLM / network calls"),
    ("yaml", "pyyaml", "ingest (frontmatter)"),
    ("pymupdf", "pymupdf", "PDF text + images"), ("pypdf", "pypdf", "PDF text (fallback)"),
    ("docx", "python-docx", ".docx"), ("openpyxl", "openpyxl", ".xlsx"),
    ("xlrd", "xlrd", ".xls"), ("striprtf", "striprtf", ".rtf"), ("PIL", "pillow", "image handling"),
    ("watchdog", "watchdog", "file watcher (else polling)"), ("lancedb", "lancedb", "vector DB (else sqlite)"),
    ("pyarrow", "pyarrow", "LanceDB"), ("numpy", "numpy", "faster vector search"),
]


def _version(mod: str):
    try:
        from importlib import metadata

        return metadata.version(mod)
    except Exception:
        return None


def _redact_cfg(cfg):
    if not isinstance(cfg, dict) or not cfg:
        return None
    ep = cfg.get("customEndpoint") or cfg.get("ollamaUrl") or cfg.get("baseUrl") or cfg.get("endpoint") or ""
    host = urlparse(ep).netloc if ep else ""
    return {"provider": cfg.get("provider"), "model": cfg.get("model"), "endpointHost": host, "hasApiKey": bool(cfg.get("apiKey"))}


@command()
def diagnostics_info() -> dict:
    libs = {}
    for imp, pip_name, purpose in _LIBS:
        libs[pip_name] = {"installed": importlib.util.find_spec(imp) is not None, "version": _version(pip_name), "usedFor": purpose}
    backend = None
    try:
        from . import vectorstore

        backend = vectorstore.backend_name()
    except Exception as e:  # report, don't fail
        backend = "error: %s" % e
    d = data_dir()
    writable = os.access(str(d), os.W_OK)
    st = app_state()
    token_src = "env" if (os.environ.get("LLM_WIKI_API_TOKEN") or "").strip() else (
        "store" if (st.get("apiConfig") or {}).get("token") else "none")
    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "libs": libs,
        "vectorBackend": backend, "dataDir": str(d), "dataDirWritable": writable,
        "tokenSource": token_src,
        "llmConfig": _redact_cfg(st.get("llmConfig")),
        "embeddingConfig": _redact_cfg(st.get("embeddingConfig")),
        "proxyEnv": sorted(k for k in os.environ if k.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy")),
    }


@command()
async def diagnostics_llm_ping(timeout_secs: float = 90) -> dict:
    """Send a tiny prompt through the configured llmConfig, exactly as ingest would."""
    cfg = app_state().get("llmConfig")
    if not isinstance(cfg, dict) or not cfg:
        return {"ok": False, "error": "llmConfig is not set (use app_state_set key=llmConfig)"}
    import asyncio

    from ..ingest import llm_client

    t0 = time.time()
    try:
        text = await asyncio.wait_for(llm_client.collect_stream_chat(
            cfg, [{"role": "user", "content": "Reply with the single word OK."}]), timeout=float(timeout_secs))
    except asyncio.TimeoutError:
        return {"ok": False, "error": "timed out after %ss" % timeout_secs, "config": _redact_cfg(cfg)}
    except Exception as e:
        return {"ok": False, "error": str(e) or type(e).__name__, "config": _redact_cfg(cfg)}
    return {"ok": bool((text or "").strip()), "reply": (text or "").strip()[:200],
            "ms": int((time.time() - t0) * 1000), "config": _redact_cfg(cfg),
            **({} if (text or "").strip() else {"error": "empty reply"})}


@command()
async def diagnostics_embedding_ping() -> dict:
    cfg = app_state().get("embeddingConfig")
    if not isinstance(cfg, dict) or not cfg or cfg.get("enabled") is False:
        return {"ok": False, "skipped": True, "error": "embeddingConfig is not set or disabled"}
    from . import search

    t0 = time.time()
    try:
        vec = await search.embedding_fetch("hello world", cfg, 1)
    except Exception as e:
        return {"ok": False, "error": getattr(e, "message", None) or str(e) or type(e).__name__, "config": _redact_cfg(cfg)}
    return {"ok": bool(vec), "dimensions": len(vec or []), "ms": int((time.time() - t0) * 1000), "config": _redact_cfg(cfg)}
