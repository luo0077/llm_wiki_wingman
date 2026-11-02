"""Replacement for src/lib/tauri-fetch.ts: outbound HTTP for the headless server.

The Tauri plugin existed to dodge browser CORS; on the server we simply use httpx.  Clients are
built per call so that the global proxy configuration applied by ``llm_wiki_server.proxy``
(``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY`` env vars via ``trust_env=True`` and the
``accept_invalid_certs`` TLS policy) always takes effect immediately, mirroring the TS code which
read the proxy settings at request time.

Tests inject an ``httpx.MockTransport`` either per call (``transport=``) or globally with
:func:`use_transport` / :func:`set_default_transport`.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Optional

import httpx


_DEFAULT_TRANSPORT: Optional[httpx.AsyncBaseTransport] = None

# No total read timeout: LLM responses can legitimately stream for a long time; callers apply
# their own backstop (see llm_client's requestTimeoutMinutes).
LLM_TIMEOUT = httpx.Timeout(connect=60.0, read=None, write=None, pool=None)
DEFAULT_TIMEOUT = httpx.Timeout(120.0)


class AbortError(Exception):
    """JS `AbortError` (DOMException name) equivalent."""

    name = "AbortError"


def set_default_transport(transport: Optional[httpx.AsyncBaseTransport]) -> None:
    global _DEFAULT_TRANSPORT
    _DEFAULT_TRANSPORT = transport


def get_default_transport() -> Optional[httpx.AsyncBaseTransport]:
    return _DEFAULT_TRANSPORT


@contextlib.contextmanager
def use_transport(transport: Optional[httpx.AsyncBaseTransport]) -> Iterator[None]:
    global _DEFAULT_TRANSPORT
    prev = _DEFAULT_TRANSPORT
    _DEFAULT_TRANSPORT = transport
    try:
        yield
    finally:
        _DEFAULT_TRANSPORT = prev


def client_kwargs(*, timeout: Optional[httpx.Timeout] = None, follow_redirects: bool = True,
                  transport: Optional[httpx.AsyncBaseTransport] = None) -> Dict[str, Any]:
    kw: Dict[str, Any] = {
        "timeout": timeout if timeout is not None else DEFAULT_TIMEOUT,
        "trust_env": True,
        "follow_redirects": follow_redirects,
    }
    transport = transport if transport is not None else _DEFAULT_TRANSPORT
    if transport is not None:
        kw["transport"] = transport
    else:
        try:
            from .. import proxy as _proxy
            kw["verify"] = not _proxy.accept_invalid_certs()
        except Exception:
            pass
    return kw


def make_async_client(*, timeout: Optional[httpx.Timeout] = None, follow_redirects: bool = True,
                      transport: Optional[httpx.AsyncBaseTransport] = None) -> httpx.AsyncClient:
    """AsyncClient honoring proxy env + TLS policy (``getHttpFetch()`` equivalent)."""
    return httpx.AsyncClient(**client_kwargs(timeout=timeout, follow_redirects=follow_redirects,
                                             transport=transport))


def is_fetch_network_error(err: Any) -> bool:
    """Port of `isFetchNetworkError`: a fetch-level network failure that is not an abort.

    httpx transport failures (DNS, TLS, refused, reset, timeouts) count, as do the generic
    JS shapes (``TypeError``, "Load failed", "Failed to fetch", messages containing
    "network error").
    """
    if not isinstance(err, BaseException):
        return False
    if isinstance(err, AbortError) or getattr(err, "name", None) == "AbortError":
        return False
    if isinstance(err, (httpx.TransportError, TypeError)):
        return True
    msg = str(err)
    if msg == "Load failed" or msg == "Failed to fetch":
        return True
    return "network error" in msg
