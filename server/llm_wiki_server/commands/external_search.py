"""Port of commands/external_search.rs (+ run_web_search / run_anytxt_search from agent/tools.rs).

Public helpers for other modules (e.g. the agent tools):
    async run_web_search(query, config, top_k) -> list[AgentReference dict]
    async run_anytxt_search(query, config, top_k) -> list[AgentReference dict]
    file_url_for_path(path) -> str
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import httpx

from ..errors import CommandError
from ..registry import command

WEB_SEARCH_TIMEOUT_SECS = 30.0
DEFAULT_ANYTXT_ENDPOINT = "http://127.0.0.1:9920"
DEFAULT_ANYTXT_LIMIT = 20
ANYTXT_LAST_MODIFY_END = 2_147_483_647


# --------------------------------------------------------------------------- helpers


def trim_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def url_encode(value: str) -> str:
    out = []
    for b in value.encode("utf-8"):
        c = chr(b)
        if c.isascii() and (c.isalnum() or c in "-_.~"):
            out.append(c)
        elif c == " ":
            out.append("+")
        else:
            out.append("%%%02X" % b)
    return "".join(out)


def _status_str(resp: httpx.Response) -> str:
    return f"{resp.status_code} {resp.reason_phrase}".strip()


def _s(v: Any) -> Optional[str]:
    return v if isinstance(v, str) else None


def _blank(v: Optional[str]) -> bool:
    return v is None or not v.strip()


# --------------------------------------------------------------------------- file URLs


def _percent_encode_file_segment(segment: str) -> str:
    out = []
    for b in segment.encode("utf-8"):
        c = chr(b)
        if c.isascii() and c.isalnum() or c in "-._~:":
            out.append(c)
        else:
            out.append("%%%02X" % b)
    return "".join(out)


def _encode_file_url_path(path: str) -> str:
    return "/".join(_percent_encode_file_segment(s) for s in path.split("/"))


def file_url_for_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if not normalized or "://" in normalized:
        return normalized
    if normalized.startswith("//"):
        return f"file:{normalized}"
    if (
        len(normalized) >= 3
        and normalized[1] == ":"
        and normalized[2] == "/"
        and normalized[0].isascii()
        and normalized[0].isalpha()
    ):
        return f"file:///{_encode_file_url_path(normalized)}"
    if normalized.startswith("/"):
        return f"file://{_encode_file_url_path(normalized)}"
    return normalized


def hostname_label(url: str) -> Optional[str]:
    try:
        sp = urlsplit(url)
        if not sp.scheme or "://" not in url:
            return None
        host = sp.hostname
    except ValueError:
        return None
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


# --------------------------------------------------------------------------- config


class WebSearchConfig:
    def __init__(self, d: Dict[str, Any], cmd: str = "web_search"):
        if not isinstance(d, dict):
            raise CommandError(f"invalid args `config` for command `{cmd}`: invalid type, expected struct WebSearchConfig")
        if "provider" not in d or not isinstance(d["provider"], str):
            raise CommandError(f"invalid args `config` for command `{cmd}`: missing field `provider`")
        self.provider: str = d["provider"]
        self.api_key: str = d.get("apiKey") or ""
        self.ollama_url: Optional[str] = d.get("ollamaUrl")
        self.sear_xng_url: Optional[str] = d.get("searXngUrl")
        self.sear_xng_categories: Optional[List[str]] = d.get("searXngCategories")
        self.serp_api_engine: Optional[str] = d.get("serpApiEngine")
        pc = d.get("providerConfigs")
        self.provider_configs: Optional[Dict[str, Dict[str, Any]]] = pc if isinstance(pc, dict) else None

    def resolved(self) -> "WebSearchConfig":
        provider = self.provider.strip().lower()
        ov = (self.provider_configs or {}).get(provider)
        if not isinstance(ov, dict):
            return self
        new = WebSearchConfig({"provider": self.provider})
        key = ov.get("apiKey")
        new.api_key = key if key is not None else self.api_key
        for attr, k in (
            ("ollama_url", "ollamaUrl"),
            ("sear_xng_url", "searXngUrl"),
            ("sear_xng_categories", "searXngCategories"),
            ("serp_api_engine", "serpApiEngine"),
        ):
            v = ov.get(k)
            setattr(new, attr, v if v is not None else getattr(self, attr))
        new.provider_configs = self.provider_configs
        return new


class AnyTxtConfig:
    def __init__(self, d: Optional[Dict[str, Any]]):
        d = d if isinstance(d, dict) else {}
        self.enabled: Optional[bool] = d.get("enabled")
        self.endpoint: Optional[str] = d.get("endpoint")
        self.filter_dir: Optional[str] = d.get("filterDir")
        self.filter_ext: Optional[str] = d.get("filterExt")
        self.limit: Optional[int] = d.get("limit")


def _as_web_config(config: Any) -> Optional[WebSearchConfig]:
    if config is None:
        return None
    if isinstance(config, WebSearchConfig):
        return config
    return WebSearchConfig(config)


# --------------------------------------------------------------------------- web search


def _normalize_web_result(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        value = {}
    metadata = value.get("metadata")
    md = metadata if isinstance(metadata, dict) else {}

    def first_str(*cands: Any) -> str:
        # Rust: .or_else chain picks the first *present* value, then as_str (non-string -> "")
        for c in cands:
            if c is not None:
                return c if isinstance(c, str) else ""
        return ""

    title = value.get("title")
    if title is None:
        title = md.get("title")
    title = title if isinstance(title, str) else "Untitled"
    url = first_str(
        value.get("url"), value.get("link"), md.get("sourceURL"), md.get("url"),
        value.get("original"), value.get("thumbnail"),
    )
    snippet = first_str(
        value.get("snippet"), value.get("content"), value.get("description"),
        md.get("description"), value.get("summary"), value.get("markdown"),
    )
    return {"title": title, "url": url, "snippet": snippet}


def _extract_nested_web_items(value: Any) -> Optional[list]:
    if not isinstance(value, dict):
        return None
    for key in ("web", "results", "items"):
        if isinstance(value.get(key), list):
            return value[key]
    return None


def _extract_web_items(value: Any, keys: List[str]) -> list:
    if not isinstance(value, dict):
        return []
    for key in keys:
        if key not in value:
            continue
        cand = value[key]
        if isinstance(cand, list):
            return cand
        nested = _extract_nested_web_items(cand)
        if nested is not None:
            return nested
    return []


def _required_api_key(config: WebSearchConfig, provider: str) -> str:
    key = config.api_key.strip()
    if not key:
        raise CommandError(f"{provider} web.search requires an API key in Settings.")
    return key


def _friendly_firecrawl_error(error: str) -> str:
    if "ip address looks suspicious" in error.lower():
        return (
            "Firecrawl Search rejected this IP for key-free access. Add a Firecrawl API key "
            "in Settings or choose another Web Search provider."
        )
    return f"Firecrawl search failed: {error}"


def _normalize_searxng_url(value: str) -> str:
    trimmed = value.strip().rstrip("/")
    if not trimmed:
        raise CommandError("SearXNG URL is required")
    url = trimmed if trimmed.startswith(("http://", "https://")) else f"https://{trimmed}"
    if not url.endswith("/search"):
        url += "/search"
    return url


def _provider_payload_error(provider: str, value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    if provider == "Bocha Search":
        code = value.get("code")
        code = code if isinstance(code, int) and not isinstance(code, bool) else None
        if code != 200:
            msg = value.get("msg")
            if not isinstance(msg, str) or not msg.strip():
                msg = "unknown API error"
            return f"{provider} failed (code {code or 0}): {msg}"
    if provider == "Brave Search" and "web" not in value:
        message = value.get("message")
        if isinstance(message, str):
            return f"{provider} search failed: {message}"
    return None


def _parse_web_json_response(resp: httpx.Response, provider: str, parse) -> List[Dict[str, str]]:
    text = resp.text
    if not resp.is_success:
        raise CommandError(f"{provider} search failed ({_status_str(resp)}): {trim_text(text, 300)}")
    try:
        value = json.loads(text)
    except ValueError:
        raise CommandError(f"{provider} returned invalid JSON: {trim_text(text, 300)}")
    if isinstance(value, dict) and isinstance(value.get("error"), str):
        raise CommandError(f"{provider} search failed: {value['error']}")
    msg = _provider_payload_error(provider, value)
    if msg:
        raise CommandError(msg)
    return parse(value)


def _list_results(value: Any, *path: str) -> list:
    cur = value
    for k in path:
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur if isinstance(cur, list) else []


async def _req(client: httpx.AsyncClient, method: str, url: str, err_prefix: str, **kw) -> httpx.Response:
    try:
        return await client.request(method, url, **kw)
    except httpx.HTTPError as e:
        raise CommandError(f"{err_prefix}: {e}")


async def _firecrawl_search(client, query, config: WebSearchConfig, max_results):
    ov = (config.provider_configs or {}).get("firecrawl")
    ov = ov if isinstance(ov, dict) else {}
    base = ov.get("baseUrl")
    base = (base if isinstance(base, str) and base.strip() else "https://api.firecrawl.dev").rstrip("/")
    headers = {"Accept": "application/json"}
    key = ov.get("apiKey")
    if isinstance(key, str) and key.strip():
        headers["Authorization"] = f"Bearer {key.strip()}"
    resp = await _req(
        client, "POST", f"{base}/v2/search", "Network error reaching Firecrawl Search",
        headers=headers, json={"query": query, "limit": max_results},
    )
    text = resp.text
    try:
        parsed = json.loads(text)
    except ValueError:
        raise CommandError(f"Firecrawl search returned invalid JSON: {trim_text(text, 300)}")
    if not resp.is_success or (isinstance(parsed, dict) and parsed.get("success") is False):
        err = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(err, str):
            raise CommandError(_friendly_firecrawl_error(err))
        raise CommandError(f"Firecrawl search failed ({_status_str(resp)})")
    return [_normalize_web_result(i) for i in _extract_web_items(parsed, ["data", "results"])]


async def _searxng_search(client, query, config: WebSearchConfig, max_results):
    if _blank(config.sear_xng_url):
        raise CommandError("SearXNG URL is required for web.search")
    url = _normalize_searxng_url(config.sear_xng_url)  # type: ignore[arg-type]
    categories = config.sear_xng_categories or ["general"]
    url += f"?q={url_encode(query)}&format=json&categories={url_encode(','.join(categories))}"
    resp = await _req(client, "GET", url, "Network error reaching SearXNG", headers={"Accept": "application/json"})
    text = resp.text
    if not resp.is_success:
        raise CommandError(f"SearXNG search failed ({_status_str(resp)}): {trim_text(text, 300)}")
    try:
        parsed = json.loads(text)
    except ValueError:
        raise CommandError(f"SearXNG returned invalid JSON: {trim_text(text, 300)}")
    items = _list_results(parsed, "results")
    return [_normalize_web_result(i) for i in items[:max_results]]


async def _tavily_search(client, query, config, max_results):
    key = _required_api_key(config, "Tavily")
    resp = await _req(
        client, "POST", "https://api.tavily.com/search", "Network error reaching Tavily",
        json={"api_key": key, "query": query, "max_results": max_results,
              "search_depth": "advanced", "include_answer": False},
    )
    return _parse_web_json_response(
        resp, "Tavily", lambda v: [_normalize_web_result(i) for i in _list_results(v, "results")]
    )


async def _ollama_search(client, query, config, max_results):
    key = _required_api_key(config, "Ollama")
    base = config.ollama_url
    base = (base.strip() if isinstance(base, str) and base.strip() else "https://ollama.com").rstrip("/")
    resp = await _req(
        client, "POST", f"{base}/api/web_search", "Network error reaching Ollama Web Search",
        headers={"Accept": "application/json", "Authorization": f"Bearer {key}"},
        json={"query": query, "max_results": max_results},
    )
    return _parse_web_json_response(
        resp, "Ollama Web Search", lambda v: [_normalize_web_result(i) for i in _list_results(v, "results")]
    )


async def _brave_search(client, query, config, max_results):
    key = _required_api_key(config, "Brave")
    url = f"https://api.search.brave.com/res/v1/web/search?q={url_encode(query)}&count={min(max_results, 20)}"
    resp = await _req(
        client, "GET", url, "Network error reaching Brave Search",
        headers={"Accept": "application/json", "X-Subscription-Token": key},
    )
    return _parse_web_json_response(
        resp, "Brave Search", lambda v: [_normalize_web_result(i) for i in _list_results(v, "web", "results")]
    )


def _parse_bocha_results(value: Any) -> List[Dict[str, str]]:
    out = []
    for item in _list_results(value, "data", "webPages", "value"):
        item = item if isinstance(item, dict) else {}
        summary = item.get("summary")
        if not isinstance(summary, str):
            summary = item.get("snippet")
        out.append({
            "title": item["name"] if isinstance(item.get("name"), str) else "Untitled",
            "url": item["url"] if isinstance(item.get("url"), str) else "",
            "snippet": summary if isinstance(summary, str) else "",
        })
    return out


async def _bocha_search(client, query, config, max_results):
    key = _required_api_key(config, "Bocha")
    resp = await _req(
        client, "POST", "https://api.bocha.cn/v1/web-search", "Network error reaching Bocha Search",
        headers={"Accept": "application/json", "Authorization": f"Bearer {key}"},
        json={"query": query, "freshness": "noLimit", "summary": True, "count": max(1, min(max_results, 50))},
    )
    return _parse_web_json_response(resp, "Bocha Search", _parse_bocha_results)


async def _serpapi_search(client, query, config, max_results):
    key = _required_api_key(config, "SerpApi")
    engine = config.serp_api_engine or "google"
    url = (
        f"https://serpapi.com/search?engine={url_encode(engine)}&q={url_encode(query)}"
        f"&api_key={url_encode(key)}&num={max_results}"
    )
    resp = await _req(client, "GET", url, "Network error reaching SerpApi", headers={"Accept": "application/json"})

    def parse(v):
        for k in ("organic_results", "news_results", "images_results", "video_results",
                  "videos_results", "shopping_results"):
            if isinstance(v, dict) and isinstance(v.get(k), list):
                return [_normalize_web_result(i) for i in v[k]]
        return []

    return _parse_web_json_response(resp, "SerpApi", parse)


def _web_search_result_limit(provider: str, requested: int) -> int:
    provider_max = 50 if provider == "bocha" else 20
    return max(1, min(requested, provider_max))


def _web_items_to_references(raw: List[Dict[str, str]], max_results: int) -> List[Dict[str, Any]]:
    out = []
    for item in raw[:max_results]:
        if not item["url"].strip():
            continue
        ref: Dict[str, Any] = {"title": item["title"], "path": item["url"], "kind": "web"}
        if item["snippet"].strip():
            ref["snippet"] = item["snippet"]
        out.append(ref)
    return out


async def run_web_search(query: str, config: Any, top_k: int) -> List[Dict[str, Any]]:
    """Returns AgentReference-shaped dicts ({title, path, kind, snippet?})."""
    query = query.strip()
    if not query:
        return []
    cfg = _as_web_config(config)
    if cfg is None:
        raise CommandError("Web search is enabled for this turn but no search provider is configured.")
    cfg = cfg.resolved()
    provider = cfg.provider.strip().lower()
    if not provider or provider == "none":
        raise CommandError("Web search provider is not configured.")
    max_results = _web_search_result_limit(provider, top_k)
    fns = {
        "firecrawl": _firecrawl_search, "searxng": _searxng_search, "tavily": _tavily_search,
        "ollama": _ollama_search, "brave": _brave_search, "bocha": _bocha_search,
        "serpapi": _serpapi_search,
    }
    fn = fns.get(provider)
    if fn is None:
        raise CommandError(f"Web search provider '{provider}' is not supported by the Rust Agent yet")
    async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT_SECS) as client:
        raw = await fn(client, query, cfg, max_results)
    return _web_items_to_references(raw, max_results)


# --------------------------------------------------------------------------- AnyTXT


def _value_at_path(value: Any, path) -> Any:
    cur = value
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


_ANYTXT_ARRAY_PATHS = [
    [], ["items"], ["files"], ["results"], ["list"], ["value"], ["data"], ["output"],
    ["output", "items"], ["output", "files"], ["output", "results"], ["output", "list"],
    ["output", "value"], ["output", "data"],
    ["data", "items"], ["data", "files"], ["data", "results"], ["data", "list"],
    ["data", "value"], ["data", "output"],
    ["data", "output", "items"], ["data", "output", "files"], ["data", "output", "results"],
    ["data", "output", "list"], ["data", "output", "value"],
]
_ANYTXT_FIELD_PATHS = [
    ["field"], ["fields"], ["output", "field"], ["output", "fields"],
    ["data", "field"], ["data", "fields"], ["data", "output", "field"], ["data", "output", "fields"],
]


def _normalize_anytxt_record(item: Any, fields: List[str]) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    if isinstance(item, list) and fields:
        return dict(zip(fields, item))
    return {"text": item}


def _string_field(record: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k not in record:
            continue
        v = record[k]
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v)
    return ""


def extract_anytxt_items(value: Any) -> List[Dict[str, str]]:
    result = value.get("result", value) if isinstance(value, dict) else value
    candidates: list = []
    for p in _ANYTXT_ARRAY_PATHS:
        c = _value_at_path(result, p)
        if isinstance(c, list):
            candidates = c
            break
    fields: List[str] = []
    for p in _ANYTXT_FIELD_PATHS:
        c = _value_at_path(result, p)
        if isinstance(c, list):
            fs = [x for x in c if isinstance(x, str)]
            if fs:
                fields = fs
                break
    items = []
    for item in candidates:
        rec = _normalize_anytxt_record(item, fields)
        fid = _string_field(rec, ["fid", "id", "fileId", "file_id"])
        raw_path = _string_field(
            rec, ["path", "file", "filePath", "file_path", "fullPath", "full_path", "filename", "fileName", "name"]
        )
        path = f"anytxt://{fid}" if (not raw_path and fid) else raw_path
        title = _string_field(rec, ["title", "name", "fileName", "filename"]).strip()
        if not title:
            base = path.rstrip("/").rsplit("/", 1)[-1] if path.rstrip("/") else ""
            title = base if base.strip() else "AnyTXT result"
        snippet = _string_field(
            rec, ["snippet", "fragment", "content", "contents", "text", "summary", "highlight", "hitText", "hit_text"]
        )
        if not path and not snippet:
            continue
        items.append({"fid": fid, "title": title, "path": path, "snippet": snippet})
    return items


def _extract_anytxt_fragment_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(t for t in (_extract_anytxt_fragment_text(i) for i in value) if t.strip())
    if not isinstance(value, dict):
        return ""
    for k in ("text", "fragment", "content", "snippet", "html"):
        if isinstance(value.get(k), str):
            return value[k]
    for k in ("output", "result", "data", "fragments", "items", "list"):
        if k in value:
            t = _extract_anytxt_fragment_text(value[k])
            if t.strip():
                return t
    return ""


def _normalize_anytxt_endpoint(value: str) -> str:
    if value.startswith(("http://", "https://")):
        return value
    return f"http://{value}"


async def _get_anytxt_fragment(client: httpx.AsyncClient, endpoint: str, fid: str, pattern: str) -> str:
    try:
        resp = await client.post(
            endpoint, headers={"Accept": "application/json"},
            json={"id": 2, "jsonrpc": "2.0", "method": "ATRpcServer.Searcher.V1.GetFragment",
                  "params": {"input": {"fid": fid, "pattern": pattern}}},
        )
    except httpx.HTTPError as e:
        raise CommandError(f"AnyTXT fragment failed: {e}")
    text = resp.text
    if not resp.is_success:
        raise CommandError(f"AnyTXT fragment HTTP {_status_str(resp)}: {trim_text(text, 300)}")
    try:
        value = json.loads(text)
    except ValueError:
        raise CommandError(f"AnyTXT fragment returned invalid JSON: {trim_text(text, 300)}")
    if isinstance(value, dict) and "error" in value:
        raise CommandError(f"AnyTXT fragment error: {trim_text(json.dumps(value['error'], ensure_ascii=False), 300)}")
    return _extract_anytxt_fragment_text(value.get("result") if isinstance(value, dict) else None)


async def run_anytxt_search(query: str, config: Any, top_k: int) -> List[Dict[str, Any]]:
    query = query.strip()
    if not query:
        return []
    cfg = config if isinstance(config, AnyTxtConfig) else AnyTxtConfig(config)
    if cfg.enabled is False:
        return []
    endpoint = cfg.endpoint if isinstance(cfg.endpoint, str) and cfg.endpoint.strip() else DEFAULT_ANYTXT_ENDPOINT
    endpoint = _normalize_anytxt_endpoint(endpoint.strip().rstrip("/"))
    cfg_limit = cfg.limit if isinstance(cfg.limit, int) and not isinstance(cfg.limit, bool) else DEFAULT_ANYTXT_LIMIT
    limit = min(max(1, min(top_k, 100)), max(1, min(cfg_limit, 100)))
    pattern = query
    filter_dir = cfg.filter_dir or ""
    filter_ext = cfg.filter_ext if isinstance(cfg.filter_ext, str) and cfg.filter_ext.strip() else "*"
    inp: Dict[str, Any] = {
        "pattern": pattern, "filterExt": filter_ext, "lastModifyBegin": 0,
        "lastModifyEnd": ANYTXT_LAST_MODIFY_END, "limit": str(limit), "offset": 0, "order": 0,
    }
    if filter_dir.strip():
        inp["filterDir"] = filter_dir
    async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT_SECS) as client:
        try:
            resp = await client.post(
                endpoint, headers={"Accept": "application/json"},
                json={"id": 1, "jsonrpc": "2.0", "method": "ATRpcServer.Searcher.V1.GetResult",
                      "params": {"input": inp}},
            )
        except httpx.HTTPError as e:
            raise CommandError(
                f"AnyTXT search failed. Check that ATGUI.exe or the AnyTXT service is running at {endpoint}: {e}"
            )
        text = resp.text
        if not resp.is_success:
            raise CommandError(f"AnyTXT HTTP {_status_str(resp)}: {trim_text(text, 300)}")
        try:
            value = json.loads(text)
        except ValueError:
            raise CommandError(f"AnyTXT returned invalid JSON: {trim_text(text, 300)}")
        if isinstance(value, dict) and "error" in value:
            raise CommandError(f"AnyTXT error: {trim_text(json.dumps(value['error'], ensure_ascii=False), 300)}")
        refs = []
        for item in extract_anytxt_items(value)[:limit]:
            fragment = ""
            if item["fid"].strip():
                try:
                    fragment = await _get_anytxt_fragment(client, endpoint, item["fid"], pattern)
                except CommandError:
                    fragment = ""
            snippet = trim_text(fragment if fragment.strip() else item["snippet"], 1200)
            ref: Dict[str, Any] = {
                "title": item["title"], "path": file_url_for_path(item["path"]), "kind": "anytxt",
            }
            if snippet.strip():
                ref["snippet"] = snippet
            refs.append(ref)
        return refs


# --------------------------------------------------------------------------- commands


def _max_results(v: Any, default: int) -> int:
    return default if v is None else int(v)


@command()
async def web_search(query: str, config: Dict[str, Any], max_results: Optional[int] = None):
    refs = await run_web_search(query, _as_web_config(config) if config is not None else None,
                                _max_results(max_results, 10))
    return [
        {
            "title": r["title"],
            "url": r["path"],
            "snippet": r.get("snippet") or "",
            "source": hostname_label(r["path"]) or "web",
        }
        for r in refs
    ]


@command()
async def anytxt_search(query: str, config: Dict[str, Any], max_results: Optional[int] = None):
    refs = await run_anytxt_search(query, config, _max_results(max_results, 20))
    return [
        {
            "title": r["title"],
            "url": file_url_for_path(r["path"]),
            "snippet": r.get("snippet") or "",
            "source": "AnyTXT",
        }
        for r in refs
    ]
