#!/usr/bin/env python3
"""LLM Wiki server client. Standard library only (no pip install needed).

Setup (once, in the shell on the machine that runs this script):
    export WIKI_URL=http://<server-ip>:19828
    export WIKI_TOKEN=<token the server was started with>

Usage:
    python wiki_client.py check
    python wiki_client.py projects
    python wiki_client.py ingest-file ./notes.md [more files ...]   # upload + wait
    python wiki_client.py ingest-dir ./docs                         # every file in a folder
    python wiki_client.py queue
    python wiki_client.py search "your question"
    python wiki_client.py chat "your question"
    python wiki_client.py set-llm --endpoint https://open.bigmodel.cn/api/paas/v4 --model glm-4.7 --api-key-env ZHIPU_API_KEY
    python wiki_client.py llm-ping                                  # test the configured model

Add --project <id-or-path> to target a project other than the current one.
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("WIKI_URL", "http://127.0.0.1:19828").rstrip("/")
TOKEN = os.environ.get("WIKI_TOKEN", "")
# Talk to the server directly. System proxy variables (http_proxy etc.) would otherwise
# route LAN requests through a proxy that usually cannot reach the server and answers
# with confusing 502/503 errors. Set WIKI_USE_PROXY=1 if you really need the proxy.
_OPENER = urllib.request.build_opener() if os.environ.get("WIKI_USE_PROXY") else \
    urllib.request.build_opener(urllib.request.ProxyHandler({}))
TEXT_EXT = {".md", ".markdown", ".txt", ".org", ".html", ".htm", ".csv", ".json"}


def call(method, path, body=None, timeout=600):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", "Bearer " + TOKEN)
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        sys.exit("HTTP %s from %s %s: %s" % (e.code, method, path, detail[:500]))
    except urllib.error.URLError as e:
        sys.exit("Cannot reach %s (%s).\nCheck WIKI_URL, that the server is running, and that "
                 "this machine can reach that IP:port (same network / firewall)." % (BASE, e.reason))


def project_path(project):
    return "/api/v1/projects/%s" % urllib.request.quote(project, safe="")


def upload_payload(path):
    name = os.path.basename(path)
    raw = open(path, "rb").read()
    if os.path.splitext(name)[1].lower() in TEXT_EXT:
        try:
            return {"name": name, "content": raw.decode("utf-8")}
        except UnicodeDecodeError:
            pass
    return {"name": name, "contentBase64": base64.b64encode(raw).decode("ascii")}  # pdf, docx, ...


def wait_for(project, task_ids, poll=2.0, max_wait=3600):
    start = time.time()
    pending = set(task_ids)
    while pending and time.time() - start < max_wait:
        q = call("GET", project_path(project) + "/ingest/queue")
        s = q.get("summary", {})
        live = {t.get("id") for t in q.get("tasks", [])}
        print("  pending=%s processing=%s completed=%s failed=%s" % (
            s.get("pending"), s.get("processing"), s.get("completed"), s.get("failed")))
        if not (pending & live):
            return q
        time.sleep(poll)
    sys.exit("Timed out waiting for ingest.")


def ingest(project, paths, wait=True):
    ids = []
    for p in paths:
        r = call("POST", project_path(project) + "/ingest", {"upload": upload_payload(p)})
        print("queued %s -> %s" % (p, r.get("taskIds")))
        ids += r.get("taskIds", [])
    if wait and ids:
        q = wait_for(project, ids)
        failed = [t for t in q.get("tasks", []) if t.get("status") == "failed"]
        for t in failed:
            print("FAILED:", t.get("source") or t.get("id"), t.get("error"))
        print("done." if not failed else "finished with failures.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="current")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("projects")
    sub.add_parser("queue")
    p = sub.add_parser("ingest-file"); p.add_argument("files", nargs="+"); p.add_argument("--no-wait", action="store_true")
    p = sub.add_parser("ingest-dir"); p.add_argument("dir"); p.add_argument("--no-wait", action="store_true")
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("--top-k", type=int, default=10)
    p = sub.add_parser("chat"); p.add_argument("message")
    p = sub.add_parser("set-llm", help="store the LLM gateway config on the server")
    p.add_argument("--endpoint", required=True); p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="LLM_API_KEY", help="name of the env var holding the API key (default LLM_API_KEY)")
    p.add_argument("--provider", default="custom"); p.add_argument("--max-context", type=int, default=128000)
    sub.add_parser("llm-ping", help="send a tiny prompt through the configured model")
    a = ap.parse_args()

    if a.cmd == "check":
        print("health:", call("GET", "/health", timeout=10))
        r = call("GET", "/api/v1/projects", timeout=10)  # also verifies the token
        print("token ok; projects:", [p.get("name") for p in r.get("projects", [])])
    elif a.cmd == "projects":
        print(json.dumps(call("GET", "/api/v1/projects"), ensure_ascii=False, indent=2))
    elif a.cmd == "queue":
        print(json.dumps(call("GET", project_path(a.project) + "/ingest/queue"), ensure_ascii=False, indent=2))
    elif a.cmd == "ingest-file":
        ingest(a.project, a.files, not a.no_wait)
    elif a.cmd == "ingest-dir":
        files = sorted(os.path.join(d, f) for d, _, fs in os.walk(a.dir) for f in fs if not f.startswith("."))
        print("%d files" % len(files))
        ingest(a.project, files, not a.no_wait)
    elif a.cmd == "search":
        r = call("POST", project_path(a.project) + "/search", {"query": a.query, "topK": a.top_k})
        for x in r.get("results", []):
            print("%6.1f  %s  %s" % (x.get("score", 0), x.get("path"), x.get("title")))
    elif a.cmd == "set-llm":
        key = os.environ.get(a.api_key_env, "")
        if not key:
            sys.exit("Environment variable %s is empty. Run:  export %s='your key'" % (a.api_key_env, a.api_key_env))
        call("POST", "/rpc/app_state_set", {"key": "llmConfig", "value": {
            "provider": a.provider, "apiKey": key, "model": a.model, "customEndpoint": a.endpoint,
            "apiMode": "chat_completions", "maxContextSize": a.max_context}})
        print("saved llmConfig: %s @ %s (key from $%s, not shown)" % (a.model, a.endpoint, a.api_key_env))
    elif a.cmd == "llm-ping":
        r = call("POST", "/rpc/diagnostics_llm_ping", {})
        if r.get("ok"):
            print("OK in %d ms: %r" % (r["ms"], r["reply"]))
        else:
            sys.exit("FAILED: %s\nconfig: %s" % (r.get("error"), r.get("config")))
    elif a.cmd == "chat":
        r = call("POST", project_path(a.project) + "/chat", {"message": a.message})
        print(r.get("message", {}).get("content", r))
        for ref in r.get("references", []):
            print("  -", ref.get("path"))


if __name__ == "__main__":
    main()
