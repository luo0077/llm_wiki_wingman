#!/usr/bin/env python3
"""End-to-end self-check for a running LLM Wiki server. Standard library only.

    export WIKI_URL=http://<server-ip>:19828
    export WIKI_TOKEN=<token>
    python selftest.py            # basics: network, auth, files, search, PDF/DOCX extraction
    python selftest.py --llm      # + your LLM gateway, embeddings, and a full ingest of a small note

Everything runs inside a throwaway project under the server's data dir and is removed
afterwards (use --keep to inspect it). The previously "current" project is restored.
Exit code is 1 if any check FAILED. Paste the whole output when asking for help.
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zipfile

BASE = os.environ.get("WIKI_URL", "http://127.0.0.1:19828").rstrip("/")
TOKEN = os.environ.get("WIKI_TOKEN", "")
_OPENER = urllib.request.build_opener() if os.environ.get("WIKI_USE_PROXY") else \
    urllib.request.build_opener(urllib.request.ProxyHandler({}))


class HttpError(Exception):
    def __init__(self, status, body):
        Exception.__init__(self, "HTTP %s: %s" % (status, body[:300]))
        self.status, self.body = status, body


def http(method, path, body=None, token=TOKEN, timeout=120):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, e.read().decode("utf-8", "replace"))


def rpc(_command, **args):
    return http("POST", "/rpc/" + _command, args)


RESULTS = []


def check(name, fn, fatal=False):
    t0 = time.time()
    try:
        out = fn()
        status, detail = ("WARN", out[1]) if isinstance(out, tuple) and out[0] == "warn" else \
                         ("SKIP", out[1]) if isinstance(out, tuple) and out[0] == "skip" else \
                         ("PASS", out if isinstance(out, str) else "")
    except Exception as e:  # noqa
        status, detail = "FAIL", "%s" % (e if str(e) else type(e).__name__)
        if os.environ.get("WIKI_DEBUG"):
            traceback.print_exc()
    RESULTS.append((name, status, detail))
    print("[%-4s] %-34s %s  (%.1fs)" % (status, name, detail, time.time() - t0), flush=True)
    if fatal and status == "FAIL":
        finish(1, "Cannot continue: fix the failure above first.")
    return status == "PASS"


def finish(code=None, note=None):
    n = {s: sum(1 for r in RESULTS if r[1] == s) for s in ("PASS", "WARN", "SKIP", "FAIL")}
    print("\nSummary: %(PASS)d passed, %(WARN)d warnings, %(SKIP)d skipped, %(FAIL)d failed" % n)
    if note:
        print(note)
    sys.exit(code if code is not None else (1 if n["FAIL"] else 0))


def tiny_pdf(text):
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"]
    stream = "BT /F1 18 Tf 20 100 Td (%s) Tj ET" % text
    objs.append("<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += ("%d 0 obj\n%s\nendobj\n" % (i, o)).encode()
    xref = len(out)
    out += ("xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)).encode()
    out += "".join("%010d 00000 n \n" % o for o in offs).encode()
    out += ("trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)).encode()
    return out


def tiny_docx(text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr("word/document.xml", '<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>%s</w:t></w:r></w:p></w:body></w:document>' % text)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", action="store_true", help="also test the LLM gateway, embeddings and a full ingest")
    ap.add_argument("--keep", action="store_true", help="keep the throwaway project")
    ap.add_argument("--ingest-timeout", type=int, default=600)
    a = ap.parse_args()
    print("Server: %s   token: %s\n" % (BASE, "set" if TOKEN else "NOT SET (WIKI_TOKEN)"))
    S = {}

    # ---- 1. network + auth
    def c_health():
        try:
            h = http("GET", "/health", timeout=15)
        except urllib.error.URLError as e:
            raise Exception("cannot reach %s (%s). Same network? firewall? WIKI_URL correct?" % (BASE, e.reason))
        if not h.get("ok"):
            raise Exception(str(h))
        return "server %s" % h.get("version")
    check("network: /health", c_health, fatal=True)

    def c_auth():
        try:
            http("GET", "/api/v1/projects", token="definitely-wrong")
            raise Exception("server accepted a wrong token - auth is OFF")
        except HttpError as e:
            if e.status != 401:
                raise Exception("wrong token gave HTTP %s, expected 401" % e.status)
        try:
            http("GET", "/api/v1/projects")
        except HttpError as e:
            if e.status == 401:
                raise Exception("the server REJECTED your WIKI_TOKEN (401). It must equal the token the server was started with (LLM_WIKI_API_TOKEN)")
            raise
        return "wrong token -> 401, your token accepted"
    check("auth: token", c_auth, fatal=True)

    # ---- 2. environment report
    def c_env():
        d = rpc("diagnostics_info")
        S["env"] = d
        py = tuple(int(x) for x in d["python"].split(".")[:2])
        if py < (3, 9):
            raise Exception("Python %s < 3.9 (unsupported)" % d["python"])
        return "python %s, vector backend: %s" % (d["python"], d["vectorBackend"])
    check("server: environment", c_env)
    env = S.get("env", {})

    def c_data():
        if not env.get("dataDirWritable"):
            raise Exception("data dir not writable: %s" % env.get("dataDir"))
        return env["dataDir"]
    check("server: data dir writable", c_data, fatal=True)

    def c_libs():
        libs = env.get("libs", {})
        need = [k for k in ("fastapi", "uvicorn", "httpx", "pyyaml") if not libs.get(k, {}).get("installed")]
        if need:
            raise Exception("missing required: %s" % ", ".join(need))
        missing = ["%s (%s)" % (k, v["usedFor"]) for k, v in libs.items() if not v["installed"]]
        if missing:
            return ("warn", "optional not installed: " + "; ".join(missing))
        return "all optional libraries present"
    check("server: libraries", c_libs)
    if env.get("vectorBackend") == "sqlite":
        pass  # reported in the line above; not an error

    # ---- 3. project + files
    root = (env.get("dataDir") or ".").rstrip("/\\") + "/selftest"
    name = "selftest-%d" % int(time.time())
    S["prev_last"] = S["prev_recent"] = None
    try:
        S["prev_last"] = rpc("app_state_get", key="lastProject")
        S["prev_recent"] = rpc("app_state_get", key="recentProjects")
    except Exception:
        pass

    def c_create():
        p = rpc("create_project", name=name, path=root)
        S["proj"] = p["path"]
        return p["path"]
    check("project: create", c_create, fatal=True)
    P = S["proj"]

    def c_files():
        f = P + "/wiki/concepts/selftest-rag.md"
        # edit history is off by default (same as the desktop app); turn it on for this project
        rpc("set_file_history_settings", project_path=P, settings={"enabled": True, "maxVersionsPerFile": 20})
        body = "---\ntype: concept\ntitle: Selftest RAG\n---\n# Selftest RAG\nRetrieval augmented generation zebrafish42.\n"
        rpc("write_file", path=f, contents=body)
        if rpc("read_file", path=f) != body:
            raise Exception("read-back differs from what was written")
        rpc("write_file", path=f, contents=body + "second version\n")
        if not rpc("file_exists", path=f):
            raise Exception("file_exists false")
        names = [n["name"] for n in rpc("list_directory", path=P + "/wiki")]
        if "concepts" not in names:
            raise Exception("list_directory missing 'concepts': %s" % names)
        return "write/read/list OK"
    check("files: write, read, list", c_files)

    def c_hist():
        h = rpc("list_file_history", project_path=P, file_path=P + "/wiki/concepts/selftest-rag.md")
        n = len(h) if isinstance(h, list) else len(h.get("versions", h.get("entries", [])))
        if n < 1:
            raise Exception("no history recorded after 2 writes")
        return "%d versions recorded" % n
    check("files: edit history", c_hist)

    def c_search():
        r = rpc("search_project", project_path=P, query="zebrafish42", top_k=5)
        hits = [x["path"] for x in r.get("results", [])]
        if not any("selftest-rag" in h for h in hits):
            raise Exception("keyword search did not find the page; got %s" % hits)
        return "found via %s" % r.get("mode")
    check("search: keyword", c_search)

    # ---- 4. document extraction
    def extract(fname, data, needle, lib):
        if not env.get("libs", {}).get(lib, {}).get("installed"):
            return ("skip", "%s not installed" % lib)
        f = P + "/raw/sources/" + fname
        rpc("write_file_base64", path=f, base64=base64.b64encode(data).decode("ascii"))
        txt = rpc("read_file", path=f)
        if needle not in txt:
            raise Exception("text not extracted; got: %r" % txt[:120])
        return "extracted text OK"
    check("extract: PDF", lambda: extract("t.pdf", tiny_pdf("Hello Pdf Zebrafish"), "Zebrafish", "pymupdf")
          if env.get("libs", {}).get("pymupdf", {}).get("installed") else extract("t.pdf", tiny_pdf("Hello Pdf Zebrafish"), "Zebrafish", "pypdf"))
    check("extract: DOCX", lambda: extract("t.docx", tiny_docx("Hello Docx Zebrafish"), "Zebrafish", "python-docx"))

    # ---- 5. optional: LLM, embeddings, ingest
    if a.llm:
        def c_llm():
            r = rpc("diagnostics_llm_ping")
            if not r.get("ok"):
                raise Exception("%s   [config: %s]" % (r.get("error"), r.get("config")))
            return "reply %r in %dms (%s/%s)" % (r["reply"][:40], r["ms"], r["config"]["provider"], r["config"]["model"])
        llm_ok = check("llm: gateway reachable", c_llm)

        def c_emb():
            r = rpc("diagnostics_embedding_ping")
            if r.get("skipped"):
                return ("skip", "embeddingConfig not set (vector search disabled; keyword search still works)")
            if not r.get("ok"):
                raise Exception("%s   [config: %s]" % (r.get("error"), r.get("config")))
            return "%d dimensions in %dms" % (r["dimensions"], r["ms"])
        check("llm: embeddings", c_emb)

        def c_ingest():
            if not llm_ok:
                return ("skip", "LLM check failed")
            rpc("register_project", path=P, make_current=True)
            note = "# Selftest note\n\nZebrafish are small freshwater fish used in biology research.\n"
            r = http("POST", "/api/v1/projects/current/ingest", {"upload": {"name": "selftest-note.md", "content": note}})
            ids = r.get("taskIds") or []
            if not ids:
                raise Exception("no task created: %s" % r)
            t0 = time.time()
            while time.time() - t0 < a.ingest_timeout:
                q = http("GET", "/api/v1/projects/current/ingest/queue")
                s = q.get("summary", {})
                if not (s.get("pending") or s.get("processing")):
                    break
                time.sleep(3)
            else:
                raise Exception("still running after %ss (see: python wiki_client.py queue)" % a.ingest_timeout)
            q = http("GET", "/api/v1/projects/current/ingest/queue")
            failed = [t for t in q.get("tasks", []) if t.get("status") == "failed"]
            if failed:
                raise Exception("ingest failed: %s" % (failed[0].get("error") or failed[0]))
            made = [n["name"] for n in rpc("list_directory", path=P + "/wiki/sources")]
            if not made:
                raise Exception("ingest completed but wrote no source page in wiki/sources")
            return "created %s; queue: %s" % (", ".join(made[:3]), q.get("summary"))
        check("ingest: full pipeline", c_ingest)

        def c_chat():
            if not llm_ok:
                return ("skip", "LLM check failed")
            r = http("POST", "/api/v1/projects/current/chat", {"message": "What are zebrafish?"}, timeout=300)
            msg = r.get("message", {})
            text = msg.get("content") if isinstance(msg, dict) else msg
            if not text:
                raise Exception("empty answer: %s" % json.dumps(r)[:300])
            if "iteration limit" in text or "did not produce a final answer" in text:
                return ("warn", "the model never produced a valid final answer (it must reply with JSON actions): %r" % text[:100])
            return "answer: %r, %d references" % (text[:60], len(r.get("references", [])))
        check("chat: Q&A with retrieval", c_chat)
    else:
        print("[SKIP] llm/ingest/chat checks              (run with --llm to include them)")

    # ---- cleanup
    if not a.keep:
        def c_clean():
            try:
                rpc("unregister_project", path=P)
                if S["prev_recent"] is not None:
                    rpc("app_state_set", key="recentProjects", value=S["prev_recent"])
                rpc("app_state_set", key="lastProject", value=S["prev_last"])
            except HttpError as e:
                raise Exception("could not restore project list: %s" % e)
            rpc("delete_file", path=P)
            return "removed %s, restored project list" % P
        check("cleanup", c_clean)
    else:
        print("kept project at %s (it is also registered as current)" % P)
    finish()


if __name__ == "__main__":
    main()
