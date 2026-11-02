# LLM Wiki headless server

A Python (FastAPI) replacement for the Tauri/Rust backend. It needs no display
server: every feature is reachable over HTTP.

* `POST /rpc/<command>` - every ported `#[tauri::command]`, same name and arguments
  (camelCase or snake_case keys). `GET /rpc` lists the registered commands.
* `/api/v1/...` - the documented REST API (port of `api_server.rs`), described below.
* `GET /health` - liveness probe (`/api/v1/health` returns the detailed status).

## Install

Requires Python >= 3.9.

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -e .                  # core: fastapi, uvicorn, httpx
pip install -e '.[extract,vector,watch]'   # optional: document extraction, LanceDB vectors, file watcher
pip install -e '.[test]' && python -m pytest
```

## Run

```bash
python -m llm_wiki_server --host 0.0.0.0 --port 19828
```

* `--host 127.0.0.1` (default) keeps the API local; use `0.0.0.0` to allow LAN access.
  This is the equivalent of the desktop "Allow LAN access" switch: the Rust app picked
  the bind host from `LLM_WIKI_BIND_HOST` or `apiConfig.allowLanAccess`; the same policy is
  available as `llm_wiki_server.commands.rest_api.configured_bind_host()`.
* State (settings, project registry, API token) lives in `app-state.json` under
  `$LLM_WIKI_DATA_DIR` (default `~/.llm-wiki-server`). An `app-state.json` exported from the
  desktop app can be dropped in as-is.

### Authentication

Everything under `/api/v1` except `/api/v1/health` requires a token unless
`apiConfig.allowUnauthenticated` is true. Even then, `pages/write`, `pages/embed` and chat
always require the token. Token sources, in order: env `LLM_WIKI_API_TOKEN`, then
`apiConfig.token` in `app-state.json`. Supply it as

* `Authorization: Bearer <token>`, or
* `X-LLM-Wiki-Token: <token>`, or
* `?token=<token>` query parameter.

```bash
export LLM_WIKI_DATA_DIR=/srv/llm-wiki
export LLM_WIKI_API_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))')
python -m llm_wiki_server --host 0.0.0.0 --port 19828
```

`apiConfig.enabled=false` makes every endpoint but `/health` answer `503`.

## RPC surface

```bash
curl -s localhost:19828/rpc | head -c 300
curl -s -X POST localhost:19828/rpc/create_project \
     -H 'Content-Type: application/json' -d '{"name":"my-wiki","path":"/srv/wikis"}'
curl -s -X POST localhost:19828/rpc/open_project \
     -H 'Content-Type: application/json' -d '{"path":"/srv/wikis/my-wiki"}'
```

Errors are `{"ok": false, "error": "..."}` with HTTP 400 (command error), 404 (unknown
command) or 500.

## REST API (`/api/v1`)

Conventions: JSON in and out; errors are `{"ok": false, "error": "<message>"}`.
`{id}` is a project id, a project path (URL-encoded) or the literal `current`.
Status codes used: 400 bad request, 401 unauthorized, 403 path not exposed, 404 not
found, 405 method not allowed, 409 conflict, 413 too large, 415 unsupported content,
429 rate limited, 499 chat cancelled, 501 agent runtime not available, 502 upstream error,
503 disabled/busy. Limits: 120 requests/s (health exempt), 64 requests in flight, 8 chat
streams, 4 page-embed jobs, 1 MiB bodies (12 MiB for `pages/write`, 40 MiB for chat).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | status, auth/config flags (no auth) |
| GET | `/api/v1/projects` | list projects + `currentProject` |
| GET | `/api/v1/projects/{id}/files?root=wiki\|sources\|all&recursive=true&maxFiles=2000` | file tree |
| GET | `/api/v1/projects/{id}/files/content?path=wiki/index.md` | read a text file (<= 2 MiB) |
| GET | `/api/v1/projects/{id}/reviews?status=unresolved\|resolved\|all&type=&limit=200` | review items |
| PATCH | `/api/v1/projects/{id}/reviews/{reviewId}` body `{resolved?, action?}` | resolve / reopen one item |
| POST | `/api/v1/projects/{id}/reviews/resolve` body `{ids:[...], action?}` | bulk resolve |
| POST | `/api/v1/projects/{id}/search` body `{query, topK?, includeContent?, queryEmbedding?}` | hybrid search |
| GET | `/api/v1/projects/{id}/graph?q=&nodeType=&offset=0&limit=200&edgeScope=page\|filtered` | knowledge graph page |
| POST | `/api/v1/projects/{id}/sources/rescan` | rescan raw sources and process the queue |
| POST | `/api/v1/projects/{id}/pages/write` body `{path, content, allowOverwrite?}` | write a wiki page (token required) |
| POST | `/api/v1/projects/{id}/pages/embed` body `{path, force?}` | (re)index one page (token required) |
| POST | `/api/v1/projects/{id}/chat` body `{message, sessionId?, runId?, stream?, ...}` | agent chat, JSON or SSE (token required) |
| POST | `/api/v1/projects/{id}/chat/{sessionId}/cancel` | cancel a running chat (token required) |

```bash
B=http://localhost:19828/api/v1
H="Authorization: Bearer $LLM_WIKI_API_TOKEN"

curl -s $B/health
curl -s -H "$H" $B/projects
curl -s -H "$H" "$B/projects/current/files?root=sources&recursive=false"
curl -s -H "$H" "$B/projects/current/files/content?path=wiki/index.md"
curl -s -H "$H" "$B/projects/current/graph?q=attention&limit=50&edgeScope=filtered"

curl -s -H "$H" "$B/projects/current/reviews?status=all&type=missing-page&limit=20"
curl -s -X PATCH -H "$H" -H 'Content-Type: application/json' \
     -d '{"resolved":true,"action":"Skip"}' $B/projects/current/reviews/review-dbdcf949
curl -s -X POST -H "$H" -H 'Content-Type: application/json' \
     -d '{"ids":["review-dbdcf949","review-fa5d9960"],"action":"Done"}' $B/projects/current/reviews/resolve

curl -s -X POST -H "$H" -H 'Content-Type: application/json' \
     -d '{"query":"transformers","topK":5,"includeContent":false}' $B/projects/current/search

curl -s -X POST -H "$H" -H 'Content-Type: application/json' \
     -d '{"path":"wiki/concepts/new-idea.md","content":"---\ntype: concept\n---\n# New idea\n","allowOverwrite":false}' \
     $B/projects/current/pages/write
curl -s -X POST -H "$H" -H 'Content-Type: application/json' \
     -d '{"path":"wiki/concepts/new-idea.md","force":false}' $B/projects/current/pages/embed
curl -s -X POST -H "$H" $B/projects/current/sources/rescan

# Chat, single JSON response
curl -s -X POST -H "$H" -H 'Content-Type: application/json' \
     -d '{"message":"Summarise the wiki"}' $B/projects/current/chat

# Chat, Server-Sent Events (stream:true or Accept: text/event-stream)
curl -N -X POST -H "$H" -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
     -d '{"message":"Summarise the wiki","sessionId":"s1","runId":"r1"}' $B/projects/current/chat

curl -s -X POST -H "$H" $B/projects/current/chat/s1/cancel
```

SSE stream: `event: meta` (`projectId`, `sessionId`, `runId`), zero or more
`event: agent` frames, a `: keepalive` comment every 10 s while idle, and exactly one terminal
frame: `event: done` (full response), `event: cancelled` or `event: error`
(`{"ok":false,"error":...}`). Closing the connection cancels the run.

The chat endpoints require the agent runtime (`llm_wiki_server.agent.runtime`); until it is
installed they answer `501 {"ok":false,"error":"agent runtime not available"}`.

## Headless setup notes

- **`/rpc/*` and `/events` auth**: stricter than `/api/v1`. If a token is configured
  (`LLM_WIKI_API_TOKEN` or `apiConfig.token`) every request needs it
  (`Authorization: Bearer ...`), even with `allowUnauthenticated`. With no token only
  loopback clients are accepted; remote callers get 403. These endpoints read/write
  arbitrary absolute paths, so always set a token before binding to `0.0.0.0`.
- **No GUI, so configure through RPC** (the desktop app did this through its store):
  - `POST /rpc/register_project {"path": ...}` adds a project to `recentProjects`/`lastProject`
    (this is what `/api/v1/projects` lists).
  - `POST /rpc/app_state_set {"key": "llmConfig", "value": {...}}` (also `embeddingConfig`,
    `searchApiConfig`, `apiConfig`, ...). `app_state_get` reads a key.
- **Optional dependencies**: `pip install '.[extract,vector,watch]'`. Without `lancedb` the
  vector index uses sqlite (separate from any index the desktop app built: re-index).
- **Not supported**: legacy binary `.doc/.ppt/.pps/.pot/.xlsb` extraction, Huffman/DRM MOBI.

## Deploying on a Linux box (pip, Python >= 3.9)

```bash
git clone <your repo> && cd llm_wiki/server        # later: git pull
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                    # optional: pip install -r requirements-optional.txt
export LLM_WIKI_API_TOKEN=<long random string>
python -m llm_wiki_server --host 127.0.0.1 --port 19828
# another terminal
export WIKI_URL=http://127.0.0.1:19828 WIKI_TOKEN=<same token>
python examples/selftest.py          # add --llm after configuring llmConfig
```
Run it from the `server/` directory (no `pip install .` needed). After `git pull`, restart the server;
re-run `pip install -r requirements.txt` only if that file changed. Verified on Python 3.9 (821 tests).
