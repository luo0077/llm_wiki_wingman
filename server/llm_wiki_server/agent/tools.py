"""Port of agent/tools.rs: builtin Agent tools, their registry, and wiki/workspace writers.

Conventions
-----------
* Rust ``Result<T, String>`` becomes ``T`` / ``raise CommandError(message)``.
* Rust serde structs become plain dicts with the exact serde (camelCase,
  skip-if-none/empty) JSON shape, so they can be json-dumped directly.
* ``ToolSpec`` is a ``dict`` subclass (JSON shape ``{name, description, effects,
  parameters?}``) with attribute accessors.

Public API (Rust-equivalent names)
----------------------------------
    TOOL_EFFECT_READ/WRITE/NETWORK/PROCESS          ("read" ...)
    ToolSpec(name, description, effects, parameters=None)
    builtin_tool_specs() -> list[ToolSpec]
    ToolContext(project_path, embedding_config=None, web_search_config=None, anytxt_config=None)
    BuiltinToolRegistry().specs() -> list[ToolSpec]
    async BuiltinToolRegistry().execute(name, input, context) -> Any (JSON value)
    async execute_tool(name, input, context)                      (same, module level)
    WebSearchConfig / AnyTxtConfig                  (re-exported from commands.external_search)
    write_wiki_page_with_options(project_path, rel_path, content, allow_overwrite) -> AgentReference dict
    write_wiki_page_with_activity(...)                             -> WikiWriteOutput dict
    write_wiki_page_verified(project_path, rel_path, content, allow_overwrite) -> VerifiedWikiWriteOutput dict
    write_workspace_file / append_workspace_file(project_path, rel_path, content) -> WorkspaceWriteOutput dict
    async run_wiki_search(project_path, query, top_k, include_content, embedding_config=None)
                                                                   -> WikiSearchToolOutput dict
    async run_shell_exec(project_path, command, timeout_secs)      -> ShellExecToolOutput dict
    async run_web_search / run_anytxt_search (re-exported)
    read_wiki_page(project_path, rel_path) -> str
    search_graph(project_path, query, top_k) -> list[AgentReference dict]
    search_sources(project_path, query, top_k) -> list[AgentReference dict]
    build_knowledge_context_index(project_path) -> {path: AgentKnowledgeContext dict}
    attach_latest_version(project_path, relative_path, context_or_none) -> None
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..commands import file_history
from ..commands import search as _search
from ..commands.external_search import (  # noqa: F401  (re-exported, Rust-equivalent location)
    AnyTxtConfig,
    WebSearchConfig,
    file_url_for_path,
    run_anytxt_search,
    run_web_search,
)
from ..errors import CommandError

# Tool I/O limits are backend security boundaries (see tools.rs).
MAX_READ_PAGE_BYTES = 2 * 1024 * 1024
MAX_WRITE_PAGE_BYTES = 2 * 1024 * 1024
MAX_WORKSPACE_WRITE_BYTES = 2 * 1024 * 1024
MAX_WORKSPACE_ROLLBACK_BYTES = 512 * 1024
MAX_SOURCE_SEARCH_FILES = 10_000
MAX_SOURCE_SNIPPET_CHARS = 500
MAX_GRAPH_SEARCH_FILES = 10_000
MAX_KNOWLEDGE_CONTEXT_ITEMS = 20
WEB_SEARCH_TIMEOUT_SECS = 30
SHELL_EXEC_TIMEOUT_SECS = 30
MAX_SHELL_COMMAND_CHARS = 4_000
MAX_SHELL_OUTPUT_CHARS = 20_000
MAX_SHELL_GENERATED_FILES = 50
SHELL_OUTPUT_DRAIN_TIMEOUT_SECS = 1

TOOL_EFFECT_READ = "read"
TOOL_EFFECT_WRITE = "write"
TOOL_EFFECT_NETWORK = "network"
TOOL_EFFECT_PROCESS = "process"

_WIKI_WRITE_LOCK = threading.Lock()


def _agent_workspace():
    from . import workspace  # owned by another agent; imported lazily

    return workspace


# --------------------------------------------------------------------------- specs


class ToolSpec(dict):
    """JSON shape ``{name, description, effects, parameters?}`` with attribute access."""

    def __init__(self, name: str, description: str, effects: List[str],
                 parameters: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, description=description, effects=list(effects))
        if parameters is not None:
            self["parameters"] = parameters

    @property
    def name(self) -> str:
        return self["name"]

    @property
    def description(self) -> str:
        return self["description"]

    @property
    def effects(self) -> List[str]:
        return self["effects"]

    @property
    def parameters(self) -> Optional[Dict[str, Any]]:
        return self.get("parameters")

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


def _query_params() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "topK": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }


def builtin_tool_specs() -> List[ToolSpec]:
    R, W, N, P = TOOL_EFFECT_READ, TOOL_EFFECT_WRITE, TOOL_EFFECT_NETWORK, TOOL_EFFECT_PROCESS
    return [
        ToolSpec("wiki.search",
                 "Search generated LLM Wiki pages using backend keyword/vector retrieval.",
                 [R], _query_params()),
        ToolSpec("wiki.read_page", "Read a project wiki markdown page by project-relative path.",
                 [R], {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}),
        ToolSpec("source.search",
                 "Search raw source files stored under raw/sources for exact keyword snippets.",
                 [R], _query_params()),
        ToolSpec("web.search", "Search external web sources when the user enables web search.",
                 [N], _query_params()),
        ToolSpec("graph.search",
                 "Retrieve graph relationships, neighbors, backlinks, dependencies, and connections "
                 "between project entities. Use concise entity or concept names rather than a full "
                 "question.", [R], _query_params()),
        ToolSpec("anytxt.search", "Search files indexed by an AnyTXT JSON-RPC service.",
                 [N, R], _query_params()),
        ToolSpec("deep_research.run",
                 "Collect broader external/local evidence for deep research turns before synthesis.",
                 [N, R],
                 {"type": "object",
                  "properties": {
                      "query": {"type": "string"},
                      "sources": {"type": "array",
                                  "items": {"enum": ["web", "anytxt", "wiki", "source"]}},
                  },
                  "required": ["query"]}),
        ToolSpec("wiki.write_page",
                 "Create a Markdown wiki page under wiki/ with project-bound path checks. Existing "
                 "files require allowOverwrite=true.",
                 [W],
                 {"type": "object",
                  "properties": {
                      "path": {"type": "string",
                               "description": "Project-relative path such as wiki/queries/new-page.md"},
                      "content": {"type": "string"},
                      "allowOverwrite": {
                          "type": "boolean",
                          "description": "Defaults to false. Set true only when the user explicitly "
                                         "asks to overwrite an existing wiki page."},
                  },
                  "required": ["path", "content"]}),
        ToolSpec("llm.generate", "Generate a final assistant answer from retrieved context.", [N], None),
        ToolSpec("skills.load", "Load instruction-only project skills from .llm-wiki/skills.", [R], None),
        ToolSpec("skill.read_file",
                 "Read a text reference file from an active skill directory by relative path.",
                 [R],
                 {"type": "object",
                  "properties": {
                      "skill": {"type": "string",
                                "description": "Optional active skill name; required when multiple "
                                               "skills are active."},
                      "path": {"type": "string",
                               "description": "Relative path inside the active skill directory, such "
                                              "as references/types.md."},
                  },
                  "required": ["path"]}),
        ToolSpec("workspace.write_file",
                 "Write a generated artifact file under the visible agent-workspace directory.",
                 [W],
                 {"type": "object",
                  "properties": {
                      "path": {"type": "string",
                               "description": "Relative path under agent-workspace, such as "
                                              "cover-image/cover.svg."},
                      "content": {"type": "string"},
                  },
                  "required": ["path", "content"]}),
        ToolSpec("workspace.append_file",
                 "Append generated artifact content under agent-workspace. Use after "
                 "workspace.write_file for large HTML/PPT files.",
                 [W],
                 {"type": "object",
                  "properties": {
                      "path": {"type": "string",
                               "description": "Relative path under agent-workspace, matching the file "
                                              "being appended."},
                      "content": {"type": "string"},
                  },
                  "required": ["path", "content"]}),
        ToolSpec("shell.exec",
                 "Run a project-scoped shell command requested by an active skill instruction.",
                 [R, P],
                 {"type": "object",
                  "properties": {
                      "command": {"type": "string"},
                      "timeoutSeconds": {"type": "integer", "minimum": 1,
                                         "maximum": SHELL_EXEC_TIMEOUT_SECS},
                  },
                  "required": ["command"]}),
    ]


# --------------------------------------------------------------------------- registry


class ToolContext:
    def __init__(self, project_path: str, embedding_config: Any = None,
                 web_search_config: Any = None, anytxt_config: Any = None):
        self.project_path = project_path
        self.embedding_config = embedding_config
        self.web_search_config = web_search_config
        self.anytxt_config = anytxt_config


def _get(input: Any, *keys: str) -> Any:
    """serde_json `input.get(k1).or_else(|| input.get(k2))` (None when absent / not an object)."""
    if not isinstance(input, dict):
        return None
    for k in keys:
        v = input.get(k)
        if v is not None:
            return v
    return None


def _as_str(v: Any) -> Optional[str]:
    return v if isinstance(v, str) else None


def _as_bool(v: Any) -> Optional[bool]:
    return v if isinstance(v, bool) else None


def _as_u64(v: Any) -> Optional[int]:
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        return None
    return v


def tool_query(input: Any, tool: str) -> str:
    q = _as_str(_get(input, "query"))
    q = q.strip() if q is not None else None
    if not q:
        raise CommandError(f"{tool} requires query")
    return q


def tool_top_k(input: Any) -> int:
    v = _as_u64(_get(input, "topK", "top_k"))
    return max(1, min(10, v if v is not None else 5))


class ToolRegistry:
    def specs(self) -> List[ToolSpec]:  # pragma: no cover - interface
        raise NotImplementedError

    async def execute(self, name: str, input: Any, context: ToolContext) -> Any:  # pragma: no cover
        raise NotImplementedError


class BuiltinToolRegistry(ToolRegistry):
    def specs(self) -> List[ToolSpec]:
        return builtin_tool_specs()

    async def execute(self, name: str, input: Any, context: ToolContext) -> Any:
        project = context.project_path
        if name == "wiki.write_page":
            path = _as_str(_get(input, "path"))
            if path is None:
                raise CommandError("wiki.write_page requires path")
            content = _as_str(_get(input, "content"))
            if content is None:
                raise CommandError("wiki.write_page requires content")
            allow = _as_bool(_get(input, "allowOverwrite", "allow_overwrite")) or False
            return await asyncio.to_thread(write_wiki_page_with_activity, project, path, content, allow)
        if name == "wiki.search":
            query = tool_query(input, "wiki.search")
            include_content = _as_bool(_get(input, "includeContent")) or False
            return await run_wiki_search(project, query, tool_top_k(input), include_content,
                                         context.embedding_config)
        if name == "wiki.read_page":
            path = _as_str(_get(input, "path"))
            path = path.strip() if path is not None else None
            if not path:
                raise CommandError("wiki.read_page requires path")
            return await asyncio.to_thread(_read_page_tool, project, path)
        if name == "workspace.write_file":
            path = _as_str(_get(input, "path"))
            if path is None:
                raise CommandError("workspace.write_file requires path")
            content = _as_str(_get(input, "content"))
            if content is None:
                raise CommandError("workspace.write_file requires content")
            return await asyncio.to_thread(write_workspace_file, project, path, content)
        if name == "workspace.append_file":
            path = _as_str(_get(input, "path"))
            if path is None:
                raise CommandError("workspace.append_file requires path")
            content = _as_str(_get(input, "content"))
            if content is None:
                raise CommandError("workspace.append_file requires content")
            return await asyncio.to_thread(append_workspace_file, project, path, content)
        if name == "source.search":
            query = tool_query(input, "source.search")
            # Synchronous filesystem walk: keep it off the event loop.
            return await asyncio.to_thread(search_sources, project, query, tool_top_k(input))
        if name == "graph.search":
            query = tool_query(input, "graph.search")
            return await asyncio.to_thread(search_graph, project, query, tool_top_k(input))
        if name == "web.search":
            query = tool_query(input, "web.search")
            return await run_web_search(query, context.web_search_config, tool_top_k(input))
        if name == "anytxt.search":
            query = tool_query(input, "anytxt.search")
            return await run_anytxt_search(query, context.anytxt_config, tool_top_k(input))
        if name == "deep_research.run":
            query = tool_query(input, "deep_research.run")
            return {"query": query, "status": "orchestrated_by_agent_runtime"}
        if name == "shell.exec":
            command = _as_str(_get(input, "command", "query"))
            if command is None:
                raise CommandError("shell.exec requires command")
            t = _as_u64(_get(input, "timeoutSeconds", "timeout_seconds"))
            timeout_secs = max(1, min(SHELL_EXEC_TIMEOUT_SECS,
                                      t if t is not None else SHELL_EXEC_TIMEOUT_SECS))
            return await run_shell_exec(project, command, timeout_secs)
        raise CommandError(f"Unknown Agent tool: {name}")


async def execute_tool(name: str, input: Any, context: ToolContext) -> Any:
    return await BuiltinToolRegistry().execute(name, input, context)


def _read_page_tool(project: str, path: str) -> Dict[str, Any]:
    content = read_wiki_page(project, path)
    normalized = normalize_rel_path(path)
    ctx = build_knowledge_context_index(project).get(normalized)
    attach_latest_version(project, normalized, ctx)
    return {"path": path, "content": content, "knowledgeContext": ctx}


# --------------------------------------------------------------------------- small text helpers


def _lines(s: str) -> List[str]:
    """Rust `str::lines`."""
    if not s:
        return []
    parts = s.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return [p[:-1] if p.endswith("\r") else p for p in parts]


def trim_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def normalize_rel_path(path: str) -> str:
    return path.strip().replace("\\", "/").lstrip("/")


def is_hidden_rel(rel: str) -> bool:
    return any(seg.startswith(".") for seg in normalize_rel_path(rel).split("/"))


def is_public_read_rel(rel: str) -> bool:
    lower = rel.lower()
    if any(seg.startswith(".") for seg in lower.split("/")):
        return False
    return (lower == "purpose.md" or lower == "schema.md" or lower.startswith("wiki/")
            or lower.startswith("raw/sources/"))


def _has_bad_component(rel: str) -> bool:
    if os.path.isabs(rel) or rel.startswith("/") or rel.startswith("\\"):
        return True
    if re.match(r"^[A-Za-z]:", rel):  # Windows drive prefix
        return True
    return any(seg == ".." for seg in rel.replace("\\", "/").split("/"))


def relative_to_project(project_path: str, path: str) -> str:
    try:
        rel = str(Path(path).relative_to(project_path))
    except ValueError:
        rel = str(path)
    return rel.replace("\\", "/").lstrip("/")


def _canon(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(2, "No such file or directory", path)
    return os.path.realpath(path)


def _starts_with(path: str, root: str) -> bool:
    p, r = Path(path), Path(root)
    return p == r or r in p.parents


def safe_project_join(project_path: str, rel: str) -> str:
    if _has_bad_component(rel):
        raise CommandError("path must be project-relative")
    joined = os.path.join(project_path, rel)
    if os.path.exists(joined):
        try:
            root_canon = _canon(project_path)
        except OSError as e:
            raise CommandError(f"Failed to resolve project path: {e}")
        try:
            joined_canon = _canon(joined)
        except OSError as e:
            raise CommandError(f"Failed to resolve requested path: {e}")
        if not _starts_with(joined_canon, root_canon):
            raise CommandError("path escapes project directory")
    return joined


def ensure_project_bound_path(project_path: str, path: str) -> None:
    try:
        root_canon = _canon(project_path)
    except OSError as e:
        raise CommandError(f"Failed to resolve project path: {e}")
    try:
        path_canon = _canon(path)
    except OSError as e:
        raise CommandError(f"Failed to resolve requested path: {e}")
    if not _starts_with(path_canon, root_canon):
        raise CommandError("path escapes project directory")


def ensure_existing_ancestor_bound(project_path: str, path: str) -> None:
    cursor = path
    while not os.path.exists(cursor):
        parent = os.path.dirname(cursor)
        if not parent or parent == cursor:
            raise CommandError("path must have an existing project ancestor")
        cursor = parent
    ensure_project_bound_path(project_path, cursor)


def validate_portable_path_segment(segment: str) -> None:
    if not segment:
        raise CommandError("wiki.write_page path contains an empty segment")
    if segment.endswith((" ", ".")):
        raise CommandError("wiki.write_page path contains a segment ending with a space or dot, "
                           "which is not portable to Windows")
    if any(ch in '<>:"|?*' or ch <= "\x1f" for ch in segment):
        raise CommandError("wiki.write_page path contains characters that are invalid on Windows")
    stem = segment.split(".")[0].rstrip(" ").upper()
    if stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}:
        raise CommandError("wiki.write_page path uses a Windows reserved device name")


def normalize_wiki_write_path(path: str) -> str:
    trimmed = path.strip()
    if trimmed.startswith("/") or trimmed.startswith("\\"):
        raise CommandError("wiki.write_page path must stay inside the project")
    rel = normalize_rel_path(path)
    lower = rel.lower()
    if not lower.startswith("wiki/") or not lower.endswith(".md"):
        raise CommandError("wiki.write_page path must be a Markdown file under wiki/")
    if any(seg.startswith(".") for seg in lower.split("/")):
        raise CommandError("wiki.write_page cannot write hidden paths")
    if _has_bad_component(rel):
        raise CommandError("wiki.write_page path must stay inside the project")
    for seg in rel.split("/"):
        validate_portable_path_segment(seg)
    return rel


def normalize_workspace_write_path(path: str) -> str:
    rel = normalize_rel_path(path)
    lower = rel.lower()
    if (not rel or lower.startswith("wiki/") or lower.startswith("raw/")
            or any(seg.startswith(".") for seg in lower.split("/"))):
        raise CommandError("workspace.write_file path must be a relative file under agent-workspace")
    if _has_bad_component(rel):
        raise CommandError("workspace.write_file path must stay inside agent-workspace")
    for seg in rel.split("/"):
        try:
            validate_portable_path_segment(seg)
        except CommandError as e:
            raise CommandError(e.message.replace("wiki.write_page", "workspace.write_file"))
    return rel


def extract_markdown_title(content: str) -> Optional[str]:
    for line in _lines(content)[:80]:
        trimmed = line.strip()
        if trimmed.startswith("title:"):
            title = trimmed[len("title:"):].strip().strip('"').strip("'")
            if title:
                return title
        if trimmed.startswith("# "):
            heading = trimmed[2:].strip()
            if heading:
                return heading
    return None


def collapse_markdown_preview(content: str) -> str:
    kept = []
    for line in _lines(content):
        t = line.strip()
        if t and t != "---" and not t.startswith("title:"):
            kept.append(line)
            if len(kept) == 12:
                break
    return " ".join(kept)


def snippet_around_char(content: str, char_idx: int, max_chars: int) -> str:
    start = max(0, min(char_idx, len(content)) - max_chars // 2)
    snippet = content[start:start + max_chars]
    if start > 0:
        snippet = "..." + snippet
    if len(content) > start + max_chars:
        snippet += "..."
    return " ".join(snippet.split())


# --------------------------------------------------------------------------- wiki writing


def _utf8_len(s: str) -> int:
    return len(s.encode("utf-8", errors="surrogatepass"))


def _is_symlink(path: str) -> bool:
    try:
        return os.path.islink(path)
    except OSError:
        return False


def write_wiki_page_with_options(project_path: str, rel_path: str, content: str,
                                 allow_overwrite: bool) -> Dict[str, Any]:
    out = write_wiki_page_with_activity(project_path, rel_path, content, allow_overwrite)
    return {k: v for k, v in out.items() if k not in ("existedBefore", "previousContent")}


def write_wiki_page_verified(project_path: str, rel_path: str, content: str,
                             allow_overwrite: bool) -> Dict[str, Any]:
    out = write_wiki_page_with_activity(project_path, rel_path, content, allow_overwrite)
    return {
        "path": out["path"],
        "bytes": _utf8_len(content),
        "allowOverwrite": bool(allow_overwrite),
        "verified": True,
        "existedBefore": out["existedBefore"],
    }


def _workspace_rollback_snapshot(path: str) -> Optional[str]:
    try:
        st = os.stat(path)
        import stat as _stat

        if not _stat.S_ISREG(st.st_mode) or st.st_size > MAX_WORKSPACE_ROLLBACK_BYTES:
            return None
        with open(path, "rb") as f:
            return f.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _atomic_replace_file(path: str, content: bytes) -> None:
    parent = os.path.dirname(path)
    file_name = os.path.basename(path)
    if not parent or not file_name:
        raise CommandError("Failed to resolve wiki page parent directory")
    nonce = time.time_ns()
    temp = os.path.join(parent, f".{file_name}.llm-wiki-{os.getpid()}-{nonce}.tmp")
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except OSError as e:
        raise CommandError(f"Failed to create wiki page temp file: {e}")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        try:
            os.remove(temp)
        except OSError:
            pass
        raise CommandError(f"Failed to write wiki page temp file: {e}")
    try:
        os.replace(temp, path)
    except OSError as e:
        try:
            os.remove(temp)
        except OSError:
            pass
        raise CommandError(f"Failed to replace wiki page: {e}")


def write_wiki_page_with_activity(project_path: str, rel_path: str, content: str,
                                  allow_overwrite: bool) -> Dict[str, Any]:
    """Returns WikiWriteOutput: AgentReference fields (flattened) + existedBefore [+ previousContent]."""
    with _WIKI_WRITE_LOCK:
        if _utf8_len(content) > MAX_WRITE_PAGE_BYTES:
            raise CommandError("wiki.write_page content is too large")
        rel = normalize_wiki_write_path(rel_path)
        path = safe_project_join(project_path, rel)
        parent = os.path.dirname(path)
        if parent:
            # Check the deepest existing ancestor before creating directories.
            ensure_existing_ancestor_bound(project_path, parent)
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as e:
                raise CommandError(f"Failed to create wiki page directory: {e}")
            ensure_project_bound_path(project_path, parent)
        # Create-only by default.
        if os.path.exists(path) and not allow_overwrite:
            raise CommandError("wiki.write_page refuses to overwrite an existing page without "
                               "allowOverwrite=true")
        if _is_symlink(path):
            raise CommandError("wiki.write_page refuses to overwrite a symlink")
        existed_before = os.path.isfile(path)
        previous_bytes: Optional[bytes] = None
        if existed_before:
            try:
                with open(path, "rb") as f:
                    previous_bytes = f.read()
            except OSError as e:
                raise CommandError(f"Failed to read existing wiki page before replacement: {e}")
        previous_content = _workspace_rollback_snapshot(path)
        file_history.record_file_version(path, "baseline", "before.wiki.write_page")
        data = content.encode("utf-8", errors="surrogatepass")
        _atomic_replace_file(path, data)
        try:
            with open(path, "rb") as f:
                persisted = f.read()
        except OSError as e:
            raise CommandError(f"Failed to verify persisted wiki page: {e}")
        if persisted != data:
            try:
                if previous_bytes is not None:
                    _atomic_replace_file(path, previous_bytes)
                else:
                    try:
                        os.remove(path)
                    except OSError as e:
                        raise CommandError(f"Failed to remove unverified page: {e}")
                rollback_err = None
            except CommandError as e:
                rollback_err = e.message
            if rollback_err is None:
                raise CommandError("Persisted wiki page did not match the requested content; the "
                                   "previous state was restored")
            raise CommandError("Persisted wiki page did not match the requested content, and "
                               f"rollback failed: {rollback_err}")
        file_history.record_file_version(path, "agent", "wiki.write_page")
        title = extract_markdown_title(content)
        if title is None:
            stem = Path(rel).stem or "Wiki page"
            title = stem.replace("-", " ")
        out: Dict[str, Any] = {"title": title, "path": rel, "kind": "wiki"}
        snippet = trim_text(collapse_markdown_preview(content), 500)
        if snippet.strip():
            out["snippet"] = snippet
        out["existedBefore"] = existed_before
        if previous_content is not None:
            out["previousContent"] = previous_content
        return out


# --------------------------------------------------------------------------- workspace files


def _resolve_workspace_write_target(project_path: str, rel_path: str,
                                    tool_name: str) -> Tuple[str, str]:
    try:
        rel = normalize_workspace_write_path(rel_path)
    except CommandError as e:
        raise CommandError(e.message.replace("workspace.write_file", tool_name))
    if not os.path.isdir(project_path):
        raise CommandError(f"{tool_name} project directory is not available")
    workspace = str(_agent_workspace().agent_workspace_path(project_path))
    try:
        os.makedirs(workspace, exist_ok=True)
    except OSError as e:
        raise CommandError(f"{tool_name} failed to create workspace: {e}")
    ensure_project_bound_path(project_path, workspace)
    path = os.path.join(workspace, rel)
    parent = os.path.dirname(path)
    if parent:
        ensure_existing_ancestor_bound(project_path, parent)
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            raise CommandError(f"{tool_name} failed to create directory: {e}")
        ensure_project_bound_path(project_path, parent)
    return rel, path


def _workspace_output(rel: str, nbytes: int, existed_before: bool,
                      previous_content: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": f"{_agent_workspace().AGENT_WORKSPACE_DIR}/{rel}",
        "bytes": nbytes,
        "existedBefore": existed_before,
    }
    if previous_content is not None:
        out["previousContent"] = previous_content
    return out


def write_workspace_file(project_path: str, rel_path: str, content: str) -> Dict[str, Any]:
    if _utf8_len(content) > MAX_WORKSPACE_WRITE_BYTES:
        raise CommandError("workspace.write_file content is too large")
    rel, path = _resolve_workspace_write_target(project_path, rel_path, "workspace.write_file")
    if _is_symlink(path):
        raise CommandError("workspace.write_file refuses to overwrite a symlink")
    existed_before = os.path.isfile(path)
    previous_content = _workspace_rollback_snapshot(path)
    file_history.record_file_version(path, "baseline", "before.workspace.write_file")
    try:
        with open(path, "wb") as f:
            f.write(content.encode("utf-8", errors="surrogatepass"))
    except OSError as e:
        raise CommandError(f"workspace.write_file failed: {e}")
    file_history.record_file_version(path, "agent", "workspace.write_file")
    return _workspace_output(rel, _utf8_len(content), existed_before, previous_content)


def append_workspace_file(project_path: str, rel_path: str, content: str) -> Dict[str, Any]:
    if _utf8_len(content) > MAX_WORKSPACE_WRITE_BYTES:
        raise CommandError("workspace.append_file content is too large")
    rel, path = _resolve_workspace_write_target(project_path, rel_path, "workspace.append_file")
    if _is_symlink(path):
        raise CommandError("workspace.append_file refuses to overwrite a symlink")
    existed_before = os.path.isfile(path)
    previous_content = _workspace_rollback_snapshot(path)
    file_history.record_file_version(path, "baseline", "before.workspace.append_file")
    try:
        with open(path, "ab") as f:
            f.write(content.encode("utf-8", errors="surrogatepass"))
    except OSError as e:
        raise CommandError(f"workspace.append_file failed: {e}")
    file_history.record_file_version(path, "agent", "workspace.append_file")
    try:
        nbytes = os.stat(path).st_size
    except OSError:
        nbytes = _utf8_len(content)
    return _workspace_output(rel, nbytes, existed_before, previous_content)


# --------------------------------------------------------------------------- wiki search


async def run_wiki_search(project_path: str, query: str, top_k: int, include_content: bool,
                          embedding_config: Any = None) -> Dict[str, Any]:
    query_embedding = await _search.resolve_query_embedding(query, None, embedding_config)
    search = await _search.search_project_inner(project_path, query, top_k, include_content,
                                                query_embedding)
    try:
        context_index = await asyncio.to_thread(build_knowledge_context_index, project_path)
    except Exception:  # noqa: BLE001 - optional enrichment must not fail retrieval
        context_index = {}
    references = []
    for result in search["results"]:
        normalized = normalize_rel_path(result["path"])
        ctx = context_index.pop(normalized, None)
        related = result.get("graphRelatedTo") or []
        if ctx is not None:
            if related:
                ctx["relatedTo"] = list(related)
            else:
                ctx.pop("relatedTo", None)
        attach_latest_version(project_path, normalized, ctx)
        ref: Dict[str, Any] = {
            "title": result["title"],
            "path": result["path"],
            "kind": "graph" if related else "wiki",
        }
        if result["snippet"].strip():
            ref["snippet"] = result["snippet"]
        ref["score"] = result["score"]
        if ctx is not None:
            ref["knowledgeContext"] = ctx
        references.append(ref)
    return {
        "mode": search["mode"],
        "tokenHits": search["tokenHits"],
        "vectorHits": search["vectorHits"],
        "graphHits": search["graphHits"],
        "references": references,
    }


# --------------------------------------------------------------------------- shell.exec


def _preserve_env(env: Dict[str, str], keys: List[str]) -> None:
    for key in keys:
        if key in os.environ:
            env[key] = os.environ[key]


def apply_sanitized_shell_env(project_path: str, workspace: str) -> Dict[str, str]:
    """Environment minimization, not an OS sandbox (see tools.rs)."""
    env: Dict[str, str] = {}
    _preserve_env(env, ["PATH", "LANG", "LC_ALL", "LC_CTYPE"])
    if os.name == "nt":
        _preserve_env(env, ["ComSpec", "SystemRoot", "WINDIR", "PATHEXT", "USERPROFILE",
                            "USERNAME", "HOMEDRIVE", "HOMEPATH", "TEMP", "TMP", "APPDATA",
                            "LOCALAPPDATA", "ProgramData"])
    else:
        _preserve_env(env, ["HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "XDG_CONFIG_HOME",
                            "XDG_CACHE_HOME", "XDG_DATA_HOME"])
    env["LLM_WIKI_PROJECT"] = project_path
    env["LLM_WIKI_PROJECT_PATH"] = project_path
    env["LLM_WIKI_AGENT_WORKSPACE"] = workspace
    return env


def _workspace_file_content_hash(path: str, length: int) -> Optional[str]:
    full_hash_limit = 8 * 1024 * 1024
    edge = 64 * 1024
    try:
        h = hashlib.blake2b(digest_size=16)
        h.update(str(length).encode())
        with open(path, "rb") as f:
            if length <= full_hash_limit:
                h.update(f.read())
            else:
                h.update(f.read(edge))
                f.seek(-edge, os.SEEK_END)
                h.update(f.read(edge))
        return h.hexdigest()
    except OSError:
        return None


def snapshot_workspace_files(workspace: str) -> Dict[str, Tuple[int, int, Optional[str]]]:
    files: Dict[str, Tuple[int, int, Optional[str]]] = {}
    for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
        for name in filenames:
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                continue
            try:
                st = os.stat(full)
            except OSError:
                continue
            import stat as _stat

            if not _stat.S_ISREG(st.st_mode):
                continue
            rel = os.path.relpath(full, workspace).replace("\\", "/")
            files[rel] = (st.st_size, st.st_mtime_ns,
                          _workspace_file_content_hash(full, st.st_size))
    return files


def changed_workspace_files(workspace: str, before: Dict[str, Tuple[int, int, Optional[str]]]
                            ) -> List[Dict[str, Any]]:
    after = snapshot_workspace_files(workspace)
    out: List[Dict[str, Any]] = []
    prefix = _agent_workspace().AGENT_WORKSPACE_DIR
    for rel in sorted(after):
        snap = after[rel]
        if before.get(rel) == snap:
            continue
        out.append({"path": f"{prefix}/{rel}", "bytes": snap[0], "existedBefore": rel in before})
        if len(out) >= MAX_SHELL_GENERATED_FILES:
            break
    return out


async def _read_limited_output(reader: asyncio.StreamReader, max_chars: int) -> str:
    kept = bytearray()
    max_bytes = max_chars * 4
    while True:
        try:
            chunk = await reader.read(8192)
        except Exception:  # noqa: BLE001
            break
        if not chunk:
            break
        if len(kept) < max_bytes:
            kept.extend(chunk[:max_bytes - len(kept)])
    text = bytes(kept).decode("utf-8", errors="replace")
    if len(text) > max_chars:
        text = trim_text(text, max_chars)
    return text


async def _await_shell_output(task: "asyncio.Task[str]", label: str) -> str:
    done, _pending = await asyncio.wait({task}, timeout=SHELL_OUTPUT_DRAIN_TIMEOUT_SECS)
    if task in done:
        try:
            return task.result()
        except Exception as e:  # noqa: BLE001
            raise CommandError(f"shell.exec {label} task failed: {e}")
    task.cancel()
    return f"[{label} output was still open after command exit and was truncated]"


async def run_shell_exec(project_path: str, command: str, timeout_secs: int) -> Dict[str, Any]:
    command = command.strip()
    if not command:
        raise CommandError("shell.exec command is empty")
    if len(command) > MAX_SHELL_COMMAND_CHARS:
        raise CommandError("shell.exec command is too long")
    if not os.path.isdir(project_path):
        raise CommandError("shell.exec project directory is not available")
    ws = _agent_workspace()
    workspace = str(ws.agent_workspace_path(project_path))
    try:
        os.makedirs(workspace, exist_ok=True)
    except OSError as e:
        raise CommandError(f"shell.exec failed to create {ws.AGENT_WORKSPACE_DIR}: {e}")
    ensure_project_bound_path(project_path, workspace)
    before_files = await asyncio.to_thread(snapshot_workspace_files, workspace)
    env = apply_sanitized_shell_env(project_path, workspace)
    if os.name == "nt":
        argv = [os.environ.get("ComSpec", "cmd"), "/C", command]
    else:
        argv = ["/bin/sh", "-c", command]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=workspace, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except OSError as e:
        raise CommandError(f"shell.exec failed to start: {e}")
    stdout_task = asyncio.ensure_future(_read_limited_output(proc.stdout, MAX_SHELL_OUTPUT_CHARS))
    stderr_task = asyncio.ensure_future(_read_limited_output(proc.stderr, MAX_SHELL_OUTPUT_CHARS))
    timed_out = False
    timeout_message: Optional[str] = None
    exit_code: Optional[int] = None
    try:
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout=timeout_secs)
            exit_code = rc if rc >= 0 else None  # killed by signal -> no code
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            timed_out = True
            timeout_message = f"Command timed out after {timeout_secs}s"
        stdout = await _await_shell_output(stdout_task, "stdout")
        stderr = await _await_shell_output(stderr_task, "stderr")
    finally:
        # kill_on_drop
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        for t in (stdout_task, stderr_task):
            if not t.done():
                t.cancel()
    if timeout_message:
        if stderr:
            stderr += "\n"
        stderr += timeout_message
    generated = await asyncio.to_thread(changed_workspace_files, workspace, before_files)
    prefix = f"{ws.AGENT_WORKSPACE_DIR}/"
    for output in generated:
        if output["path"].startswith(prefix):
            file_history.record_file_version(os.path.join(workspace, output["path"][len(prefix):]),
                                             "agent", "shell.exec")
    return {
        "command": command,
        "exitCode": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "timedOut": timed_out,
        "generatedFiles": generated,
    }


# --------------------------------------------------------------------------- knowledge graph


class KnowledgePage:
    __slots__ = ("path", "title", "stem", "tags", "links", "matches_query", "neighbors")

    def __init__(self, path, title, stem, tags, links, matches_query):
        self.path = path
        self.title = title
        self.stem = stem
        self.tags = tags
        self.links = links
        self.matches_query = matches_query
        self.neighbors: List[str] = []


_QUERY_SPLIT = re.compile(r"[\s,，;；:：/|]+")
_SOURCE_QUERY_SPLIT = re.compile(r"[\s,，;；:：]+")


def graph_query_terms(query: str) -> List[str]:
    return [t.strip() for t in _QUERY_SPLIT.split(query) if t.strip()]


def source_query_terms(query: str) -> List[str]:
    stop = {"raw", "source", "sources", "file", "files", "原始资料", "原始文件", "源文件"}
    return [t for t in (x.strip() for x in _SOURCE_QUERY_SPLIT.split(query))
            if len(t) >= 2 and t not in stop]


def normalize_wiki_link(value: str) -> str:
    v = value.split("#")[0].strip()
    while v.endswith(".md"):
        v = v[:-3]
    return v.replace("\\", "/").lower().replace(" ", "-")


def extract_wikilinks(content: str) -> List[str]:
    links: List[str] = []
    rest = content
    while True:
        start = rest.find("[[")
        if start < 0:
            break
        rest = rest[start + 2:]
        end = rest.find("]]")
        if end < 0:
            break
        target = rest[:end].split("|")[0].strip()
        if target:
            links.append(target)
        rest = rest[end + 2:]
    return links


def extract_frontmatter_list(content: str, key: str) -> List[str]:
    normalized = content.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return []
    rest = normalized[4:]
    end = rest.find("\n---")
    if end < 0:
        return []
    prefix = f"{key}:"
    lines = _lines(rest[:end])
    for index, line in enumerate(lines):
        t = line.strip()
        if not t.startswith(prefix):
            continue
        inline = t[len(prefix):].strip()
        if inline.startswith("[") and inline.endswith("]") and len(inline) >= 2:
            items = (i.strip().strip("'\"") for i in inline[1:-1].split(","))
            return [i for i in items if i]
        values: List[str] = []
        for nxt in lines[index + 1:]:
            tr = nxt.strip()
            if not tr.startswith("-"):
                break
            item = tr[1:].strip().strip("'\"")
            if item:
                values.append(item)
        return values
    return []


def _walk_files(root: str) -> List[str]:
    """Regular (non-symlink) files under root, deterministic order, symlinked dirs not followed."""
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                continue
            out.append(full)
    return out


def build_knowledge_graph_snapshot(project_path: str, query: Optional[str]
                                   ) -> Tuple[List[KnowledgePage], Dict[str, Dict[str, Any]]]:
    wiki_root = os.path.join(project_path, "wiki")
    if not os.path.isdir(wiki_root):
        return [], {}
    pages: List[KnowledgePage] = []
    for full in _walk_files(wiki_root):
        if len(pages) >= MAX_GRAPH_SEARCH_FILES or Path(full).suffix != ".md":
            continue
        try:
            with open(full, "rb") as f:
                content = f.read().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        path = relative_to_project(project_path, full)
        if is_hidden_rel(path):
            continue
        title = _search.extract_title(content, os.path.basename(full))
        matches = False
        if query is not None:
            haystack = f"{title} {path} {content}".lower()
            matches = query in haystack or any(t in haystack for t in graph_query_terms(query))
        pages.append(KnowledgePage(path, title, Path(full).stem, extract_frontmatter_list(content, "tags"),
                                   extract_wikilinks(content), matches))

    aliases: Dict[str, str] = {}
    for page in pages:
        wiki_rel = page.path[5:] if page.path.startswith("wiki/") else page.path
        for alias in (page.stem, page.title, page.path, wiki_rel):
            aliases[normalize_wiki_link(alias)] = page.path

    backlinks: Dict[str, set] = {}
    neighbors: Dict[str, set] = {}
    for page in pages:
        for link in page.links:
            target = aliases.get(normalize_wiki_link(link))
            if target is not None and target != page.path:
                backlinks.setdefault(target, set()).add(page.path)
                neighbors.setdefault(page.path, set()).add(target)
                neighbors.setdefault(target, set()).add(page.path)

    for page in pages:
        page.neighbors = sorted(neighbors.pop(page.path, set()))[:MAX_KNOWLEDGE_CONTEXT_ITEMS]

    contexts: Dict[str, Dict[str, Any]] = {}
    for page in pages:
        outgoing = sorted(set(page.links))
        total_outgoing = len(outgoing)
        outgoing = outgoing[:MAX_KNOWLEDGE_CONTEXT_ITEMS]
        page_backlinks = sorted(backlinks.pop(page.path, set()))
        total_backlinks = len(page_backlinks)
        page_backlinks = page_backlinks[:MAX_KNOWLEDGE_CONTEXT_ITEMS]
        ctx: Dict[str, Any] = {}
        tags = page.tags[:MAX_KNOWLEDGE_CONTEXT_ITEMS]
        if tags:
            ctx["tags"] = tags
        if outgoing:
            ctx["outgoingLinks"] = outgoing
        if page_backlinks:
            ctx["backlinks"] = page_backlinks
        ctx["linkCount"] = total_outgoing + total_backlinks
        contexts[page.path] = ctx
    return pages, contexts


def build_knowledge_context_index(project_path: str) -> Dict[str, Dict[str, Any]]:
    return build_knowledge_graph_snapshot(project_path, None)[1]


def attach_latest_version(project_path: str, relative_path: str,
                          context: Optional[Dict[str, Any]]) -> None:
    if context is None:
        return
    v = file_history.latest_file_version(os.path.join(project_path, relative_path))
    if v is None:
        context.pop("latestVersion", None)
    else:
        context["latestVersion"] = {"timestamp": v[0], "author": v[1], "tool": v[2]}


# --------------------------------------------------------------------------- read / graph / source


def read_wiki_page(project_path: str, rel_path: str) -> str:
    rel = normalize_rel_path(rel_path)
    if not is_public_read_rel(rel) or not rel.lower().startswith("wiki/"):
        raise CommandError("wiki.read_page path must stay under wiki/")
    path = safe_project_join(project_path, rel)
    try:
        st = os.stat(path)
    except OSError as e:
        raise CommandError(f"Failed to read page metadata: {e}")
    import stat as _stat

    if not _stat.S_ISREG(st.st_mode):
        raise CommandError("wiki.read_page path is not a file")
    if st.st_size > MAX_READ_PAGE_BYTES:
        raise CommandError("wiki.read_page file is too large")
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise CommandError(f"Failed to read wiki page: {e}")


def search_graph(project_path: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    query = query.strip().lower()
    if not query:
        return []
    pages, context_index = build_knowledge_graph_snapshot(project_path, query)
    seeds = {p.path for p in pages if p.matches_query}
    if not seeds:
        return []
    refs: List[Dict[str, Any]] = []
    for page in pages:
        page_ctx = context_index.get(page.path)
        if page_ctx is None:
            continue
        link_count = page_ctx["linkCount"]
        connected = any(n in seeds for n in page.neighbors)
        if not page.matches_query and not connected:
            continue
        relation = "matched entity" if page.matches_query else "direct neighbor"
        ctx = context_index.pop(page.path, None)
        attach_latest_version(project_path, page.path, ctx)
        ref: Dict[str, Any] = {
            "title": page.title,
            "path": page.path,
            "kind": "graph",
            "snippet": f"{relation}; {link_count} related link(s)",
            "score": (10_000.0 if page.matches_query else 5_000.0) + float(link_count),
        }
        if ctx is not None:
            ref["knowledgeContext"] = ctx
        refs.append(ref)
    refs.sort(key=lambda r: (-r["score"], r["path"]))
    return refs[:max(1, min(10, top_k))]


_TEXT_EXTS = {"md", "markdown", "org", "txt", "json", "csv", "tsv", "yaml", "yml", "xml", "html"}
_CACHED_EXTS = {"pdf", "doc", "docx", "docm", "ppt", "pps", "pot", "pptx", "pptm", "ppsx", "ppsm",
                "xls", "xlsx", "xlsm", "xlsb", "odt", "ods", "odp", "rtf", "epub", "mobi"}


def search_sources(project_path: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    query = query.strip()
    if not query:
        raise CommandError("source.search query is required")
    root = os.path.join(project_path, "raw", "sources")
    if not os.path.exists(root):
        return []
    lower_query = query.lower()
    terms = source_query_terms(lower_query)
    limit = max(1, min(10, top_k))
    refs: List[Dict[str, Any]] = []
    seen = 0
    for full in _walk_files(root):
        rel = relative_to_project(project_path, full)
        if is_hidden_rel(rel):
            continue
        seen += 1
        if seen > MAX_SOURCE_SEARCH_FILES:
            import sys

            print(f"[Agent] source.search stopped after {MAX_SOURCE_SEARCH_FILES} files in "
                  f"{project_path}", file=sys.stderr)
            break
        ext = Path(full).suffix[1:].lower() if Path(full).suffix else ""
        if not ext:
            continue
        if ext in _TEXT_EXTS:
            try:
                with open(full, "rb") as f:
                    content = f.read().decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
        elif ext in _CACHED_EXTS:
            from ..commands import fs as _fs

            content = _fs.read_preprocessed_cache(full)
            if content is None:
                continue
        else:
            continue
        lower = content.lower()
        idx = -1
        for term in [lower_query, *terms]:
            i = lower.find(term)
            if i >= 0:
                idx = i
                break
        if idx < 0:
            continue
        refs.append({
            "title": os.path.basename(full) or rel,
            "path": rel,
            "kind": "source",
            "snippet": snippet_around_char(content, idx, MAX_SOURCE_SNIPPET_CHARS),
        })
        if len(refs) >= limit:
            break
    return refs
