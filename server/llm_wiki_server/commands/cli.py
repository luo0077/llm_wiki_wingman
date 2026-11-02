"""Port of commands/cli_resolver.rs, claude_cli.rs and codex_cli.rs.

Events (published on the shared bus, delivered over SSE ``GET /events``):
  * ``claude-cli:<stream_id>``        payload: str  (one stdout line, no newline)
  * ``claude-cli:<stream_id>:done``   payload: {"code": int|None, "stderr": str}
  * ``codex-cli:<stream_id>``         payload: str  (one stdout line)
  * ``codex-cli:<stream_id>:done``    payload: {"code": int|None, "stderr": str, "stdout": str}
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

from .. import events
from ..errors import CommandError
from ..registry import command

CLI_VERSION_PROBE_TIMEOUT = 10.0
LOGIN_SHELL_PATH_TIMEOUT = 10.0
PATH_MARKER = "\x1e"
_IS_WINDOWS = sys.platform.startswith("win")
_STREAM_LIMIT = 256 * 1024 * 1024  # asyncio readline limit; stream-json lines can be huge


# --------------------------------------------------------------------------
# cli_resolver.rs
# --------------------------------------------------------------------------

def cli_version_timeout_error(cmd: str) -> str:
    return f"`{cmd} --version` timed out after {int(CLI_VERSION_PROBE_TIMEOUT)}s"


_resolved_commands: Dict[str, str] = {}
_shell_path_cache: Optional[str] = None
_shell_path_lock = threading.Lock()


def parse_shell_path_output(stdout: str) -> Optional[str]:
    prefix = f"{PATH_MARKER}PATH="
    start = stdout.find(prefix)
    while start != -1:
        ps = start + len(prefix)
        end = stdout.find(PATH_MARKER, ps)
        if end != -1:
            path = stdout[ps:end]
            if path and "\r" not in path and "\n" not in path:
                return path
        start = stdout.find(prefix, start + 1)
    return None


def login_shell_path(timeout: float = LOGIN_SHELL_PATH_TIMEOUT) -> Optional[str]:
    shell = os.environ.get("SHELL") or "/bin/sh"
    name = os.path.basename(shell).lower()
    flag = "-ic" if name in ("sh", "dash", "ash") else "-ilc"
    script = "printf '\\036PATH=%s\\036\\n' \"$PATH\""
    try:
        proc = subprocess.run([shell, flag, script], stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_shell_path_output(proc.stdout.decode("utf-8", "replace"))


def resolve_login_shell_path() -> Optional[str]:
    global _shell_path_cache
    if _IS_WINDOWS:
        return None
    with _shell_path_lock:
        if _shell_path_cache is not None:
            return _shell_path_cache
        path = login_shell_path()
        if path is None:
            return None  # failures are not cached
        _shell_path_cache = path
        return path


def merge_child_path_env(shell_path: str, inherited: Optional[str]) -> str:
    if inherited:
        return f"{shell_path}:{inherited}"
    return shell_path


async def child_path_env() -> Optional[str]:
    if _IS_WINDOWS:
        return None
    sp = await asyncio.get_running_loop().run_in_executor(None, resolve_login_shell_path)
    if sp is None:
        return None
    return merge_child_path_env(sp, os.environ.get("PATH"))


def _find_uncached(cmd: str, windows_candidates: List[str]) -> str:
    if _IS_WINDOWS:
        for cand in list(windows_candidates) + [cmd]:
            p = shutil.which(cand)
            if p:
                return p
        raise CommandError(f"`{cmd}` not found on PATH")
    p = shutil.which(cmd)
    if p:
        return p
    full = resolve_login_shell_path()
    if full:
        p = shutil.which(cmd, path=full)
        if p:
            return p
    raise CommandError(f"`{cmd}` not found on PATH")


async def find_cli_command(cmd: str, windows_candidates: Optional[List[str]] = None) -> str:
    cached = _resolved_commands.get(cmd)
    if cached:
        if os.path.exists(cached):
            return cached
        _resolved_commands.pop(cmd, None)
    path = await asyncio.get_running_loop().run_in_executor(
        None, _find_uncached, cmd, list(windows_candidates or []))
    _resolved_commands[cmd] = path
    return path


async def _child_env() -> Optional[Dict[str, str]]:
    path_env = await child_path_env()
    if path_env is None:
        return None
    env = dict(os.environ)
    env["PATH"] = path_env
    return env


def _status_text(code: Optional[int]) -> str:
    if code is None:
        return "signal"
    if code < 0:
        return f"signal: {-code}"
    return f"exit status: {code}"


async def _detect(name: str, finder, extra_quarantine: bool) -> Dict[str, Any]:
    try:
        path = await finder()
    except CommandError as e:
        return {"installed": False, "version": None, "path": None, "error": e.message}
    env = await _child_env()
    try:
        proc = await asyncio.create_subprocess_exec(
            path, "--version", stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
    except OSError as e:
        return {"installed": False, "version": None, "path": path,
                "error": f"Failed to spawn `{name}`: {e}"}
    try:
        out, err = await asyncio.wait_for(proc.communicate(), CLI_VERSION_PROBE_TIMEOUT)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return {"installed": False, "version": None, "path": path,
                "error": cli_version_timeout_error(name)}
    if proc.returncode == 0:
        return {"installed": True, "version": out.decode("utf-8", "replace").strip(),
                "path": path, "error": None}
    stderr = err.decode("utf-8", "replace").strip()
    if extra_quarantine and ("quarantine" in stderr or "damaged" in stderr):
        error = f"Binary quarantined — try: xattr -d com.apple.quarantine {path}"
    elif not stderr:
        error = f"`{name} --version` exited with {_status_text(proc.returncode)}"
    else:
        error = stderr
    return {"installed": False, "version": None, "path": path, "error": error}


async def _resolve_working_directory(value: Optional[str], label: str, log: str) -> str:
    raw = (value or "").strip()
    if not raw:
        raise CommandError(f"{label} requires an active project working directory")
    if not os.path.isabs(raw):
        raise CommandError(f"{label} working directory must be an absolute project path")
    try:
        st = os.stat(raw)
    except OSError as e:
        print(f"[{log}] failed to read working directory metadata {raw}: {e}", file=sys.stderr)
        raise CommandError(f"{label} working directory does not exist or cannot be read: {raw}")
    import stat as _stat
    if not _stat.S_ISDIR(st.st_mode):
        raise CommandError(f"{label} working directory is not a directory: {raw}")
    index = os.path.join(raw, "wiki", "index.md")
    try:
        ist = os.stat(index)
    except OSError as e:
        print(f"[{log}] failed to read wiki/index.md metadata for {raw}: {e}", file=sys.stderr)
        raise CommandError(
            f"{label} working directory must be an LLM Wiki project containing wiki/index.md: {raw}")
    if not _stat.S_ISREG(ist.st_mode):
        raise CommandError(
            f"{label} working directory must be an LLM Wiki project containing wiki/index.md: {raw}")
    try:
        return os.path.realpath(raw)
    except OSError as e:
        raise CommandError(f"Failed to canonicalize {label} working directory {raw}: {e}")


def _kill_proc(proc) -> None:
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


# --------------------------------------------------------------------------
# claude_cli.rs
# --------------------------------------------------------------------------

ISOLATED_MCP_CONFIG = '{"mcpServers":{}}'
_claude_children: Dict[str, Any] = {}


async def _find_claude() -> str:
    return await find_cli_command("claude", ["claude.cmd", "claude.exe"])


def claude_content_text_only(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def claude_content_blocks(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out: List[Dict[str, Any]] = []
    for b in content:
        t = b.get("type") if isinstance(b, dict) else None
        if t == "text":
            out.append({"type": "text", "text": b["text"]})
        elif t == "image":
            out.append({"type": "image", "source": {
                "type": "base64", "media_type": b["mediaType"], "data": b["dataBase64"]}})
        else:
            raise CommandError("invalid args `messages`: unknown content block")
    return out


def merge_system_preamble_into_user_content(content: List[Dict[str, Any]], preamble: str) -> None:
    if not preamble:
        return
    for i, block in enumerate(content):
        if block.get("type") != "text" or not isinstance(block.get("text"), str):
            continue
        content[i] = {"type": "text", "text": f"{preamble}\n\n{block['text']}"}
        return
    content.insert(0, {"type": "text", "text": preamble})


def build_claude_cli_args(model: str, isolate_local_config: bool) -> List[str]:
    args = ["-p", "--output-format", "stream-json", "--input-format", "stream-json", "--verbose"]
    if isolate_local_config:
        args += ["--setting-sources", "project", "--disable-slash-commands", "--tools", "",
                 "--no-session-persistence", "--prompt-suggestions", "false"]
    args += ["--model", model]
    if isolate_local_config:
        args += ["--strict-mcp-config", "--mcp-config", ISOLATED_MCP_CONFIG]
    return args


@command()
async def claude_cli_detect():
    return await _detect("claude", _find_claude, True)


async def _drain(proc, stream_id: str, prefix: str, children: Dict[str, Any], *,
                 collect_stdout: bool, stderr_limit: Optional[int], timeout_state: Optional[dict]):
    topic = f"{prefix}:{stream_id}"
    done_topic = f"{prefix}:{stream_id}:done"

    async def read_stderr() -> str:
        collected = ""
        while True:
            try:
                raw = await proc.stderr.readline()
            except (ValueError, OSError):
                break
            if not raw:
                break
            line = raw.decode("utf-8", "replace").rstrip("\n").rstrip("\r")
            print(f"[{prefix} stderr] {line}", file=sys.stderr)
            if stderr_limit is None:
                collected += line + "\n"
            else:
                collected = append_capped_line(collected, line, stderr_limit)
        return collected

    stderr_task = asyncio.ensure_future(read_stderr())
    stdout_text = ""
    while True:
        try:
            raw = await proc.stdout.readline()
        except (ValueError, OSError) as e:
            print(f"[{prefix} stdout] read error: {e}", file=sys.stderr)
            break
        if not raw:
            break
        line = raw.decode("utf-8", "replace").rstrip("\n").rstrip("\r")
        if collect_stdout:
            stdout_text = append_capped_line(stdout_text, line, STDOUT_LIMIT_BYTES)
        events.publish(topic, line)

    owned = children.pop(stream_id, None)
    exit_code: Optional[int] = None
    try:
        rc = await proc.wait()
        # Killed through claude_cli_kill / timeout -> removed from map -> code None.
        if owned is not None:
            exit_code = rc if rc >= 0 else None
    except Exception:
        exit_code = None
    try:
        stderr_text = await stderr_task
    except Exception:
        stderr_text = ""
    return exit_code, stderr_text, stdout_text, done_topic


@command()
async def claude_cli_spawn(stream_id: str, model: str, messages: List[dict],
                           isolate_local_config: bool = False,
                           working_directory: Optional[str] = None):
    system_preamble = "\n\n".join(
        claude_content_text_only(m["content"]) for m in messages if m.get("role") == "system")
    conversation = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not conversation:
        raise CommandError("No user/assistant messages to send to claude CLI")

    first_user_seen = False
    turns = []
    for m in conversation:
        role = m["role"]
        content = claude_content_blocks(m["content"])
        if not first_user_seen and role == "user" and system_preamble:
            merge_system_preamble_into_user_content(content, system_preamble)
            first_user_seen = True
        turns.append((role, content))

    cwd = await _resolve_working_directory(
        working_directory, "Claude Code CLI", "claude-cli")
    claude = await _find_claude()
    env = await _child_env()
    try:
        proc = await asyncio.create_subprocess_exec(
            claude, *build_claude_cli_args(model, isolate_local_config), cwd=cwd, env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=_STREAM_LIMIT)
    except OSError as e:
        raise CommandError(f"Failed to spawn claude: {e}")

    try:
        for role, content in turns:
            event = {"type": role, "message": {"role": role, "content": content}}
            proc.stdin.write((json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        await proc.stdin.drain()
    except (OSError, ConnectionError) as e:
        _kill_proc(proc)
        raise CommandError(f"Failed to write to claude stdin: {e}")
    try:
        proc.stdin.close()
    except OSError:
        pass

    _claude_children[stream_id] = proc

    async def run():
        code, stderr_text, _, done_topic = await _drain(
            proc, stream_id, "claude-cli", _claude_children,
            collect_stdout=False, stderr_limit=None, timeout_state=None)
        events.publish(done_topic, {"code": code, "stderr": stderr_text})

    _spawn_task(run())
    return None


_tasks: set = set()


def _spawn_task(coro) -> None:
    t = asyncio.ensure_future(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


@command()
async def claude_cli_kill(stream_id: str):
    proc = _claude_children.pop(stream_id, None)
    if proc is not None:
        _kill_proc(proc)
    return None


# --------------------------------------------------------------------------
# codex_cli.rs
# --------------------------------------------------------------------------

DEFAULT_CODEX_SPAWN_TIMEOUT_MINUTES = 10
MIN_CODEX_SPAWN_TIMEOUT_MINUTES = 1
MAX_CODEX_SPAWN_TIMEOUT_MINUTES = 240
STDERR_LIMIT_BYTES = 1024 * 1024
STDOUT_LIMIT_BYTES = 1024 * 1024
_codex_children: Dict[str, Any] = {}


def append_capped_line(collected: str, line: str, limit_bytes: int) -> str:
    """Byte-capped append (limit counted in UTF-8 bytes, never splitting a char)."""
    size = len(collected.encode("utf-8"))
    if size >= limit_bytes:
        return collected
    out = []
    for ch in line:
        n = len(ch.encode("utf-8"))
        if size + n > limit_bytes:
            break
        out.append(ch)
        size += n
    if size < limit_bytes:
        out.append("\n")
    return collected + "".join(out)


def codex_spawn_timeout_minutes(value: Optional[int]) -> int:
    v = DEFAULT_CODEX_SPAWN_TIMEOUT_MINUTES if value is None else int(value)
    return max(MIN_CODEX_SPAWN_TIMEOUT_MINUTES, min(MAX_CODEX_SPAWN_TIMEOUT_MINUTES, v))


def build_codex_cli_args(model: str, isolate_local_config: bool) -> List[str]:
    args = ["-a", "never", "exec"]
    if isolate_local_config:
        args += ["--ignore-user-config", "--ignore-rules"]
    args += ["--json", "--skip-git-repo-check", "--sandbox", "read-only", "--ephemeral",
             "--model", model, "-"]
    return args


async def _find_codex() -> str:
    return await find_cli_command("codex", ["codex.cmd", "codex.exe"])


@command()
async def codex_cli_detect():
    return await _detect("codex", _find_codex, False)


@command()
async def codex_cli_spawn(stream_id: str, model: str, prompt: str,
                          isolate_local_config: bool = False,
                          timeout_minutes: Optional[int] = None,
                          working_directory: Optional[str] = None):
    if not prompt.strip():
        raise CommandError("No prompt to send to codex CLI")
    cwd = await _resolve_working_directory(working_directory, "Codex CLI", "codex-cli")
    codex = await _find_codex()
    env = await _child_env()
    try:
        proc = await asyncio.create_subprocess_exec(
            codex, *build_codex_cli_args(model, isolate_local_config), cwd=cwd, env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=_STREAM_LIMIT)
    except OSError as e:
        raise CommandError(f"Failed to spawn codex: {e}")
    try:
        proc.stdin.write(prompt.encode("utf-8"))
        await proc.stdin.drain()
    except (OSError, ConnectionError) as e:
        _kill_proc(proc)
        raise CommandError(f"Failed to write to codex stdin: {e}")
    try:
        proc.stdin.close()
    except OSError:
        pass

    _codex_children[stream_id] = proc
    minutes = codex_spawn_timeout_minutes(timeout_minutes)
    timed_out = {"v": False}

    async def watchdog():
        await asyncio.sleep(minutes * 60)
        p = _codex_children.pop(stream_id, None)
        if p is not None:
            timed_out["v"] = True
            _kill_proc(p)

    wd = asyncio.ensure_future(watchdog())

    async def run():
        try:
            code, stderr_text, stdout_text, done_topic = await _drain(
                proc, stream_id, "codex-cli", _codex_children,
                collect_stdout=True, stderr_limit=STDERR_LIMIT_BYTES, timeout_state=timed_out)
        finally:
            wd.cancel()
        if timed_out["v"]:
            if stderr_text:
                stderr_text += "\n"
            stderr_text += f"Codex CLI timed out after {minutes} minutes."
        elif len(stderr_text.encode("utf-8")) >= STDERR_LIMIT_BYTES:
            stderr_text += "\n[stderr truncated]"
        if len(stdout_text.encode("utf-8")) >= STDOUT_LIMIT_BYTES:
            stdout_text += "\n[stdout truncated]"
        events.publish(done_topic, {
            "code": -1 if timed_out["v"] else code,
            "stderr": stderr_text,
            "stdout": stdout_text,
        })

    _spawn_task(run())
    return None


@command()
async def codex_cli_kill(stream_id: str):
    proc = _codex_children.pop(stream_id, None)
    if proc is not None:
        _kill_proc(proc)
    return None
