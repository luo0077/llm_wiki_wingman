from __future__ import annotations

import asyncio
import os
import stat

import pytest

from llm_wiki_server import events
from llm_wiki_server.commands import cli
from llm_wiki_server.errors import CommandError


def run(coro):
    return asyncio.run(coro)


def test_claude_args():
    a = cli.build_claude_cli_args("sonnet", False)
    assert "--model" in a and "--setting-sources" not in a and "--mcp-config" not in a
    b = cli.build_claude_cli_args("sonnet", True)
    assert b[-3:] == ["--strict-mcp-config", "--mcp-config", cli.ISOLATED_MCP_CONFIG]
    assert "--no-session-persistence" in b


def test_codex_args_and_timeout():
    assert cli.build_codex_cli_args("m", False)[:3] == ["-a", "never", "exec"]
    assert "--ignore-rules" in cli.build_codex_cli_args("m", True)
    assert cli.codex_spawn_timeout_minutes(None) == 10
    assert cli.codex_spawn_timeout_minutes(0) == 1
    assert cli.codex_spawn_timeout_minutes(9999) == 240


def test_append_capped_line():
    assert cli.append_capped_line("", "hello", 16) == "hello\n"
    out = cli.append_capped_line("", "abcdef", 4)
    assert out == "abcd" and cli.append_capped_line(out, "x", 4) == "abcd"
    assert cli.append_capped_line("", "é水x", 5) == "é水"


def test_preamble_merge():
    blocks = [{"type": "text", "text": "hi"}, {"type": "image", "source": {}}]
    cli.merge_system_preamble_into_user_content(blocks, "SYS")
    assert blocks[0]["text"] == "SYS\n\nhi" and len(blocks) == 2
    img = [{"type": "image", "source": {}}]
    cli.merge_system_preamble_into_user_content(img, "SYS")
    assert img[0] == {"type": "text", "text": "SYS"} and len(img) == 2
    assert cli.claude_content_blocks([{"type": "image", "mediaType": "image/png", "dataBase64": "abc"}]) == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}]


def test_parse_shell_path_output():
    assert cli.parse_shell_path_output("Welcome\n\x1ePATH=/a:/b\x1e\nbye") == "/a:/b"
    assert cli.parse_shell_path_output("nothing") is None


def test_working_directory_validation(tmp_path):
    f = lambda v: run(cli._resolve_working_directory(v, "Claude Code CLI", "claude-cli"))
    with pytest.raises(CommandError, match="active project"):
        f(None)
    with pytest.raises(CommandError, match="active project"):
        f("   ")
    with pytest.raises(CommandError, match="absolute"):
        f("relative/path")
    with pytest.raises(CommandError, match="wiki/index.md"):
        f(str(tmp_path))
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "index.md").write_text("# i")
    assert f(str(tmp_path)) == os.path.realpath(tmp_path)


def _fake_cli(dirpath, name, body):
    p = dirpath / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def fake_path(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    cli._resolved_commands.clear()
    monkeypatch.setattr(cli, "_shell_path_cache", None)
    monkeypatch.setattr(cli, "login_shell_path", lambda *a, **k: None)
    return bindir


def test_detect_missing_and_present(fake_path, monkeypatch):
    monkeypatch.setenv("PATH", str(fake_path))
    r = run(cli.claude_cli_detect())
    assert r["installed"] is False and r["path"] is None and "not found" in r["error"]
    _fake_cli(fake_path, "claude", 'echo "1.2.3 (Claude Code)"\n')
    r = run(cli.claude_cli_detect())
    assert r == {"installed": True, "version": "1.2.3 (Claude Code)", "path": str(fake_path / "claude"), "error": None}
    _fake_cli(fake_path, "codex", 'echo "damaged quarantine" >&2; exit 3\n')
    cli._resolved_commands.clear()
    r = run(cli.codex_cli_detect())
    assert r["installed"] is False and r["error"] == "damaged quarantine"


def test_claude_spawn_streams_and_done(fake_path, tmp_path):
    _fake_cli(fake_path, "claude",
              'cat > "$0.stdin"\necho \'{"type":"a"}\'\necho second\necho err >&2\nexit 2\n')
    proj = tmp_path / "proj"
    (proj / "wiki").mkdir(parents=True)
    (proj / "wiki" / "index.md").write_text("x")
    seen = []
    remove = events.add_listener(lambda t, p: seen.append((t, p)))

    async def go():
        await cli.claude_cli_spawn(
            "s1", "sonnet",
            [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}],
            False, str(proj))
        for _ in range(100):
            if any(t == "claude-cli:s1:done" for t, _ in seen):
                break
            await asyncio.sleep(0.05)

    try:
        run(go())
    finally:
        remove()
    assert ("claude-cli:s1", '{"type":"a"}') in seen and ("claude-cli:s1", "second") in seen
    done = [p for t, p in seen if t == "claude-cli:s1:done"][0]
    assert done == {"code": 2, "stderr": "err\n"}
    sent = (fake_path / "claude.stdin").read_text()
    assert '"text":"SYS\\n\\nhello"' in sent


def test_codex_spawn_and_kill(fake_path, tmp_path):
    _fake_cli(fake_path, "codex", 'cat\nexec sleep 30\n')
    proj = tmp_path / "proj"
    (proj / "wiki").mkdir(parents=True)
    (proj / "wiki" / "index.md").write_text("x")
    seen = []
    remove = events.add_listener(lambda t, p: seen.append((t, p)))

    async def go():
        with pytest.raises(CommandError, match="No prompt"):
            await cli.codex_cli_spawn("c0", "m", "  ", False, None, str(proj))
        await cli.codex_cli_spawn("c1", "m", "ping\n", False, None, str(proj))
        for _ in range(100):
            if any(t == "codex-cli:c1" for t, _ in seen):
                break
            await asyncio.sleep(0.05)
        await cli.codex_cli_kill("c1")
        for _ in range(100):
            if any(t == "codex-cli:c1:done" for t, _ in seen):
                break
            await asyncio.sleep(0.05)

    try:
        run(go())
    finally:
        remove()
    assert ("codex-cli:c1", "ping") in seen
    done = [p for t, p in seen if t == "codex-cli:c1:done"][0]
    assert done["code"] is None and done["stdout"] == "ping\n"
