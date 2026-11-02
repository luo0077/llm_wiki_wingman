"""Port of agent/skills.rs (skill discovery/loading + the `agent_list_skills` command)."""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..registry import command

MAX_SKILL_FILE_BYTES = 64_000
MAX_SKILL_SCAN_DEPTH = 8

_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


@dataclass
class AgentSkill:
    name: str
    description: str
    instructions: str
    base_dir: str
    location: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "baseDir": self.base_dir,
            "location": self.location,
        }


@dataclass
class AvailableAgentSkill:
    id: str
    name: str
    description: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "description": self.description, "source": self.source}


@dataclass
class SkillRoot:
    path: Path
    source: str


@dataclass
class SkillCandidate:
    id: str
    path: Path


class _SkillError(Exception):
    pass


@command()
def agent_list_skills(project_path: str) -> List[Dict[str, Any]]:
    return [s.to_dict() for s in list_available_skills(project_path)]


def load_project_skills(project_path: str, requested: List[str]) -> List[AgentSkill]:
    if not requested:
        return []
    roots = skill_roots(project_path)
    names = sorted({n for n in (normalize_skill_name(r) for r in requested) if n is not None})
    out: List[AgentSkill] = []
    for name in names:
        s = load_one_skill_from_roots(roots, name)
        if s is not None:
            out.append(s)
    return out


def list_available_skills(project_path: str) -> List[AvailableAgentSkill]:
    skills: Dict[str, AvailableAgentSkill] = {}
    for root in skill_roots(project_path):
        for cand in discover_skill_candidates(root.path):
            try:
                skill = load_skill_path(cand.path, cand.id)
            except _SkillError:
                continue
            # First occurrence wins: project-local overrides user-level skills.
            if cand.id not in skills:
                skills[cand.id] = AvailableAgentSkill(cand.id, skill.name, skill.description, root.source)
    return [skills[k] for k in sorted(skills)]


def home_dir() -> Optional[Path]:
    if os.name == "nt":
        up = os.environ.get("USERPROFILE")
        if up:
            return Path(up)
        drive, hp = os.environ.get("HOMEDRIVE"), os.environ.get("HOMEPATH")
        if drive and hp:
            return Path(drive) / hp
        h = os.environ.get("HOME")
        return Path(h) if h else None
    h = os.environ.get("HOME")
    return Path(h) if h else None


def skill_roots(project_path: str) -> List[SkillRoot]:
    roots = [SkillRoot(Path(project_path) / ".llm-wiki" / "skills", "project")]
    home = home_dir()
    if home is not None:
        roots.append(SkillRoot(home / ".claude" / "skills", "claude"))
        roots.append(SkillRoot(home / ".codex" / "skills", "codex"))
        roots.append(SkillRoot(home / ".agents" / "skills", "agents"))
    return roots


def load_one_skill_from_roots(roots: List[SkillRoot], name: str) -> Optional[AgentSkill]:
    name = normalize_skill_name(name)
    if name is None:
        return None
    for root in roots:
        s = load_one_skill(root.path, name)
        if s is not None:
            return s
    return None


def load_one_skill(root: Path, name: str) -> Optional[AgentSkill]:
    try:
        return load_skill_file(root / f"{name}.md", name)
    except _SkillError:
        pass
    try:
        return load_skill_directory(root / name, name)
    except _SkillError:
        pass
    for cand in discover_skill_candidates(root):
        if cand.id == name:
            try:
                return load_skill_path(cand.path, name)
            except _SkillError:
                return None
    return None


def discover_skill_candidates(root: Path) -> List[SkillCandidate]:
    out: List[SkillCandidate] = []
    _discover(Path(root), 0, out)
    return out


def _lstat(path: Path):
    try:
        return os.lstat(path)
    except OSError:
        return None


def _discover(d: Path, depth: int, out: List[SkillCandidate]) -> None:
    if depth > MAX_SKILL_SCAN_DEPTH:
        return
    meta = _lstat(d)
    if meta is None or stat.S_ISLNK(meta.st_mode) or not stat.S_ISDIR(meta.st_mode):
        return
    try:
        entries = sorted(os.listdir(d))
    except OSError:
        return
    for entry in entries:
        path = d / entry
        m = _lstat(path)
        if m is None or stat.S_ISLNK(m.st_mode):
            continue
        if stat.S_ISREG(m.st_mode):
            if entry.isascii() and entry.lower() == "skill.md":
                sid = normalize_skill_name(d.name)
                if sid is not None:
                    out.append(SkillCandidate(sid, path))
                continue
            if path.suffix.isascii() and path.suffix.lower() == ".md":
                sid = normalize_skill_name(path.stem)
                if sid is not None:
                    out.append(SkillCandidate(sid, path))
            continue
        if stat.S_ISDIR(m.st_mode):
            if _is_hidden_or_unsafe_dir(path):
                continue
            _discover(path, depth + 1, out)


def _is_hidden_or_unsafe_dir(path: Path) -> bool:
    name = path.name
    return name.startswith(".") or name == "node_modules" or normalize_skill_name(name) is None


def load_skill_path(path: Path, fallback_name: str) -> AgentSkill:
    if path.name == "SKILL.md":
        return load_skill_directory(path.parent, fallback_name)
    return load_skill_file(path, fallback_name)


def load_skill_file(path: Path, fallback_name: str) -> AgentSkill:
    try:
        meta = os.lstat(path)
    except OSError as e:
        raise _SkillError(f"Skill not found: {e}")
    if stat.S_ISLNK(meta.st_mode) or not stat.S_ISREG(meta.st_mode) or meta.st_size > MAX_SKILL_FILE_BYTES:
        raise _SkillError("Skill file is not readable or is too large")
    try:
        raw = Path(path).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise _SkillError(f"Failed to read skill: {e}")
    frontmatter, instructions = split_frontmatter(raw)
    name = (yaml_string_field(frontmatter, "name") if frontmatter is not None else None) or fallback_name
    description = (yaml_string_field(frontmatter, "description") if frontmatter is not None else None) or ""
    if not description.strip():
        raise _SkillError("Skill description is required")
    skill = AgentSkill(
        name=name,
        description=description,
        instructions=instructions.strip(),
        base_dir=str(Path(path).parent).replace("\\", "/"),
        location=str(path).replace("\\", "/"),
    )
    if not skill.instructions:
        raise _SkillError("Skill instructions are empty")
    return skill


def load_skill_directory(d: Path, fallback_name: str) -> AgentSkill:
    try:
        meta = os.lstat(d)
    except OSError as e:
        raise _SkillError(f"Skill folder not found: {e}")
    if stat.S_ISLNK(meta.st_mode) or not stat.S_ISDIR(meta.st_mode):
        raise _SkillError("Skill folder is not readable")
    main = find_skill_main_file(d) or (Path(d) / "SKILL.md")
    return load_skill_file(main, fallback_name)


def find_skill_main_file(d: Path) -> Optional[Path]:
    try:
        for entry in os.listdir(d):
            if entry.lower() == "skill.md" and entry.isascii():
                return Path(d) / entry
    except OSError:
        return None
    return None


def normalize_skill_name(value: str) -> Optional[str]:
    t = value.strip()
    if not t or "/" in t or "\\" in t or ".." in t or not is_portable_skill_name(t):
        return None
    return t


def split_frontmatter(raw: str):
    n = raw[1:] if raw.startswith("﻿") else raw
    n = n.replace("\r\n", "\n").replace("\r", "\n")
    if not n.startswith("---\n"):
        return None, n
    rest = n[4:]
    end = rest.find("\n---")
    if end < 0:
        return None, n
    fm = rest[:end]
    after = rest[end + 4:]
    if after.startswith("\n"):
        after = after[1:]
    return fm, after


def is_portable_skill_name(value: str) -> bool:
    if value.endswith((" ", ".")):
        return False
    if any(ch in '<>:"|?*' or ch <= "\x1f" for ch in value):
        return False
    stem = value.split(".")[0].rstrip(" ")
    stem = "".join(c.upper() if c.isascii() else c for c in stem)
    return stem not in _RESERVED


def yaml_string_field(frontmatter: str, key: str) -> Optional[str]:
    prefix = f"{key}:"
    for line in frontmatter.split("\n"):
        t = line.strip()
        if not t.startswith(prefix):
            continue
        value = t[len(prefix):].strip()
        if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif len(value) >= 2 and value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        if value:
            return value
    return None
