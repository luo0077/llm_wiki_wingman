"""Port of commands/project_maintenance.rs.

Headless note: in the desktop app `destination` / `archive_path` came from a
native save/open dialog; here the caller passes explicit absolute paths on the
server's filesystem (the Rust commands already took them as arguments).
"""
from __future__ import annotations

import os
import stat
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..errors import CommandError
from ..registry import command

MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 100_000


def _io(e: BaseException) -> str:
    if isinstance(e, FileNotFoundError):
        return "No such file or directory (os error 2)"
    if isinstance(e, PermissionError):
        return "Permission denied (os error 13)"
    if isinstance(e, NotADirectoryError):
        return "Not a directory (os error 20)"
    return str(e)


def _canonicalize(p: str) -> Path:
    if not os.path.exists(p):
        raise CommandError("No such file or directory (os error 2)")
    return Path(os.path.realpath(p))


def _normal_parts(name: str) -> Optional[List[str]]:
    """Rust `safe_relative` + component normalisation: None if unsafe, else the Normal components."""
    if name.startswith("/") or (os.name == "nt" and ("\\" in name or (len(name) > 1 and name[1] == ":"))):
        return None
    segs = name.split("/")
    parts: List[str] = []
    for i, seg in enumerate(segs):
        if seg == "":
            continue  # repeated / trailing separators collapse
        if seg == ".":
            if i == 0:
                return None  # leading CurDir component is not Normal
            continue
        if seg == "..":
            return None
        if "\x00" in seg:
            return None
        parts.append(seg)
    return parts


def _within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- export

def _resolve_export_destination(root: Path, output: Path) -> Path:
    if output.exists():
        resolved = Path(os.path.realpath(output))
    else:
        parent = output.parent
        if str(parent) == str(output) or not output.name:
            raise CommandError("Export destination must have a parent directory")
        if not parent.exists():
            raise CommandError("No such file or directory (os error 2)")
        resolved = Path(os.path.realpath(parent)) / output.name
    if _within(resolved, root):
        raise CommandError("Export destination must be outside the project directory")
    return resolved


@command()
def export_project_archive(project_path: str, destination: str):
    if not os.path.isabs(project_path) or not os.path.isabs(destination):
        raise CommandError("Project and archive paths must be absolute")
    root = _canonicalize(project_path)
    output = _resolve_export_destination(root, Path(destination))
    try:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames.sort()
                base = Path(dirpath)
                entries = [(d, True) for d in dirnames] + [(f, False) for f in sorted(filenames)]
                for name, is_dir_hint in sorted(entries, key=lambda t: t[0]):
                    p = base / name
                    if p.is_symlink():
                        continue
                    rel = str(p.relative_to(root)).replace("\\", "/")
                    if p.is_dir():
                        zf.write(p, arcname=rel + "/")
                    elif p.is_file():
                        zf.write(p, arcname=rel, compress_type=zipfile.ZIP_DEFLATED)
    except CommandError:
        raise
    except OSError as e:
        raise CommandError(_io(e))
    return None


# ---------------------------------------------------------------- import

@command()
def import_project_archive(archive_path: str, destination: str):
    if not os.path.isabs(archive_path) or not os.path.isabs(destination):
        raise CommandError("Archive and destination paths must be absolute")
    try:
        f = open(archive_path, "rb")
    except OSError as e:
        raise CommandError(_io(e))
    with f:
        try:
            zf = zipfile.ZipFile(f)
        except Exception as e:
            raise CommandError(str(e))
        with zf:
            infos = zf.infolist()
            if len(infos) > MAX_ARCHIVE_ENTRIES:
                raise CommandError("Project archive contains too many entries")
            expanded = 0
            has_index = False
            plan: List[Tuple[zipfile.ZipInfo, List[str], bool]] = []
            for info in infos:
                mode = info.external_attr >> 16
                if mode and (mode & 0o170000) == 0o120000:
                    raise CommandError(f"Archive contains an unsupported symbolic link: {info.filename}")
                parts = _normal_parts(info.filename)
                if parts is None:
                    raise CommandError(f"Unsafe archive path: {info.filename}")
                is_dir = info.is_dir()
                if parts == ["wiki", "index.md"] and not is_dir:
                    has_index = True
                expanded += info.file_size
                if expanded > MAX_ARCHIVE_BYTES:
                    raise CommandError("Project archive exceeds 4 GB expanded limit")
                plan.append((info, parts, is_dir))
            if not has_index:
                raise CommandError("Archive is not an LLM Wiki project (wiki/index.md is missing)")
            root = Path(destination)
            try:
                if root.exists() and any(True for _ in os.scandir(root)):
                    raise CommandError("Import destination must be empty")
                root.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise CommandError(_io(e))
            real_root = Path(os.path.realpath(root))
            try:
                for info, parts, is_dir in plan:
                    target = root.joinpath(*parts)
                    # Defence in depth beyond the Rust checks: never write outside the root.
                    if not _within(Path(os.path.realpath(target.parent if not is_dir else target)), real_root):
                        raise CommandError(f"Unsafe archive path: {info.filename}")
                    if is_dir:
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as out:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
            except CommandError:
                raise
            except (OSError, zipfile.BadZipFile, RuntimeError) as e:
                raise CommandError(_io(e))
    return str(root)


# ---------------------------------------------------------------- rebuild index

def frontmatter_value(content: str, key: str) -> Optional[str]:
    normalized = content.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return None
    rest = normalized[4:]
    i = rest.find("\n---")
    if i < 0:
        return None
    body = rest[:i]
    for line in body.split("\n"):
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        if name.strip() == key:
            v = value.strip().strip("\"'")
            return v or None
    return None


@command()
def rebuild_wiki_index(project_path: str):
    wiki = Path(project_path) / "wiki"
    groups: Dict[str, List[Tuple[str, str]]] = {}
    if not wiki.exists():
        raise CommandError(
            f"Failed to enumerate wiki pages: IO error for operation on {wiki}: No such file or directory (os error 2)"
        )
    for dirpath, dirnames, filenames in os.walk(wiki, followlinks=False):
        dirnames.sort()
        for fname in sorted(filenames):
            p = Path(dirpath) / fname
            st = os.lstat(p)
            if not stat.S_ISREG(st.st_mode) or not fname.endswith(".md") or fname == ".md":
                continue
            stem = fname[:-3]
            if stem.lower() in ("index", "overview", "log"):
                continue
            try:
                content = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise CommandError("stream did not contain valid UTF-8")
            except OSError as e:
                raise CommandError(_io(e))
            kind = frontmatter_value(content, "type") or "other"
            title = frontmatter_value(content, "title") or stem
            target = str(p.relative_to(wiki).with_suffix("")).replace("\\", "/")
            groups.setdefault(kind, []).append((target, title))
    for pages in groups.values():
        pages.sort(key=lambda t: t[1].lower())
    count = sum(len(v) for v in groups.values())
    out = "# Wiki Index\n\n"
    for kind in sorted(groups, key=lambda k: k.encode("utf-8")):
        out += f"## {kind}\n\n"
        for slug, title in groups[kind]:
            out += f"- [[{slug}|{title}]]\n"
        out += "\n"
    index_path = wiki / "index.md"
    tmp = wiki / ".index.md.rebuild.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(out)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, index_path)
    except OSError as e:
        raise CommandError(_io(e))
    return {"pages": count, "groups": len(groups)}
