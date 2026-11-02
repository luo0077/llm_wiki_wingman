"""Port of src/lib/parsed-source-output.ts (generated raw/parsed/*.md copies). All functions are sync."""
from __future__ import annotations

from typing import Optional

from .path_utils import get_file_name, get_relative_path, is_absolute_path, normalize_path
import re

PARSED_MARKDOWN_EXTENSIONS = frozenset(
    [
        "pdf", "doc", "docx", "docm", "ppt", "pps", "pot", "pptx", "pptm", "ppsx", "ppsm",
        "xls", "xlsx", "xlsm", "xlsb", "odt", "ods", "odp", "rtf", "epub", "mobi", "org",
    ]
)


def parsed_markdown_output_path(project_path: str, source_path: str) -> Optional[str]:
    project = re.sub(r"/+\Z", "", normalize_path(project_path))
    source = normalize_path(source_path)
    source_root = f"{project}/raw/sources"
    if source.lower().startswith("raw/sources/"):
        relative = source[len("raw/sources/"):]
    else:
        relative = get_relative_path(source, source_root)
    if (
        not relative
        or (relative == source and not source.lower().startswith("raw/sources/"))
        or is_absolute_path(relative)
    ):
        return None
    if any((not part) or part == "." or part == ".." for part in relative.split("/")):
        return None

    file_name = get_file_name(relative)
    extension = file_name.split(".")[-1].lower() if "." in file_name else ""
    if extension not in PARSED_MARKDOWN_EXTENSIONS:
        return None
    return f"{project}/raw/parsed/{relative}.md"


def persist_parsed_markdown(project_path: str, source_path: str, markdown: str) -> Optional[str]:
    from ..commands import fs

    output_path = parsed_markdown_output_path(project_path, source_path)
    if not output_path:
        return None
    parent = output_path[: output_path.rfind("/")]
    fs.create_directory(parent)
    fs.write_file_atomic(output_path, markdown)
    return output_path


def remove_parsed_markdown(project_path: str, source_path: str) -> None:
    from ..commands import fs

    output_path = parsed_markdown_output_path(project_path, source_path)
    if not output_path:
        return
    try:
        fs.delete_file(output_path)
    except Exception:
        pass


def move_parsed_markdown(project_path: str, old_source_path: str, new_source_path: str) -> None:
    from ..commands import fs

    old_output = parsed_markdown_output_path(project_path, old_source_path)
    new_output = parsed_markdown_output_path(project_path, new_source_path)
    if not old_output or not new_output or old_output == new_output:
        return
    try:
        markdown = fs.read_file(old_output)
    except Exception:
        return
    fs.create_directory(new_output[: new_output.rfind("/")])
    fs.write_file_atomic(new_output, markdown)
    try:
        fs.delete_file(old_output)
    except Exception:
        pass
