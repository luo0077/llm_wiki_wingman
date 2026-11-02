"""Port of commands/project.rs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..common import fwd, validate_wiki_project_root
from ..errors import CommandError
from ..registry import command

_DIRS = [
    "raw/sources", "raw/assets", "wiki/entities", "wiki/concepts",
    "wiki/sources", "wiki/queries", "wiki/comparisons", "wiki/synthesis",
]

_SCHEMA = """# Wiki Schema

## Page Types

| Type | Directory | Purpose |
|------|-----------|---------|
| entity | wiki/entities/ | Named things (models, companies, people, datasets) |
| concept | wiki/concepts/ | Ideas, techniques, phenomena |
| source | wiki/sources/ | Papers, articles, talks, blog posts |
| query | wiki/queries/ | Open questions under investigation |
| comparison | wiki/comparisons/ | Side-by-side analysis of related entities |
| synthesis | wiki/synthesis/ | Cross-cutting summaries and conclusions |

## Naming Conventions

- Files: `kebab-case.md`
- Entities: match official name where possible (e.g., `gpt-4.md`, `openai.md`)
- Concepts: descriptive noun phrases (e.g., `chain-of-thought.md`)
- Sources: `author-year-slug.md` (e.g., `wei-2022-chain-of-thought.md`)
- Queries: question as slug (e.g., `does-scale-improve-reasoning.md`)

## Frontmatter

All pages must include YAML frontmatter:

```yaml
---
type: entity | concept | source | query | comparison | synthesis | overview
title: Human-readable title
tags: []
related: []
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Source pages also include:
```yaml
authors: []
year: YYYY
url: ""
venue: ""
```

## Index Format

`wiki/index.md` lists all pages grouped by type. Each entry:
```
- [[page-slug]] — one-line description
```

## Log Format

`wiki/log.md` records research activity in reverse chronological order:
```
## YYYY-MM-DD

- Action taken / finding noted
```

## Cross-referencing Rules

- Use `[[page-slug]]` syntax to link between wiki pages
- Every entity and concept should appear in `wiki/index.md`
- Queries link to the sources and concepts they draw on
- Synthesis pages cite all contributing sources via `related:`

## Contradiction Handling

When sources contradict each other:
1. Note the contradiction in the relevant concept or entity page
2. Create or update a query page to track the open question
3. Link both sources from the query page
4. Resolve in a synthesis page once sufficient evidence exists
"""

_PURPOSE = """# Project Purpose

## Goal

<!-- What are you trying to understand or build? -->

## Key Questions

<!-- List the primary questions driving this research -->

1.
2.
3.

## Scope

<!-- What is in scope? What is explicitly out of scope? -->

**In scope:**
-

**Out of scope:**
-

## Thesis

<!-- Your current working hypothesis or conclusion (update as research progresses) -->

> TBD
"""

_INDEX = """# Wiki Index

## Entities

## Concepts

## Sources

## Queries

## Comparisons

## Synthesis
"""

_OVERVIEW = """---
type: overview
title: Project Overview
tags: []
related: []
---

# Overview

<!-- Provide a high-level summary of what this wiki covers and its current state. Update regularly as understanding deepens. -->
"""

_OBSIDIAN = {
    "app.json": """{
  "attachmentFolderPath": "raw/assets",
  "userIgnoreFilters": [
    ".cache",
    ".llm-wiki",
    ".superpowers"
  ],
  "useMarkdownLinks": false,
  "newLinkFormat": "shortest",
  "showUnsupportedFiles": false
}""",
    "appearance.json": """{
  "baseFontSize": 16,
  "theme": "obsidian"
}""",
    "core-plugins.json": """{
  "file-explorer": true,
  "global-search": true,
  "graph": true,
  "backlink": true,
  "tag-pane": true,
  "page-preview": true,
  "outgoing-link": true,
  "starred": true
}""",
}


def _write(path: Path, contents: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    except OSError as e:
        raise CommandError(f"Failed to write file '{path}': {e}")


def ensure_metadata_dir(root: Path) -> None:
    """Create `<project>/.llm-wiki`. On the desktop the frontend does this when a project is
    opened (persist.ts); headless there is no frontend, and history/queue/review features fail
    with ENOENT without it."""
    try:
        (Path(root) / ".llm-wiki").mkdir(exist_ok=True)
    except OSError:
        pass


@command()
def create_project(name: str, path: str) -> dict:
    root = Path(path) / name
    if root.exists():
        raise CommandError(f"Directory already exists: '{root}'")
    for d in _DIRS:
        try:
            (root / d).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise CommandError(f"Failed to create directory '{d}': {e}")
    today = datetime.now().strftime("%Y-%m-%d")
    _write(root / "schema.md", _SCHEMA)
    _write(root / "purpose.md", _PURPOSE)
    _write(root / "wiki/index.md", _INDEX)
    _write(root / "wiki/log.md", f"# Research Log\n\n## {today}\n\n- Project created\n")
    _write(root / "wiki/overview.md", _OVERVIEW)
    for fname, body in _OBSIDIAN.items():
        _write(root / ".obsidian" / fname, body)
    ensure_metadata_dir(root)
    return {"name": name, "path": fwd(root)}


@command()
def open_project(path: str) -> dict:
    root = validate_wiki_project_root(path)
    ensure_metadata_dir(root)
    return {"name": root.name or "Unknown", "path": path.replace("\\", "/")}


@command()
def open_project_folder(path: str) -> None:
    """Headless: there is no file manager to open. Validate and succeed (the Rust version only validated + launched the OS opener)."""
    validate_wiki_project_root(path)
    return None


@command()
def open_path_in_project(project_path: str, target_path: str) -> None:
    root = validate_wiki_project_root(project_path).resolve()
    target = Path(target_path)
    target = target if target.is_absolute() else root / target
    try:
        target = target.resolve(strict=True)
    except OSError as e:
        raise CommandError(f"Failed to resolve target path '{target}': {e}")
    try:
        target.relative_to(root)
    except ValueError:
        raise CommandError(f"Refusing to open a path outside the project: '{target}'")
    return None
