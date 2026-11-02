"""Headless replacements for what the desktop frontend did through tauri-plugin-store.

The GUI wrote `recentProjects` / `lastProject` / configs into app-state.json
itself; with no GUI, API clients do it through these commands.
"""
from __future__ import annotations

from ..common import fwd, validate_wiki_project_root
from ..registry import command
from ..state import app_state


@command()
def app_state_get(key: str):
    return app_state().get(key)


@command()
def app_state_set(key: str, value) -> None:
    app_state().set(key, value)
    try:
        from . import rest_api

        rest_api.invalidate_config_cache()
    except Exception:
        pass
    return None


@command()
def register_project(path: str, make_current: bool = True) -> dict:
    """Validate a wiki project and add it to recentProjects (and lastProject), like the GUI's saveLastProject."""
    root = validate_wiki_project_root(path)
    from .project import ensure_metadata_dir

    ensure_metadata_dir(root)
    project = {"name": root.name or "Unknown", "path": fwd(path)}
    st = app_state()
    existing = st.get("recentProjects") or []
    st.set("recentProjects", ([project] + [p for p in existing if p.get("path") != project["path"]])[:10])
    if make_current:
        st.set("lastProject", project)
    return project


@command()
def unregister_project(path: str) -> None:
    """Remove a project from recentProjects (and lastProject if it points at it). Files are not touched."""
    st = app_state()
    norm = fwd(path)
    st.set("recentProjects", [p for p in (st.get("recentProjects") or []) if p.get("path") != norm])
    last = st.get("lastProject")
    if isinstance(last, dict) and last.get("path") == norm:
        st.set("lastProject", None)
    return None
