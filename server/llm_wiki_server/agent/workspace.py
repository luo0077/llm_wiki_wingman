"""Port of agent/workspace.rs."""
from __future__ import annotations

import os
from pathlib import Path

AGENT_WORKSPACE_DIR = "agent-workspace"


def agent_workspace_path(project_path) -> Path:
    return Path(project_path) / AGENT_WORKSPACE_DIR


def agent_workspace_display(project_path) -> str:
    return str(agent_workspace_path(project_path)).replace("\\", "/")
