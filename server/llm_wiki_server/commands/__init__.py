"""Importing this package registers every ported command (see registry.command)."""
import importlib

# One module per Rust file under src-tauri/src/commands/ (plus agent / REST
# layers). A module that has not been ported yet is skipped; a module that
# exists but fails to import raises, so real errors are never swallowed.
_MODULES = [
    "project",
    "app_state_rpc",
    "diagnostics",
    "fs",
    "file_history",
    "project_maintenance",
    "extract_images",
    "search",
    "external_search",
    "vectorstore",
    "page_embedding",
    "file_sync",
    "cli",
    "ingest",
    "rest_api",
    "runtime_config",
    "agent",
]

for _name in _MODULES:
    try:
        importlib.import_module(f"{__name__}.{_name}")
    except ModuleNotFoundError as e:
        if e.name != f"{__name__}.{_name}":
            raise
