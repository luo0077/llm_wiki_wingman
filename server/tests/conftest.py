import os
import tempfile

import pytest

# Tests must be hermetic: whatever LLM_WIKI_* variables the developer exported to run the real
# server (API token, data dir, bind host, ...) must neither leak in nor be written to.
# This runs before any llm_wiki_server import.
for _k in [k for k in os.environ if k.startswith("LLM_WIKI_")]:
    del os.environ[_k]
os.environ["LLM_WIKI_DISABLE_CLIP_SERVER"] = "1"
os.environ["LLM_WIKI_DISABLE_INGEST_RESTORE"] = "1"
os.environ["LLM_WIKI_DATA_DIR"] = tempfile.mkdtemp(prefix="llm-wiki-test-")


@pytest.fixture(autouse=True)
def _no_token_unless_test_sets_one(monkeypatch):
    monkeypatch.delenv("LLM_WIKI_API_TOKEN", raising=False)
