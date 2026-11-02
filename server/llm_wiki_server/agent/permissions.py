"""Port of agent/permissions.rs."""
from __future__ import annotations

import enum
from typing import Iterable, List


class AgentCapability(str, enum.Enum):
    READ_PROJECT = "read_project"
    READ_SOURCE = "read_source"
    SEARCH_WIKI = "search_wiki"
    SEARCH_WEB = "search_web"
    SEARCH_ANY_TXT = "search_any_txt"
    WRITE_WIKI = "write_wiki"
    RUN_DEEP_RESEARCH = "run_deep_research"
    NETWORK = "network"
    PROCESS = "process"


_RUST_NAMES = {
    "read_project": "ReadProject", "read_source": "ReadSource", "search_wiki": "SearchWiki",
    "search_web": "SearchWeb", "search_any_txt": "SearchAnyTxt", "write_wiki": "WriteWiki",
    "run_deep_research": "RunDeepResearch", "network": "Network", "process": "Process",
}


class PermissionPolicy:
    def __init__(self, allowed: Iterable[AgentCapability]):
        self.allowed: List[AgentCapability] = list(allowed)

    @classmethod
    def api_default(cls) -> "PermissionPolicy":
        C = AgentCapability
        return cls([C.READ_PROJECT, C.READ_SOURCE, C.SEARCH_WIKI, C.SEARCH_WEB, C.SEARCH_ANY_TXT,
                    C.WRITE_WIKI, C.NETWORK, C.PROCESS])

    def allows(self, capability: AgentCapability) -> bool:
        return capability in self.allowed

    def require(self, capability: AgentCapability) -> None:
        """Raises Exception(f"Agent capability '{Name}' is not allowed")."""
        if not self.allows(capability):
            raise PermissionError(f"Agent capability '{_RUST_NAMES[AgentCapability(capability).value]}' is not allowed")
