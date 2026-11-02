"""Port of agent/router.rs."""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict

from .types import AgentMode, AgentToolOptions


class QueryIntent(str, enum.Enum):
    NEEDS_INTERNAL_SEARCH = "needs_internal_search"
    NEEDS_EXTERNAL_SEARCH = "needs_external_search"
    NEEDS_RAW_SOURCE_SEARCH = "needs_raw_source_search"
    NEEDS_GRAPH = "needs_graph"
    NEEDS_WRITE = "needs_write"
    SIMPLE_CONVERSATIONAL = "simple_conversational"
    AMBIGUOUS = "ambiguous"


@dataclass
class RouterDecision:
    intent: QueryIntent
    should_search_wiki: bool
    should_hint_web: bool
    should_hint_anytxt: bool
    should_include_sources: bool
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "shouldSearchWiki": self.should_search_wiki,
            "shouldHintWeb": self.should_hint_web,
            "shouldHintAnytxt": self.should_hint_anytxt,
            "shouldIncludeSources": self.should_include_sources,
            "rationale": self.rationale,
        }


def _contains_any(value: str, needles) -> bool:
    return any(n in value for n in needles)


_RATIONALE = {
    QueryIntent.NEEDS_EXTERNAL_SEARCH: "User appears to request current/external information.",
    QueryIntent.SIMPLE_CONVERSATIONAL: "Short conversational turn; avoid unnecessary retrieval.",
    QueryIntent.NEEDS_RAW_SOURCE_SEARCH: "User explicitly referenced raw/source material.",
    QueryIntent.NEEDS_GRAPH: "User asks about graph/relationships.",
    QueryIntent.NEEDS_WRITE: "User asks to create or update wiki content.",
    QueryIntent.NEEDS_INTERNAL_SEARCH: "User question likely benefits from project retrieval.",
    QueryIntent.AMBIGUOUS: "Ambiguous request; let the tool planner decide whether retrieval is useful.",
}


def route_query(message: str, mode: AgentMode, tools: AgentToolOptions) -> RouterDecision:
    lower = message.lower()
    trimmed = message.strip()
    explicit_web = _contains_any(lower, ["web search", "search the web", "internet", "online", "latest",
                                         "today", "新闻", "联网", "网上", "最新"])
    explicit_raw = _contains_any(lower, ["raw source", "source file", "原始资料", "原始文件", "源文件"])
    explicit_graph = _contains_any(lower, ["graph", "relationship", "知识图谱", "关系图"])
    explicit_write = _contains_any(lower, ["write to wiki", "create page", "写入", "创建页面"])
    # Rust compares byte length (`trimmed.len() < 32`).
    conversational = len(trimmed.encode("utf-8")) < 32 and _contains_any(
        lower, ["hi", "hello", "thanks", "谢谢", "你好", "好的", "ok"])

    if explicit_write:
        intent = QueryIntent.NEEDS_WRITE
    elif explicit_graph:
        intent = QueryIntent.NEEDS_GRAPH
    elif explicit_raw:
        intent = QueryIntent.NEEDS_RAW_SOURCE_SEARCH
    elif explicit_web:
        intent = QueryIntent.NEEDS_EXTERNAL_SEARCH
    elif conversational:
        intent = QueryIntent.SIMPLE_CONVERSATIONAL
    else:
        intent = QueryIntent.AMBIGUOUS

    return RouterDecision(
        intent=intent,
        should_search_wiki=False,
        should_hint_web=tools.web,
        should_hint_anytxt=tools.anytxt,
        should_include_sources=explicit_raw or AgentMode.parse(mode) == AgentMode.DEEP,
        rationale=_RATIONALE[intent],
    )
