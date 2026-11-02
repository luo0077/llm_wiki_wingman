"""Port of src/lib/reasoning-capabilities.ts.

``config`` (LlmConfig) and reasoning configs are plain dicts with the TS
camelCase keys (``provider``, ``model``, ``customEndpoint``, ``azureModelFamily``,
``ingestReasoning``; reasoning = ``{"mode": ..., "budgetTokens": ...}``);
attribute-style objects are accepted too.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from ._js import js_trim, rx

AUTO_ONLY = ("auto",)
OPENAI_LEVELS = ("auto", "low", "medium", "high")
BUDGET_LEVELS = ("auto", "off", "low", "medium", "high", "max", "custom")
THINKING_REQUIRED_BUDGETS = ("auto", "low", "medium", "high", "max", "custom")
THINKING_REQUIRED_LEVELS = ("auto", "low", "medium", "high", "max")
OLLAMA_LEVELS = ("auto", "off", "low", "medium", "high", "max")
TOGGLE_LEVELS = ("auto", "off")
DEEPSEEK_LEVELS = ("auto", "off", "high", "max")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class ReasoningCapabilities:
    modes: Tuple[str, ...]
    customBudgetRange: Dict[str, int]

    def normalize(self, requested: Any) -> Dict[str, Any]:
        """A saved mode can outlive a model/provider change; normalize before building wire payloads."""
        mode = _get(requested, "mode")
        if mode not in self.modes:
            return {"mode": "auto"}
        if mode != "custom":
            return {"mode": mode}
        raw = _get(requested, "budgetTokens")
        raw = 0 if raw is None else raw
        try:
            if isinstance(raw, float) and math.isnan(raw):
                return {"mode": "auto"}
            floored = math.floor(raw)
        except (TypeError, ValueError, OverflowError):
            floored = math.inf if raw == math.inf else (-math.inf if raw == -math.inf else 0)
        budget = min(self.customBudgetRange["max"], floored)
        if 0 < budget < self.customBudgetRange["min"]:
            return {"mode": "custom", "budgetTokens": self.customBudgetRange["min"]}
        return {"mode": "custom", "budgetTokens": int(budget)} if budget > 0 else {"mode": "auto"}


def _capabilities(modes: Sequence[str], custom_budget_range: Optional[Dict[str, int]] = None) -> ReasoningCapabilities:
    return ReasoningCapabilities(tuple(modes), custom_budget_range or {"min": 1, "max": 32_768})


_CLAUDE46_RE = rx(r"claude-(?:opus|sonnet|haiku)-4[-_.]?(?:[6-9]|\d{2,})(?:[-_.]|$)".replace("$", r"\Z"), re.I)
_GEMINI25PRO_RE = rx(r"gemini[-_.]?2\.5[-_.]?pro(?:[-_.]|$)".replace("$", r"\Z"), re.I)
_GEMINI3_RE = rx(r"gemini[-_.]?3(?:[-_.]|$)".replace("$", r"\Z"), re.I)
_OPENAI_REASONING_RE = rx(r"\A(?:gpt-5|o\d+)(?:[.\-_]|\Z)")
_DEEPSEEK_RE = rx(r"api\.deepseek\.(?:com|cn)(?:[:/]|\Z)")
_XIAOMI_RE = rx(r"xiaomimimo\.com(?:[:/]|\Z)")


def _is_claude46_or_later(model: str) -> bool:
    return bool(_CLAUDE46_RE.search(model))


def _is_gemini25_pro(model: str) -> bool:
    return bool(_GEMINI25PRO_RE.search(model))


def _is_gemini3(model: str) -> bool:
    return bool(_GEMINI3_RE.search(model))


def _is_openai_reasoning_model(config: Any) -> bool:
    if _get(config, "provider") == "azure" and _get(config, "azureModelFamily") == "gpt5":
        return True
    model = js_trim(_get(config, "model", "") or "").lower()
    return bool(_OPENAI_REASONING_RE.search(model))


def is_openrouter_endpoint(endpoint: str) -> bool:
    try:
        parts = urlsplit(endpoint)
        if not parts.scheme or parts.hostname is None:
            return False
        hostname = parts.hostname.lower()
        return hostname == "openrouter.ai" or hostname.endswith(".openrouter.ai")
    except ValueError:
        return False


def resolve_reasoning_capabilities(config: Any) -> ReasoningCapabilities:
    provider = _get(config, "provider")
    model = _get(config, "model", "") or ""
    if provider in ("claude-code", "codex-cli"):
        return _capabilities(AUTO_ONLY)
    if provider == "ollama":
        return _capabilities(OLLAMA_LEVELS)
    if provider == "google":
        if _is_gemini3(model):
            return _capabilities(THINKING_REQUIRED_LEVELS)
        if _is_gemini25_pro(model):
            return _capabilities(THINKING_REQUIRED_BUDGETS, {"min": 128, "max": 32_768})
        return _capabilities(BUDGET_LEVELS)
    if provider == "anthropic":
        return _capabilities(
            THINKING_REQUIRED_LEVELS if _is_claude46_or_later(model) else BUDGET_LEVELS,
            {"min": 1024, "max": 32_768},
        )
    if provider == "minimax":
        return _capabilities(AUTO_ONLY)
    if provider in ("openai", "azure"):
        return _capabilities(OPENAI_LEVELS if _is_openai_reasoning_model(config) else AUTO_ONLY)
    if provider == "custom":
        endpoint = (_get(config, "customEndpoint", "") or "").lower()
        if is_openrouter_endpoint(endpoint):
            return _capabilities(BUDGET_LEVELS)
        if _DEEPSEEK_RE.search(endpoint):
            return _capabilities(DEEPSEEK_LEVELS)
        if _XIAOMI_RE.search(endpoint):
            return _capabilities(TOGGLE_LEVELS)
        return _capabilities(AUTO_ONLY)
    return _capabilities(AUTO_ONLY)


def normalize_reasoning_for_provider(config: Any, requested: Any) -> Dict[str, Any]:
    return resolve_reasoning_capabilities(config).normalize(requested)


def resolve_ingest_reasoning(config: Any) -> Any:
    """config.ingestReasoning, defaulting to {"mode": "off"}."""
    value = _get(config, "ingestReasoning")
    return value if value is not None else {"mode": "off"}


def is_adaptive_anthropic_model(config: Any) -> bool:
    return _get(config, "provider") == "anthropic" and _is_claude46_or_later(_get(config, "model", "") or "")


def is_gemini_thinking_level_model(config: Any) -> bool:
    return _get(config, "provider") == "google" and _is_gemini3(_get(config, "model", "") or "")
