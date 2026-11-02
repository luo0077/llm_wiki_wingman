"""Port of agent/events.rs.

`AgentEvent` is a `dict` subclass holding exactly the serde JSON shape
(`{"type": "toolStart", "tool": ..., "input": ...}`), so it can be json-dumped,
published on the event bus, or put in `AgentChatResponse.events` directly.
Option<T> fields without skip_serializing_if are always present (null when None).
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class AgentEvent(dict):
    @property
    def type(self) -> str:
        return self.get("type", "")

    # ---- constructors (enum variants) ----
    @classmethod
    def agent_start(cls, session_id: str) -> "AgentEvent":
        return cls({"type": "agentStart", "sessionId": session_id})

    @classmethod
    def turn_start(cls, mode: str) -> "AgentEvent":
        return cls({"type": "turnStart", "mode": str(getattr(mode, "value", mode))})

    @classmethod
    def tool_start(cls, tool: str, input: Optional[str] = None) -> "AgentEvent":
        return cls({"type": "toolStart", "tool": tool, "input": input})

    @classmethod
    def tool_end(cls, tool: str, output: Optional[str] = None) -> "AgentEvent":
        return cls({"type": "toolEnd", "tool": tool, "output": output})

    @classmethod
    def reference_added(cls, reference: Any) -> "AgentEvent":
        """`reference`: AgentReference or its dict."""
        ref = reference.to_dict() if hasattr(reference, "to_dict") else dict(reference)
        return cls({"type": "referenceAdded", "reference": ref})

    @classmethod
    def file_changed(cls, path: str, tool: str, existed_before: bool,
                     previous_content: Optional[str] = None) -> "AgentEvent":
        ev: Dict[str, Any] = {"type": "fileChanged", "path": path, "tool": tool, "existedBefore": existed_before}
        if previous_content is not None:
            ev["previousContent"] = previous_content
        return cls(ev)

    @classmethod
    def message_delta(cls, text: str) -> "AgentEvent":
        return cls({"type": "messageDelta", "text": text})

    @classmethod
    def error(cls, message: str) -> "AgentEvent":
        return cls({"type": "error", "message": message})

    @classmethod
    def user_input_required(cls, request: Any) -> "AgentEvent":
        """`request`: AgentUserInputRequest or its dict."""
        req = request.to_dict() if hasattr(request, "to_dict") else dict(request)
        return cls({"type": "userInputRequired", "request": req})

    @classmethod
    def done(cls, session_id: str) -> "AgentEvent":
        return cls({"type": "done", "sessionId": session_id})

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

    def redact_for_external_api(self) -> "AgentEvent":
        """In-place (like Rust `&mut self`); also returns self."""
        if self.get("type") == "fileChanged":
            self.pop("previousContent", None)
        return self


def redact_for_external_api(event: Dict[str, Any]) -> Dict[str, Any]:
    """Functional form (matches rest_api's runtime.redact_for_external_api contract): returns a redacted copy."""
    out = AgentEvent(event)
    return out.redact_for_external_api()


# Rust-style aliases
tool_start = AgentEvent.tool_start
tool_end = AgentEvent.tool_end
