"""Port of agent/types.rs. Dataclasses mirror the serde shapes (camelCase JSON).

Every type has `from_dict(d)` (serde Deserialize, with defaults; raises
ValueError on invalid input) and `to_dict()` (serde Serialize, honoring
skip_serializing_if).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class _StrEnum(str, enum.Enum):
    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)

    @classmethod
    def parse(cls, value: Any, default: Optional["_StrEnum"] = None):
        if value is None:
            if default is not None:
                return default
            raise ValueError("missing enum value")
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"unknown variant `{value}` for {cls.__name__}")


class AgentMode(_StrEnum):
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"
    LOCAL_FIRST = "local_first"


class AgentRetrievalMode(_StrEnum):
    STANDARD = "standard"
    SMART = "smart"
    FAITHFUL = "faithful"


class AgentSkillMode(_StrEnum):
    AUTO = "auto"
    EXPLICIT = "explicit"


def _opt_str(d: dict, key: str) -> Optional[str]:
    v = d.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"invalid type for `{key}`: expected a string")
    return v


def _req_str(d: dict, key: str) -> str:
    v = d.get(key)
    if not isinstance(v, str):
        raise ValueError(f"missing field `{key}`")
    return v


def _opt_uint(d: dict, key: str) -> Optional[int]:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"invalid value for `{key}`: expected an unsigned integer")
    return v


def _bool(d: dict, key: str, default: bool) -> bool:
    v = d.get(key)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise ValueError(f"invalid type for `{key}`: expected a boolean")
    return v


def _str_list(d: dict, key: str) -> List[str]:
    v = d.get(key)
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError(f"invalid type for `{key}`: expected an array of strings")
    return list(v)


def _obj(v: Any, what: str) -> dict:
    if not isinstance(v, dict):
        raise ValueError(f"invalid type for {what}: expected an object")
    return v


@dataclass
class AgentToolOptions:
    wiki: bool = True
    web: bool = False
    anytxt: bool = False

    @classmethod
    def from_dict(cls, d: Any) -> "AgentToolOptions":
        if d is None:
            return cls()
        d = _obj(d, "tools")
        return cls(wiki=_bool(d, "wiki", True), web=_bool(d, "web", False), anytxt=_bool(d, "anytxt", False))

    def to_dict(self) -> Dict[str, Any]:
        return {"wiki": self.wiki, "web": self.web, "anytxt": self.anytxt}


@dataclass
class AgentUsage:
    prompt_chars: int = 0
    completion_chars: int = 0
    reference_count: int = 0
    tool_event_count: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "AgentUsage":
        d = _obj(d, "usage")
        return cls(
            _opt_uint(d, "promptChars") or 0,
            _opt_uint(d, "completionChars") or 0,
            _opt_uint(d, "referenceCount") or 0,
            _opt_uint(d, "toolEventCount") or 0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "promptChars": self.prompt_chars,
            "completionChars": self.completion_chars,
            "referenceCount": self.reference_count,
            "toolEventCount": self.tool_event_count,
        }


@dataclass
class AgentImage:
    media_type: str
    data_base64: str

    @classmethod
    def from_dict(cls, d: Any) -> "AgentImage":
        d = _obj(d, "image")
        return cls(_req_str(d, "mediaType"), _req_str(d, "dataBase64"))

    def to_dict(self) -> Dict[str, Any]:
        return {"mediaType": self.media_type, "dataBase64": self.data_base64}


@dataclass
class AgentConversationMessage:
    role: str
    content: str

    @classmethod
    def from_dict(cls, d: Any) -> "AgentConversationMessage":
        d = _obj(d, "history message")
        return cls(_req_str(d, "role"), _req_str(d, "content"))

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class AgentVersionSummary:
    timestamp: int
    author: str
    tool: str

    @classmethod
    def from_dict(cls, d: Any) -> "AgentVersionSummary":
        d = _obj(d, "latestVersion")
        ts = d.get("timestamp")
        if isinstance(ts, bool) or not isinstance(ts, int):
            raise ValueError("missing field `timestamp`")
        return cls(ts, _req_str(d, "author"), _req_str(d, "tool"))

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "author": self.author, "tool": self.tool}


@dataclass
class AgentKnowledgeContext:
    link_count: int = 0
    related_to: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    outgoing_links: List[str] = field(default_factory=list)
    backlinks: List[str] = field(default_factory=list)
    latest_version: Optional[AgentVersionSummary] = None

    @classmethod
    def from_dict(cls, d: Any) -> "AgentKnowledgeContext":
        d = _obj(d, "knowledgeContext")
        lv = d.get("latestVersion")
        link_count = d.get("linkCount")
        if isinstance(link_count, bool) or not isinstance(link_count, int) or link_count < 0:
            raise ValueError("missing field `linkCount`")
        return cls(
            link_count=link_count,
            related_to=_str_list(d, "relatedTo"),
            tags=_str_list(d, "tags"),
            outgoing_links=_str_list(d, "outgoingLinks"),
            backlinks=_str_list(d, "backlinks"),
            latest_version=AgentVersionSummary.from_dict(lv) if lv is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.related_to:
            out["relatedTo"] = list(self.related_to)
        if self.tags:
            out["tags"] = list(self.tags)
        if self.outgoing_links:
            out["outgoingLinks"] = list(self.outgoing_links)
        if self.backlinks:
            out["backlinks"] = list(self.backlinks)
        out["linkCount"] = self.link_count
        if self.latest_version is not None:
            out["latestVersion"] = self.latest_version.to_dict()
        return out


@dataclass
class AgentReference:
    title: str
    path: str
    kind: str
    snippet: Optional[str] = None
    score: Optional[float] = None
    knowledge_context: Optional[AgentKnowledgeContext] = None

    @classmethod
    def from_dict(cls, d: Any) -> "AgentReference":
        d = _obj(d, "reference")
        score = d.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
            raise ValueError("invalid type for `score`")
        kc = d.get("knowledgeContext")
        return cls(
            title=_req_str(d, "title"),
            path=_req_str(d, "path"),
            kind=_req_str(d, "kind"),
            snippet=_opt_str(d, "snippet"),
            score=float(score) if score is not None else None,
            knowledge_context=AgentKnowledgeContext.from_dict(kc) if kc is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"title": self.title, "path": self.path, "kind": self.kind}
        if self.snippet is not None:
            out["snippet"] = self.snippet
        if self.score is not None:
            out["score"] = self.score
        if self.knowledge_context is not None:
            out["knowledgeContext"] = self.knowledge_context.to_dict()
        return out


@dataclass
class AgentToolEvent:
    tool: str
    status: str
    detail: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Any) -> "AgentToolEvent":
        d = _obj(d, "toolEvent")
        return cls(_req_str(d, "tool"), _req_str(d, "status"), _opt_str(d, "detail"))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"tool": self.tool, "status": self.status}
        if self.detail is not None:
            out["detail"] = self.detail
        return out


@dataclass
class AgentUserInputOption:
    label: str
    value: str
    description: Optional[str] = None
    recommended: Optional[bool] = None

    @classmethod
    def from_dict(cls, d: Any) -> "AgentUserInputOption":
        d = _obj(d, "option")
        rec = d.get("recommended")
        if rec is not None and not isinstance(rec, bool):
            raise ValueError("invalid type for `recommended`")
        return cls(_req_str(d, "label"), _req_str(d, "value"), _opt_str(d, "description"), rec)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"label": self.label, "value": self.value}
        if self.description is not None:
            out["description"] = self.description
        if self.recommended is not None:
            out["recommended"] = self.recommended
        return out


@dataclass
class AgentUserInputField:
    id: str
    field_type: str  # serialized as "type"
    label: str
    description: Optional[str] = None
    placeholder: Optional[str] = None
    options: List[AgentUserInputOption] = field(default_factory=list)
    default_value: Optional[Any] = None

    @classmethod
    def from_dict(cls, d: Any) -> "AgentUserInputField":
        d = _obj(d, "field")
        opts = d.get("options") or []
        if not isinstance(opts, list):
            raise ValueError("invalid type for `options`")
        return cls(
            id=_req_str(d, "id"),
            field_type=_req_str(d, "type"),
            label=_req_str(d, "label"),
            description=_opt_str(d, "description"),
            placeholder=_opt_str(d, "placeholder"),
            options=[AgentUserInputOption.from_dict(o) for o in opts],
            default_value=d.get("defaultValue"),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.id, "type": self.field_type, "label": self.label}
        if self.description is not None:
            out["description"] = self.description
        if self.placeholder is not None:
            out["placeholder"] = self.placeholder
        if self.options:
            out["options"] = [o.to_dict() for o in self.options]
        if self.default_value is not None:
            out["defaultValue"] = self.default_value
        return out


@dataclass
class AgentUserInputRequest:
    request_id: str
    title: str
    fields: List[AgentUserInputField]
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Any) -> "AgentUserInputRequest":
        d = _obj(d, "userInputRequest")
        fields = d.get("fields")
        if not isinstance(fields, list):
            raise ValueError("missing field `fields`")
        return cls(
            request_id=_req_str(d, "requestId"),
            title=_req_str(d, "title"),
            fields=[AgentUserInputField.from_dict(f) for f in fields],
            description=_opt_str(d, "description"),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"requestId": self.request_id, "title": self.title}
        if self.description is not None:
            out["description"] = self.description
        out["fields"] = [f.to_dict() for f in self.fields]
        return out


@dataclass
class AgentChatRequest:
    message: str = ""
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    mode: AgentMode = AgentMode.STANDARD
    retrieval_mode: AgentRetrievalMode = AgentRetrievalMode.STANDARD
    tools: AgentToolOptions = field(default_factory=AgentToolOptions)
    top_k: Optional[int] = None
    include_content: Optional[bool] = None
    history: List[AgentConversationMessage] = field(default_factory=list)
    history_explicit: bool = False
    skills: List[str] = field(default_factory=list)
    context_files: List[str] = field(default_factory=list)
    skill_mode: AgentSkillMode = AgentSkillMode.EXPLICIT
    approved_shell_commands: List[str] = field(default_factory=list)
    shell_command: Optional[str] = None
    images: List[AgentImage] = field(default_factory=list)
    stream: Optional[bool] = None
    allow_empty_retrieval: bool = False
    persist_session: bool = True

    @classmethod
    def from_dict(cls, d: Any) -> "AgentChatRequest":
        d = _obj(d, "AgentChatRequest")
        if not isinstance(d.get("message"), str):
            raise ValueError("missing field `message`")
        ic = d.get("includeContent")
        if ic is not None and not isinstance(ic, bool):
            raise ValueError("invalid type for `includeContent`")
        st = d.get("stream")
        if st is not None and not isinstance(st, bool):
            raise ValueError("invalid type for `stream`")
        hist = d.get("history") or []
        imgs = d.get("images") or []
        if not isinstance(hist, list) or not isinstance(imgs, list):
            raise ValueError("invalid type for `history`/`images`")
        return cls(
            message=d["message"],
            session_id=_opt_str(d, "sessionId"),
            run_id=_opt_str(d, "runId"),
            mode=AgentMode.parse(d.get("mode"), AgentMode.STANDARD),
            retrieval_mode=AgentRetrievalMode.parse(d.get("retrievalMode"), AgentRetrievalMode.STANDARD),
            tools=AgentToolOptions.from_dict(d.get("tools")),
            top_k=_opt_uint(d, "topK"),
            include_content=ic,
            history=[AgentConversationMessage.from_dict(m) for m in hist],
            history_explicit=_bool(d, "historyExplicit", False),
            skills=_str_list(d, "skills"),
            context_files=_str_list(d, "contextFiles"),
            skill_mode=AgentSkillMode.parse(d.get("skillMode"), AgentSkillMode.EXPLICIT),
            approved_shell_commands=_str_list(d, "approvedShellCommands"),
            shell_command=_opt_str(d, "shellCommand"),
            images=[AgentImage.from_dict(i) for i in imgs],
            stream=st,
            allow_empty_retrieval=_bool(d, "allowEmptyRetrieval", False),
            persist_session=_bool(d, "persistSession", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "sessionId": self.session_id,
            "runId": self.run_id,
            "mode": self.mode.value,
            "retrievalMode": self.retrieval_mode.value,
            "tools": self.tools.to_dict(),
            "topK": self.top_k,
            "includeContent": self.include_content,
            "history": [m.to_dict() for m in self.history],
            "historyExplicit": self.history_explicit,
            "skills": list(self.skills),
            "contextFiles": list(self.context_files),
            "skillMode": self.skill_mode.value,
            "approvedShellCommands": list(self.approved_shell_commands),
            "shellCommand": self.shell_command,
            "images": [i.to_dict() for i in self.images],
            "stream": self.stream,
            "allowEmptyRetrieval": self.allow_empty_retrieval,
            "persistSession": self.persist_session,
        }


@dataclass
class AgentChatResponse:
    ok: bool
    project_id: str
    session_id: str
    mode: AgentMode
    message: str
    references: List[AgentReference] = field(default_factory=list)
    tool_events: List[AgentToolEvent] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)  # AgentEvent dicts
    user_input_request: Optional[AgentUserInputRequest] = None
    usage: Optional[AgentUsage] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ok": self.ok,
            "projectId": self.project_id,
            "sessionId": self.session_id,
            "mode": AgentMode.parse(self.mode).value,
            "message": self.message,
            "references": [r.to_dict() for r in self.references],
            "toolEvents": [t.to_dict() for t in self.tool_events],
        }
        if self.events:
            out["events"] = [dict(e) for e in self.events]
        if self.user_input_request is not None:
            out["userInputRequest"] = self.user_input_request.to_dict()
        if self.usage is not None:
            out["usage"] = self.usage.to_dict()
        return out

    @classmethod
    def from_dict(cls, d: Any) -> "AgentChatResponse":
        d = _obj(d, "AgentChatResponse")
        uir = d.get("userInputRequest")
        us = d.get("usage")
        return cls(
            ok=bool(d.get("ok")),
            project_id=_req_str(d, "projectId"),
            session_id=_req_str(d, "sessionId"),
            mode=AgentMode.parse(d.get("mode")),
            message=_req_str(d, "message"),
            references=[AgentReference.from_dict(r) for r in d.get("references") or []],
            tool_events=[AgentToolEvent.from_dict(t) for t in d.get("toolEvents") or []],
            events=[dict(e) for e in d.get("events") or []],
            user_input_request=AgentUserInputRequest.from_dict(uir) if uir is not None else None,
            usage=AgentUsage.from_dict(us) if us is not None else None,
        )
