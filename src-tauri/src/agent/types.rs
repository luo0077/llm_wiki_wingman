use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    Fast,
    Standard,
    Deep,
    LocalFirst,
}

impl Default for AgentMode {
    fn default() -> Self {
        Self::Standard
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolOptions {
    #[serde(default = "default_true")]
    pub wiki: bool,
    #[serde(default)]
    pub web: bool,
    #[serde(default)]
    pub anytxt: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentUsage {
    pub prompt_chars: usize,
    pub completion_chars: usize,
    pub reference_count: usize,
    pub tool_event_count: usize,
}

const fn default_true() -> bool {
    true
}

impl Default for AgentToolOptions {
    fn default() -> Self {
        Self {
            wiki: true,
            web: false,
            anytxt: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentChatRequest {
    pub message: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub mode: AgentMode,
    #[serde(default)]
    pub tools: AgentToolOptions,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub include_content: Option<bool>,
    #[serde(default)]
    pub history: Vec<AgentConversationMessage>,
    #[serde(default)]
    pub skills: Vec<String>,
    #[serde(default)]
    pub images: Vec<AgentImage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default = "default_true")]
    pub persist_session: bool,
}

impl Default for AgentChatRequest {
    fn default() -> Self {
        Self {
            message: String::new(),
            session_id: None,
            run_id: None,
            mode: AgentMode::default(),
            tools: AgentToolOptions::default(),
            top_k: None,
            include_content: None,
            history: Vec::new(),
            skills: Vec::new(),
            images: Vec::new(),
            stream: None,
            persist_session: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentImage {
    pub media_type: String,
    pub data_base64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AgentReference {
    pub title: String,
    pub path: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolEvent {
    pub tool: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentChatResponse {
    pub ok: bool,
    pub project_id: String,
    pub session_id: String,
    pub mode: AgentMode,
    pub message: String,
    pub references: Vec<AgentReference>,
    pub tool_events: Vec<AgentToolEvent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<super::events::AgentEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<AgentUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentConversationMessage {
    pub role: String,
    pub content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_accepts_camelcase_api_shape_with_defaults() {
        let req: AgentChatRequest = serde_json::from_value(serde_json::json!({
            "message": "hello",
            "sessionId": "s1",
            "topK": 7
        }))
        .unwrap();

        assert_eq!(req.message, "hello");
        assert_eq!(req.session_id.as_deref(), Some("s1"));
        assert!(req.run_id.is_none());
        assert_eq!(req.mode, AgentMode::Standard);
        assert_eq!(req.top_k, Some(7));
        assert!(req.tools.wiki);
        assert!(!req.tools.web);
        assert!(!req.tools.anytxt);
        assert!(req.persist_session);
    }

    #[test]
    fn chat_request_accepts_tool_overrides() {
        let req: AgentChatRequest = serde_json::from_value(serde_json::json!({
            "message": "hello",
            "mode": "local_first",
            "tools": {
                "wiki": false,
                "web": true,
                "anytxt": true
            }
        }))
        .unwrap();

        assert_eq!(req.mode, AgentMode::LocalFirst);
        assert!(!req.tools.wiki);
        assert!(req.tools.web);
        assert!(req.tools.anytxt);
        assert!(req.images.is_empty());
    }
}
