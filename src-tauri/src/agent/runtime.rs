use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

use serde::Deserialize;
use uuid::Uuid;

use crate::commands::search::SearchEmbeddingConfig;

use super::cancel::AgentCancellationToken;
use super::context::{
    build_agent_context, collapse_whitespace, intent_label, load_project_context, trim_chars,
    AgentContextInput, BuiltAgentContext,
};
use super::events::AgentEvent;
use super::permissions::{AgentCapability, PermissionPolicy};
use super::provider::{AgentLlmProvider, LlmClient, LlmConfig};
use super::router::route_query;
use super::skills::load_project_skills;
use super::tools::{self, AnyTxtConfig, ToolRegistry, WebSearchConfig};
use super::types::{
    AgentChatRequest, AgentChatResponse, AgentMode, AgentReference, AgentToolEvent, AgentUsage,
};

// These limits are intentionally enforced in the backend Agent rather than the
// React UI. API and MCP callers bypass the UI, so safety and cost boundaries
// must live here.
const DEFAULT_CHAT_SEARCH_RESULTS: usize = 5;
const MAX_CHAT_SEARCH_RESULTS: usize = 10;
const MAX_IMAGES_PER_TURN: usize = 5;
const MAX_IMAGE_BASE64_BYTES: usize = 7 * 1024 * 1024;

pub type AgentEventSink = Arc<dyn Fn(AgentEvent) + Send + Sync>;

#[derive(Debug, Clone)]
pub struct AgentRuntime {
    project_id: String,
    project_path: String,
    embedding_config: Option<SearchEmbeddingConfig>,
    llm_config: Option<LlmConfig>,
    web_search_config: Option<WebSearchConfig>,
    anytxt_config: Option<AnyTxtConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelToolPlan {
    #[serde(default)]
    tool_calls: Vec<ModelToolCall>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelToolCall {
    tool: String,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    content: Option<String>,
    // Overwriting existing wiki pages is destructive. The planner may set this
    // only when the user explicitly asks to update/overwrite an existing page;
    // the tool defaults to create-only when the field is absent.
    #[serde(default)]
    allow_overwrite: Option<bool>,
}

impl AgentRuntime {
    pub fn new(
        project_id: impl Into<String>,
        project_path: impl Into<String>,
        embedding_config: Option<SearchEmbeddingConfig>,
        llm_config: Option<LlmConfig>,
        web_search_config: Option<WebSearchConfig>,
        anytxt_config: Option<AnyTxtConfig>,
    ) -> Self {
        Self {
            project_id: project_id.into(),
            project_path: project_path.into(),
            embedding_config,
            llm_config,
            web_search_config,
            anytxt_config,
        }
    }

    #[allow(dead_code)]
    pub async fn run_once(&self, request: AgentChatRequest) -> Result<AgentChatResponse, String> {
        self.run_once_with_cancel(request, None).await
    }

    pub async fn run_once_with_cancel(
        &self,
        request: AgentChatRequest,
        cancellation: Option<AgentCancellationToken>,
    ) -> Result<AgentChatResponse, String> {
        self.run_once_with_cancel_and_events(request, cancellation, None)
            .await
    }

    pub async fn run_once_with_cancel_and_events(
        &self,
        request: AgentChatRequest,
        cancellation: Option<AgentCancellationToken>,
        event_sink: Option<AgentEventSink>,
    ) -> Result<AgentChatResponse, String> {
        let message = request.message.trim();
        if message.is_empty() {
            return Err("message is required".to_string());
        }
        validate_images(&request.images)?;
        check_cancel(cancellation.as_ref())?;

        let session_id = request
            .session_id
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_else(|| format!("api_{}", Uuid::new_v4()));
        let mut tool_events = Vec::new();
        let mut events = Vec::new();
        emit_event(
            &mut events,
            &event_sink,
            AgentEvent::AgentStart {
                session_id: session_id.clone(),
            },
        );
        emit_event(
            &mut events,
            &event_sink,
            AgentEvent::TurnStart {
                mode: mode_label(request.mode).to_string(),
            },
        );
        let mut references = Vec::new();
        let permission_policy = PermissionPolicy::api_default();
        let router = route_query(message, request.mode, &request.tools);
        let skills = load_project_skills(&self.project_path, &request.skills);
        check_cancel(cancellation.as_ref())?;
        let model_plan = self
            .plan_tools_with_model(message, request.mode, &request.tools, cancellation.as_ref())
            .await
            .unwrap_or_default();
        if !model_plan.tool_calls.is_empty() {
            tool_emit_event(
                &mut tool_events,
                &mut events,
                &event_sink,
                AgentToolEvent {
                    tool: "agent.plan_tools".to_string(),
                    status: "completed".to_string(),
                    detail: Some(format!(
                        "{} planned tool call(s)",
                        model_plan.tool_calls.len()
                    )),
                },
            );
        }
        let planned_queries = planned_tool_queries(&model_plan, message);
        let planned_has = |tool: &str| planned_queries.contains_key(tool);
        if !request.skills.is_empty() {
            permission_policy.require(AgentCapability::ReadProject)?;
            tool_emit_event(
                &mut tool_events,
                &mut events,
                &event_sink,
                AgentToolEvent {
                    tool: "skills.load".to_string(),
                    status: "completed".to_string(),
                    detail: Some(format!("{} skill(s) active", skills.len())),
                },
            );
            emit_event(
                &mut events,
                &event_sink,
                AgentEvent::tool_end(
                    "skills.load",
                    Some(format!("{} skill(s) active", skills.len())),
                ),
            );
        }

        if request.tools.web {
            permission_policy.require(AgentCapability::SearchWeb)?;
            tool_emit_event(&mut tool_events, &mut events, &event_sink, AgentToolEvent {
                tool: "web.search".to_string(),
                status: "available".to_string(),
                detail: Some("Web search is enabled for this turn. Router decides whether to execute it immediately.".to_string()),
            });
        }
        if request.tools.anytxt {
            permission_policy.require(AgentCapability::SearchAnyTxt)?;
            tool_emit_event(&mut tool_events, &mut events, &event_sink, AgentToolEvent {
                tool: "anytxt.search".to_string(),
                status: "available".to_string(),
                detail: Some("AnyTXT search is enabled for this turn. Router decides whether to execute it immediately.".to_string()),
            });
        }

        let mut retrieval_parts = Vec::new();
        let tool_registry = tools::BuiltinToolRegistry::default();
        let should_search_wiki = router.should_search_wiki || planned_has("wiki.search");
        let should_include_sources = router.should_include_sources || planned_has("source.search");
        let should_search_graph = matches!(router.intent, super::router::QueryIntent::NeedsGraph)
            || planned_has("graph.search");
        let should_run_web = request.tools.web
            && (matches!(
                router.intent,
                super::router::QueryIntent::NeedsExternalSearch
            ) || planned_has("web.search")
                || matches!(request.mode, AgentMode::Deep));
        let should_run_anytxt = request.tools.anytxt
            && (should_include_sources
                || planned_has("anytxt.search")
                || matches!(request.mode, AgentMode::Deep));
        let deep_research = matches!(request.mode, AgentMode::Deep)
            && (should_run_web || should_run_anytxt || should_include_sources);

        if let Some(write_call) = model_plan
            .tool_calls
            .iter()
            .find(|call| call.tool.trim() == "wiki.write_page")
        {
            check_cancel(cancellation.as_ref())?;
            permission_policy.require(AgentCapability::WriteWiki)?;
            let path = write_call
                .path
                .as_deref()
                .map(str::trim)
                .filter(|path| !path.is_empty())
                .ok_or_else(|| "wiki.write_page requires path".to_string())?;
            let content = write_call
                .content
                .as_deref()
                .map(str::trim)
                .filter(|content| !content.is_empty())
                .ok_or_else(|| "wiki.write_page requires content".to_string())?;
            tool_emit_event(
                &mut tool_events,
                &mut events,
                &event_sink,
                AgentToolEvent {
                    tool: "wiki.write_page".to_string(),
                    status: "started".to_string(),
                    detail: Some(path.to_string()),
                },
            );
            emit_event(
                &mut events,
                &event_sink,
                AgentEvent::tool_start("wiki.write_page", Some(path.to_string())),
            );
            let tool_context = self.tool_context();
            match execute_tool_with_cancellation(
                tool_registry.execute(
                    "wiki.write_page",
                    serde_json::json!({
                        "path": path,
                        "content": content,
                        "allowOverwrite": write_call.allow_overwrite.unwrap_or(false),
                    }),
                    tool_context,
                ),
                cancellation.as_ref(),
            )
            .await
            .and_then(|value| {
                serde_json::from_value::<AgentReference>(value)
                    .map_err(|err| format!("Invalid wiki.write_page result: {err}"))
            }) {
                Ok(reference) => {
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::ReferenceAdded {
                            reference: reference.clone(),
                        },
                    );
                    references.push(reference);
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "wiki.write_page".to_string(),
                            status: "completed".to_string(),
                            detail: Some(path.to_string()),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end("wiki.write_page", Some(path.to_string())),
                    );
                    retrieval_parts.push(format!("wiki.write_page wrote {path}."));
                }
                Err(err) => {
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "wiki.write_page".to_string(),
                            status: "failed".to_string(),
                            detail: Some(err.clone()),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end("wiki.write_page", Some(format!("failed: {err}"))),
                    );
                }
            }
        }

        if deep_research {
            tool_emit_event(
                &mut tool_events,
                &mut events,
                &event_sink,
                AgentToolEvent {
                    tool: "deep_research.run".to_string(),
                    status: "started".to_string(),
                    detail: Some(message.to_string()),
                },
            );
            emit_event(
                &mut events,
                &event_sink,
                AgentEvent::tool_start("deep_research.run", Some(message.to_string())),
            );
        }

        if should_search_wiki {
            check_cancel(cancellation.as_ref())?;
            permission_policy.require(AgentCapability::SearchWiki)?;
            let wiki_query = planned_queries
                .get("wiki.search")
                .map(String::as_str)
                .unwrap_or(message);
            tool_emit_event(
                &mut tool_events,
                &mut events,
                &event_sink,
                AgentToolEvent {
                    tool: "wiki.search".to_string(),
                    status: "started".to_string(),
                    detail: Some(wiki_query.to_string()),
                },
            );
            emit_event(
                &mut events,
                &event_sink,
                AgentEvent::tool_start("wiki.search", Some(wiki_query.to_string())),
            );
            let top_k = request
                .top_k
                .unwrap_or(DEFAULT_CHAT_SEARCH_RESULTS)
                .clamp(1, MAX_CHAT_SEARCH_RESULTS);
            let wiki_search = execute_tool_with_cancellation(
                tool_registry.execute(
                    "wiki.search",
                    serde_json::json!({
                        "query": wiki_query,
                        "topK": top_k,
                        "includeContent": request.include_content.unwrap_or(false)
                    }),
                    self.tool_context(),
                ),
                cancellation.as_ref(),
            )
            .await
            .and_then(|value| {
                serde_json::from_value::<tools::WikiSearchToolOutput>(value)
                    .map_err(|err| format!("Invalid wiki.search result: {err}"))
            });
            match wiki_search {
                Ok(search) => {
                    check_cancel(cancellation.as_ref())?;
                    let search_refs = search.references;
                    for reference in &search_refs {
                        emit_event(
                            &mut events,
                            &event_sink,
                            AgentEvent::ReferenceAdded {
                                reference: reference.clone(),
                            },
                        );
                    }
                    let search_count = search_refs.len();
                    references.extend(search_refs);
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "wiki.search".to_string(),
                            status: "completed".to_string(),
                            detail: Some(format!(
                                "{} result(s), mode={}, tokenHits={}, vectorHits={}",
                                search_count, search.mode, search.token_hits, search.vector_hits
                            )),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end(
                            "wiki.search",
                            Some(format!("{search_count} result(s)")),
                        ),
                    );
                    retrieval_parts.push(build_retrieval_answer(message, &references));
                }
                Err(err) => {
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "wiki.search".to_string(),
                            status: "failed".to_string(),
                            detail: Some(err.clone()),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end("wiki.search", Some(format!("failed: {err}"))),
                    );
                }
            }
            if matches!(request.mode, AgentMode::Deep) && !references.is_empty() {
                permission_policy.require(AgentCapability::ReadProject)?;
                let excerpts = references
                    .iter()
                    .filter(|reference| reference.kind == "wiki")
                    .take(2)
                    .filter_map(|reference| {
                        tools::read_wiki_page(&self.project_path, &reference.path)
                            .ok()
                            .map(|content| {
                                format!(
                                    "Excerpt from {}:\n{}",
                                    reference.path,
                                    collapse_whitespace(&content)
                                        .chars()
                                        .take(2_000)
                                        .collect::<String>()
                                )
                            })
                    })
                    .collect::<Vec<_>>();
                if !excerpts.is_empty() {
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "wiki.read_page".to_string(),
                            status: "completed".to_string(),
                            detail: Some(format!("{} excerpt(s)", excerpts.len())),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end(
                            "wiki.read_page",
                            Some(format!("{} excerpt(s)", excerpts.len())),
                        ),
                    );
                    retrieval_parts.push(excerpts.join("\n\n"));
                }
            }
        } else if request.tools.wiki {
            retrieval_parts.push(format!(
                "Router intent={} did not require immediate wiki.search for this turn.",
                intent_label(router.intent)
            ));
        }

        if should_include_sources {
            check_cancel(cancellation.as_ref())?;
            permission_policy.require(AgentCapability::ReadSource)?;
            let source_query = planned_queries
                .get("source.search")
                .map(String::as_str)
                .unwrap_or(message);
            tool_emit_event(
                &mut tool_events,
                &mut events,
                &event_sink,
                AgentToolEvent {
                    tool: "source.search".to_string(),
                    status: "started".to_string(),
                    detail: Some(source_query.to_string()),
                },
            );
            emit_event(
                &mut events,
                &event_sink,
                AgentEvent::tool_start("source.search", Some(source_query.to_string())),
            );
            match execute_tool_with_cancellation(
                tool_registry.execute(
                    "source.search",
                    serde_json::json!({
                        "query": source_query,
                        "topK": request
                            .top_k
                            .unwrap_or(DEFAULT_CHAT_SEARCH_RESULTS)
                            .clamp(1, MAX_CHAT_SEARCH_RESULTS)
                    }),
                    self.tool_context(),
                ),
                cancellation.as_ref(),
            )
            .await
            .and_then(|value| {
                serde_json::from_value::<Vec<AgentReference>>(value)
                    .map_err(|err| format!("Invalid source.search result: {err}"))
            }) {
                Ok(source_refs) => {
                    for reference in &source_refs {
                        emit_event(
                            &mut events,
                            &event_sink,
                            AgentEvent::ReferenceAdded {
                                reference: reference.clone(),
                            },
                        );
                    }
                    let count = source_refs.len();
                    references.extend(source_refs);
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "source.search".to_string(),
                            status: "completed".to_string(),
                            detail: Some(format!("{count} result(s)")),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end("source.search", Some(format!("{count} result(s)"))),
                    );
                }
                Err(err) => {
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "source.search".to_string(),
                            status: "failed".to_string(),
                            detail: Some(err.clone()),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end("source.search", Some(format!("failed: {err}"))),
                    );
                }
            }
        }

        if should_search_graph {
            check_cancel(cancellation.as_ref())?;
            permission_policy.require(AgentCapability::ReadProject)?;
            let graph_query = planned_queries
                .get("graph.search")
                .map(String::as_str)
                .unwrap_or(message);
            tool_emit_event(
                &mut tool_events,
                &mut events,
                &event_sink,
                AgentToolEvent {
                    tool: "graph.search".to_string(),
                    status: "started".to_string(),
                    detail: Some(graph_query.to_string()),
                },
            );
            emit_event(
                &mut events,
                &event_sink,
                AgentEvent::tool_start("graph.search", Some(graph_query.to_string())),
            );
            match execute_tool_with_cancellation(
                tool_registry.execute(
                    "graph.search",
                    serde_json::json!({
                        "query": graph_query,
                        "topK": request
                            .top_k
                            .unwrap_or(DEFAULT_CHAT_SEARCH_RESULTS)
                            .clamp(1, MAX_CHAT_SEARCH_RESULTS)
                    }),
                    self.tool_context(),
                ),
                cancellation.as_ref(),
            )
            .await
            .and_then(|value| {
                serde_json::from_value::<Vec<AgentReference>>(value)
                    .map_err(|err| format!("Invalid graph.search result: {err}"))
            }) {
                Ok(graph_refs) => {
                    for reference in &graph_refs {
                        emit_event(
                            &mut events,
                            &event_sink,
                            AgentEvent::ReferenceAdded {
                                reference: reference.clone(),
                            },
                        );
                    }
                    let count = graph_refs.len();
                    references.extend(graph_refs);
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "graph.search".to_string(),
                            status: "completed".to_string(),
                            detail: Some(format!("{count} result(s)")),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end("graph.search", Some(format!("{count} result(s)"))),
                    );
                }
                Err(err) => {
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "graph.search".to_string(),
                            status: "failed".to_string(),
                            detail: Some(err.clone()),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end("graph.search", Some(format!("failed: {err}"))),
                    );
                }
            }
        }

        if should_run_web {
            check_cancel(cancellation.as_ref())?;
            permission_policy.require(AgentCapability::Network)?;
            let web_query = planned_queries
                .get("web.search")
                .map(String::as_str)
                .unwrap_or(message);
            tool_emit_event(
                &mut tool_events,
                &mut events,
                &event_sink,
                AgentToolEvent {
                    tool: "web.search".to_string(),
                    status: "started".to_string(),
                    detail: Some(web_query.to_string()),
                },
            );
            emit_event(
                &mut events,
                &event_sink,
                AgentEvent::tool_start("web.search", Some(web_query.to_string())),
            );
            match execute_tool_with_cancellation(
                tool_registry.execute(
                    "web.search",
                    serde_json::json!({
                        "query": web_query,
                        "topK": request
                            .top_k
                            .unwrap_or(DEFAULT_CHAT_SEARCH_RESULTS)
                            .clamp(1, MAX_CHAT_SEARCH_RESULTS)
                    }),
                    self.tool_context(),
                ),
                cancellation.as_ref(),
            )
            .await
            .and_then(|value| {
                serde_json::from_value::<Vec<AgentReference>>(value)
                    .map_err(|err| format!("Invalid web.search result: {err}"))
            }) {
                Ok(web_refs) => {
                    check_cancel(cancellation.as_ref())?;
                    for reference in &web_refs {
                        emit_event(
                            &mut events,
                            &event_sink,
                            AgentEvent::ReferenceAdded {
                                reference: reference.clone(),
                            },
                        );
                    }
                    let count = web_refs.len();
                    references.extend(web_refs);
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "web.search".to_string(),
                            status: "completed".to_string(),
                            detail: Some(format!("{count} result(s)")),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end("web.search", Some(format!("{count} result(s)"))),
                    );
                }
                Err(err) => {
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "web.search".to_string(),
                            status: "failed".to_string(),
                            detail: Some(err.clone()),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end("web.search", Some(format!("failed: {err}"))),
                    );
                }
            }
        }

        if should_run_anytxt {
            check_cancel(cancellation.as_ref())?;
            permission_policy.require(AgentCapability::Network)?;
            let anytxt_query = planned_queries
                .get("anytxt.search")
                .map(String::as_str)
                .unwrap_or(message);
            tool_emit_event(
                &mut tool_events,
                &mut events,
                &event_sink,
                AgentToolEvent {
                    tool: "anytxt.search".to_string(),
                    status: "started".to_string(),
                    detail: Some(anytxt_query.to_string()),
                },
            );
            emit_event(
                &mut events,
                &event_sink,
                AgentEvent::tool_start("anytxt.search", Some(anytxt_query.to_string())),
            );
            match execute_tool_with_cancellation(
                tool_registry.execute(
                    "anytxt.search",
                    serde_json::json!({
                        "query": anytxt_query,
                        "topK": request
                            .top_k
                            .unwrap_or(DEFAULT_CHAT_SEARCH_RESULTS)
                            .clamp(1, MAX_CHAT_SEARCH_RESULTS)
                    }),
                    self.tool_context(),
                ),
                cancellation.as_ref(),
            )
            .await
            .and_then(|value| {
                serde_json::from_value::<Vec<AgentReference>>(value)
                    .map_err(|err| format!("Invalid anytxt.search result: {err}"))
            }) {
                Ok(anytxt_refs) => {
                    check_cancel(cancellation.as_ref())?;
                    for reference in &anytxt_refs {
                        emit_event(
                            &mut events,
                            &event_sink,
                            AgentEvent::ReferenceAdded {
                                reference: reference.clone(),
                            },
                        );
                    }
                    let count = anytxt_refs.len();
                    references.extend(anytxt_refs);
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "anytxt.search".to_string(),
                            status: "completed".to_string(),
                            detail: Some(format!("{count} result(s)")),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end("anytxt.search", Some(format!("{count} result(s)"))),
                    );
                }
                Err(err) => {
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "anytxt.search".to_string(),
                            status: "failed".to_string(),
                            detail: Some(err.clone()),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::tool_end("anytxt.search", Some(format!("failed: {err}"))),
                    );
                }
            }
        }

        if deep_research {
            tool_emit_event(
                &mut tool_events,
                &mut events,
                &event_sink,
                AgentToolEvent {
                    tool: "deep_research.run".to_string(),
                    status: "completed".to_string(),
                    detail: Some(format!("{} reference(s)", references.len())),
                },
            );
            emit_event(
                &mut events,
                &event_sink,
                AgentEvent::tool_end(
                    "deep_research.run",
                    Some(format!("{} reference(s)", references.len())),
                ),
            );
        }

        if retrieval_parts.is_empty() {
            if !request.tools.wiki && !request.tools.web && !request.tools.anytxt {
                retrieval_parts.push("No Agent tools were enabled for this request. Enable wiki, web, or AnyTXT tools to let the backend Agent retrieve supporting context.".to_string());
            } else {
                retrieval_parts.push(
                    "No Agent tools ran before generation. Available tools were exposed as model hints."
                        .to_string(),
                );
            }
        }
        let retrieval_summary = retrieval_parts.join("\n\n");
        let project_context = load_project_context(&self.project_path);
        let built_context = fit_context_to_model(
            build_agent_context(AgentContextInput {
                query: message,
                project: &project_context,
                router: &router,
                history: &request.history,
                skills: &skills,
                references: &references,
                retrieval_summary: &retrieval_summary,
            }),
            self.llm_config.as_ref(),
        );

        let answer = if let Some(config) = self
            .llm_config
            .as_ref()
            .filter(|cfg| cfg.is_usable_for_backend_http())
        {
            check_cancel(cancellation.as_ref())?;
            let client = LlmClient::new(config.clone())?;
            tool_emit_event(
                &mut tool_events,
                &mut events,
                &event_sink,
                AgentToolEvent {
                    tool: "llm.generate".to_string(),
                    status: "started".to_string(),
                    detail: Some(format!(
                        "{}:{}",
                        client.provider_name(),
                        client.model_name()
                    )),
                },
            );
            let generation = if event_sink.is_some() {
                generate_with_cancellation_stream(
                    &client,
                    &built_context.system,
                    &built_context.user,
                    &request.images,
                    cancellation.as_ref(),
                    |delta| {
                        emit_event(
                            &mut events,
                            &event_sink,
                            AgentEvent::MessageDelta {
                                text: delta.to_string(),
                            },
                        );
                    },
                )
                .await
            } else {
                generate_with_cancellation(
                    &client,
                    &built_context.system,
                    &built_context.user,
                    &request.images,
                    cancellation.as_ref(),
                )
                .await
            };
            match generation {
                Ok(answer) => {
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "llm.generate".to_string(),
                            status: "completed".to_string(),
                            detail: None,
                        },
                    );
                    answer
                }
                Err(err) => {
                    tool_emit_event(
                        &mut tool_events,
                        &mut events,
                        &event_sink,
                        AgentToolEvent {
                            tool: "llm.generate".to_string(),
                            status: "failed".to_string(),
                            detail: Some(err.clone()),
                        },
                    );
                    emit_event(
                        &mut events,
                        &event_sink,
                        AgentEvent::Error {
                            message: err.clone(),
                        },
                    );
                    return Err(err);
                }
            }
        } else {
            retrieval_summary
        };
        emit_event(
            &mut events,
            &event_sink,
            AgentEvent::Done {
                session_id: session_id.clone(),
            },
        );
        let usage = AgentUsage {
            prompt_chars: built_context.system.len() + built_context.user.len(),
            completion_chars: answer.len(),
            reference_count: references.len(),
            tool_event_count: tool_events.len(),
        };

        Ok(AgentChatResponse {
            ok: true,
            project_id: self.project_id.clone(),
            session_id,
            mode: request.mode,
            message: answer,
            references,
            tool_events,
            events,
            usage: Some(usage),
        })
    }
}

impl AgentRuntime {
    fn tool_context(&self) -> tools::ToolContext<'_> {
        tools::ToolContext {
            project_path: &self.project_path,
            embedding_config: self.embedding_config.clone(),
            web_search_config: self.web_search_config.clone(),
            anytxt_config: self.anytxt_config.clone(),
        }
    }

    async fn plan_tools_with_model(
        &self,
        message: &str,
        mode: AgentMode,
        tools: &super::types::AgentToolOptions,
        cancellation: Option<&AgentCancellationToken>,
    ) -> Result<ModelToolPlan, String> {
        if !should_plan_tools_with_model(message, mode, tools) {
            return Ok(ModelToolPlan::default());
        }
        let Some(config) = self
            .llm_config
            .as_ref()
            .filter(|cfg| cfg.is_usable_for_backend_http())
        else {
            return Ok(ModelToolPlan::default());
        };
        check_cancel(cancellation)?;
        let mut available = vec![
            "wiki.search",
            "source.search",
            "graph.search",
            "wiki.write_page",
        ];
        if tools.web {
            available.push("web.search");
        }
        if tools.anytxt {
            available.push("anytxt.search");
        }
        let system = "You are an agent tool planner. Return only compact JSON. Do not explain.";
        let user = format!(
            "User request:\n{message}\n\nAvailable tools: {}\n\nReturn JSON exactly like {{\"toolCalls\":[{{\"tool\":\"wiki.search\",\"query\":\"short query\"}}]}}. Use an empty array when no tool is needed. Prefer web.search only for current/external information. Prefer anytxt.search only for user files outside the wiki. Prefer wiki.search for project knowledge. Use wiki.write_page only when the user explicitly asks to create a wiki page; include path under wiki/ ending in .md and full Markdown content. Existing pages are create-only by default; include allowOverwrite:true only when the user explicitly asks to overwrite or update an existing wiki page.",
            available.join(", ")
        );
        let client = LlmClient::new(config.clone())?;
        let raw = generate_with_cancellation(&client, system, &user, &[], cancellation).await?;
        parse_model_tool_plan(&raw)
    }
}

fn should_plan_tools_with_model(
    message: &str,
    mode: AgentMode,
    tools: &super::types::AgentToolOptions,
) -> bool {
    if matches!(mode, AgentMode::Fast) {
        return false;
    }
    tools.web || tools.anytxt || looks_like_wiki_write_request(message)
}

fn looks_like_wiki_write_request(message: &str) -> bool {
    let lower = message.to_lowercase();
    let mentions_wiki = lower.contains("wiki")
        || lower.contains("page")
        || lower.contains("页面")
        || lower.contains("词条")
        || lower.contains("笔记");
    let asks_write = lower.contains("create")
        || lower.contains("write")
        || lower.contains("update")
        || lower.contains("save")
        || lower.contains("生成")
        || lower.contains("创建")
        || lower.contains("新建")
        || lower.contains("写入")
        || lower.contains("保存")
        || lower.contains("更新");
    mentions_wiki && asks_write
}

fn planned_tool_queries(plan: &ModelToolPlan, fallback_query: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for call in &plan.tool_calls {
        let tool = call.tool.trim();
        if !matches!(
            tool,
            "wiki.search"
                | "source.search"
                | "graph.search"
                | "web.search"
                | "anytxt.search"
                | "wiki.write_page"
        ) {
            continue;
        }
        let query = call
            .query
            .as_deref()
            .map(str::trim)
            .filter(|query| !query.is_empty())
            .unwrap_or(fallback_query);
        out.entry(tool.to_string())
            .or_insert_with(|| query.to_string());
    }
    out
}

fn parse_model_tool_plan(raw: &str) -> Result<ModelToolPlan, String> {
    let json_text = extract_json_object(raw).unwrap_or(raw).trim();
    serde_json::from_str(json_text).map_err(|err| format!("Invalid Agent tool plan JSON: {err}"))
}

fn extract_json_object(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    Some(&raw[start..=end])
}

fn check_cancel(cancellation: Option<&AgentCancellationToken>) -> Result<(), String> {
    if let Some(token) = cancellation {
        token.check()?;
    }
    Ok(())
}

fn fit_context_to_model(
    mut context: BuiltAgentContext,
    config: Option<&LlmConfig>,
) -> BuiltAgentContext {
    let Some(max_context_size) = config.and_then(|cfg| cfg.max_context_size) else {
        return context;
    };
    let max_chars = max_context_size.clamp(8_000, 400_000);
    let total_chars = context.system.chars().count() + context.user.chars().count();
    if total_chars <= max_chars {
        return context;
    }
    let user_budget = max_chars
        .saturating_sub(context.system.chars().count())
        .max(4_000);
    context.user = trim_chars(&context.user, user_budget);
    context
}

// Tool futures may include network I/O or blocking-pool filesystem scans.
// Cancelling the turn should stop waiting for them immediately. A blocking task
// already running in Tokio's blocking pool cannot be force-killed, so the
// contract is "stop the Agent turn promptly", not "terminate the OS work".
async fn execute_tool_with_cancellation<F>(
    future: F,
    cancellation: Option<&AgentCancellationToken>,
) -> Result<serde_json::Value, String>
where
    F: Future<Output = Result<serde_json::Value, String>>,
{
    if let Some(token) = cancellation {
        tokio::select! {
            biased;
            _ = token.cancelled() => Err("Agent turn cancelled".to_string()),
            result = future => result,
        }
    } else {
        future.await
    }
}

fn validate_images(images: &[super::types::AgentImage]) -> Result<(), String> {
    if images.len() > MAX_IMAGES_PER_TURN {
        return Err(format!(
            "At most {MAX_IMAGES_PER_TURN} images can be attached to one Agent turn"
        ));
    }
    for image in images {
        let media_type = image.media_type.trim();
        if !matches!(
            media_type,
            "image/png" | "image/jpeg" | "image/webp" | "image/gif"
        ) {
            return Err(format!("Unsupported image media type: {media_type}"));
        }
        if image.data_base64.len() > MAX_IMAGE_BASE64_BYTES {
            return Err("Attached image is too large".to_string());
        }
        if image.data_base64.trim().is_empty() {
            return Err("Attached image is empty".to_string());
        }
    }
    Ok(())
}

fn emit_event(
    events: &mut Vec<AgentEvent>,
    event_sink: &Option<AgentEventSink>,
    event: AgentEvent,
) {
    if let Some(sink) = event_sink {
        sink(event.clone());
    }
    events.push(event);
}

fn tool_emit_event(
    tool_events: &mut Vec<AgentToolEvent>,
    _events: &mut Vec<AgentEvent>,
    _event_sink: &Option<AgentEventSink>,
    tool_event: AgentToolEvent,
) {
    tool_events.push(tool_event);
}

async fn generate_with_cancellation<P: AgentLlmProvider>(
    provider: &P,
    system: &str,
    user: &str,
    images: &[super::types::AgentImage],
    cancellation: Option<&AgentCancellationToken>,
) -> Result<String, String> {
    if let Some(token) = cancellation {
        tokio::select! {
            biased;
            _ = token.cancelled() => Err("Agent turn cancelled".to_string()),
            result = provider.generate_text(system, user, images) => result,
        }
    } else {
        provider.generate_text(system, user, images).await
    }
}

async fn generate_with_cancellation_stream<P, F>(
    provider: &P,
    system: &str,
    user: &str,
    images: &[super::types::AgentImage],
    cancellation: Option<&AgentCancellationToken>,
    on_delta: F,
) -> Result<String, String>
where
    P: AgentLlmProvider,
    F: FnMut(&str) + Send,
{
    if let Some(token) = cancellation {
        tokio::select! {
            biased;
            _ = token.cancelled() => Err("Agent turn cancelled".to_string()),
            result = provider.generate_text_stream(system, user, images, Box::new(on_delta)) => result,
        }
    } else {
        provider
            .generate_text_stream(system, user, images, Box::new(on_delta))
            .await
    }
}

fn build_retrieval_answer(query: &str, references: &[AgentReference]) -> String {
    if references.is_empty() {
        return format!(
            "I searched the current LLM Wiki project for \"{query}\" but did not find matching wiki pages."
        );
    }

    let mut out = format!(
        "I searched the current LLM Wiki project for \"{query}\" and found {} relevant page(s):",
        references.len()
    );
    for (idx, reference) in references.iter().take(MAX_CHAT_SEARCH_RESULTS).enumerate() {
        out.push_str(&format!(
            "\n{}. {} ({})",
            idx + 1,
            reference.title,
            reference.path
        ));
        if let Some(snippet) = reference.snippet.as_deref() {
            out.push_str(&format!("\n   {}", collapse_whitespace(snippet)));
        }
    }
    out
}

fn mode_label(mode: AgentMode) -> &'static str {
    match mode {
        AgentMode::Fast => "fast",
        AgentMode::Standard => "standard",
        AgentMode::Deep => "deep",
        AgentMode::LocalFirst => "local_first",
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;
    use crate::agent::types::{AgentMode, AgentToolOptions};

    fn temp_project(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("llm-wiki-agent-test-{name}-{}", Uuid::new_v4()));
        fs::create_dir_all(root.join("wiki").join("concepts")).unwrap();
        root
    }

    #[tokio::test]
    async fn run_once_searches_wiki_and_returns_references() {
        let project = temp_project("search");
        fs::write(
            project.join("wiki").join("concepts").join("agent-runtime.md"),
            "---\ntitle: Agent Runtime\n---\n# Agent Runtime\n\nRust backend agent substrate with tool calls.",
        )
        .unwrap();

        let runtime = AgentRuntime::new(
            "project-1",
            project.to_string_lossy(),
            None,
            None,
            None,
            None,
        );
        let response = runtime
            .run_once(AgentChatRequest {
                message: "agent runtime".to_string(),
                session_id: Some("s1".to_string()),
                mode: AgentMode::Standard,
                tools: AgentToolOptions {
                    wiki: true,
                    web: false,
                    anytxt: false,
                },
                top_k: Some(3),
                include_content: Some(false),
                history: Vec::new(),
                skills: Vec::new(),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(response.ok);
        assert_eq!(response.session_id, "s1");
        assert_eq!(response.references.len(), 1);
        assert_eq!(
            response.references[0].path,
            "wiki/concepts/agent-runtime.md"
        );
        assert!(response.message.contains("Agent Runtime"));
        assert_eq!(response.tool_events[0].tool, "wiki.search");
    }

    #[tokio::test]
    async fn run_once_rejects_empty_message() {
        let project = temp_project("empty");
        let runtime = AgentRuntime::new(
            "project-1",
            project.to_string_lossy(),
            None,
            None,
            None,
            None,
        );
        let err = runtime
            .run_once(AgentChatRequest {
                message: "   ".to_string(),
                session_id: None,
                mode: AgentMode::Fast,
                tools: AgentToolOptions::default(),
                top_k: None,
                include_content: None,
                history: Vec::new(),
                skills: Vec::new(),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert_eq!(err, "message is required");
    }

    #[test]
    fn fit_context_to_model_honors_configured_context_size() {
        let context = BuiltAgentContext {
            system: "system".repeat(100),
            user: "user".repeat(10_000),
        };
        let config = LlmConfig {
            provider: "custom".to_string(),
            api_key: String::new(),
            model: "local".to_string(),
            ollama_url: String::new(),
            custom_endpoint: "http://127.0.0.1:11434/v1".to_string(),
            azure_api_version: None,
            api_mode: None,
            reasoning: None,
            max_context_size: Some(8_000),
        };
        let fitted = fit_context_to_model(context, Some(&config));
        assert!(fitted.system.contains("system"));
        assert!(fitted.user.chars().count() <= 7_400);
        assert!(fitted.user.ends_with("..."));
    }

    #[tokio::test]
    async fn run_once_can_disable_wiki_tool() {
        let project = temp_project("disabled");
        let runtime = AgentRuntime::new(
            "project-1",
            project.to_string_lossy(),
            None,
            None,
            None,
            None,
        );
        let response = runtime
            .run_once(AgentChatRequest {
                message: "anything".to_string(),
                session_id: None,
                mode: AgentMode::LocalFirst,
                tools: AgentToolOptions {
                    wiki: false,
                    web: false,
                    anytxt: false,
                },
                top_k: None,
                include_content: None,
                history: Vec::new(),
                skills: Vec::new(),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(response.references.is_empty());
        assert!(response.tool_events.is_empty());
        assert!(response.message.contains("No Agent tools were enabled"));
    }

    #[tokio::test]
    async fn run_once_in_fast_mode_exposes_tools_without_presearching() {
        let project = temp_project("fast");
        fs::write(
            project.join("overview.md"),
            "This project covers search routing.",
        )
        .unwrap();
        fs::write(
            project.join("wiki").join("concepts").join("routing.md"),
            "# Routing\n\nThis should not be searched in fast mode.",
        )
        .unwrap();

        let runtime = AgentRuntime::new(
            "project-1",
            project.to_string_lossy(),
            None,
            None,
            None,
            None,
        );
        let response = runtime
            .run_once(AgentChatRequest {
                message: "routing details?".to_string(),
                session_id: None,
                mode: AgentMode::Fast,
                tools: AgentToolOptions {
                    wiki: true,
                    web: true,
                    anytxt: false,
                },
                top_k: Some(3),
                include_content: Some(false),
                history: Vec::new(),
                skills: Vec::new(),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(response.references.is_empty());
        assert!(response
            .tool_events
            .iter()
            .any(|event| event.tool == "web.search" && event.status == "available"));
        assert!(response.message.contains("Router intent"));
    }

    #[tokio::test]
    async fn optional_tool_failure_does_not_abort_turn() {
        let project = temp_project("web-fail");
        let runtime = AgentRuntime::new(
            "project-1",
            project.to_string_lossy(),
            None,
            None,
            None,
            None,
        );
        let response = runtime
            .run_once(AgentChatRequest {
                message: "latest external update".to_string(),
                session_id: None,
                mode: AgentMode::Standard,
                tools: AgentToolOptions {
                    wiki: false,
                    web: true,
                    anytxt: false,
                },
                top_k: Some(3),
                include_content: Some(false),
                history: Vec::new(),
                skills: Vec::new(),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(response.ok);
        assert!(response
            .tool_events
            .iter()
            .any(|event| event.tool == "web.search" && event.status == "failed"));
        assert!(!response
            .events
            .iter()
            .any(|event| matches!(event, AgentEvent::Error { .. })));
    }

    #[tokio::test]
    async fn run_once_can_include_raw_source_search_for_source_questions() {
        let project = temp_project("source");
        let source_dir = project.join("raw").join("sources");
        fs::create_dir_all(&source_dir).unwrap();
        fs::write(source_dir.join("coal.txt"), "煤矿安全治理 source details.").unwrap();

        let runtime = AgentRuntime::new(
            "project-1",
            project.to_string_lossy(),
            None,
            None,
            None,
            None,
        );
        let response = runtime
            .run_once(AgentChatRequest {
                message: "原始资料 煤矿安全".to_string(),
                session_id: None,
                mode: AgentMode::Deep,
                tools: AgentToolOptions::default(),
                top_k: Some(3),
                include_content: Some(false),
                history: Vec::new(),
                skills: Vec::new(),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(response
            .references
            .iter()
            .any(|reference| reference.kind == "source"));
        assert!(response
            .tool_events
            .iter()
            .any(|event| event.tool == "source.search"));
    }

    #[tokio::test]
    async fn deep_mode_reads_top_wiki_pages_after_search() {
        let project = temp_project("deep-read");
        fs::write(
            project.join("wiki").join("concepts").join("deep-agent.md"),
            "---\ntitle: Deep Agent\n---\n# Deep Agent\n\nDetailed evidence that should be read in deep mode.",
        )
        .unwrap();

        let runtime = AgentRuntime::new(
            "project-1",
            project.to_string_lossy(),
            None,
            None,
            None,
            None,
        );
        let response = runtime
            .run_once(AgentChatRequest {
                message: "deep agent evidence".to_string(),
                session_id: None,
                mode: AgentMode::Deep,
                tools: AgentToolOptions::default(),
                top_k: Some(3),
                include_content: Some(false),
                history: Vec::new(),
                skills: Vec::new(),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(response
            .tool_events
            .iter()
            .any(|event| event.tool == "wiki.read_page" && event.status == "completed"));
        assert!(response.message.contains("Detailed evidence"));
    }

    #[tokio::test]
    async fn turn_start_event_uses_api_mode_label() {
        let project = temp_project("mode-label");
        let runtime = AgentRuntime::new(
            "project-1",
            project.to_string_lossy(),
            None,
            None,
            None,
            None,
        );
        let response = runtime
            .run_once(AgentChatRequest {
                message: "hello".to_string(),
                session_id: None,
                mode: AgentMode::LocalFirst,
                tools: AgentToolOptions {
                    wiki: false,
                    web: false,
                    anytxt: false,
                },
                top_k: None,
                include_content: None,
                history: Vec::new(),
                skills: Vec::new(),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(response
            .events
            .iter()
            .any(|event| matches!(event, AgentEvent::TurnStart { mode } if mode == "local_first")));
    }

    #[tokio::test]
    async fn graph_questions_run_graph_search_tool() {
        let project = temp_project("graph");
        fs::write(
            project.join("wiki").join("concepts").join("graph.md"),
            "---\ntitle: Graph Relations\n---\n# Graph Relations\n\n[[A]] links to [[B]].",
        )
        .unwrap();

        let runtime = AgentRuntime::new(
            "project-1",
            project.to_string_lossy(),
            None,
            None,
            None,
            None,
        );
        let response = runtime
            .run_once(AgentChatRequest {
                message: "知识图谱 Graph Relations".to_string(),
                session_id: None,
                mode: AgentMode::Standard,
                tools: AgentToolOptions::default(),
                top_k: Some(3),
                include_content: None,
                history: Vec::new(),
                skills: Vec::new(),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(response
            .tool_events
            .iter()
            .any(|event| event.tool == "graph.search" && event.status == "completed"));
        assert!(response
            .references
            .iter()
            .any(|reference| reference.kind == "graph"));
    }

    #[test]
    fn parses_model_tool_plan_from_wrapped_json() {
        let plan = parse_model_tool_plan(
            "```json\n{\"toolCalls\":[{\"tool\":\"web.search\",\"query\":\"llm wiki release\"}]}\n```",
        )
        .unwrap();
        let queries = planned_tool_queries(&plan, "fallback");
        assert_eq!(
            queries.get("web.search").map(String::as_str),
            Some("llm wiki release")
        );
    }

    #[test]
    fn ignores_unknown_model_tool_names() {
        let plan = parse_model_tool_plan(
            "{\"toolCalls\":[{\"tool\":\"shell.exec\",\"query\":\"rm -rf\"},{\"tool\":\"wiki.search\"}]}",
        )
        .unwrap();
        let queries = planned_tool_queries(&plan, "safe query");
        assert_eq!(queries.len(), 1);
        assert_eq!(
            queries.get("wiki.search").map(String::as_str),
            Some("safe query")
        );
    }

    #[test]
    fn write_requests_enable_model_tool_planning_without_external_tools() {
        let tools = AgentToolOptions {
            wiki: true,
            web: false,
            anytxt: false,
        };
        assert!(should_plan_tools_with_model(
            "请创建一个 wiki 页面总结今天的发现",
            AgentMode::Standard,
            &tools
        ));
        assert!(!should_plan_tools_with_model(
            "普通问答不需要额外规划",
            AgentMode::Standard,
            &tools
        ));
        assert!(!should_plan_tools_with_model(
            "请创建一个 wiki 页面",
            AgentMode::Fast,
            &tools
        ));
    }

    #[test]
    fn validates_agent_image_limits() {
        let valid = super::super::types::AgentImage {
            media_type: "image/png".to_string(),
            data_base64: "abcd".to_string(),
        };
        assert!(validate_images(std::slice::from_ref(&valid)).is_ok());

        let invalid = super::super::types::AgentImage {
            media_type: "image/bmp".to_string(),
            data_base64: "abcd".to_string(),
        };
        assert!(validate_images(&[invalid]).is_err());

        let too_many = vec![valid; MAX_IMAGES_PER_TURN + 1];
        assert!(validate_images(&too_many).is_err());
    }
}
