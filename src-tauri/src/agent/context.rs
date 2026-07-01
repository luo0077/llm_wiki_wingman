use std::fs;
use std::path::Path;

use super::router::{QueryIntent, RouterDecision};
use super::skills::AgentSkill;
use super::types::{AgentConversationMessage, AgentReference};

const MAX_OVERVIEW_CHARS: usize = 8_000;
const MAX_SCHEMA_CHARS: usize = 6_000;
const MAX_HISTORY_CHARS: usize = 12_000;
const MAX_REFERENCE_CHARS: usize = 24_000;
const MAX_SKILL_CHARS: usize = 18_000;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectContext {
    pub overview: Option<String>,
    pub schema: Option<String>,
}

pub fn load_project_context(project_path: &str) -> ProjectContext {
    let root = Path::new(project_path);
    ProjectContext {
        overview: read_trimmed(root.join("overview.md"), MAX_OVERVIEW_CHARS)
            .or_else(|| read_trimmed(root.join("wiki").join("overview.md"), MAX_OVERVIEW_CHARS)),
        schema: read_trimmed(root.join("schema.md"), MAX_SCHEMA_CHARS)
            .or_else(|| read_trimmed(root.join("wiki").join("schema.md"), MAX_SCHEMA_CHARS)),
    }
}

#[derive(Debug, Clone)]
pub struct AgentContextInput<'a> {
    pub query: &'a str,
    pub project: &'a ProjectContext,
    pub router: &'a RouterDecision,
    pub history: &'a [AgentConversationMessage],
    pub skills: &'a [AgentSkill],
    pub references: &'a [AgentReference],
    pub retrieval_summary: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltAgentContext {
    pub system: String,
    pub user: String,
}

pub fn build_agent_context(input: AgentContextInput<'_>) -> BuiltAgentContext {
    BuiltAgentContext {
        system: build_system_context(input.project, input.router, input.skills),
        user: build_user_context(input),
    }
}

fn build_system_context(
    project: &ProjectContext,
    router: &RouterDecision,
    skills: &[AgentSkill],
) -> String {
    let mut out = [
        "You are the LLM Wiki backend Agent.",
        "Answer using the current project context, available tools, and cited references.",
        "If evidence is insufficient, say what is missing instead of inventing facts.",
        "When using references, mention the relevant page paths naturally.",
        "Do not claim that internet or local-source search is unavailable when those tools are enabled; use the provided tool context and tool hints.",
    ]
    .join("\n");

    out.push_str("\n\nTool policy:\n");
    out.push_str("- wiki.search can search generated wiki pages.\n");
    if router.should_hint_web {
        out.push_str("- web.search is available when current or external information is useful.\n");
    }
    if router.should_hint_anytxt {
        out.push_str(
            "- anytxt.search is available for local or remote file content indexed by AnyTXT.\n",
        );
    }
    out.push_str(&format!(
        "- Router hint: {:?}. {}\n",
        router.intent, router.rationale
    ));

    if let Some(overview) = project.overview.as_deref().filter(|v| !v.trim().is_empty()) {
        out.push_str("\n\nProject overview:\n");
        out.push_str(&trim_chars(overview, MAX_OVERVIEW_CHARS));
    }
    if let Some(schema) = project.schema.as_deref().filter(|v| !v.trim().is_empty()) {
        out.push_str("\n\nProject schema:\n");
        out.push_str(&trim_chars(schema, MAX_SCHEMA_CHARS));
    }
    if !skills.is_empty() {
        out.push_str("\n\nActive skills:\n");
        let mut remaining = MAX_SKILL_CHARS;
        for skill in skills {
            if remaining == 0 {
                break;
            }
            let rendered = format!(
                "\n## {}\n{}\n{}",
                skill.name,
                skill.description.trim(),
                skill.instructions.trim()
            );
            let piece = trim_chars(&rendered, remaining);
            remaining = remaining.saturating_sub(piece.len());
            out.push_str(&piece);
            out.push('\n');
        }
    }
    out
}

fn build_user_context(input: AgentContextInput<'_>) -> String {
    let mut out = String::new();
    if !input.history.is_empty() {
        out.push_str("Recent conversation history:\n");
        let mut history = String::new();
        for item in input.history.iter().rev().take(12).rev() {
            history.push_str(&format!(
                "{}: {}\n",
                item.role,
                collapse_whitespace(&item.content)
            ));
        }
        out.push_str(&trim_chars(&history, MAX_HISTORY_CHARS));
        out.push_str("\n\n");
    }

    out.push_str("Retrieved project context:\n");
    if input.references.is_empty() {
        out.push_str("No matching wiki references were found.\n\n");
    } else {
        let mut rendered = String::new();
        for (idx, reference) in input.references.iter().enumerate() {
            rendered.push_str(&format!(
                "{}. [{}] {} ({})\n",
                idx + 1,
                reference.kind,
                reference.title,
                reference.path
            ));
            if let Some(snippet) = reference.snippet.as_deref() {
                rendered.push_str(&format!("Snippet: {}\n", collapse_whitespace(snippet)));
            }
        }
        out.push_str(&trim_chars(&rendered, MAX_REFERENCE_CHARS));
        out.push('\n');
    }

    out.push_str("Retrieval summary:\n");
    out.push_str(&trim_chars(input.retrieval_summary, 8_000));
    out.push_str("\n\nLatest user request:\n");
    out.push_str(input.query.trim());
    out
}

fn read_trimmed(path: impl AsRef<Path>, max_chars: usize) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trim_chars(trimmed, max_chars))
    }
}

pub fn trim_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = value
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    out.push_str("...");
    out
}

pub fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn intent_label(intent: QueryIntent) -> &'static str {
    match intent {
        QueryIntent::NeedsInternalSearch => "internal_search",
        QueryIntent::NeedsExternalSearch => "external_search",
        QueryIntent::NeedsRawSourceSearch => "raw_source_search",
        QueryIntent::NeedsGraph => "graph",
        QueryIntent::NeedsWrite => "write",
        QueryIntent::SimpleConversational => "conversation",
        QueryIntent::Ambiguous => "ambiguous",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::router::route_query;
    use crate::agent::types::{AgentMode, AgentToolOptions};

    #[test]
    fn context_keeps_stable_project_context_before_latest_request() {
        let project = ProjectContext {
            overview: Some("Project overview text".to_string()),
            schema: Some("Schema text".to_string()),
        };
        let router = route_query(
            "latest policy",
            AgentMode::Standard,
            &AgentToolOptions::default(),
        );
        let ctx = build_agent_context(AgentContextInput {
            query: "latest policy",
            project: &project,
            router: &router,
            history: &[],
            skills: &[],
            references: &[],
            retrieval_summary: "None",
        });

        assert!(ctx.system.contains("Project overview text"));
        assert!(ctx.system.contains("Schema text"));
        assert!(ctx.user.ends_with("latest policy"));
    }

    #[test]
    fn trim_chars_is_utf8_safe() {
        assert_eq!(trim_chars("煤矿安全治理", 5), "煤矿...");
    }
}
