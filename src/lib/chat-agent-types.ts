/**
 * UI-facing Agent metadata types.
 *
 * The Agent execution engine lives in Rust (`src-tauri/src/agent`). Keep this
 * file intentionally limited to display/persistence shapes used by the React UI.
 * Do not reintroduce routing, retrieval, tool execution, or prompt-building
 * logic here; those belong in the Rust Agent runtime so API, MCP, and UI callers
 * share one backend behavior.
 */

export type ChatAgentEventStage =
  | "understanding"
  | "routing"
  | "tool_call"
  | "tool_result"
  | "searching_wiki"
  | "searching_graph"
  | "searching_web"
  | "searching_anytxt"
  | "reading_context"
  | "writing"

export interface ChatAgentEvent {
  stage: ChatAgentEventStage
  query?: string
  tool?: ChatAgentToolName
  message?: string
  count?: number
  status?: "running" | "success" | "error" | "skipped"
}

export type ChatAgentMode = "fast" | "standard" | "deep" | "local_first"

export type ChatAgentToolName =
  | "project_files"
  | "project_file_read"
  | "wiki_search"
  | "graph_search"
  | "web_search"
  | "anytxt_search"
  | "unknown_tool"

export interface ChatAgentStep {
  id: string
  type: "understanding" | "routing" | "tool_call" | "tool_result" | "final"
  tool?: ChatAgentToolName
  query?: string
  message?: string
  count?: number
  status?: "running" | "success" | "error" | "skipped"
}
