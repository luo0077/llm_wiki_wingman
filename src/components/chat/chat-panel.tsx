import { useRef, useEffect, useCallback, useState } from "react"
import { useTranslation } from "react-i18next"
import { invoke } from "@tauri-apps/api/core"
import { listen } from "@tauri-apps/api/event"
import { BookOpen, Plus, Trash2, MessageSquare, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ChatMessage, StreamingMessage, useSourceFiles, type ChatReferencePreview } from "./chat-message"
import { ChatInput, type ChatSendOptions } from "./chat-input"
import { useChatStore, chatMessagesToLLM, type MessageImage, type MessageReference } from "@/stores/chat-store"
import { useWikiStore } from "@/stores/wiki-store"
import { isReasoningOnlyResponseError, streamChat } from "@/lib/llm-client"
import { supportsImageInput } from "@/lib/llm-providers"
import { executeIngestWrites } from "@/lib/ingest"
import { deleteFile } from "@/commands/fs"
import { getFileName, normalizePath } from "@/lib/path-utils"
import { hasConfiguredAnyTxt } from "@/lib/anytxt-search"
import type { ChatAgentEvent } from "@/lib/chat-agent-types"
import type { ChatMessage as LlmChatMessage, ContentBlock } from "@/lib/llm-client"
import { FilePreview } from "@/components/editor/file-preview"
import { WikiReader } from "@/components/editor/wiki-reader"
import { FrontmatterPanel } from "@/components/editor/frontmatter-panel"
import { parseFrontmatter } from "@/lib/frontmatter"
import { getFileCategory } from "@/lib/file-types"
import { refreshProjectFileTree } from "@/lib/project-file-tree-refresh"

interface BackendAgentReference {
  title: string
  path: string
  kind: string
  snippet?: string
  score?: number
}

interface BackendAgentToolEvent {
  tool: string
  status: string
  detail?: string
}

interface BackendAgentEventPayload {
  sessionId: string
  runId?: string
  event: {
    type: string
    text?: string
    tool?: string
    input?: string
    output?: string
    message?: string
    reference?: BackendAgentReference
    sessionId?: string
  }
}

interface BackendAgentResponse {
  sessionId: string
  mode?: string
  message: string | { role?: string; content?: string }
  references?: BackendAgentReference[]
  toolEvents?: BackendAgentToolEvent[]
}

// Store the page mapping from the last query so SourceFilesBar can show which pages were cited
export let lastQueryPages: { title: string; path: string }[] = []

function formatDate(timestamp: number): string {
  const d = new Date(timestamp)
  const now = new Date()
  const isToday = d.toDateString() === now.toDateString()
  if (isToday) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  }
  return d.toLocaleDateString([], { month: "short", day: "numeric" })
}

function ConversationSidebar() {
  const { t } = useTranslation()
  const conversations = useChatStore((s) => s.conversations)
  const activeConversationId = useChatStore((s) => s.activeConversationId)
  const messages = useChatStore((s) => s.messages)
  const createConversation = useChatStore((s) => s.createConversation)
  const deleteConversation = useChatStore((s) => s.deleteConversation)
  const setActiveConversation = useChatStore((s) => s.setActiveConversation)

  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)

  function getMessageCount(convId: string): number {
    return messages.filter((m) => m.conversationId === convId).length
  }

  return (
    <div className="flex h-full w-[200px] flex-shrink-0 flex-col border-r bg-muted/30">
      <div className="border-b p-2">
        <Button
          variant="outline"
          size="sm"
          className="w-full gap-2"
          onClick={() => createConversation()}
        >
          <Plus className="h-3.5 w-3.5" />
          {t("chat.newChat")}
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto py-1">
        {sorted.length === 0 ? (
          <p className="px-3 py-4 text-xs text-muted-foreground text-center">
            {t("chat.noConversationsYet")}
          </p>
        ) : (
          sorted.map((conv) => {
            const isActive = conv.id === activeConversationId
            const msgCount = getMessageCount(conv.id)
            return (
              <div
                key={conv.id}
                className={`group relative mx-1 my-0.5 flex cursor-pointer flex-col rounded-md px-2 py-1.5 text-sm transition-colors ${
                  isActive
                    ? "bg-primary/10 text-primary"
                    : "hover:bg-accent text-foreground"
                }`}
                onClick={() => setActiveConversation(conv.id)}
                onMouseEnter={() => setHoveredId(conv.id)}
                onMouseLeave={() => setHoveredId(null)}
              >
                <div className="flex items-start justify-between gap-1">
                  <span className="line-clamp-2 flex-1 text-xs font-medium leading-snug">
                    {conv.title}
                  </span>
                  {hoveredId === conv.id && (
                    <button
                      className="flex-shrink-0 rounded p-0.5 text-muted-foreground hover:text-destructive"
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteConversation(conv.id)
                        // Delete persisted chat file
                        const proj = useWikiStore.getState().project
                        if (proj) {
                          deleteFile(`${proj.path}/.llm-wiki/chats/${conv.id}.json`).catch(() => {})
                        }
                      }}
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  )}
                </div>
                <div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-muted-foreground">
                  <span>{formatDate(conv.updatedAt)}</span>
                  {msgCount > 0 && (
                    <>
                      <span>·</span>
                      <span>{msgCount} {t("chat.msgCount")}</span>
                    </>
                  )}
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

function backendReferenceToMessageReference(ref: BackendAgentReference): MessageReference {
  const isWiki = ref.kind === "wiki" || ref.path.startsWith("wiki/")
  const isWeb = ref.kind === "web" || /^https?:\/\//i.test(ref.path)
  const source =
    ref.kind === "anytxt" ? "AnyTXT"
      : ref.kind === "web" ? "Web"
        : ref.kind === "source" ? "Source"
          : ref.kind === "graph" ? "Graph"
            : undefined
  return {
    title: ref.title,
    path: ref.path,
    kind: isWiki ? "wiki" : "external",
    source,
    url: isWeb ? ref.path : undefined,
    snippet: ref.snippet,
  }
}

function backendToolToAgentStep(event: BackendAgentToolEvent, index: number) {
  if (event.tool === "agent.plan_tools") {
    return {
      id: `backend-${index}-${event.tool}-${event.status}`,
      type: "routing" as const,
      message: event.detail ?? event.tool,
      status: event.status === "failed" ? "error" as const : "success" as const,
    }
  }
  if (event.tool === "llm.generate") {
    return {
      id: `backend-${index}-${event.tool}-${event.status}`,
      type: "final" as const,
      message: event.detail ?? event.tool,
      status: event.status === "failed" ? "error" as const
        : event.status === "started" ? "running" as const
          : "success" as const,
    }
  }
  const tool = normalizeBackendToolName(event.tool)
  return {
    id: `backend-${index}-${event.tool}-${event.status}`,
    type: event.status === "started" ? "tool_call" as const : "tool_result" as const,
    tool,
    message: event.detail ?? event.tool,
    status: event.status === "failed" ? "error" as const
      : event.status === "available" ? "skipped" as const
        : event.status === "started" ? "running" as const
          : "success" as const,
  }
}

function normalizeBackendToolName(tool: string) {
  const normalized = tool.split(".").join("_")
  if (normalized === "wiki_search") return "wiki_search" as const
  if (normalized === "wiki_read_page") return "project_file_read" as const
  if (normalized === "source_search") return "project_file_read" as const
  if (normalized === "graph_search") return "graph_search" as const
  if (normalized === "web_search") return "web_search" as const
  if (normalized === "anytxt_search") return "anytxt_search" as const
  if (normalized === "deep_research_run") return "project_file_read" as const
  return "unknown_tool" as const
}

function backendToolToAgentEvent(event: BackendAgentToolEvent): ChatAgentEvent {
  if (event.tool === "agent.plan_tools") {
    return {
      stage: "routing",
      message: event.detail ?? event.tool,
      status: event.status === "failed" ? "error" : "success",
    }
  }
  if (event.tool === "llm.generate") {
    return {
      stage: "writing",
      message: event.detail ?? event.tool,
      status: event.status === "failed" ? "error"
        : event.status === "started" ? "running"
          : "success",
    }
  }
  const tool = normalizeBackendToolName(event.tool)
  const stage =
    tool === "web_search" ? "searching_web"
      : tool === "anytxt_search" ? "searching_anytxt"
        : tool === "graph_search" ? "searching_graph"
          : tool === "project_file_read" ? "reading_context"
            : tool === "wiki_search" ? "searching_wiki"
              : event.status === "started" ? "tool_call"
                : "tool_result"
  return {
    stage,
    tool,
    message: event.detail ?? event.tool,
    status: event.status === "failed" ? "error"
      : event.status === "started" ? "running"
        : event.status === "available" ? "skipped"
          : "success",
  }
}

function backendResponseText(response: BackendAgentResponse): string {
  if (typeof response.message === "string") return response.message
  return response.message?.content ?? ""
}

export function ChatPanel() {
  const { t } = useTranslation()
  useSourceFiles() // Keep source file cache warm
  const activeConversationId = useChatStore((s) => s.activeConversationId)
  const isStreaming = useChatStore((s) => s.isStreaming)
  const streamingContent = useChatStore((s) => s.streamingContent)
  const mode = useChatStore((s) => s.mode)
  const addMessage = useChatStore((s) => s.addMessage)
  const setStreaming = useChatStore((s) => s.setStreaming)
  const appendStreamToken = useChatStore((s) => s.appendStreamToken)
  const finalizeStream = useChatStore((s) => s.finalizeStream)
  const createConversation = useChatStore((s) => s.createConversation)
  const removeLastAssistantMessage = useChatStore((s) => s.removeLastAssistantMessage)
  const maxHistoryMessages = useChatStore((s) => s.maxHistoryMessages)
  const useWebSearch = useChatStore((s) => s.useWebSearch)
  const useAnyTxtSearch = useChatStore((s) => s.useAnyTxtSearch)
  const agentMode = useChatStore((s) => s.agentMode)
  const setUseWebSearch = useChatStore((s) => s.setUseWebSearch)
  const setUseAnyTxtSearch = useChatStore((s) => s.setUseAnyTxtSearch)
  const setAgentMode = useChatStore((s) => s.setAgentMode)

  // Derive active messages via selector to re-render on message changes
  const allMessages = useChatStore((s) => s.messages)
  const activeMessages = activeConversationId
    ? allMessages.filter((m) => m.conversationId === activeConversationId)
    : []

  const project = useWikiStore((s) => s.project)
  const llmConfig = useWikiStore((s) => s.llmConfig)
  const searchApiConfig = useWikiStore((s) => s.searchApiConfig)
  const anyTxtAvailable = hasConfiguredAnyTxt(searchApiConfig.anyTxt)
  const imageInputAvailable = supportsImageInput(llmConfig)

  const abortRef = useRef<AbortController | null>(null)
  const activeRunSessionIdRef = useRef<string | null>(null)
  const activeRunIdRef = useRef<string | null>(null)
  const runIdRef = useRef(0)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const [agentEvents, setAgentEvents] = useState<ChatAgentEvent[]>([])
  const [referencePreview, setReferencePreview] = useState<ChatReferencePreview | null>(null)
  const [referencePreviewWidth, setReferencePreviewWidth] = useState(420)
  const lastMessage = activeMessages[activeMessages.length - 1]
  const scrollKey = [
    activeConversationId ?? "",
    activeMessages.length,
    lastMessage?.id ?? "",
    lastMessage?.content.length ?? 0,
    isStreaming ? streamingContent.length : 0,
  ].join(":")

  // Auto-scroll to bottom when messages change or streaming content updates
  useEffect(() => {
    const container = scrollContainerRef.current
    if (container) {
      container.scrollTop = container.scrollHeight
    }
  }, [scrollKey])

  const handleSend = useCallback(
    async (
      text: string,
      images: MessageImage[] = [],
      options?: ChatSendOptions,
    ) => {
      const sendOptions = options ?? {
        useWebSearch: useChatStore.getState().useWebSearch,
        useAnyTxtSearch: useChatStore.getState().useAnyTxtSearch,
        agentMode: useChatStore.getState().agentMode,
      }
      // Auto-create a conversation if none is active
      let convId = useChatStore.getState().activeConversationId
      if (!convId) {
        convId = createConversation()
      }

      addMessage("user", text, images)
      setStreaming(true)
      setAgentEvents([])
      let finalized = false
      const runId = ++runIdRef.current
      const backendRunId = `ui-${Date.now()}-${runId}`

      try {
        const controller = new AbortController()
        abortRef.current = controller
        activeRunSessionIdRef.current = convId
        activeRunIdRef.current = backendRunId
        const isCurrentRun = () => runIdRef.current === runId && !controller.signal.aborted

        const useBackendAgent =
          llmConfig.provider !== "claude-code" &&
          llmConfig.provider !== "codex-cli"

        if (useBackendAgent) {
          setAgentEvents([
            {
              stage: "routing",
              status: "running",
              message: t("chat.agent.routing"),
            },
          ])
          const activeConvMessages = useChatStore.getState().getActiveMessages()
            .filter((m) => m.role === "user" || m.role === "assistant")
            .slice(0, -1)
            .slice(-maxHistoryMessages)
            .map((m) => ({ role: m.role, content: m.content }))
          let accumulated = ""
          const references: MessageReference[] = []
          const backendEvents: BackendAgentToolEvent[] = []
          const seenRefs = new Set<string>()
          let streamFinished = false
          let streamUnlisten: (() => void) | null = null
          let resolveStream: (() => void) | null = null
          let rejectStream: ((err: Error) => void) | null = null
          const streamDone = new Promise<void>((resolve, reject) => {
            resolveStream = resolve
            rejectStream = reject
          })
          void streamDone.catch(() => {})
          const timeout = window.setTimeout(() => {
            if (!streamFinished) {
              streamFinished = true
              rejectStream?.(new Error("Agent stream timed out"))
            }
          }, 240_000)
          streamUnlisten = await listen<BackendAgentEventPayload>("agent-event", (event) => {
            const payload = event.payload
            if (payload.sessionId !== convId || payload.runId !== backendRunId || !isCurrentRun()) return
            const agentEvent = payload.event
            if (agentEvent.type === "done") {
              if (!streamFinished) {
                streamFinished = true
                window.clearTimeout(timeout)
                resolveStream?.()
              }
              return
            }
            if (agentEvent.type === "messageDelta" && agentEvent.text) {
              accumulated += agentEvent.text
              appendStreamToken(agentEvent.text)
              return
            }
            if (agentEvent.type === "referenceAdded" && agentEvent.reference) {
              const ref = backendReferenceToMessageReference(agentEvent.reference)
              const key = `${ref.kind ?? "wiki"}:${ref.url ?? ref.path}`.toLowerCase()
              if (!seenRefs.has(key)) {
                seenRefs.add(key)
                references.push(ref)
              }
              return
            }
            if (agentEvent.type === "toolStart" && agentEvent.tool) {
              const toolEvent: BackendAgentToolEvent = {
                tool: agentEvent.tool,
                status: "started",
                detail: agentEvent.input,
              }
              backendEvents.push(toolEvent)
              setAgentEvents((prev) => [...prev, backendToolToAgentEvent(toolEvent)].slice(-6))
              return
            }
            if (agentEvent.type === "toolEnd" && agentEvent.tool) {
              const failed = typeof agentEvent.output === "string" && agentEvent.output.startsWith("failed:")
              const toolEvent: BackendAgentToolEvent = {
                tool: agentEvent.tool,
                status: failed ? "failed" : "completed",
                detail: agentEvent.output,
              }
              backendEvents.push(toolEvent)
              setAgentEvents((prev) => [...prev, backendToolToAgentEvent(toolEvent)].slice(-6))
              return
            }
            if (agentEvent.type === "error" && agentEvent.message) {
              const toolEvent: BackendAgentToolEvent = {
                tool: "agent",
                status: "failed",
                detail: agentEvent.message,
              }
              backendEvents.push(toolEvent)
              setAgentEvents((prev) => [...prev, backendToolToAgentEvent(toolEvent)].slice(-6))
              if (!streamFinished) {
                streamFinished = true
                window.clearTimeout(timeout)
                rejectStream?.(new Error(agentEvent.message))
              }
            }
          })
          try {
            await invoke<string>("agent_start_turn_stream", {
              projectId: project?.id ?? "current",
              request: {
                message: text,
                sessionId: convId,
                runId: backendRunId,
                mode: sendOptions.agentMode,
                stream: true,
                tools: {
                  wiki: true,
                  web: sendOptions.useWebSearch,
                  anytxt: sendOptions.useAnyTxtSearch,
                },
                topK: sendOptions.agentMode === "deep" ? 8 : 5,
                includeContent: sendOptions.agentMode === "deep",
                history: activeConvMessages,
                images: images.map((image) => ({
                  mediaType: image.mediaType,
                  dataBase64: image.dataBase64,
                })),
              },
            })
            await streamDone
          } finally {
            window.clearTimeout(timeout)
            streamUnlisten?.()
          }
          if (!isCurrentRun()) return
          lastQueryPages = references
            .filter((ref) => ref.kind === "wiki")
            .map((ref) => ({ title: ref.title, path: ref.path }))
          const steps = backendEvents.map(backendToolToAgentStep)
          finalized = true
          finalizeStream(accumulated, references, steps)
          setAgentEvents([])
          abortRef.current = null
          activeRunSessionIdRef.current = null
          activeRunIdRef.current = null
          return
        }

        const activeConvMessages = useChatStore.getState().getActiveMessages()
          .filter((m) => m.role === "user" || m.role === "assistant")
          .slice(-maxHistoryMessages)
        const priorMessages = activeConvMessages.slice(0, -1)
        const backendResponse = await invoke<BackendAgentResponse>("agent_start_turn", {
          projectId: project?.id ?? "current",
          request: {
            message: text,
            sessionId: convId,
            runId: backendRunId,
            persistSession: false,
            mode: sendOptions.agentMode,
            tools: {
              wiki: true,
              web: sendOptions.useWebSearch,
              anytxt: sendOptions.useAnyTxtSearch,
            },
            topK: sendOptions.agentMode === "deep" ? 8 : 5,
            includeContent: sendOptions.agentMode === "deep",
            history: chatMessagesToLLM(priorMessages).map((m) => ({
              role: m.role,
              content: typeof m.content === "string"
                ? m.content
                : m.content
                    .filter((block) => block.type === "text")
                    .map((block) => block.text)
                    .join("\n"),
            })),
            images: images.map((image) => ({
              mediaType: image.mediaType,
              dataBase64: image.dataBase64,
            })),
          },
        })
        if (!isCurrentRun()) return

        const backendReferences = (backendResponse.references ?? []).map(backendReferenceToMessageReference)
        const backendSteps = (backendResponse.toolEvents ?? []).map(backendToolToAgentStep)
        const backendEvents = (backendResponse.toolEvents ?? []).map(backendToolToAgentEvent)
        setAgentEvents(backendEvents.slice(-6))
        lastQueryPages = backendReferences
          .filter((ref) => ref.kind === "wiki")
          .map((ref) => ({ title: ref.title, path: ref.path }))

        const contextText = [
          "You have access to the current LLM Wiki project context below. Use it as retrieved evidence when it is relevant.",
          "",
          backendResponseText(backendResponse),
          "",
          `User request: ${text}`,
        ].join("\n")
        const userContent: string | ContentBlock[] = images.length > 0
          ? [
              { type: "text", text: contextText },
              ...images.map((image) => ({
                type: "image" as const,
                mediaType: image.mediaType,
                dataBase64: image.dataBase64,
              })),
            ]
          : contextText
        const finalMessages: LlmChatMessage[] = [
          {
            role: "system",
            content: "Answer using the provided LLM Wiki context and references. If the context is insufficient, say what is missing instead of inventing details.",
          },
          ...chatMessagesToLLM(priorMessages),
          { role: "user", content: userContent },
        ]

        let accumulated = ""
        let thinkingOpen = false

        const appendReasoning = (token: string) => {
          if (!token) return
          if (!thinkingOpen) {
            thinkingOpen = true
            accumulated += "<think>"
            appendStreamToken("<think>")
          }
          accumulated += token
          appendStreamToken(token)
        }

        const closeReasoning = () => {
          if (!thinkingOpen) return
          thinkingOpen = false
          accumulated += "</think>"
          appendStreamToken("</think>")
        }

        const streamFinalAnswer = async (reasoningOff: boolean) => {
          let streamError: Error | null = null
          await streamChat(
            llmConfig,
            finalMessages,
            {
              onToken: (token) => {
                if (!isCurrentRun()) return
                closeReasoning()
                accumulated += token
                appendStreamToken(token)
              },
              onReasoningToken: (token) => {
                if (!isCurrentRun()) return
                if (reasoningOff) return
                appendReasoning(token)
              },
              onDone: () => {},
              onError: (err) => {
                streamError = err
              },
            },
            controller.signal,
            reasoningOff ? { reasoning: { mode: "off" } } : undefined,
          )
          if (streamError) throw streamError
        }

        try {
          await streamFinalAnswer(false)
        } catch (err) {
          if (!isCurrentRun()) return
          if (isReasoningOnlyResponseError(err)) {
            accumulated = ""
            thinkingOpen = false
            useChatStore.setState({ streamingContent: "" })
            await streamFinalAnswer(true)
          } else {
            throw err
          }
        }

        if (!isCurrentRun()) return
        closeReasoning()
        finalized = true
        finalizeStream(accumulated, backendReferences, backendSteps)
        setAgentEvents([])
        abortRef.current = null
        activeRunSessionIdRef.current = null
        activeRunIdRef.current = null
        // save-worthy detection removed — user has direct "Save to Wiki" button on each message
      } catch (err) {
        if (!finalized) {
          if (isAbortLikeError(err) || runIdRef.current !== runId) {
            setStreaming(false)
            setAgentEvents([])
            abortRef.current = null
            activeRunSessionIdRef.current = null
            activeRunIdRef.current = null
            return
          }
          const message = err instanceof Error ? err.message : String(err)
          finalizeStream(`Error: ${message}`, undefined)
          setAgentEvents([])
        }
        abortRef.current = null
        activeRunSessionIdRef.current = null
        activeRunIdRef.current = null
      }
    },
    [project, llmConfig, searchApiConfig, addMessage, setStreaming, appendStreamToken, finalizeStream, createConversation, maxHistoryMessages, t],
  )

  const handleStop = useCallback(() => {
    runIdRef.current += 1
    const sessionId = activeRunSessionIdRef.current
    const backendRunId = activeRunIdRef.current
    if (sessionId) {
      void invoke("agent_cancel_turn", {
        projectId: project?.id ?? "current",
        sessionId,
        runId: backendRunId ?? undefined,
      }).catch(() => {})
    }
    abortRef.current?.abort()
    abortRef.current = null
    activeRunSessionIdRef.current = null
    activeRunIdRef.current = null
    setStreaming(false)
    setAgentEvents([])
  }, [project, setStreaming])

  const handleRegenerate = useCallback(async () => {
    if (isStreaming) return
    // Find the last user message in active conversation
    const active = useChatStore.getState().getActiveMessages()
    const lastUserMsg = [...active].reverse().find((m) => m.role === "user")
    if (!lastUserMsg) return
    // Remove the last assistant reply, then re-send
    removeLastAssistantMessage()
    // Small delay to let state update
    await new Promise((r) => setTimeout(r, 50))
    // Trigger send with the same text (handleSend will add a new user message,
    // so also remove the original to avoid duplication)
    // Actually: just call handleSend — but it adds a user message. To avoid dupe,
    // we remove the last user message too and let handleSend re-add it.
    const store = useChatStore.getState()
    const updatedActive = store.getActiveMessages()
    const lastUser = [...updatedActive].reverse().find((m) => m.role === "user")
    if (lastUser) {
      useChatStore.setState((s) => ({
        messages: s.messages.filter((m) => m.id !== lastUser.id),
      }))
    }
    // Re-send with the original text AND images so a regenerated turn
    // keeps the same vision context.
    handleSend(lastUserMsg.content, lastUserMsg.images ?? [])
  }, [isStreaming, removeLastAssistantMessage, handleSend])

  const handleWriteToWiki = useCallback(async () => {
    if (!project) return
    const pp = normalizePath(project.path)
    try {
      await executeIngestWrites(pp, llmConfig, undefined, undefined)
      await refreshProjectFileTree(pp, { bumpDataVersion: true })
    } catch (err) {
      console.error("Failed to write to wiki:", err)
    }
  }, [project, llmConfig])

  const hasAssistantMessages = activeMessages.some((m) => m.role === "assistant")
  const showWriteButton = mode === "ingest" && !isStreaming && hasAssistantMessages

  return (
    <div className="flex h-full flex-row overflow-hidden">
      <ConversationSidebar />

      <div className="flex flex-1 flex-col overflow-hidden">
        {!activeConversationId ? (
          <div className="flex flex-1 items-center justify-center text-muted-foreground">
            <div className="text-center">
              <MessageSquare className="mx-auto mb-3 h-8 w-8 opacity-30" />
              <p className="text-sm">{t("chat.startNewConversation")}</p>
              <p className="mt-1 text-xs opacity-60">{t("chat.clickNewChatToBegin")}</p>
            </div>
          </div>
        ) : (
          <>
            <div
              ref={scrollContainerRef}
              className="flex-1 overflow-y-auto px-3 py-2"
            >
              <div className="flex flex-col gap-3">
                {activeMessages.map((msg, idx) => {
                  // Check if this is the last assistant message
                  const isLastAssistant = msg.role === "assistant" &&
                    !activeMessages.slice(idx + 1).some((m) => m.role === "assistant")
                  return (
                    <ChatMessage
                      key={msg.id}
                      message={msg}
                      isLastAssistant={isLastAssistant && !isStreaming}
                      onRegenerate={isLastAssistant ? handleRegenerate : undefined}
                      onOpenReferencePreview={setReferencePreview}
                    />
                  )
                })}
                {isStreaming && <StreamingMessage content={streamingContent} agentEvents={agentEvents} />}
                <div ref={bottomRef} />
              </div>
            </div>

            {showWriteButton && (
              <div className="border-t px-3 py-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleWriteToWiki}
                  className="w-full gap-2"
                >
                  <BookOpen className="h-4 w-4" />
                  {t("chat.writeToWiki")}
                </Button>
              </div>
            )}
          </>
        )}

        <ChatInput
          onSend={handleSend}
          onStop={handleStop}
          isStreaming={isStreaming}
          useWebSearch={useWebSearch}
          useAnyTxtSearch={useAnyTxtSearch}
          agentMode={agentMode}
          onUseWebSearchChange={setUseWebSearch}
          onUseAnyTxtSearchChange={setUseAnyTxtSearch}
          onAgentModeChange={setAgentMode}
          anyTxtAvailable={anyTxtAvailable}
          imageInputAvailable={imageInputAvailable}
          placeholder={
            mode === "ingest"
              ? t("chat.ingestPlaceholder")
              : t("chat.typeAMessage")
          }
        />
      </div>

      {referencePreview && (
        <ChatReferencePreviewPanel
          preview={referencePreview}
          width={referencePreviewWidth}
          onResize={setReferencePreviewWidth}
          onClose={() => setReferencePreview(null)}
        />
      )}
    </div>
  )
}

function ChatReferencePreviewPanel({
  preview,
  width,
  onResize,
  onClose,
}: {
  preview: ChatReferencePreview
  width: number
  onResize: (width: number) => void
  onClose: () => void
}) {
  const { t } = useTranslation()
  const displayTitle = preview.title || getFileName(preview.path)
  const dragStartRef = useRef<{ x: number; width: number } | null>(null)

  const startResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault()
    dragStartRef.current = { x: event.clientX, width }
    event.currentTarget.setPointerCapture(event.pointerId)
  }, [width])

  const handleResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStartRef.current) return
    const delta = dragStartRef.current.x - event.clientX
    onResize(clampReferencePreviewWidth(dragStartRef.current.width + delta))
  }, [onResize])

  const stopResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    dragStartRef.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }, [])

  return (
    <aside
      className="relative flex h-full min-w-[320px] max-w-[56%] shrink-0 flex-col border-l bg-background"
      style={{ width }}
    >
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label={t("chat.resizeReferencePreview")}
        tabIndex={0}
        onPointerDown={startResize}
        onPointerMove={handleResize}
        onPointerUp={stopResize}
        onPointerCancel={stopResize}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") {
            event.preventDefault()
            onResize(clampReferencePreviewWidth(width + 32))
          } else if (event.key === "ArrowRight") {
            event.preventDefault()
            onResize(clampReferencePreviewWidth(width - 32))
          }
        }}
        className="absolute -left-1 top-0 z-10 h-full w-2 cursor-col-resize outline-none transition-colors hover:bg-primary/15 focus-visible:bg-primary/20"
      />
      <div className="flex min-h-10 items-center gap-2 border-b px-3 py-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-xs font-medium" title={displayTitle}>
            {displayTitle}
          </div>
          <div className="mt-0.5 truncate text-[10px] text-muted-foreground" title={preview.path}>
            {preview.source ?? t("chat.referencePreview")} · {preview.path}
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="shrink-0 rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
          title={t("chat.closeReferencePreview")}
          aria-label={t("chat.closeReferencePreview")}
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {preview.external ? (
          <ExternalReferencePreview preview={preview} />
        ) : getFileCategory(preview.path) === "markdown" ? (
          <ChatMarkdownReferencePreview preview={preview} />
        ) : (
          <FilePreview
            key={preview.path}
            filePath={preview.path}
            textContent={preview.content}
          />
        )}
      </div>
    </aside>
  )
}

function clampReferencePreviewWidth(width: number): number {
  return Math.min(760, Math.max(320, Math.round(width)))
}

function ChatMarkdownReferencePreview({ preview }: { preview: ChatReferencePreview }) {
  const { frontmatter, body } = parseFrontmatter(preview.content)
  return (
    <div className="h-full overflow-auto px-6 py-6">
      {frontmatter && <FrontmatterPanel data={frontmatter} />}
      <WikiReader body={body} filePath={preview.path} />
    </div>
  )
}

function ExternalReferencePreview({ preview }: { preview: ChatReferencePreview }) {
  const { t } = useTranslation()
  return (
    <div className="flex h-full flex-col overflow-auto p-5">
      <div className="mb-4 space-y-2">
        <div className="flex items-center gap-2">
          {preview.source && (
            <span className="rounded border border-border/60 bg-muted px-1.5 py-0.5 text-[10px] font-medium uppercase text-muted-foreground">
              {preview.source}
            </span>
          )}
          <h3 className="truncate text-sm font-medium" title={preview.title}>{preview.title}</h3>
        </div>
        <div className="break-all rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
          {preview.path.replace(/^[a-z]+-preview:\/\//, "")}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto rounded-lg border border-border/60 bg-muted/20 p-4">
        <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-6">
          {preview.snippet?.trim() || t("chat.noReferencePreviewFragment")}
        </pre>
      </div>
    </div>
  )
}

function isAbortLikeError(err: unknown): boolean {
  if (err instanceof DOMException && err.name === "AbortError") return true
  if (!(err instanceof Error)) return false
  return err.name === "AbortError" || /abort|cancel/i.test(err.message)
}
