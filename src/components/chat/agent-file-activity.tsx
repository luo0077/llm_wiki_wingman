import { useState } from "react"
import { ChevronDown, ChevronRight, FilePlus2, FileText, RotateCcw } from "lucide-react"
import { useTranslation } from "react-i18next"
import { deleteFile, readFile, writeFile } from "@/commands/fs"
import type { ChatAgentFileChange } from "@/lib/chat-agent-types"
import { getFileName } from "@/lib/path-utils"
import { refreshProjectFileTree } from "@/lib/project-file-tree-refresh"
import { useWikiStore } from "@/stores/wiki-store"

export function AgentFileActivity({ changes }: { changes: ChatAgentFileChange[] }) {
  const { t } = useTranslation()
  const project = useWikiStore((state) => state.project)
  const openFileInPreview = useWikiStore((state) => state.openFileInPreview)
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [undoing, setUndoing] = useState<string | null>(null)
  const [undone, setUndone] = useState<Set<string>>(() => new Set())
  const [error, setError] = useState<string | null>(null)

  if (changes.length === 0) return null

  const undo = async (change: ChatAgentFileChange) => {
    if (!project || change.beforeContent === undefined || change.afterContent === undefined || undone.has(change.id)) return
    setUndoing(change.id)
    setError(null)
    try {
      const currentContent = await readFile(change.path).catch(() => null)
      if (currentContent !== change.afterContent) {
        throw new Error(t("chat.agentChanges.undoConflict"))
      }
      if (change.operation === "created" && change.beforeContent === null) {
        await deleteFile(change.path)
      } else if (typeof change.beforeContent === "string") {
        await writeFile(change.path, change.beforeContent)
      }
      await refreshProjectFileTree(project.path, { bumpDataVersion: true })
      setUndone((current) => new Set(current).add(change.id))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setUndoing(null)
    }
  }

  return (
    <section className="rounded-md border border-border/60 bg-background/60" aria-label={t("chat.agentChanges.title") }>
      <div className="flex items-center justify-between gap-2 border-b border-border/50 px-2.5 py-1.5">
        <span className="text-xs font-medium">{t("chat.agentChanges.title")}</span>
        <span className="text-[10px] tabular-nums text-muted-foreground">
          {t("chat.agentChanges.fileCount", { count: changes.length })}
        </span>
      </div>
      <div>
        {changes.map((change) => {
          const isExpanded = Boolean(expanded[change.id])
          const canUndo = change.beforeContent !== undefined
            && change.afterContent !== undefined
            && !undone.has(change.id)
          const Icon = change.operation === "created" ? FilePlus2 : FileText
          return (
            <div key={change.id} className="border-b border-border/40 last:border-b-0">
              <div className="flex min-w-0 items-center gap-1.5 px-2 py-1.5 text-xs">
                <button
                  type="button"
                  onClick={() => setExpanded((current) => ({ ...current, [change.id]: !isExpanded }))}
                  className="rounded p-0.5 text-muted-foreground hover:bg-accent hover:text-foreground"
                  aria-expanded={isExpanded}
                  aria-label={isExpanded ? t("chat.agentChanges.collapseDiff") : t("chat.agentChanges.expandDiff")}
                >
                  {isExpanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                </button>
                <Icon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <button
                  type="button"
                  onClick={() => {
                    void readFile(change.path)
                      .catch(() => "")
                      .then((content) => openFileInPreview(change.path, content))
                  }}
                  className="min-w-0 flex-1 truncate text-left hover:underline"
                  title={change.path}
                >
                  {getFileName(change.path)}
                </button>
                <span className="shrink-0 font-mono text-[10px] text-emerald-600 dark:text-emerald-400">+{change.additions}</span>
                <span className="shrink-0 font-mono text-[10px] text-red-600 dark:text-red-400">-{change.deletions}</span>
                <span className="shrink-0 text-[10px] text-muted-foreground">{new Date(change.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
                <button
                  type="button"
                  disabled={!canUndo || undoing === change.id}
                  onClick={() => void undo(change)}
                  className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-30"
                  title={canUndo ? t("chat.agentChanges.undo") : t("chat.agentChanges.undoUnavailable")}
                  aria-label={canUndo ? t("chat.agentChanges.undo") : t("chat.agentChanges.undoUnavailable")}
                >
                  <RotateCcw className={`h-3.5 w-3.5 ${undoing === change.id ? "animate-spin" : ""}`} />
                </button>
              </div>
              {isExpanded && (
                <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-words border-t border-border/40 bg-muted/30 px-3 py-2 font-mono text-[10px] leading-4">
                  {change.diff}
                </pre>
              )}
            </div>
          )
        })}
      </div>
      {error && <div className="border-t border-destructive/20 px-2.5 py-1.5 text-[10px] text-destructive">{error}</div>}
    </section>
  )
}
