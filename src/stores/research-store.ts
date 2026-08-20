import { create } from "zustand"
import type { WebSearchResult } from "@/lib/web-search"

export interface ResearchTask {
  id: string
  topic: string
  /** Review item that requested this research. It is resolved only after the
   * generated page has been written successfully. */
  sourceReviewId?: string
  searchQueries?: string[]
  status: "queued" | "searching" | "synthesizing" | "saving" | "done" | "error"
  webResults: WebSearchResult[]
  synthesis: string
  savedPath: string | null
  error: string | null
  createdAt: number
}

interface ResearchState {
  tasks: ResearchTask[]
  panelOpen: boolean
  maxConcurrent: number

  addTask: (topic: string) => string
  addTasks: (inputs: Array<Pick<ResearchTask, "topic" | "searchQueries" | "sourceReviewId">>) => string[]
  updateTask: (id: string, updates: Partial<ResearchTask>) => void
  removeTask: (id: string) => void
  setPanelOpen: (open: boolean) => void
  getRunningCount: () => number
  getNextQueued: () => ResearchTask | undefined
}

let counter = 0

function createResearchTask(
  input: Pick<ResearchTask, "topic" | "searchQueries" | "sourceReviewId">,
): ResearchTask {
  return {
    id: `research-${++counter}`,
    topic: input.topic,
    ...(input.searchQueries?.length ? { searchQueries: input.searchQueries } : {}),
    ...(input.sourceReviewId ? { sourceReviewId: input.sourceReviewId } : {}),
    status: "queued",
    webResults: [],
    synthesis: "",
    savedPath: null,
    error: null,
    createdAt: Date.now(),
  }
}

export const useResearchStore = create<ResearchState>((set, get) => ({
  tasks: [],
  panelOpen: false,
  maxConcurrent: 3,

  addTask: (topic) => {
    const task = createResearchTask({ topic })
    set((state) => ({
      tasks: [...state.tasks, task],
      panelOpen: true,
    }))
    return task.id
  },

  addTasks: (inputs) => {
    const tasks = inputs.map(createResearchTask)
    if (tasks.length === 0) return []
    set((state) => ({ tasks: [...state.tasks, ...tasks], panelOpen: true }))
    return tasks.map((task) => task.id)
  },

  updateTask: (id, updates) =>
    set((state) => ({
      tasks: state.tasks.map((t) => (t.id === id ? { ...t, ...updates } : t)),
    })),

  removeTask: (id) =>
    set((state) => ({
      tasks: state.tasks.filter((t) => t.id !== id),
    })),

  setPanelOpen: (panelOpen) => set({ panelOpen }),

  getRunningCount: () => {
    const { tasks } = get()
    return tasks.filter((t) =>
      t.status === "searching" || t.status === "synthesizing" || t.status === "saving"
    ).length
  },

  getNextQueued: () => {
    const { tasks } = get()
    return tasks.find((t) => t.status === "queued")
  },
}))
