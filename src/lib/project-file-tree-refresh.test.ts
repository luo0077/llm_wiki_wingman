import { beforeEach, describe, expect, it, vi } from "vitest"
import { refreshProjectFileTree } from "@/lib/project-file-tree-refresh"
import { useWikiStore } from "@/stores/wiki-store"
import type { FileNode, WikiProject } from "@/types/wiki"

const mocks = vi.hoisted(() => ({
  listDirectory: vi.fn(),
}))

vi.mock("@/commands/fs", () => ({
  listDirectory: mocks.listDirectory,
}))

const project: WikiProject = {
  id: "project-1",
  name: "Project",
  path: "/tmp/project",
}

const shallowTree: FileNode[] = [
  {
    name: "wiki",
    path: "/tmp/project/wiki",
    is_dir: true,
    children: [],
  },
]

const fullTree: FileNode[] = [
  {
    name: "wiki",
    path: "/tmp/project/wiki",
    is_dir: true,
    children: [
      {
        name: "entities",
        path: "/tmp/project/wiki/entities",
        is_dir: true,
        children: [
          {
            name: "alpha.md",
            path: "/tmp/project/wiki/entities/alpha.md",
            is_dir: false,
          },
        ],
      },
    ],
  },
]

async function flushMicrotasks() {
  await Promise.resolve()
  await Promise.resolve()
}

describe("refreshProjectFileTree", () => {
  beforeEach(() => {
    mocks.listDirectory.mockReset()
    useWikiStore.setState({
      project,
      fileTree: [],
      projectPathIndex: { byPath: new Map(), filesByName: new Map() },
      dataVersion: 0,
    })
  })

  it("updates the visible tree shallowly and refreshes the full resolver index in the background", async () => {
    mocks.listDirectory.mockImplementation(async (_path: string, options?: { maxDepth?: number }) =>
      options?.maxDepth === 2 ? shallowTree : fullTree
    )

    await refreshProjectFileTree(project.path, {
      projectId: project.id,
      bumpDataVersion: true,
    })
    await flushMicrotasks()

    expect(mocks.listDirectory).toHaveBeenNthCalledWith(1, project.path, { maxDepth: 2 })
    expect(mocks.listDirectory).toHaveBeenNthCalledWith(2, project.path, undefined)
    expect(useWikiStore.getState().fileTree).toEqual(shallowTree)
    expect(useWikiStore.getState().projectPathIndex.byPath.has("/tmp/project/wiki/entities/alpha.md")).toBe(true)
    expect(useWikiStore.getState().dataVersion).toBe(1)
  })

  it("does not write stale results after the active project changes", async () => {
    mocks.listDirectory.mockImplementation(async (_path: string, options?: { maxDepth?: number }) =>
      options?.maxDepth === 2 ? shallowTree : fullTree
    )
    useWikiStore.setState({
      project: { ...project, id: "other-project", path: "/tmp/other" },
    })

    await refreshProjectFileTree(project.path, {
      projectId: project.id,
      clearDisplayTreeFirst: true,
      bumpDataVersion: true,
    })
    await flushMicrotasks()

    expect(useWikiStore.getState().fileTree).toEqual([])
    expect(useWikiStore.getState().projectPathIndex.byPath.size).toBe(0)
    expect(useWikiStore.getState().dataVersion).toBe(0)
  })
})
