import { describe, expect, it } from "vitest"
import { mergeAgentFileChange, summarizeAgentFileChange } from "@/lib/agent-file-activity"

describe("agent file activity", () => {
  it("summarizes a created file", () => {
    const change = summarizeAgentFileChange({
      id: "run:file",
      path: "/project/agent-workspace/file.md",
      tool: "workspace.write_file",
      beforeContent: null,
      afterContent: "one\ntwo",
      timestamp: 1,
    })
    expect(change.operation).toBe("created")
    expect(change.additions).toBe(2)
    expect(change.deletions).toBe(0)
    expect(change.diff).toContain("+one")
  })

  it("keeps unchanged prefix and suffix outside the displayed hunk", () => {
    const change = summarizeAgentFileChange({
      id: "run:file",
      path: "/project/wiki/page.md",
      tool: "wiki.write_page",
      beforeContent: "head\nold\ntail",
      afterContent: "head\nnew\ntail",
    })
    expect(change.additions).toBe(1)
    expect(change.deletions).toBe(1)
    expect(change.diff).toContain("-old\n+new")
    expect(change.diff).not.toContain(" head")
  })

  it("preserves the first rollback snapshot when writes are coalesced", () => {
    const first = summarizeAgentFileChange({
      id: "first",
      path: "/project/file",
      tool: "workspace.write_file",
      beforeContent: "original",
      afterContent: "middle",
    })
    const second = summarizeAgentFileChange({
      id: "second",
      path: "/project/file",
      tool: "workspace.append_file",
      beforeContent: "middle",
      afterContent: "middle\nend",
    })
    const merged = mergeAgentFileChange(first, second)
    expect(merged.id).toBe("first")
    expect(merged.beforeContent).toBe("original")
    expect(merged.afterContent).toBe("middle\nend")
  })
})
