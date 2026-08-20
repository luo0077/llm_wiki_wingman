import { beforeEach, describe, expect, it } from "vitest"
import { useResearchStore } from "./research-store"

beforeEach(() => {
  useResearchStore.setState({ tasks: [], panelOpen: false, maxConcurrent: 3 })
})

describe("research store batch queue", () => {
  it("adds a large batch in one state update with review metadata intact", () => {
    let updates = 0
    const unsubscribe = useResearchStore.subscribe(() => { updates += 1 })

    const ids = useResearchStore.getState().addTasks(Array.from({ length: 100 }, (_, index) => ({
      topic: `Topic ${index}`,
      searchQueries: [`query ${index}`],
      sourceReviewId: `review-${index}`,
      rerunOfTaskId: index === 99 ? "research-original" : undefined,
    })))
    unsubscribe()

    expect(ids).toHaveLength(100)
    expect(updates).toBe(1)
    expect(useResearchStore.getState().tasks[99]).toMatchObject({
      topic: "Topic 99",
      searchQueries: ["query 99"],
      sourceReviewId: "review-99",
      rerunOfTaskId: "research-original",
      status: "queued",
    })
    expect(useResearchStore.getState().panelOpen).toBe(true)
  })

  it("does not update state for an empty batch", () => {
    expect(useResearchStore.getState().addTasks([])).toEqual([])
    expect(useResearchStore.getState().tasks).toEqual([])
  })
})
