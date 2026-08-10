import { beforeEach, describe, expect, it } from "vitest"
import { useAppDialog, useAppDialogStore } from "./app-dialog-store"

describe("app dialog store", () => {
  beforeEach(() => {
    useAppDialogStore.setState({ current: null, queue: [] })
  })

  it("queues dialogs and resolves them in order", async () => {
    const dialogs = useAppDialog()
    const first = dialogs.confirm({ message: "first" })
    const second = dialogs.confirm({ message: "second" })

    expect(useAppDialogStore.getState().current?.message).toBe("first")
    expect(useAppDialogStore.getState().queue).toHaveLength(1)

    useAppDialogStore.getState().settle(true)
    await expect(first).resolves.toBe(true)
    expect(useAppDialogStore.getState().current?.message).toBe("second")

    useAppDialogStore.getState().settle(false)
    await expect(second).resolves.toBe(false)
    expect(useAppDialogStore.getState().current).toBeNull()
  })

  it("resolves alerts after dismissal", async () => {
    const pending = useAppDialog().alert({ message: "notice" })
    useAppDialogStore.getState().settle(false)
    await expect(pending).resolves.toBeUndefined()
  })
})

