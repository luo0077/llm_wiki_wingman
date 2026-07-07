import { beforeEach, describe, expect, it } from "vitest"
import { useChatStore } from "./chat-store"

describe("chat-store conversation isolation", () => {
  beforeEach(() => {
    useChatStore.setState({
      conversations: [],
      activeConversationId: null,
      messages: [],
      isStreaming: false,
      streamingContent: "",
      mode: "chat",
      ingestSource: null,
      useWebSearch: false,
      useAnyTxtSearch: false,
      agentMode: "standard",
      selectedSkills: [],
      disabledSkills: [],
    })
  })

  it("writes async assistant results back to the original conversation", () => {
    const store = useChatStore.getState()
    const first = store.createConversation()
    store.addMessageToConversation(first, "user", "first question")

    const second = useChatStore.getState().createConversation()
    expect(useChatStore.getState().activeConversationId).toBe(second)

    useChatStore
      .getState()
      .finalizeStreamForConversation(first, "first answer")

    const state = useChatStore.getState()
    const firstMessages = state.messages.filter((message) => message.conversationId === first)
    const secondMessages = state.messages.filter((message) => message.conversationId === second)

    expect(firstMessages.map((message) => message.content)).toEqual([
      "first question",
      "first answer",
    ])
    expect(secondMessages).toEqual([])
  })

  it("clears stale stream content when a new stream starts", () => {
    useChatStore.setState({
      streamingContent: "old conversation tokens",
      isStreaming: false,
    })

    useChatStore.getState().setStreaming(true)

    expect(useChatStore.getState().streamingContent).toBe("")
    expect(useChatStore.getState().isStreaming).toBe(true)
  })

  it("clears stale stream content when creating or switching conversations", () => {
    const first = useChatStore.getState().createConversation()
    useChatStore.setState({
      streamingContent: "old conversation tokens",
      isStreaming: true,
    })

    const second = useChatStore.getState().createConversation()

    expect(useChatStore.getState().activeConversationId).toBe(second)
    expect(useChatStore.getState().streamingContent).toBe("")
    expect(useChatStore.getState().isStreaming).toBe(false)

    useChatStore.setState({
      streamingContent: "more stale tokens",
      isStreaming: true,
    })
    useChatStore.getState().setActiveConversation(first)

    expect(useChatStore.getState().activeConversationId).toBe(first)
    expect(useChatStore.getState().streamingContent).toBe("")
  })

  it("stores selected skills per conversation and starts new conversations empty", () => {
    const first = useChatStore.getState().createConversation()
    useChatStore.getState().setSelectedSkills(["cover-image"])

    const second = useChatStore.getState().createConversation()

    expect(useChatStore.getState().activeConversationId).toBe(second)
    expect(useChatStore.getState().selectedSkills).toEqual([])

    useChatStore.getState().setSelectedSkills(["ppt"])
    useChatStore.getState().setActiveConversation(first)

    expect(useChatStore.getState().selectedSkills).toEqual(["cover-image"])

    useChatStore.getState().setActiveConversation(second)

    expect(useChatStore.getState().selectedSkills).toEqual(["ppt"])
  })
})
