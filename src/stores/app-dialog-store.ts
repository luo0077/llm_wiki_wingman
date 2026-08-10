import { create } from "zustand"

export type AppDialogVariant = "default" | "destructive"

export interface AppDialogOptions {
  title?: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  variant?: AppDialogVariant
}

interface DialogRequest extends AppDialogOptions {
  id: number
  kind: "alert" | "confirm"
  resolve: (value: boolean) => void
}

interface AppDialogState {
  current: DialogRequest | null
  queue: DialogRequest[]
  enqueue: (request: DialogRequest) => void
  settle: (value: boolean) => void
}

export const useAppDialogStore = create<AppDialogState>((set, get) => ({
  current: null,
  queue: [],
  enqueue: (request) => {
    if (!get().current) {
      set({ current: request })
      return
    }
    set((state) => ({ queue: [...state.queue, request] }))
  },
  settle: (value) => {
    const { current, queue } = get()
    if (!current) return
    const [next, ...rest] = queue
    set({ current: next ?? null, queue: rest })
    current.resolve(value)
  },
}))

let nextDialogId = 1

function enqueueDialog(
  kind: DialogRequest["kind"],
  options: AppDialogOptions,
): Promise<boolean> {
  return new Promise((resolve) => {
    useAppDialogStore.getState().enqueue({
      ...options,
      id: nextDialogId++,
      kind,
      resolve,
    })
  })
}

const appDialogApi = {
  alert: async (options: AppDialogOptions): Promise<void> => {
    await enqueueDialog("alert", options)
  },
  confirm: (options: AppDialogOptions): Promise<boolean> =>
    enqueueDialog("confirm", options),
}

export function useAppDialog() {
  return appDialogApi
}
