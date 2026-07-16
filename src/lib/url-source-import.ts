import { writeFile } from "@/commands/fs"
import type { LlmConfig, SourceWatchConfig } from "@/stores/wiki-store"
import type { WikiProject } from "@/types/wiki"
import { getHttpFetch } from "@/lib/tauri-fetch"
import { normalizeSourceWatchConfig } from "@/lib/source-watch-config"
import { enqueueSourceIngest, getUniqueDestPath } from "@/lib/source-lifecycle"
import { normalizePath } from "@/lib/path-utils"

export const MAX_BATCH_URLS = 50

export interface UrlImportResult {
  url: string
  path?: string
  error?: string
}

export function parseImportUrls(input: string): string[] {
  const unique = new Set<string>()
  for (const line of input.split(/\r?\n/)) {
    const candidate = line.trim()
    if (!candidate) continue
    let parsed: URL
    try {
      parsed = new URL(candidate)
    } catch {
      throw new Error(`Invalid URL: ${candidate}`)
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error(`Unsupported URL scheme: ${candidate}`)
    }
    if (parsed.username || parsed.password) {
      throw new Error(`URLs with embedded credentials are not allowed: ${candidate}`)
    }
    parsed.hash = ""
    unique.add(parsed.toString())
    if (unique.size > MAX_BATCH_URLS) {
      throw new Error(`A batch can contain at most ${MAX_BATCH_URLS} URLs`)
    }
  }
  return [...unique]
}

function safeSlug(value: string): string {
  const slug = value
    .normalize("NFKC")
    .replace(/[^\p{L}\p{N}._-]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 100)
  const stem = slug.split(".")[0]?.toUpperCase()
  return /^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$/.test(stem) ? `${slug}-web` : slug
}

export function urlSourceFileName(url: string, contentType: string, body: string): string {
  const parsed = new URL(url)
  const html = /(?:text\/html|application\/xhtml\+xml)/i.test(contentType)
  const title = html ? body.match(/<title\b[^>]*>([\s\S]*?)<\/title>/i)?.[1] : undefined
  const decodedTitle = title?.replace(/<[^>]+>/g, " ").replace(/&(?:amp|#38);/gi, "&").trim()
  const encodedLeaf = parsed.pathname.split("/").filter(Boolean).pop() ?? ""
  let pathLeaf = encodedLeaf
  try {
    pathLeaf = decodeURIComponent(encodedLeaf)
  } catch {
    // Keep the encoded path leaf. A malformed percent sequence in a remote
    // URL should not prevent importing an otherwise valid response.
  }
  const base = safeSlug(decodedTitle || pathLeaf.replace(/\.[^.]+$/, "") || parsed.hostname) || "web-page"
  return `${base}.${html ? "html" : "txt"}`
}

function attachSourceUrl(url: string, contentType: string, body: string): string {
  if (/(?:text\/html|application\/xhtml\+xml)/i.test(contentType)) {
    const escaped = url.replace(/&/g, "&amp;").replace(/"/g, "&quot;")
    const meta = `<meta name="llm-wiki-source-url" content="${escaped}">`
    return /<head\b[^>]*>/i.test(body)
      ? body.replace(/<head\b[^>]*>/i, (head) => `${head}\n${meta}`)
      : `${meta}\n${body}`
  }
  return `Source URL: ${url}\n\n${body}`
}

export async function importSourceUrls(
  project: WikiProject,
  urls: string[],
  llmConfig: LlmConfig,
  sourceWatchConfig?: SourceWatchConfig,
): Promise<UrlImportResult[]> {
  const fetch = await getHttpFetch()
  const maxBytes = normalizeSourceWatchConfig(sourceWatchConfig).maxFileSizeMb * 1024 * 1024
  const sourceRoot = `${normalizePath(project.path)}/raw/sources`
  const results: UrlImportResult[] = []
  const importedPaths: string[] = []

  for (const url of urls) {
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 60_000)
      let response: Response
      try {
        response = await fetch(url, { redirect: "follow", signal: controller.signal })
      } finally {
        clearTimeout(timeout)
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const declaredSize = Number(response.headers.get("content-length") ?? "0")
      if (Number.isFinite(declaredSize) && declaredSize > maxBytes) {
        throw new Error("Response exceeds the source file size limit")
      }
      const bytes = new Uint8Array(await response.arrayBuffer())
      if (bytes.byteLength > maxBytes) throw new Error("Response exceeds the source file size limit")
      const contentType = response.headers.get("content-type") ?? "text/plain"
      if (!/(?:text\/|application\/(?:xhtml\+xml|json|xml))/i.test(contentType)) {
        throw new Error(`Unsupported content type: ${contentType.split(";")[0]}`)
      }
      const body = new TextDecoder().decode(bytes)
      const fileName = urlSourceFileName(url, contentType, body)
      const path = await getUniqueDestPath(sourceRoot, fileName)
      await writeFile(path, attachSourceUrl(url, contentType, body))
      importedPaths.push(path)
      results.push({ url, path })
    } catch (error) {
      results.push({ url, error: error instanceof Error ? error.message : String(error) })
    }
  }

  try {
    await enqueueSourceIngest(project, importedPaths, llmConfig)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    for (const result of results) {
      if (result.path) result.error = `Saved, but failed to queue ingest: ${message}`
    }
  }
  return results
}
