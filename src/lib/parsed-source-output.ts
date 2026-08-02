import { createDirectory, writeFileAtomic } from "@/commands/fs"
import { getFileName, getRelativePath, isAbsolutePath, normalizePath } from "@/lib/path-utils"

const PARSED_MARKDOWN_EXTENSIONS = new Set([
  "pdf",
  "doc",
  "docx",
  "pptx",
  "xls",
  "xlsx",
  "odt",
  "ods",
  "odp",
  "epub",
  "mobi",
  "org",
])

/**
 * Map a parsed source to a visible, generated Markdown path. The source must
 * live below this project's raw/sources directory; refusing arbitrary paths
 * keeps this convenience export inside the project sandbox. The original
 * extension remains in the filename (report.pdf.md), avoiding collisions
 * between report.pdf, report.docx, and a user-authored report.md.
 */
export function parsedMarkdownOutputPath(
  projectPath: string,
  sourcePath: string,
): string | null {
  const project = normalizePath(projectPath).replace(/\/+$/, "")
  const source = normalizePath(sourcePath)
  const sourceRoot = `${project}/raw/sources`
  const relative = getRelativePath(source, sourceRoot)
  if (!relative || relative === source || isAbsolutePath(relative)) return null
  if (relative.split("/").some((part) => !part || part === "." || part === "..")) return null

  const fileName = getFileName(relative)
  const extension = fileName.includes(".")
    ? fileName.split(".").pop()?.toLowerCase() ?? ""
    : ""
  if (!PARSED_MARKDOWN_EXTENSIONS.has(extension)) return null
  return `${project}/raw/parsed/${relative}.md`
}

export async function persistParsedMarkdown(
  projectPath: string,
  sourcePath: string,
  markdown: string,
): Promise<string | null> {
  const outputPath = parsedMarkdownOutputPath(projectPath, sourcePath)
  if (!outputPath) return null
  const parent = outputPath.slice(0, outputPath.lastIndexOf("/"))
  await createDirectory(parent)
  await writeFileAtomic(outputPath, markdown)
  return outputPath
}
