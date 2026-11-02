"""Port of src/lib/vision-caption.ts: caption one image with a vision-capable LLM.

``caption_image(image_base64, media_type, llm_config, signal=None, options=None)`` is async and
returns the trimmed caption text; any LLM error is raised.  ``options`` is a dict using the TS keys
(``maxTokens``, ``temperature``, ``contextBefore``, ``contextAfter``, ``outputLanguage``) or a
:class:`CaptionOptions`; snake_case keys are accepted too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from . import llm_client
from .llm_client import StreamCallbacks
from .llm_jscompat import js_trim
from .reasoning_capabilities import resolve_ingest_reasoning

CAPTION_PROMPT = (
    "Describe this image factually for a knowledge-base index. Include: any visible text verbatim, "
    "chart axes and values, diagram structure (boxes/arrows/labels), key visual elements. Do NOT "
    "speculate or editorialize. 2 to 4 sentences. Output plain text only — no markdown, no preamble."
)


def _caption_language_instruction(output_language: Optional[str]) -> str:
    language = js_trim(output_language) if output_language else ""
    if not language or language.lower() == "auto":
        return ""
    return (f"Write the description in {language}. "
            "Preserve visible text verbatim in its original language.")


def build_caption_prompt_with_context(before: str, after: str, output_language: Optional[str] = None) -> str:
    def fmt(s: str) -> str:
        trimmed = js_trim(s)
        return trimmed if len(trimmed) > 0 else "(none)"

    parts = [
        "The image is embedded in a longer document. Here is the text that appears IMMEDIATELY BEFORE and "
        "AFTER this image in the source:",
        "",
        "--- Text before image ---",
        fmt(before),
        "--- Text after image ---",
        fmt(after),
        "--- End surrounding text ---",
        "",
        "This surrounding text MAY help describe the image — for example, a sentence like "
        "\"Figure 3: Q2 revenue chart\" tells you what the chart actually plots. It MAY ALSO be unrelated "
        "body text that just happens to flank the image. Use your judgment: if a passage clearly "
        "identifies, references, or labels the image, anchor your caption to it; if not, ignore the "
        "surrounding text and describe what you see.",
        "",
        "Now describe the image factually for a knowledge-base index. Include: any visible text verbatim, "
        "chart axes and values, diagram structure (boxes/arrows/labels), key visual elements. If the "
        "surrounding text contains a relevant figure number / caption / referent, incorporate that "
        "specifically. Do NOT invent details that aren't visible in the image or directly stated in the "
        "surrounding text. 2 to 4 sentences. Output plain text only — no markdown, no preamble.",
        _caption_language_instruction(output_language),
    ]
    return "\n".join(p for p in parts if p)


@dataclass
class CaptionOptions:
    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    contextBefore: Optional[str] = None
    contextAfter: Optional[str] = None
    outputLanguage: Optional[str] = None

    @classmethod
    def coerce(cls, value: Any) -> "CaptionOptions":
        if value is None:
            return cls()
        if isinstance(value, CaptionOptions):
            return value
        if isinstance(value, Mapping):
            def pick(camel: str, snake: str) -> Any:
                v = value.get(camel)
                return v if v is not None else value.get(snake)
            return cls(pick("maxTokens", "max_tokens"), pick("temperature", "temperature"),
                       pick("contextBefore", "context_before"), pick("contextAfter", "context_after"),
                       pick("outputLanguage", "output_language"))
        raise TypeError("options must be a dict or CaptionOptions")


async def caption_image(
    image_base64: str,
    media_type: str,
    llm_config: Mapping[str, Any],
    signal: Any = None,
    options: Any = None,
    *,
    transport: Any = None,
) -> str:
    """Caption a single image (``image_base64`` is raw base64, NOT a data: URL)."""
    cfg = llm_client._as_config_dict(llm_config)
    if cfg.get("provider") == "codex-cli":
        raise Exception("Codex CLI transport does not support image input for captioning yet.")

    opts = CaptionOptions.coerce(options)
    before = js_trim(opts.contextBefore) if opts.contextBefore else ""
    after = js_trim(opts.contextAfter) if opts.contextAfter else ""
    if len(before) > 0 or len(after) > 0:
        prompt_text = build_caption_prompt_with_context(before, after, opts.outputLanguage)
    else:
        prompt_text = "\n".join(p for p in (CAPTION_PROMPT, _caption_language_instruction(opts.outputLanguage)) if p)

    messages: List[Dict[str, Any]] = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image", "mediaType": media_type, "dataBase64": image_base64},
        ],
    }]

    tokens: List[str] = []
    errors: List[Exception] = []
    callbacks = StreamCallbacks(on_token=tokens.append, on_done=lambda *a: None, on_error=errors.append)
    overrides = {
        "temperature": opts.temperature if opts.temperature is not None else 0,
        "max_tokens": opts.maxTokens if opts.maxTokens is not None else 4096,
        # Captioning is a short factual vision task, so reasoning defaults off. The ingest
        # preference can enable it for models that require thinking.
        "reasoning": resolve_ingest_reasoning(cfg),
    }
    extra = {"transport": transport} if transport is not None else {}
    await llm_client.stream_chat(llm_config, messages, callbacks, signal, overrides, **extra)

    if errors:
        # stream_chat reports HTTP / network errors via on_error but returns cleanly; re-raise so the
        # caller can try/except the caption call as a unit.
        raise errors[0]
    return js_trim("".join(tokens))
