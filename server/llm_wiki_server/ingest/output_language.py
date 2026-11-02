"""Port of src/lib/output-language.ts.

The TS version reads ``useWikiStore.getState().outputLanguage``; here the
configured value is an explicit ``configured`` parameter (``"auto"``/empty/None
means "detect from text").
"""
from __future__ import annotations

from typing import Optional

from .detect_language import detect_language
from .language_metadata import get_language_prompt_name


def get_output_language(fallback_text: str = "", configured: Optional[str] = None) -> str:
    if configured and configured != "auto":
        return configured
    return detect_language(fallback_text or "English")


def build_language_directive(fallback_text: str = "", configured: Optional[str] = None) -> str:
    lang = get_output_language(fallback_text, configured)
    prompt_lang = get_language_prompt_name(lang)
    return "\n".join(
        [
            f"## ⚠️ MANDATORY OUTPUT LANGUAGE: {prompt_lang}",
            "",
            f"Write surrounding natural-language prose in **{prompt_lang}**.",
            f"All generated prose, including prose titles and section headings, must be in {prompt_lang}.",
            "Do not translate, transliterate, or describe proper nouns and technical identifiers unless the source already uses a well-established localized form.",
            "Preserve organization names, product names, model names, dataset names, tool/library names, acronyms, code identifiers, file names, URLs, paper titles, citation strings, and technical terms that have no widely-used localized equivalent in their standard original form.",
            f"The source material or wiki content may be in a different language; use it as evidence, but keep generated prose in {prompt_lang}.",
            "This language rule overrides weaker style instructions, but it does not override the proper-noun and technical-identifier preservation rule above.",
        ]
    )


def build_language_reminder(fallback_text: str = "", configured: Optional[str] = None) -> str:
    lang = get_output_language(fallback_text, configured)
    return f"REMINDER: Write prose in {get_language_prompt_name(lang)}; preserve names, acronyms, identifiers, URLs, file names, and paper titles in their standard original form."
