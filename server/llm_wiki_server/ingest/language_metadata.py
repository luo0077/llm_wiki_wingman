"""Port of src/lib/language-metadata.ts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ._js import Record


@dataclass
class LanguageMetadata(Record):
    promptName: str
    direction: str  # "ltr" | "rtl"
    scriptFamily: str  # "arabic" | "cjk" | "latin" | "other"
    htmlLang: Optional[str] = None


_LANGUAGE_METADATA: Dict[str, LanguageMetadata] = {
    "English": LanguageMetadata("English", "ltr", "latin", "en"),
    "Arabic": LanguageMetadata("Arabic / العربية", "rtl", "arabic", "ar"),
    "Persian": LanguageMetadata("Persian (Farsi / فارسی)", "rtl", "arabic", "fa"),
    "Hebrew": LanguageMetadata("Hebrew / עברית", "rtl", "other", "he"),
    "Chinese": LanguageMetadata("Chinese", "ltr", "cjk", "zh-Hans"),
    "Traditional Chinese": LanguageMetadata("Traditional Chinese", "ltr", "cjk", "zh-Hant"),
    "Japanese": LanguageMetadata("Japanese", "ltr", "cjk", "ja"),
    "Korean": LanguageMetadata("Korean", "ltr", "cjk", "ko"),
    "Czech": LanguageMetadata("Czech / čeština", "ltr", "latin", "cs"),
}


def get_language_metadata(language: str) -> LanguageMetadata:
    found = _LANGUAGE_METADATA.get(language)
    if found is not None:
        return found
    return LanguageMetadata(promptName=language or "English", direction="ltr", scriptFamily="latin", htmlLang=None)


def get_language_prompt_name(language: str) -> str:
    return get_language_metadata(language).promptName


def get_text_direction(language: str) -> str:
    return get_language_metadata(language).direction


def get_html_lang(language: str) -> Optional[str]:
    return get_language_metadata(language).htmlLang


def same_script_family(a: str, b: str) -> bool:
    return get_language_metadata(a).scriptFamily == get_language_metadata(b).scriptFamily
