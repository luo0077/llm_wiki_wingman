"""Port of src/lib/detect-language.ts (Unicode-script based language detection)."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ._js import rx

# (script, ranges) in the exact order the TS getScript() checks them.
_SCRIPT_RANGES = [
    ("Chinese", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0xF900, 0xFAFF)]),
    ("Japanese", [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF65, 0xFF9F)]),
    ("Korean", [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)]),
    ("Arabic", [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]),
    ("Hebrew", [(0x0590, 0x05FF), (0xFB1D, 0xFB4F)]),
    ("Thai", [(0x0E00, 0x0E7F)]),
    ("Hindi", [(0x0900, 0x097F)]),
    ("Bengali", [(0x0980, 0x09FF)]),
    ("Tamil", [(0x0B80, 0x0BFF)]),
    ("Telugu", [(0x0C00, 0x0C7F)]),
    ("Kannada", [(0x0C80, 0x0CFF)]),
    ("Malayalam", [(0x0D00, 0x0D7F)]),
    ("Gujarati", [(0x0A80, 0x0AFF)]),
    ("Punjabi", [(0x0A00, 0x0A7F)]),
    ("Burmese", [(0x1000, 0x109F)]),
    ("Khmer", [(0x1780, 0x17FF)]),
    ("Lao", [(0x0E80, 0x0EFF)]),
    ("Georgian", [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)]),
    ("Armenian", [(0x0530, 0x058F)]),
    ("Amharic", [(0x1200, 0x137F)]),
    ("Tibetan", [(0x0F00, 0x0FFF)]),
    ("Sinhala", [(0x0D80, 0x0DFF)]),
    ("Russian", [(0x0400, 0x04FF), (0x0500, 0x052F)]),
    ("Greek", [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
]

_RTL_PERSIAN = {"پ": 3, "چ": 3, "ژ": 3, "گ": 3, "ک": 1, "ی": 1}
_ARABIC_ONLY = set("كيةىإأؤئ")

_PERSIAN_WORDS = ["این", "است", "که", "برای", "های", "را", "در", "به", "از", "می", "یک"]
_ARABIC_WORDS = ["ال", "في", "من", "على", "هذا", "هذه", "إلى", "التي", "الذي", "كان"]

_NON_LN_RUN_RE = None


def _get_script(cp: int) -> Optional[str]:
    for name, ranges in _SCRIPT_RANGES:
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return "Russian" if name == "Cyrillic" else name
    return None


def detect_language(text: str) -> str:
    """Detect the primary language of `text`; returns an English language name."""
    counts: Dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if not cp or cp < 0x80:
            continue
        script = _get_script(cp)
        if script:
            counts[script] = counts.get(script, 0) + 1

    if counts.get("Japanese", 0) > 0 and counts.get("Chinese", 0) > 0:
        return "Japanese"

    max_script = ""
    max_count = 0
    for script, count in counts.items():
        if count > max_count:
            max_script = script
            max_count = count

    if max_script == "Arabic" and max_count >= 2:
        return _detect_arabic_script_language(text)

    if max_script and max_count >= 2:
        return max_script

    latin = _detect_latin_language(text)
    if latin:
        return latin

    return "English"


def _letters_numbers_to_spaces(text: str) -> str:
    """JS text.replace(/[^\\p{L}\\p{N}]+/gu, " ")."""
    import unicodedata

    out: List[str] = []
    in_run = False
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] == "L" or cat[0] == "N":
            out.append(ch)
            in_run = False
        elif not in_run:
            out.append(" ")
            in_run = True
    return "".join(out)


def _detect_arabic_script_language(text: str) -> str:
    persian_score = 0
    arabic_score = 0
    for ch in text:
        if ch in _RTL_PERSIAN:
            persian_score += _RTL_PERSIAN[ch]
        elif ch in _ARABIC_ONLY:
            arabic_score += 1

    normalized = " " + _letters_numbers_to_spaces(text) + " "
    for word in _PERSIAN_WORDS:
        if (" " + word + " ") in normalized:
            persian_score += 2
    for word in _ARABIC_WORDS:
        if (" " + word + " ") in normalized:
            arabic_score += 2

    return "Persian" if persian_score >= 3 and persian_score > arabic_score else "Arabic"


def _re(pattern: str) -> "re.Pattern[str]":
    return rx(pattern)


_VI = _re("[ảạắằẳẵặấầẩẫậđẻẽẹếềểễệỉĩịỏọốồổỗộơớờởỡợủũụưứừửữựỷỹỵ]")
_TR_CH = _re("[ğış]")
_TR_W = _re(r"\b(bir|ve|için|ile|bu|da|de|değil|ama)\b")
_PL = _re("[ąćęłńśźż]")
_CS = _re("[ěšžřďťňů]")
_RO_CH = _re("[ăâîșț]")
_RO_W = _re(r"\b(și|este|sau|care|pentru)\b")
_HU = _re("[őű]")
_DE_W = _re(r"\b(?:und|der|die|das|ist|nicht|ein|eine)\b")
_DE_CH = _re("[äöüß]")
_FR_CH = _re("[àâçéèêëïîôùûüÿœæ]")
_FR_W1 = _re(r"\b(le|la|les|de|des|est|et|un|une|du|au)\b")
_FR_W2 = _re(r"\b(le|la|les|est|une|des)\b")
_PT_CH = _re("[ãõç]")
_PT_W = _re(r"\b(o|a|os|as|de|do|da|é|em|um|uma|não|que)\b")
_ES_CH = _re("[áéíóúñ¿¡]")
_ES_W1 = _re(r"\b(el|la|los|las|de|del|es|en|por|que|un|una)\b")
_ES_W2 = _re(r"\b(el|los|las|del|por)\b")
_ES_CH2 = _re("[ñ¿¡]")
_IT_W1 = _re(r"\b(il|lo|la|gli|le|di|del|della|è|e|un|una|che|non|per)\b")
_IT_W2 = _re(r"\b(il|della|gli|che|è)\b")
_NL_W1 = _re(r"\b(het|de|een|van|en|in|is|dat|op|te|met)\b")
_NL_W2 = _re(r"\b(het|een|van|dat)\b")
_SV_CH = _re("[åäö]")
_SV_W = _re(r"\b(och|att|det|en|ett|är|för|med)\b")
_NO_CH = _re("[åæø]")
_NO_W = _re(r"\b(og|er|det|en|et|for|med|på)\b")
_DA_W = _re(r"\b(og|er|det|en|et|til|med|af)\b")
_FI_CH = _re("[äö]")
_FI_W = _re(r"\b(ja|on|ei|se|että|tai|kun|niin)\b")
_ID_W1 = _re(r"\b(dan|yang|di|dari|untuk|dengan|ini|itu|adalah|tidak|ada)\b")
_ID_W2 = _re(r"\b(yang|dari|untuk|dengan|adalah)\b")
_SW_W = _re(r"\b(na|ya|wa|ni|kwa|katika|hii|hiyo)\b")


def _detect_latin_language(text: str) -> Optional[str]:
    lower = text.lower()

    if _VI.search(lower):
        return "Vietnamese"
    if _TR_CH.search(lower) and _TR_W.search(lower):
        return "Turkish"
    if _PL.search(lower):
        return "Polish"
    if _CS.search(lower):
        return "Czech"
    if _RO_CH.search(lower) and _RO_W.search(lower):
        return "Romanian"
    if _HU.search(lower):
        return "Hungarian"

    german_signals = set(_DE_W.findall(lower))
    if len(german_signals) >= 2 or (_DE_CH.search(lower) and len(german_signals) >= 1):
        return "German"

    if _FR_CH.search(lower) or _FR_W1.search(lower):
        if _FR_W2.search(lower):
            return "French"

    if _PT_CH.search(lower) and _PT_W.search(lower):
        return "Portuguese"

    if _ES_CH.search(lower) or _ES_W1.search(lower):
        if _ES_W2.search(lower) or _ES_CH2.search(lower):
            return "Spanish"

    if _IT_W1.search(lower):
        if _IT_W2.search(lower):
            return "Italian"

    if _NL_W1.search(lower):
        if _NL_W2.search(lower):
            return "Dutch"

    if _SV_CH.search(lower) and _SV_W.search(lower):
        return "Swedish"
    if _NO_CH.search(lower) and _NO_W.search(lower):
        return "Norwegian"
    if _NO_CH.search(lower) and _DA_W.search(lower):
        return "Danish"
    if _FI_CH.search(lower) and _FI_W.search(lower):
        return "Finnish"

    if _ID_W1.search(lower):
        if _ID_W2.search(lower):
            return "Indonesian"

    if _SW_W.search(lower):
        return "Swahili"

    return None
