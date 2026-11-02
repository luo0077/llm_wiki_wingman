"""Translations of detect-language(.property).test.ts, language-metadata.test.ts, output-language.test.ts."""
from __future__ import annotations

import random
import re

import pytest

from llm_wiki_server.ingest.detect_language import detect_language
from llm_wiki_server.ingest.language_metadata import (
    get_html_lang,
    get_language_prompt_name,
    get_text_direction,
    same_script_family,
)
from llm_wiki_server.ingest.output_language import (
    build_language_directive,
    build_language_reminder,
    get_output_language,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", "English"),
        ("abc xyz 123", "English"),
        ("注意力机制是什么", "Chinese"),
        ("これはテストです", "Japanese"),
        ("안녕하세요", "Korean"),
        ("مرحبا بالعالم", "Arabic"),
        ("سلام دنیا، این یک متن فارسی برای آزمایش است", "Persian"),
        ("اللغة العربية مهمة في العالم", "Arabic"),
        ("سلام", "Arabic"),
        ("สวัสดีครับ", "Thai"),
        ("नमस्ते दुनिया", "Hindi"),
        ("привет мир", "Russian"),
        ("Γειά σου κόσμε", "Greek"),
        ("x中", "English"),
        ("机器学习 machine learning 深度学习 神经网络", "Chinese"),
        ("Xử lý nước thải", "Vietnamese"),
        ("Việt Nam là một quốc gia xinh đẹp", "Vietnamese"),
        ("Příliš žluťoučký kůň úpěl ďábelské ódy", "Czech"),
        ("le chat noir et les chiens blancs, un homme et une femme", "French"),
        ("le chat est là, les étudiants préfèrent le café", "French"),
        ("der Hund und die Katze sind nicht das Problem", "German"),
        ("# Cell lifecycle\nDamaged cells die when repair pathways fail.\nThe remaining cells continue to divide and support the tissue.", "English"),
        ("el nino y los libros del colegio que son para todos", "Spanish"),
        ("dzień dobry świat", "Polish"),
        ("o coração do Brasil é um lugar de paz e que encanta", "Portuguese"),
        ("123 !@#$", "English"),
        ("a" * 10000, "English"),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_detect_language_extra_latin():
    # Branches beyond the TS tests (parity with the JS regex semantics).
    assert detect_language("bu bir test ve ğ için") == "Turkish"
    assert detect_language("och att det är för med å") == "Swedish"
    assert detect_language("il gatto della casa che è bello") == "Italian"
    assert detect_language("het is een van de dat") == "Dutch"
    assert detect_language("yang dari untuk dengan") == "Indonesian"
    assert detect_language("na ya wa ni kwa") == "Swahili"


KNOWN_LATIN = {
    "English", "French", "German", "Spanish", "Portuguese", "Italian", "Dutch", "Swedish", "Norwegian", "Danish",
    "Finnish", "Indonesian", "Swahili", "Polish", "Czech", "Romanian", "Hungarian", "Vietnamese", "Turkish",
}


def test_detect_language_properties():
    rng = random.Random(7)
    for _ in range(300):
        s = "".join(chr(rng.randint(0x20, 0x7E)) for _ in range(rng.randint(0, 200)))
        assert detect_language(s) in KNOWN_LATIN
    for lo, hi, lang in [(0x4E00, 0x9FFF, "Chinese"), (0x3041, 0x3096, "Japanese"), (0xAC00, 0xD7AF, "Korean")]:
        for _ in range(100):
            s = "".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(2, 50)))
            assert detect_language(s) == lang
    for _ in range(200):
        s = "".join(chr(rng.choice([rng.randint(0, 0x2FFF), rng.randint(0x10000, 0x2FFFF)])) for _ in range(rng.randint(0, 60)))
        detect_language(s)  # never throws


# ── language-metadata ─────────────────────────────────────────────

def test_persian_metadata():
    assert get_language_prompt_name("Persian") == "Persian (Farsi / فارسی)"
    assert get_text_direction("Persian") == "rtl"
    assert get_html_lang("Persian") == "fa"
    assert same_script_family("Persian", "Arabic") is True


def test_czech_metadata():
    assert get_language_prompt_name("Czech") == "Czech / čeština"
    assert get_text_direction("Czech") == "ltr"
    assert get_html_lang("Czech") == "cs"
    assert same_script_family("Czech", "English") is True


def test_unknown_language_defaults():
    assert get_language_prompt_name("Vietnamese") == "Vietnamese"
    assert get_text_direction("Vietnamese") == "ltr"
    assert get_html_lang("Vietnamese") is None
    assert get_language_prompt_name("") == "English"


# ── output-language (store value is an explicit parameter) ────────

def test_output_language_explicit_beats_detection():
    assert get_output_language("whatever fallback text", "Chinese") == "Chinese"
    assert get_output_language("This is clearly English text", "Japanese") == "Japanese"


def test_output_language_auto():
    assert get_output_language("注意力机制是什么", "auto") == "Chinese"
    assert get_output_language("پردازش زبان طبیعی در فارسی کاربردهای زیادی دارد", "auto") == "Persian"
    assert get_output_language("English fallback text", "Czech") == "Czech"
    assert get_output_language("Příliš žluťoučký kůň úpěl ďábelské ódy", "auto") == "Czech"
    assert get_output_language("", "auto") == "English"
    assert get_output_language() == "English"
    assert get_output_language("注意力机制是什么", None) == "Chinese"


def test_directive():
    d = build_language_directive("", "Chinese")
    assert "MANDATORY OUTPUT LANGUAGE: Chinese" in d
    d = build_language_directive("", "Japanese")
    assert d.count("Japanese") >= 3
    for s in [
        "Write surrounding natural-language prose",
        "prose titles and section headings",
        "Preserve organization names",
        "model names",
        "paper titles",
        "technical terms that have no widely-used localized equivalent",
        "does not override the proper-noun and technical-identifier preservation rule",
    ]:
        assert s in d
    assert not re.search("transliteration when appropriate", d, re.I)
    assert "overrides all other instructions" not in d


def test_directive_misc():
    d = build_language_directive("这段文字是中文", "Vietnamese")
    assert "Vietnamese" in d and "MANDATORY OUTPUT LANGUAGE: Chinese" not in d
    assert "Vietnamese" in build_language_directive("Xử lý nước thải là vấn đề quan trọng", "auto")
    assert "MANDATORY OUTPUT LANGUAGE: Persian (Farsi / فارسی)" in build_language_directive("", "Persian")
    assert "MANDATORY OUTPUT LANGUAGE: Czech / čeština" in build_language_directive("", "Czech")
    d = build_language_directive("", "English")
    assert "use it as evidence" in d and "proper nouns and technical identifiers" in d
    # exact first lines (byte-for-byte parity with the TS template)
    assert build_language_directive("", "English").split("\n")[:3] == [
        "## ⚠️ MANDATORY OUTPUT LANGUAGE: English",
        "",
        "Write surrounding natural-language prose in **English**.",
    ]


def test_reminder():
    r = build_language_reminder("", "Chinese")
    assert re.search("Write prose in Chinese", r)
    assert "preserve names" in r
    assert len(r.split("\n")) == 1
    assert "Korean" in build_language_reminder("ignored fallback", "Korean")
    assert "Japanese" in build_language_reminder("これは日本語です", "auto")
    assert "Persian (Farsi / فارسی)" in build_language_reminder("", "Persian")
