"""Conservative subtitle cleaning and Sorani Arabic-script normalization.

The Arabic heh character is genuinely ambiguous in Kurdish. By default it is
preserved and flagged rather than silently turning a consonant into a vowel.
The optional positional policy is lossy and should be reviewed by a speaker.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

HehPolicy = Literal["preserve", "positional"]

# NFKC first resolves Arabic presentation forms. Keep ZWNJ and Kurdish letters.
_CHAR_MAP = str.maketrans(
    {
        "ك": "ک",
        "ي": "ی",
        "ى": "ی",
        "ة": "ە",
        "ـ": "",
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "–": "—",
        "−": "—",
        "\u200e": "",
        "\u200f": "",
        "\u202a": "",
        "\u202b": "",
        "\u202c": "",
        "\u202d": "",
        "\u202e": "",
        "\u2066": "",
        "\u2067": "",
        "\u2068": "",
        "\u2069": "",
        "\ufeff": "",
    }
)
_PUNCTUATION = str.maketrans({"?": "؟", ",": "،", ";": "؛"})
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]*?>")
_ASS_TAG = re.compile(r"\{\\[^{}]*\}")
_WEBVTT_TIMESTAMP = re.compile(r"<(?:\d{2,}:)?\d{2}:\d{2}\.\d{3}>")
_SFX = re.compile(r"\[([^\[\]\n]{1,120})\]")
_SPEAKER = re.compile(r"^\s*[-—]?\s*([A-Z][A-Z0-9 ._'-]{0,39}):\s*(.*)$")
_AD = re.compile(
    r"^\s*(?:"
    r"(?:subtitles?|captions?|translation|translated|sync(?:ed|hronized)?|resync(?:ed)?)"
    r"\s*(?:by|from|at|:).+|"
    r"(?:www\.|https?://|opensubtitles\b|subscene\b|addic7ed\b).+|"
    r"(?:download(?:ed)?|visit|follow us).*(?:www\.|https?://|\.com\b|\.org\b)"
    r")\s*$",
    re.IGNORECASE,
)
_MUSIC = str.maketrans("", "", "♪♫♬♩♭♮♯")


@dataclass(frozen=True, slots=True)
class CleanedText:
    text: str
    speakers: tuple[str, ...]
    sound_effects: tuple[str, ...]
    removed_ad_lines: int


def _normalize_common(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)
    text = re.sub(r"\.{3,}", "…", text)
    # Preserve cue line boundaries, including ASS/VTT line breaks.
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()


def normalize_english(text: str) -> str:
    """Normalize typography without changing English punctuation or register."""
    return _normalize_common(text)


def normalize_sorani(text: str, heh_policy: HehPolicy = "preserve") -> str:
    """Normalize Sorani characters and RTL punctuation.

    ``positional`` maps word-final ه to ە and other ه to ھ; this is only a
    heuristic, not a linguistic disambiguator. The default leaves ه unchanged.
    Arabic-script ة is normalized to ە in either mode.
    """
    if heh_policy not in {"preserve", "positional"}:
        raise ValueError(f"Unknown heh policy: {heh_policy}")
    text = _normalize_common(text).translate(_PUNCTUATION)
    if heh_policy == "positional":
        # A following Arabic-script letter/mark or ZWNJ indicates a nonfinal
        # position; punctuation, Latin letters and spaces end the word.
        chars = list(text)
        for position, char in enumerate(chars):
            if char != "ه":
                continue
            following = chars[position + 1] if position + 1 < len(chars) else ""
            continues = bool(following) and (
                following == "\u200c"
                or (
                    unicodedata.category(following)[0] in {"L", "M"}
                    and "ARABIC" in unicodedata.name(following, "")
                )
            )
            chars[position] = "ھ" if continues else "ە"
        text = "".join(chars)
    return text


def clean_subtitle_text(text: str, *, named_speaker: str | None = None) -> CleanedText:
    """Strip subtitle markup and advertising while retaining speaker/SFX metadata.

    Only square-bracketed annotations are treated as sound effects. Parentheses
    can contain dialogue and are therefore left intact. Profanity is never
    modified. Labels in VTT ``<v>`` and ASS ``Name`` are passed as named_speaker.
    """
    text = _ASS_TAG.sub("", text)
    text = _WEBVTT_TIMESTAMP.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = html.unescape(text).replace(r"\N", "\n").replace(r"\n", "\n")
    text = text.replace(r"\h", " ").translate(_MUSIC)
    speakers: list[str] = []
    if named_speaker and named_speaker.strip():
        speakers.append(" ".join(named_speaker.split()))
    effects: list[str] = []
    kept_lines: list[str] = []
    removed_ads = 0
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if _AD.match(line):
            removed_ads += 1
            continue
        match = _SPEAKER.match(line)
        if match:
            speaker, line = match.groups()
            speaker = speaker.strip()
            if speaker not in speakers:
                speakers.append(speaker)
        for effect in _SFX.findall(line):
            cleaned = " ".join(effect.split())
            if cleaned:
                effects.append(cleaned)
        line = " ".join(_SFX.sub(" ", line).split())
        if line:
            kept_lines.append(line)
    return CleanedText("\n".join(kept_lines), tuple(speakers), tuple(effects), removed_ads)


def normalization_warnings(text: str) -> list[str]:
    """Return audit flags for orthography that cannot safely be inferred."""
    return ["ambiguous_arabic_heh"] if "ه" in text else []
