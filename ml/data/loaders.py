"""Read SRT, WebVTT, ASS and SSA cues without altering their timings."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pysubs2
import webvtt

from ml.data.normalize import clean_subtitle_text

SUPPORTED_EXTENSIONS = frozenset({".srt", ".vtt", ".ass", ".ssa"})


class SubtitleLoadError(ValueError):
    """A subtitle file cannot be decoded, parsed, or contains invalid timing."""


@dataclass(frozen=True, slots=True)
class Cue:
    # Index is the 1-based dialogue order in the file, not an SRT/VTT identifier.
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None
    style: str | None = None
    speaker_labels: tuple[str, ...] = ()
    sound_effects: tuple[str, ...] = ()
    removed_ad_lines: int = 0


def _cue(
    index: int,
    start_ms: int,
    end_ms: int,
    text: str,
    speaker: str | None,
    style: str | None,
    path: Path,
) -> Cue:
    if start_ms < 0 or end_ms <= start_ms:
        raise SubtitleLoadError(
            f"{path}: cue {index} has invalid timestamps {start_ms}–{end_ms} ms"
        )
    cleaned = clean_subtitle_text(text, named_speaker=speaker)
    return Cue(
        index=index,
        start_ms=start_ms,
        end_ms=end_ms,
        text=cleaned.text,
        speaker=cleaned.speakers[0] if cleaned.speakers else None,
        style=style,
        speaker_labels=cleaned.speakers,
        sound_effects=cleaned.sound_effects,
        removed_ad_lines=cleaned.removed_ad_lines,
    )


def load_subtitles(path: str | Path, *, encoding: str = "utf-8-sig") -> list[Cue]:
    """Parse a local subtitle file. Empty/SFX-only cues remain as structured cues.

    VTT uses webvtt-py so voice spans survive parsing; pysubs2 handles SRT and
    ASS/SSA. All cue indices and millisecond timestamps are preserved thereafter.
    """
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise SubtitleLoadError(f"{path}: unsupported subtitle extension {suffix!r}")
    try:
        content = path.read_text(encoding=encoding, errors="strict")
        if suffix == ".vtt":
            captions = webvtt.from_buffer(StringIO(content))
            cues = [
                _cue(
                    index,
                    caption.start_time.in_milliseconds(),
                    caption.end_time.in_milliseconds(),
                    caption.raw_text,
                    caption.voice,
                    None,
                    path,
                )
                for index, caption in enumerate(captions, 1)
            ]
        else:
            subtitle_file = pysubs2.SSAFile.from_string(content, format_=suffix[1:])
            events = [event for event in subtitle_file if not event.is_comment]
            cues = [
                _cue(
                    index,
                    event.start,
                    event.end,
                    event.plaintext,
                    event.name or None,
                    event.style if suffix in {".ass", ".ssa"} else None,
                    path,
                )
                for index, event in enumerate(events, 1)
            ]
    except SubtitleLoadError:
        raise
    except (
        OSError,
        UnicodeError,
        LookupError,
        ValueError,
        TypeError,
        webvtt.errors.WebVTTError,
    ) as exc:
        raise SubtitleLoadError(f"Could not read {path}: {exc}") from exc
    if not cues:
        raise SubtitleLoadError(f"{path}: no subtitle cues found; check the format and encoding")
    return cues
