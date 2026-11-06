from pathlib import Path

import pysubs2
import pytest

from ml.data.loaders import SubtitleLoadError, load_subtitles


def test_srt_loading_bom_tags_and_sfx(tmp_path: Path):
    path = tmp_path / "en.srt"
    path.write_text(
        "42\n00:00:01,025 --> 00:00:03,450\n<i>JOHN:</i> Hello [DOOR SLAMS]\n\n"
        "99\n00:00:04,000 --> 00:00:05,000\n[WIND BLOWS]\n",
        encoding="utf-8-sig",
    )
    first, second = load_subtitles(path)
    assert (first.index, first.start_ms, first.end_ms) == (1, 1025, 3450)
    assert first.text == "Hello"
    assert first.speaker == "JOHN"
    assert first.sound_effects == ("DOOR SLAMS",)
    assert second.text == ""
    assert second.sound_effects == ("WIND BLOWS",)


def test_vtt_voice_identifier_and_millisecond_precision(tmp_path: Path):
    path = tmp_path / "en.vtt"
    path.write_text(
        "WEBVTT\n\nNOTE a comment\nignored\n\nmy-cue\n"
        "00:00:01.123 --> 00:00:02.456 align:start\n<v JOHN>Hello &amp; goodbye</v>\n\n",
        encoding="utf-8",
    )
    cue = load_subtitles(path)[0]
    assert (cue.start_ms, cue.end_ms, cue.text, cue.speaker) == (
        1123,
        2456,
        "Hello & goodbye",
        "JOHN",
    )


@pytest.mark.parametrize("extension", ["ass", "ssa"])
def test_ass_and_ssa_preserve_style_speaker_and_times(tmp_path: Path, extension: str):
    subtitle = pysubs2.SSAFile()
    subtitle.events.append(
        pysubs2.SSAEvent(
            start=1050,
            end=2350,
            text=r"{\i1}Hello\Nthere{\i0}",
            name="JOHN",
            style="Default",
        )
    )
    path = tmp_path / f"en.{extension}"
    path.write_text(subtitle.to_string(format_=extension), encoding="utf-8")
    cue = load_subtitles(path)[0]
    assert (cue.start_ms, cue.end_ms) == (1050, 2350)
    assert cue.text == "Hello\nthere"
    assert cue.speaker == "JOHN"
    assert cue.style == "Default"


def test_invalid_encoding_and_unsupported_extension_fail_clearly(tmp_path: Path):
    invalid = tmp_path / "bad.srt"
    invalid.write_bytes(b"\xff")
    with pytest.raises(SubtitleLoadError, match="Could not read"):
        load_subtitles(invalid)
    with pytest.raises(SubtitleLoadError, match="unsupported"):
        load_subtitles(tmp_path / "bad.txt")


def test_no_cues_and_invalid_timing_fail_clearly(tmp_path: Path):
    path = tmp_path / "bad.srt"
    path.write_text("Not subtitle content", encoding="utf-8")
    with pytest.raises(SubtitleLoadError, match="no subtitle cues"):
        load_subtitles(path)
    path.write_text("1\n00:00:02,000 --> 00:00:01,000\nBackwards\n", encoding="utf-8")
    with pytest.raises(SubtitleLoadError, match="invalid timestamps"):
        load_subtitles(path)
