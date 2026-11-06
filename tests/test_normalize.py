import pytest

from ml.data.normalize import clean_subtitle_text, normalization_warnings, normalize_sorani


def test_sorani_normalizes_codepoints_punctuation_and_preserves_kurdish_letters():
    text = "ك ي ى ة ڕ ڵ ێ ۆ ڤ پ چ ژ گ? , ; ... “بەڵێ”"
    assert normalize_sorani(text) == 'ک ی ی ە ڕ ڵ ێ ۆ ڤ پ چ ژ گ؟ ، ؛ … "بەڵێ"'


def test_heh_is_preserved_and_flagged_by_default():
    text = "هەموو ئەوە باشە، شاه"
    normalized = normalize_sorani(text)
    assert normalized == text
    assert normalization_warnings(normalized) == ["ambiguous_arabic_heh"]


def test_positional_heh_is_an_explicit_lossy_opt_in_and_idempotent():
    normalized = normalize_sorani("هەموو راه وه", heh_policy="positional")
    assert normalized == "ھەموو راە وە"
    assert normalize_sorani(normalized, heh_policy="positional") == normalized


def test_zwnj_is_preserved_bidi_controls_are_removed():
    assert normalize_sorani("\u200fبە\u200cی\u2066ABC\u2069") == "بە\u200cیABC"


def test_unrecognized_heh_policy_is_rejected():
    with pytest.raises(ValueError, match="heh policy"):
        normalize_sorani("سڵاو", heh_policy="unknown")


def test_clean_extracts_speakers_sfx_and_removes_only_ad_lines():
    result = clean_subtitle_text(
        "<i>JOHN:</i> Hello, damn it! [DOOR SLAMS] ♪\n"
        "MARY: I am here.\nSubtitles by Somebody\nwww.opensubtitles.org",
    )
    assert result.text == "Hello, damn it!\nI am here."
    assert result.speakers == ("JOHN", "MARY")
    assert result.sound_effects == ("DOOR SLAMS",)
    assert result.removed_ad_lines == 2


def test_html_entities_do_not_turn_into_markup_and_parentheses_are_kept():
    result = clean_subtitle_text("ALICE: &lt;3 (I am fine) &amp; you?")
    assert result.text == "<3 (I am fine) & you?"
    assert result.speakers == ("ALICE",)


def test_voice_speaker_and_sfx_only_cue_are_preserved():
    result = clean_subtitle_text("[RAIN FALLS] ♪", named_speaker="Narrator")
    assert result.text == ""
    assert result.speakers == ("Narrator",)
    assert result.sound_effects == ("RAIN FALLS",)
