import pytest

from ml.inference.compressor import compress
from ml.inference.qa import score_cue, summarize_qa
from ml.inference.text import cps, isolate_ltr, normalize_target, visible_characters, wrap_lines


def test_unicode_graphemes_rtl_punctuation_and_latin_isolates():
    assert visible_characters("سَڵاو \u2066John 12\u2069") == 12
    assert visible_characters("👨‍👩‍👧\n") == 1
    raw = normalize_target("كوردى, John 12:30?", "en-ckb")
    assert raw == "کوردی، John 12:30؟"
    text = isolate_ltr(raw)
    assert "\u2066John\u2069" in text and "\u206612:30\u2069" in text
    assert isolate_ltr(text) == text
    assert cps(text, 2000) == visible_characters(text) / 2
    accented = isolate_ltr("سڵاو José Łódź e\u0301 ٢٠٢٦")
    assert "\u2066José\u2069" in accented
    assert "\u2066Łódź\u2069" in accented
    assert "\u2066e\u0301\u2069" in accented
    assert "\u2066٢٠٢٦\u2069" in accented
    assert isolate_ltr(accented) == accented


def test_wrap_never_breaks_words_or_discards_content():
    text = "one two three four five six seven eight"
    wrapped = wrap_lines(text, max_chars=22)
    assert len(wrapped.splitlines()) == 2
    assert wrapped.replace("\n", " ") == text
    assert all(visible_characters(line) <= 22 for line in wrapped.splitlines())
    word = "a" * 50
    assert wrap_lines(word, max_chars=42) == word
    long_text = " ".join(["word"] * 2000)
    assert wrap_lines(long_text).replace("\n", " ") == long_text
    result = score_cue(source="source", target=word, duration_ms=1000, direction="ckb-en")
    assert "line_too_long" in result.warnings and result.flagged


def test_compression_is_conservative_protects_terms_and_flags_unresolved_speed():
    original = "Um, in order to leave we need the car immediately"
    result = compress(original, 2000, direction="ckb-en", target_cps=17)
    assert "remove_hesitation" in result.applied_rules
    assert "concise_purpose" in result.applied_rules
    assert "in order to" not in result.text
    locked = compress(
        original, 2000, direction="ckb-en", target_cps=17, protected_terms=("in order to",)
    )
    assert "in order to" in locked.text
    assert compress("Unbreakablemeaningfulword", 300, direction="ckb-en").still_above_target


def test_cps_boundaries_and_invalid_values():
    values = []
    for count in (17, 21, 22):
        values.append(
            score_cue(
                source="word",
                target="a" * count,
                duration_ms=1000,
                direction="ckb-en",
                token_logprobs=(-0.1,),
            )
        )
    assert "cps_above_target" not in values[0].warnings
    assert "cps_above_target" in values[1].warnings
    assert "cps_above_hard_cap" not in values[1].warnings
    assert "cps_above_hard_cap" in values[2].warnings
    with pytest.raises(ValueError, match="finite"):
        cps("text", float("nan"))
    with pytest.raises(ValueError, match="target CPS"):
        compress("text", 1000, direction="ckb-en", target_cps=float("nan"))


def test_qa_combines_signals_without_claiming_calibration():
    quality = score_cue(
        source="Take the car",
        target="كۆن John " + "word" * 20,
        duration_ms=500,
        direction="en-ckb",
        token_logprobs=(-1.0, -2.0),
        missing_terms=("سەیارە",),
        timing_overlap=True,
    )
    assert quality.mean_token_logprob == pytest.approx(-1.5)
    assert {
        "cps_above_hard_cap",
        "line_too_long",
        "glossary_mismatch",
        "untranslated_latin",
        "original_timing_overlap",
    }.issubset(quality.warnings)
    assert 0 <= quality.confidence_score < 70 and quality.flagged
    summary = summarize_qa(
        [
            {
                "qa_warnings": quality.warnings,
                "flagged": quality.flagged,
                "confidence_score": quality.confidence_score,
            }
        ]
    )
    assert summary["flagged_count"] == 1
    assert "Uncalibrated" in summary["confidence_note"]
