from pathlib import Path

import fasttext
import pytest

from ml.data.filter_dedupe import (
    FastTextLanguageIdentifier,
    arabic_script_ratio,
    deduplicate_pairs,
    filter_pair,
)
from tests.conftest import LID_PAIRS, PERSIAN_TEXT


def test_script_ratio_ignores_punctuation_and_digits_but_not_latin_letters():
    assert arabic_script_ratio("سڵاو، 123!") == 1.0
    assert arabic_script_ratio("سڵاو A") == 0.8
    assert arabic_script_ratio("123 !") == 0.0


@pytest.mark.parametrize(
    ("english", "sorani", "reason"),
    [
        ("", "سڵاو", "empty_text"),
        ("hello", "hello", "identical_text"),
        ("this is a very long line of dialogue", "بە", "length_ratio"),
        ("hello world", "hello سڵاو", "sorani_script_ratio"),
    ],
)
def test_bad_pairs_are_rejected(english: str, sorani: str, reason: str):
    result = filter_pair(english, sorani, detector=None)
    assert not result.accepted
    assert result.reason == reason


def test_script_only_mode_is_explicitly_unverified():
    result = filter_pair("hello friend", "سڵاو هاوڕێ", detector=None)
    assert result.accepted
    assert result.english_language is None
    assert result.sorani_language is None


def test_real_fasttext_verifies_english_and_sorani_and_rejects_persian(fasttext_model: Path):
    detector = FastTextLanguageIdentifier(fasttext_model, min_confidence=0.3)
    english, sorani = LID_PAIRS[0]
    good = filter_pair(english, sorani, detector=detector)
    assert good.accepted
    assert good.english_language.label == "en"
    assert good.sorani_language.label == "ckb"
    wrong = filter_pair(english, PERSIAN_TEXT, detector=detector)
    assert not wrong.accepted
    assert wrong.reason == "sorani_language_id"


def test_generic_ku_is_not_misrepresented_as_sorani(tmp_path: Path):
    training = tmp_path / "data.txt"
    training.write_text(
        "__label__en hello my friend\n__label__ku slaw heval\n" * 10, encoding="utf-8"
    )
    model = fasttext.train_supervised(input=str(training), epoch=5, thread=1, verbose=0)
    path = tmp_path / "lid.bin"
    model.save_model(str(path))
    with pytest.raises(ValueError, match="generic ku"):
        FastTextLanguageIdentifier(path)


def make_row(film: str, en: str, ckb: str, provenance: str, identifier: str) -> dict:
    return {
        "film_id": film,
        "english": en,
        "sorani": ckb,
        "provenance": provenance,
        "quality_score": 0.8,
        "english_start_ms": 0,
        "pair_id": identifier,
    }


def test_minhash_deduplicates_both_languages_and_prefers_human():
    english = "Could you please open the window and close the wooden door for me."
    sorani = "تکایە دەتوانیت پەنجەرەکە بکەیتەوە و دەرگای دارین دابخەیت بۆ من."
    machine = make_row("a", english, sorani, "machine", "1")
    human = make_row("b", english + "!", sorani, "human", "2")
    kept, removed = deduplicate_pairs([machine, human])
    assert removed == 1
    assert kept == [human]


def test_distinct_translations_are_not_removed_for_same_english():
    english = "Please bring the books to the table."
    rows = [
        make_row("a", english, "تکایە کتێبەکان بۆ سەر مێزەکە بهێنە", "human", "1"),
        make_row("b", english, "تکایە دەفتەرەکان لە ژوورەکەدا دابنێ", "human", "2"),
    ]
    kept, removed = deduplicate_pairs(rows)
    assert len(kept) == 2
    assert removed == 0


def test_exact_duplicates_are_always_removed_case_insensitively():
    rows = [
        make_row("a", "Hello friend", "سڵاو هاوڕێ", "unknown", "1"),
        make_row("b", "hello   friend", "سڵاو هاوڕێ", "unknown", "2"),
    ]
    kept, removed = deduplicate_pairs(rows)
    assert len(kept) == 1
    assert removed == 1
