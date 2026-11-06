from pathlib import Path

import pytest

pytest.importorskip("sacrebleu")
pytest.importorskip("sentencepiece")
pytest.importorskip("regex")

import sentencepiece as spm

from ml.eval.metrics import (
    GlossaryTerm,
    compute_metrics,
    cue_cps,
    load_glossary,
    visible_characters,
)
from ml.eval.records import EvaluationRecord
from ml.train.dataset import ContextExample


def record(pair_id: str, source: str, reference: str, duration_ms: int = 1000) -> EvaluationRecord:
    example = ContextExample(pair_id, "film", source, reference, (), "human", 1.5)
    return EvaluationRecord(pair_id, "film", source, reference, duration_ms, "human", example)


@pytest.fixture(name="sentencepiece_model")
def make_sentencepiece_model(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (
            "Hello my friend, bring the car to the station.\n"
            "سڵاو هاوڕێکم سەیارەکە بۆ وێستگەکە بهێنە.\n"
            "We need to walk home before the rain starts.\n"
            "پێویستە پێش باران بگەڕێینەوە بۆ ماڵەوە.\n"
        )
        * 10,
        encoding="utf-8",
    )
    prefix = tmp_path / "spm"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=80,
        model_type="unigram",
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return prefix.with_suffix(".model")


def test_grapheme_cps_ignores_bidi_controls_and_preserves_joined_clusters():
    assert visible_characters("سَڵاو") == 4
    assert visible_characters("\u2066abc\u2069") == 3
    assert visible_characters("a\t\x00b\u2028c\n") == 3
    assert visible_characters("👨‍👩‍👧") == 1
    assert cue_cps("سَڵاو", 2000) == 2
    with pytest.raises(ValueError, match="positive duration"):
        cue_cps("text", 0)


def test_metrics_are_explicit_about_cps_glossary_and_ratios(sentencepiece_model):
    records = [
        record("a", "Bring the car now", "سەیارە بهێنە", 1000),
        record("b", "The carpet is beautiful", "ڕاخەکە جوانە", 1000),
    ]
    hypotheses = ["سەیارە بهێنە", "ڕاخەکە جوانە"]
    result = compute_metrics(
        records,
        hypotheses,
        direction="en-ckb",
        glossary=[GlossaryTerm("car", "سەیارە")],
        sentencepiece_model=sentencepiece_model,
    )
    assert result["chrfpp"] == pytest.approx(100)
    assert result["spbleu"]["score"] == pytest.approx(100)
    assert len(result["spbleu"]["sentencepiece_sha256"]) == 64
    assert result["spbleu"]["sentencepiece_version"] == spm.__version__
    assert result["comet"] is None
    assert result["glossary"]["applicable_locked_terms"] == 1  # not the substring in carpet
    assert result["glossary"]["adherence_percent"] == 100
    assert result["length_ratio_to_reference"]["median"] == 1
    assert result["cps"]["above_hard_cap_count"] == 0


def test_missing_glossary_and_cps_violations_are_not_fabricated():
    sample = record("a", "Where is the car?", "سەیارەکە لە کوێیە؟", 500)
    result = compute_metrics([sample], ["ئەم دەقە زۆر زۆر درێژە"], direction="en-ckb")
    assert result["glossary"]["provided"] is False
    assert result["glossary"]["adherence_percent"] is None
    assert result["cps"]["above_target_percent"] == 100
    assert result["cps"]["above_hard_cap_percent"] == 100
    assert result["spbleu"] is None


def test_cps_thresholds_and_length_percentile_math():
    records = [record(str(index), "source", "abcde") for index in range(3)]
    result = compute_metrics(records, ["x" * 17, "x" * 21, "x" * 22], direction="en-ckb")
    assert result["cps"]["above_target_count"] == 2
    assert result["cps"]["above_hard_cap_count"] == 1
    assert result["cps"]["above_hard_cap_percent"] == pytest.approx(100 / 3)
    ratios = result["length_ratio_to_reference"]
    assert ratios["min"] == pytest.approx(3.4)
    assert ratios["median"] == pytest.approx(4.2)
    assert ratios["p90"] == pytest.approx(4.36)
    assert ratios["max"] == pytest.approx(4.4)


def test_reverse_glossary_and_isolated_latin_term_match():
    sample = record("a", "سەیارە", "The car is here", 1000)
    result = compute_metrics(
        [sample],
        ["The \u2066car\u2069 is here"],
        direction="ckb-en",
        glossary=[GlossaryTerm("car", "سەیارە")],
    )
    assert result["glossary"]["adherence_percent"] == 100


def test_glossary_validation(tmp_path):
    glossary = tmp_path / "terms.json"
    glossary.write_text('[{"english":"Car","sorani":"سەیارە","locked":true}]', encoding="utf-8")
    assert load_glossary(glossary) == [GlossaryTerm("Car", "سەیارە", True)]
    glossary.write_text('[{"english":"Car","sorani":"سەیارە","locked":"yes"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="boolean"):
        load_glossary(glossary)
