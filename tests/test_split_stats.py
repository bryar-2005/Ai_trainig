import pytest

from ml.data.split import sample_gold_for_review, split_by_film
from ml.data.stats import dataset_statistics, markdown_report


def row(film: str, index: int, provenance: str = "human") -> dict:
    return {
        "film_id": film,
        "pair_id": f"{film}-{index}",
        "english": f"hello word{index}",
        "sorani": f"سڵاو وشە{index}",
        "provenance": provenance,
        "english_start_ms": index * 1000,
    }


def test_film_disjoint_splits_are_deterministic_and_all_nonempty():
    rows = [row(f"film-{film}", line) for film in range(10) for line in range(3)]
    splits = split_by_film(rows, seed=19)
    assert splits == split_by_film(rows, seed=19)
    ids = [{r["film_id"] for r in splits[name]} for name in ("train", "val", "test")]
    assert all(ids)
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == {r["film_id"] for r in rows}
    assert [len(group) for group in ids] == [8, 1, 1]


def test_insufficient_films_fail_without_explicit_development_override():
    rows = [row("one", 1), row("two", 1)]
    with pytest.raises(ValueError, match="at least 3"):
        split_by_film(rows)
    splits = split_by_film(rows, allow_empty_splits=True)
    assert len(splits["train"]) == len(splits["test"]) == 1
    assert splits["val"] == []


def test_gold_sample_is_stratified_unverified_and_not_padded():
    rows = [row("large", n) for n in range(20)] + [row("small", 1, "machine")]
    gold = sample_gold_for_review(rows, size=5, seed=9)
    assert len(gold) == 5
    assert {r["film_id"] for r in gold} == {"large", "small"}
    assert all(r["review_status"] == "unverified" for r in gold)
    assert gold == sample_gold_for_review(rows, size=5, seed=9)
    assert len(sample_gold_for_review(rows, size=1000)) == len(rows)


def test_stats_report_correct_oov_definition_and_no_missing_empty_split():
    splits = {"train": [row("a", 1)], "val": [row("b", 2)], "test": []}
    stats = dataset_statistics(
        splits, diagnostics={}, gold_count=0, gold_requested=1000, warnings=[]
    )
    assert stats["pair_count"] == 2
    assert stats["splits"]["val"]["token_oov_vs_train_vocabulary"]["english"]["rate"] == 0.5
    assert stats["splits"]["test"]["token_oov_vs_train_vocabulary"]["english"]["rate"] is None
    assert "not NLLB tokenizer" in stats["oov_definition"]
    assert "Subtitle dataset statistics" in markdown_report(stats)
