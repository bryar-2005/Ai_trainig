"""Synthetic-only tests for deterministic, bounded overlap candidate retrieval."""

import random
import string

import pytest

from ml.data.overlap_matching import (
    AuditPair,
    OverlapLimitError,
    ReferenceIndex,
    WorkBudget,
    _near_score,
    canonical_text,
    prepare_query,
)


def pair(pair_id: str, english: str, sorani: str) -> AuditPair:
    return AuditPair(pair_id, canonical_text(english), canonical_text(sorani))


def signals(index: ReferenceIndex, value: AuditPair, budget: WorkBudget):
    return {
        reference.pair_id: match
        for reference, match in index.find(prepare_query(value, budget), budget)
    }


def test_exact_single_side_pair_id_and_swapped_signals_do_not_discard_duplicates():
    budget = WorkBudget()
    references = [
        pair("one", "HELLO   FRIEND", "سڵاو هاوڕێ"),
        pair("two", "hello friend", "وشەی تر"),
        pair("three", "سڵاو هاوڕێ", "hello friend"),
        pair("same-id", "completely different", "وشەیەکی جیاواز"),
    ]
    index = ReferenceIndex(references, budget)
    found = signals(index, pair("same-id", "Hello friend", "سڵاو هاوڕێ"), budget)
    assert found["one"].signals >= {"english_exact", "sorani_exact", "bilingual_exact"}
    assert found["two"].signals == {"english_exact"}
    assert found["three"].signals == {"swapped_bilingual_exact"}
    assert found["same-id"].signals == {"same_pair_id"}
    assert found["one"].similarities == {"english": 1.0, "sorani": 1.0}


def test_normalization_preserves_zwnj_and_ambiguous_heh():
    assert canonical_text("  ＡＢＣ\tDEF  ") == "abc def"
    assert canonical_text("ئه\u200cم") == "ئه\u200cم"
    assert canonical_text("ئه\u200cم") != canonical_text("ئهم")
    assert canonical_text("ه") != canonical_text("ە")
    assert canonical_text("hello?") != canonical_text("hello!")


def test_short_text_only_gets_exact_signals_and_near_is_per_language():
    budget = WorkBudget()
    index = ReferenceIndex(
        [
            pair("short", "abcdefghijk", "دەق"),
            pair("long", "The garden needs water!", "باخچەکە پێویستی بە ئاوە."),
        ],
        budget,
    )
    assert "short" not in signals(index, pair("q", "abcdefghijkx", "تر"), budget)
    result = signals(
        index, pair("q2", "The garden needs water.", "باخچەکە پێویستی بە ئاوە."), budget
    )
    assert result["long"].signals == {"english_near", "sorani_exact", "bilingual_near"}
    assert 0.9 <= result["long"].similarities["english"] < 1


def test_jaccard_threshold_is_inclusive_and_uses_integer_comparison():
    left = frozenset(str(i) for i in range(9))
    right = frozenset(str(i) for i in range(10))
    assert _near_score(left, right) == 0.9
    assert _near_score(left, right | {"eleven"}) is None


def _trigrams(text: str) -> set[str]:
    return {text[index : index + 3] for index in range(len(text) - 2)}


def _oracle(left: str, right: str) -> bool:
    if len(left) < 12 or len(right) < 12 or left == right:
        return False
    a, b = _trigrams(left), _trigrams(right)
    return 10 * len(a & b) >= 9 * len(a | b)


def test_prefix_index_matches_exhaustive_jaccard_oracle_on_random_fixtures():
    rng = random.Random(2781)
    alphabet = string.ascii_lowercase + "    " + "ەک\u200c"
    texts = ["".join(rng.choice(alphabet) for _ in range(rng.randint(12, 55))) for _ in range(180)]
    # Include one-character edits, repeats, a ZWNJ, and a short non-near input.
    queries = [text[:-1] + rng.choice(alphabet) for text in texts[:90]]
    texts += ["aaaaaaaaaaaa", "text\u200cwith-zwnj", "short"]
    queries += ["aaaaaaaaaaaaa", "textwith-zwnj", "shorter"]
    references = [pair(f"ref-{i:04}", text, f"بەشی تایبەت {i}") for i, text in enumerate(texts)]
    budget = WorkBudget()
    index = ReferenceIndex(references, budget)
    for query_text in queries:
        query = pair("query", query_text, "شێوەی تر")
        result = signals(index, query, budget)
        actual = {key for key, value in result.items() if "english_near" in value.signals}
        expected = {ref.pair_id for ref in references if _oracle(query.english, ref.english)}
        assert actual == expected


def test_work_limits_abort_instead_of_dropping_candidates():
    with pytest.raises(OverlapLimitError, match="reference windows"):
        ReferenceIndex(
            [pair("one", "Some long English text.", "دەقێکی زۆر درێژ")],
            WorkBudget(max_reference_windows=1),
        )
    budget = WorkBudget(max_posting_visits=0)
    index = ReferenceIndex([pair("one", "short", "کورت")], budget)
    with pytest.raises(OverlapLimitError, match="posting visits"):
        signals(index, pair("two", "short", "تر"), budget)
