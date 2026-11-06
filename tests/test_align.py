import pytest

from ml.data.align import align_cues, temporal_iou
from ml.data.loaders import Cue


def cue(index: int, start: int, end: int) -> Cue:
    return Cue(index=index, start_ms=start, end_ms=end, text=f"line {index}")


def test_iou_math_and_invalid_intervals():
    assert temporal_iou(0, 1000, 0, 2000) == 0.5
    assert temporal_iou(0, 1000, 1000, 2000) == 0
    with pytest.raises(ValueError):
        temporal_iou(0, 0, 0, 1000)


def test_one_to_one_threshold_inclusive_and_timing_unmodified():
    english = [cue(1, 0, 1000)]
    sorani = [cue(1, 0, 500)]
    result = align_cues(english, sorani)
    assert len(result) == 1
    assert result[0].temporal_iou == 0.5
    assert result[0].english_cues[0].end_ms == 1000
    assert result[0].sorani_cues[0].end_ms == 500


def test_tolerance_does_not_fabricate_iou_or_shift_timestamps():
    assert align_cues([cue(1, 0, 400)], [cue(1, 300, 700)]) == []
    assert align_cues([cue(1, 0, 2000)], [cue(1, 600, 2000)]) == []


def test_one_to_many_aligns_merged_span():
    result = align_cues([cue(1, 0, 2000)], [cue(10, 0, 1000), cue(11, 1000, 2000)])
    assert len(result) == 1
    assert [c.index for c in result[0].sorani_cues] == [10, 11]
    assert result[0].temporal_iou == 1.0


def test_many_to_one_aligns_merged_span():
    result = align_cues([cue(1, 0, 900), cue(2, 900, 2000)], [cue(9, 0, 2000)])
    assert len(result) == 1
    assert [c.index for c in result[0].english_cues] == [1, 2]
    assert result[0].temporal_iou == 1.0


def test_global_matching_is_ordered_and_never_reuses_cues():
    english = [cue(1, 0, 1000), cue(2, 1000, 2000), cue(3, 2000, 3000)]
    sorani = [cue(8, 0, 1000), cue(9, 1000, 2000), cue(10, 2000, 3000)]
    result = align_cues(english, sorani)
    assert [(p.english_cues[0].index, p.sorani_cues[0].index) for p in result] == [
        (1, 8),
        (2, 9),
        (3, 10),
    ]
    assert all(len(p.english_cues) == len(p.sorani_cues) == 1 for p in result)


def test_global_search_can_skip_an_early_weaker_match():
    english = [cue(1, 1800, 2500), cue(2, 1900, 3100), cue(3, 2100, 2600)]
    sorani = [cue(1, 1100, 2100), cue(2, 1300, 2000), cue(3, 2100, 2600)]
    result = align_cues(english, sorani, max_group_size=1)
    assert len(result) == 1
    assert result[0].english_cues[0].index == 3
    assert result[0].sorani_cues[0].index == 3
    assert result[0].temporal_iou == 1.0


def test_candidate_limit_and_duplicate_indices_are_rejected():
    with pytest.raises(ValueError, match="Too many"):
        align_cues([cue(1, 0, 1000)], [cue(1, 0, 1000), cue(2, 0, 1000)], max_candidates=1)
    with pytest.raises(ValueError, match="Duplicate"):
        align_cues([cue(1, 0, 1000), cue(1, 1000, 2000)], [cue(1, 0, 1000)])
