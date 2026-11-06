from pathlib import Path

from datasets import load_from_disk

from ml.data.build_dataset import BuildConfig, build_dataset


def test_merged_pair_preserves_each_original_cue_and_structured_metadata(tmp_path: Path):
    film = tmp_path / "raw" / "film"
    film.mkdir(parents=True)
    (film / "en.srt").write_text(
        "1\n00:00:01,000 --> 00:00:05,000\n"
        "JOHN: Please take this and bring it back. [DOOR OPENS]\n",
        encoding="utf-8",
    )
    (film / "ckb.srt").write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nتکایە ئەمە ببە و\n\n"
        "2\n00:00:03,000 --> 00:00:05,000\nدووبارە بیگەڕێنەوە.\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    build_dataset(
        BuildConfig(
            input_dir=film.parent,
            output_dir=output,
            gold_path=tmp_path / "gold.jsonl",
            skip_language_id=True,
            allow_empty_splits=True,
        )
    )
    record = load_from_disk(str(output / "dataset"))["train"][0]
    assert record["english_cue_ids"] == [1]
    assert record["sorani_cue_ids"] == [1, 2]
    assert record["sorani_cue_starts_ms"] == [1000, 3000]
    assert record["sorani_cue_ends_ms"] == [3000, 5000]
    assert record["sorani_cue_texts"] == ["تکایە ئەمە ببە و", "دووبارە بیگەڕێنەوە."]
    assert record["english_cue_speaker_labels"] == [["JOHN"]]
    assert record["english_cue_sound_effects"] == [["DOOR OPENS"]]
    assert record["english_cue_styles"] == [None]
