import json
import subprocess
import sys
from pathlib import Path

import pytest
from datasets import DatasetDict, load_from_disk

from ml.data.build_dataset import BuildConfig, build_dataset
from tests.conftest import LID_PAIRS, write_srt


def test_cli_end_to_end_saves_dataset_reports_and_held_out_review_sample(
    tmp_path: Path, local_films: Path
):
    output = tmp_path / "processed"
    gold = tmp_path / "gold" / "gold_test.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "ml.data.build_dataset",
        "--input-dir",
        str(local_films),
        "--output-dir",
        str(output),
        "--gold-path",
        str(gold),
        "--skip-language-id",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    summary = json.loads(result.stdout)
    assert summary["pair_count"] == 12
    assert summary["film_count"] == 6
    assert summary["split_counts"] == {"train": 8, "val": 2, "test": 2}
    assert summary["gold_written"] == 2
    assert any("skipped" in warning for warning in summary["warnings"])
    dataset = load_from_disk(str(output / "dataset"))
    assert isinstance(dataset, DatasetDict)
    assert set(dataset) == {"train", "val", "test"}
    film_sets = [{r["film_id"] for r in dataset[name]} for name in ("train", "val", "test")]
    assert not (
        film_sets[0] & film_sets[1] or film_sets[0] & film_sets[2] or film_sets[1] & film_sets[2]
    )
    sample = dataset["train"][0]
    assert sample["english_cue_ids"] and sample["sorani_cue_ids"]
    assert sample["english_speakers"] == ["JOHN"]
    assert sample["english_cue_speaker_labels"] == [["JOHN"]]
    assert sample["english_cue_texts"] == [sample["english"]]
    assert sample["english_cue_starts_ms"] == [sample["english_start_ms"]]
    assert sample["temporal_iou"] == 1.0
    assert sample["quality_score"] == 0.9
    assert sample["quality_source"] == "metadata"
    assert sample["language_verified"] is False
    assert sample["sorani_lid_label"] is None
    gold_rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    assert {r["pair_id"] for r in gold_rows} <= {r["pair_id"] for r in dataset["test"]}
    assert all(row["review_status"] == "unverified" for row in gold_rows)
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    assert stats["gold_review_sample"]["source_split"] == "test"
    assert stats["diagnostics"]["inputs"][0]["english_sha256"]
    assert "OOV" in (output / "stats.md").read_text(encoding="utf-8")

    old_stats = (output / "stats.json").read_text(encoding="utf-8")
    old_gold = gold.read_text(encoding="utf-8")
    without_overwrite = subprocess.run(cmd, text=True, capture_output=True)
    assert without_overwrite.returncode == 2
    assert "already exists" in without_overwrite.stderr
    subprocess.run([*cmd, "--overwrite"], text=True, capture_output=True, check=True)
    assert old_stats == (output / "stats.json").read_text(encoding="utf-8")
    assert old_gold == gold.read_text(encoding="utf-8")


def test_actual_fasttext_lid_is_used_in_a_full_build(tmp_path: Path, fasttext_model: Path):
    root = tmp_path / "raw"
    root.mkdir()
    for index, (english, sorani) in enumerate(LID_PAIRS):
        directory = root / f"film-{index}"
        directory.mkdir()
        write_srt(directory / "en.srt", [english])
        write_srt(directory / "ckb.srt", [sorani])
    output = tmp_path / "processed"
    stats = build_dataset(
        BuildConfig(
            input_dir=root,
            output_dir=output,
            gold_path=tmp_path / "gold.jsonl",
            lid_model=fasttext_model,
        )
    )
    assert stats["pair_count"] == 3
    assert stats["diagnostics"]["language_id"] == "fasttext"
    dataset = load_from_disk(str(output / "dataset"))
    for split in dataset.values():
        for item in split:
            assert item["language_verified"] is True
            assert item["english_lid_label"] == "en"
            assert item["sorani_lid_label"] == "ckb"
            assert item["quality_source"] == "temporal_iou"


def test_empty_result_reports_filter_reasons_without_writing_outputs(tmp_path: Path):
    film = tmp_path / "raw" / "film"
    film.mkdir(parents=True)
    write_srt(film / "en.srt", ["hello friend"])
    write_srt(film / "ckb.srt", ["hello friend"])
    output, gold = tmp_path / "out", tmp_path / "gold.jsonl"
    with pytest.raises(ValueError, match="identical_text"):
        build_dataset(
            BuildConfig(
                input_dir=film.parent,
                output_dir=output,
                gold_path=gold,
                skip_language_id=True,
            )
        )
    assert not output.exists()
    assert not gold.exists()


def test_unmarked_output_cannot_be_overwritten(tmp_path: Path, local_films: Path):
    output = tmp_path / "precious"
    output.mkdir()
    (output / "keep.txt").write_text("do not remove", encoding="utf-8")
    config = BuildConfig(
        input_dir=local_films,
        output_dir=output,
        gold_path=tmp_path / "gold.jsonl",
        skip_language_id=True,
        overwrite=True,
    )
    with pytest.raises(ValueError, match="unmarked"):
        build_dataset(config)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "do not remove"


def test_output_cannot_overlap_input_or_silently_skip_lid(tmp_path: Path, local_films: Path):
    with pytest.raises(ValueError, match="exactly one"):
        build_dataset(
            BuildConfig(
                input_dir=local_films, output_dir=tmp_path / "out", gold_path=tmp_path / "g.jsonl"
            )
        )
    with pytest.raises(ValueError, match="must not contain"):
        build_dataset(
            BuildConfig(
                input_dir=local_films,
                output_dir=local_films / "out",
                gold_path=tmp_path / "g.jsonl",
                skip_language_id=True,
            )
        )
    assert not (local_films / "out").exists()


def test_insufficient_films_fail_unless_development_override(tmp_path: Path, local_films: Path):
    root = tmp_path / "one-film"
    root.mkdir()
    original = next(local_films.iterdir())
    import shutil

    shutil.copytree(original, root / original.name)
    config = BuildConfig(
        input_dir=root,
        output_dir=tmp_path / "out",
        gold_path=tmp_path / "gold.jsonl",
        skip_language_id=True,
    )
    with pytest.raises(ValueError, match="at least 3"):
        build_dataset(config)
    assert not config.output_dir.exists()
    stats = build_dataset(
        BuildConfig(
            input_dir=root,
            output_dir=config.output_dir,
            gold_path=config.gold_path,
            skip_language_id=True,
            allow_empty_splits=True,
        )
    )
    assert stats["splits"]["train"]["pair_count"] == 2
    assert stats["splits"]["test"]["pair_count"] == 0
    assert config.gold_path.read_text(encoding="utf-8") == ""
    assert any("empty" in warning for warning in stats["warnings"])
