import os
from pathlib import Path

import pytest

from ml.data.build_dataset import BuildConfig, build_dataset


def test_symlinked_marker_does_not_authorize_overwrite(tmp_path: Path, local_films: Path):
    output = tmp_path / "precious"
    output.mkdir()
    marker = tmp_path / "marker"
    marker.write_text("sorani-subtitle-dataset-v1\n", encoding="utf-8")
    (output / ".sorani-dataset-output").symlink_to(marker)
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unmarked"):
        build_dataset(
            BuildConfig(
                input_dir=local_films,
                output_dir=output,
                gold_path=tmp_path / "gold.jsonl",
                skip_language_id=True,
                overwrite=True,
            )
        )
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_failed_gold_install_rolls_back_existing_dataset(
    tmp_path: Path, local_films: Path, monkeypatch: pytest.MonkeyPatch
):
    output, gold = tmp_path / "processed", tmp_path / "gold.jsonl"
    common = dict(input_dir=local_films, output_dir=output, gold_path=gold, skip_language_id=True)
    build_dataset(BuildConfig(**common))
    original_stats = (output / "stats.json").read_bytes()
    original_gold = gold.read_bytes()
    real_replace = os.replace

    def fail_gold_only(src, dst):
        if Path(dst) == gold:
            raise OSError("simulated install failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_gold_only)
    with pytest.raises(OSError, match="simulated install failure"):
        build_dataset(BuildConfig(**common, overwrite=True))
    assert (output / "stats.json").read_bytes() == original_stats
    assert gold.read_bytes() == original_gold
    assert not list(tmp_path.glob(".gold-*"))
    assert not list(tmp_path.glob(".processed.build-*"))
