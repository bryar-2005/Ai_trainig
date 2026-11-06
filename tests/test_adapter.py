import json
from pathlib import Path

import pytest

from ml.data.adapters.local import LocalDataError, discover_films


def make_film(root: Path) -> Path:
    film = root / "film-a"
    film.mkdir(parents=True)
    (film / "en.srt").write_text("source", encoding="utf-8")
    (film / "ckb.vtt").write_text("target", encoding="utf-8")
    return film


def test_discovers_local_files_and_metadata(tmp_path: Path):
    film = make_film(tmp_path)
    (film / "metadata.json").write_text(
        json.dumps(
            {
                "provenance": "human",
                "quality_score": 0.9,
                "license": "test",
                "attribution": "author",
            }
        ),
        encoding="utf-8",
    )
    result = discover_films(tmp_path)
    assert len(result) == 1
    assert result[0].film_id == "film-a"
    assert result[0].quality_score == 0.9
    assert result[0].source_license == "test"
    assert result[0].sorani_path.name == "ckb.vtt"


def test_missing_or_ambiguous_subtitles_are_not_silently_skipped(tmp_path: Path):
    film = make_film(tmp_path)
    (film / "en.vtt").write_text("duplicate", encoding="utf-8")
    with pytest.raises(LocalDataError, match="exactly one en"):
        discover_films(tmp_path)
    (film / "en.vtt").unlink()
    (film / "ckb.vtt").unlink()
    with pytest.raises(LocalDataError, match="exactly one ckb"):
        discover_films(tmp_path)


@pytest.mark.parametrize(
    "metadata",
    [
        {"provenance": "automated"},
        {"quality_score": 1.01},
        {"quality_score": True},
        {"quality_score": -1},
        {"license": 123},
    ],
)
def test_bad_metadata_rejected(tmp_path: Path, metadata: dict):
    film = make_film(tmp_path)
    (film / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(LocalDataError):
        discover_films(tmp_path)


def test_local_adapter_rejects_symlinked_subtitles(tmp_path: Path):
    film = make_film(tmp_path)
    (film / "ckb.vtt").unlink()
    (film / "ckb.vtt").symlink_to(film / "en.srt")
    with pytest.raises(LocalDataError, match="symlinks"):
        discover_films(tmp_path)
