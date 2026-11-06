"""Discover explicitly supplied, local per-film bilingual subtitle folders."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ml.data.loaders import SUPPORTED_EXTENSIONS

PROVENANCES = frozenset({"human", "machine", "unknown"})


class LocalDataError(ValueError):
    """The local directory or its metadata does not follow the documented layout."""


@dataclass(frozen=True, slots=True)
class FilmFiles:
    film_id: str
    english_path: Path
    sorani_path: Path
    provenance: str = "unknown"
    quality_score: float | None = None
    source_license: str | None = None
    attribution: str | None = None


def _read_metadata(film_dir: Path) -> dict:
    path = film_dir / "metadata.json"
    if path.is_symlink():
        raise LocalDataError(f"{path}: metadata may not be a symlink")
    if not path.exists():
        return {}
    try:
        metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalDataError(f"Invalid metadata in {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise LocalDataError(f"{path}: expected a JSON object")
    return metadata


def discover_films(root: str | Path) -> list[FilmFiles]:
    """Read ``root/<film_id>/{en,ckb}.<format>``; never crawl or download.

    A film must have exactly one file for each language. Child symlinks are
    rejected so a data directory cannot silently include files outside it.
    """
    root = Path(root)
    if not root.is_dir():
        raise LocalDataError(f"Input directory does not exist or is not a directory: {root}")
    films: list[FilmFiles] = []
    for film_dir in sorted(root.iterdir(), key=lambda item: item.name):
        if film_dir.name.startswith("."):
            continue
        if film_dir.is_symlink():
            raise LocalDataError(f"{film_dir}: film folders may not be symlinks")
        if not film_dir.is_dir():
            continue
        by_language: dict[str, list[Path]] = {"en": [], "ckb": []}
        for item in sorted(film_dir.iterdir(), key=lambda path: path.name):
            language = item.stem.casefold()
            if language in by_language and item.suffix.casefold() in SUPPORTED_EXTENSIONS:
                if item.is_symlink() or not item.is_file():
                    raise LocalDataError(
                        f"{item}: subtitle inputs must be regular files, not symlinks"
                    )
                by_language[language].append(item)
        for language, paths in by_language.items():
            if len(paths) != 1:
                raise LocalDataError(
                    f"{film_dir}: expected exactly one {language}.srt/.vtt/.ass/.ssa file; "
                    f"found {len(paths)}"
                )
        metadata = _read_metadata(film_dir)
        provenance = metadata.get("provenance", "unknown")
        if not isinstance(provenance, str) or provenance not in PROVENANCES:
            raise LocalDataError(f"{film_dir}: provenance must be human, machine, or unknown")
        quality = metadata.get("quality_score")
        if quality is not None and (
            isinstance(quality, bool)
            or not isinstance(quality, int | float)
            or not math.isfinite(quality)
            or not 0 <= quality <= 1
        ):
            raise LocalDataError(f"{film_dir}: quality_score must be a number between 0 and 1")
        for key in ("license", "attribution"):
            if metadata.get(key) is not None and not isinstance(metadata[key], str):
                raise LocalDataError(f"{film_dir}: {key} must be a string or null")
        films.append(
            FilmFiles(
                film_id=film_dir.name,
                english_path=by_language["en"][0],
                sorani_path=by_language["ckb"][0],
                provenance=provenance,
                quality_score=float(quality) if quality is not None else None,
                source_license=metadata.get("license"),
                attribution=metadata.get("attribution"),
            )
        )
    if not films:
        raise LocalDataError(f"No film directories found in {root}")
    return films
