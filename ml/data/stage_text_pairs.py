"""Stage local English–Sorani line pairs for review, not model training.

This offline importer reads two explicitly supplied UTF-8 text files. It does
not download corpora, infer film boundaries, verify rights or language, create
train/validation/test splits, or start any model job.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.data.normalize import normalization_warnings, normalize_english, normalize_sorani

_MAX_MANIFEST_BYTES = 128 * 1024
_COMMERCIAL_USES = frozenset({"commercial_training", "commercial_validation", "commercial_testing"})
_PROVENANCES = frozenset({"human", "machine", "unknown"})
_OUTPUT_FILES = ("pairs.jsonl", "quarantine.jsonl", "source-manifest.json", "staging-report.json")


class TextPairStagingError(ValueError):
    """Inputs are unsafe, inconsistent, or not authorized for this staging workflow."""


@dataclass(frozen=True, slots=True)
class TextPairStagingConfig:
    english_file: Path
    sorani_file: Path
    source_manifest: Path
    output_dir: Path
    max_chars: int = 5000


@dataclass(frozen=True, slots=True)
class _FileInfo:
    sha256: str
    lines: int
    bytes: int


def _required_text(values: dict[str, Any], key: str, location: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise TextPairStagingError(
            f"{location}.{key} must be a nonempty string of at most 2048 characters"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TextPairStagingError(f"{location}.{key} must contain valid Unicode text") from exc
    return value.strip()


def _unique_json_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise TextPairStagingError("Source manifest contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise TextPairStagingError("Source manifest may not contain non-finite JSON numbers")


def _read_manifest(path: Path) -> tuple[dict[str, Any], str, bytes]:
    with path.open("rb") as handle:
        raw = handle.read(_MAX_MANIFEST_BYTES + 1)
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise TextPairStagingError("Source manifest is too large")
    try:
        document = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TextPairStagingError("Source manifest must be a UTF-8 JSON object") from exc
    if not isinstance(document, dict):
        raise TextPairStagingError("Source manifest must be a JSON object")
    provenance = document.get("provenance")
    if not isinstance(provenance, str) or provenance not in _PROVENANCES:
        raise TextPairStagingError("manifest.provenance must be human, machine, or unknown")
    rights = document.get("rights")
    if not isinstance(rights, dict):
        raise TextPairStagingError("manifest.rights must be an object")
    status = rights.get("status")
    if not isinstance(status, str) or status not in {"synthetic", "user_attested"}:
        raise TextPairStagingError("rights.status must be synthetic or user_attested")
    if rights.get("affirmed_by_user") is not True:
        raise TextPairStagingError("rights.affirmed_by_user must be true")
    uses = rights.get("permitted_uses")
    if (
        not isinstance(uses, list)
        or not uses
        or not all(isinstance(use, str) for use in uses)
        or len(set(uses)) != len(uses)
    ):
        raise TextPairStagingError("rights.permitted_uses must be a nonempty list of distinct uses")
    allowed = {"synthetic_testing"} if status == "synthetic" else _COMMERCIAL_USES
    if not set(uses) <= allowed:
        raise TextPairStagingError(
            "rights.permitted_uses are incompatible with the declared status"
        )
    # These are user-supplied claims and references, not legal verification.
    source = {
        "source_id": _required_text(document, "source_id", "manifest"),
        "source_revision": _required_text(document, "source_revision", "manifest"),
        "source_reference": _required_text(document, "source_reference", "manifest"),
        "provenance": provenance,
        "rights": {
            "status": status,
            "affirmed_by_user": True,
            "original_text_evidence": _required_text(rights, "original_text_evidence", "rights"),
            "translation_evidence": _required_text(rights, "translation_evidence", "rights"),
            "license_or_permission": _required_text(rights, "license_or_permission", "rights"),
            "attribution": _required_text(rights, "attribution", "rights"),
            "permitted_uses": list(uses),
        },
    }
    return source, hashlib.sha256(raw).hexdigest(), raw


def _paths(config: TextPairStagingConfig) -> tuple[Path, Path, Path, Path]:
    if isinstance(config.max_chars, bool) or not isinstance(config.max_chars, int):
        raise TextPairStagingError("max_chars must be an integer between 1 and 100000")
    if not 1 <= config.max_chars <= 100_000:
        raise TextPairStagingError("max_chars must be an integer between 1 and 100000")
    inputs: list[Path] = []
    for name, supplied in (
        ("English file", config.english_file),
        ("Sorani file", config.sorani_file),
        ("source manifest", config.source_manifest),
    ):
        path = Path(supplied).expanduser()
        if path.is_symlink() or not path.is_file():
            raise TextPairStagingError(f"{name} must be an existing regular file, not a symlink")
        inputs.append(path.resolve())
    for index, left in enumerate(inputs):
        for right in inputs[index + 1 :]:
            if os.path.samefile(left, right):
                raise TextPairStagingError("English, Sorani and manifest must be distinct files")
    output_path = Path(config.output_dir).expanduser()
    if output_path.is_symlink() or output_path.exists():
        raise TextPairStagingError("Output path already exists; choose a new directory")
    output = output_path.resolve()
    if any(path == output or path.is_relative_to(output) for path in inputs):
        raise TextPairStagingError("Output must not contain an input file")
    return inputs[0], inputs[1], inputs[2], output


def _file_info(path: Path) -> _FileInfo:
    """Count LF/CRLF physical lines and hash bytes without loading the file."""
    digest = hashlib.sha256()
    total = newlines = 0
    last = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
            newlines += chunk.count(b"\n")
            last = chunk[-1:]
    lines = newlines + int(total > 0 and last != b"\n")
    return _FileInfo(digest.hexdigest(), lines, total)


def _identity(*values: str | int) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode_line(
    raw: bytes, *, first: bool, language: str, max_chars: int
) -> tuple[str | None, list[str]]:
    if raw.endswith(b"\n"):
        raw = raw[:-1]
        if raw.endswith(b"\r"):
            raw = raw[:-1]
    try:
        text = raw.decode("utf-8-sig" if first else "utf-8", errors="strict")
    except UnicodeDecodeError:
        return None, [f"{language}_invalid_utf8"]
    return text, _validate_decoded_text(text, language=language, max_chars=max_chars)


def _validate_decoded_text(
    text: str, *, language: str, max_chars: int, allow_multiline: bool = False
) -> list[str]:
    """Validate a decoded side without discarding its mate or changing its text."""
    reasons = []
    if not text.strip():
        reasons.append(f"{language}_blank")
    allowed_controls = "\t\n\r" if allow_multiline else "\t"
    if any(
        (unicodedata.category(char) == "Cc" and char not in allowed_controls)
        or char in "\u2028\u2029"
        for char in text
    ):
        reasons.append(f"{language}_control_or_line_separator")
    if len(text) > max_chars:
        reasons.append(f"{language}_too_long")
    return reasons


def _has_letter_in_script(text: str, script: str) -> bool:
    return any(
        unicodedata.category(char).startswith("L") and script in unicodedata.name(char, "")
        for char in text
    )


def _write_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _pair_flags(
    english: str, sorani: str, first_occurrence: dict[str, int], position: int
) -> tuple[list[str], int | None]:
    """Flag review concerns without treating them as language or alignment checks."""
    flags: list[str] = []
    if not _has_letter_in_script(english, "LATIN"):
        flags.append("english_without_latin_letters")
    if not _has_letter_in_script(sorani, "ARABIC"):
        flags.append("sorani_without_arabic_letters")
    if " ".join(english.split()).casefold() == " ".join(sorani.split()).casefold():
        flags.append("identical_text")
    if max(len(english), len(sorani)) / min(len(english), len(sorani)) > 5:
        flags.append("unusual_length_ratio")
    flags.extend(normalization_warnings(sorani))
    duplicate_key = _identity(
        " ".join(english.split()).casefold(), " ".join(sorani.split()).casefold()
    )
    duplicate_of = first_occurrence.get(duplicate_key)
    if duplicate_of is not None:
        flags.append("duplicate_pair")
    else:
        first_occurrence[duplicate_key] = position
    return flags, duplicate_of


def _process(
    english_file: Path,
    sorani_file: Path,
    english_info: _FileInfo,
    sorani_info: _FileInfo,
    source: dict[str, Any],
    stage: Path,
    *,
    max_chars: int,
) -> tuple[dict[str, Any], str]:
    # Unknown document boundaries must not become separate splits merely because
    # the same collection was sharded or released again with different bytes.
    group_id = _identity("text-pair-collection-v1", source["source_id"])
    english_digest, sorani_digest = hashlib.sha256(), hashlib.sha256()
    reasons: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    first_occurrence: dict[str, int] = {}
    processed = staged = quarantined = 0
    with (
        english_file.open("rb") as english_handle,
        sorani_file.open("rb") as sorani_handle,
        (stage / "pairs.jsonl").open("w", encoding="utf-8", newline="\n") as pairs,
        (stage / "quarantine.jsonl").open("w", encoding="utf-8", newline="\n") as quarantine,
    ):
        for line_number, (en_raw, ckb_raw) in enumerate(
            itertools.zip_longest(english_handle, sorani_handle), 1
        ):
            if en_raw is None or ckb_raw is None:
                raise TextPairStagingError("An input file changed after the line-count check")
            processed += 1
            english_digest.update(en_raw)
            sorani_digest.update(ckb_raw)
            pair_id = _identity(
                "text-pair-v1",
                group_id,
                source["source_revision"],
                english_info.sha256,
                sorani_info.sha256,
                line_number,
            )
            original_en, en_reasons = _decode_line(
                en_raw, first=line_number == 1, language="english", max_chars=max_chars
            )
            original_ckb, ckb_reasons = _decode_line(
                ckb_raw, first=line_number == 1, language="sorani", max_chars=max_chars
            )
            rejected = en_reasons + ckb_reasons
            english = sorani = ""
            if not rejected:
                assert original_en is not None and original_ckb is not None
                english = normalize_english(original_en)
                sorani = normalize_sorani(original_ckb, heh_policy="preserve")
                for label, text in (("english", english), ("sorani", sorani)):
                    if not text.strip():
                        rejected.append(f"{label}_blank_after_normalization")
                    if len(text) > max_chars:
                        rejected.append(f"{label}_too_long_after_normalization")
            if rejected:
                quarantined += 1
                reasons.update(rejected)
                _write_jsonl(
                    quarantine,
                    {"pair_id": pair_id, "source_line": line_number, "reasons": rejected},
                )
                continue
            flags, duplicate_of_line = _pair_flags(english, sorani, first_occurrence, line_number)
            warnings.update(flags)
            staged += 1
            _write_jsonl(
                pairs,
                {
                    "record_type": "text_pair",
                    "pair_id": pair_id,
                    "source_group_id": group_id,
                    "source_id": source["source_id"],
                    "source_revision": source["source_revision"],
                    "source_line": line_number,
                    "english": english,
                    "sorani": sorani,
                    "original_english": original_en,
                    "original_sorani": original_ckb,
                    "original_direction": "unknown",
                    "source_context": [],
                    "provenance": source["provenance"],
                    "language_verified": False,
                    "alignment_verified": False,
                    "flags": flags,
                    "duplicate_of_line": duplicate_of_line,
                },
            )
    if (
        processed != english_info.lines
        or processed != sorani_info.lines
        or english_digest.hexdigest() != english_info.sha256
        or sorani_digest.hexdigest() != sorani_info.sha256
    ):
        raise TextPairStagingError("An input file changed after the line-count check")
    return (
        {
            "input_pairs": processed,
            "staged_pairs": staged,
            "quarantined_pairs": quarantined,
            "quarantine_reasons": dict(sorted(reasons.items())),
            "warning_counts": dict(sorted(warnings.items())),
        },
        group_id,
    )


def _publish(stage: Path, output: Path) -> None:
    """Create the output exclusively; copy the completion report last."""
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise TextPairStagingError("Output path already exists; choose a new directory") from exc
    created: list[Path] = []
    try:
        for name in _OUTPUT_FILES:
            destination = output / name
            with destination.open("xb") as target:
                created.append(destination)
                with (stage / name).open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
    except BaseException:
        # Never recursively remove files someone else may have placed here.
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            output.rmdir()
        except OSError:
            pass
        raise


def stage_text_pairs(config: TextPairStagingConfig) -> dict[str, Any]:
    """Stage declared local line pairs and return a non-dialogue audit report."""
    english_file, sorani_file, manifest_file, output = _paths(config)
    source, manifest_hash, manifest_bytes = _read_manifest(manifest_file)
    english_info, sorani_info = _file_info(english_file), _file_info(sorani_file)
    if english_info.lines != sorani_info.lines:
        raise TextPairStagingError(
            "Physical line counts differ "
            f"(English: {english_info.lines}, Sorani: {sorani_info.lines}); no pairs staged. "
            "Do not delete blank lines separately."
        )
    if not english_info.lines:
        raise TextPairStagingError("Both input files are empty; no pairs staged")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temp:
        stage = Path(temp)
        counts, group_id = _process(
            english_file,
            sorani_file,
            english_info,
            sorani_info,
            source,
            stage,
            max_chars=config.max_chars,
        )
        pairs_info = _file_info(stage / "pairs.jsonl")
        quarantine_info = _file_info(stage / "quarantine.jsonl")
        report = {
            "schema": "sorani-text-pair-staging-v1",
            "status": "staged_for_review",
            "training_ready": False,
            "source": source,
            "source_group_id": group_id,
            "inputs": {
                "english": {
                    "filename": english_file.name,
                    "sha256": english_info.sha256,
                    "bytes": english_info.bytes,
                },
                "sorani": {
                    "filename": sorani_file.name,
                    "sha256": sorani_info.sha256,
                    "bytes": sorani_info.bytes,
                },
                "manifest_sha256": manifest_hash,
                "encoding": "utf-8-sig",
                "physical_line_endings": "LF or CRLF",
            },
            "outputs": {
                "pairs_sha256": pairs_info.sha256,
                "quarantine_sha256": quarantine_info.sha256,
            },
            "normalization": {"sorani_heh_policy": "preserve", "max_chars": config.max_chars},
            "counts": counts,
            "limitations": [
                "Rights and permitted uses are user declarations, not verified legal clearance.",
                "Language, dialect, translation quality and alignment have not been verified.",
                "Matching line counts do not establish semantic alignment.",
                "All rows share one conservative source group; "
                "no film grouping or splits are inferred.",
                "Text pairs have no subtitle timings, speakers, context, or reading-speed metrics.",
                "This staging output is not a dataset usable by the current trainer or web UI.",
            ],
        }
        (stage / "source-manifest.json").write_bytes(manifest_bytes)
        (stage / "staging-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _publish(stage, output)
    return report


def _demo_inputs(directory: Path) -> tuple[Path, Path, Path]:
    """Write independently authored sample pairs, not external subtitle data."""
    english = directory / "demo-en.txt"
    sorani = directory / "demo-ckb.txt"
    manifest = directory / "demo-source.json"
    english.write_text(
        "Please open the window.\nThe garden needs water.\nThank you, friend.\n", encoding="utf-8"
    )
    sorani.write_text(
        "تکایە پەنجەرەکە بکەوە.\nباخچەکە پێویستی بە ئاوە.\nسوپاس هاوڕێ.\n", encoding="utf-8"
    )
    evidence = "Independently authored synthetic examples for offline testing only"
    manifest.write_text(
        json.dumps(
            {
                "source_id": "synthetic-text-pair-demo",
                "source_revision": "demo-v1",
                "source_reference": evidence,
                "provenance": "unknown",
                "rights": {
                    "status": "synthetic",
                    "affirmed_by_user": True,
                    "original_text_evidence": evidence,
                    "translation_evidence": evidence,
                    "license_or_permission": "Synthetic testing only",
                    "attribution": "No external corpus used",
                    "permitted_uses": ["synthetic_testing"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return english, sorani, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use synthetic test pairs only")
    parser.add_argument("--english-file", type=Path, help="Local UTF-8 English line file")
    parser.add_argument("--sorani-file", type=Path, help="Local UTF-8 Sorani line file")
    parser.add_argument("--source-manifest", type=Path, help="Local source and rights JSON")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New private output directory"
    )
    parser.add_argument("--max-chars", type=int, default=5000, help="Maximum characters per line")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    supplied = (args.english_file, args.sorani_file, args.source_manifest)
    if args.demo and any(item is not None for item in supplied):
        parser.error("--demo cannot be combined with input files or a source manifest")
    if not args.demo and any(item is None for item in supplied):
        parser.error("supply --english-file, --sorani-file and --source-manifest, or use --demo")
    try:
        if args.demo:
            with tempfile.TemporaryDirectory(prefix="sorani-synthetic-pairs-") as temp:
                english, sorani, manifest = _demo_inputs(Path(temp))
                report = stage_text_pairs(
                    TextPairStagingConfig(
                        english, sorani, manifest, args.output_dir, args.max_chars
                    )
                )
        else:
            report = stage_text_pairs(
                TextPairStagingConfig(
                    args.english_file,
                    args.sorani_file,
                    args.source_manifest,
                    args.output_dir,
                    args.max_chars,
                )
            )
    except (TextPairStagingError, OSError) as exc:
        print(f"Text pair staging failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "counts": report["counts"],
                "training_ready": False,
                "note": "Staged for review only; rights, language and alignment are not verified.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
