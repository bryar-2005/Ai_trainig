"""Stage local, plain-segment English–Sorani TMX for review, not training.

This offline tool never downloads a corpus or starts a job. It preserves a
reference to each original translation unit, but it does not infer works,
timings, context, rights, translation quality, or the actual original language.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from ml.data.normalize import normalize_english, normalize_sorani
from ml.data.stage_text_pairs import (
    TextPairStagingError,
    _demo_inputs,
    _file_info,
    _identity,
    _pair_flags,
    _publish,
    _read_manifest,
    _validate_decoded_text,
    _write_jsonl,
)

_MAX_TMX_BYTES = 256 * 1024 * 1024
_MAX_XML_DEPTH = 32
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_LANGUAGE_TAG = re.compile(r"^(en|ckb)(?:-[a-z0-9]{1,8})*$", re.IGNORECASE | re.ASCII)
_PLACEHOLDER_OR_MARKUP = re.compile(r"<[^<>\n]{1,256}>|\{[^{}\n]{1,256}\}|%(?:\d+\$)?[a-zA-Z]")


@dataclass(frozen=True, slots=True)
class TmxPairStagingConfig:
    tmx_file: Path
    source_manifest: Path
    output_dir: Path
    max_chars: int = 5000


@dataclass(slots=True)
class _ReviewState:
    first_pair: dict[str, int] = field(default_factory=dict)
    first_unit_id: dict[str, int] = field(default_factory=dict)
    reasons: Counter[str] = field(default_factory=Counter)
    warnings: Counter[str] = field(default_factory=Counter)
    units: int = 0
    staged: int = 0
    quarantined: int = 0


class _HashingReader:
    """Hash exactly the bytes passed to the XML parser, including read-ahead."""

    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size: int = -1) -> bytes:
        data = self.handle.read(size)
        self.bytes += len(data)
        if self.bytes > _MAX_TMX_BYTES:
            raise TextPairStagingError("TMX file exceeds the 256 MiB safety limit")
        self.digest.update(data)
        return data


def _paths(config: TmxPairStagingConfig) -> tuple[Path, Path, Path]:
    if isinstance(config.max_chars, bool) or not isinstance(config.max_chars, int):
        raise TextPairStagingError("max_chars must be an integer between 1 and 100000")
    if not 1 <= config.max_chars <= 100_000:
        raise TextPairStagingError("max_chars must be an integer between 1 and 100000")
    paths: list[Path] = []
    for name, supplied in (
        ("TMX file", config.tmx_file),
        ("source manifest", config.source_manifest),
    ):
        path = Path(supplied).expanduser()
        if path.is_symlink() or not path.is_file():
            raise TextPairStagingError(f"{name} must be an existing regular file, not a symlink")
        paths.append(path.resolve())
    if os.path.samefile(*paths):
        raise TextPairStagingError("TMX file and manifest must be distinct files")
    supplied_output = Path(config.output_dir).expanduser()
    if supplied_output.is_symlink() or supplied_output.exists():
        raise TextPairStagingError("Output path already exists; choose a new directory")
    output = supplied_output.resolve()
    if any(path == output or path.is_relative_to(output) for path in paths):
        raise TextPairStagingError("Output must not contain an input file")
    return paths[0], paths[1], output


def _safe_metadata(value: str | None, limit: int) -> bool:
    return value is None or (
        len(value) <= limit
        and not any(unicodedata.category(char) == "Cc" or char in "\u2028\u2029" for char in value)
    )


def _language_kind(tag: str | None) -> str | None:
    if tag is None or len(tag) > 64:
        return None
    match = _LANGUAGE_TAG.fullmatch(tag)
    if not match:
        return None
    return "english" if match.group(1).lower() == "en" else "sorani"


def _stage_unit(
    unit: ET.Element,
    *,
    source: dict[str, Any],
    input_sha256: str,
    group_id: str,
    header_source_language: str,
    max_chars: int,
    state: _ReviewState,
    pairs: TextIO,
    quarantine: TextIO,
) -> None:
    state.units += 1
    ordinal = state.units
    pair_id = _identity(
        "tmx-text-pair-v1", group_id, source["source_revision"], input_sha256, ordinal
    )
    reasons: list[str] = []
    unit_id = unit.get("tuid")
    declared_source = unit.get("srclang")
    if not _safe_metadata(unit_id, 2048):
        reasons.append("invalid_source_unit_id")
    if not _safe_metadata(declared_source, 128):
        reasons.append("invalid_declared_source_language")

    duplicate_unit_id_of = None
    if unit_id and unit_id.strip() and _safe_metadata(unit_id, 2048):
        duplicate_unit_id_of = state.first_unit_id.get(unit_id)
        if duplicate_unit_id_of is None:
            state.first_unit_id[unit_id] = ordinal

    variants = [child for child in unit if child.tag == "tuv"]
    if len(variants) != 2:
        reasons.append("expected_two_language_variants")
    if any(child.tag not in {"tuv", "prop", "note"} for child in unit):
        reasons.append("unsupported_unit_element")
    if (unit.text and unit.text.strip()) or any(
        child.tail and child.tail.strip() for child in unit
    ):
        reasons.append("unexpected_unit_text")

    by_language: dict[str, list[tuple[str, str]]] = {"english": [], "sorani": []}
    language_order: list[str] = []
    for variant in variants:
        tag = variant.get(_XML_LANG)
        kind = _language_kind(tag)
        if kind is None:
            reasons.append("unsupported_or_missing_variant_language")
            continue
        assert tag is not None
        language_order.append(tag)
        if any(child.tag not in {"seg", "prop", "note"} for child in variant):
            reasons.append(f"{kind}_unsupported_variant_element")
        if variant.text and variant.text.strip():
            reasons.append(f"{kind}_unexpected_variant_text")
        segments = [child for child in variant if child.tag == "seg"]
        if len(segments) != 1:
            reasons.append(f"{kind}_expected_one_segment")
            continue
        segment = segments[0]
        if len(segment):
            # Flattening inline TMX XML could silently discard markup or context.
            reasons.append(f"{kind}_inline_xml_markup")
            continue
        if any(child.tail and child.tail.strip() for child in variant):
            reasons.append(f"{kind}_unexpected_variant_text")
        original = segment.text or ""
        reasons.extend(
            _validate_decoded_text(
                original, language=kind, max_chars=max_chars, allow_multiline=True
            )
        )
        by_language[kind].append((tag, original))

    for kind in ("english", "sorani"):
        if len(by_language[kind]) != 1:
            reasons.append(f"{kind}_expected_one_variant")

    english = sorani = ""
    if not reasons:
        original_en = by_language["english"][0][1]
        original_ckb = by_language["sorani"][0][1]
        english = normalize_english(original_en)
        sorani = normalize_sorani(original_ckb, heh_policy="preserve")
        for kind, text in (("english", english), ("sorani", sorani)):
            if not text.strip():
                reasons.append(f"{kind}_blank_after_normalization")
            if len(text) > max_chars:
                reasons.append(f"{kind}_too_long_after_normalization")

    if reasons:
        reasons = list(dict.fromkeys(reasons))
        state.quarantined += 1
        state.reasons.update(reasons)
        _write_jsonl(
            quarantine,
            {"pair_id": pair_id, "source_unit_index": ordinal, "reasons": reasons},
        )
        return

    flags, duplicate_pair_of = _pair_flags(english, sorani, state.first_pair, ordinal)
    if not unit_id or not unit_id.strip():
        flags.append("missing_source_unit_id")
    elif duplicate_unit_id_of is not None:
        flags.append("duplicate_source_unit_id")
    if (
        declared_source
        and declared_source != "*all*"
        and declared_source.casefold() not in {tag.casefold() for tag in language_order}
    ):
        flags.append("declared_source_language_not_in_variants")
    if any(char in original_en or char in original_ckb for char in "\n\r"):
        flags.append("multiline_segment")
    if _PLACEHOLDER_OR_MARKUP.search(original_en) or _PLACEHOLDER_OR_MARKUP.search(original_ckb):
        flags.append("potential_placeholder_or_markup")
    state.staged += 1
    state.warnings.update(flags)
    _write_jsonl(
        pairs,
        {
            "record_type": "text_pair",
            "pair_id": pair_id,
            "source_group_id": group_id,
            "source_id": source["source_id"],
            "source_revision": source["source_revision"],
            "source_unit_index": ordinal,
            "source_unit_id": unit_id,
            "header_declared_source_language": header_source_language,
            "unit_declared_source_language": declared_source,
            "variant_language_order": language_order,
            "english_language_tag": by_language["english"][0][0],
            "sorani_language_tag": by_language["sorani"][0][0],
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
            "duplicate_of_unit": duplicate_pair_of,
            "duplicate_source_unit_id_of": duplicate_unit_id_of,
        },
    )


def _process_tmx(
    tmx_file: Path,
    *,
    source: dict[str, Any],
    input_sha256: str,
    input_bytes: int,
    stage: Path,
    max_chars: int,
) -> tuple[dict[str, Any], str, str]:
    group_id = _identity("text-pair-collection-v1", source["source_id"])
    state = _ReviewState()
    stack: list[ET.Element] = []
    header_seen = body_seen = False
    header_source_language: str | None = None
    header: ET.Element | None = None
    body: ET.Element | None = None
    pending_unit: ET.Element | None = None

    def discard_completed_unit() -> None:
        nonlocal pending_unit
        if pending_unit is not None:
            if pending_unit.tail and pending_unit.tail.strip():
                raise TextPairStagingError("Unexpected text between TMX translation units")
            assert body is not None
            body.remove(pending_unit)
            pending_unit.clear()
            pending_unit = None

    try:
        with (
            tmx_file.open("rb") as original,
            (stage / "pairs.jsonl").open("w", encoding="utf-8", newline="\n") as pairs,
            (stage / "quarantine.jsonl").open("w", encoding="utf-8", newline="\n") as quarantine,
        ):
            reader = _HashingReader(original)
            events = SafeET.iterparse(
                reader,
                events=("start", "end"),
                forbid_dtd=True,
                forbid_entities=True,
                forbid_external=True,
            )
            for event, element in events:
                if event == "start":
                    if len(stack) >= _MAX_XML_DEPTH:
                        raise TextPairStagingError("TMX XML nesting exceeds the safety limit")
                    if not stack:
                        if element.tag != "tmx" or element.get("version") != "1.4":
                            raise TextPairStagingError(
                                "Expected an unnamespaced TMX version 1.4 root"
                            )
                    elif len(stack) == 1:
                        if stack[0].text and stack[0].text.strip():
                            raise TextPairStagingError("Unexpected text in TMX root")
                        if element.tag == "header" and not header_seen and not body_seen:
                            header_seen = True
                            header = element
                            header_source_language = element.get("srclang")
                            if (
                                not header_source_language
                                or not header_source_language.strip()
                                or not _safe_metadata(header_source_language, 128)
                            ):
                                raise TextPairStagingError(
                                    "TMX header must declare a source language"
                                )
                        elif element.tag == "body" and header_seen and not body_seen:
                            if header is not None and header.tail and header.tail.strip():
                                raise TextPairStagingError("Unexpected text in TMX root")
                            body_seen = True
                            body = element
                        else:
                            raise TextPairStagingError(
                                "Expected one TMX header followed by one body"
                            )
                    elif stack[-1] is body:
                        # The previous unit's tail is complete once the next child starts.
                        discard_completed_unit()
                        if element.tag != "tu":
                            raise TextPairStagingError(
                                "TMX body must contain only translation units"
                            )
                    stack.append(element)
                else:
                    if len(stack) == 3 and stack[-2] is body and element.tag == "tu":
                        assert header_source_language is not None and body is not None
                        _stage_unit(
                            element,
                            source=source,
                            input_sha256=input_sha256,
                            group_id=group_id,
                            header_source_language=header_source_language,
                            max_chars=max_chars,
                            state=state,
                            pairs=pairs,
                            quarantine=quarantine,
                        )
                        # Wait for the next sibling (or body end) before checking its tail.
                        pending_unit = element
                    elif element is body:
                        if body.text and body.text.strip():
                            raise TextPairStagingError("Unexpected text in TMX body")
                        discard_completed_unit()
                    elif len(stack) == 1 and body is not None and body.tail and body.tail.strip():
                        raise TextPairStagingError("Unexpected text in TMX root")
                    stack.pop()
            if reader.bytes != input_bytes or reader.digest.hexdigest() != input_sha256:
                raise TextPairStagingError("TMX input changed after the hash check")
    except (ET.ParseError, DefusedXmlException, LookupError, UnicodeError) as exc:
        raise TextPairStagingError("Invalid or unsafe TMX XML; no pairs staged") from exc
    if not header_seen or not body_seen or not header_source_language:
        raise TextPairStagingError("Expected one TMX header followed by one body")
    if not state.units:
        raise TextPairStagingError("TMX contains no translation units; no pairs staged")
    return (
        {
            "input_units": state.units,
            "staged_pairs": state.staged,
            "quarantined_units": state.quarantined,
            "quarantine_reasons": dict(sorted(state.reasons.items())),
            "warning_counts": dict(sorted(state.warnings.items())),
        },
        group_id,
        header_source_language,
    )


def stage_tmx_pairs(config: TmxPairStagingConfig) -> dict[str, Any]:
    """Stage one explicitly declared local TMX file and return its review report."""
    tmx_file, manifest_file, output = _paths(config)
    source, manifest_hash, manifest_bytes = _read_manifest(manifest_file)
    input_info = _file_info(tmx_file)
    if input_info.bytes > _MAX_TMX_BYTES:
        raise TextPairStagingError("TMX file exceeds the 256 MiB safety limit")
    if not input_info.bytes:
        raise TextPairStagingError("TMX file is empty; no pairs staged")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temp:
        stage = Path(temp)
        counts, group_id, header_source_language = _process_tmx(
            tmx_file,
            source=source,
            input_sha256=input_info.sha256,
            input_bytes=input_info.bytes,
            stage=stage,
            max_chars=config.max_chars,
        )
        pairs_info = _file_info(stage / "pairs.jsonl")
        quarantine_info = _file_info(stage / "quarantine.jsonl")
        report = {
            "schema": "sorani-tmx-pair-staging-v1",
            "status": "staged_for_review",
            "training_ready": False,
            "source": source,
            "source_group_id": group_id,
            "inputs": {
                "tmx": {
                    "filename": tmx_file.name,
                    "sha256": input_info.sha256,
                    "bytes": input_info.bytes,
                    "version": "1.4",
                    "header_declared_source_language": header_source_language,
                },
                "manifest_sha256": manifest_hash,
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
                "Source language tags are declarations, not proof "
                "of original authorship or direction.",
                "All units share one conservative source group; "
                "no work grouping or splits are inferred.",
                "Keep the exact original TMX: XML parsing normalizes syntax "
                "and not all metadata is copied.",
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


def _demo_files(directory: Path) -> tuple[Path, Path]:
    english, sorani, manifest = _demo_inputs(directory)
    root = ET.Element("tmx", version="1.4")
    ET.SubElement(root, "header", srclang="en-US", datatype="plaintext")
    body = ET.SubElement(root, "body")
    for ordinal, (en, ckb) in enumerate(
        zip(
            english.read_text(encoding="utf-8").splitlines(),
            sorani.read_text(encoding="utf-8").splitlines(),
            strict=True,
        ),
        1,
    ):
        unit = ET.SubElement(body, "tu", tuid=f"synthetic-unit-{ordinal}", srclang="en-US")
        for tag, text in (("en-US", en), ("ckb", ckb)):
            variant = ET.SubElement(unit, "tuv", {_XML_LANG: tag})
            ET.SubElement(variant, "seg").text = text
    tmx = directory / "synthetic.tmx"
    ET.ElementTree(root).write(tmx, encoding="utf-8", xml_declaration=True)
    return tmx, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use synthetic test pairs only")
    parser.add_argument("--tmx-file", type=Path, help="Local plain-segment English–Sorani TMX")
    parser.add_argument("--source-manifest", type=Path, help="Local source and rights JSON")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New private output directory"
    )
    parser.add_argument(
        "--max-chars", type=int, default=5000, help="Maximum characters per segment"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    supplied = (args.tmx_file, args.source_manifest)
    if args.demo and any(item is not None for item in supplied):
        parser.error("--demo cannot be combined with an input file or source manifest")
    if not args.demo and any(item is None for item in supplied):
        parser.error("supply --tmx-file and --source-manifest, or use --demo")
    try:
        if args.demo:
            with tempfile.TemporaryDirectory(prefix="sorani-synthetic-tmx-") as temp:
                tmx, manifest = _demo_files(Path(temp))
                report = stage_tmx_pairs(
                    TmxPairStagingConfig(tmx, manifest, args.output_dir, args.max_chars)
                )
        else:
            report = stage_tmx_pairs(
                TmxPairStagingConfig(
                    args.tmx_file, args.source_manifest, args.output_dir, args.max_chars
                )
            )
    except (TextPairStagingError, OSError) as exc:
        print(f"TMX staging failed: {exc}", file=sys.stderr)
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
