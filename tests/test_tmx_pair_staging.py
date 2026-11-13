"""TMX staging tests: independently authored synthetic XML, no real corpus."""

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import ml.data.stage_tmx_pairs as staging
from ml.data.stage_text_pairs import TextPairStagingError, _identity
from ml.data.stage_tmx_pairs import TmxPairStagingConfig, stage_tmx_pairs
from ml.train.dataset import validate_splits

_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def synthetic_manifest() -> dict:
    return {
        "source_id": "synthetic-tmx-collection",
        "source_revision": "test-v1",
        "source_reference": "Independent synthetic test fixture",
        "provenance": "unknown",
        "rights": {
            "status": "synthetic",
            "affirmed_by_user": True,
            "original_text_evidence": "Independently authored for tests",
            "translation_evidence": "Independently authored for tests",
            "license_or_permission": "Synthetic testing only",
            "attribution": "No external corpus",
            "permitted_uses": ["synthetic_testing"],
        },
        "extra_condition": "Keep this source manifest exactly as supplied",
    }


def new_document() -> tuple[ET.Element, ET.Element]:
    root = ET.Element("tmx", version="1.4")
    ET.SubElement(root, "header", srclang="en-US", datatype="plaintext")
    return root, ET.SubElement(root, "body")


def add_unit(
    body: ET.Element,
    variants: list[tuple[str, str]],
    *,
    unit_id: str | None = None,
    declared_source: str | None = None,
) -> ET.Element:
    unit = ET.SubElement(body, "tu")
    if unit_id is not None:
        unit.set("tuid", unit_id)
    if declared_source is not None:
        unit.set("srclang", declared_source)
    for tag, text in variants:
        variant = ET.SubElement(unit, "tuv", {_XML_LANG: tag})
        ET.SubElement(variant, "seg").text = text
    return unit


def xml_bytes(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def make_config(
    directory: Path,
    raw: bytes | None = None,
    manifest: dict | None = None,
    *,
    max_chars: int = 5000,
) -> TmxPairStagingConfig:
    directory.mkdir(parents=True, exist_ok=True)
    if raw is None:
        root, body = new_document()
        add_unit(body, [("en-US", "A small test."), ("ckb", "تاقیکردنەوەیەکی بچووک.")])
        raw = xml_bytes(root)
    tmx_path = directory / "input.tmx"
    manifest_path = directory / "manifest.json"
    tmx_path.write_bytes(raw)
    manifest_path.write_text(
        json.dumps(manifest if manifest is not None else synthetic_manifest(), ensure_ascii=False),
        encoding="utf-8",
    )
    return TmxPairStagingConfig(tmx_path, manifest_path, directory / "output", max_chars)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ml.data.stage_tmx_pairs", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_demo_is_offline_review_only_and_not_current_trainer_input(tmp_path: Path):
    output = tmp_path / "demo"
    result = cli("--demo", "--output-dir", str(output))
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["counts"]["input_units"] == summary["counts"]["staged_pairs"] == 3
    assert summary["counts"]["quarantined_units"] == 0
    assert summary["training_ready"] is False
    report = json.loads((output / "staging-report.json").read_text(encoding="utf-8"))
    rows = read_jsonl(output / "pairs.jsonl")
    assert report["schema"] == "sorani-tmx-pair-staging-v1"
    assert report["source"]["rights"]["status"] == "synthetic"
    assert report["training_ready"] is False
    assert report["inputs"]["tmx"]["header_declared_source_language"] == "en-US"
    assert read_jsonl(output / "quarantine.jsonl") == []
    for name in ("pairs", "quarantine"):
        assert (
            report["outputs"][f"{name}_sha256"]
            == hashlib.sha256((output / f"{name}.jsonl").read_bytes()).hexdigest()
        )
    for row in rows:
        assert row["record_type"] == "text_pair"
        assert row["original_direction"] == "unknown"
        assert row["source_context"] == []
        assert not row["language_verified"] and not row["alignment_verified"]
        assert not {"film_id", "english_start_ms", "sorani_start_ms", "speakers"} & row.keys()
    with pytest.raises(ValueError, match="missing fields"):
        validate_splits({"train": [rows[0]], "val": [rows[1]]}, allow_unverified=True)
    assert "Please open" not in result.stdout
    assert "Please open" not in (output / "staging-report.json").read_text(encoding="utf-8")


def test_preserves_direction_declarations_metadata_newlines_zwnj_and_duplicate_ids(tmp_path: Path):
    root, body = new_document()
    en = "Open { $window } & wait.\nAnother line."
    ckb = "ئه\u200cم دەرگایە بکەوە.\nچاوەڕێ بکە."
    first = add_unit(
        body,
        [("ckb-Arab-IQ", ckb), ("en-GB", en)],
        unit_id="fixture:duplicate-id",
        declared_source="ckb-Arab-IQ",
    )
    ET.SubElement(first, "prop", type="note").text = "Extra synthetic metadata lives in input"
    add_unit(
        body,
        [("EN-gb", en), ("CKB-arab-iq", ckb)],
        unit_id="fixture:duplicate-id",
        declared_source="en-GB",
    )
    add_unit(body, [("en", "Last pair."), ("ckb", "کۆتا دێڕ.")])
    raw = xml_bytes(root)
    config = make_config(tmp_path, raw)
    manifest_bytes = config.source_manifest.read_bytes()
    report = stage_tmx_pairs(config)
    rows = read_jsonl(config.output_dir / "pairs.jsonl")
    assert config.tmx_file.read_bytes() == raw
    assert (config.output_dir / "source-manifest.json").read_bytes() == manifest_bytes
    assert report["inputs"]["tmx"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["counts"]["staged_pairs"] == 3
    assert [row["source_unit_index"] for row in rows] == [1, 2, 3]
    assert len({row["pair_id"] for row in rows}) == 3
    assert {row["source_group_id"] for row in rows} == {
        _identity("text-pair-collection-v1", "synthetic-tmx-collection")
    }
    assert rows[0]["source_unit_id"] == "fixture:duplicate-id"
    assert rows[0]["header_declared_source_language"] == "en-US"
    assert rows[0]["unit_declared_source_language"] == "ckb-Arab-IQ"
    assert rows[0]["variant_language_order"] == ["ckb-Arab-IQ", "en-GB"]
    assert rows[0]["english_language_tag"] == "en-GB"
    assert rows[0]["sorani_language_tag"] == "ckb-Arab-IQ"
    assert rows[0]["original_english"] == en and rows[0]["original_sorani"] == ckb
    assert "\u200c" in rows[0]["sorani"] and "ه" in rows[0]["sorani"]
    assert "\n" in rows[0]["english"] and "\n" in rows[0]["sorani"]
    assert {"multiline_segment", "potential_placeholder_or_markup", "ambiguous_arabic_heh"} <= set(
        rows[0]["flags"]
    )
    assert {"duplicate_pair", "duplicate_source_unit_id"} <= set(rows[1]["flags"])
    assert rows[1]["duplicate_of_unit"] == rows[1]["duplicate_source_unit_id_of"] == 1
    assert "missing_source_unit_id" in rows[2]["flags"]
    assert all(
        row["original_direction"] == "unknown" and row["source_context"] == [] for row in rows
    )
    assert report["counts"]["warning_counts"]["duplicate_pair"] == 1
    assert report["counts"]["warning_counts"]["duplicate_source_unit_id"] == 1
    assert "Extra synthetic metadata" not in (config.output_dir / "staging-report.json").read_text()


def test_invalid_units_are_quarantined_whole_without_shifting_later_pairs(tmp_path: Path):
    root, body = new_document()
    add_unit(body, [("en", "First okay."), ("ckb", "یەکەم باشە.")])
    inline = add_unit(body, [("en", "Do not flatten."), ("ckb", "مەگۆڕە.")])
    segment = inline.find("./tuv/seg")
    assert segment is not None
    ET.SubElement(segment, "ph").text = "markup"
    add_unit(body, [("en", " "), ("ckb", "بەتاڵە.")])
    add_unit(body, [("en", "Unknown dialect."), ("ku", "دەق.")])
    add_unit(body, [("en", "Duplicate variant."), ("en-US", "Other."), ("ckb", "دەق.")])
    add_unit(body, [("en", "Bad\u0085control"), ("ckb", "دەق.")])
    add_unit(body, [("en", "\ufeff"), ("ckb", "دەق.")])
    add_unit(body, [("ckb", "دواتر."), ("en", "Last okay.")])
    config = make_config(tmp_path, xml_bytes(root))
    report = stage_tmx_pairs(config)
    rows = read_jsonl(config.output_dir / "pairs.jsonl")
    bad = read_jsonl(config.output_dir / "quarantine.jsonl")
    assert [row["source_unit_index"] for row in rows] == [1, 8]
    assert rows[1]["english"] == "Last okay." and rows[1]["sorani"] == "دواتر."
    assert [row["source_unit_index"] for row in bad] == [2, 3, 4, 5, 6, 7]
    assert "english_inline_xml_markup" in bad[0]["reasons"]
    assert "english_blank" in bad[1]["reasons"]
    assert "unsupported_or_missing_variant_language" in bad[2]["reasons"]
    assert "sorani_expected_one_variant" in bad[2]["reasons"]
    assert "expected_two_language_variants" in bad[3]["reasons"]
    assert "english_expected_one_variant" in bad[3]["reasons"]
    assert "english_control_or_line_separator" in bad[4]["reasons"]
    assert "english_blank_after_normalization" in bad[5]["reasons"]
    assert report["counts"]["input_units"] == 8
    assert report["counts"]["staged_pairs"] == 2
    assert report["counts"]["quarantined_units"] == 6
    quarantined_text = (config.output_dir / "quarantine.jsonl").read_text(encoding="utf-8")
    assert "Do not flatten" not in quarantined_text and "Unknown dialect" not in quarantined_text


@pytest.mark.parametrize("tag", ["ku", "cKb", "ckb-İ", "en-uſ"])
def test_non_sorani_and_non_ascii_confusable_language_tags_are_rejected(tmp_path: Path, tag: str):
    root, body = new_document()
    add_unit(body, [("en", "Test."), (tag, "دەق.")])
    config = make_config(tmp_path, xml_bytes(root))
    report = stage_tmx_pairs(config)
    assert report["counts"]["quarantined_units"] == 1
    reasons = read_jsonl(config.output_dir / "quarantine.jsonl")[0]["reasons"]
    assert "unsupported_or_missing_variant_language" in reasons


def test_escaped_markup_is_preserved_and_flagged_not_silently_stripped(tmp_path: Path):
    root, body = new_document()
    add_unit(
        body,
        [("en", "<b>Hello</b>, %s!"), ("ckb", "<b>سڵاو</b>، %s!")],
        declared_source="fr",
    )
    config = make_config(tmp_path, xml_bytes(root))
    stage_tmx_pairs(config)
    row = read_jsonl(config.output_dir / "pairs.jsonl")[0]
    assert row["original_english"] == "<b>Hello</b>, %s!"
    assert "<b>" in row["english"] and "<b>" in row["sorani"]
    assert "potential_placeholder_or_markup" in row["flags"]
    assert "declared_source_language_not_in_variants" in row["flags"]
    assert row["unit_declared_source_language"] == "fr"
    assert row["original_direction"] == "unknown"


def test_stable_ids_bind_source_revision_and_input_bytes_not_output_path(tmp_path: Path):
    first = make_config(tmp_path / "one")
    second = make_config(tmp_path / "two", first.tmx_file.read_bytes())
    report1 = stage_tmx_pairs(first)
    report2 = stage_tmx_pairs(second)
    row1 = read_jsonl(first.output_dir / "pairs.jsonl")[0]
    row2 = read_jsonl(second.output_dir / "pairs.jsonl")[0]
    assert row1["pair_id"] == row2["pair_id"]
    assert report1["source_group_id"] == report2["source_group_id"]
    third_manifest = synthetic_manifest()
    third_manifest["source_revision"] = "test-v2"
    third = make_config(tmp_path / "three", first.tmx_file.read_bytes(), third_manifest)
    stage_tmx_pairs(third)
    row3 = read_jsonl(third.output_dir / "pairs.jsonl")[0]
    fourth = make_config(tmp_path / "four", first.tmx_file.read_bytes() + b"\n")
    stage_tmx_pairs(fourth)
    row4 = read_jsonl(fourth.output_dir / "pairs.jsonl")[0]
    assert row1["pair_id"] != row3["pair_id"] != row4["pair_id"]
    assert row1["pair_id"] != row4["pair_id"]
    assert row1["source_group_id"] == row3["source_group_id"] == row4["source_group_id"]


def test_utf16_xml_uses_declared_parser_encoding_and_hashes_exact_input(tmp_path: Path):
    xml = (
        '<?xml version="1.0" encoding="UTF-16"?>'
        '<tmx version="1.4"><header srclang="en"/><body>'
        '<tu><tuv xml:lang="en"><seg>Test.</seg></tuv>'
        '<tuv xml:lang="ckb"><seg>تاقیکردنەوە.</seg></tuv></tu></body></tmx>'
    ).encode("utf-16")
    config = make_config(tmp_path, xml)
    report = stage_tmx_pairs(config)
    assert report["inputs"]["tmx"]["sha256"] == hashlib.sha256(xml).hexdigest()
    assert read_jsonl(config.output_dir / "pairs.jsonl")[0]["original_english"] == "Test."


@pytest.mark.parametrize(
    "raw",
    [
        b'<!DOCTYPE tmx><tmx version="1.4"><header srclang="en"/><body/></tmx>',
        b'<!DOCTYPE tmx [<!ENTITY e "hello">]><tmx version="1.4">&e;</tmx>',
        b'<!DOCTYPE tmx [<!ENTITY e SYSTEM "file:///etc/passwd">]><tmx version="1.4">&e;</tmx>',
        b'<tmx version="1.4"><header srclang="en"/><body><tu>\xff</tu></body></tmx>',
        b'<tmx version="1.4"><header srclang="en"/><body><tu></body></tmx>',
        b'<?xml version="1.0" encoding="X-UNKNOWN"?><tmx/>',
    ],
)
def test_invalid_or_unsafe_xml_fails_whole_run_without_publishing(tmp_path: Path, raw: bytes):
    config = make_config(tmp_path, raw)
    with pytest.raises(TextPairStagingError, match="Invalid or unsafe TMX"):
        stage_tmx_pairs(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".output.staging-*"))
    assert config.tmx_file.read_bytes() == raw


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"", "empty"),
        (b'<tmx version="1.3"><header srclang="en"/><body/></tmx>', "version 1.4"),
        (b'<tmx version="1.4"><body/></tmx>', "header followed"),
        (b'<tmx version="1.4"><header/><body/></tmx>', "declare a source"),
        (b'<tmx version="1.4"><header srclang="en"/><body/></tmx>', "no translation units"),
        (
            b'<tmx version="1.4"><header srclang="en"/><body><unexpected/></body></tmx>',
            "body must contain only",
        ),
    ],
)
def test_unsupported_document_structure_fails_without_output(tmp_path: Path, raw, message):
    config = make_config(tmp_path, raw)
    with pytest.raises(TextPairStagingError, match=message):
        stage_tmx_pairs(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".output.staging-*"))


@pytest.mark.parametrize("placement", ["body_before", "between_units", "root_after"])
def test_unexpected_outer_text_fails_even_across_parser_chunks(tmp_path: Path, placement: str):
    root, body = new_document()
    first = add_unit(body, [("en", "First."), ("ckb", "یەکەم.")])
    add_unit(body, [("en", "Second."), ("ckb", "دووەم.")])
    if placement == "body_before":
        body.text = "Unexpected outside of a translation unit"
    elif placement == "between_units":
        # The second unit starts after multiple XML parser reads.
        first.tail = " " * 70_000 + "Unexpected between units"
    else:
        body.tail = "Unexpected after the body"
    config = make_config(tmp_path, xml_bytes(root))
    with pytest.raises(TextPairStagingError, match="Unexpected text"):
        stage_tmx_pairs(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".output.staging-*"))


def test_unexpected_text_inside_unit_is_quarantined(tmp_path: Path):
    root, body = new_document()
    unit = add_unit(body, [("en", "First."), ("ckb", "یەکەم.")])
    unit[0].tail = "Unexpected outside of a variant"
    config = make_config(tmp_path, xml_bytes(root))
    report = stage_tmx_pairs(config)
    assert report["counts"]["quarantined_units"] == 1
    assert (
        "unexpected_unit_text" in read_jsonl(config.output_dir / "quarantine.jsonl")[0]["reasons"]
    )


def test_xml_size_and_depth_limits_fail_without_output(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path / "size")
    monkeypatch.setattr(staging, "_MAX_TMX_BYTES", 10)
    with pytest.raises(TextPairStagingError, match="256 MiB safety limit"):
        stage_tmx_pairs(config)
    assert not config.output_dir.exists()
    monkeypatch.undo()
    root, body = new_document()
    unit = add_unit(body, [("en", "Test."), ("ckb", "دەق.")])
    inner = ET.SubElement(unit, "prop")
    for _ in range(40):
        inner = ET.SubElement(inner, "prop")
    deep = make_config(tmp_path / "deep", xml_bytes(root))
    with pytest.raises(TextPairStagingError, match="nesting exceeds"):
        stage_tmx_pairs(deep)
    assert not deep.output_dir.exists()


def test_size_and_char_limits_and_user_rights_attestation(tmp_path: Path):
    root, body = new_document()
    add_unit(body, [("en", "This side is too long."), ("ckb", "دەق.")])
    config = make_config(tmp_path / "long", xml_bytes(root), max_chars=10)
    report = stage_tmx_pairs(config)
    assert report["counts"]["quarantined_units"] == 1
    assert "english_too_long" in read_jsonl(config.output_dir / "quarantine.jsonl")[0]["reasons"]
    manifest = synthetic_manifest()
    manifest["rights"]["affirmed_by_user"] = False
    no_rights = make_config(tmp_path / "rights", manifest=manifest)
    with pytest.raises(TextPairStagingError, match="affirmed_by_user"):
        stage_tmx_pairs(no_rights)
    assert not no_rights.output_dir.exists()
    invalid = make_config(tmp_path / "invalid-max", max_chars=True)
    with pytest.raises(TextPairStagingError, match="max_chars"):
        stage_tmx_pairs(invalid)
    assert not invalid.output_dir.exists()


def test_input_modified_after_hash_is_detected_and_no_output_published(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    real_file_info = staging._file_info
    changed = False

    def mutate_after_hash(path):
        nonlocal changed
        info = real_file_info(path)
        if path == config.tmx_file.resolve() and not changed:
            config.tmx_file.write_bytes(config.tmx_file.read_bytes() + b"\n")
            changed = True
        return info

    monkeypatch.setattr(staging, "_file_info", mutate_after_hash)
    with pytest.raises(TextPairStagingError, match="changed after the hash check"):
        stage_tmx_pairs(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".output.staging-*"))


def test_paths_and_racing_output_never_overwrite_existing_files(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    with pytest.raises(TextPairStagingError, match="distinct files"):
        stage_tmx_pairs(TmxPairStagingConfig(config.tmx_file, config.tmx_file, config.output_dir))
    try:
        link = tmp_path / "linked-input"
        link.symlink_to(config.tmx_file)
        output_link = tmp_path / "linked-output"
        output_link.symlink_to(tmp_path / "missing", target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this platform")
    with pytest.raises(TextPairStagingError, match="not a symlink"):
        stage_tmx_pairs(TmxPairStagingConfig(link, config.source_manifest, config.output_dir))
    with pytest.raises(TextPairStagingError, match="already exists"):
        stage_tmx_pairs(TmxPairStagingConfig(config.tmx_file, config.source_manifest, output_link))
    real_publish = staging._publish

    def competing_process(stage, output):
        output.mkdir()
        (output / "keep.txt").write_text("someone else's data", encoding="utf-8")
        real_publish(stage, output)

    monkeypatch.setattr(staging, "_publish", competing_process)
    with pytest.raises(TextPairStagingError, match="already exists"):
        stage_tmx_pairs(config)
    assert (config.output_dir / "keep.txt").read_text(encoding="utf-8") == "someone else's data"
    assert not list(tmp_path.glob(".output.staging-*"))


def test_cli_local_inputs_and_invalid_combinations(tmp_path: Path):
    config = make_config(tmp_path)
    result = cli(
        "--tmx-file",
        str(config.tmx_file),
        "--source-manifest",
        str(config.source_manifest),
        "--output-dir",
        str(config.output_dir),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["counts"]["staged_pairs"] == 1
    assert "A small test" not in result.stdout
    rejected = cli(
        "--demo", "--tmx-file", str(config.tmx_file), "--output-dir", str(tmp_path / "bad")
    )
    assert rejected.returncode == 2 and "cannot be combined" in rejected.stderr
    assert not (tmp_path / "bad").exists()
    repeat = cli(
        "--tmx-file",
        str(config.tmx_file),
        "--source-manifest",
        str(config.source_manifest),
        "--output-dir",
        str(config.output_dir),
    )
    assert repeat.returncode == 2 and "already exists" in repeat.stderr


def test_thousands_of_synthetic_units_remain_counted_and_review_only(tmp_path: Path):
    root, body = new_document()
    for index in range(5000):
        add_unit(
            body,
            [("en", f"Synthetic example {index}."), ("ckb", f"دەقی تاقیکردنەوە {index}.")],
            unit_id=f"test-{index}",
        )
    config = make_config(tmp_path, xml_bytes(root))
    report = stage_tmx_pairs(config)
    assert report["counts"]["input_units"] == report["counts"]["staged_pairs"] == 5000
    assert report["counts"]["quarantined_units"] == 0
    assert len(read_jsonl(config.output_dir / "pairs.jsonl")) == 5000
    assert report["training_ready"] is False
