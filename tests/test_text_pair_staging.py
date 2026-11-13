"""Offline tests using independently authored synthetic examples only."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import ml.data.stage_text_pairs as staging
from ml.data.stage_text_pairs import TextPairStagingConfig, TextPairStagingError, stage_text_pairs
from ml.train.dataset import validate_splits


def synthetic_manifest() -> dict:
    return {
        "source_id": "test-source",
        "source_revision": "test-revision-1",
        "source_reference": "Independently authored synthetic examples",
        "provenance": "unknown",
        "rights": {
            "status": "synthetic",
            "affirmed_by_user": True,
            "original_text_evidence": "Synthetic test fixture",
            "translation_evidence": "Synthetic test fixture",
            "license_or_permission": "Test use only",
            "attribution": "No external source",
            "permitted_uses": ["synthetic_testing"],
        },
    }


def make_config(
    tmp_path: Path,
    english: bytes = b"Hello friend.\nPlease wait.\n",
    sorani: bytes = "سڵاو هاوڕێ.\nتکایە چاوەڕێ بکە.\n".encode(),
    manifest: dict | None = None,
    *,
    max_chars: int = 5000,
) -> TextPairStagingConfig:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    english_path, sorani_path, manifest_path = raw / "en.txt", raw / "ckb.txt", raw / "source.json"
    english_path.write_bytes(english)
    sorani_path.write_bytes(sorani)
    manifest_path.write_text(
        json.dumps(manifest or synthetic_manifest(), ensure_ascii=False), encoding="utf-8"
    )
    return TextPairStagingConfig(
        english_path, sorani_path, manifest_path, tmp_path / "out", max_chars
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def demo_cli(output: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.data.stage_text_pairs",
            "--demo",
            "--output-dir",
            str(output),
            *extra,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_offline_demo_stages_synthetic_pairs_but_cannot_feed_current_trainer(tmp_path: Path):
    output = tmp_path / "demo"
    result = demo_cli(output)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["counts"]["input_pairs"] == summary["counts"]["staged_pairs"] == 3
    assert summary["counts"]["quarantined_pairs"] == 0
    assert summary["training_ready"] is False
    rows = read_jsonl(output / "pairs.jsonl")
    report = json.loads((output / "staging-report.json").read_text(encoding="utf-8"))
    assert report["source"]["rights"]["status"] == "synthetic"
    assert report["training_ready"] is False
    for name in ("pairs", "quarantine"):
        actual_hash = hashlib.sha256((output / f"{name}.jsonl").read_bytes()).hexdigest()
        assert report["outputs"][f"{name}_sha256"] == actual_hash
    assert len({row["pair_id"] for row in rows}) == 3
    assert len({row["source_group_id"] for row in rows}) == 1
    assert [row["source_line"] for row in rows] == [1, 2, 3]
    for row in rows:
        assert row["record_type"] == "text_pair"
        assert row["source_context"] == []
        assert row["language_verified"] is False and row["alignment_verified"] is False
        assert row["original_direction"] == "unknown"
        assert not {"film_id", "english_start_ms", "sorani_start_ms", "speakers"} & row.keys()
    assert not (output / "dataset").exists()
    assert read_jsonl(output / "quarantine.jsonl") == []
    with pytest.raises(ValueError, match="missing fields"):
        validate_splits({"train": [rows[0]], "val": [rows[1]]}, allow_unverified=True)
    assert "Please open" not in result.stdout
    assert "Please open" not in (output / "staging-report.json").read_text(encoding="utf-8")


def test_demo_ids_are_reproducible_without_using_the_output_path(tmp_path: Path):
    assert demo_cli(tmp_path / "a").returncode == 0
    assert demo_cli(tmp_path / "b").returncode == 0
    one = read_jsonl(tmp_path / "a" / "pairs.jsonl")
    two = read_jsonl(tmp_path / "b" / "pairs.jsonl")
    assert [row["pair_id"] for row in one] == [row["pair_id"] for row in two]
    assert one[0]["source_group_id"] == two[0]["source_group_id"]


def test_blank_rows_quarantine_both_sides_without_shifting_and_preserve_zwnj(tmp_path: Path):
    english = b"\xef\xbb\xbfFirst sentence.\r\n\r\nThird sentence.\r\nFourth sentence.\r\n"
    sorani = "ئه\u200cم یەکەمە.\r\nدووەمە.\r\n\r\nچوارەمە.\r\n".encode()
    config = make_config(tmp_path, english, sorani)
    report = stage_text_pairs(config)
    assert config.english_file.read_bytes() == english
    assert config.sorani_file.read_bytes() == sorani
    rows = read_jsonl(config.output_dir / "pairs.jsonl")
    invalid = read_jsonl(config.output_dir / "quarantine.jsonl")
    assert [row["source_line"] for row in rows] == [1, 4]
    assert rows[1]["english"] == "Fourth sentence."
    assert rows[1]["sorani"] == "چوارەمە."
    assert "\u200c" in rows[0]["sorani"] and "\u200c" in rows[0]["original_sorani"]
    assert "ه" in rows[0]["sorani"] and "ambiguous_arabic_heh" in rows[0]["flags"]
    assert invalid[0]["source_line"] == 2 and invalid[0]["reasons"] == ["english_blank"]
    assert invalid[1]["source_line"] == 3 and invalid[1]["reasons"] == ["sorani_blank"]
    assert report["counts"]["input_pairs"] == 4
    assert report["counts"]["staged_pairs"] == 2
    assert report["counts"]["quarantined_pairs"] == 2
    assert "Third sentence" not in (config.output_dir / "quarantine.jsonl").read_text()


def test_invalid_utf8_control_and_long_lines_are_quarantined_atomically(tmp_path: Path):
    english = b"Hello.\ncorrupt\xff\ncontrol\x00\na very long line\nWorld.\n"
    sorani = "سڵاو.\nدەقی دوو.\nدەقی سێ.\nدەقی چوار.\n".encode() + b"bad\xff\n"
    config = make_config(tmp_path, english, sorani, max_chars=12)
    report = stage_text_pairs(config)
    assert report["counts"]["input_pairs"] == 5
    assert report["counts"]["staged_pairs"] == 1
    assert report["counts"]["quarantined_pairs"] == 4
    invalid = read_jsonl(config.output_dir / "quarantine.jsonl")
    assert [row["source_line"] for row in invalid] == [2, 3, 4, 5]
    assert "english_invalid_utf8" in invalid[0]["reasons"]
    assert "english_control_or_line_separator" in invalid[1]["reasons"]
    assert "english_too_long" in invalid[2]["reasons"]
    assert "sorani_invalid_utf8" in invalid[3]["reasons"]
    assert [row["source_line"] for row in read_jsonl(config.output_dir / "pairs.jsonl")] == [1]


def test_alignment_script_and_duplicate_warnings_do_not_delete_pairs(tmp_path: Path):
    english = b"HELLO\nhello\nsame\n123\nA\n"
    sorani = "سڵاه\nسڵاه\nsame\n456\nدەقێکی زۆر درێژ بۆ تاقیکردنەوە\n".encode()
    config = make_config(tmp_path, english, sorani)
    report = stage_text_pairs(config)
    rows = read_jsonl(config.output_dir / "pairs.jsonl")
    assert len(rows) == 5 and report["counts"]["staged_pairs"] == 5
    assert "ambiguous_arabic_heh" in rows[0]["flags"]
    assert "duplicate_pair" in rows[1]["flags"] and rows[1]["duplicate_of_line"] == 1
    assert "identical_text" in rows[2]["flags"]
    assert "sorani_without_arabic_letters" in rows[2]["flags"]
    assert "english_without_latin_letters" in rows[3]["flags"]
    assert "unusual_length_ratio" in rows[4]["flags"]
    assert report["counts"]["warning_counts"]["duplicate_pair"] == 1
    assert all(not row["alignment_verified"] for row in rows)


def test_different_physical_line_counts_fail_before_filtering_and_leave_no_output(tmp_path: Path):
    config = make_config(tmp_path, b"first\n\nlast\n", "یەک\nدوو\n".encode())
    with pytest.raises(TextPairStagingError, match="Physical line counts differ"):
        stage_text_pairs(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))
    assert config.english_file.read_bytes() == b"first\n\nlast\n"


@pytest.mark.parametrize("content", [b"", b"\xef\xbb\xbf"])
def test_empty_or_only_a_bom_produces_no_usable_pair(tmp_path: Path, content: bytes):
    config = make_config(tmp_path, content, content)
    if not content:
        with pytest.raises(TextPairStagingError, match="empty"):
            stage_text_pairs(config)
        assert not config.output_dir.exists()
    else:
        report = stage_text_pairs(config)
        assert report["counts"]["staged_pairs"] == 0
        assert report["counts"]["quarantined_pairs"] == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["rights"].update(status="not_reviewed"),
        lambda m: m["rights"].update(status=[]),
        lambda m: m["rights"].update(affirmed_by_user=False),
        lambda m: m["rights"].update(permitted_uses=[]),
        lambda m: m["rights"].update(permitted_uses=["commercial_training"]),
        lambda m: m["rights"].update(original_text_evidence=""),
        lambda m: m.update(provenance=[]),
        lambda m: m.update(source_revision=""),
    ],
)
def test_manifest_gate_rejects_missing_unreviewed_or_inconsistent_claims(tmp_path: Path, change):
    manifest = synthetic_manifest()
    change(manifest)
    config = make_config(tmp_path, manifest=manifest)
    with pytest.raises(TextPairStagingError):
        stage_text_pairs(config)
    assert not config.output_dir.exists()


def test_duplicate_json_keys_are_not_silently_accepted(tmp_path: Path):
    config = make_config(tmp_path)
    config.source_manifest.write_text('{"source_id":"a","source_id":"b"}', encoding="utf-8")
    with pytest.raises(TextPairStagingError, match="duplicate JSON key"):
        stage_text_pairs(config)
    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b'{"provenance":NaN}', "non-finite"),
        (b"[]", "JSON object"),
        (b"{" + b" " * (128 * 1024) + b"}", "too large"),
    ],
)
def test_invalid_manifest_is_reported_without_creating_output(tmp_path: Path, contents, message):
    config = make_config(tmp_path)
    config.source_manifest.write_bytes(contents)
    with pytest.raises(TextPairStagingError, match=message):
        stage_text_pairs(config)
    assert not config.output_dir.exists()


def test_unpaired_surrogate_in_manifest_is_rejected(tmp_path: Path):
    config = make_config(tmp_path)
    manifest = synthetic_manifest()
    manifest["source_id"] = "\ud800"
    config.source_manifest.write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")
    with pytest.raises(TextPairStagingError, match="valid Unicode"):
        stage_text_pairs(config)
    assert not config.output_dir.exists()


def test_attested_source_metadata_is_preserved_but_not_treated_as_verification(tmp_path: Path):
    manifest = synthetic_manifest()
    manifest["rights"].update(
        status="user_attested",
        permitted_uses=["commercial_training", "commercial_validation"],
        original_text_evidence="document:original",
        translation_evidence="document:translation",
        license_or_permission="document:license",
        attribution="document:attribution",
    )
    manifest["additional_conditions"] = "See the referenced agreement"
    config = make_config(tmp_path, manifest=manifest)
    report = stage_text_pairs(config)
    assert report["source"]["rights"] == manifest["rights"]
    assert report["source"]["source_reference"] == manifest["source_reference"]
    assert (
        config.output_dir / "source-manifest.json"
    ).read_bytes() == config.source_manifest.read_bytes()
    assert len(report["inputs"]["english"]["sha256"]) == 64
    assert len(report["inputs"]["sorani"]["sha256"]) == 64
    assert len(report["inputs"]["manifest_sha256"]) == 64
    assert report["training_ready"] is False
    assert all(
        not row["language_verified"] for row in read_jsonl(config.output_dir / "pairs.jsonl")
    )
    assert any("not verified legal clearance" in text for text in report["limitations"])


def test_ids_bind_source_revision_and_both_content_hashes(tmp_path: Path):
    config = make_config(tmp_path)
    original = stage_text_pairs(config)
    row_one = read_jsonl(config.output_dir / "pairs.jsonl")[0]
    config.english_file.write_bytes(b"Changed text.\nPlease wait.\n")
    changed_config = TextPairStagingConfig(
        config.english_file, config.sorani_file, config.source_manifest, tmp_path / "out-2"
    )
    changed = stage_text_pairs(changed_config)
    row_two = read_jsonl(changed_config.output_dir / "pairs.jsonl")[0]
    assert original["source_group_id"] == changed["source_group_id"]
    assert row_one["pair_id"] != row_two["pair_id"]
    manifest = synthetic_manifest()
    manifest["source_revision"] = "test-revision-2"
    config.source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    revised_config = TextPairStagingConfig(
        config.english_file, config.sorani_file, config.source_manifest, tmp_path / "out-3"
    )
    revised = stage_text_pairs(revised_config)
    revised_row = read_jsonl(revised_config.output_dir / "pairs.jsonl")[0]
    assert revised["source_group_id"] == changed["source_group_id"]
    assert revised_row["pair_id"] != row_two["pair_id"]


def test_existing_output_is_never_overwritten(tmp_path: Path):
    config = make_config(tmp_path)
    config.output_dir.mkdir()
    keep = config.output_dir / "keep.txt"
    keep.write_text("do not remove", encoding="utf-8")
    with pytest.raises(TextPairStagingError, match="already exists"):
        stage_text_pairs(config)
    assert keep.read_text(encoding="utf-8") == "do not remove"
    assert list(config.output_dir.iterdir()) == [keep]


def test_input_and_output_symlinks_and_same_files_are_rejected(tmp_path: Path):
    config = make_config(tmp_path)
    try:
        link = tmp_path / "linked-input"
        link.symlink_to(config.english_file)
        output_link = tmp_path / "linked-output"
        output_link.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this platform")
    with pytest.raises(TextPairStagingError, match="not a symlink"):
        stage_text_pairs(
            TextPairStagingConfig(
                link, config.sorani_file, config.source_manifest, config.output_dir
            )
        )
    with pytest.raises(TextPairStagingError, match="already exists"):
        stage_text_pairs(
            TextPairStagingConfig(
                config.english_file, config.sorani_file, config.source_manifest, output_link
            )
        )
    with pytest.raises(TextPairStagingError, match="distinct files"):
        stage_text_pairs(
            TextPairStagingConfig(
                config.english_file, config.english_file, config.source_manifest, config.output_dir
            )
        )
    assert not config.output_dir.exists()


def test_input_modified_after_preflight_is_detected(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    real_file_info = staging._file_info
    calls = 0

    def mutate_after_scan(path):
        nonlocal calls
        info = real_file_info(path)
        calls += 1
        if calls == 2:
            config.english_file.write_bytes(b"Changed text.\nPlease wait.\n")
        return info

    monkeypatch.setattr(staging, "_file_info", mutate_after_scan)
    with pytest.raises(TextPairStagingError, match="changed after"):
        stage_text_pairs(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_racing_output_creation_preserves_the_other_directory(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    real_publish = staging._publish

    def another_process_creates_output(stage, output):
        output.mkdir()
        (output / "keep.txt").write_text("someone else's data", encoding="utf-8")
        real_publish(stage, output)

    monkeypatch.setattr(staging, "_publish", another_process_creates_output)
    with pytest.raises(TextPairStagingError, match="already exists"):
        stage_text_pairs(config)
    assert (config.output_dir / "keep.txt").read_text() == "someone else's data"
    assert not list(tmp_path.glob(".out.staging-*"))


def test_failed_publication_removes_only_created_files(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    real_copy = staging.shutil.copyfileobj
    calls = 0

    def fail_second_copy(source, target, length=0):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write(b"partial")
            raise OSError("simulated copy failure")
        return real_copy(source, target, length=length)

    monkeypatch.setattr(staging.shutil, "copyfileobj", fail_second_copy)
    with pytest.raises(OSError, match="simulated copy failure"):
        stage_text_pairs(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_cli_accepts_explicit_local_synthetic_files(tmp_path: Path):
    config = make_config(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.data.stage_text_pairs",
            "--english-file",
            str(config.english_file),
            "--sorani-file",
            str(config.sorani_file),
            "--source-manifest",
            str(config.source_manifest),
            "--output-dir",
            str(config.output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["counts"]["staged_pairs"] == 2
    assert len(read_jsonl(config.output_dir / "pairs.jsonl")) == 2
    assert "Hello friend" not in result.stdout
    assert "سڵاو" not in result.stdout


def test_cli_rejects_mixing_demo_with_real_inputs_and_existing_output(tmp_path: Path):
    result = demo_cli(tmp_path / "bad", "--english-file", str(tmp_path / "x"))
    assert result.returncode == 2
    assert "cannot be combined" in result.stderr
    assert not (tmp_path / "bad").exists()
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("keep", encoding="utf-8")
    result = demo_cli(output)
    assert result.returncode == 2
    assert "already exists" in result.stderr
    assert (output / "keep").read_text() == "keep"


def test_large_synthetic_input_reports_every_line(tmp_path: Path):
    count = 5000
    english = "".join(f"Example number {index}.\n" for index in range(count)).encode()
    sorani = "".join(f"دەقی تاقیکردنەوەی ژمارە {index}.\n" for index in range(count)).encode()
    config = make_config(tmp_path, english, sorani)
    report = stage_text_pairs(config)
    assert report["counts"]["input_pairs"] == count
    assert report["counts"]["staged_pairs"] == count
    assert report["counts"]["quarantined_pairs"] == 0
    assert len(read_jsonl(config.output_dir / "pairs.jsonl")) == count
