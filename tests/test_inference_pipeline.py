import json
from pathlib import Path

import pytest

from ml.inference.base import BaseTranslator, TranslationResult
from ml.inference.glossary import GlossaryEntry
from ml.inference.pipeline import PipelineOptions, _write_new_json, translate_subtitles


class RecordingTranslator(BaseTranslator):
    def __init__(self, direction="en-ckb"):
        self.direction = direction
        self.model_version_id = "unit-test-model"
        self.requests = []

    def translate_batch(self, requests):
        self.requests.extend(requests)
        outputs = []
        for request in requests:
            if self.direction == "ckb-en":
                text = "The car is here."
            elif "friend" in request.text:
                text = "سڵاو هاوڕێ." if request.force_terms else "سڵاو friend."
            elif "bridge" in request.text.lower():
                text = "پردەکە لە پێشەوەیە."
            else:
                text = "تکایە چاوەڕێ بکە."
            outputs.append(
                TranslationResult(text, (-0.4, -0.8), {"constrained": bool(request.force_terms)})
            )
        return outputs


@pytest.fixture
def subtitle(tmp_path: Path) -> Path:
    path = tmp_path / "original.srt"
    path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nJOHN: Hello,\n\n"
        "2\n00:00:02,000 --> 00:00:03,250\nJOHN: my friend.\n\n"
        "3\n00:00:03,200 --> 00:00:04,700\nJOHN: Please wait.\n\n"
        "4\n00:00:05,000 --> 00:00:06,000\n[DOOR SLAMS]\n\n"
        "5\n00:00:06,000 --> 00:00:07,500\nThe bridge\n\n"
        "6\n00:00:07,500 --> 00:00:09,000\nis near.\n",
        encoding="utf-8",
    )
    return path


def test_twelve_stages_context_glossary_offsets_and_original_timings(subtitle):
    translator = RecordingTranslator()
    events = []
    result = translate_subtitles(
        subtitle,
        translator,
        glossary=[GlossaryEntry("friend", "هاوڕێ", True)],
        options=PipelineOptions(batch_size=2),
        progress=events.append,
    )
    assert [cue["start_ms"] for cue in result["cues"]] == [1000, 2000, 3200, 5000, 6000, 7500]
    assert [cue["end_ms"] for cue in result["cues"]] == [2000, 3250, 4700, 6000, 7500, 9000]
    assert result["sentences"][0]["source"] == "Hello, my friend."
    assert result["sentences"][0]["cue_spans"] == [
        {"cue_index": 1, "start": 0, "end": 6},
        {"cue_index": 2, "start": 7, "end": 17},
    ]
    assert result["cues"][1]["source_cue_ids"] == [2]
    assert result["cues"][1]["sentence_source_cue_ids"] == [1, 2]
    assert result["cues"][1]["source_offsets"] == [7, 17]
    for cue in result["cues"]:
        if cue["sentence_id"] is not None:
            sentence = result["sentences"][cue["sentence_id"] - 1]["source"]
            start, end = cue["source_offsets"]
            assert sentence[start:end] == " ".join(cue["normalized_source_text"].split())
    assert result["sentences"][0]["enforcement"] == ["constrained_retry_used"]
    assert any(request.force_terms == ("هاوڕێ",) for request in translator.requests)
    assert translator.requests[1].context == ("[JOHN] Hello, my friend.",)
    assert result["cues"][3]["sound_effects"] == ["DOOR SLAMS"]
    assert "sfx_untranslated" in result["cues"][3]["qa_warnings"]
    assert "original_timing_overlap" in result["cues"][1]["qa_warnings"]
    assert all(
        cue["model_version_id"] == "unit-test-model" and cue["human_text"] is None
        for cue in result["cues"]
    )
    assert all(cue["revision_history"] == [] for cue in result["cues"])
    assert {event["stage_number"] for event in events} == set(range(1, 13))
    assert [event["percent"] for event in events] == sorted(event["percent"] for event in events)
    assert events[-1]["percent"] == 100


def test_reverse_direction_and_cancellation_are_supported(tmp_path):
    path = tmp_path / "source.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:04,000\nسەیارە لێرەیە.\n", encoding="utf-8")
    translator = RecordingTranslator("ckb-en")
    result = translate_subtitles(path, translator, glossary=[GlossaryEntry("car", "سەیارە")])
    assert result["cues"][0]["machine_text"] == "The car is here."
    assert translator.requests[0].glossary_hints == (("سەیارە", "car"),)

    def cancel(event):
        if event["stage"] == "reconstruct_sentences":
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        translate_subtitles(path, translator, progress=cancel)


def test_opt_in_softening_uses_only_user_rules_and_flags_changes(tmp_path):
    source = tmp_path / "source.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:04,000\nسەیارە لێرەیە.\n", encoding="utf-8")
    rules = (("car", "vehicle"),)
    unchanged = translate_subtitles(
        source, RecordingTranslator("ckb-en"), options=PipelineOptions(profanity_replacements=rules)
    )
    assert unchanged["cues"][0]["machine_text"] == "The car is here."
    softened = translate_subtitles(
        source,
        RecordingTranslator("ckb-en"),
        glossary=[GlossaryEntry("car", "سەیارە")],
        options=PipelineOptions(soften_profanity=True, profanity_replacements=rules),
    )
    assert softened["cues"][0]["machine_text"] == "The vehicle is here."
    assert "profanity_softened_needs_review" in softened["cues"][0]["qa_warnings"]
    assert "glossary_mismatch" in softened["cues"][0]["qa_warnings"]
    with pytest.raises(ValueError, match="requires"):
        translate_subtitles(
            source, RecordingTranslator("ckb-en"), options=PipelineOptions(soften_profanity=True)
        )


def test_failed_constraint_uses_only_safe_literal_replacement(subtitle):
    class NoConstraints(RecordingTranslator):
        def translate_batch(self, requests):
            if any(request.force_terms for request in requests):
                raise ValueError("unsupported")
            return super().translate_batch(requests)

    result = translate_subtitles(
        subtitle, NoConstraints(), glossary=[GlossaryEntry("friend", "هاوڕێ")]
    )
    assert "constrained_retry_failed" in result["sentences"][0]["enforcement"]
    assert "literal_replacement_needs_review" in result["sentences"][0]["enforcement"]
    assert all("glossary_mismatch" not in cue["qa_warnings"] for cue in result["cues"][:2])
    assert all(
        "literal_replacement_needs_review" in cue["qa_warnings"] for cue in result["cues"][:2]
    )


def test_sfx_only_cue_never_calls_the_model(tmp_path):
    source = tmp_path / "sfx.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:03,000\n[WIND BLOWS]\n", encoding="utf-8")
    translator = RecordingTranslator()
    result = translate_subtitles(source, translator)
    assert not translator.requests and result["sentences"] == []
    assert "WIND" in result["cues"][0]["machine_text"]
    assert "sfx_untranslated" in result["cues"][0]["qa_warnings"]


def test_out_of_order_timestamps_are_flagged_without_shifting(tmp_path):
    path = tmp_path / "unordered.srt"
    path.write_text(
        "1\n00:00:05,000 --> 00:00:06,000\nFirst.\n\n2\n00:00:01,000 --> 00:00:02,000\nSecond.\n",
        encoding="utf-8",
    )
    result = translate_subtitles(path, RecordingTranslator())
    times = [(cue["start_ms"], cue["end_ms"]) for cue in result["cues"]]
    assert times == [(5000, 6000), (1000, 2000)]
    assert "original_timing_out_of_order" in result["cues"][1]["qa_warnings"]


def test_sound_effect_does_not_mask_missing_dialogue(tmp_path):
    class EmptyTranslator(RecordingTranslator):
        def translate_batch(self, requests):
            return [TranslationResult("") for request in requests]

    path = tmp_path / "dialogue-and-effect.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:04,000\nHello [KNOCK]\n", encoding="utf-8")
    result = translate_subtitles(path, EmptyTranslator())
    assert "KNOCK" in result["cues"][0]["machine_text"]
    assert "missing_translation" in result["cues"][0]["qa_warnings"]


def test_register_conflicts_are_flagged_not_silently_rewritten(tmp_path):
    class RegisterTranslator(RecordingTranslator):
        def translate_batch(self, requests):
            return [TranslationResult("تۆ و ئێوە.", (-0.1,)) for request in requests]

    path = tmp_path / "register.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:04,000\nJOHN: Hello.\n", encoding="utf-8")
    result = translate_subtitles(
        path, RegisterTranslator(), options=PipelineOptions(register="formal")
    )
    assert "register_needs_review" in result["cues"][0]["qa_warnings"]
    assert "تۆ" in result["cues"][0]["machine_text"] and "ئێوە" in result["cues"][0]["machine_text"]


def test_json_output_is_exclusive_and_cannot_overlap_input(subtitle, tmp_path):
    target = tmp_path / "result.json"
    _write_new_json(target, {"text": "سڵاو"}, [subtitle])
    assert json.loads(target.read_text())["text"] == "سڵاو"
    with pytest.raises(ValueError, match="already exists"):
        _write_new_json(target, {}, [subtitle])
    with pytest.raises(ValueError, match="already exists"):
        _write_new_json(subtitle, {}, [subtitle])
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "not-created.json")
    with pytest.raises(ValueError, match="symlink"):
        _write_new_json(link, {}, [subtitle])
    assert subtitle.exists()


def test_input_change_during_parsing_is_detected(subtitle, monkeypatch):
    from ml.inference import pipeline

    fingerprints = iter(("before", "after"))
    monkeypatch.setattr(pipeline, "_sha256", lambda path: next(fingerprints))
    with pytest.raises(ValueError, match="changed while it was being parsed"):
        translate_subtitles(subtitle, RecordingTranslator())


def test_pipeline_rejects_invalid_options_and_wrong_result_count(subtitle):
    translator = RecordingTranslator()
    with pytest.raises(ValueError, match="CPS"):
        translate_subtitles(
            subtitle, translator, options=PipelineOptions(target_cps=23, hard_cps=21)
        )
    with pytest.raises(ValueError, match="CPS"):
        translate_subtitles(subtitle, translator, options=PipelineOptions(target_cps=float("nan")))
    with pytest.raises(ValueError, match="more than once"):
        translate_subtitles(
            subtitle,
            translator,
            glossary=[GlossaryEntry("car", "سەیارە"), GlossaryEntry("CAR", "ئۆتۆمبێل")],
        )

    class BadTranslator(RecordingTranslator):
        def translate_batch(self, requests):
            return []

    with pytest.raises(ValueError, match="different number"):
        translate_subtitles(subtitle, BadTranslator())
