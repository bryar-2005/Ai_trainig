"""Twelve-stage, timestamp-preserving subtitle translation orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import regex

from ml.data.loaders import Cue, load_subtitles
from ml.data.normalize import normalize_english, normalize_sorani
from ml.inference.base import BaseTranslator, TranslationRequest, TranslationResult
from ml.inference.compressor import compress
from ml.inference.glossary import (
    EntityExtractor,
    GlossaryEntry,
    HeuristicEntityExtractor,
    TransformersEntityExtractor,
    load_glossary,
    merge_entities,
    missing_locked,
    relevant_locked,
    replace_literal_sources,
)
from ml.inference.qa import score_cue, summarize_qa
from ml.inference.segmenter import Sentence, reconstruct_sentences, split_translation
from ml.inference.text import isolate_ltr, normalize_target, wrap_lines


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    target_cps: float = 17.0
    hard_cps: float = 21.0
    max_chars: int = 42
    max_lines: int = 2
    context_sentences: int = 3
    batch_size: int = 4
    register: str = "auto"
    soften_profanity: bool = False
    profanity_replacements: tuple[tuple[str, str], ...] = ()

    def validate(self) -> None:
        speeds = (self.target_cps, self.hard_cps)
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in speeds
            )
            or not 0 < self.target_cps <= self.hard_cps
        ):
            raise ValueError(
                "CPS limits must be finite, positive and target must not exceed hard cap"
            )
        integers = (self.max_chars, self.max_lines, self.context_sentences, self.batch_size)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise ValueError("Line, context and batch limits must be integers")
        if self.max_chars < 1 or not 1 <= self.max_lines <= 2:
            raise ValueError("Line limits must be positive and at most two lines")
        if not 3 <= self.context_sentences <= 5 or self.batch_size < 1:
            raise ValueError("Context must be 3–5 sentences and batch size must be positive")
        if not isinstance(self.register, str) or self.register not in {
            "auto",
            "formal",
            "informal",
        }:
            raise ValueError("register must be auto, formal or informal")
        if not isinstance(self.soften_profanity, bool):
            raise ValueError("soften_profanity must be a boolean")
        if (
            not isinstance(self.profanity_replacements, tuple)
            or len(self.profanity_replacements) > 50
            or any(
                not isinstance(rule, tuple)
                or len(rule) != 2
                or any(
                    not isinstance(text, str)
                    or not text.strip()
                    or len(text) > 120
                    or "\n" in text
                    or "\r" in text
                    for text in rule
                )
                for rule in self.profanity_replacements
            )
        ):
            raise ValueError("Softening replacements must be bounded nonempty text pairs")
        if self.soften_profanity and not self.profanity_replacements:
            raise ValueError("Softening requires explicit target-language replacement rules")
        keys = [source.casefold() for source, _ in self.profanity_replacements]
        if len(keys) != len(set(keys)):
            raise ValueError("Softening source terms must be unique")


ProgressCallback = Callable[[dict[str, Any]], None]
_STAGES = (
    "parse",
    "extract_metadata",
    "reconstruct_sentences",
    "extract_glossary",
    "translate",
    "enforce_glossary",
    "resplit",
    "compress",
    "line_break",
    "normalize",
    "quality_assurance",
    "emit",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_results(results: list[TranslationResult], count: int) -> list[TranslationResult]:
    if not isinstance(results, list) or len(results) != count:
        raise ValueError("Translator returned a different number of results than requests")
    for result in results:
        if (
            not isinstance(result, TranslationResult)
            or not isinstance(result.text, str)
            or not isinstance(result.metadata, dict)
        ):
            raise ValueError("Translator returned an invalid result")
        if any(
            not isinstance(score, (int, float)) or not math.isfinite(score) or score > 0
            for score in result.token_logprobs
        ):
            raise ValueError("Translator returned invalid token log probabilities")
    return results


def _speaker_context(sentence: Sentence) -> str:
    return f"[{sentence.speaker}] {sentence.text}" if sentence.speaker else sentence.text


def _overlap_ids(cues: list[Cue]) -> set[int]:
    overlaps: set[int] = set()
    active_end = -1
    active_id = -1
    for cue in sorted(cues, key=lambda item: (item.start_ms, item.end_ms, item.index)):
        if cue.start_ms < active_end:
            overlaps.update((active_id, cue.index))
        if cue.end_ms > active_end:
            active_end, active_id = cue.end_ms, cue.index
    return overlaps


def _register_warnings(
    cues: list[Cue], texts: dict[int, str], direction: str, register: str
) -> set[int]:
    if direction != "en-ckb":
        return set()
    forms: dict[str, set[str]] = {}
    cue_forms: dict[int, set[str]] = {}
    for cue in cues:
        if not cue.speaker:
            continue
        text = texts[cue.index]
        found = set()
        if regex.search(r"(?<!\p{L})تۆ(?!\p{L})", text):
            found.add("informal")
        if regex.search(r"(?<!\p{L})ئێوە(?!\p{L})", text):
            found.add("formal")
        cue_forms[cue.index] = found
        forms.setdefault(cue.speaker.casefold(), set()).update(found)
    flagged: set[int] = set()
    for cue in cues:
        if not cue.speaker:
            continue
        found = cue_forms.get(cue.index, set())
        if register == "auto" and len(forms[cue.speaker.casefold()]) > 1 and found:
            flagged.add(cue.index)
        elif register != "auto" and found - {register}:
            flagged.add(cue.index)
    return flagged


def translate_subtitles(
    input_path: str | Path,
    translator: BaseTranslator,
    *,
    glossary: list[GlossaryEntry] | None = None,
    extractor: EntityExtractor | None = None,
    options: PipelineOptions | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Translate a local subtitle file and return auditable JSON-ready metadata.

    A callback may raise to cancel; no output file is written by this function.
    SFX stay in their original language and are explicitly flagged for review.
    """
    options = options or PipelineOptions()
    options.validate()
    direction = getattr(translator, "direction", None)
    version_id = getattr(translator, "model_version_id", None)
    if (
        direction not in {"en-ckb", "ckb-en"}
        or not isinstance(version_id, str)
        or not version_id.strip()
    ):
        raise ValueError("Translator needs a valid direction and model version ID")
    available_context = getattr(translator, "context_window", options.context_sentences)
    if not isinstance(available_context, int) or available_context < options.context_sentences:
        raise ValueError("Requested context window exceeds the translator's supported window")
    path = Path(input_path).expanduser()

    def notify(index: int, fraction: float = 1.0, detail: str = "") -> None:
        if progress:
            progress(
                {
                    "stage": _STAGES[index - 1],
                    "stage_number": index,
                    "stage_count": len(_STAGES),
                    "stage_fraction": fraction,
                    "percent": round(100 * (index - 1 + fraction) / len(_STAGES), 2),
                    "detail": detail,
                }
            )

    notify(1, 0.0)
    digest = _sha256(path)
    original = load_subtitles(path)
    if _sha256(path) != digest:
        raise ValueError("Input subtitle changed while it was being parsed")
    notify(1, detail=f"Parsed {len(original)} cues")
    notify(2, detail=f"Extracted {sum(len(cue.sound_effects) for cue in original)} sound effects")
    normalizer = normalize_english if direction == "en-ckb" else normalize_sorani
    cues = [replace(cue, text=normalizer(cue.text)) for cue in original]
    sentences = reconstruct_sentences(cues)
    notify(3, detail=f"Reconstructed {len(sentences)} source sentences")
    extractor = extractor or HeuristicEntityExtractor()
    ner_texts = [sentence.text for sentence in sentences]
    ner_texts.extend(label for cue in cues for label in cue.speaker_labels)
    entities = extractor.extract(ner_texts)
    terms = merge_entities(glossary or [], entities)
    locked_sources: set[str] = set()
    for term in terms:
        if not term.locked:
            continue
        source = term.english if direction == "en-ckb" else term.sorani
        key = " ".join(source.casefold().split())
        if key in locked_sources:
            raise ValueError(f"Multiple locked terms have the same source: {source!r}")
        locked_sources.add(key)
    notify(4, detail=f"Prepared {len(terms)} glossary entries")

    requests: list[TranslationRequest] = []
    matching: list[list[GlossaryEntry]] = []
    for index, sentence in enumerate(sentences):
        applicable = relevant_locked(sentence.text, terms, direction)
        matching.append(applicable)
        hints = tuple(
            (item.english, item.sorani) if direction == "en-ckb" else (item.sorani, item.english)
            for item in applicable
        )
        requests.append(
            TranslationRequest(
                text=sentence.text,
                speaker=sentence.speaker,
                context=tuple(
                    _speaker_context(item)
                    for item in sentences[max(0, index - options.context_sentences) : index]
                ),
                glossary_hints=hints,
            )
        )
    generated: list[TranslationResult] = []
    for start in range(0, len(requests), options.batch_size):
        batch = requests[start : start + options.batch_size]
        generated.extend(_checked_results(translator.translate_batch(batch), len(batch)))
        notify(
            5,
            min(1.0, len(generated) / len(requests)),
            f"Translated {len(generated)} of {len(requests)} sentences",
        )
    if not requests:
        notify(5, detail="No spoken dialogue to translate")

    enforcement: list[list[str]] = [[] for _ in sentences]
    for index, result in enumerate(generated):
        applicable = matching[index]
        missing = missing_locked(normalize_target(result.text, direction), applicable, direction)
        if not missing:
            continue
        retry = replace(
            requests[index],
            force_terms=tuple(target for _, target in requests[index].glossary_hints),
        )
        try:
            alternative = _checked_results(translator.translate_batch([retry]), 1)[0]
            if len(
                missing_locked(normalize_target(alternative.text, direction), applicable, direction)
            ) < len(missing):
                generated[index] = alternative
                result = alternative
                enforcement[index].append("constrained_retry_used")
        except (ValueError, RuntimeError) as exc:
            enforcement[index].append("constrained_retry_failed")
            enforcement[index].append(f"retry_error:{type(exc).__name__}")
        missing = missing_locked(normalize_target(result.text, direction), applicable, direction)
        if missing:
            replacement = replace_literal_sources(result.text, missing, direction)
            if replacement != result.text:
                generated[index] = replace(
                    result,
                    text=replacement,
                    metadata={**result.metadata, "raw_generated_text": result.text},
                )
                enforcement[index].append("literal_replacement_needs_review")
    notify(6, detail="Checked locked terms and retried missing constraints")

    by_cue: dict[int, str] = {cue.index: "" for cue in cues}
    sentence_for_cue: dict[int, int] = {}
    span_for_cue: dict[int, tuple[int, int]] = {}
    for index, (sentence, result) in enumerate(zip(sentences, generated, strict=True)):
        for span, text in zip(
            sentence.spans, split_translation(sentence, result.text), strict=True
        ):
            by_cue[span.cue_index] = text
            sentence_for_cue[span.cue_index] = index
            span_for_cue[span.cue_index] = (span.start, span.end)
    empty_translations = {
        cue.index for cue in cues if cue.text.strip() and not by_cue[cue.index].strip()
    }
    notify(7, detail="Re-split on word boundaries without changing timing")

    compression_rules: dict[int, tuple[str, ...]] = {}
    for cue in cues:
        effects = " ".join(f"[{effect}]" for effect in cue.sound_effects)
        text = " ".join(part for part in (by_cue[cue.index], effects) if part)
        sentence_index = sentence_for_cue.get(cue.index)
        protected = (
            tuple(target for _, target in requests[sentence_index].glossary_hints)
            if sentence_index is not None
            else ()
        )
        compressed = compress(
            text,
            cue.end_ms - cue.start_ms,
            direction=direction,
            target_cps=options.target_cps,
            protected_terms=protected,
        )
        by_cue[cue.index] = compressed.text
        compression_rules[cue.index] = compressed.applied_rules
    softened: set[int] = set()
    if options.soften_profanity:
        for cue in cues:
            text = by_cue[cue.index]
            for source, replacement in options.profanity_replacements:
                source = normalize_target(source, direction)
                replacement = normalize_target(replacement, direction)
                boundary = r"[\p{L}\p{M}\p{N}_]"
                pattern = rf"(?<!{boundary}){regex.escape(source)}(?!{boundary})"
                updated = regex.sub(
                    pattern, lambda _, value=replacement: value, text, flags=regex.IGNORECASE
                )
                if updated != text:
                    softened.add(cue.index)
                    text = updated
            by_cue[cue.index] = text
    notify(8, detail="Applied conservative compression and any user-specified softening rules")
    for cue in cues:
        by_cue[cue.index] = wrap_lines(
            by_cue[cue.index], max_chars=options.max_chars, max_lines=options.max_lines
        )
    notify(9, detail="Balanced lines without splitting words")
    for cue in cues:
        text = normalize_target(by_cue[cue.index], direction)
        by_cue[cue.index] = isolate_ltr(text) if direction == "en-ckb" else text
    notify(10, detail="Normalized target text and isolated embedded Latin/numbers")

    group_missing: dict[int, tuple[str, ...]] = {}
    for index, sentence in enumerate(sentences):
        final_text = " ".join(by_cue[cue_id] for cue_id in sentence.cue_ids)
        group_missing[index] = tuple(
            item.sorani if direction == "en-ckb" else item.english
            for item in missing_locked(final_text, matching[index], direction)
        )
    overlaps = _overlap_ids(original)
    out_of_order = {
        current.index
        for previous, current in zip(original[:-1], original[1:], strict=True)
        if current.start_ms < previous.start_ms
    }
    register_flags = _register_warnings(cues, by_cue, direction, options.register)
    original_by_id = {cue.index: cue for cue in original}
    output_cues: list[dict[str, Any]] = []
    for cue in cues:
        sentence_index = sentence_for_cue.get(cue.index)
        result = generated[sentence_index] if sentence_index is not None else TranslationResult("")
        extra: list[str] = []
        if cue.index in empty_translations:
            extra.append("missing_translation")
        if cue.sound_effects:
            extra.append("sfx_untranslated")
        if cue.index in out_of_order:
            extra.append("original_timing_out_of_order")
        if cue.removed_ad_lines:
            extra.append("source_ad_removed")
        if compression_rules[cue.index]:
            extra.append("compression_needs_review")
        if cue.index in softened:
            extra.append("profanity_softened_needs_review")
        if cue.index in register_flags:
            extra.append("register_needs_review")
        if result.metadata.get("source_truncated"):
            extra.append("source_truncated")
        if result.metadata.get("context_truncated"):
            extra.append("context_truncated")
        if result.metadata.get("invalid_scores"):
            extra.append("invalid_model_scores")
        if (
            sentence_index is not None
            and "literal_replacement_needs_review" in enforcement[sentence_index]
        ):
            extra.append("literal_replacement_needs_review")
        missing = group_missing.get(sentence_index, ())
        quality = score_cue(
            source=cue.text,
            target=by_cue[cue.index],
            duration_ms=cue.end_ms - cue.start_ms,
            direction=direction,
            token_logprobs=result.token_logprobs,
            missing_terms=missing,
            target_cps=options.target_cps,
            hard_cps=options.hard_cps,
            max_chars=options.max_chars,
            max_lines=options.max_lines,
            timing_overlap=cue.index in overlaps,
            extra_warnings=tuple(extra),
        )
        original_cue = original_by_id[cue.index]
        output_cues.append(
            {
                "cue_id": cue.index,
                "source_cue_ids": [cue.index],
                "sentence_source_cue_ids": list(sentences[sentence_index].cue_ids)
                if sentence_index is not None
                else [cue.index],
                "sentence_id": sentence_index + 1 if sentence_index is not None else None,
                "source_offsets": list(span_for_cue[cue.index])
                if cue.index in span_for_cue
                else None,
                "start_ms": cue.start_ms,
                "end_ms": cue.end_ms,
                "source_text": original_cue.text,
                "normalized_source_text": cue.text,
                "speaker": cue.speaker,
                "speaker_labels": list(cue.speaker_labels),
                "sound_effects": list(cue.sound_effects),
                "style": cue.style,
                "machine_text": by_cue[cue.index],
                "human_text": None,
                "revision_history": [],
                "model_version_id": translator.model_version_id,
                "confidence_score": quality.confidence_score,
                "cps": quality.cps,
                "line_lengths": list(quality.line_lengths),
                "length_ratio": quality.length_ratio,
                "mean_token_logprob": quality.mean_token_logprob,
                "glossary_mismatches": list(missing),
                "compression_rules": list(compression_rules[cue.index]),
                "qa_warnings": list(quality.warnings),
                "flagged": quality.flagged,
            }
        )
    notify(11, detail="Scored cues and retained original timing warnings")
    report = {
        "schema_version": 1,
        "direction": direction,
        "input_sha256": digest,
        "model_version_id": translator.model_version_id,
        "model": getattr(
            translator, "model_info", {"model_version_id": translator.model_version_id}
        ),
        "options": asdict(options),
        "entity_extractor": {
            "type": type(extractor).__name__,
            "model": getattr(extractor, "model_source", None),
            "revision": getattr(extractor, "model_revision", None),
        },
        "glossary": [asdict(term) for term in terms],
        "sentences": [
            {
                "sentence_id": index + 1,
                "source": sentence.text,
                "cue_spans": [asdict(span) for span in sentence.spans],
                "translation_before_split": generated[index].text,
                "token_logprobs": list(generated[index].token_logprobs),
                "logprob_basis": "raw model output before deterministic post-processing",
                "generation": generated[index].metadata,
                "enforcement": enforcement[index],
            }
            for index, sentence in enumerate(sentences)
        ],
        "cues": output_cues,
        "qa_summary": summarize_qa(output_cues),
        "warnings": [
            "Automatic transliterations and compression require human review.",
            "SFX annotations remain in the source language and must be reviewed.",
            "CPS and confidence are heuristics; timestamps have not been shifted.",
        ],
    }
    notify(12, detail=f"Prepared {len(output_cues)} cues")
    return report


def _output_path(path: Path, inputs: list[Path]) -> Path:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError("Output may not be a symlink")
    target = requested.resolve()
    if target.exists():
        raise ValueError("Output file already exists")
    for item in inputs:
        source = item.expanduser().resolve()
        if source == target or (source.is_dir() and target.is_relative_to(source)):
            raise ValueError("Output must not overlap an input file or model directory")
    return target


def _write_new_json(path: Path, value: dict[str, Any], inputs: list[Path]) -> None:
    target = _output_path(path, inputs)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            temp = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, target)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Local SRT, VTT, ASS or SSA subtitle"
    )
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file")
    parser.add_argument("--direction", choices=["en-ckb", "ckb-en"], required=True)
    parser.add_argument("--model-version-id", required=True, help="Stable model version identifier")
    parser.add_argument(
        "--model", help="Base or merged local/Hugging Face model; optional with adapter"
    )
    parser.add_argument("--model-revision", help="Optional immutable 40-character Hub commit")
    parser.add_argument("--adapter-dir", type=Path, help="Completed Phase 2 LoRA adapter directory")
    parser.add_argument("--glossary", type=Path, help="JSON array of user glossary entries")
    parser.add_argument("--ner-model", help="Optional token-classification model for English NER")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-smoke-model", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-cps", type=float, default=17.0)
    parser.add_argument("--hard-cps", type=float, default=21.0)
    parser.add_argument("--max-chars", type=int, default=42)
    parser.add_argument("--max-lines", type=int, choices=[1, 2], default=2)
    parser.add_argument("--context-sentences", type=int, choices=[3, 4, 5], default=3)
    parser.add_argument("--register", choices=["auto", "formal", "informal"], default="auto")
    args = parser.parse_args(argv)
    try:
        inputs = [args.input] + [
            item for item in (args.adapter_dir, args.glossary) if item is not None
        ]
        for model in (args.model, args.ner_model):
            if model and Path(model).expanduser().is_dir():
                inputs.append(Path(model))
        _output_path(args.output, inputs)
        if not args.input.is_file():
            raise ValueError("Input subtitle file does not exist")
        options = PipelineOptions(
            target_cps=args.target_cps,
            hard_cps=args.hard_cps,
            max_chars=args.max_chars,
            max_lines=args.max_lines,
            context_sentences=args.context_sentences,
            batch_size=args.batch_size,
            register=args.register,
        )
        options.validate()
        terms = load_glossary(args.glossary) if args.glossary else None
        from ml.inference.nllb_translator import NllbTranslator

        translator = NllbTranslator(
            direction=args.direction,
            model_version_id=args.model_version_id,
            model=args.model,
            adapter_dir=args.adapter_dir,
            model_revision=args.model_revision,
            local_files_only=args.local_files_only,
            allow_smoke_model=args.allow_smoke_model,
            device=args.device,
            batch_size=args.batch_size,
            context_window=args.context_sentences,
        )
        for resolved_model in (
            translator.model_info["source"],
            translator.model_info["resolved_source"],
        ):
            if Path(resolved_model).expanduser().is_dir():
                inputs.append(Path(resolved_model))
        _output_path(args.output, inputs)
        extractor = (
            TransformersEntityExtractor(
                args.ner_model, local_files_only=args.local_files_only, device=args.device
            )
            if args.ner_model
            else None
        )
        if extractor and Path(extractor.resolved_source).expanduser().is_dir():
            inputs.append(Path(extractor.resolved_source))
        _output_path(args.output, inputs)
        report = translate_subtitles(
            args.input,
            translator,
            glossary=terms,
            extractor=extractor,
            options=options,
            progress=lambda event: print(json.dumps(event, ensure_ascii=False), file=sys.stderr),
        )
        _write_new_json(args.output, report, inputs)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Translation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cue_count": len(report["cues"]),
                "qa": report["qa_summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
