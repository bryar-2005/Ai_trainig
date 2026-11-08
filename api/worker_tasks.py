"""Real translation, reviewed-feedback training, and model comparison jobs."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import math
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.config import Settings
from api.db.models import Cue, Job, MediaFile, ModelVersion, Project, User
from api.review import cue_snapshot, log_change, recalculate_cue, terms_for, update_project_status
from api.storage import create_storage
from api.worker import JobCancelled, PreparedWork, Reporter
from ml.data.loaders import load_subtitles
from ml.data.normalize import normalize_english, normalize_sorani
from ml.inference.base import BaseTranslator, TranslationRequest, TranslationResult
from ml.inference.pipeline import PipelineOptions, translate_subtitles
from ml.inference.qa import summarize_qa
from ml.inference.segmenter import reconstruct_sentences


def _public_model(info: dict) -> dict:
    return {
        key: value
        for key, value in info.items()
        if key not in {"source", "resolved_source", "adapter"}
    }


def _translator(model: ModelVersion, settings: Settings):
    from ml.inference.nllb_translator import NllbTranslator

    return NllbTranslator(
        direction=model.direction,
        model_version_id=str(model.id),
        model=model.source,
        adapter_dir=model.adapter_dir,
        model_revision=model.revision if not model.adapter_dir else None,
        local_files_only=settings.local_files_only,
        allow_smoke_model=settings.allow_smoke_models,
        device=settings.inference_device,
    )


async def _job_and_model(
    db: AsyncSession, job_id: uuid.UUID, model_id: str
) -> tuple[Job, ModelVersion]:
    job = await db.get(Job, job_id)
    model = await db.get(ModelVersion, uuid.UUID(model_id))
    owner = await db.get(User, job.owner_id) if job else None
    if not job or not model or not owner or not owner.is_active:
        raise ValueError("The job, owner, or model is unavailable")
    if model.owner_id is not None and model.owner_id != owner.id and owner.role != "admin":
        raise ValueError("The selected model is not available to this user")
    return job, model


class _SelectiveTranslator(BaseTranslator):
    """Translate only the target reconstructed sentence but keep full source context."""

    def __init__(self, translator: BaseTranslator, sentence: str, extra: tuple[str, ...], hint):
        self.translator = translator
        self.sentence = sentence
        self.extra = extra
        self.hint = hint
        self.direction = translator.direction
        self.model_version_id = translator.model_version_id
        self.model_info = translator.model_info
        self.context_window = translator.context_window

    def translate_batch(self, requests: list[TranslationRequest]) -> list[TranslationResult]:
        results: list[TranslationResult | None] = [None] * len(requests)
        indices, selected = [], []
        for index, request in enumerate(requests):
            if request.text != self.sentence:
                results[index] = TranslationResult("", metadata={"not_requested": True})
                continue
            hints = request.glossary_hints
            if self.hint:
                pair = (self.hint["english"], self.hint["sorani"])
                hints += (pair if self.direction == "en-ckb" else tuple(reversed(pair)),)
            indices.append(index)
            selected.append(replace(request, additional_context=self.extra, glossary_hints=hints))
        for index, result in zip(indices, self.translator.translate_batch(selected), strict=True):
            results[index] = result
        if any(result is None for result in results):
            raise ValueError("Selective translation returned an incomplete batch")
        return [result for result in results if result is not None]


async def _prepare_translation(
    job_id: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
    reporter: Reporter,
) -> PreparedWork:
    async with sessions() as db:
        job = await db.get(Job, job_id)
        if not job or not job.project_id:
            raise ValueError("Translation job has no project")
        job, model = await _job_and_model(db, job_id, job.payload["model_id"])
        project = await db.get(Project, job.project_id)
        media = await db.get(MediaFile, uuid.UUID(job.payload["subtitle_id"]))
        if (
            not project
            or not media
            or media.project_id != project.id
            or media.sha256 != job.payload["subtitle_sha256"]
        ):
            raise ValueError("The source subtitle changed after the job was queued")
        terms = await terms_for(db, project.id)
        options = PipelineOptions(
            target_cps=project.target_cps,
            hard_cps=project.hard_cps,
            max_chars=project.max_chars,
            max_lines=project.max_lines,
            register=project.register,
            soften_profanity=project.soften_profanity,
            profanity_replacements=tuple(
                (rule["source"], rule["replacement"]) for rule in project.profanity_replacements
            ),
        )
        media_key, name, expected = media.storage_key, media.original_name, media.sha256
        payload, kind = dict(job.payload), job.kind
    await reporter.update("loading_model", 0, "Loading the selected trusted model")
    storage = create_storage(settings)
    temp_root = settings.storage_root.expanduser().resolve()
    temp_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".inference-", dir=temp_root) as temp:
        source = Path(temp) / ("source" + Path(name).suffix.casefold())
        await asyncio.to_thread(storage.download_file, media_key, source)
        digest = await asyncio.to_thread(lambda: hashlib.sha256(source.read_bytes()).hexdigest())
        if digest != expected:
            raise ValueError("Stored subtitle hash does not match its upload record")
        translator = await asyncio.to_thread(_translator, model, settings)
        if kind == "retranslate":
            original = await asyncio.to_thread(load_subtitles, source)
            normalizer = normalize_english if project.direction == "en-ckb" else normalize_sorani
            normalized = [replace(cue, text=normalizer(cue.text)) for cue in original]
            sentence = next(
                (s for s in reconstruct_sentences(normalized) if payload["cue_index"] in s.cue_ids),
                None,
            )
            if sentence is None:
                raise ValueError("The selected cue contains no spoken sentence")
            translator = _SelectiveTranslator(
                translator,
                sentence.text,
                tuple(payload.get("custom_context", [])),
                payload.get("glossary_hint"),
            )
        report = await asyncio.to_thread(
            translate_subtitles,
            source,
            translator,
            glossary=terms,
            options=options,
            progress=reporter.callback,
        )
        report["model"] = _public_model(report["model"])
        return PreparedWork(kind=kind, value=report)


async def _prepare_sample(job_id, sessions, settings: Settings, reporter: Reporter) -> PreparedWork:
    async with sessions() as db:
        job = await db.get(Job, job_id)
        if not job:
            raise ValueError("Sample job no longer exists")
        payload = dict(job.payload)
        _, model = await _job_and_model(db, job_id, payload["model_version_id"])
        if model.direction != payload["direction"]:
            raise ValueError("Sample model direction changed")
    await reporter.update("loading_model", 5, "Loading the selected model")
    translator = await asyncio.to_thread(_translator, model, settings)
    requests = [
        TranslationRequest(
            text=line, context=tuple(payload["lines"][:i][-translator.context_window :])
        )
        for i, line in enumerate(payload["lines"])
    ]
    generated = await asyncio.to_thread(translator.translate_batch, requests)
    await reporter.update("sample", 95, "Translated the sample")
    return PreparedWork(
        kind="sample",
        value={
            "direction": payload["direction"],
            "lines": payload["lines"],
            "outputs": [item.text for item in generated],
            "model": _public_model(translator.model_info),
        },
    )


async def _prepare_compare(
    job_id, sessions, settings: Settings, reporter: Reporter
) -> PreparedWork:
    async with sessions() as db:
        job = await db.get(Job, job_id)
        if not job:
            raise ValueError("Comparison job no longer exists")
        payload = dict(job.payload)
        _, first = await _job_and_model(db, job_id, payload["model_a"])
        _, second = await _job_and_model(db, job_id, payload["model_b"])
        if first.direction != payload["direction"] or second.direction != payload["direction"]:
            raise ValueError("Comparison model direction changed")
    outputs = []
    for index, model in enumerate((first, second)):
        await reporter.update("loading_model", 5 + index * 45, f"Loading model {index + 1} of 2")
        translator = await asyncio.to_thread(_translator, model, settings)
        requests = [
            TranslationRequest(
                text=line,
                context=tuple(
                    (payload["context"] + payload["lines"][:i])[-translator.context_window :]
                ),
            )
            for i, line in enumerate(payload["lines"])
        ]
        generated = await asyncio.to_thread(translator.translate_batch, requests)
        outputs.append(
            {
                "model_version_id": str(model.id),
                "model": _public_model(translator.model_info),
                "outputs": [
                    {
                        "text": item.text,
                        "token_logprobs": list(item.token_logprobs),
                        "metadata": item.metadata,
                    }
                    for item in generated
                ],
            }
        )
        del translator
        gc.collect()
        if settings.inference_device == "cuda":
            import torch

            torch.cuda.empty_cache()
    await reporter.update("comparison", 95, "Compared both models")
    return PreparedWork(kind="compare", value={"lines": payload["lines"], "models": outputs})


def _feedback_dataset(
    rows: list[tuple[Cue, Project]], source_cues: list[Cue], path: Path
) -> dict[str, int]:
    from datasets import Dataset, DatasetDict, disable_progress_bars

    from api.routes.models import _feedback_row
    from ml.train.dataset import make_context_examples

    direction = rows[0][1].direction
    source_records = []
    for cue in source_cues:
        source = (
            normalize_english(cue.source_text)
            if direction == "en-ckb"
            else normalize_sorani(cue.source_text)
        )
        if not source.strip():
            continue
        source_records.append(
            {
                "pair_id": str(cue.id),
                "film_id": str(cue.project_id),
                "english": source if direction == "en-ckb" else ".",
                "sorani": source if direction == "ckb-en" else ".",
                "english_start_ms": cue.original_start_ms,
                "sorani_start_ms": cue.original_start_ms,
                "english_speakers": cue.speaker_labels if direction == "en-ckb" else [],
                "sorani_speakers": cue.speaker_labels if direction == "ckb-en" else [],
                "provenance": "human",
            }
        )
    contexts = {
        example.pair_id: list(example.context)
        for example in make_context_examples(
            source_records, direction=direction, context_sentences=5
        )
    }

    films = sorted(
        {str(project.id) for _, project in rows},
        key=lambda value: hashlib.sha256(value.encode()).digest(),
    )
    if len(films) < 2:
        raise ValueError("Film-disjoint feedback training requires at least two films")
    val_film = films[0]
    test_film = films[1] if len(films) >= 3 else None
    groups: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for cue, project in rows:
        item = _feedback_row(cue, project)
        item["english"] = normalize_english(item["english"])
        item["sorani"] = normalize_sorani(item["sorani"])
        required = (
            "pair_id",
            "film_id",
            "english",
            "sorani",
            "english_start_ms",
            "sorani_start_ms",
            "english_speakers",
            "sorani_speakers",
            "provenance",
            "language_verified",
        )
        clean = {key: item[key] for key in required}
        if str(cue.id) not in contexts:
            raise ValueError("Approved feedback has no nonempty source context")
        # Context comes from every source cue in the film, never unreviewed targets.
        clean["source_context"] = contexts[str(cue.id)]
        group = (
            "val"
            if str(project.id) == val_film
            else "test"
            if str(project.id) == test_film
            else "train"
        )
        groups[group].append(clean)
    columns = list(groups["train"][0])
    disable_progress_bars()
    dataset = DatasetDict(
        {
            name: Dataset.from_list(items)
            if items
            else Dataset.from_dict({key: [] for key in columns})
            for name, items in groups.items()
        }
    )
    dataset.save_to_disk(
        str(path), num_shards={name: 1 for name, items in groups.items() if not items}
    )
    return {name: len(items) for name, items in groups.items()}


async def _prepare_train(job_id, sessions, settings: Settings, reporter: Reporter) -> PreparedWork:
    from ml.train.config import LoraSettings, load_config
    from ml.train.train_lora import run_training

    async with sessions() as db:
        job = await db.get(Job, job_id)
        if not job:
            raise ValueError("Training job no longer exists")
        job, model = await _job_and_model(db, job_id, job.payload["model_id"])
        payload = dict(job.payload)
        versions = {uuid.UUID(cue_id): revision for cue_id, revision in payload["cue_versions"]}
        rows = (
            await db.execute(
                select(Cue, Project)
                .join(Project, Project.id == Cue.project_id)
                .where(
                    Cue.id.in_(versions),
                    Project.owner_id == job.owner_id,
                    Project.direction == payload["direction"],
                )
                .order_by(Project.id, Cue.cue_index)
            )
        ).all()
        if len(rows) != len(versions) or any(
            cue.revision != versions[cue.id]
            or cue.status != "approved"
            or cue.human_text is None
            or cue.human_text == cue.machine_text
            or not cue.human_text.strip()
            for cue, _ in rows
        ):
            raise ValueError("Feedback changed after the training job was queued")
        source_cues = (
            await db.scalars(
                select(Cue)
                .where(Cue.project_id.in_({project.id for _, project in rows}))
                .order_by(Cue.project_id, Cue.cue_index)
            )
        ).all()
    await reporter.update("prepare_feedback", 1, "Preparing film-disjoint reviewed feedback")
    root = settings.storage_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_root = (
        settings.model_root.expanduser().resolve()
        / "feedback"
        / str(job_id)
        / str(reporter.lease_token)
    ).resolve()
    if not output_root.is_relative_to(settings.model_root.expanduser().resolve()):
        raise ValueError("Training output escapes the model root")
    with tempfile.TemporaryDirectory(prefix=".feedback-", dir=root) as temp:
        dataset_path = Path(temp) / "dataset"
        counts = await asyncio.to_thread(_feedback_dataset, rows, source_cues, dataset_path)
        config = load_config(Path(__file__).resolve().parents[1] / "ml/train/config.yaml")
        initial_adapter = model.adapter_dir if model.kind == "adapter" else None
        if initial_adapter:
            adapter_manifest = json.loads(
                (Path(initial_adapter) / "training_manifest.json").read_text(encoding="utf-8")
            )
            original = adapter_manifest["config"]
            raw_lora = original["lora"]
            lora = LoraSettings(
                r=raw_lora["r"],
                alpha=raw_lora["alpha"],
                dropout=raw_lora["dropout"],
                target_modules=tuple(raw_lora["target_modules"]),
            )
            config = replace(
                config,
                lora=lora,
                context_sentences=original["context_sentences"],
                max_source_length=original["max_source_length"],
                max_target_length=original["max_target_length"],
            )
        steps = max(
            1,
            math.ceil(
                counts["train"]
                / (
                    config.training.per_device_train_batch_size
                    * config.training.gradient_accumulation_steps
                )
            )
            * math.ceil(payload["epochs"]),
        )
        training = replace(
            config.training,
            epochs=payload["epochs"],
            learning_rate=payload["learning_rate"],
            warmup_steps=min(50, steps // 10),
            eval_save_steps=max(1, min(200, steps // 3)),
            logging_steps=1,
            report_to=(),
        )
        config = replace(
            config,
            base_model=model.source,
            base_revision=model.revision,
            direction=payload["direction"],
            dataset_dir=str(dataset_path),
            output_dir=str(output_root),
            local_files_only=settings.local_files_only,
            training=training,
        )
        await reporter.update(
            "training",
            5,
            f"Training on {counts['train']} reviewed cues; validating on {counts['val']}",
        )
        manifest = await asyncio.to_thread(
            run_training,
            config,
            allow_unverified_data=True,
            initial_adapter=initial_adapter,
            progress=reporter.training_callback,
        )
    await reporter.update("register_model", 98, "Preparing the trained model version")
    return PreparedWork(
        kind="train", value={"manifest": manifest, "counts": counts, "base_model_id": str(model.id)}
    )


async def prepare_work(job_id, sessions, settings: Settings, reporter: Reporter) -> PreparedWork:
    async with sessions() as db:
        job = await db.get(Job, job_id)
        if not job or job.status != "running":
            raise JobCancelled()
        kind = job.kind
    if kind in {"translate", "retranslate"}:
        return await _prepare_translation(job_id, sessions, settings, reporter)
    if kind == "compare":
        return await _prepare_compare(job_id, sessions, settings, reporter)
    if kind == "sample":
        return await _prepare_sample(job_id, sessions, settings, reporter)
    if kind == "train":
        return await _prepare_train(job_id, sessions, settings, reporter)
    raise ValueError("Unsupported job kind")


async def _apply_translation(db: AsyncSession, job: Job, report: dict) -> dict:
    project = await db.scalar(select(Project).where(Project.id == job.project_id).with_for_update())
    media = await db.get(MediaFile, uuid.UUID(job.payload["subtitle_id"]))
    owner = await db.get(User, job.owner_id)
    model = await db.get(ModelVersion, uuid.UUID(job.payload["model_id"]))
    if (
        not project
        or not media
        or not owner
        or not model
        or media.sha256 != job.payload["subtitle_sha256"]
    ):
        raise ValueError("The project or model changed before results could be saved")
    cues = (
        await db.scalars(
            select(Cue)
            .where(Cue.project_id == project.id)
            .order_by(Cue.cue_index)
            .with_for_update()
        )
    ).all()
    results = {item["cue_id"]: item for item in report["cues"]}
    if len(cues) != len(results) or any(cue.cue_index not in results for cue in cues):
        raise ValueError("Translation cue lineage no longer matches the project")
    terms = await terms_for(db, project.id)
    updated = 0
    for cue in cues:
        if job.kind == "retranslate" and str(cue.id) != job.payload["cue_id"]:
            continue
        if job.kind == "retranslate" and cue.revision != job.payload["expected_revision"]:
            raise ValueError("The selected cue was edited during retranslation")
        result = results[cue.cue_index]
        if (cue.original_start_ms, cue.original_end_ms) != (result["start_ms"], result["end_ms"]):
            raise ValueError("Original cue timing changed during translation")
        before = cue_snapshot(cue)
        cue.machine_text = result["machine_text"]
        cue.model_version_id = model.id
        cue.source_offsets = result["source_offsets"]
        cue.sentence_source_cue_ids = result["sentence_source_cue_ids"]
        cue.normalized_source_text = result["normalized_source_text"]
        cue.mean_token_logprob = result["mean_token_logprob"]
        cue.qa_warnings = result["qa_warnings"]
        if cue.human_text is None:
            cue.status = (
                "edited"
                if (cue.start_ms, cue.end_ms) != (cue.original_start_ms, cue.original_end_ms)
                else "unreviewed"
            )
        await recalculate_cue(db, cue, project, terms)
        if job.kind == "retranslate" and len(result["sentence_source_cue_ids"]) > 1:
            cue.qa_warnings = list(
                dict.fromkeys([*cue.qa_warnings, "partial_sentence_retranslation_needs_review"])
            )
            cue.flagged = True
        log_change(db, cue, owner, before, "machine_translation")
        updated += 1
    if job.kind == "retranslate" and updated != 1:
        raise ValueError("The retranslation cue no longer exists")
    await update_project_status(db, project)
    summary = summarize_qa(
        [
            {
                "qa_warnings": cue.qa_warnings,
                "flagged": cue.flagged,
                "confidence_score": cue.confidence_score if cue.confidence_score is not None else 0,
            }
            for cue in cues
        ]
    )
    scored = [cue.confidence_score for cue in cues if cue.confidence_score is not None]
    summary["average_confidence"] = sum(scored) / len(scored) if scored else None
    sentences = report["sentences"]
    if job.kind == "retranslate":
        target_index = job.payload["cue_index"]
        sentences = [
            sentence
            for sentence in sentences
            if any(span["cue_index"] == target_index for span in sentence["cue_spans"])
        ]
    return {
        "cue_count": updated,
        "qa_summary": summary,
        "model": _public_model(report["model"]),
        "warnings": report["warnings"],
        "sentences": sentences,
        "input_sha256": report["input_sha256"],
    }


async def apply_work(db: AsyncSession, job: Job, work: PreparedWork, settings: Settings) -> dict:
    if work.kind in {"translate", "retranslate"}:
        return await _apply_translation(db, job, work.value)
    if work.kind in {"compare", "sample"}:
        return work.value
    if work.kind == "train":
        versions = {uuid.UUID(cue_id): revision for cue_id, revision in job.payload["cue_versions"]}
        project_ids = (
            await db.scalars(select(Cue.project_id).where(Cue.id.in_(versions)).distinct())
        ).all()
        projects = (
            await db.scalars(
                select(Project)
                .where(Project.id.in_(project_ids))
                .order_by(Project.id)
                .with_for_update()
            )
        ).all()
        selected = (
            await db.scalars(
                select(Cue).where(Cue.id.in_(versions)).order_by(Cue.id).with_for_update()
            )
        ).all()
        if (
            len(selected) != len(versions)
            or any(
                p.owner_id != job.owner_id or p.direction != job.payload["direction"]
                for p in projects
            )
            or any(
                cue.revision != versions[cue.id]
                or cue.status != "approved"
                or not cue.human_text
                or cue.human_text == cue.machine_text
                for cue in selected
            )
        ):
            raise ValueError("Feedback changed during training; model was not registered")
        manifest, counts = work.value["manifest"], work.value["counts"]
        model = ModelVersion(
            id=uuid.uuid4(),
            owner_id=job.owner_id,
            name=f"feedback-{str(job.id)[:8]}",
            direction=job.payload["direction"],
            kind="adapter",
            source=manifest["actual_base_model"],
            adapter_dir=manifest["adapter_dir"],
            revision=None,
            metrics={"validation_chrf": manifest.get("final_eval_chrf")},
            metadata_json={
                "training_identity_sha256": manifest["identity_sha256"],
                "dataset_sha256": manifest["dataset_sha256"],
                "base_model_id": work.value["base_model_id"],
                "counts": counts,
                "warnings": manifest["warnings"],
                "smoke_test": manifest["smoke_test"],
            },
        )
        db.add(model)
        return {
            "model_version_id": str(model.id),
            "validation_chrf": manifest.get("final_eval_chrf"),
            "feedback_counts": counts,
            "warnings": manifest["warnings"],
        }
    raise ValueError("Unsupported prepared work")
