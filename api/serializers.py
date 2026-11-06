"""Stable response shapes without password hashes, storage keys, or internal paths."""

from __future__ import annotations

from api.db.models import Cue, CueRevision, GlossaryTerm, Job, MediaFile, ModelVersion, Project


def project_json(item: Project) -> dict:
    return {
        "id": str(item.id),
        "name": item.name,
        "description": item.description,
        "direction": item.direction,
        "status": item.status,
        "model_version_id": str(item.model_version_id) if item.model_version_id else None,
        "target_cps": item.target_cps,
        "hard_cps": item.hard_cps,
        "max_chars": item.max_chars,
        "max_lines": item.max_lines,
        "register": item.register,
        "soften_profanity": item.soften_profanity,
        "profanity_replacements": item.profanity_replacements,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


def cue_json(item: Cue) -> dict:
    return {
        "id": str(item.id),
        "project_id": str(item.project_id),
        "cue_index": item.cue_index,
        "original_start_ms": item.original_start_ms,
        "original_end_ms": item.original_end_ms,
        "start_ms": item.start_ms,
        "end_ms": item.end_ms,
        "source_cue_ids": item.source_cue_ids,
        "sentence_source_cue_ids": item.sentence_source_cue_ids,
        "source_offsets": item.source_offsets,
        "source_text": item.source_text,
        "normalized_source_text": item.normalized_source_text,
        "speaker": item.speaker,
        "speaker_labels": item.speaker_labels,
        "sound_effects": item.sound_effects,
        "style": item.style,
        "machine_text": item.machine_text,
        "human_text": item.human_text,
        "effective_text": item.human_text if item.human_text is not None else item.machine_text,
        "model_version_id": str(item.model_version_id) if item.model_version_id else None,
        "confidence_score": item.confidence_score,
        "mean_token_logprob": item.mean_token_logprob,
        "cps": item.cps,
        "line_lengths": item.line_lengths,
        "length_ratio": item.length_ratio,
        "glossary_mismatches": item.glossary_mismatches,
        "qa_warnings": item.qa_warnings,
        "flagged": item.flagged,
        "status": item.status,
        "revision": item.revision,
        "updated_at": item.updated_at,
    }


def revision_json(item: CueRevision) -> dict:
    return {
        "id": str(item.id),
        "revision": item.revision,
        "reason": item.reason,
        "actor_id": str(item.actor_id),
        "before": item.before,
        "after": item.after,
        "created_at": item.created_at,
    }


def term_json(item: GlossaryTerm) -> dict:
    return {
        "id": str(item.id),
        "project_id": str(item.project_id),
        "english": item.english,
        "sorani": item.sorani,
        "category": item.category,
        "locked": item.locked,
        "origin": item.origin,
        "created_at": item.created_at,
    }


def media_json(item: MediaFile) -> dict:
    return {
        "id": str(item.id),
        "kind": item.kind,
        "original_name": item.original_name,
        "content_type": item.content_type,
        "size_bytes": item.size_bytes,
        "sha256": item.sha256,
    }


def job_json(item: Job) -> dict:
    # Job payload can contain dialogue and server-side details; results are
    # intentionally visible only after the owner check in the calling route.
    return {
        "id": str(item.id),
        "project_id": str(item.project_id) if item.project_id else None,
        "kind": item.kind,
        "status": item.status,
        "stage": item.stage,
        "progress": item.progress,
        "result": item.result,
        "error": item.error,
        "created_at": item.created_at,
        "started_at": item.started_at,
        "finished_at": item.finished_at,
        "updated_at": item.updated_at,
    }


def model_json(item: ModelVersion, *, active: bool = False) -> dict:
    return {
        "id": str(item.id),
        "name": item.name,
        "direction": item.direction,
        "kind": item.kind,
        "owner_id": str(item.owner_id) if item.owner_id else None,
        "metrics": item.metrics,
        "metadata": item.metadata_json,
        "active": active,
        "created_at": item.created_at,
    }
