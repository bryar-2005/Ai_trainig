"""Recalculate human-visible QA and record auditable cue changes."""

from __future__ import annotations

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import Cue, CueRevision, GlossaryTerm, Project, ReviewEvent, User
from ml.inference.glossary import GlossaryEntry, missing_locked, relevant_locked
from ml.inference.qa import score_cue


def cue_snapshot(cue: Cue) -> dict:
    return {
        "human_text": cue.human_text,
        "machine_text": cue.machine_text,
        "model_version_id": str(cue.model_version_id) if cue.model_version_id else None,
        "start_ms": cue.start_ms,
        "end_ms": cue.end_ms,
        "status": cue.status,
    }


async def terms_for(db: AsyncSession, project_id) -> list[GlossaryEntry]:
    rows = (
        await db.scalars(select(GlossaryTerm).where(GlossaryTerm.project_id == project_id))
    ).all()
    return [GlossaryEntry(x.english, x.sorani, x.locked, x.category, x.origin) for x in rows]


async def recalculate_cue(
    db: AsyncSession, cue: Cue, project: Project, terms: list[GlossaryEntry] | None = None
) -> None:
    terms = terms if terms is not None else await terms_for(db, project.id)
    text = cue.human_text if cue.human_text is not None else cue.machine_text
    applicable = relevant_locked(cue.normalized_source_text, terms, project.direction)
    missing = missing_locked(text, applicable, project.direction)
    targets = tuple(x.sorani if project.direction == "en-ckb" else x.english for x in missing)
    # Keep model/input warnings which cannot be reconstructed from the displayed cue.
    retained = {
        "source_truncated",
        "context_truncated",
        "invalid_model_scores",
        "original_timing_out_of_order",
        "literal_replacement_needs_review",
        "register_needs_review",
        "source_ad_removed",
        "profanity_softened_needs_review",
        "partial_sentence_retranslation_needs_review",
    }
    extras = [x for x in cue.qa_warnings if x in retained]
    if cue.human_text is None and "missing_translation" in cue.qa_warnings:
        extras.append("missing_translation")
    if cue.sound_effects:
        extras.append("sfx_untranslated")
    if cue.human_text is not None and cue.mean_token_logprob is not None:
        extras.append("human_edit_model_confidence_is_stale")
    overlap = await db.scalar(
        select(Cue.id)
        .where(
            Cue.project_id == cue.project_id,
            Cue.id != cue.id,
            Cue.start_ms < cue.end_ms,
            Cue.end_ms > cue.start_ms,
        )
        .limit(1)
    )
    quality = score_cue(
        source=cue.normalized_source_text,
        target=text,
        duration_ms=cue.end_ms - cue.start_ms,
        direction=project.direction,
        token_logprobs=(cue.mean_token_logprob,) if cue.mean_token_logprob is not None else (),
        missing_terms=targets,
        target_cps=project.target_cps,
        hard_cps=project.hard_cps,
        max_chars=project.max_chars,
        max_lines=project.max_lines,
        timing_overlap=bool(overlap),
        extra_warnings=tuple(extras),
    )
    cue.cps = quality.cps
    cue.line_lengths = list(quality.line_lengths)
    cue.length_ratio = quality.length_ratio
    cue.confidence_score = quality.confidence_score
    cue.glossary_mismatches = list(targets)
    cue.qa_warnings = [
        "timing_overlap" if warning == "original_timing_overlap" else warning
        for warning in quality.warnings
    ]
    cue.flagged = quality.flagged


async def recalculate_timing_neighbors(
    db: AsyncSession, cue: Cue, project: Project, previous_start: int, previous_end: int
) -> None:
    """Refresh both old and new overlap neighbors after a deliberate time edit."""
    if (previous_start, previous_end) == (cue.start_ms, cue.end_ms):
        return
    await db.flush()
    terms = await terms_for(db, project.id)
    neighbors = (
        await db.scalars(
            select(Cue)
            .where(
                Cue.project_id == project.id,
                Cue.id != cue.id,
                or_(
                    and_(Cue.start_ms < previous_end, Cue.end_ms > previous_start),
                    and_(Cue.start_ms < cue.end_ms, Cue.end_ms > cue.start_ms),
                ),
            )
            .order_by(Cue.cue_index)
            .with_for_update()
        )
    ).all()
    for neighbor in neighbors:
        await recalculate_cue(db, neighbor, project, terms)


def log_change(db: AsyncSession, cue: Cue, user: User, before: dict, reason: str) -> None:
    cue.revision += 1
    db.add(
        CueRevision(
            cue_id=cue.id,
            actor_id=user.id,
            revision=cue.revision,
            reason=reason,
            before=before,
            after=cue_snapshot(cue),
        )
    )
    db.add(ReviewEvent(cue_id=cue.id, actor_id=user.id, action=reason, cue_revision=cue.revision))


async def update_project_status(db: AsyncSession, project: Project) -> None:
    # Flush edits before counting because sessions intentionally disable autoflush.
    await db.flush()
    cues = (await db.scalars(select(Cue).where(Cue.project_id == project.id))).all()
    if not cues:
        project.status = "draft"
    elif all(cue.status == "approved" for cue in cues):
        project.status = "completed"
    elif any(cue.machine_text or cue.human_text is not None for cue in cues):
        project.status = "needs_review"
    else:
        project.status = "draft"
