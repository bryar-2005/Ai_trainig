"""Owner-scoped, verified-review feedback selection and snapshot formatting."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import Cue, Project, User


async def approved_feedback(db: AsyncSession, user: User, direction: str | None = None):
    # A workspace's feedback export never silently includes another user's text,
    # even when an administrator can inspect their projects individually.
    query = (
        select(Cue, Project)
        .join(Project, Project.id == Cue.project_id)
        .where(
            Project.owner_id == user.id,
            Cue.status == "approved",
            Cue.human_text.is_not(None),
            Cue.human_text != Cue.machine_text,
            func.trim(Cue.source_text) != "",
            func.trim(Cue.human_text) != "",
        )
        .order_by(Project.id, Cue.cue_index)
    )
    if direction:
        query = query.where(Project.direction == direction)
    return (await db.execute(query)).all()


def feedback_row(cue: Cue, project: Project) -> dict:
    source, target = cue.source_text, cue.human_text or ""
    english, sorani = (source, target) if project.direction == "en-ckb" else (target, source)
    return {
        "pair_id": str(cue.id),
        "film_id": str(project.id),
        "english": english,
        "sorani": sorani,
        "english_start_ms": cue.start_ms,
        "english_end_ms": cue.end_ms,
        "sorani_start_ms": cue.start_ms,
        "sorani_end_ms": cue.end_ms,
        "english_speakers": cue.speaker_labels if project.direction == "en-ckb" else [],
        "sorani_speakers": cue.speaker_labels if project.direction == "ckb-en" else [],
        "provenance": "human",
        "review_status": "verified",
        "language_verified": False,
        "machine_text": cue.machine_text,
        "human_text": cue.human_text,
        "model_version_id": str(cue.model_version_id) if cue.model_version_id else None,
        "source_cue_ids": cue.source_cue_ids,
        "cue_revision": cue.revision,
    }
