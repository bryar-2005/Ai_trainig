"""Revision-safe cue editing, approvals, compression, and QA."""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.access import cue_for, project_for
from api.db.models import Cue, CueRevision, Project, User
from api.db.session import get_session
from api.review import (
    cue_snapshot,
    log_change,
    recalculate_cue,
    recalculate_timing_neighbors,
    terms_for,
    update_project_status,
)
from api.routes.projects import _no_active_job
from api.schemas import BulkApproval, CuePatch, RetranslateIn, RevisionRequest
from api.security import current_user
from api.serializers import cue_json, revision_json
from ml.inference.compressor import compress
from ml.inference.glossary import relevant_locked
from ml.inference.text import isolate_ltr, normalize_target, wrap_lines

router = APIRouter(prefix="/api", tags=["subtitle editor"])
CueFilter = Literal[
    "all", "flagged", "low_confidence", "cps_violation", "glossary_mismatch", "unreviewed", "edited"
]


async def _locked_cue(db: AsyncSession, cue_id: uuid.UUID, user: User):
    cue, project = await cue_for(db, cue_id, user)
    project = await project_for(db, project.id, user, lock=True)
    cue, _ = await cue_for(db, cue.id, user, lock=True)
    return cue, project


def _check_revision(cue: Cue, expected: int) -> None:
    if cue.revision != expected:
        raise HTTPException(
            status_code=409,
            detail={"message": "Cue changed since it was loaded", "current_revision": cue.revision},
        )


def _approvable(cue: Cue) -> None:
    effective = cue.human_text if cue.human_text is not None else cue.machine_text
    if cue.source_text.strip() and not effective.strip():
        raise HTTPException(status_code=422, detail="A spoken cue must have text before approval")


@router.get("/projects/{project_id}/cues")
async def list_cues(
    project_id: uuid.UUID,
    filter: CueFilter = "all",
    q: str = Query(default="", max_length=200),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user)
    conditions = [Cue.project_id == project.id]
    if filter == "flagged":
        conditions.append(Cue.flagged.is_(True))
    elif filter == "low_confidence":
        conditions.append(Cue.confidence_score < 70)
    elif filter == "cps_violation":
        conditions.append(Cue.cps > project.target_cps)
    elif filter == "glossary_mismatch":
        conditions.append(func.json_array_length(Cue.glossary_mismatches) > 0)
    elif filter == "unreviewed":
        conditions.append(Cue.status == "unreviewed")
    elif filter == "edited":
        conditions.append(Cue.human_text.is_not(None))
    if q.strip():
        escaped = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        term = f"%{escaped}%"
        conditions.append(
            or_(
                Cue.source_text.ilike(term, escape="\\"),
                func.coalesce(Cue.human_text, Cue.machine_text).ilike(term, escape="\\"),
            )
        )
    total = await db.scalar(select(func.count(Cue.id)).where(*conditions))
    items = (
        await db.scalars(
            select(Cue)
            .where(*conditions)
            .order_by(Cue.cue_index)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).all()
    return {
        "items": [cue_json(cue) for cue in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/projects/{project_id}/cues/locate/{cue_id}")
async def locate_cue(
    project_id: uuid.UUID,
    cue_id: uuid.UUID,
    page_size: int = Query(default=100, ge=1, le=500),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    """Locate an authorized cue in the unfiltered ordered editor, including sparse indices."""
    project = await project_for(db, project_id, user)
    cue = await db.scalar(select(Cue).where(Cue.id == cue_id, Cue.project_id == project.id))
    if not cue:
        raise HTTPException(status_code=404, detail="Cue not found")
    preceding = await db.scalar(
        select(func.count(Cue.id)).where(
            Cue.project_id == project.id, Cue.cue_index < cue.cue_index
        )
    )
    return {"page": (preceding or 0) // page_size + 1, "cue_id": str(cue.id)}


@router.get("/cues/{cue_id}/context")
async def cue_context(
    cue_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    cue, project = await cue_for(db, cue_id, user)
    previous = (
        await db.scalars(
            select(Cue)
            .where(Cue.project_id == project.id, Cue.cue_index < cue.cue_index)
            .order_by(Cue.cue_index.desc())
            .limit(3)
        )
    ).all()
    following = (
        await db.scalars(
            select(Cue)
            .where(Cue.project_id == project.id, Cue.cue_index > cue.cue_index)
            .order_by(Cue.cue_index)
            .limit(3)
        )
    ).all()
    return {
        "previous": [cue_json(item) for item in reversed(previous)],
        "next": [cue_json(item) for item in following],
    }


@router.patch("/cues/{cue_id}")
async def patch_cue(
    cue_id: uuid.UUID,
    payload: CuePatch,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    cue, project = await _locked_cue(db, cue_id, user)
    await _no_active_job(db, project.id)
    _check_revision(cue, payload.expected_revision)
    changes = payload.model_dump(exclude_unset=True, exclude={"expected_revision"})
    if not changes:
        raise HTTPException(status_code=422, detail="No cue changes supplied")
    if any(changes.get(key, 1) is None for key in ("start_ms", "end_ms", "status")):
        raise HTTPException(status_code=422, detail="Timestamps and status cannot be null")
    start, end = changes.get("start_ms", cue.start_ms), changes.get("end_ms", cue.end_ms)
    if end <= start:
        raise HTTPException(status_code=422, detail="Cue end must follow its start")
    before = cue_snapshot(cue)
    requested_status = changes.pop("status", None)
    for key, value in changes.items():
        setattr(cue, key, value)
    if requested_status is None and cue_snapshot(cue) == before:
        return cue_json(cue)
    if requested_status == "approved":
        _approvable(cue)
    cue.status = requested_status or (
        "edited"
        if cue.human_text is not None
        or (start, end) != (cue.original_start_ms, cue.original_end_ms)
        else "unreviewed"
    )
    if cue_snapshot(cue) == before:
        return cue_json(cue)
    await recalculate_cue(db, cue, project)
    await recalculate_timing_neighbors(db, cue, project, before["start_ms"], before["end_ms"])
    log_change(db, cue, user, before, "edit")
    await update_project_status(db, project)
    await db.commit()
    return cue_json(cue)


@router.get("/cues/{cue_id}/revisions")
async def cue_revisions(
    cue_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_session)
):
    cue, _ = await cue_for(db, cue_id, user)
    rows = (
        await db.scalars(
            select(CueRevision).where(CueRevision.cue_id == cue.id).order_by(CueRevision.revision)
        )
    ).all()
    return {"items": [revision_json(row) for row in rows]}


@router.post("/cues/{cue_id}/revisions/{revision_id}/restore")
async def restore_revision(
    cue_id: uuid.UUID,
    revision_id: uuid.UUID,
    payload: RevisionRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    cue, project = await _locked_cue(db, cue_id, user)
    await _no_active_job(db, project.id)
    _check_revision(cue, payload.expected_revision)
    revision = await db.scalar(
        select(CueRevision).where(CueRevision.id == revision_id, CueRevision.cue_id == cue.id)
    )
    if not revision:
        raise HTTPException(status_code=404, detail="Revision not found")
    before = cue_snapshot(cue)
    # Restore only editable fields, never an old machine model or approval.
    snapshot = revision.after
    cue.human_text = snapshot["human_text"]
    cue.start_ms, cue.end_ms = snapshot["start_ms"], snapshot["end_ms"]
    cue.status = (
        "edited"
        if cue.human_text is not None
        or (cue.start_ms, cue.end_ms) != (cue.original_start_ms, cue.original_end_ms)
        else "unreviewed"
    )
    if before != cue_snapshot(cue):
        await recalculate_cue(db, cue, project)
        await recalculate_timing_neighbors(db, cue, project, before["start_ms"], before["end_ms"])
        log_change(db, cue, user, before, "restore_revision")
        await update_project_status(db, project)
        await db.commit()
    return cue_json(cue)


@router.post("/cues/{cue_id}/approve")
async def approve_cue(
    cue_id: uuid.UUID,
    payload: RevisionRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    cue, project = await _locked_cue(db, cue_id, user)
    await _no_active_job(db, project.id)
    _check_revision(cue, payload.expected_revision)
    _approvable(cue)
    if cue.status != "approved":
        before = cue_snapshot(cue)
        cue.status = "approved"
        log_change(db, cue, user, before, "approve")
        await update_project_status(db, project)
        await db.commit()
    return cue_json(cue)


@router.post("/projects/{project_id}/cues/bulk-approve")
async def bulk_approve(
    project_id: uuid.UUID,
    payload: BulkApproval,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user, lock=True)
    await _no_active_job(db, project.id)
    requested = dict(payload.cues)
    rows = (
        await db.scalars(
            select(Cue)
            .where(Cue.project_id == project.id, Cue.id.in_(requested))
            .order_by(Cue.cue_index)
            .with_for_update()
        )
    ).all()
    if len(rows) != len(requested):
        raise HTTPException(
            status_code=404, detail="One or more cues were not found in this project"
        )
    for cue in rows:
        _check_revision(cue, requested[cue.id])
        _approvable(cue)
    changed = 0
    for cue in rows:
        if cue.status != "approved":
            before = cue_snapshot(cue)
            cue.status = "approved"
            log_change(db, cue, user, before, "approve")
            changed += 1
    await update_project_status(db, project)
    await db.commit()
    return {"approved": changed, "items": [cue_json(cue) for cue in rows]}


async def _compress_one(db: AsyncSession, cue: Cue, project: Project, user: User, terms) -> bool:
    text = cue.human_text if cue.human_text is not None else cue.machine_text
    applicable = relevant_locked(cue.normalized_source_text, terms, project.direction)
    protected = tuple(x.sorani if project.direction == "en-ckb" else x.english for x in applicable)
    result = compress(
        text,
        cue.end_ms - cue.start_ms,
        direction=project.direction,
        target_cps=project.target_cps,
        protected_terms=protected,
    )
    if not result.applied_rules:
        return False
    before = cue_snapshot(cue)
    text = normalize_target(
        wrap_lines(result.text, max_chars=project.max_chars, max_lines=project.max_lines),
        project.direction,
    )
    cue.human_text = isolate_ltr(text) if project.direction == "en-ckb" else text
    cue.status = "edited"
    await recalculate_cue(db, cue, project, terms)
    cue.qa_warnings = list(dict.fromkeys([*cue.qa_warnings, "compression_needs_review"]))
    cue.flagged = True
    log_change(db, cue, user, before, "compress")
    return True


@router.post("/cues/{cue_id}/compress")
async def compress_cue(
    cue_id: uuid.UUID,
    payload: RevisionRequest,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    cue, project = await _locked_cue(db, cue_id, user)
    await _no_active_job(db, project.id)
    _check_revision(cue, payload.expected_revision)
    changed = await _compress_one(db, cue, project, user, await terms_for(db, project.id))
    if changed:
        await update_project_status(db, project)
        await db.commit()
    return {"changed": changed, "cue": cue_json(cue)}


@router.post("/cues/{cue_id}/retranslate", status_code=202)
async def retranslate_cue(
    cue_id: uuid.UUID,
    payload: RetranslateIn,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    from api.jobs import enqueue_retranslation

    cue, project = await _locked_cue(db, cue_id, user)
    _check_revision(cue, payload.expected_revision)
    job = await enqueue_retranslation(db, request, user, project, cue, payload)
    return {"job_id": str(job.id), "status": job.status}


@router.get("/projects/{project_id}/qa")
async def project_qa(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user)
    cues = (
        await db.scalars(select(Cue).where(Cue.project_id == project.id).order_by(Cue.cue_index))
    ).all()
    counts: Counter[str] = Counter()
    for cue in cues:
        counts.update(cue.qa_warnings)
    return {
        "cue_count": len(cues),
        "flagged_count": sum(c.flagged for c in cues),
        "warning_counts": dict(sorted(counts.items())),
        "items": [cue_json(c) for c in cues if c.flagged],
        "confidence_note": "Uncalibrated heuristic, not a correctness probability.",
    }


@router.post("/projects/{project_id}/qa/auto-fix")
async def auto_fix(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user, lock=True)
    await _no_active_job(db, project.id)
    terms = await terms_for(db, project.id)
    cues = (
        await db.scalars(
            select(Cue)
            .where(Cue.project_id == project.id, Cue.flagged.is_(True))
            .order_by(Cue.cue_index)
            .with_for_update()
        )
    ).all()
    changed = 0
    for cue in cues:
        changed += int(await _compress_one(db, cue, project, user, terms))
    await update_project_status(db, project)
    await db.commit()
    return {"changed": changed, "review_required": changed}
