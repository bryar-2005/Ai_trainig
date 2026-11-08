"""Transactional job creation and broker dispatch."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.access import model_for, project_for
from api.db.models import Cue, Job, JobEvent, MediaFile, Project, User, now
from api.routes.projects import _no_active_job
from api.schemas import RetranslateIn, TranslationIn


async def dispatch(db: AsyncSession, request: Request, job: Job) -> Job:
    settings = request.app.state.settings
    try:
        if settings.task_mode == "celery":
            from celery import Celery

            publisher = Celery("sorani-publisher", broker=settings.broker_url)
            try:
                # send_task itself is blocking and may perform broker I/O.
                await asyncio.to_thread(
                    publisher.send_task,
                    "api.execute_job",
                    args=[str(job.id)],
                    task_id=str(job.id),
                    retry=False,
                )
            finally:
                publisher.close()
            job.enqueued_at = now()
            await db.commit()
        elif settings.task_mode == "eager":
            from api.worker import run_job

            await run_job(job.id, settings)
            await db.refresh(job)
        elif settings.environment != "test":
            raise RuntimeError("Manual dispatch is only available in tests")
    except Exception as exc:
        job.status, job.stage, job.error, job.finished_at = (
            "failed",
            "dispatch_failed",
            "Could not enqueue the job",
            now(),
        )
        db.add(JobEvent(job_id=job.id, stage=job.stage, progress=job.progress, message=job.error))
        if job.kind == "translate" and job.project_id:
            project = await db.get(Project, job.project_id)
            if project:
                project.status = "failed"
        await db.commit()
        raise HTTPException(status_code=503, detail="The job broker is unavailable") from exc
    return job


async def _model_choice(db: AsyncSession, user: User, project: Project, selected: uuid.UUID | None):
    model_id = selected or project.model_version_id
    if not model_id:
        raw = user.preferences.get(f"active_model_{project.direction}")
        try:
            model_id = uuid.UUID(raw) if raw else None
        except (TypeError, ValueError):
            model_id = None
    if not model_id:
        raise HTTPException(status_code=422, detail="Select an available model version first")
    model = await model_for(db, model_id, user)
    if model.direction != project.direction:
        raise HTTPException(status_code=422, detail="Model direction does not match the project")
    return model


async def enqueue_translation(
    db: AsyncSession, request: Request, user: User, project_id: uuid.UUID, payload: TranslationIn
) -> Job:
    project = await project_for(db, project_id, user, lock=True)
    await _no_active_job(db, project.id)
    media = await db.scalar(
        select(MediaFile).where(MediaFile.project_id == project.id, MediaFile.kind == "subtitle")
    )
    if not media:
        raise HTTPException(status_code=422, detail="Upload subtitles before translation")
    existing = await db.scalar(
        select(Cue.id).where(Cue.project_id == project.id, Cue.machine_text != "").limit(1)
    )
    if existing and not payload.allow_replace_machine:
        raise HTTPException(
            status_code=409, detail="Machine text already exists; explicitly allow replacement"
        )
    model = await _model_choice(db, user, project, payload.model_version_id)
    job = Job(
        id=uuid.uuid4(),
        owner_id=user.id,
        project_id=project.id,
        kind="translate",
        status="queued",
        stage="queued",
        progress=0,
        payload={
            "model_id": str(model.id),
            "subtitle_id": str(media.id),
            "subtitle_sha256": media.sha256,
        },
    )
    db.add(job)
    db.add(JobEvent(job_id=job.id, stage="queued", progress=0, message="Translation queued"))
    project.status = "translating"
    await db.commit()
    return await dispatch(db, request, job)


async def enqueue_retranslation(
    db: AsyncSession,
    request: Request,
    user: User,
    project: Project,
    cue: Cue,
    payload: RetranslateIn,
    selected_model: uuid.UUID | None = None,
) -> Job:
    project = await project_for(db, project.id, user, lock=True)
    await _no_active_job(db, project.id)
    media = await db.scalar(
        select(MediaFile).where(MediaFile.project_id == project.id, MediaFile.kind == "subtitle")
    )
    if not media:
        raise HTTPException(status_code=422, detail="Upload subtitles before translation")
    model = await _model_choice(db, user, project, selected_model)
    job = Job(
        id=uuid.uuid4(),
        owner_id=user.id,
        project_id=project.id,
        kind="retranslate",
        status="queued",
        stage="queued",
        progress=0,
        payload={
            "model_id": str(model.id),
            "subtitle_id": str(media.id),
            "subtitle_sha256": media.sha256,
            "cue_id": str(cue.id),
            "cue_index": cue.cue_index,
            "expected_revision": cue.revision,
            "custom_context": payload.custom_context,
            "glossary_hint": payload.glossary_hint.model_dump() if payload.glossary_hint else None,
        },
    )
    db.add(job)
    db.add(
        JobEvent(
            job_id=job.id, stage="queued", progress=0, message="Contextual retranslation queued"
        )
    )
    await db.commit()
    return await dispatch(db, request, job)


async def enqueue_user_job(
    db: AsyncSession, request: Request, user: User, kind: str, payload: dict
) -> Job:
    # Serialize expensive per-user work without locking unrelated projects.
    await db.scalar(select(User).where(User.id == user.id).with_for_update())
    if kind in {"train", "sample"}:
        existing = await db.scalar(
            select(Job.id)
            .where(
                Job.owner_id == user.id,
                Job.kind == kind,
                Job.status.in_(("queued", "running", "cancel_requested")),
            )
            .limit(1)
        )
        if existing:
            raise HTTPException(status_code=409, detail=f"A {kind} job is already active")
    job = Job(
        id=uuid.uuid4(),
        owner_id=user.id,
        project_id=None,
        kind=kind,
        status="queued",
        stage="queued",
        progress=0,
        payload=payload,
    )
    db.add(job)
    db.add(
        JobEvent(job_id=job.id, stage="queued", progress=0, message=f"{kind.capitalize()} queued")
    )
    await db.commit()
    return await dispatch(db, request, job)
