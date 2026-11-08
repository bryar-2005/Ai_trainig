"""Opt-in nightly feedback scheduling and private snapshot lifecycle."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from celery import Celery
from sqlalchemy import and_, exists, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.config import Settings
from api.db.models import Cue, FeedbackExport, Job, JobEvent, Project, User, now
from api.db.session import create_engine, create_session_factory
from api.security import _aware
from api.storage import StorageBackend, create_storage

logger = logging.getLogger("sorani.feedback")


def feedback_scratch_path(settings: Settings, export_id: uuid.UUID) -> Path:
    return settings.storage_root.expanduser().resolve() / ".feedback-export-work" / str(export_id)


async def discard_pending_export(
    export_id: uuid.UUID,
    lease_token: uuid.UUID,
    sessions: async_sessionmaker[AsyncSession],
    storage: StorageBackend,
    settings: Settings,
) -> bool:
    """Remove only this failed attempt; never delete an already committed export."""
    async with sessions() as db:
        item = await db.scalar(
            select(FeedbackExport)
            .where(
                FeedbackExport.id == export_id,
                FeedbackExport.status == "pending",
                FeedbackExport.lease_token == lease_token,
            )
            .with_for_update()
        )
        if not item:
            return False
        await asyncio.to_thread(storage.delete, item.storage_key)
        await asyncio.to_thread(
            shutil.rmtree, feedback_scratch_path(settings, item.id), ignore_errors=True
        )
        await db.delete(item)
        await db.commit()
        return True


async def cleanup_feedback_exports(settings: Settings) -> int:
    """Delete expired snapshots and abandoned uploads; failures remain retryable."""
    engine = create_engine(settings.database_url)
    sessions = create_session_factory(engine)
    storage = create_storage(settings)
    removed = 0
    try:
        async with sessions() as db:
            candidates = (
                await db.scalars(
                    select(FeedbackExport.id)
                    .join(Job, Job.id == FeedbackExport.job_id)
                    .where(
                        or_(
                            FeedbackExport.expires_at <= now(),
                            and_(
                                FeedbackExport.status == "pending",
                                or_(
                                    Job.status.not_in(("running", "cancel_requested")),
                                    Job.lease_token.is_(None),
                                    Job.lease_token != FeedbackExport.lease_token,
                                    Job.updated_at
                                    <= now() - timedelta(seconds=settings.job_lease_seconds),
                                ),
                            ),
                        )
                    )
                    .order_by(FeedbackExport.expires_at, FeedbackExport.created_at)
                    .limit(500)
                )
            ).all()
        for export_id in candidates:
            try:
                async with sessions() as db:
                    item = await db.scalar(
                        select(FeedbackExport)
                        .where(FeedbackExport.id == export_id)
                        .with_for_update(skip_locked=True)
                    )
                    if not item:
                        continue
                    expired = _aware(item.expires_at) <= now()
                    if not expired and item.status == "pending":
                        job = await db.get(Job, item.job_id)
                        live = (
                            job
                            and job.status in {"running", "cancel_requested"}
                            and job.lease_token == item.lease_token
                            and _aware(job.updated_at)
                            > now() - timedelta(seconds=settings.job_lease_seconds)
                        )
                        if live:
                            continue
                    elif not expired:
                        continue
                    await asyncio.to_thread(storage.delete, item.storage_key)
                    await asyncio.to_thread(
                        shutil.rmtree, feedback_scratch_path(settings, item.id), ignore_errors=True
                    )
                    await db.delete(item)
                    await db.commit()
                    removed += 1
            except Exception as exc:
                # Never log subtitle text or the private object key.
                logger.error("feedback_export_cleanup_failed error_type=%s", type(exc).__name__)
        # A killed process may leave a staging directory after its DB row was
        # committed ready or removed. Do not touch a live pending writer.
        scratch_root = settings.storage_root.expanduser().resolve() / ".feedback-export-work"
        if scratch_root.is_dir():
            for path in list(scratch_root.iterdir())[:500]:
                try:
                    if path.is_symlink() or not path.is_dir():
                        continue
                    export_id = uuid.UUID(path.name)
                    async with sessions() as db:
                        item = await db.get(FeedbackExport, export_id)
                    if (item and item.status == "ready") or (
                        not item and path.stat().st_mtime < time.time() - settings.job_lease_seconds
                    ):
                        await asyncio.to_thread(shutil.rmtree, path, ignore_errors=True)
                except (OSError, ValueError):
                    continue
        return removed
    finally:
        await engine.dispose()


async def schedule_nightly_feedback_exports(
    settings: Settings, *, when: datetime | None = None
) -> int:
    """Create at most one nightly job per opted-in owner and UTC date.

    Jobs are committed before publication. A broker outage leaves them in the
    durable outbox, which the existing periodic recovery task republishes.
    """
    if not settings.nightly_feedback_export_enabled:
        return 0
    stamp = _aware(when or now()).date().isoformat()
    engine = create_engine(settings.database_url)
    sessions = create_session_factory(engine)
    queued: list[uuid.UUID] = []
    publisher = Celery("sorani-feedback-scheduler", broker=settings.broker_url)
    try:
        async with sessions() as db:
            eligible = exists(
                select(Cue.id)
                .join(Project, Project.id == Cue.project_id)
                .where(
                    Project.owner_id == User.id,
                    Cue.status == "approved",
                    Cue.human_text.is_not(None),
                    Cue.human_text != Cue.machine_text,
                    func.trim(Cue.source_text) != "",
                    func.trim(Cue.human_text) != "",
                )
            )
            candidates = (
                await db.scalars(select(User.id).where(User.is_active.is_(True), eligible))
            ).all()
        for user_id in candidates:
            async with sessions() as db:
                user = await db.scalar(select(User).where(User.id == user_id).with_for_update())
                if (
                    not user
                    or not user.is_active
                    or user.preferences.get("nightly_feedback_export") is not True
                ):
                    continue
                schedule_key = f"feedback:{user_id}:{stamp}"
                already = await db.scalar(select(Job.id).where(Job.schedule_key == schedule_key))
                active = await db.scalar(
                    select(Job.id)
                    .where(
                        Job.owner_id == user_id,
                        Job.kind == "feedback_export",
                        Job.status.in_(("queued", "running", "cancel_requested")),
                    )
                    .limit(1)
                )
                if already or active:
                    continue
                job = Job(
                    id=uuid.uuid4(),
                    owner_id=user_id,
                    kind="feedback_export",
                    status="queued",
                    stage="queued",
                    progress=0,
                    schedule_key=schedule_key,
                    payload={"direction": None, "scheduled": True},
                )
                db.add(job)
                db.add(
                    JobEvent(
                        job_id=job.id,
                        stage="queued",
                        progress=0,
                        message="Nightly feedback export queued",
                    )
                )
                await db.commit()
                queued.append(job.id)
            if settings.task_mode == "celery":
                try:
                    await asyncio.to_thread(
                        publisher.send_task,
                        "api.execute_job",
                        args=[str(job.id)],
                        task_id=str(job.id),
                        retry=False,
                    )
                    async with sessions() as db:
                        await db.execute(
                            update(Job)
                            .where(Job.id == job.id)
                            .values(enqueued_at=now(), updated_at=Job.updated_at)
                        )
                        await db.commit()
                except Exception as exc:
                    logger.error(
                        "feedback_schedule_publish_failed error_type=%s", type(exc).__name__
                    )
            elif settings.task_mode == "eager":
                from api.worker import run_job

                await run_job(job.id, settings)
        return len(queued)
    finally:
        publisher.close()
        await engine.dispose()
