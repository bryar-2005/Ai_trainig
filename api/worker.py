"""Celery execution with persisted progress, leases, and cooperative cancellation."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, timedelta
from typing import Any

from celery import Celery
from celery.schedules import crontab
from dotenv import dotenv_values
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.config import Settings, get_settings
from api.db.models import Job, JobEvent, Project, now
from api.db.session import create_engine, create_session_factory
from api.review import update_project_status

logger = logging.getLogger("sorani.worker")
_broker_url = (
    os.environ.get("SORANI_BROKER_URL")
    or dotenv_values(".env").get("SORANI_BROKER_URL")
    or "redis://localhost:6379/1"
)
celery_app = Celery("sorani", broker=_broker_url)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={"visibility_timeout": 7200},
    task_ignore_result=True,
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "recover-undelivered-jobs": {"task": "api.requeue_stale_jobs", "schedule": 30.0},
        "nightly-private-feedback-exports": {
            "task": "api.schedule_feedback_exports",
            "schedule": crontab(hour=2, minute=0),
        },
        "clean-expired-feedback-exports": {
            "task": "api.cleanup_feedback_exports",
            "schedule": 3600.0,
        },
    },
)


class JobCancelled(Exception):
    """Raised at a safe boundary when the owner requested cancellation."""


@dataclass
class PreparedWork:
    kind: str
    value: Any
    cleanup_path: str | None = None
    cleanup_export_id: uuid.UUID | None = None


class Reporter:
    def __init__(
        self,
        job_id: uuid.UUID,
        lease_token: uuid.UUID,
        sessions: async_sessionmaker[AsyncSession],
        loop,
    ):
        self.job_id, self.lease_token, self.sessions, self.loop = (
            job_id,
            lease_token,
            sessions,
            loop,
        )

    async def update(self, stage: str, percent: float, detail: str = "") -> None:
        async with self.sessions() as db:
            job = await db.scalar(select(Job).where(Job.id == self.job_id).with_for_update())
            if not job or job.lease_token != self.lease_token or job.status == "cancel_requested":
                raise JobCancelled()
            if job.status != "running":
                raise JobCancelled()
            progress = min(99.0, max(job.progress, float(percent)))
            changed = stage != job.stage or progress >= job.progress + 1.0
            job.stage, job.progress, job.updated_at = stage[:80], progress, now()
            if changed:
                db.add(
                    JobEvent(
                        job_id=job.id, stage=job.stage, progress=progress, message=detail[:500]
                    )
                )
            await db.commit()

    def callback(self, event: dict) -> None:
        # ML code runs in a thread, so the loop remains available for DB writes.
        future = asyncio.run_coroutine_threadsafe(
            self.update(
                event.get("stage", "processing"), event.get("percent", 0), event.get("detail", "")
            ),
            self.loop,
        )
        future.result()

    def training_callback(self, event: dict) -> None:
        step, total = event.get("step", 0), max(1, event.get("total_steps", 1))
        self.callback(
            {
                "stage": event.get("stage", "training"),
                "percent": 5 + 90 * step / total,
                "detail": f"Optimizer step {step} of {total}",
            }
        )


async def _claim(job_id: uuid.UUID, lease_token: uuid.UUID, sessions, settings: Settings) -> str:
    async with sessions() as db:
        job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if not job or job.status in {"completed", "failed", "cancelled"}:
            return "finished"
        if job.status in {"running", "cancel_requested"}:
            updated = (
                job.updated_at.replace(tzinfo=UTC)
                if job.updated_at.tzinfo is None
                else job.updated_at
            )
            if updated > now() - timedelta(seconds=settings.job_lease_seconds):
                return "busy"
            if job.status == "cancel_requested":
                job.status, job.stage, job.finished_at = "cancelled", "cancelled", now()
                job.lease_token = None
                db.add(
                    JobEvent(
                        job_id=job.id,
                        stage="cancelled",
                        progress=job.progress,
                        message="Cancelled after worker lease expired",
                    )
                )
                if job.project_id and job.kind == "translate":
                    project = await db.get(Project, job.project_id)
                    if project:
                        await update_project_status(db, project)
                await db.commit()
                return "finished"
            message = "Recovered an expired worker lease; restarting the job"
            job.progress = 0
        else:
            message = "Worker started"
        job.status, job.stage, job.started_at, job.updated_at = "running", "starting", now(), now()
        job.lease_token = lease_token
        db.add(JobEvent(job_id=job.id, stage="starting", progress=job.progress, message=message))
        await db.commit()
        return "claimed"


async def _heartbeat(job_id: uuid.UUID, lease_token: uuid.UUID, sessions, interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        async with sessions() as db:
            changed = await db.execute(
                update(Job)
                .where(
                    Job.id == job_id,
                    Job.lease_token == lease_token,
                    Job.status.in_(("running", "cancel_requested")),
                )
                .values(updated_at=now())
            )
            await db.commit()
            if not changed.rowcount:
                return


async def _finish_failure(
    job_id: uuid.UUID, lease_token: uuid.UUID, sessions, *, cancelled: bool, error: str
) -> None:
    async with sessions() as db:
        job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if (
            not job
            or job.lease_token != lease_token
            or job.status in {"completed", "failed", "cancelled"}
        ):
            return
        cancelled = cancelled or job.status == "cancel_requested"
        job.status = job.stage = "cancelled" if cancelled else "failed"
        job.lease_token = None
        job.finished_at = now()
        job.error = None if cancelled else error
        db.add(
            JobEvent(
                job_id=job.id,
                stage=job.stage,
                progress=job.progress,
                message="Job cancelled" if cancelled else error,
            )
        )
        if job.project_id and job.kind == "translate":
            project = await db.get(Project, job.project_id)
            if project:
                if cancelled:
                    await update_project_status(db, project)
                else:
                    project.status = "failed"
        await db.commit()


async def run_job(job_id: uuid.UUID, settings: Settings) -> str:
    """Claim and execute a job; return busy when another live worker owns it."""
    from api.worker_tasks import apply_work, prepare_work

    engine = create_engine(settings.database_url)
    sessions = create_session_factory(engine)
    heartbeat = None
    prepared: PreparedWork | None = None
    claimed = False
    completed = False
    lease_token = uuid.uuid4()
    try:
        state = await _claim(job_id, lease_token, sessions, settings)
        if state != "claimed":
            return state
        claimed = True
        reporter = Reporter(job_id, lease_token, sessions, asyncio.get_running_loop())
        heartbeat = asyncio.create_task(
            _heartbeat(job_id, lease_token, sessions, settings.job_heartbeat_seconds)
        )
        try:
            prepared = await prepare_work(job_id, sessions, settings, reporter)
            async with sessions() as db:
                job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
                if not job or job.lease_token != lease_token or job.status == "cancel_requested":
                    raise JobCancelled()
                if job.status != "running":
                    raise JobCancelled()
                result = await apply_work(db, job, prepared, settings)
                job.result, job.status, job.stage, job.progress, job.finished_at = (
                    result,
                    "completed",
                    "completed",
                    100,
                    now(),
                )
                job.lease_token = None
                db.add(
                    JobEvent(
                        job_id=job.id, stage="completed", progress=100, message="Job completed"
                    )
                )
                await db.commit()
            completed = True
            return "completed"
        except JobCancelled:
            await _finish_failure(job_id, lease_token, sessions, cancelled=True, error="")
            return "cancelled"
        except Exception as exc:
            # Do not write subtitle text, credentials, or full local paths into logs.
            logger.error("job_failed job_id=%s error_type=%s", job_id, type(exc).__name__)
            await _finish_failure(
                job_id,
                lease_token,
                sessions,
                cancelled=False,
                error=f"{type(exc).__name__}: job failed; inspect worker logs",
            )
            return "failed"
    finally:
        if heartbeat:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
        if prepared and prepared.cleanup_path:
            await asyncio.to_thread(shutil.rmtree, prepared.cleanup_path, ignore_errors=True)
        if prepared and prepared.cleanup_export_id and not completed:
            from api.feedback_tasks import discard_pending_export
            from api.storage import create_storage

            try:
                await discard_pending_export(
                    prepared.cleanup_export_id,
                    lease_token,
                    sessions,
                    create_storage(settings),
                    settings,
                )
            except Exception:
                # The tracked pending object remains eligible for periodic cleanup.
                logger.error("feedback_cleanup_failed job_id=%s", job_id)
        if claimed and not completed:
            # Only this job's private partial training output is removed; a busy
            # duplicate delivery must never remove a live worker's artifacts.
            partial = (
                settings.model_root.expanduser().resolve()
                / "feedback"
                / str(job_id)
                / str(lease_token)
            )
            await asyncio.to_thread(shutil.rmtree, partial, ignore_errors=True)
        await engine.dispose()


async def requeue_stale_jobs(settings: Settings) -> int:
    """Recover a queued-but-unpublished message or an expired worker lease.

    The jobs table is the durable outbox; duplicate broker deliveries are safe
    because workers claim each job under a row lock. Run Celery beat alongside
    the worker to invoke this periodic recovery task.
    """
    engine = create_engine(settings.database_url)
    sessions = create_session_factory(engine)
    publisher = Celery("sorani-recovery", broker=settings.broker_url)
    try:
        async with sessions() as db:
            cutoff = now() - timedelta(seconds=30)
            stale_lease = now() - timedelta(seconds=settings.job_lease_seconds)
            rows = (
                await db.scalars(
                    select(Job)
                    .where(
                        or_(
                            Job.status == "queued",
                            (Job.status.in_(("running", "cancel_requested")))
                            & (Job.updated_at < stale_lease),
                        ),
                        or_(Job.enqueued_at.is_(None), Job.enqueued_at < cutoff),
                    )
                    .order_by(Job.created_at)
                    .limit(100)
                    .with_for_update(skip_locked=True)
                )
            ).all()
            for job in rows:
                await asyncio.to_thread(
                    publisher.send_task,
                    "api.execute_job",
                    args=[str(job.id)],
                    task_id=str(job.id),
                    retry=False,
                )
                # Do not renew a dead worker's lease when recording publication.
                await db.execute(
                    update(Job)
                    .where(Job.id == job.id)
                    .values(enqueued_at=now(), updated_at=Job.updated_at)
                )
            await db.commit()
            return len(rows)
    finally:
        publisher.close()
        await engine.dispose()


@celery_app.task(name="api.requeue_stale_jobs")
def recover_jobs():
    return asyncio.run(requeue_stale_jobs(get_settings()))


@celery_app.task(name="api.schedule_feedback_exports")
def schedule_feedback_exports():
    from api.feedback_tasks import schedule_nightly_feedback_exports

    return asyncio.run(schedule_nightly_feedback_exports(get_settings()))


@celery_app.task(name="api.cleanup_feedback_exports")
def cleanup_exports():
    from api.feedback_tasks import cleanup_feedback_exports

    return asyncio.run(cleanup_feedback_exports(get_settings()))


@celery_app.task(bind=True, name="api.execute_job", max_retries=None)
def execute_job(self, job_id: str):
    settings = get_settings()
    state = asyncio.run(run_job(uuid.UUID(job_id), settings))
    if state == "busy":
        raise self.retry(countdown=min(30, settings.job_lease_seconds // 3))
    return state
