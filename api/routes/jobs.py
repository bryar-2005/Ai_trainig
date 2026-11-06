"""Job lifecycle, cooperative cancellation, and resumable SSE progress."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.access import job_for
from api.db.models import Job, JobEvent, Project, User, now
from api.db.session import get_session
from api.jobs import enqueue_translation
from api.review import update_project_status
from api.schemas import TranslationIn
from api.security import current_user
from api.serializers import job_json

router = APIRouter(prefix="/api", tags=["jobs"])
_TERMINAL = {"completed", "failed", "cancelled"}


@router.post("/projects/{project_id}/translate", status_code=202)
async def translate_project(
    project_id: uuid.UUID,
    payload: TranslationIn,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    job = await enqueue_translation(db, request, user, project_id, payload)
    return {"job_id": str(job.id), "status": job.status}


@router.get("/jobs")
async def list_jobs(
    status: Literal["queued", "running", "cancel_requested", "cancelled", "completed", "failed"]
    | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    conditions = [] if user.role == "admin" else [Job.owner_id == user.id]
    if status:
        conditions.append(Job.status == status)
    total = await db.scalar(select(func.count(Job.id)).where(*conditions))
    rows = (
        await db.scalars(
            select(Job)
            .where(*conditions)
            .order_by(Job.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).all()
    return {
        "items": [job_json(item) for item in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_session)
):
    return job_json(await job_for(db, job_id, user))


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_session)
):
    job = await job_for(db, job_id, user, lock=True)
    if job.status in _TERMINAL:
        raise HTTPException(status_code=409, detail="Job is already finished")
    if job.status == "queued":
        job.status, job.stage, job.finished_at = "cancelled", "cancelled", now()
        if job.project_id and job.kind == "translate":
            project = await db.get(Project, job.project_id)
            if project:
                await update_project_status(db, project)
    else:
        job.status, job.stage = "cancel_requested", "cancel_requested"
    db.add(
        JobEvent(
            job_id=job.id, stage=job.stage, progress=job.progress, message="Cancellation requested"
        )
    )
    await db.commit()
    return job_json(job)


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(
    job_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_session)
):
    job = await job_for(db, job_id, user, lock=True)
    if job.status not in _TERMINAL:
        raise HTTPException(status_code=409, detail="Cancel or finish the job before deleting it")
    await db.delete(job)
    await db.commit()
    return Response(status_code=204)


def _eta(job: Job) -> int | None:
    if not job.started_at or not 0 < job.progress < 100:
        return None
    started = (
        job.started_at.replace(tzinfo=UTC) if job.started_at.tzinfo is None else job.started_at
    )
    elapsed = max(0.0, (now() - started).total_seconds())
    return round(elapsed * (100 - job.progress) / job.progress)


@router.get("/jobs/{job_id}/stream")
async def stream_job(
    job_id: uuid.UUID,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    await job_for(db, job_id, user)
    # The stream uses short-lived sessions; do not retain the request's DB
    # connection for the entire SSE lifetime.
    await db.rollback()
    try:
        last = int(request.headers.get("last-event-id", "0"))
        if last < 0:
            raise ValueError
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Last-Event-ID") from exc

    async def generate():
        nonlocal last
        heartbeat = 0
        while not await request.is_disconnected():
            async with request.app.state.sessions() as session:
                try:
                    # Re-check expiry and revocation while a long stream is open.
                    fresh_user = await current_user(request, session)
                    job = await job_for(session, job_id, fresh_user)
                except HTTPException:
                    yield "event: auth_expired\ndata: {}\n\n"
                    return
                events = (
                    await session.scalars(
                        select(JobEvent)
                        .where(JobEvent.job_id == job.id, JobEvent.id > last)
                        .order_by(JobEvent.id)
                        .limit(200)
                    )
                ).all()
                for event in events:
                    last = event.id
                    data = {
                        "stage": event.stage,
                        "progress": event.progress,
                        "message": event.message,
                        "created_at": event.created_at.isoformat(),
                        "eta_seconds": _eta(job),
                    }
                    encoded = json.dumps(data, ensure_ascii=False)
                    yield f"id: {event.id}\nevent: progress\ndata: {encoded}\n\n"
                if job.status in _TERMINAL and len(events) < 200:
                    done = json.dumps({"status": job.status, "job_id": str(job.id)})
                    yield f"event: done\ndata: {done}\n\n"
                    return
            heartbeat += 1
            if heartbeat % 15 == 0:
                yield ": keepalive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
