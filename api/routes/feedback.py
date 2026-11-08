"""Private retained feedback-export jobs and owner-only downloads."""

from __future__ import annotations

import asyncio
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import FeedbackExport, User, now
from api.db.session import get_session
from api.jobs import enqueue_user_job
from api.security import current_user

router = APIRouter(prefix="/api", tags=["feedback"])


class FeedbackExportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: Literal["en-ckb", "ckb-en"] | None = None


def export_json(item: FeedbackExport) -> dict:
    return {
        "id": str(item.id),
        "job_id": str(item.job_id),
        "direction": item.direction,
        "cue_count": item.cue_count,
        "size_bytes": item.size_bytes,
        "sha256": item.sha256,
        "created_at": item.created_at,
        "expires_at": item.expires_at,
    }


@router.post("/feedback/exports", status_code=202)
async def create_export(
    payload: FeedbackExportIn,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    job = await enqueue_user_job(
        db, request, user, "feedback_export", {"direction": payload.direction, "scheduled": False}
    )
    return {"job_id": str(job.id), "status": job.status}


@router.get("/feedback/exports")
async def list_exports(
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    # Admin privileges never grant access to another person's training text.
    rows = (
        await db.scalars(
            select(FeedbackExport)
            .where(
                FeedbackExport.owner_id == user.id,
                FeedbackExport.status == "ready",
                FeedbackExport.expires_at > now(),
            )
            .order_by(FeedbackExport.created_at.desc())
            .limit(100)
        )
    ).all()
    settings = request.app.state.settings
    return {
        "items": [export_json(item) for item in rows],
        "nightly_schedule_enabled": settings.nightly_feedback_export_enabled,
        "retention_days": settings.feedback_export_retention_days,
    }


async def _owned_export(db: AsyncSession, export_id: uuid.UUID, user: User, *, lock=False):
    query = select(FeedbackExport).where(
        FeedbackExport.id == export_id, FeedbackExport.owner_id == user.id
    )
    if lock:
        query = query.with_for_update()
    item = await db.scalar(query)
    if not item:
        raise HTTPException(status_code=404, detail="Feedback export not found")
    return item


@router.get("/feedback/exports/{export_id}/download")
async def download_export(
    export_id: uuid.UUID,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    item = await _owned_export(db, export_id, user)
    # SQLite strips timezone info, so normalize for tests as well as PostgreSQL.
    from api.security import _aware

    if item.status != "ready" or _aware(item.expires_at) <= now():
        raise HTTPException(status_code=404, detail="Feedback export not found")
    return StreamingResponse(
        request.app.state.storage.iter_range(item.storage_key, 0, item.size_bytes - 1),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": 'attachment; filename="approved-feedback.jsonl"',
            "Content-Length": str(item.size_bytes),
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Content-SHA256": item.sha256,
        },
    )


@router.delete("/feedback/exports/{export_id}", status_code=204)
async def delete_export(
    export_id: uuid.UUID,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    item = await _owned_export(db, export_id, user, lock=True)
    await asyncio.to_thread(request.app.state.storage.delete, item.storage_key)
    await db.delete(item)
    await db.commit()
    return Response(status_code=204)
