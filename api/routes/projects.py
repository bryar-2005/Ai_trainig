"""Owner-scoped projects, bounded uploads, and authenticated media streaming."""

from __future__ import annotations

import asyncio
import hashlib
import re
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import case, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.access import model_for, owner_clause, project_for
from api.db.models import Cue, Job, MediaFile, Project, ReviewEvent, User
from api.db.session import get_session
from api.schemas import ProjectCreate, ProjectPatch
from api.security import current_user
from api.serializers import job_json, media_json, project_json
from api.uploads import parse_project_upload
from ml.data.loaders import SubtitleLoadError, load_subtitles

router = APIRouter(prefix="/api", tags=["projects"])
_ACTIVE = ("queued", "running", "cancel_requested")
_VIDEO_TYPES = {".mp4": "video/mp4", ".mkv": "video/x-matroska", ".webm": "video/webm"}
_SUBTITLE_TYPES = {
    ".srt": "application/x-subrip",
    ".vtt": "text/vtt",
    ".ass": "text/plain",
    ".ssa": "text/plain",
}


async def _no_active_job(db: AsyncSession, project_id: uuid.UUID) -> None:
    active = await db.scalar(
        select(Job.id).where(Job.project_id == project_id, Job.status.in_(_ACTIVE)).limit(1)
    )
    if active:
        raise HTTPException(status_code=409, detail="Wait for the active project job to finish")


async def _stage_upload(upload: UploadFile, path: Path, limit: int) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    try:
        with path.open("xb") as destination:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413, detail="Uploaded file exceeds the size limit"
                    )
                digest.update(chunk)
                await asyncio.to_thread(destination.write, chunk)
    finally:
        await upload.close()
    if not size:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")
    return size, digest.hexdigest()


def _filename(upload: UploadFile, allowed: dict[str, str]) -> tuple[str, str, str]:
    name = Path((upload.filename or "").replace("\\", "/")).name
    name = "".join(char for char in name if char.isprintable())[:255]
    suffix = Path(name).suffix.casefold()
    if suffix not in allowed:
        raise HTTPException(status_code=422, detail="Unsupported file extension")
    return name, suffix, allowed[suffix]


def _check_video(path: Path, suffix: str) -> None:
    with path.open("rb") as handle:
        header = handle.read(16)
    valid = (suffix == ".mp4" and len(header) >= 8 and header[4:8] == b"ftyp") or (
        suffix in {".webm", ".mkv"} and header.startswith(b"\x1a\x45\xdf\xa3")
    )
    if not valid:
        raise HTTPException(status_code=422, detail="Video signature does not match its extension")


@router.post("/subtitles/inspect")
async def inspect_subtitle(
    request: Request,
    subtitle: UploadFile = File(),
    _user: User = Depends(current_user),
):
    """Parse an authorized upload for a live count without creating or storing a project."""
    _, suffix, _ = _filename(subtitle, _SUBTITLE_TYPES)
    root = request.app.state.settings.storage_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".inspect-", dir=root) as temporary:
        path = Path(temporary) / f"subtitle{suffix}"
        await _stage_upload(subtitle, path, request.app.state.settings.max_subtitle_bytes)
        try:
            cues = await asyncio.to_thread(load_subtitles, path)
        except SubtitleLoadError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid subtitle: {exc}") from exc
        return {"cue_count": len(cues), "duration_ms": max(cue.end_ms for cue in cues)}


@router.get("/dashboard/summary")
async def dashboard(user: User = Depends(current_user), db: AsyncSession = Depends(get_session)):
    project_filter = owner_clause(user)
    total_projects = await db.scalar(select(func.count(Project.id)).where(project_filter))
    summary = (
        await db.execute(
            select(
                func.count(Cue.id),
                func.avg(Cue.confidence_score),
                func.sum(case((Cue.status != "approved", 1), else_=0)),
            )
            .join(Project, Project.id == Cue.project_id)
            .where(project_filter)
        )
    ).one()
    recent = (
        await db.scalars(
            select(Project).where(project_filter).order_by(Project.updated_at.desc()).limit(5)
        )
    ).all()
    jobs = select(Job).where(Job.status.in_(_ACTIVE)).order_by(Job.created_at.desc()).limit(10)
    if user.role != "admin":
        jobs = jobs.where(Job.owner_id == user.id)
    return {
        "total_projects": total_projects or 0,
        "cue_count": summary[0] or 0,
        "average_confidence": summary[1],
        "pending_review": summary[2] or 0,
        "recent_projects": [project_json(item) for item in recent],
        "active_jobs": [job_json(item) for item in (await db.scalars(jobs)).all()],
    }


@router.get("/projects")
async def list_projects(
    q: str = Query(default="", max_length=200),
    status: str | None = Query(
        default=None, pattern="^(draft|translating|needs_review|completed|failed)$"
    ),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: Literal[
        "updated_desc", "created_desc", "created_asc", "confidence_desc", "confidence_asc"
    ] = "updated_desc",
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    conditions = [owner_clause(user)]
    if q.strip():
        escaped = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        conditions.append(Project.name.ilike(f"%{escaped}%", escape="\\"))
    if status:
        conditions.append(Project.status == status)
    count = await db.scalar(select(func.count(Project.id)).where(*conditions))
    averages = (
        select(Cue.project_id.label("project_id"), func.avg(Cue.confidence_score).label("score"))
        .group_by(Cue.project_id)
        .subquery()
    )
    ordering = {
        "updated_desc": Project.updated_at.desc(),
        "created_desc": Project.created_at.desc(),
        "created_asc": Project.created_at.asc(),
        "confidence_desc": averages.c.score.desc().nulls_last(),
        "confidence_asc": averages.c.score.asc().nulls_last(),
    }
    items = (
        await db.execute(
            select(Project, averages.c.score)
            .outerjoin(averages, averages.c.project_id == Project.id)
            .where(*conditions)
            .order_by(ordering[sort], Project.id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).all()
    return {
        "items": [{**project_json(item), "average_confidence": score} for item, score in items],
        "total": count,
        "page": page,
        "page_size": page_size,
    }


@router.post("/projects", status_code=201)
async def create_project(
    payload: ProjectCreate,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    if payload.model_version_id:
        model = await model_for(db, payload.model_version_id, user)
        if model.direction != payload.direction:
            raise HTTPException(
                status_code=422, detail="Model direction does not match the project"
            )
    item = Project(id=uuid.uuid4(), owner_id=user.id, **payload.model_dump(by_alias=True))
    db.add(item)
    await db.commit()
    return project_json(item)


@router.get("/projects/{project_id}")
async def get_project(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user)
    media = (await db.scalars(select(MediaFile).where(MediaFile.project_id == project.id))).all()
    cues = await db.scalar(select(func.count(Cue.id)).where(Cue.project_id == project.id))
    return {
        **project_json(project),
        "media": [media_json(item) for item in media],
        "cue_count": cues,
    }


@router.get("/projects/{project_id}/activity")
async def project_activity(
    project_id: uuid.UUID,
    limit: int = Query(default=10, ge=1, le=50),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user)
    rows = (
        await db.execute(
            select(ReviewEvent, Cue.cue_index, User.display_name)
            .join(Cue, Cue.id == ReviewEvent.cue_id)
            .join(User, User.id == ReviewEvent.actor_id)
            .where(Cue.project_id == project.id)
            .order_by(ReviewEvent.created_at.desc(), ReviewEvent.id.desc())
            .limit(limit)
        )
    ).all()
    return {
        "items": [
            {
                "id": str(event.id),
                "cue_id": str(event.cue_id),
                "cue_index": index,
                "action": event.action,
                "revision": event.cue_revision,
                "actor": actor,
                "created_at": event.created_at,
            }
            for event, index, actor in rows
        ]
    }


@router.patch("/projects/{project_id}")
async def update_project(
    project_id: uuid.UUID,
    payload: ProjectPatch,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user, lock=True)
    await _no_active_job(db, project.id)
    changes = payload.model_dump(exclude_unset=True, by_alias=True)
    if any(value is None for key, value in changes.items() if key != "model_version_id"):
        raise HTTPException(status_code=422, detail="Only model_version_id may be cleared")
    if changes.get("model_version_id"):
        model = await model_for(db, changes["model_version_id"], user)
        if model.direction != project.direction:
            raise HTTPException(
                status_code=422, detail="Model direction does not match the project"
            )
    if changes.get("target_cps", project.target_cps) > changes.get("hard_cps", project.hard_cps):
        raise HTTPException(status_code=422, detail="Target CPS may not exceed the hard cap")
    rules = changes.get("profanity_replacements", project.profanity_replacements)
    if changes.get("soften_profanity", project.soften_profanity) and not rules:
        raise HTTPException(
            status_code=422, detail="Provide target-language softening replacements"
        )
    keys = [rule["source"].casefold() for rule in rules]
    if len(keys) != len(set(keys)):
        raise HTTPException(status_code=422, detail="Softening source terms must be unique")
    for key, value in changes.items():
        setattr(project, key, value)
    if {"target_cps", "hard_cps", "max_chars", "max_lines"} & changes.keys():
        from api.review import recalculate_cue, terms_for

        terms = await terms_for(db, project.id)
        cues = (await db.scalars(select(Cue).where(Cue.project_id == project.id))).all()
        for cue in cues:
            await recalculate_cue(db, cue, project, terms)
    await db.commit()
    return project_json(project)


@router.delete("/projects/{project_id}", status_code=204)
async def delete_project(
    project_id: uuid.UUID,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user, lock=True)
    await _no_active_job(db, project.id)
    media = (await db.scalars(select(MediaFile).where(MediaFile.project_id == project.id))).all()
    # Snapshots may contain this project's text. Invalidate all of the owner's
    # snapshots, including in-progress uploads, before deleting the project.
    from api.db.models import FeedbackExport

    snapshots = (
        await db.scalars(
            select(FeedbackExport)
            .where(FeedbackExport.owner_id == project.owner_id)
            .order_by(FeedbackExport.id)
            .with_for_update()
        )
    ).all()
    for item in [*media, *snapshots]:
        await asyncio.to_thread(request.app.state.storage.delete, item.storage_key)
    for item in snapshots:
        await db.delete(item)
    await db.delete(project)
    await db.commit()
    return Response(status_code=204)


@router.post(
    "/projects/{project_id}/upload",
    status_code=201,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["subtitle"],
                        "properties": {
                            "subtitle": {"type": "string", "format": "binary"},
                            "video": {"type": "string", "format": "binary"},
                        },
                    }
                }
            },
        }
    },
)
async def upload_project(
    project_id: uuid.UUID,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
    uploads: tuple[UploadFile, UploadFile | None] = Depends(parse_project_upload),
):
    subtitle, video = uploads
    project = await project_for(db, project_id, user)
    await _no_active_job(db, project.id)
    if await db.scalar(select(MediaFile.id).where(MediaFile.project_id == project.id)):
        raise HTTPException(status_code=409, detail="This project already has an upload")
    subtitle_name, subtitle_suffix, subtitle_type = _filename(subtitle, _SUBTITLE_TYPES)
    video_details = _filename(video, _VIDEO_TYPES) if video else None
    # Do not hold a database transaction open during potentially large transfers.
    await db.commit()
    settings = request.app.state.settings
    staging_root = settings.storage_root.expanduser().resolve()
    staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    stored: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix=".upload-", dir=staging_root) as temp:
            subtitle_path = Path(temp) / f"subtitle{subtitle_suffix}"
            subtitle_size, subtitle_hash = await _stage_upload(
                subtitle, subtitle_path, settings.max_subtitle_bytes
            )
            try:
                parsed = await asyncio.to_thread(load_subtitles, subtitle_path)
            except SubtitleLoadError as exc:
                raise HTTPException(status_code=422, detail=f"Invalid subtitle: {exc}") from exc
            video_path: Path | None = None
            video_size, video_hash = 0, ""
            if video and video_details:
                video_path = Path(temp) / f"video{video_details[1]}"
                video_size, video_hash = await _stage_upload(
                    video, video_path, settings.max_video_bytes
                )
                _check_video(video_path, video_details[1])
            project = await project_for(db, project_id, user, lock=True)
            await _no_active_job(db, project.id)
            if await db.scalar(select(MediaFile.id).where(MediaFile.project_id == project.id)):
                raise HTTPException(status_code=409, detail="This project already has an upload")
            files = [
                (
                    "subtitle",
                    subtitle_path,
                    subtitle_name,
                    subtitle_suffix,
                    subtitle_type,
                    subtitle_size,
                    subtitle_hash,
                )
            ]
            if video_path and video_details:
                files.append(("video", video_path, *video_details, video_size, video_hash))
            media: list[MediaFile] = []
            for kind, path, name, suffix, content_type, size, digest in files:
                key = f"{project.id}/{uuid.uuid4().hex}{suffix}"
                await asyncio.to_thread(request.app.state.storage.put_file, path, key, content_type)
                stored.append(key)
                media.append(
                    MediaFile(
                        id=uuid.uuid4(),
                        project_id=project.id,
                        kind=kind,
                        storage_key=key,
                        original_name=name,
                        content_type=content_type,
                        size_bytes=size,
                        sha256=digest,
                    )
                )
            db.add_all(media)
            for cue in parsed:
                db.add(
                    Cue(
                        id=uuid.uuid4(),
                        project_id=project.id,
                        cue_index=cue.index,
                        original_start_ms=cue.start_ms,
                        original_end_ms=cue.end_ms,
                        start_ms=cue.start_ms,
                        end_ms=cue.end_ms,
                        source_cue_ids=[cue.index],
                        sentence_source_cue_ids=[cue.index],
                        source_text=cue.text,
                        normalized_source_text=cue.text,
                        speaker=cue.speaker,
                        speaker_labels=list(cue.speaker_labels),
                        sound_effects=list(cue.sound_effects),
                        style=cue.style,
                        machine_text="",
                        human_text=None,
                        qa_warnings=["untranslated"],
                        flagged=True,
                        status="unreviewed",
                    )
                )
            await db.commit()
            stored.clear()
            return {
                "media": [media_json(item) for item in media],
                "cue_count": len(parsed),
                "duration_ms": max(item.end_ms for item in parsed),
            }
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="An upload already exists") from exc
    finally:
        for key in stored:
            await asyncio.to_thread(request.app.state.storage.delete, key)


def _range(value: str | None, size: int) -> tuple[int, int, bool]:
    if value is None:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or not any(match.groups()):
        raise HTTPException(
            status_code=416,
            detail="Only one valid byte range is supported",
            headers={"Content-Range": f"bytes */{size}"},
        )
    first, last = match.groups()
    if not first:
        amount = int(last)
        start, end = max(0, size - amount), size - 1
    else:
        start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise HTTPException(
            status_code=416,
            detail="Unsatisfiable byte range",
            headers={"Content-Range": f"bytes */{size}"},
        )
    return start, end, True


@router.get("/projects/{project_id}/media/{media_id}")
async def stream_media(
    project_id: uuid.UUID,
    media_id: uuid.UUID,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user)
    media = await db.scalar(
        select(MediaFile).where(MediaFile.id == media_id, MediaFile.project_id == project.id)
    )
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")
    start, end, partial = _range(request.headers.get("range"), media.size_bytes)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff",
    }
    if partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{media.size_bytes}"
    if media.kind == "subtitle":
        headers["Content-Disposition"] = "attachment"
    return StreamingResponse(
        request.app.state.storage.iter_range(media.storage_key, start, end),
        status_code=206 if partial else 200,
        media_type=media.content_type,
        headers=headers,
    )
