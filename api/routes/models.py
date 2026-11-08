"""Trusted model registry, per-user activation, and reviewed feedback."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.access import model_for
from api.db.models import ModelVersion, User
from api.db.session import get_session
from api.feedback import approved_feedback as _feedback
from api.feedback import feedback_row as _feedback_row
from api.jobs import enqueue_user_job
from api.routes.feedback import FeedbackExportIn
from api.schemas import CompareIn, ModelCreate, SampleIn, TrainIn
from api.security import current_user, require_admin
from api.serializers import model_json

router = APIRouter(prefix="/api", tags=["models and feedback"])


def _local_model_path(value: str, root: Path) -> str:
    base = root.expanduser().resolve()
    path = Path(value).expanduser().resolve()
    if not path.is_dir() or not path.is_relative_to(base):
        raise HTTPException(
            status_code=422, detail="Model directories must exist inside the configured model root"
        )
    return str(path)


def _model_source(value: str, root: Path) -> str:
    if Path(value).is_absolute() or value.startswith((".", "~")) or Path(value).is_dir():
        return _local_model_path(value, root)
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value)
        or ".." in value
    ):
        raise HTTPException(
            status_code=422, detail="Model source must be a trusted local directory or Hub ID"
        )
    return value


@router.get("/models")
async def list_models(user: User = Depends(current_user), db: AsyncSession = Depends(get_session)):
    query = select(ModelVersion)
    if user.role != "admin":
        query = query.where(or_(ModelVersion.owner_id.is_(None), ModelVersion.owner_id == user.id))
    rows = (await db.scalars(query.order_by(ModelVersion.created_at.desc()))).all()
    active = {
        str(value) for key, value in user.preferences.items() if key.startswith("active_model_")
    }
    return {"items": [model_json(row, active=str(row.id) in active) for row in rows]}


@router.post("/models", status_code=201)
async def register_model(
    payload: ModelCreate,
    request: Request,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
):
    root = request.app.state.settings.model_root
    source = _model_source(payload.source, root)
    adapter = _local_model_path(payload.adapter_dir, root) if payload.adapter_dir else None
    local = Path(source)
    if local.is_dir():
        has_tokenizer = any(
            (local / name).is_file() for name in ("tokenizer.json", "sentencepiece.bpe.model")
        )
        has_weights = any(local.glob("*.safetensors")) or any(local.glob("pytorch_model*.bin"))
        if not (local / "config.json").is_file() or not has_tokenizer or not has_weights:
            raise HTTPException(status_code=422, detail="Local model assets are incomplete")
        if (
            local / "smoke_model.json"
        ).is_file() and not request.app.state.settings.allow_smoke_models:
            raise HTTPException(status_code=422, detail="Random smoke models are test-only")
    if payload.kind == "merged":
        if not local.is_dir() or not (local / "merge_manifest.json").is_file():
            raise HTTPException(
                status_code=422, detail="Merged models require a local merge manifest"
            )
        try:
            merged = json.loads((local / "merge_manifest.json").read_text(encoding="utf-8"))
            if merged.get("direction") != payload.direction:
                raise ValueError("direction mismatch")
            if merged.get("smoke_test") and not request.app.state.settings.allow_smoke_models:
                raise ValueError("smoke model")
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail="Invalid or test-only merge manifest"
            ) from exc
    if adapter:
        directory = Path(adapter)
        if any(
            not (directory / name).is_file()
            for name in (
                "training_manifest.json",
                "adapter_config.json",
                "adapter_model.safetensors",
            )
        ):
            raise HTTPException(
                status_code=422, detail="Adapter assets or training manifest are missing"
            )
        try:
            manifest = json.loads(
                (directory / "training_manifest.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("status") != "completed"
                or manifest.get("config", {}).get("direction") != payload.direction
                or (
                    manifest.get("smoke_test") and not request.app.state.settings.allow_smoke_models
                )
            ):
                raise ValueError("invalid or test-only adapter")
            expected = manifest.get("base_model_sha256")
            if local.is_dir():
                from ml.train.model_identity import local_model_digest

                if not expected or await asyncio.to_thread(local_model_digest, source) != expected:
                    raise ValueError("base identity mismatch")
            elif (
                expected is not None
                or manifest.get("actual_base_model") != source
                or not manifest.get("base_revision_resolved")
            ):
                raise ValueError("base identity mismatch")
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail="Adapter manifest or base identity is invalid or test-only"
            ) from exc
    duplicate = await db.scalar(
        select(ModelVersion.id).where(
            ModelVersion.owner_id.is_(None),
            ModelVersion.name == payload.name,
            ModelVersion.direction == payload.direction,
        )
    )
    if duplicate:
        raise HTTPException(
            status_code=409, detail="A global model with this name and direction exists"
        )
    model = ModelVersion(
        id=uuid.uuid4(),
        owner_id=None,
        name=payload.name,
        direction=payload.direction,
        kind=payload.kind,
        source=source,
        adapter_dir=adapter,
        revision=payload.revision,
        metrics=payload.metrics,
        metadata_json={
            "registered_by": str(admin.id),
            "metrics_provenance": "administrator-supplied, not independently verified",
        },
    )
    db.add(model)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409, detail="A global model with this name and direction exists"
        ) from exc
    return model_json(model)


@router.get("/models/feedback-stats")
async def feedback_stats(
    user: User = Depends(current_user), db: AsyncSession = Depends(get_session)
):
    rows = await _feedback(db, user)
    directions = {}
    for direction in ("en-ckb", "ckb-en"):
        matching = [(cue, project) for cue, project in rows if project.direction == direction]
        projects = {project.id for _, project in matching}
        directions[direction] = {
            "approved_edits": len(matching),
            "films": len(projects),
            "film_disjoint_training_possible": len(projects) >= 2,
        }
    return {"approved_edits": len(rows), "directions": directions, "language_id_verified": False}


@router.post("/models/{model_id}/activate")
async def activate_model(
    model_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_session)
):
    model = await model_for(db, model_id, user)
    prefs = dict(user.preferences)
    prefs[f"active_model_{model.direction}"] = str(model.id)
    user.preferences = prefs
    await db.commit()
    return model_json(model, active=True)


@router.post("/feedback/export")
async def export_feedback(
    payload: FeedbackExportIn,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    rows = await _feedback(db, user, payload.direction)
    output = "".join(
        json.dumps(_feedback_row(cue, project), ensure_ascii=False, separators=(",", ":")) + "\n"
        for cue, project in rows
    )
    return Response(
        output,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": 'attachment; filename="approved-feedback.jsonl"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/models/train", status_code=202)
async def train_model(
    payload: TrainIn,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    model = await model_for(db, payload.model_version_id, user)
    if model.direction != payload.direction:
        raise HTTPException(status_code=422, detail="Model direction does not match the feedback")
    if not payload.allow_unverified_language:
        raise HTTPException(
            status_code=422,
            detail="Approved edits have not been language-ID verified; explicit opt-in is required",
        )
    rows = await _feedback(db, user, payload.direction)
    if len({project.id for _, project in rows}) < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                "At least two films with approved edits are required "
                "for film-disjoint train/validation"
            ),
        )
    job = await enqueue_user_job(
        db,
        request,
        user,
        "train",
        {
            "direction": payload.direction,
            "model_id": str(model.id),
            "epochs": payload.epochs,
            "learning_rate": payload.learning_rate,
            "allow_unverified_language": True,
            "cue_versions": [[str(cue.id), cue.revision] for cue, _ in rows],
        },
    )
    return {"job_id": str(job.id), "status": job.status}


@router.post("/models/sample", status_code=202)
async def sample_model(
    payload: SampleIn,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    model = await model_for(db, payload.model_version_id, user)
    if model.direction != payload.direction:
        raise HTTPException(status_code=422, detail="Model direction does not match the sample")
    job = await enqueue_user_job(db, request, user, "sample", payload.model_dump(mode="json"))
    return {"job_id": str(job.id), "status": job.status}


@router.post("/models/compare", status_code=202)
async def compare_models(
    payload: CompareIn,
    request: Request,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    first, second = (
        await model_for(db, payload.model_a, user),
        await model_for(db, payload.model_b, user),
    )
    if first.direction != payload.direction or second.direction != payload.direction:
        raise HTTPException(
            status_code=422, detail="Both models must match the requested direction"
        )
    job = await enqueue_user_job(db, request, user, "compare", payload.model_dump(mode="json"))
    return {"job_id": str(job.id), "status": job.status}
