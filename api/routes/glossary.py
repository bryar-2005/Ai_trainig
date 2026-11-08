"""Project terminology, conservative entity proposals, and bounded imports."""

from __future__ import annotations

import csv
import io
import json
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.access import glossary_for, project_for
from api.db.models import Cue, GlossaryTerm, Project, User
from api.db.session import get_session
from api.review import recalculate_cue, terms_for
from api.routes.projects import _no_active_job
from api.schemas import GlossaryIn, GlossaryPatch
from api.security import current_user
from api.serializers import term_json
from ml.data.normalize import normalize_english, normalize_sorani
from ml.inference.glossary import HeuristicEntityExtractor, merge_entities

router = APIRouter(prefix="/api", tags=["glossary"])
_MAX_IMPORT = 1024 * 1024


async def _locked_term(db: AsyncSession, term_id: uuid.UUID, user: User):
    term, project = await glossary_for(db, term_id, user)
    project = await project_for(db, project.id, user, lock=True)
    term, _ = await glossary_for(db, term.id, user, lock=True)
    return term, project


def _values(payload: GlossaryIn) -> dict:
    english, sorani = normalize_english(payload.english), normalize_sorani(payload.sorani)
    if not english or not sorani or "\n" in english or "\n" in sorani:
        raise HTTPException(status_code=422, detail="Terms must be nonempty single-line text")
    return {
        "english": english,
        "english_key": english.casefold(),
        "sorani": sorani,
        "category": payload.category,
        "locked": payload.locked,
    }


async def _check_unique_reverse(
    db: AsyncSession, project: Project, values: dict, exclude=None
) -> None:
    if project.direction != "ckb-en" or not values["locked"]:
        return
    existing = (
        await db.scalars(
            select(GlossaryTerm).where(
                GlossaryTerm.project_id == project.id, GlossaryTerm.locked.is_(True)
            )
        )
    ).all()
    key = values["sorani"].casefold()
    if any(term.id != exclude and term.sorani.casefold() == key for term in existing):
        raise HTTPException(status_code=409, detail="A locked Sorani source term already exists")


async def _recheck(db: AsyncSession, project: Project) -> None:
    await db.flush()
    terms = await terms_for(db, project.id)
    cues = (await db.scalars(select(Cue).where(Cue.project_id == project.id))).all()
    for cue in cues:
        await recalculate_cue(db, cue, project, terms)


@router.get("/projects/{project_id}/glossary")
async def list_terms(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user)
    rows = (
        await db.scalars(
            select(GlossaryTerm)
            .where(GlossaryTerm.project_id == project.id)
            .order_by(GlossaryTerm.english_key)
        )
    ).all()
    return {"items": [term_json(item) for item in rows]}


@router.post("/projects/{project_id}/glossary", status_code=201)
async def add_term(
    project_id: uuid.UUID,
    payload: GlossaryIn,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user, lock=True)
    await _no_active_job(db, project.id)
    values = _values(payload)
    await _check_unique_reverse(db, project, values)
    term = GlossaryTerm(id=uuid.uuid4(), project_id=project.id, origin="user", **values)
    db.add(term)
    try:
        await _recheck(db, project)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="The English term already exists") from exc
    return term_json(term)


@router.patch("/glossary/{term_id}")
async def update_term(
    term_id: uuid.UUID,
    payload: GlossaryPatch,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    term, project = await _locked_term(db, term_id, user)
    await _no_active_job(db, project.id)
    changes = payload.model_dump(exclude_unset=True)
    if not changes or any(value is None for value in changes.values()):
        raise HTTPException(status_code=422, detail="Supply non-null glossary changes")
    values = _values(
        GlossaryIn(
            english=changes.get("english", term.english),
            sorani=changes.get("sorani", term.sorani),
            category=changes.get("category", term.category),
            locked=changes.get("locked", term.locked),
        )
    )
    await _check_unique_reverse(db, project, values, term.id)
    for key, value in values.items():
        setattr(term, key, value)
    term.origin = "user"
    try:
        await _recheck(db, project)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="The English term already exists") from exc
    return term_json(term)


@router.delete("/glossary/{term_id}", status_code=204)
async def delete_term(
    term_id: uuid.UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_session)
):
    term, project = await _locked_term(db, term_id, user)
    await _no_active_job(db, project.id)
    await db.delete(term)
    await _recheck(db, project)
    await db.commit()


@router.post("/projects/{project_id}/glossary/recheck")
async def recheck_terms(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    """Recalculate glossary and QA warnings without changing subtitle text or timing."""
    project = await project_for(db, project_id, user, lock=True)
    await _no_active_job(db, project.id)
    count = await db.scalar(select(func.count(Cue.id)).where(Cue.project_id == project.id))
    await _recheck(db, project)
    await db.commit()
    return {"checked": count or 0}


@router.post("/projects/{project_id}/glossary/extract")
async def extract_terms(
    project_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user, lock=True)
    await _no_active_job(db, project.id)
    cues = (
        await db.scalars(select(Cue).where(Cue.project_id == project.id).order_by(Cue.cue_index))
    ).all()
    texts = [cue.source_text for cue in cues] + [
        label for cue in cues for label in cue.speaker_labels
    ]
    current = await terms_for(db, project.id)
    proposals = merge_entities(current, HeuristicEntityExtractor().extract(texts))
    added = [term for term in proposals if term.origin == "ner_proposal" and term not in current]
    for term in added:
        db.add(
            GlossaryTerm(
                id=uuid.uuid4(),
                project_id=project.id,
                english=term.english,
                english_key=term.english.casefold(),
                sorani=term.sorani,
                category=term.category,
                locked=False,
                origin="ner_proposal",
            )
        )
    await db.commit()
    return {"added": len(added), "proposals_require_review": True}


@router.post("/projects/{project_id}/glossary/import")
async def import_terms(
    project_id: uuid.UUID,
    file: UploadFile = File(),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    project = await project_for(db, project_id, user, lock=True)
    await _no_active_job(db, project.id)
    try:
        raw = await file.read(_MAX_IMPORT + 1)
    finally:
        await file.close()
    if len(raw) > _MAX_IMPORT:
        raise HTTPException(status_code=413, detail="Glossary import exceeds 1 MiB")
    try:
        text = raw.decode("utf-8-sig")
        if (file.filename or "").casefold().endswith(".json"):
            items = json.loads(text)
        elif (file.filename or "").casefold().endswith(".csv"):
            items = list(csv.DictReader(io.StringIO(text)))
            for item in items:
                value = (item.get("locked") or "true").strip().casefold()
                if value not in {"true", "false", "1", "0"}:
                    raise ValueError("CSV locked must be true or false")
                item["locked"] = value in {"true", "1"}
        else:
            raise ValueError("Use a .csv or .json file")
        if not isinstance(items, list) or not 1 <= len(items) <= 10000:
            raise ValueError("Import must contain 1–10000 entries")
        values = [_values(GlossaryIn.model_validate(item)) for item in items]
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid glossary import: {exc}") from exc
    keys = [item["english_key"] for item in values]
    reverse_keys = [item["sorani"].casefold() for item in values if item["locked"]]
    if len(keys) != len(set(keys)) or (
        project.direction == "ckb-en" and len(reverse_keys) != len(set(reverse_keys))
    ):
        raise HTTPException(
            status_code=409, detail="Import contains duplicate locked or English terms"
        )
    existing = (
        await db.scalars(select(GlossaryTerm).where(GlossaryTerm.project_id == project.id))
    ).all()
    if set(keys) & {item.english_key for item in existing} or (
        project.direction == "ckb-en"
        and set(reverse_keys) & {item.sorani.casefold() for item in existing if item.locked}
    ):
        raise HTTPException(status_code=409, detail="An imported term already exists")
    db.add_all(
        [
            GlossaryTerm(id=uuid.uuid4(), project_id=project.id, origin="import", **item)
            for item in values
        ]
    )
    try:
        await _recheck(db, project)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="An imported term already exists") from exc
    return {"imported": len(values)}
