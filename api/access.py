"""Central owner checks for every project-scoped resource."""

from __future__ import annotations

import uuid

from fastapi import HTTPException
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import Cue, GlossaryTerm, Job, ModelVersion, Project, User


def owner_clause(user: User):
    return True if user.role == "admin" else Project.owner_id == user.id


async def project_for(
    db: AsyncSession, project_id: uuid.UUID, user: User, *, lock: bool = False
) -> Project:
    query = select(Project).where(Project.id == project_id, owner_clause(user))
    if lock:
        query = query.with_for_update().execution_options(populate_existing=True)
    value = await db.scalar(query)
    if value is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return value


async def cue_for(db: AsyncSession, cue_id: uuid.UUID, user: User, *, lock: bool = False):
    query = (
        select(Cue, Project)
        .join(Project, Project.id == Cue.project_id)
        .where(Cue.id == cue_id, owner_clause(user))
    )
    if lock:
        query = query.with_for_update(of=Cue).execution_options(populate_existing=True)
    row = (await db.execute(query)).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Cue not found")
    return row.Cue, row.Project


async def glossary_for(db: AsyncSession, term_id: uuid.UUID, user: User, *, lock: bool = False):
    query = (
        select(GlossaryTerm, Project)
        .join(Project, Project.id == GlossaryTerm.project_id)
        .where(GlossaryTerm.id == term_id, owner_clause(user))
    )
    if lock:
        query = query.with_for_update(of=GlossaryTerm).execution_options(populate_existing=True)
    row = (await db.execute(query)).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Glossary term not found")
    return row.GlossaryTerm, row.Project


async def job_for(db: AsyncSession, job_id: uuid.UUID, user: User, *, lock: bool = False) -> Job:
    query = select(Job).where(Job.id == job_id)
    if user.role != "admin":
        query = query.where(Job.owner_id == user.id)
    if lock:
        query = query.with_for_update().execution_options(populate_existing=True)
    job = await db.scalar(query)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


async def model_for(db: AsyncSession, model_id: uuid.UUID, user: User) -> ModelVersion:
    query = select(ModelVersion).where(ModelVersion.id == model_id)
    if user.role != "admin":
        query = query.where(or_(ModelVersion.owner_id.is_(None), ModelVersion.owner_id == user.id))
    model = await db.scalar(query)
    if model is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    return model
