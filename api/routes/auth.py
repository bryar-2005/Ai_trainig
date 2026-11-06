"""Cookie-based account registration and revocable token rotation."""

from __future__ import annotations

import hmac
import secrets
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import RefreshSession, User, now
from api.db.session import get_session
from api.schemas import LoginIn, RegisterIn
from api.security import (
    REFRESH_COOKIE,
    _aware,
    clear_session_cookies,
    current_user,
    decode_token,
    hash_password,
    set_session_cookies,
    token_hash,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["authentication"])


def user_output(user: User) -> dict:
    return {
        "id": str(user.id),
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role,
        "preferences": user.preferences,
        "created_at": user.created_at,
    }


async def _new_session(user: User, request: Request, response: Response, db: AsyncSession) -> None:
    settings = request.app.state.settings
    session_id, jti = uuid.uuid4(), secrets.token_urlsafe(32)
    db.add(
        RefreshSession(
            id=session_id,
            user_id=user.id,
            token_hash=token_hash(jti),
            expires_at=now() + timedelta(days=settings.refresh_days),
        )
    )
    await db.commit()
    set_session_cookies(response, settings, user.id, session_id, jti)


@router.post("/register", status_code=201)
async def register(
    payload: RegisterIn,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_session),
):
    user = User(
        id=uuid.uuid4(),
        email=str(payload.email).casefold(),
        display_name=payload.display_name,
        password_hash=await hash_password(payload.password),
        role="user",
        preferences={},
    )
    db.add(user)
    try:
        await db.flush()
        await _new_session(user, request, response, db)
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409, detail="An account with that email already exists"
        ) from exc
    return user_output(user)


@router.post("/login")
async def login(
    payload: LoginIn, request: Request, response: Response, db: AsyncSession = Depends(get_session)
):
    user = await db.scalar(select(User).where(User.email == str(payload.email).casefold()))
    valid = await verify_password(user.password_hash if user else None, payload.password)
    if not valid or not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    await _new_session(user, request, response, db)
    return user_output(user)


@router.post("/refresh")
async def refresh(request: Request, response: Response, db: AsyncSession = Depends(get_session)):
    value = request.cookies.get(REFRESH_COOKIE)
    if not value:
        raise HTTPException(status_code=401, detail="Refresh session required")
    settings = request.app.state.settings
    payload = decode_token(settings, value, "refresh")
    row = await db.scalar(
        select(RefreshSession)
        .where(RefreshSession.id == uuid.UUID(payload["sid"]))
        .with_for_update()
    )
    if (
        row is None
        or row.user_id != uuid.UUID(payload["sub"])
        or row.revoked_at is not None
        or _aware(row.expires_at) <= now()
    ):
        raise HTTPException(status_code=401, detail="Session expired or revoked")
    if not hmac.compare_digest(row.token_hash, token_hash(payload["jti"])):
        # Reuse of a rotated token revokes the whole session, including access tokens.
        row.revoked_at = now()
        await db.commit()
        denied = JSONResponse({"detail": "Refresh token already used"}, status_code=401)
        clear_session_cookies(denied, settings)
        return denied
    user = await db.get(User, row.user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Session expired or revoked")
    jti = secrets.token_urlsafe(32)
    row.token_hash = token_hash(jti)
    row.rotated_at = now()
    # The original session's absolute expiration is not extended by refresh.
    await db.commit()
    set_session_cookies(response, settings, user.id, row.id, jti)
    return user_output(user)


@router.post("/logout", status_code=204)
async def logout(request: Request, response: Response, db: AsyncSession = Depends(get_session)):
    value = request.cookies.get(REFRESH_COOKIE)
    if value:
        try:
            payload = decode_token(request.app.state.settings, value, "refresh")
        except HTTPException:
            payload = None
        if payload:
            row = await db.get(RefreshSession, uuid.UUID(payload["sid"]))
            if row and row.user_id == uuid.UUID(payload["sub"]):
                row.revoked_at = now()
                await db.commit()
    clear_session_cookies(response, request.app.state.settings)
    return response


@router.get("/me")
async def me(user: User = Depends(current_user)):
    return user_output(user)
