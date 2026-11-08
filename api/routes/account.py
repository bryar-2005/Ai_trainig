"""Account settings, expiring personal API tokens, and SMTP password reset."""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.account_schemas import (
    ApiTokenIn,
    ForgotPasswordIn,
    PasswordIn,
    PreferencesIn,
    ProfileIn,
    ResetPasswordIn,
)
from api.db.models import PasswordReset, PersonalApiToken, RefreshSession, User, now
from api.db.session import get_session
from api.email import send_reset_email
from api.routes.auth import user_output
from api.security import (
    _aware,
    clear_session_cookies,
    current_user,
    hash_password,
    token_hash,
    verify_password,
)

router = APIRouter(prefix="/api", tags=["account"])
logger = logging.getLogger("sorani.account")


def _public_token(item: PersonalApiToken) -> dict:
    return {
        "id": str(item.id),
        "name": item.name,
        "created_at": item.created_at,
        "expires_at": item.expires_at,
        "expired": _aware(item.expires_at) <= now(),
        "revoked_at": item.revoked_at,
    }


async def _lock_user(db: AsyncSession, user_id: uuid.UUID) -> User:
    user = await db.scalar(
        select(User)
        .where(User.id == user_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Account unavailable")
    return user


async def _revoke_credentials(db: AsyncSession, user_id: uuid.UUID) -> None:
    timestamp = now()
    await db.execute(
        update(RefreshSession)
        .where(RefreshSession.user_id == user_id, RefreshSession.revoked_at.is_(None))
        .values(revoked_at=timestamp)
    )
    await db.execute(
        update(PersonalApiToken)
        .where(PersonalApiToken.user_id == user_id, PersonalApiToken.revoked_at.is_(None))
        .values(revoked_at=timestamp)
    )
    await db.execute(
        update(PasswordReset)
        .where(PasswordReset.user_id == user_id, PasswordReset.used_at.is_(None))
        .values(used_at=timestamp)
    )


@router.patch("/account/profile")
async def update_profile(
    payload: ProfileIn, user: User = Depends(current_user), db: AsyncSession = Depends(get_session)
):
    locked = await _lock_user(db, user.id)
    locked.display_name = payload.display_name
    await db.commit()
    return user_output(locked)


@router.put("/account/preferences")
async def update_preferences(
    payload: PreferencesIn,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    locked = await _lock_user(db, user.id)
    # Keep private active-model selections; only known preference keys are writable here.
    locked.preferences = {**locked.preferences, **payload.model_dump()}
    await db.commit()
    return user_output(locked)


@router.post("/account/password", status_code=204)
async def change_password(
    payload: PasswordIn,
    request: Request,
    response: Response,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    locked = await _lock_user(db, user.id)
    if not await verify_password(locked.password_hash, payload.current_password):
        raise HTTPException(status_code=403, detail="Current password is incorrect")
    locked.password_hash = await hash_password(payload.new_password)
    await _revoke_credentials(db, locked.id)
    await db.commit()
    clear_session_cookies(response, request.app.state.settings)
    response.status_code = 204
    return response


@router.get("/account/tokens")
async def list_tokens(user: User = Depends(current_user), db: AsyncSession = Depends(get_session)):
    items = (
        await db.scalars(
            select(PersonalApiToken)
            .where(PersonalApiToken.user_id == user.id)
            .order_by(PersonalApiToken.created_at.desc())
        )
    ).all()
    return {"items": [_public_token(item) for item in items]}


@router.post("/account/tokens", status_code=201)
async def create_token(
    payload: ApiTokenIn,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    locked = await _lock_user(db, user.id)
    if not await verify_password(locked.password_hash, payload.password):
        raise HTTPException(status_code=403, detail="Current password is incorrect")
    secret = "sst_" + secrets.token_urlsafe(40)
    item = PersonalApiToken(
        id=uuid.uuid4(),
        user_id=locked.id,
        name=payload.name,
        token_hash=token_hash(secret),
        expires_at=now() + timedelta(days=payload.expires_days),
    )
    db.add(item)
    await db.commit()
    # The plaintext secret is returned once and is never persisted or logged.
    return {**_public_token(item), "token": secret}


@router.delete("/account/tokens/{token_id}", status_code=204)
async def revoke_token(
    token_id: uuid.UUID,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_session),
):
    item = await db.scalar(
        select(PersonalApiToken)
        .where(PersonalApiToken.id == token_id, PersonalApiToken.user_id == user.id)
        .with_for_update()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Token not found")
    if item.revoked_at is None:
        item.revoked_at = now()
        await db.commit()
    return Response(status_code=204)


def _safe_mail(settings, recipient: str, token: str) -> None:
    try:
        send_reset_email(settings, recipient, token)
    except Exception:
        # Do not put credentials, recipient addresses, message bodies, or tokens in logs.
        logger.error("Password reset email delivery failed")


@router.post("/auth/forgot-password", status_code=202)
async def forgot_password(
    payload: ForgotPasswordIn,
    request: Request,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_session),
):
    settings = request.app.state.settings
    if not settings.smtp_host or not settings.smtp_sender or not settings.public_web_origin:
        raise HTTPException(status_code=503, detail="Password reset email is not configured")
    user = await db.scalar(
        select(User).where(User.email == str(payload.email).casefold(), User.is_active.is_(True))
    )
    if user:
        secret = secrets.token_urlsafe(40)
        db.add(
            PasswordReset(
                id=uuid.uuid4(),
                user_id=user.id,
                token_hash=token_hash(secret),
                expires_at=now() + timedelta(minutes=settings.reset_minutes),
            )
        )
        await db.commit()
        background.add_task(_safe_mail, settings, user.email, secret)
    # The same response is returned for existing and unknown addresses.
    return {"message": "If the account exists, a reset link will be sent."}


@router.post("/auth/reset-password", status_code=204)
async def reset_password(
    payload: ResetPasswordIn,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_session),
):
    digest = token_hash(payload.token)
    record = await db.scalar(select(PasswordReset).where(PasswordReset.token_hash == digest))
    if not record:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")
    # Lock the account first so different valid reset links cannot race each other.
    try:
        user = await _lock_user(db, record.user_id)
    except HTTPException as exc:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link") from exc
    record = await db.scalar(
        select(PasswordReset)
        .where(PasswordReset.token_hash == digest)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if not record or record.used_at is not None or _aware(record.expires_at) <= now():
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")
    user.password_hash = await hash_password(payload.new_password)
    await _revoke_credentials(db, user.id)
    await db.commit()
    clear_session_cookies(response, request.app.state.settings)
    response.status_code = 204
    return response
