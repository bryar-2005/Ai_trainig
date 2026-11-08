"""Argon2 passwords, short-lived JWTs, and rotating server-side refresh sessions."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from api.config import Settings
from api.db.models import PersonalApiToken, RefreshSession, User, now
from api.db.session import get_session

ACCESS_COOKIE = "sorani_access"
REFRESH_COOKIE = "sorani_refresh"
CSRF_COOKIE = "sorani_csrf"
ISSUER = "sorani-subtitles"
AUDIENCE = "sorani-web"
_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
# A genuine Argon2 hash equalizes unknown-email and incorrect-password work.
_dummy_hash = _hasher.hash(secrets.token_urlsafe(32))


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


async def hash_password(password: str) -> str:
    return await run_in_threadpool(_hasher.hash, password)


async def verify_password(encoded: str | None, password: str) -> bool:
    def verify() -> bool:
        try:
            return _hasher.verify(encoded or _dummy_hash, password) and encoded is not None
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False

    return await run_in_threadpool(verify)


def _token(
    settings: Settings, user_id: uuid.UUID, session_id: uuid.UUID, kind: str, jti: str
) -> str:
    current = now()
    expires = current + (
        timedelta(minutes=settings.access_minutes)
        if kind == "access"
        else timedelta(days=settings.refresh_days)
    )
    return jwt.encode(
        {
            "sub": str(user_id),
            "sid": str(session_id),
            "typ": kind,
            "jti": jti,
            "iat": current,
            "exp": expires,
            "iss": ISSUER,
            "aud": AUDIENCE,
        },
        settings.secret_key.get_secret_value(),
        algorithm="HS256",
    )


def decode_token(settings: Settings, value: str, kind: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            value,
            settings.secret_key.get_secret_value(),
            algorithms=["HS256"],
            issuer=ISSUER,
            audience=AUDIENCE,
            options={"require": ["sub", "sid", "typ", "jti", "iat", "exp", "iss", "aud"]},
        )
        if payload["typ"] != kind:
            raise ValueError("Wrong token type")
        uuid.UUID(payload["sub"])
        uuid.UUID(payload["sid"])
        if not isinstance(payload["jti"], str):
            raise ValueError("Invalid token ID")
        return payload
    except (jwt.PyJWTError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session"
        ) from exc


def token_hash(jti: str) -> str:
    return hashlib.sha256(jti.encode("utf-8")).hexdigest()


def set_session_cookies(
    response: Response, settings: Settings, user_id: uuid.UUID, session_id: uuid.UUID, jti: str
) -> None:
    common = {"secure": settings.cookie_secure, "samesite": "lax", "path": "/"}
    response.set_cookie(
        ACCESS_COOKIE,
        _token(settings, user_id, session_id, "access", secrets.token_urlsafe(32)),
        httponly=True,
        max_age=settings.access_minutes * 60,
        **common,
    )
    response.set_cookie(
        REFRESH_COOKIE,
        _token(settings, user_id, session_id, "refresh", jti),
        httponly=True,
        max_age=settings.refresh_days * 86400,
        **common,
    )
    nonce = secrets.token_urlsafe(32)
    signature = hmac.new(
        settings.secret_key.get_secret_value().encode(),
        f"csrf:{session_id}:{nonce}".encode(),
        hashlib.sha256,
    ).hexdigest()
    response.set_cookie(
        CSRF_COOKIE,
        f"{nonce}.{signature}",
        httponly=False,
        max_age=settings.refresh_days * 86400,
        **common,
    )


def clear_session_cookies(response: Response, settings: Settings) -> None:
    for name in (ACCESS_COOKIE, REFRESH_COOKIE, CSRF_COOKIE):
        response.delete_cookie(name, path="/", secure=settings.cookie_secure, samesite="lax")


def check_csrf(request: Request) -> bool:
    cookie = request.cookies.get(CSRF_COOKIE, "")
    header = request.headers.get("x-csrf-token", "")
    if (
        not cookie
        or not header
        or len(cookie) > 256
        or len(header) > 256
        or not cookie.isascii()
        or not header.isascii()
        or not hmac.compare_digest(cookie, header)
    ):
        return False
    try:
        nonce, signature = cookie.split(".", 1)
    except ValueError:
        return False
    settings = request.app.state.settings
    for name, kind in ((ACCESS_COOKIE, "access"), (REFRESH_COOKIE, "refresh")):
        value = request.cookies.get(name)
        if not value:
            continue
        try:
            session_id = decode_token(settings, value, kind)["sid"]
        except HTTPException:
            continue
        expected = hmac.new(
            settings.secret_key.get_secret_value().encode(),
            f"csrf:{session_id}:{nonce}".encode(),
            hashlib.sha256,
        ).hexdigest()
        if hmac.compare_digest(expected, signature):
            return True
    return False


async def current_user(request: Request, db: AsyncSession = Depends(get_session)) -> User:
    authorization = request.headers.get("authorization")
    if authorization is not None:
        scheme, separator, secret = authorization.partition(" ")
        if (
            not separator
            or scheme.casefold() != "bearer"
            or not secret.startswith("sst_")
            or len(secret) > 256
        ):
            raise HTTPException(status_code=401, detail="Invalid API token")
        row = (
            await db.execute(
                select(User, PersonalApiToken)
                .join(PersonalApiToken, PersonalApiToken.user_id == User.id)
                .where(PersonalApiToken.token_hash == token_hash(secret))
            )
        ).one_or_none()
        if (
            row is None
            or not row.User.is_active
            or row.PersonalApiToken.revoked_at is not None
            or _aware(row.PersonalApiToken.expires_at) <= now()
        ):
            raise HTTPException(status_code=401, detail="Invalid or expired API token")
        return row.User
    value = request.cookies.get(ACCESS_COOKIE)
    if not value:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = decode_token(request.app.state.settings, value, "access")
    session_id, user_id = uuid.UUID(payload["sid"]), uuid.UUID(payload["sub"])
    result = await db.execute(
        select(User, RefreshSession)
        .join(RefreshSession, RefreshSession.user_id == User.id)
        .where(User.id == user_id, RefreshSession.id == session_id)
    )
    row = result.one_or_none()
    if (
        row is None
        or not row.User.is_active
        or row.RefreshSession.revoked_at is not None
        or _aware(row.RefreshSession.expires_at) <= now()
    ):
        raise HTTPException(status_code=401, detail="Session expired or revoked")
    return row.User


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return user
