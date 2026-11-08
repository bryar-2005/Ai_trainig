"""FastAPI application factory for authenticated subtitle workspaces."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from api.config import Settings, get_settings
from api.db.session import create_engine, create_session_factory
from api.rate_limit import RateLimiter
from api.routes import account, auth, cues, export, glossary, jobs, models, projects
from api.security import ACCESS_COOKIE, REFRESH_COOKIE, check_csrf
from api.storage import create_storage

logger = logging.getLogger("sorani.http")


class BodyTooLarge(Exception):
    """Stop consuming an oversized streaming request."""


class BodyLimitMiddleware:
    def __init__(self, app, settings: Settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        upload = scope.get("method") == "POST" and scope.get("path", "").endswith("/upload")
        inspect = scope.get("method") == "POST" and scope.get("path") == "/api/subtitles/inspect"
        limit = (
            self.settings.max_subtitle_bytes + self.settings.max_video_bytes + 1024 * 1024
            if upload
            else self.settings.max_subtitle_bytes + 1024 * 1024
            if inspect
            else 2 * 1024 * 1024
        )
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = -1
        if declared < 0 or declared > limit:
            await JSONResponse({"detail": "Request body is too large or invalid"}, status_code=413)(
                scope, receive, send
            )
            return
        count, started, exceeded, rejection_sent = 0, False, False, False

        async def bounded_receive():
            nonlocal count, exceeded
            message = await receive()
            if message["type"] == "http.request":
                count += len(message.get("body", b""))
                if count > limit:
                    exceeded = True
                    raise BodyTooLarge()
            return message

        async def reject():
            nonlocal started, rejection_sent
            rejection_sent, started = True, True
            await JSONResponse({"detail": "Request body is too large"}, status_code=413)(
                scope, receive, send
            )

        async def capture_send(message):
            nonlocal started
            # FastAPI's form/JSON parser can catch receive exceptions and try to
            # emit a generic 400. Replace it with the correct bounded-body result.
            if exceeded:
                if started and not rejection_sent:
                    raise BodyTooLarge()
                if not rejection_sent:
                    await reject()
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, capture_send)
        except BodyTooLarge:
            if started and not rejection_sent:
                raise
            if not rejection_sent:
                await reject()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.engine = create_engine(settings.database_url)
        app.state.sessions = create_session_factory(app.state.engine)
        app.state.storage = create_storage(settings)
        app.state.limiter = RateLimiter(settings)
        try:
            yield
        finally:
            await app.state.limiter.close()
            await app.state.engine.dispose()

    app = FastAPI(title="Sorani Subtitle Workspace API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, settings=settings)
    registry = CollectorRegistry()
    request_count = Counter(
        "sorani_http_requests_total",
        "HTTP requests",
        ["method", "route", "status"],
        registry=registry,
    )
    latency = Histogram(
        "sorani_http_request_seconds",
        "HTTP request duration",
        ["method", "route"],
        registry=registry,
    )

    @app.middleware("http")
    async def safeguards(request: Request, call_next):
        started = time.monotonic()
        response = None
        path, method = request.url.path, request.method
        try:
            if path.startswith("/api/"):
                identity = request.client.host if request.client else "unknown"
                # Limit credential attempts independently from routine session
                # reads. Counting GET /auth/me as a login attempt locks normal
                # users out while navigating, especially behind a shared proxy.
                credential_paths = {
                    "/api/auth/register",
                    "/api/auth/login",
                    "/api/auth/refresh",
                    "/api/auth/forgot-password",
                    "/api/auth/reset-password",
                    "/api/account/password",
                    "/api/account/tokens",
                }
                category = "auth" if method == "POST" and path in credential_paths else "api"
                limit = settings.auth_rate_limit if category == "auth" else settings.api_rate_limit
                try:
                    allowed = await request.app.state.limiter.check(identity, category, limit)
                except RedisError:
                    response = JSONResponse(
                        {"detail": "Rate-limit service unavailable"}, status_code=503
                    )
                    return response
                if not allowed:
                    response = JSONResponse(
                        {"detail": "Too many requests"},
                        status_code=429,
                        headers={"Retry-After": str(settings.rate_window_seconds)},
                    )
                    return response
                if method not in {"GET", "HEAD", "OPTIONS"}:
                    origin = request.headers.get("origin")
                    same_origin = f"{request.url.scheme}://{request.url.netloc}"
                    if origin and origin.rstrip("/") not in {
                        *settings.allowed_origins,
                        same_origin,
                    }:
                        response = JSONResponse(
                            {"detail": "Origin is not allowed"}, status_code=403
                        )
                        return response
                    authenticated_cookie = request.cookies.get(
                        ACCESS_COOKIE
                    ) or request.cookies.get(REFRESH_COOKIE)
                    if (
                        authenticated_cookie
                        and path not in {"/api/auth/register", "/api/auth/login"}
                        and not check_csrf(request)
                    ):
                        response = JSONResponse({"detail": "CSRF token required"}, status_code=403)
                        return response
            response = await call_next(request)
            return response
        finally:
            route = getattr(request.scope.get("route"), "path", "unmatched")
            status = str(response.status_code or 500 if response is not None else 500)
            request_count.labels(method, route, status).inc()
            latency.labels(method, route).observe(time.monotonic() - started)
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "time": datetime.now(UTC).isoformat(),
                        "method": method,
                        "route": route,
                        "status": int(status),
                        "duration_ms": round(1000 * (time.monotonic() - started), 2),
                    }
                )
            )
            if response is not None:
                response.headers.setdefault("X-Content-Type-Options", "nosniff")
                response.headers.setdefault("Referrer-Policy", "no-referrer")
                if path.startswith("/api/"):
                    response.headers.setdefault("Cache-Control", "private, no-store")

    # Install CORS outside the safeguards so allowed origins can read errors too.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-CSRF-Token", "Last-Event-ID", "Range"],
        expose_headers=["Content-Disposition", "Content-Range", "X-Subtitle-Origin"],
    )

    @app.get("/health", tags=["operations"])
    async def health(request: Request):
        database_ok = False
        try:
            async with request.app.state.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
                database_ok = True
        except SQLAlchemyError:
            database_ok = False
        redis_ok = await request.app.state.limiter.healthy()
        return JSONResponse(
            {
                "status": "ok" if database_ok and redis_ok else "degraded",
                "database": database_ok,
                "redis": redis_ok,
            },
            status_code=200 if database_ok and redis_ok else 503,
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    for router in (
        auth.router,
        account.router,
        projects.router,
        cues.router,
        glossary.router,
        export.router,
        jobs.router,
        models.router,
    ):
        app.include_router(router)
    return app
