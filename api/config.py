"""Explicit service configuration; no usable default signing key is shipped."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import EmailStr, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SORANI_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    environment: Literal["development", "production", "test"] = "development"
    database_url: str = "postgresql+asyncpg://sorani:sorani@localhost:5432/sorani"
    redis_url: str = "redis://localhost:6379/0"
    broker_url: str = "redis://localhost:6379/1"
    secret_key: SecretStr
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    cookie_secure: bool = True
    access_minutes: int = Field(default=15, ge=1, le=60)
    refresh_days: int = Field(default=7, ge=1, le=90)
    # Password reset is disabled until an SMTP host, sender, and public web origin are configured.
    smtp_host: str | None = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None
    smtp_starttls: bool = True
    smtp_sender: EmailStr | None = None
    public_web_origin: str | None = None
    reset_minutes: int = Field(default=30, ge=5, le=120)
    storage_backend: Literal["local", "s3"] = "local"
    storage_root: Path = Path("data/private-storage")
    model_root: Path = Path("outputs/models")
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    max_subtitle_bytes: int = Field(default=5 * 1024 * 1024, ge=1, le=50 * 1024 * 1024)
    max_video_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=1, le=4 * 1024**3)
    auth_rate_limit: int = Field(default=20, ge=1)
    api_rate_limit: int = Field(default=300, ge=1)
    rate_window_seconds: int = Field(default=60, ge=1, le=3600)
    local_files_only: bool = True
    inference_device: Literal["cpu", "cuda"] = "cpu"
    allow_smoke_models: bool = False
    job_lease_seconds: int = Field(default=120, ge=30, le=3600)
    job_heartbeat_seconds: int = Field(default=10, ge=2, le=300)
    # Scheduled exports are disabled until both operator and user opt in.
    nightly_feedback_export_enabled: bool = False
    feedback_export_retention_days: int = Field(default=7, ge=1, le=30)
    # Tests may execute Celery tasks synchronously or suppress dispatch explicitly.
    # Neither mode is permitted in production.
    task_mode: Literal["celery", "eager", "manual"] = "celery"

    @field_validator("secret_key")
    @classmethod
    def strong_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("SORANI_SECRET_KEY must contain at least 32 bytes")
        return value

    @field_validator("database_url")
    @classmethod
    def async_database(cls, value: str) -> str:
        if not value.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
            raise ValueError("Use an async PostgreSQL URL (or SQLite only in tests)")
        return value

    @field_validator("allowed_origins")
    @classmethod
    def explicit_origins(cls, value: list[str]) -> list[str]:
        origins = [item.rstrip("/") for item in value]
        if not origins:
            raise ValueError("allowed_origins must not be empty")
        for origin in origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or "*" in origin
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("allowed_origins must contain explicit HTTP(S) origins")
            _ = parsed.port  # Validate the optional port as well.
        return origins

    @field_validator("public_web_origin")
    @classmethod
    def valid_web_origin(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.rstrip("/")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Public web origin must be an explicit HTTP(S) origin")
        _ = parsed.port
        return value

    @model_validator(mode="after")
    def production_safety(self) -> Settings:
        mail_settings = (self.smtp_host, self.smtp_sender, self.public_web_origin)
        if any(mail_settings) and not all(mail_settings):
            raise ValueError("SMTP host, sender, and public web origin are required together")
        if bool(self.smtp_username) != bool(self.smtp_password):
            raise ValueError("SMTP username and password must be supplied together")
        if self.smtp_username and not self.smtp_starttls:
            raise ValueError("SMTP authentication requires STARTTLS")
        if self.environment == "production" and self.smtp_host:
            if not self.smtp_starttls or not self.public_web_origin.startswith("https://"):
                raise ValueError("Production password reset requires HTTPS and SMTP STARTTLS")
        if self.storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("SORANI_S3_BUCKET is required for S3 storage")
        if self.job_heartbeat_seconds * 2 >= self.job_lease_seconds:
            raise ValueError("Job heartbeat must be less than half the lease duration")
        if self.environment != "test" and self.database_url.startswith("sqlite"):
            raise ValueError("SQLite is supported only for tests; use PostgreSQL")
        if self.environment != "test" and (self.allow_smoke_models or self.task_mode == "manual"):
            raise ValueError("Smoke models and manual task dispatch are restricted to tests")
        if self.environment == "production":
            if self.database_url == "postgresql+asyncpg://sorani:sorani@localhost:5432/sorani":
                raise ValueError("Production requires an explicitly configured database URL")
            if not self.cookie_secure or self.task_mode != "celery":
                raise ValueError("Production requires secure cookies and a Celery worker")
            if not self.allowed_origins or any(
                x.startswith("http://") for x in self.allowed_origins
            ):
                raise ValueError("Production origins must be explicit HTTPS origins")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
