"""Validated account settings and credential-management inputs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreferencesIn(Input):
    default_target_cps: float = Field(default=17, gt=0, le=100, allow_inf_nan=False)
    default_max_chars: int = Field(default=42, ge=1, le=200)
    default_export_encoding: Literal["utf-8", "utf-8-sig"] = "utf-8"
    nightly_feedback_export: bool = False


class ProfileIn(Input):
    display_name: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def not_blank(self):
        self.display_name = self.display_name.strip()
        if not self.display_name:
            raise ValueError("Display name must not be blank")
        return self


class PasswordIn(Input):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=12, max_length=128)


class ApiTokenIn(Input):
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=128)
    expires_days: int = Field(default=90, ge=1, le=365)

    @model_validator(mode="after")
    def not_blank(self):
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("Token name must not be blank")
        return self


class ForgotPasswordIn(Input):
    email: EmailStr


class ResetPasswordIn(Input):
    token: str = Field(min_length=32, max_length=256, pattern=r"^[A-Za-z0-9_-]+$")
    new_password: str = Field(min_length=12, max_length=128)
