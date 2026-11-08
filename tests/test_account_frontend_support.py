"""Real account, editor deep-link, retry, and styled-export integration tests."""

from __future__ import annotations

from datetime import timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")
pytest.importorskip("celery")

import pysubs2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import update  # noqa: E402

from api.db.models import PasswordReset, now  # noqa: E402
from api.main import create_app  # noqa: E402
from api.security import token_hash  # noqa: E402
from tests.test_api import PASSWORD, Backend  # noqa: E402
from tests.test_api import api as shared_api  # noqa: E402

api = shared_api


def test_profile_preferences_and_personal_tokens(api: Backend):
    user = api.register()
    client = api.client
    preflight = client.options(
        "/api/account/preferences",
        headers={
            "Origin": "http://client.example",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "content-type,x-csrf-token",
        },
    )
    assert preflight.status_code == 200
    assert "PUT" in preflight.headers["access-control-allow-methods"]
    profile = client.patch("/api/account/profile", json={"display_name": "  Kurdish Reviewer  "})
    assert profile.status_code == 200 and profile.json()["display_name"] == "Kurdish Reviewer"
    assert client.patch("/api/account/profile", json={"display_name": "  "}).status_code == 422
    preferences = {
        "default_target_cps": 19,
        "default_max_chars": 38,
        "default_export_encoding": "utf-8-sig",
        "nightly_feedback_export": False,
    }
    assert (
        client.put("/api/account/preferences", json=preferences).json()["preferences"]
        == preferences
    )
    assert (
        client.put(
            "/api/account/preferences", json={**preferences, "active_model_en-ckb": "fake"}
        ).status_code
        == 422
    )
    assert (
        client.post("/api/account/tokens", json={"name": "CLI", "password": "wrong"}).status_code
        == 403
    )
    created = client.post(
        "/api/account/tokens", json={"name": "CLI", "password": PASSWORD, "expires_days": 7}
    )
    assert created.status_code == 201, created.text
    secret, token_id = created.json()["token"], created.json()["id"]
    assert secret.startswith("sst_") and len(secret) > 40
    listing = client.get("/api/account/tokens").json()
    assert secret not in str(listing) and "token_hash" not in str(listing)
    assert listing["items"][0]["expired"] is False
    # A bearer-only client needs no cookies or CSRF and acts as its owner.
    with TestClient(create_app(api.settings)) as bearer:
        bearer.headers["Authorization"] = f"Bearer {secret}"
        assert bearer.get("/api/auth/me").json()["id"] == user["id"]
        assert (
            bearer.post(
                "/api/projects", json={"name": "Token project", "direction": "en-ckb"}
            ).status_code
            == 201
        )
        assert client.delete(f"/api/account/tokens/{token_id}").status_code == 204
        assert bearer.get("/api/auth/me").status_code == 401
    second = client.post(
        "/api/account/tokens", json={"name": "Second", "password": PASSWORD}
    ).json()
    new_password = "a-different-long-password-456"
    assert (
        client.post(
            "/api/account/password",
            json={"current_password": PASSWORD, "new_password": new_password},
        ).status_code
        == 204
    )
    assert client.get("/api/auth/me").status_code == 401
    with TestClient(create_app(api.settings)) as bearer:
        assert (
            bearer.get(
                "/api/auth/me", headers={"Authorization": f"Bearer {second['token']}"}
            ).status_code
            == 401
        )
    assert (
        client.post(
            "/api/auth/login", json={"email": user["email"], "password": PASSWORD}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/auth/login", json={"email": user["email"], "password": new_password}
        ).status_code
        == 200
    )
    api.csrf()
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").status_code == 401


def test_password_reset_expiry_replay_and_session_revocation(api: Backend, monkeypatch):
    user = api.register()
    client = api.client
    assert (
        client.post("/api/auth/forgot-password", json={"email": user["email"]}).status_code == 503
    )
    api.settings.smtp_host = "smtp.example.test"
    api.settings.smtp_sender = "studio@example.test"
    api.settings.public_web_origin = "https://studio.example.test"
    delivered: list[str] = []
    monkeypatch.setattr(
        "api.routes.account.send_reset_email", lambda _s, _r, token: delivered.append(token)
    )
    other = client.post("/api/auth/forgot-password", json={"email": "nobody@example.com"})
    known = client.post("/api/auth/forgot-password", json={"email": user["email"]})
    assert other.status_code == known.status_code == 202 and other.json() == known.json()
    assert len(delivered) == 1
    expired_token = delivered[0]
    api.execute(
        update(PasswordReset)
        .where(PasswordReset.token_hash == token_hash(expired_token))
        .values(expires_at=now() - timedelta(seconds=1))
    )
    assert (
        client.post(
            "/api/auth/reset-password",
            json={"token": expired_token, "new_password": "new-strong-password-123"},
        ).status_code
        == 400
    )
    assert (
        client.post("/api/auth/forgot-password", json={"email": user["email"]}).status_code == 202
    )
    valid = delivered[-1]
    assert (
        client.post(
            "/api/auth/reset-password",
            json={"token": valid, "new_password": "new-strong-password-123"},
        ).status_code
        == 204
    )
    assert client.get("/api/auth/me").status_code == 401
    assert (
        client.post(
            "/api/auth/reset-password",
            json={"token": valid, "new_password": "another-long-password-123"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/auth/login", json={"email": user["email"], "password": "new-strong-password-123"}
        ).status_code
        == 200
    )


def test_deep_link_glossary_recheck_and_styled_ass(api: Backend):
    api.register()
    project = api.project()
    api.upload(project)
    client = api.client
    inspect = client.post(
        "/api/subtitles/inspect",
        files={"subtitle": ("preview.srt", b"1\n00:00:01,000 --> 00:00:03,000\nOriginal text\n")},
    )
    assert inspect.status_code == 200
    assert inspect.json() == {"cue_count": 1, "duration_ms": 3000}
    assert not list(api.settings.storage_root.glob(".inspect-*"))
    cues = api.cues(project)
    assert (
        client.get(f"/api/projects/{project}/cues/locate/{cues[2]['id']}?page_size=1").json()[
            "page"
        ]
        == 3
    )
    context = client.get(f"/api/cues/{cues[1]['id']}/context").json()
    assert [cue["id"] for cue in context["previous"]] == [cues[0]["id"]]
    assert [cue["id"] for cue in context["next"]] == [cues[2]["id"]]
    before = [(cue["start_ms"], cue["end_ms"], cue["revision"]) for cue in cues]
    assert client.post(f"/api/projects/{project}/glossary/recheck").json()["checked"] == 3
    after = [(cue["start_ms"], cue["end_ms"], cue["revision"]) for cue in api.cues(project)]
    assert before == after
    edited = client.patch(
        f"/api/cues/{cues[0]['id']}", json={"expected_revision": 0, "human_text": "سڵاو"}
    )
    assert edited.status_code == 200
    stats = client.get(f"/api/projects/{project}/stats").json()
    assert stats["translated_count"] == 1
    assert sum(stats["confidence_histogram"]) + stats["unscored_count"] == 3
    activity = client.get(f"/api/projects/{project}/activity").json()["items"]
    assert activity[0]["cue_id"] == cues[0]["id"] and activity[0]["action"] == "edit"
    listing = client.get("/api/projects?sort=confidence_desc").json()["items"]
    assert listing[0]["id"] == project and listing[0]["average_confidence"] is not None
    styled = client.get(
        f"/api/projects/{project}/export",
        params={
            "format": "ass",
            "ass_font": "Vazirmatn",
            "ass_font_size": 34,
            "ass_color": "#A1B2C3",
            "ass_margin": 42,
        },
    )
    assert styled.status_code == 200, styled.text
    style = pysubs2.SSAFile.from_string(styled.text, format_="ass").styles["Default"]
    assert style.fontname == "Vazirmatn" and style.fontsize == 34 and style.marginv == 42
    assert (style.primarycolor.r, style.primarycolor.g, style.primarycolor.b) == (161, 178, 195)
    assert (
        client.get(
            f"/api/projects/{project}/export", params={"format": "ass", "ass_font": "Bad\n[Events]"}
        ).status_code
        == 422
    )
    client.cookies.clear()
    api.register("other@example.com")
    assert client.get(f"/api/projects/{project}/cues/locate/{cues[2]['id']}").status_code == 404
    assert client.post(f"/api/projects/{project}/glossary/recheck").status_code == 404


def test_cancelled_job_retry_is_new_and_revalidated(api: Backend):
    api.register()
    api.admin()
    project = api.project()
    api.upload(project)
    client = api.client
    model = client.post(
        "/api/models",
        json={
            "name": "retry-test",
            "direction": "en-ckb",
            "kind": "base",
            "source": "fixture/not-cached",
        },
    ).json()["id"]
    job = client.post(
        f"/api/projects/{project}/translate", json={"model_version_id": model}
    ).json()["job_id"]
    assert client.post(f"/api/jobs/{job}/retry").status_code == 409
    assert client.post(f"/api/jobs/{job}/cancel").status_code == 200
    retried = client.post(f"/api/jobs/{job}/retry")
    assert retried.status_code == 202 and retried.json()["job_id"] != job
    assert client.get(f"/api/jobs/{retried.json()['job_id']}").json()["status"] == "queued"
    assert client.post(f"/api/jobs/{job}/retry").status_code == 409
