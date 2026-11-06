"""Backend integration tests. Optional URLs point only to disposable test services."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")
pytest.importorskip("asyncpg")
pytest.importorskip("celery")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import update  # noqa: E402

from api.config import Settings  # noqa: E402
from api.db.models import Base, User  # noqa: E402
from api.db.session import create_engine, create_session_factory  # noqa: E402
from api.main import create_app  # noqa: E402
from api.security import CSRF_COOKIE, REFRESH_COOKIE  # noqa: E402

PASSWORD = "test-only-long-password-123"
SUBTITLES = (
    b"1\n00:00:01,000 --> 00:00:03,000\nJOHN: Please find the\n\n"
    b"2\n00:00:03,000 --> 00:00:06,000\nobject and bring it back.\n\n"
    b"3\n00:00:07,000 --> 00:00:09,000\nWe should go home.\n"
)
VIDEO = b"\x00\x00\x00\x18ftypisom" + b"original-fixture-video"


@dataclass
class Backend:
    client: TestClient
    settings: Settings

    def execute(self, statement):
        async def run():
            engine = create_engine(self.settings.database_url)
            try:
                async with create_session_factory(engine)() as session:
                    result = await session.execute(statement)
                    await session.commit()
                    return result
            finally:
                await engine.dispose()

        return asyncio.run(run())

    def register(self, email: str = "owner@example.com") -> dict:
        response = self.client.post(
            "/api/auth/register",
            json={"email": email, "display_name": "Test reviewer", "password": PASSWORD},
        )
        assert response.status_code == 201, response.text
        self.csrf()
        return response.json()

    def csrf(self) -> None:
        self.client.headers["X-CSRF-Token"] = self.client.cookies[CSRF_COOKIE]

    def admin(self, email: str = "owner@example.com") -> None:
        self.execute(update(User).where(User.email == email).values(role="admin"))

    def project(self, name: str = "Test film", model_id: str | None = None) -> str:
        response = self.client.post(
            "/api/projects",
            json={"name": name, "direction": "en-ckb", "model_version_id": model_id},
        )
        assert response.status_code == 201, response.text
        return response.json()["id"]

    def upload(self, project_id: str, *, video: bool = False, data: bytes = SUBTITLES) -> dict:
        files = {"subtitle": ("../fixture.srt", data, "text/plain")}
        if video:
            files["video"] = ("fixture.mp4", VIDEO, "video/mp4")
        response = self.client.post(f"/api/projects/{project_id}/upload", files=files)
        assert response.status_code == 201, response.text
        return response.json()

    def cues(self, project_id: str) -> list[dict]:
        response = self.client.get(f"/api/projects/{project_id}/cues")
        assert response.status_code == 200, response.text
        return response.json()["items"]

    def edit_and_approve(self, cue: dict, text: str = "ئێمە دەگەڕێینەوە ماڵەوە.") -> dict:
        response = self.client.patch(
            f"/api/cues/{cue['id']}",
            json={"expected_revision": cue["revision"], "human_text": text},
        )
        assert response.status_code == 200, response.text
        edited = response.json()
        response = self.client.post(
            f"/api/cues/{cue['id']}/approve", json={"expected_revision": edited["revision"]}
        )
        assert response.status_code == 200, response.text
        return response.json()


@pytest.fixture
def api(tmp_path: Path):
    database_url = os.environ.get(
        "SORANI_TEST_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'test.sqlite'}"
    )
    settings = Settings(
        environment="test",
        secret_key="not-a-production-secret-used-only-for-backend-tests",
        database_url=database_url,
        redis_url=os.environ.get("SORANI_TEST_REDIS_URL", "memory://"),
        broker_url=os.environ.get("SORANI_TEST_BROKER_URL", "redis://localhost:6379/15"),
        allowed_origins=["http://client.example"],
        cookie_secure=False,
        task_mode="manual",
        storage_root=tmp_path / "private",
        model_root=tmp_path / "models",
        auth_rate_limit=10000,
        api_rate_limit=10000,
        max_subtitle_bytes=4096,
        max_video_bytes=4096,
    )

    async def reset():
        engine = create_engine(database_url)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.drop_all)
                await connection.run_sync(Base.metadata.create_all)
        finally:
            await engine.dispose()

    asyncio.run(reset())
    with TestClient(create_app(settings), raise_server_exceptions=True) as client:
        yield Backend(client, settings)


def test_auth_csrf_origin_rotation_and_replay(api: Backend):
    client = api.client
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/auth/me").status_code == 401
    user = api.register("Alice@example.com")
    assert user["role"] == "user" and user["email"] == "alice@example.com"
    assert (
        "HttpOnly"
        in client.post(
            "/api/auth/login", json={"email": "alice@example.com", "password": PASSWORD}
        ).headers["set-cookie"]
    )
    api.csrf()
    assert (
        client.post("/api/projects", json={"name": "A", "direction": "en-ckb"}).status_code == 201
    )
    del client.headers["X-CSRF-Token"]
    forbidden = client.post(
        "/api/projects",
        json={"name": "A", "direction": "en-ckb"},
        headers={"Origin": "http://client.example"},
    )
    assert forbidden.status_code == 403
    assert forbidden.headers["access-control-allow-origin"] == "http://client.example"
    assert forbidden.headers["x-content-type-options"] == "nosniff"
    api.csrf()
    assert (
        client.post(
            "/api/projects",
            json={"name": "A", "direction": "en-ckb"},
            headers={"Origin": "https://attacker.example"},
        ).status_code
        == 403
    )
    old_refresh, old_csrf = client.cookies[REFRESH_COOKIE], client.cookies[CSRF_COOKIE]
    rotated = client.post("/api/auth/refresh")
    assert rotated.status_code == 200
    assert client.cookies[REFRESH_COOKIE] != old_refresh
    new_refresh = client.cookies[REFRESH_COOKIE]
    client.cookies.set(REFRESH_COOKIE, old_refresh, domain="testserver.local", path="/")
    client.cookies.set(CSRF_COOKIE, old_csrf, domain="testserver.local", path="/")
    api.csrf()
    assert client.post("/api/auth/refresh").status_code == 401
    client.cookies.set(REFRESH_COOKIE, new_refresh, domain="testserver.local", path="/")
    client.cookies.set(CSRF_COOKIE, old_csrf, domain="testserver.local", path="/")
    api.csrf()
    assert client.post("/api/auth/refresh").status_code == 401
    assert client.get("/api/auth/me").status_code == 401


def test_admin_bootstrap_and_signed_csrf(api: Backend, monkeypatch):
    from api.db.seed import create_admin

    monkeypatch.setattr("api.db.seed.get_settings", lambda: api.settings)
    asyncio.run(create_admin("admin@example.com", "Administrator", PASSWORD))
    with pytest.raises(ValueError, match="already exists"):
        asyncio.run(create_admin("admin@example.com", "Administrator", PASSWORD))
    login = api.client.post(
        "/api/auth/login", json={"email": "admin@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200 and login.json()["role"] == "admin"
    api.client.cookies.set(CSRF_COOKIE, "forged.value", domain="testserver.local", path="/")
    api.client.headers["X-CSRF-Token"] = "forged.value"
    response = api.client.post("/api/projects", json={"name": "A", "direction": "en-ckb"})
    assert response.status_code == 403


def test_owner_isolation_and_input_validation(api: Backend):
    api.register()
    client = api.client
    project = api.project()
    api.upload(project)
    cue = api.cues(project)[0]
    client.cookies.clear()
    api.register("other@example.com")
    assert client.get(f"/api/projects/{project}").status_code == 404
    assert client.get(f"/api/projects/{project}/cues").status_code == 404
    assert (
        client.patch(
            f"/api/cues/{cue['id']}", json={"expected_revision": 0, "human_text": "stolen"}
        ).status_code
        == 404
    )
    assert client.get("/api/projects").json()["total"] == 0
    assert (
        client.post(
            "/api/projects", json={"name": "", "direction": "en-ckb", "owner_id": user_id()}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/models",
            json={"name": "x", "direction": "en-ckb", "kind": "base", "source": "/tmp"},
        ).status_code
        == 403
    )
    assert client.get(f"/api/projects/{project}/export?format=json").status_code == 404


def user_id() -> str:
    return str(uuid.uuid4())


def test_upload_limits_chunking_private_media_and_range(api: Backend):
    api.register()
    client = api.client
    project = api.project()
    url = f"/api/projects/{project}/upload"
    assert client.post(url, files={"subtitle": ("large.srt", b"x" * 4097)}).status_code == 413

    def multipart_chunks():
        yield (
            b"--test-boundary\r\nContent-Disposition: form-data; "
            b'name="subtitle"; filename="x.srt"\r\n\r\n'
        )
        for _ in range(6):
            yield b"x" * 1000
        yield b"\r\n--test-boundary--\r\n"

    streamed = client.post(
        url,
        content=multipart_chunks(),
        headers={"Content-Type": "multipart/form-data; boundary=test-boundary"},
    )
    assert streamed.status_code == 413
    assert client.post(url, files={"subtitle": ("bad.exe", b"text")}).status_code == 422
    assert client.post(url, files={"subtitle": ("empty.srt", b"")}).status_code == 422
    assert (
        client.post(
            url, files={"subtitle": ("s.srt", SUBTITLES), "video": ("bad.mp4", b"no")}
        ).status_code
        == 422
    )
    uploaded = api.upload(project, video=True)
    assert uploaded["media"][0]["original_name"] == "fixture.srt"
    assert uploaded["cue_count"] == 3
    media = next(item for item in uploaded["media"] if item["kind"] == "video")
    video_url = f"/api/projects/{project}/media/{media['id']}"
    assert client.get(video_url).content == VIDEO
    response = client.get(video_url, headers={"Range": "bytes=4-11"})
    assert response.status_code == 206 and response.content == VIDEO[4:12]
    assert response.headers["content-range"] == f"bytes 4-11/{len(VIDEO)}"
    assert client.get(video_url, headers={"Range": "bytes=999-"}).status_code == 416
    assert client.post(url, files={"subtitle": ("again.srt", SUBTITLES)}).status_code == 409

    def chunks():
        for _ in range(230):
            yield b"x" * 10000

    response = client.post(
        "/api/auth/login", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Request body is too large"
    assert list(api.settings.storage_root.rglob("*.mp4"))
    assert client.delete(f"/api/projects/{project}").status_code == 204
    assert not list(api.settings.storage_root.rglob("*.mp4"))


def test_cue_review_glossary_export_and_feedback(api: Backend):
    api.register()
    client = api.client
    project = api.project()
    api.upload(project)
    first, second, third = api.cues(project)
    response = client.patch(
        f"/api/cues/{first['id']}", json={"expected_revision": 0, "human_text": "سڵاو هاوڕێ."}
    )
    assert response.status_code == 200
    edited = response.json()
    assert (edited["start_ms"], edited["end_ms"]) == (1000, 3000)
    assert (
        client.patch(
            f"/api/cues/{first['id']}", json={"expected_revision": 0, "human_text": "stale"}
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/cues/{first['id']}/approve", json={"expected_revision": edited["revision"]}
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/api/projects/{project}/cues/bulk-approve",
            json={"cues": [[second["id"], second["revision"]], [third["id"], third["revision"]]]},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/projects/{project}/glossary", json={"english": "Please", "sorani": "تکایە"}
        ).status_code
        == 201
    )
    assert (
        client.post(
            f"/api/projects/{project}/glossary", json={"english": "please", "sorani": "تکایە"}
        ).status_code
        == 409
    )
    current = api.cues(project)[0]
    assert "glossary_mismatch" in current["qa_warnings"]
    imported = client.post(
        f"/api/projects/{project}/glossary/import",
        files={"file": ("terms.csv", "english,sorani,locked\nhome,ماڵ,true\n".encode())},
    )
    assert imported.status_code == 200 and imported.json()["imported"] == 1
    assert client.get(f"/api/projects/{project}/qa").json()["flagged_count"] >= 1
    for fmt in ("srt", "vtt", "ass", "json", "txt"):
        exported = client.get(f"/api/projects/{project}/export?format={fmt}&bilingual=true")
        assert exported.status_code == 200, exported.text
        assert "machine" in exported.headers["x-subtitle-origin"]
        assert "سڵاو" in exported.text
    srt = client.get(f"/api/projects/{project}/export?format=srt").text
    assert "00:00:01,000 --> 00:00:03,000" in srt
    assert "Machine-assisted" in srt
    stats = client.get(f"/api/projects/{project}/stats").json()
    assert stats["cue_count"] == 3 and stats["approved_count"] == 1
    feedback = client.post("/api/feedback/export", json={}).text.splitlines()
    assert len(feedback) == 1
    row = json.loads(feedback[0])
    assert row["review_status"] == "verified" and row["language_verified"] is False
    revisions = client.get(f"/api/cues/{first['id']}/revisions").json()["items"]
    assert len(revisions) == 2
    restored = client.post(
        f"/api/cues/{first['id']}/revisions/{revisions[0]['id']}/restore",
        json={"expected_revision": current["revision"]},
    )
    assert restored.status_code == 200 and restored.json()["status"] == "edited"


def test_project_softening_is_explicit_and_cue_status_is_revisioned(api: Backend):
    api.register()
    project = api.project()
    api.upload(project)
    settings = api.client.get(f"/api/projects/{project}").json()
    assert settings["soften_profanity"] is False and settings["profanity_replacements"] == []
    assert (
        api.client.patch(f"/api/projects/{project}", json={"soften_profanity": True}).status_code
        == 422
    )
    response = api.client.patch(
        f"/api/projects/{project}",
        json={
            "soften_profanity": True,
            "profanity_replacements": [{"source": "example", "replacement": "alternative"}],
        },
    )
    assert response.status_code == 200 and response.json()["soften_profanity"] is True
    cue = api.cues(project)[0]
    assert (
        api.client.patch(
            f"/api/cues/{cue['id']}", json={"expected_revision": 0, "status": "approved"}
        ).status_code
        == 422
    )
    approved = api.client.patch(
        f"/api/cues/{cue['id']}",
        json={"expected_revision": 0, "human_text": "دەقی دەستکاریکراو", "status": "approved"},
    )
    assert approved.status_code == 200 and approved.json()["status"] == "approved"
    assert len(api.client.get(f"/api/cues/{cue['id']}/revisions").json()["items"]) == 1


def test_timing_edits_recalculate_both_neighbors_and_ass_is_exact(api: Backend):
    api.register()
    project = api.project()
    api.upload(project)
    first = api.cues(project)[0]
    changed = api.client.patch(
        f"/api/cues/{first['id']}", json={"expected_revision": 0, "end_ms": 3501}
    )
    assert changed.status_code == 200
    cues = api.cues(project)
    assert all("timing_overlap" in cue["qa_warnings"] for cue in cues[:2])
    assert api.client.get(f"/api/projects/{project}/export?format=ass").status_code == 422
    restored = api.client.patch(
        f"/api/cues/{first['id']}",
        json={"expected_revision": changed.json()["revision"], "end_ms": 3000},
    )
    assert restored.status_code == 200
    assert all("timing_overlap" not in cue["qa_warnings"] for cue in api.cues(project)[:2])
    assert api.client.get(f"/api/projects/{project}/export?format=ass").status_code == 200


def test_rate_limit_fails_closed_and_returns_retry_after(api: Backend):
    settings = api.settings.model_copy(update={"redis_url": "memory://", "auth_rate_limit": 2})
    with TestClient(create_app(settings)) as client:
        payload = {"email": "unknown@example.com", "password": PASSWORD}
        assert client.post("/api/auth/login", json=payload).status_code == 401
        assert client.post("/api/auth/login", json=payload).status_code == 401
        limited = client.post("/api/auth/login", json=payload)
        assert limited.status_code == 429 and "Retry-After" in limited.headers


def test_job_cancellation_and_resumable_events(api: Backend):
    api.register()
    client = api.client
    project = api.project()
    api.upload(project)
    # Create a model directly; production registration remains admin-only and validated.
    api.admin()
    model = client.post(
        "/api/models",
        json={
            "name": "test",
            "direction": "en-ckb",
            "kind": "base",
            "source": "fixture/not-cached",
        },
    )
    assert model.status_code == 201, model.text
    job = client.post(
        f"/api/projects/{project}/translate", json={"model_version_id": model.json()["id"]}
    )
    assert job.status_code == 202
    job_id = job.json()["job_id"]
    assert client.delete(f"/api/projects/{project}").status_code == 409
    assert (
        client.patch(
            f"/api/cues/{api.cues(project)[0]['id']}",
            json={"expected_revision": 0, "human_text": "not while running"},
        ).status_code
        == 409
    )
    assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    stream = client.get(f"/api/jobs/{job_id}/stream")
    assert stream.status_code == 200 and "event: done" in stream.text
    ids = [int(value) for value in re.findall(r"^id: (\d+)$", stream.text, re.M)]
    assert len(ids) >= 2
    resumed = client.get(f"/api/jobs/{job_id}/stream", headers={"Last-Event-ID": str(ids[0])})
    assert f"id: {ids[0]}\n" not in resumed.text and f"id: {ids[-1]}\n" in resumed.text
    assert (
        client.get(f"/api/jobs/{job_id}/stream", headers={"Last-Event-ID": "-1"}).status_code == 400
    )
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
