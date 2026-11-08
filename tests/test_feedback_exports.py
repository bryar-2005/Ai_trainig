"""Private feedback snapshots, consent, cleanup, and concurrent edits."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")
pytest.importorskip("celery")

from sqlalchemy import update  # noqa: E402

from api.db.models import Cue, FeedbackExport, Job, now  # noqa: E402
from api.db.session import create_engine, create_session_factory  # noqa: E402
from api.feedback_tasks import cleanup_feedback_exports, schedule_nightly_feedback_exports  # noqa: E402
from api.storage import LocalStorage  # noqa: E402
from api.worker import run_job  # noqa: E402
from tests.test_api import PASSWORD, Backend  # noqa: E402
from tests.test_api import api as shared_api  # noqa: E402

api = shared_api


def _eligible_project(api: Backend) -> str:
    project = api.project()
    api.upload(project)
    cues = api.cues(project)
    api.edit_and_approve(cues[0], "ئەمە دەستکارییەکی پەسەندکراوە.")
    # Unapproved content must never be exported.
    changed = api.client.patch(
        f"/api/cues/{cues[1]['id']}",
        json={"expected_revision": cues[1]["revision"], "human_text": "PRIVATE UNAPPROVED"},
    )
    assert changed.status_code == 200
    return project


def _job_and_export(api: Backend):
    api.settings.task_mode = "eager"
    response = api.client.post("/api/feedback/exports", json={})
    assert response.status_code == 202, response.text
    job = api.client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "completed", job
    return job, api.client.get("/api/feedback/exports").json()


def test_private_snapshot_lifecycle_and_project_deletion(api: Backend):
    api.register()
    project = _eligible_project(api)
    job, listing = _job_and_export(api)
    assert job["result"]["cue_count"] == 1
    assert "ئەمە" not in json.dumps(job, ensure_ascii=False)
    assert not listing["nightly_schedule_enabled"] and listing["retention_days"] == 7
    item = listing["items"][0]
    assert item["id"] == job["result"]["export_id"]
    response = api.client.get(f"/api/feedback/exports/{item['id']}/download")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert hashlib.sha256(response.content).hexdigest() == item["sha256"]
    record = json.loads(response.text)
    assert record["provenance"] == "human" and record["review_status"] == "verified"
    assert record["language_verified"] is False and "PRIVATE UNAPPROVED" not in response.text

    # Even another administrator cannot download someone else's training text.
    api.client.cookies.clear()
    api.register("other@example.com")
    api.admin("other@example.com")
    assert api.client.get("/api/feedback/exports").json()["items"] == []
    assert api.client.get(f"/api/feedback/exports/{item['id']}/download").status_code == 404
    assert api.client.delete(f"/api/feedback/exports/{item['id']}").status_code == 404
    assert (
        api.client.post(
            "/api/auth/login", json={"email": "owner@example.com", "password": PASSWORD}
        ).status_code
        == 200
    )
    api.csrf()
    assert api.client.delete(f"/api/feedback/exports/{item['id']}").status_code == 204
    assert not list(api.settings.storage_root.rglob("*.jsonl"))
    _, listing = _job_and_export(api)
    next_id = listing["items"][0]["id"]
    assert api.client.delete(f"/api/projects/{project}").status_code == 204
    assert api.client.get(f"/api/feedback/exports/{next_id}/download").status_code == 404
    assert not list(api.settings.storage_root.rglob("*.jsonl"))


def test_empty_and_expired_snapshots_are_safe(api: Backend):
    api.register()
    job, listing = _job_and_export(api)
    assert job["result"] == {"export_id": None, "cue_count": 0}
    assert listing["items"] == [] and not list(api.settings.storage_root.rglob("*.jsonl"))
    _eligible_project(api)
    _, listing = _job_and_export(api)
    export_id = listing["items"][0]["id"]
    api.execute(
        update(FeedbackExport)
        .where(FeedbackExport.id == uuid.UUID(export_id))
        .values(expires_at=now() - timedelta(seconds=1))
    )
    assert api.client.get(f"/api/feedback/exports/{export_id}/download").status_code == 404
    assert api.client.get("/api/feedback/exports").json()["items"] == []
    assert asyncio.run(cleanup_feedback_exports(api.settings)) == 1
    assert not list(api.settings.storage_root.rglob("*.jsonl"))


def test_changed_feedback_discards_pending_snapshot(api: Backend, monkeypatch):
    api.register()
    project = _eligible_project(api)
    cue = api.cues(project)[0]
    original = LocalStorage.put_file

    def change_after_upload(storage, source, key, content_type):
        original(storage, source, key, content_type)
        if key.startswith("feedback-exports/"):
            api.execute(
                update(Cue).where(Cue.id == uuid.UUID(cue["id"])).values(revision=Cue.revision + 1)
            )

    monkeypatch.setattr(LocalStorage, "put_file", change_after_upload)
    api.settings.task_mode = "eager"
    response = api.client.post("/api/feedback/exports", json={})
    assert response.status_code == 202
    job = api.client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert api.client.get("/api/feedback/exports").json()["items"] == []
    assert not list(api.settings.storage_root.rglob("*.jsonl"))
    assert asyncio.run(cleanup_feedback_exports(api.settings)) == 0


def test_nightly_requires_operator_and_owner_opt_in_and_deduplicates(api: Backend):
    api.register()
    _eligible_project(api)
    assert (
        api.client.put(
            "/api/account/preferences", json={"nightly_feedback_export": True}
        ).status_code
        == 200
    )
    assert asyncio.run(schedule_nightly_feedback_exports(api.settings)) == 0
    api.settings.nightly_feedback_export_enabled = True
    assert asyncio.run(schedule_nightly_feedback_exports(api.settings)) == 1
    assert asyncio.run(schedule_nightly_feedback_exports(api.settings)) == 0
    jobs = api.client.get("/api/jobs").json()["items"]
    job = next(item for item in jobs if item["kind"] == "feedback_export")
    assert asyncio.run(run_job(uuid.UUID(job["id"]), api.settings)) == "completed"
    assert len(api.client.get("/api/feedback/exports").json()["items"]) == 1
    assert asyncio.run(schedule_nightly_feedback_exports(api.settings)) == 0
    tomorrow = now() + timedelta(days=1)
    assert asyncio.run(schedule_nightly_feedback_exports(api.settings, when=tomorrow)) == 1
    next_job = api.client.get("/api/jobs").json()["items"][0]
    assert (
        api.client.put(
            "/api/account/preferences", json={"nightly_feedback_export": False}
        ).status_code
        == 200
    )
    assert asyncio.run(run_job(uuid.UUID(next_job["id"]), api.settings)) == "cancelled"
    assert len(api.client.get("/api/feedback/exports").json()["items"]) == 1


def test_cleanup_removes_abandoned_upload_but_not_live_attempt(api: Backend):
    user = api.register()
    user_id, job_id, lease, export_id = (uuid.uuid4() for _ in range(4))
    user_id = uuid.UUID(user["id"])
    key = f"feedback-exports/{user_id}/{job_id}/{export_id}.jsonl"
    path = api.settings.storage_root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("private\n", encoding="utf-8")

    async def insert():
        engine = create_engine(api.settings.database_url)
        sessions = create_session_factory(engine)
        async with sessions() as db:
            db.add(
                Job(
                    id=job_id,
                    owner_id=user_id,
                    kind="feedback_export",
                    status="running",
                    stage="working",
                    progress=1,
                    payload={},
                    lease_token=lease,
                )
            )
            await db.flush()
            db.add(
                FeedbackExport(
                    id=export_id,
                    owner_id=user_id,
                    job_id=job_id,
                    lease_token=lease,
                    storage_key=key,
                    status="pending",
                    cue_count=1,
                    size_bytes=8,
                    sha256="0" * 64,
                    expires_at=now() + timedelta(days=1),
                )
            )
            await db.commit()
        await engine.dispose()

    asyncio.run(insert())
    assert asyncio.run(cleanup_feedback_exports(api.settings)) == 0 and path.exists()
    api.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(updated_at=now() - timedelta(seconds=api.settings.job_lease_seconds + 1))
    )
    assert asyncio.run(cleanup_feedback_exports(api.settings)) == 1 and not path.exists()


def test_live_celery_on_demand_and_nightly_exports(api: Backend, tmp_path):
    import os
    import subprocess
    import sys
    import time

    from celery import Celery

    if not all(
        os.environ.get(name)
        for name in ("SORANI_TEST_DATABASE_URL", "SORANI_TEST_REDIS_URL", "SORANI_TEST_BROKER_URL")
    ):
        pytest.skip("Live Celery test requires disposable PostgreSQL and Redis URLs")
    api.register()
    _eligible_project(api)
    assert (
        api.client.put(
            "/api/account/preferences", json={"nightly_feedback_export": True}
        ).status_code
        == 200
    )
    api.settings.task_mode = "celery"
    api.settings.nightly_feedback_export_enabled = True
    environment = {
        **os.environ,
        "SORANI_ENVIRONMENT": "test",
        "SORANI_SECRET_KEY": api.settings.secret_key.get_secret_value(),
        "SORANI_DATABASE_URL": api.settings.database_url,
        "SORANI_REDIS_URL": api.settings.redis_url,
        "SORANI_BROKER_URL": api.settings.broker_url,
        "SORANI_STORAGE_ROOT": str(api.settings.storage_root),
        "SORANI_MODEL_ROOT": str(api.settings.model_root),
        "SORANI_COOKIE_SECURE": "false",
        "SORANI_TASK_MODE": "celery",
        "SORANI_NIGHTLY_FEEDBACK_EXPORT_ENABLED": "true",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
    }
    log = tmp_path / "feedback-worker.log"
    with log.open("w+") as stream:
        worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "celery",
                "-A",
                "api.worker:celery_app",
                "worker",
                "--loglevel=INFO",
                "--pool=solo",
                "--concurrency=1",
                "--without-gossip",
                "--without-mingle",
                "--without-heartbeat",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        publisher = Celery("feedback-test-publisher", broker=api.settings.broker_url)
        try:
            queued = api.client.post("/api/feedback/exports", json={})
            assert queued.status_code == 202
            job_id = queued.json()["job_id"]
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                job = api.client.get(f"/api/jobs/{job_id}").json()
                if (
                    job["status"] in {"completed", "failed", "cancelled"}
                    or worker.poll() is not None
                ):
                    break
                time.sleep(0.1)
            stream.flush()
            assert job["status"] == "completed", (job, log.read_text()[-8000:])
            publisher.send_task("api.schedule_feedback_exports", retry=False)
            while time.monotonic() < deadline:
                items = api.client.get("/api/feedback/exports").json()["items"]
                if len(items) == 2 or worker.poll() is not None:
                    break
                time.sleep(0.1)
            stream.flush()
            assert len(items) == 2, log.read_text()[-8000:]
            for item in items:
                response = api.client.get(f"/api/feedback/exports/{item['id']}/download")
                assert (
                    response.status_code == 200
                    and json.loads(response.text)["language_verified"] is False
                )
        finally:
            publisher.close()
            worker.terminate()
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)


def test_postgres_concurrent_nightly_schedulers_create_only_one_job(api: Backend):
    from concurrent.futures import ThreadPoolExecutor

    if not api.settings.database_url.startswith("postgresql"):
        pytest.skip("Row-lock concurrency requires disposable PostgreSQL")
    api.register()
    _eligible_project(api)
    api.client.put("/api/account/preferences", json={"nightly_feedback_export": True})
    api.settings.nightly_feedback_export_enabled = True
    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(
            pool.map(
                lambda _: asyncio.run(schedule_nightly_feedback_exports(api.settings)), range(4)
            )
        )
    assert sum(counts) == 1
    jobs = api.client.get("/api/jobs").json()["items"]
    assert len([job for job in jobs if job["kind"] == "feedback_export"]) == 1


def test_cleanup_removes_ready_and_abandoned_private_scratch(api: Backend):
    import os
    import time

    from api.feedback_tasks import feedback_scratch_path

    api.register()
    _eligible_project(api)
    _, listing = _job_and_export(api)
    export_id = uuid.UUID(listing["items"][0]["id"])
    ready_dir = feedback_scratch_path(api.settings, export_id)
    ready_dir.mkdir(parents=True)
    (ready_dir / "private.jsonl").write_text("private\n", encoding="utf-8")
    orphan_dir = feedback_scratch_path(api.settings, uuid.uuid4())
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "private.jsonl").write_text("private\n", encoding="utf-8")
    stale = time.time() - api.settings.job_lease_seconds - 10
    os.utime(orphan_dir, (stale, stale))
    assert asyncio.run(cleanup_feedback_exports(api.settings)) == 0
    assert not ready_dir.exists() and not orphan_dir.exists()
    assert len(api.client.get("/api/feedback/exports").json()["items"]) == 1


def test_nightly_does_not_export_another_user_without_consent(api: Backend):
    api.register()
    _eligible_project(api)
    api.client.put("/api/account/preferences", json={"nightly_feedback_export": True})
    api.client.cookies.clear()
    api.register("no-consent@example.com")
    _eligible_project(api)
    api.settings.nightly_feedback_export_enabled = True
    assert asyncio.run(schedule_nightly_feedback_exports(api.settings)) == 1
    assert api.client.get("/api/jobs").json()["items"] == []
    assert api.client.get("/api/feedback/exports").json()["items"] == []


def test_disabling_operator_schedule_cancels_queued_nightly_work(api: Backend):
    api.register()
    _eligible_project(api)
    api.client.put("/api/account/preferences", json={"nightly_feedback_export": True})
    api.settings.nightly_feedback_export_enabled = True
    assert asyncio.run(schedule_nightly_feedback_exports(api.settings)) == 1
    job_id = api.client.get("/api/jobs").json()["items"][0]["id"]
    api.settings.nightly_feedback_export_enabled = False
    assert asyncio.run(run_job(uuid.UUID(job_id), api.settings)) == "cancelled"
    assert api.client.get("/api/feedback/exports").json()["items"] == []
