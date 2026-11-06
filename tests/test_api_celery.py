"""Live Celery/Redis dispatch and durable-outbox recovery on disposable services."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("celery")
pytest.importorskip("sqlalchemy")

from sqlalchemy import update  # noqa: E402

from api.db.models import Job, now  # noqa: E402
from api.db.session import create_engine, create_session_factory  # noqa: E402
from api.worker import _claim, requeue_stale_jobs  # noqa: E402
from tests.test_api import Backend  # noqa: E402
from tests.test_api import api as shared_api  # noqa: E402
from tests.test_api_workers import register_synthetic  # noqa: E402
from tests.test_api_workers import synthetic_model as shared_synthetic_model  # noqa: E402

api = shared_api
synthetic_model = shared_synthetic_model


def test_celery_redis_dispatch_and_outbox_recovery(
    api: Backend, synthetic_model: Path, tmp_path: Path
):
    if not all(
        os.environ.get(name)
        for name in ("SORANI_TEST_DATABASE_URL", "SORANI_TEST_REDIS_URL", "SORANI_TEST_BROKER_URL")
    ):
        pytest.skip("Live Celery test requires disposable PostgreSQL and Redis URLs")
    api.register()
    api.settings.allow_smoke_models = True
    model = register_synthetic(api, synthetic_model)
    queued_project = api.project("Recover queued job", model)
    api.upload(queued_project)
    response = api.client.post(f"/api/projects/{queued_project}/translate", json={})
    assert response.status_code == 202 and response.json()["status"] == "queued"
    first_job = response.json()["job_id"]
    expired_project = api.project("Recover expired lease", model)
    api.upload(expired_project)
    expired_response = api.client.post(f"/api/projects/{expired_project}/translate", json={})
    expired_job = expired_response.json()["job_id"]

    async def expire_lease():
        engine = create_engine(api.settings.database_url)
        sessions = create_session_factory(engine)
        try:
            job_id = uuid.UUID(expired_job)
            assert await _claim(job_id, uuid.uuid4(), sessions, api.settings) == "claimed"
            async with sessions() as session:
                await session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        updated_at=now() - timedelta(seconds=api.settings.job_lease_seconds + 5)
                    )
                )
                await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(expire_lease())
    api.settings.task_mode = "celery"
    # Simulate both a missing broker publication and a worker that lost its lease.
    assert asyncio.run(requeue_stale_jobs(api.settings)) >= 2
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
        "SORANI_ALLOW_SMOKE_MODELS": "true",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    log = tmp_path / "worker.log"
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
        try:
            project = api.project("Broker dispatch", model)
            api.upload(project)
            dispatched = api.client.post(f"/api/projects/{project}/translate", json={})
            assert dispatched.status_code == 202
            jobs = [first_job, expired_job, dispatched.json()["job_id"]]
            deadline = time.monotonic() + 90
            states = {}
            while time.monotonic() < deadline:
                states = {job: api.client.get(f"/api/jobs/{job}").json() for job in jobs}
                if all(
                    value["status"] in {"completed", "failed", "cancelled"}
                    for value in states.values()
                ):
                    break
                if worker.poll() is not None:
                    break
                time.sleep(0.1)
            stream.flush()
            detail = log.read_text(encoding="utf-8")[-8000:]
            assert all(value["status"] == "completed" for value in states.values()), (
                states,
                detail,
            )
            assert api.cues(queued_project)[0]["machine_text"]
            assert api.cues(expired_project)[0]["machine_text"]
            assert api.cues(project)[0]["machine_text"]
        finally:
            worker.terminate()
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
