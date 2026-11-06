"""Real CPU model execution and worker lifecycle against the API database."""

from __future__ import annotations

import asyncio
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")
pytest.importorskip("celery")

from sqlalchemy import select, update  # noqa: E402

from api.db.models import Cue, Job, Project, now  # noqa: E402
from api.db.session import create_engine, create_session_factory  # noqa: E402
from api.worker import JobCancelled, Reporter, _claim, _finish_failure  # noqa: E402
from tests.test_api import Backend  # noqa: E402
from tests.test_api import api as shared_api  # noqa: E402

api = shared_api


@pytest.fixture(scope="module")
def synthetic_model(tmp_path_factory: pytest.TempPathFactory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from datasets import DatasetDict, disable_progress_bars

    from ml.train.config import load_config
    from ml.train.train_lora import run_training
    from tests.test_train_integration import _rows

    root = tmp_path_factory.mktemp("api-random-model")
    disable_progress_bars()
    DatasetDict(
        {name: _rows(name, count) for name, count in (("train", 200), ("val", 20), ("test", 20))}
    ).save_to_disk(str(root / "data"))
    config = replace(
        load_config("ml/train/config.yaml"),
        dataset_dir=str(root / "data"),
        output_dir=str(root / "run"),
    )
    manifest = run_training(config, smoke_test=True)
    assert manifest["smoke_test"] and manifest["status"] == "completed"
    return root / "run"


def register_synthetic(api: Backend, synthetic_model: Path) -> str:
    api.admin()
    root = api.settings.model_root
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(synthetic_model / "smoke_base", root / "base")
    shutil.copytree(synthetic_model / "adapter", root / "adapter")
    response = api.client.post(
        "/api/models",
        json={
            "name": "random-test-adapter",
            "direction": "en-ckb",
            "kind": "adapter",
            "source": str(root / "base"),
            "adapter_dir": str(root / "adapter"),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_real_translation_retranslation_feedback_training_and_compare(
    api: Backend, synthetic_model
):
    api.register()
    api.settings.allow_smoke_models = True
    api.settings.task_mode = "eager"
    model_id = register_synthetic(api, synthetic_model)
    first_project = api.project("Original synthetic film", model_id)
    api.upload(first_project)
    initial = api.cues(first_project)
    original_times = [(cue["start_ms"], cue["end_ms"]) for cue in initial]
    # An unapproved target may be used as neither feedback nor source context.
    draft = api.client.patch(
        f"/api/cues/{initial[1]['id']}",
        json={"expected_revision": 0, "human_text": "UNREVIEWED SECRET TARGET"},
    )
    assert draft.status_code == 200
    response = api.client.post(f"/api/projects/{first_project}/translate", json={})
    assert response.status_code == 202
    job = api.client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "completed", job
    assert any("random" in item.lower() for item in job["result"]["model"]["warnings"])
    assert "source" not in job["result"]["model"]
    cues = api.cues(first_project)
    assert [(cue["start_ms"], cue["end_ms"]) for cue in cues] == original_times
    assert cues[1]["human_text"] == "UNREVIEWED SECRET TARGET"
    assert cues[0]["sentence_source_cue_ids"] == [1, 2]
    assert cues[0]["machine_text"]
    approved = api.edit_and_approve(cues[2])
    before_human = approved["human_text"]
    second_project = api.project("Second synthetic film", model_id)
    api.upload(second_project)
    api.edit_and_approve(api.cues(second_project)[0], "سڵاو هاوڕێ.")
    assert api.client.get("/api/models/feedback-stats").json()["approved_edits"] == 2
    denied = api.client.post(
        "/api/models/train", json={"direction": "en-ckb", "model_version_id": model_id}
    )
    assert denied.status_code == 422
    training = api.client.post(
        "/api/models/train",
        json={
            "direction": "en-ckb",
            "model_version_id": model_id,
            "epochs": 1,
            "allow_unverified_language": True,
        },
    )
    assert training.status_code == 202
    job = api.client.get(f"/api/jobs/{training.json()['job_id']}").json()
    assert job["status"] == "completed", job
    assert job["result"]["feedback_counts"] == {"train": 1, "val": 1, "test": 0}
    assert any("random" in message.lower() for message in job["result"]["warnings"])
    feedback_model = job["result"]["model_version_id"]
    comparison = api.client.post(
        "/api/models/compare",
        json={
            "direction": "en-ckb",
            "model_a": model_id,
            "model_b": feedback_model,
            "lines": ["We should go home."],
            "context": ["The work is finished."],
        },
    )
    assert comparison.status_code == 202
    compared = api.client.get(f"/api/jobs/{comparison.json()['job_id']}").json()
    assert compared["status"] == "completed", compared
    assert len(compared["result"]["models"]) == 2
    selected = api.cues(first_project)[0]
    retranslated = api.client.post(
        f"/api/cues/{selected['id']}/retranslate",
        json={"expected_revision": selected["revision"], "custom_context": ["A new clue."]},
    )
    assert retranslated.status_code == 202
    job = api.client.get(f"/api/jobs/{retranslated.json()['job_id']}").json()
    assert job["status"] == "completed", job
    after = api.cues(first_project)
    assert "partial_sentence_retranslation_needs_review" in after[0]["qa_warnings"]
    assert after[2]["human_text"] == before_human and after[2]["status"] == "approved"
    assert [(cue["start_ms"], cue["end_ms"]) for cue in after] == original_times


def test_model_registry_rejects_smoke_without_test_opt_in_and_wrong_direction(
    api: Backend, synthetic_model: Path
):
    api.register()
    api.admin()
    root = api.settings.model_root
    root.mkdir(parents=True)
    shutil.copytree(synthetic_model / "smoke_base", root / "base")
    shutil.copytree(synthetic_model / "adapter", root / "adapter")
    payload = {
        "name": "random",
        "direction": "en-ckb",
        "kind": "adapter",
        "source": str(root / "base"),
        "adapter_dir": str(root / "adapter"),
    }
    assert api.client.post("/api/models", json=payload).status_code == 422
    api.settings.allow_smoke_models = True
    assert (
        api.client.post("/api/models", json={**payload, "direction": "ckb-en"}).status_code == 422
    )
    assert api.client.post("/api/models", json={**payload, "source": "/tmp"}).status_code == 422


def test_feedback_context_uses_intervening_source_not_unapproved_targets(
    api: Backend, tmp_path: Path
):
    pytest.importorskip("torch")
    from datasets import load_from_disk

    from api.worker_tasks import _feedback_dataset
    from ml.train.dataset import make_context_examples, validate_splits

    api.register()
    first = api.project("First")
    second = api.project("Second")
    api.upload(first)
    api.upload(second)
    cues = api.cues(first)
    assert (
        api.client.patch(
            f"/api/cues/{cues[1]['id']}",
            json={"expected_revision": 0, "human_text": "UNREVIEWED SECRET TARGET"},
        ).status_code
        == 200
    )
    selected = api.edit_and_approve(cues[2])
    api.edit_and_approve(api.cues(second)[0])

    async def records():
        engine = create_engine(api.settings.database_url)
        try:
            async with create_session_factory(engine)() as session:
                rows = (
                    await session.execute(
                        select(Cue, Project).join(Project).where(Cue.status == "approved")
                    )
                ).all()
                sources = (await session.scalars(select(Cue))).all()
                return rows, sources
        finally:
            await engine.dispose()

    rows, sources = asyncio.run(records())
    output = tmp_path / "feedback-data"
    assert _feedback_dataset(rows, sources, output) == {"train": 1, "val": 1, "test": 0}
    data = load_from_disk(str(output))
    splits = {name: list(data[name]) for name in ("train", "val", "test")}
    validate_splits(splits, allow_unverified=True)
    row = next(
        item for split in splits.values() for item in split if item["pair_id"] == selected["id"]
    )
    assert any(
        "Please find the object and bring it back." in text for text in row["source_context"]
    )
    assert "UNREVIEWED SECRET TARGET" not in str(splits)
    example = make_context_examples([row], direction="en-ckb")[0]
    assert example.context == tuple(row["source_context"])


def test_training_rejects_feedback_changed_after_enqueue_and_cleans_up(
    api: Backend, synthetic_model
):
    from api.worker import run_job

    api.register()
    api.settings.allow_smoke_models = True
    model = register_synthetic(api, synthetic_model)
    first = api.project("First reviewed film")
    second = api.project("Second reviewed film")
    api.upload(first)
    api.upload(second)
    approved = api.edit_and_approve(api.cues(first)[0])
    api.edit_and_approve(api.cues(second)[0])
    response = api.client.post(
        "/api/models/train",
        json={
            "direction": "en-ckb",
            "model_version_id": model,
            "epochs": 1,
            "allow_unverified_language": True,
        },
    )
    assert response.status_code == 202 and response.json()["status"] == "queued"
    edited = api.client.patch(
        f"/api/cues/{approved['id']}",
        json={"expected_revision": approved["revision"], "human_text": "Changed after enqueue"},
    )
    assert edited.status_code == 200 and edited.json()["status"] == "edited"
    job_id = response.json()["job_id"]
    assert asyncio.run(run_job(uuid.UUID(job_id), api.settings)) == "failed"
    job = api.client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed" and "Changed after enqueue" not in job["error"]
    assert not (api.settings.model_root / "feedback" / job_id).exists()


def test_worker_leases_fence_stale_attempts(api: Backend):
    api.register()
    api.admin()
    model = api.client.post(
        "/api/models",
        json={
            "name": "empty",
            "kind": "base",
            "direction": "en-ckb",
            "source": "fixture/not-cached",
        },
    ).json()["id"]
    project = api.project(model_id=model)
    api.upload(project)
    job = api.client.post(f"/api/projects/{project}/translate", json={}).json()["job_id"]

    async def exercise():
        engine = create_engine(api.settings.database_url)
        sessions = create_session_factory(engine)
        try:
            job_id, old, fresh = uuid.UUID(job), uuid.uuid4(), uuid.uuid4()
            assert await _claim(job_id, old, sessions, api.settings) == "claimed"
            assert await _claim(job_id, fresh, sessions, api.settings) == "busy"
            async with sessions() as session:
                await session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        updated_at=now() - timedelta(seconds=api.settings.job_lease_seconds + 5)
                    )
                )
                await session.commit()
            assert await _claim(job_id, fresh, sessions, api.settings) == "claimed"
            stale = Reporter(job_id, old, sessions, asyncio.get_running_loop())
            with pytest.raises(JobCancelled):
                await stale.update("stale", 50)
            await _finish_failure(job_id, old, sessions, cancelled=True, error="")
            live = Reporter(job_id, fresh, sessions, asyncio.get_running_loop())
            await live.update("live", 25)
            async with sessions() as session:
                stored = await session.get(Job, job_id)
                assert stored.status == "running" and stored.lease_token == fresh
        finally:
            await engine.dispose()

    asyncio.run(exercise())


def test_failed_worker_reports_failure_without_dialogue(api: Backend):
    api.register()
    api.admin()
    api.settings.task_mode = "eager"
    model = api.client.post(
        "/api/models",
        json={"name": "bad", "kind": "base", "direction": "en-ckb", "source": "fixture/not-cached"},
    ).json()["id"]
    project = api.project(model_id=model)
    api.upload(project)
    response = api.client.post(f"/api/projects/{project}/translate", json={})
    assert response.status_code == 202
    job = api.client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert "Please find" not in job["error"]
    assert api.client.get(f"/api/projects/{project}").json()["status"] == "failed"
    assert all(not cue["machine_text"] for cue in api.cues(project))


def test_concurrent_revision_updates_are_serialized(api: Backend):
    if not api.settings.database_url.startswith("postgresql"):
        pytest.skip("Row-lock semantics require the production PostgreSQL backend")
    api.register()
    project = api.project()
    api.upload(project)
    cue = api.cues(project)[0]
    barrier = threading.Barrier(2)

    def edit(text):
        barrier.wait(timeout=10)
        return api.client.patch(
            f"/api/cues/{cue['id']}", json={"expected_revision": 0, "human_text": text}
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(edit, ["First human edit", "Second human edit"]))
    assert sorted(results) == [200, 409]
    assert api.cues(project)[0]["revision"] == 1
