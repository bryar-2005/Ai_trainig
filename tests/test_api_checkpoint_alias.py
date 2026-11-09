"""The worker can use a verified M2M100 checkpoint with a retained tied alias."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")
pytest.importorskip("celery")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from tests.test_api import Backend  # noqa: E402
from tests.test_api import api as shared_api  # noqa: E402
from tests.test_inference_checkpoint_alias import make_checkpoint  # noqa: E402

api = shared_api


def test_alias_checkpoint_can_translate_through_real_worker(api: Backend):
    api.register()
    api.admin()
    api.settings.allow_smoke_models = True
    api.settings.task_mode = "eager"
    root = api.settings.model_root
    root.mkdir(parents=True, exist_ok=True)
    path = root / "alias-checkpoint"
    make_checkpoint(path, "model.decoder.embed_tokens.weight")
    registered = api.client.post(
        "/api/models",
        json={
            "name": "random-alias-checkpoint",
            "direction": "en-ckb",
            "kind": "base",
            "source": str(path),
        },
    )
    assert registered.status_code == 201, registered.text
    project = api.project("Synthetic checkpoint repair", registered.json()["id"])
    api.upload(project)
    response = api.client.post(f"/api/projects/{project}/translate", json={})
    assert response.status_code == 202
    job = api.client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "completed", job
    # A tiny random model can legitimately emit only special tokens.
    assert isinstance(api.cues(project)[0]["machine_text"], str)
    assert any("random" in item.lower() for item in job["result"]["model"]["warnings"])
