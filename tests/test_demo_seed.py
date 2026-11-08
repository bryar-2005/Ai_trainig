"""The optional demo never installs a model or overwrites an account."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")
pytest.importorskip("celery")

from sqlalchemy import select  # noqa: E402

from api.db.models import ModelVersion  # noqa: E402
from api.db.seed_demo import create_demo  # noqa: E402
from tests.test_api import PASSWORD, Backend  # noqa: E402
from tests.test_api import api as shared_api  # noqa: E402

api = shared_api


def test_demo_is_explicit_synthetic_untranslated_and_non_destructive(api: Backend):
    settings = api.settings
    project_id = asyncio.run(create_demo("demo@example.com", "Demo Reviewer", PASSWORD, settings))
    login = api.client.post(
        "/api/auth/login", json={"email": "demo@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200 and login.json()["role"] == "user"
    api.csrf()
    project = api.client.get(f"/api/projects/{project_id}").json()
    assert "synthetic" in project["description"].lower() and project["model_version_id"] is None
    cues = api.cues(str(project_id))
    assert len(cues) == 3 and all(
        cue["human_text"] is None and cue["machine_text"] == "" for cue in cues
    )
    assert api.client.get("/api/models").json()["items"] == []
    assert not api.execute(select(ModelVersion)).scalars().all()
    with pytest.raises(ValueError, match="already exists"):
        asyncio.run(create_demo("demo@example.com", "Again", PASSWORD, settings))
    production = settings.model_copy(update={"environment": "production"})
    with pytest.raises(ValueError, match="disabled in production"):
        asyncio.run(create_demo("second@example.com", "Demo", PASSWORD, production))
