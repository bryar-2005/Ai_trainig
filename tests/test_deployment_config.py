"""Static safeguards for the container deployment (a real Docker smoke test runs in CI)."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]


def _compose(name: str):
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def test_compose_keeps_backing_services_private_and_uses_persistent_volumes():
    compose = _compose("compose.yaml")
    services = compose["services"]
    assert {
        "web",
        "api",
        "worker-cpu",
        "beat",
        "postgres",
        "redis",
        "minio",
        "minio-init",
        "migrate",
    } <= set(services)
    assert all("ports" not in service for name, service in services.items() if name != "web")
    assert services["web"]["ports"] == ["${WEB_BIND:-127.0.0.1}:${WEB_PORT:-3000}:3000"]
    assert services["api"]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
    assert services["web"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert services["api"]["environment"]["SORANI_STORAGE_BACKEND"] == "s3"
    assert "MINIO_APP_ACCESS_KEY" in services["api"]["environment"]["AWS_ACCESS_KEY_ID"]
    assert "MINIO_ROOT_PASSWORD" not in str(services["api"]["environment"])
    assert "env_file" not in services["api"]
    assert services["worker-cpu"]["environment"]["SORANI_INFERENCE_DEVICE"] == "cpu"
    assert services["web"]["environment"]["INTERNAL_API_ORIGIN"] == "http://api:8000"
    assert {
        "postgres-data",
        "redis-data",
        "minio-data",
        "private-scratch",
        "model-assets",
        "huggingface-cache",
    } <= set(compose["volumes"])
    for service, secret in (
        ("postgres", "POSTGRES_PASSWORD"),
        ("redis", "REDIS_PASSWORD"),
        ("minio", "MINIO_ROOT_PASSWORD"),
    ):
        assert ":?" in services[service]["environment"][secret]


def test_gpu_overlay_extends_worker_but_disables_cpu_by_default():
    gpu = _compose("compose.gpu.yaml")["services"]
    assert gpu["worker-cpu"]["profiles"] == ["cpu"]
    worker = gpu["worker-gpu"]
    assert worker["extends"] == {"file": "compose.yaml", "service": "worker-cpu"}
    assert worker["profiles"] == ["gpu"]
    assert worker["environment"]["SORANI_INFERENCE_DEVICE"] == "cuda"
    assert worker["deploy"]["resources"]["reservations"]["devices"][0]["capabilities"] == ["gpu"]
    assert worker["build"]["args"]["TORCH_INDEX_URL"].endswith("/cu124")


def test_templates_do_not_contain_default_credentials_or_bundle_private_inputs():
    example = (ROOT / ".env.compose.example").read_text(encoding="utf-8")
    for name in (
        "POSTGRES_PASSWORD",
        "REDIS_PASSWORD",
        "MINIO_ROOT_USER",
        "MINIO_ROOT_PASSWORD",
        "MINIO_APP_ACCESS_KEY",
        "MINIO_APP_SECRET_KEY",
        "SORANI_SECRET_KEY",
    ):
        assert f"{name}=\n" in example
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for name in (".env*", "/data", "/outputs", "*.srt", "*.jsonl", "*.safetensors"):
        assert name in ignore
    assert "USER 10001:10001" in (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER node" in (ROOT / "web/Dockerfile").read_text(encoding="utf-8")
