"""Exercise an already-running Compose stack without model weights or external data."""

from __future__ import annotations

import hashlib
import http.cookiejar
import json
import os
import secrets
import time
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener

BASE = os.environ.get("SORANI_SMOKE_BASE", "http://127.0.0.1:3000").rstrip("/")
jar = http.cookiejar.CookieJar()
opener = build_opener(HTTPCookieProcessor(jar))


def call(method: str, path: str, body: bytes | None = None, content_type: str | None = None):
    headers = {"Origin": BASE}
    csrf = next((cookie.value for cookie in jar if cookie.name == "sorani_csrf"), None)
    if csrf:
        headers["X-CSRF-Token"] = csrf
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(BASE + path, data=body, headers=headers, method=method)
    try:
        with opener.open(request, timeout=30) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def json_call(method: str, path: str, payload: dict | None = None):
    status, body = call(
        method,
        path,
        json.dumps(payload).encode() if payload is not None else None,
        "application/json" if payload is not None else None,
    )
    return status, json.loads(body) if body else None


def expect(status: int, actual: int, label: str) -> None:
    if actual != status:
        raise RuntimeError(f"{label}: expected HTTP {status}, got {actual}")


def main() -> None:
    email = f"compose-{secrets.token_hex(6)}@example.com"
    password = secrets.token_urlsafe(24)
    status, user = json_call(
        "POST",
        "/api/auth/register",
        {"email": email, "display_name": "Compose smoke", "password": password},
    )
    expect(201, status, "registration")
    assert user["role"] == "user"
    status, project = json_call(
        "POST",
        "/api/projects",
        {"name": "Original synthetic compose fixture", "direction": "en-ckb"},
    )
    expect(201, status, "project creation")
    project_id = project["id"]
    boundary = "sorani" + secrets.token_hex(12)
    subtitle = b"1\n00:00:01,000 --> 00:00:03,000\nThe lantern is lit.\n"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="subtitle"; filename="synthetic.srt"\r\n'
            "Content-Type: application/x-subrip\r\n\r\n"
        ).encode()
        + subtitle
        + f"\r\n--{boundary}--\r\n".encode()
    )
    status, _ = call(
        "POST",
        f"/api/projects/{project_id}/upload",
        body,
        f"multipart/form-data; boundary={boundary}",
    )
    expect(201, status, "private subtitle upload")
    status, listing = json_call("GET", f"/api/projects/{project_id}/cues")
    expect(200, status, "cue listing")
    cue = listing["items"][0]
    status, cue = json_call(
        "PATCH",
        f"/api/cues/{cue['id']}",
        {"expected_revision": cue["revision"], "human_text": "چرا هەڵگیرساوە."},
    )
    expect(200, status, "human edit")
    status, cue = json_call(
        "POST", f"/api/cues/{cue['id']}/approve", {"expected_revision": cue["revision"]}
    )
    expect(200, status, "human approval")
    status, queued = json_call("POST", "/api/feedback/exports", {})
    expect(202, status, "feedback job")
    job_id = queued["job_id"]
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        status, job = json_call("GET", f"/api/jobs/{job_id}")
        expect(200, status, "job status")
        if job["status"] == "completed":
            break
        if job["status"] in {"failed", "cancelled"}:
            raise RuntimeError(f"Feedback job ended as {job['status']}")
        time.sleep(1)
    else:
        raise RuntimeError("Feedback worker did not complete within 90 seconds")
    export_id = job["result"]["export_id"]
    assert job["result"]["cue_count"] == 1 and export_id
    status, raw = call("GET", f"/api/feedback/exports/{export_id}/download")
    expect(200, status, "private feedback download")
    assert hashlib.sha256(raw).hexdigest() == job["result"]["sha256"]
    record = json.loads(raw)
    assert record["provenance"] == "human" and record["language_verified"] is False
    status, _ = call("DELETE", f"/api/projects/{project_id}")
    expect(204, status, "project deletion")
    status, _ = call("GET", f"/api/feedback/exports/{export_id}/download")
    expect(404, status, "snapshot invalidation")
    print(
        "Deployment smoke passed: registration, private upload, review, "
        "Celery export, download and deletion."
    )


if __name__ == "__main__":
    main()
