import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";

// All dialogue below is an original synthetic fixture. No model is required.
test("private feedback snapshot is queued, downloaded and deleted", async ({ page }) => {
  test.skip(!process.env.E2E_FEEDBACK_WORKER, "Requires a real Celery worker on the test broker");
  const email = `feedback-${Date.now()}-${Math.random().toString(36).slice(2)}@example.com`;
  await page.goto("/");
  const fixture = await page.evaluate(async (email) => {
    const csrf = () => decodeURIComponent(document.cookie.split("; ").find((v) => v.startsWith("sorani_csrf="))?.split("=").slice(1).join("=") ?? "");
    async function json(path: string, method: string, payload: unknown) {
      const response = await fetch(path, { method, headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf() }, body: JSON.stringify(payload) });
      if (!response.ok) throw new Error(`${path}: ${response.status}`);
      return response.json();
    }
    await json("/api/auth/register", "POST", { email, password: "feedback-test-password-123", display_name: "Feedback Reviewer" });
    const project = await json("/api/projects", "POST", { name: "Synthetic feedback fixture", direction: "en-ckb" });
    const form = new FormData();
    form.set("subtitle", new File(["1\n00:00:01,000 --> 00:00:03,000\nThe lantern is lit.\n"], "synthetic.srt", { type: "application/x-subrip" }));
    const upload = await fetch(`/api/projects/${project.id}/upload`, { method: "POST", headers: { "X-CSRF-Token": csrf() }, body: form });
    if (!upload.ok) throw new Error(`upload: ${upload.status}`);
    const cues = await (await fetch(`/api/projects/${project.id}/cues`)).json();
    const cue = await json(`/api/cues/${cues.items[0].id}`, "PATCH", { expected_revision: cues.items[0].revision, human_text: "چرا هەڵگیرساوە." });
    await json(`/api/cues/${cue.id}/approve`, "POST", { expected_revision: cue.revision });
    return { projectId: project.id };
  }, email);
  await page.goto("/settings");
  await page.getByLabel("Create private nightly feedback snapshots").check();
  await page.locator("section").filter({ has: page.getByRole("heading", { name: "Subtitle preferences" }) }).getByRole("button", { name: "Save changes" }).click();
  await expect.poll(async () => page.evaluate(async () => (await (await fetch("/api/auth/me")).json()).preferences.nightly_feedback_export)).toBe(true);
  await page.goto("/models");
  await expect(page.getByText("Nightly scheduling is disabled by the operator.")).toBeVisible();
  await page.getByRole("button", { name: "Create snapshot job" }).click();
  await expect(page).toHaveURL(/\/jobs\/[0-9a-f-]{36}/);
  const jobId = page.url().split("/").at(-1)!;
  await expect.poll(async () => page.evaluate(async (id) => (await (await fetch(`/api/jobs/${id}`)).json()).status, jobId), { timeout: 60_000 }).toBe("completed");
  await expect(page.getByRole("link", { name: "Private feedback snapshots" })).toBeVisible();
  await page.getByRole("link", { name: "Private feedback snapshots" }).click();
  await expect(page).toHaveURL(/\/models/);
  const panel = page.locator("section").filter({ has: page.getByRole("heading", { name: "Private feedback snapshots" }) });
  await expect(panel.getByText(/1 approved human edits/i)).toBeVisible();
  const pending = page.waitForEvent("download");
  await panel.getByRole("button", { name: "Download" }).click();
  const file = await pending;
  const record = JSON.parse(await readFile((await file.path())!, "utf8"));
  expect(record.human_text).toBe("چرا هەڵگیرساوە.");
  expect(record.language_verified).toBe(false);
  await panel.getByRole("button", { name: "Delete snapshot" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete snapshot" }).click();
  await expect(panel.getByText("No retained snapshots yet.")).toBeVisible();
  expect(fixture.projectId).toBeTruthy();
});
