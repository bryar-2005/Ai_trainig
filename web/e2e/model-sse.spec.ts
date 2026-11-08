import { expect, test } from "@playwright/test";

test("real synthetic model sample and streamed worker progress", async ({ page }) => {
  test.skip(!process.env.E2E_SYNTHETIC_MODEL, "Requires an explicitly configured disposable random model and worker");
  test.setTimeout(180_000);
  const email = `model-${Date.now()}-${Math.random().toString(36).slice(2)}@example.com`;
  await page.goto("/");
  const status = await page.evaluate(async (email) => (await fetch("/api/auth/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password: "model-test-password-123", display_name: "Model Reviewer" }) })).status, email);
  expect(status).toBe(201);
  await page.reload();
  await page.getByLabel("Model", { exact: true }).selectOption({ label: process.env.E2E_SYNTHETIC_MODEL! });
  await page.getByLabel("Your lines · one per line, up to three").fill("The door is open.\nWe should go home.");
  const submitted = page.waitForResponse((response) => response.url().endsWith("/api/models/sample") && response.request().method() === "POST");
  await page.getByRole("button", { name: "Translate sample" }).click();
  const response = await submitted;
  expect(response.status()).toBe(202);
  const { job_id: jobId } = await response.json() as { job_id: string };
  await page.goto(`/jobs/${jobId}`);
  await expect(page.getByRole("heading", { name: "Live progress" })).toBeVisible();
  // Worker event messages are rendered from the real SSE stream, not from
  // the polling endpoint. A separate test verifies an open stream while queued.
  await expect(page.getByText("Loading the selected model").first()).toBeVisible({ timeout: 60_000 });
  await expect.poll(async () => page.evaluate(async (id) => (await (await fetch(`/api/jobs/${id}`)).json()).status, jobId), { timeout: 150_000, intervals: [1000] }).toBe("completed");
  await expect(page.getByText("Translated the sample").first()).toBeVisible({ timeout: 30_000 });
  const result = await page.evaluate(async (id) => (await (await fetch(`/api/jobs/${id}`)).json()).result, jobId);
  expect(result.outputs).toHaveLength(2);
  expect(result.model.warnings.join(" ").toLowerCase()).toContain("random");
  await page.goto("/#demo");
  await expect(page.getByLabel("Model", { exact: true })).toBeVisible();
});
