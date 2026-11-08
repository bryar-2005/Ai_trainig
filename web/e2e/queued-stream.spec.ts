import { expect, test } from "@playwright/test";

test("an open proxied event stream delivers queued progress before cancellation", async ({ page }) => {
  test.skip(!process.env.E2E_EXPECT_QUEUED || !process.env.E2E_SYNTHETIC_MODEL, "Requires a disposable model and a deliberately stopped worker");
  const email = `stream-${Date.now()}-${Math.random().toString(36).slice(2)}@example.com`;
  await page.goto("/");
  expect(await page.evaluate(async (email) => (await fetch("/api/auth/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password: "stream-test-password-123", display_name: "Stream Reviewer" }) })).status, email)).toBe(201);
  await page.reload();
  await page.getByLabel("Model", { exact: true }).selectOption({ label: process.env.E2E_SYNTHETIC_MODEL! });
  await page.getByLabel("Your lines · one per line, up to three").fill("An original test line.");
  const response = page.waitForResponse((item) => item.url().endsWith("/api/models/sample") && item.request().method() === "POST");
  await page.getByRole("button", { name: "Translate sample" }).click();
  const body = await (await response).json() as { job_id: string };
  await page.goto(`/jobs/${body.job_id}`);
  await expect(page.getByRole("heading", { name: "Live progress" })).toBeVisible();
  await expect(page.getByText("Sample queued").first()).toBeVisible();
  expect(await page.evaluate(async (id) => (await (await fetch(`/api/jobs/${id}`)).json()).status, body.job_id)).toBe("queued");
  await page.getByRole("button", { name: "Cancel job" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Cancel job" }).click();
  await expect(page.getByText("Cancellation requested").first()).toBeVisible();
  await expect.poll(() => page.evaluate(async (id) => (await (await fetch(`/api/jobs/${id}`)).json()).status, body.job_id)).toBe("cancelled");
});
