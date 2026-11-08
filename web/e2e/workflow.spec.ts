import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";

// Original synthetic fixture, created specifically for this test; no subtitle corpus is required.
const subtitle = Buffer.from(`1\n00:00:01,000 --> 00:00:03,000\nMira, can you hear me?\n\n2\n00:00:03,500 --> 00:00:06,000\nThe path is quiet tonight.\n\n3\n00:00:07,000 --> 00:00:09,000\nWe should go home.\n`, "utf8");

test("registration, authorized upload, review, glossary, export, settings and login", async ({ page }) => {
  const email = `browser-${Date.now()}-${Math.random().toString(36).slice(2)}@example.com`;
  const password = "browser-test-password-123";
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("/dashboard");
  await expect(page).toHaveURL(/\/login\?next=/);
  await page.getByRole("link", { name: "Create account" }).click();
  await page.getByLabel("Display name").fill("Browser Reviewer");
  await page.getByLabel("Email address").fill(email);
  await page.getByLabel("Password", { exact: true }).fill(password);
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page).toHaveURL(/\/dashboard/);
  await expect(page.getByRole("heading", { name: /Welcome back, Browser/ })).toBeVisible();

  await page.locator('input[type="file"]').first().setInputFiles({ name: "original-test.srt", mimeType: "application/x-subrip", buffer: subtitle });
  await expect(page).toHaveURL(/\/projects\/new/);
  await page.getByLabel("Project name").fill("Original browser fixture");
  await expect(page.getByText(/3 cues · duration 00:00:09\.000/)).toBeVisible();
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByText("Configure translation")).toBeVisible();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "+ Add name" }).click();
  await page.getByLabel("Character 1 English").fill("Mira");
  await page.getByLabel("Character 1 Sorani").fill("میرا");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Create draft" }).click();
  await expect(page).toHaveURL(/\/projects\/[0-9a-f-]{36}$/);
  const projectId = page.url().split("/").at(-1)!;
  await expect(page.getByRole("heading", { name: "Original browser fixture" })).toBeVisible();
  await expect(page.getByText("Subtitle lines")).toBeVisible();

  await page.getByRole("link", { name: "Open editor" }).first().click();
  await expect(page).toHaveURL(new RegExp(`/projects/${projectId}/editor`));
  const edit = page.getByLabel("Translation for cue 1");
  await edit.fill("میرا، گوێم لێ دەگریت؟");
  await expect.poll(async () => page.evaluate(async (id) => {
    const response = await fetch(`/api/projects/${id}/cues`);
    return (await response.json()).items[0].human_text;
  }, projectId)).toBe("میرا، گوێم لێ دەگریت؟");
  await page.getByRole("button", { name: "Approve cue 1" }).click();
  await expect.poll(async () => page.evaluate(async (id) => {
    const response = await fetch(`/api/projects/${id}/cues`);
    return (await response.json()).items[0].status;
  }, projectId)).toBe("approved");

  await page.goto(`/projects/${projectId}/glossary`);
  await expect(page.getByRole("cell", { name: "Mira", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Re-check script" }).click();
  await expect(page.getByText("3 cues rechecked.")).toBeVisible();
  await page.getByRole("button", { name: "Add term" }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByLabel("English").fill("home");
  await dialog.getByLabel("Sorani").fill("ماڵ");
  await dialog.getByRole("button", { name: "Save changes" }).click();
  await expect(page.getByRole("cell", { name: "home", exact: true })).toBeVisible();

  await page.goto(`/projects/${projectId}/qa`);
  await expect(page.getByRole("heading", { name: "Quality assurance" })).toBeVisible();
  await page.getByRole("link", { name: /Open in editor/ }).first().click();
  await expect(page).toHaveURL(/editor\?cueId=/);

  await page.goto(`/projects/${projectId}/export`);
  await page.getByLabel("Approved cues only").check();
  await expect(page.getByText("Machine-assisted", { exact: false }).first()).toBeVisible();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download" }).click();
  const file = await downloadPromise;
  const output = await readFile((await file.path())!, "utf8");
  expect(output).toContain("Machine-assisted translation");
  expect(output).toContain("00:00:01,000 --> 00:00:03,000");
  expect(output).toContain("میرا");
  expect(output).not.toContain("00:00:03,500");

  await page.goto(`/projects/${projectId}/preview`);
  await expect(page.getByText("No video was uploaded for this project.")).toBeVisible();
  await page.getByRole("button", { name: "+100 ms" }).click();
  await expect.poll(async () => page.evaluate(async (id) => {
    const response = await fetch(`/api/projects/${id}/cues`);
    return (await response.json()).items[0].start_ms;
  }, projectId)).toBe(1100);

  await page.goto("/settings");
  const name = page.getByLabel("Display name");
  await name.fill("Updated Reviewer");
  await page.getByRole("button", { name: "Save changes" }).first().click();
  await expect(page.getByText("Profile updated")).toBeVisible();
  await page.getByRole("button", { name: "Create token" }).click();
  const tokenDialog = page.getByRole("dialog");
  await tokenDialog.getByLabel("Token name").fill("Browser test token");
  await tokenDialog.getByLabel("Current password").fill(password);
  await tokenDialog.getByRole("button", { name: "Create", exact: true }).click();
  await expect(tokenDialog.getByText(/sst_/)).toBeVisible();
  await tokenDialog.getByRole("button", { name: "Done" }).click();
  await expect(page.getByText("Browser test token")).toBeVisible();
  await page.getByRole("button", { name: "Revoke" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Revoke token" }).click();
  await expect(page.getByText(/Revoked/)).toBeVisible();

  await page.getByRole("combobox", { name: "Language" }).selectOption("ckb");
  await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
  await page.getByRole("combobox", { name: "زمان" }).selectOption("en");
  await expect(page.locator("html")).toHaveAttribute("dir", "ltr");
  await page.getByRole("combobox", { name: "Theme" }).selectOption("dark");
  await expect(page.locator("html")).toHaveClass(/dark/);
  await page.getByRole("combobox", { name: "Theme" }).selectOption("light");

  await page.locator("header summary").last().click();
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page).toHaveURL(/\/login/);
  await page.getByLabel("Email address").fill(email);
  await page.getByLabel("Password", { exact: true }).fill(password);
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page).toHaveURL(/\/dashboard/);
  expect(errors).toEqual([]);
});
