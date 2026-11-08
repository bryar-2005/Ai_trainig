import { expect, test } from "@playwright/test";

test("original synthetic video plays with a bidi subtitle overlay and inline edits", async ({ page }) => {
  const email = `video-${Date.now()}-${Math.random().toString(36).slice(2)}@example.com`;
  await page.goto("/");
  const registered = await page.evaluate(async (email) => {
    const response = await fetch("/api/auth/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password: "video-test-password-123", display_name: "Video Reviewer" }) });
    return response.status;
  }, email);
  expect(registered).toBe(201);
  // Generate our own short video in the browser; no external media is used or retained.
  const bytes = await page.evaluate(async () => {
    const canvas = document.createElement("canvas"); canvas.width = 320; canvas.height = 180;
    const ctx = canvas.getContext("2d")!; const stream = canvas.captureStream(12);
    const recorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp8" });
    const chunks: Blob[] = []; recorder.ondataavailable = (event) => chunks.push(event.data);
    const finished = new Promise<Blob>((resolve) => { recorder.onstop = () => resolve(new Blob(chunks, { type: "video/webm" })); });
    recorder.start(); let frame = 0;
    const paint = setInterval(() => { ctx.fillStyle = frame++ % 2 ? "#3d765b" : "#19372a"; ctx.fillRect(0, 0, 320, 180); }, 60);
    await new Promise((resolve) => setTimeout(resolve, 1800)); clearInterval(paint); recorder.stop();
    const output = await finished; stream.getTracks().forEach((track) => track.stop());
    return Array.from(new Uint8Array(await output.arrayBuffer()));
  });
  const subtitles = Buffer.from("1\n00:00:00,200 --> 00:00:01,400\nA line with Mira.\n\n2\n00:00:01,500 --> 00:00:01,750\nThe next line.\n");
  await page.goto("/projects/new");
  await page.getByLabel("Project name").fill("Synthetic video fixture");
  await page.getByLabel("Subtitle file").setInputFiles({ name: "video-test.srt", mimeType: "application/x-subrip", buffer: subtitles });
  await page.getByLabel("Video file (optional)").setInputFiles({ name: "original-canvas.webm", mimeType: "video/webm", buffer: Buffer.from(bytes) });
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("button", { name: "Create draft" }).click();
  await expect(page).toHaveURL(/\/projects\/[0-9a-f-]{36}$/);
  const projectId = page.url().split("/").at(-1)!;
  await page.goto(`/projects/${projectId}/preview`);
  await expect(page.getByLabel("Project video")).toBeVisible();
  await expect.poll(() => page.locator("video").evaluate((video) => (video as HTMLVideoElement).readyState)).toBeGreaterThan(0);
  const text = page.getByLabel("Translation", { exact: true });
  await text.fill("سڵاو Mira");
  await page.getByRole("button", { name: "Save changes" }).click();
  await page.locator("video").evaluate((element) => { const video = element as HTMLVideoElement; video.currentTime = 0.5; video.dispatchEvent(new Event("seeked")); });
  const overlay = page.locator("video + div");
  await expect(overlay).toContainText("سڵاو Mira");
  await expect(overlay.locator("bdi", { hasText: "Mira" })).toHaveAttribute("dir", "ltr");
  await text.fill("گۆڕانکاری نوێ");
  await page.getByRole("button", { name: /#2/ }).click(); // switching cues saves the pending edit
  await expect.poll(async () => page.evaluate(async (id) => (await (await fetch(`/api/projects/${id}/cues`)).json()).items[0].human_text, projectId)).toBe("گۆڕانکاری نوێ");
  await page.goto(`/projects/${projectId}/editor`);
  await expect(page.getByLabel("Translation for cue 1")).toHaveValue("گۆڕانکاری نوێ");
});
