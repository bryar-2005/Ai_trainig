import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, currentUser } from "@/lib/api";

afterEach(() => vi.unstubAllGlobals());

describe("same-origin API client", () => {
  it("sends CSRF on mutations and shares one refresh across simultaneous 401s", async () => {
    document.cookie = "sorani_csrf=old-token; path=/";
    let refreshes = 0; const counts = new Map<string, number>(); const tokens: string[] = [];
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path === "/api/auth/refresh") {
        refreshes++;
        await new Promise((resolve) => setTimeout(resolve, 15));
        document.cookie = "sorani_csrf=new-token; path=/";
        return new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } });
      }
      tokens.push(new Headers(init?.headers).get("X-CSRF-Token") ?? "");
      const count = (counts.get(path) ?? 0) + 1; counts.set(path, count);
      if (count === 1) return new Response('{"detail":"expired"}', { status: 401 });
      return new Response(JSON.stringify({ path }), { status: 200, headers: { "Content-Type": "application/json" } });
    });
    vi.stubGlobal("fetch", fetcher);
    const result = await Promise.all([api<{ path: string }>("/api/one", { method: "POST", body: { x: 1 } }), api<{ path: string }>("/api/two", { method: "POST", body: { x: 2 } })]);
    expect(result.map((item) => item.path)).toEqual(["/api/one", "/api/two"]);
    expect(refreshes).toBe(1);
    expect(tokens.filter((value) => value === "old-token")).toHaveLength(2);
    expect(tokens.filter((value) => value === "new-token")).toHaveLength(2);
  });
  it("does not attempt refresh on an anonymous visit without a CSRF cookie", async () => {
    const fetcher = vi.fn(async () => new Response('{"detail":"not signed in"}', { status: 401 }));
    vi.stubGlobal("fetch", fetcher);
    await expect(currentUser()).resolves.toBeNull();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it("surfaces backend validation details without treating 403 as expiry", async () => {
    const fetcher = vi.fn(async () => new Response('{"detail":"Current password is incorrect"}', { status: 403 }));
    vi.stubGlobal("fetch", fetcher);
    await expect(api("/api/account/password", { method: "POST", body: {} })).rejects.toMatchObject({ status: 403, message: "Current password is incorrect" } satisfies Partial<ApiError>);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});
