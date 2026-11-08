import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobStream } from "@/lib/use-job-stream";

const auth = vi.hoisted(() => ({ refreshSession: vi.fn(), api: vi.fn() }));
vi.mock("@/lib/api", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/lib/api")>()), ...auth }));
class FakeEventSource {
  static CLOSED = 2;
  static instances: FakeEventSource[] = [];
  readyState = 1; onopen: (() => void) | null = null; onerror: (() => void) | null = null;
  listeners = new Map<string, (event: MessageEvent<string>) => void>();
  constructor(public url: string) { FakeEventSource.instances.push(this); }
  addEventListener(name: string, fn: (event: MessageEvent<string>) => void) { this.listeners.set(name, fn); }
  close() { this.readyState = FakeEventSource.CLOSED; }
  send(name: string, data = "{}", id = "") { this.listeners.get(name)?.(new MessageEvent(name, { data, lastEventId: id })); }
}
const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{children}</QueryClientProvider>;
beforeEach(() => { FakeEventSource.instances = []; vi.stubGlobal("EventSource", FakeEventSource); });
afterEach(() => vi.unstubAllGlobals());

describe("durable job progress", () => {
  it("resumes after session refresh without replaying prior events", async () => {
    auth.refreshSession.mockResolvedValue(true);
    const { result, unmount } = renderHook(() => useJobStream("job-1"), { wrapper });
    const initial = FakeEventSource.instances[0];
    act(() => { initial.onopen?.(); initial.send("progress", '{"stage":"loading","progress":5}', "12"); });
    expect(result.current.state).toBe("live");
    expect(result.current.events).toHaveLength(1);
    act(() => initial.send("auth_expired"));
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(2));
    const resumed = FakeEventSource.instances[1];
    expect(resumed.url).toBe("/api/jobs/job-1/stream?after=12");
    act(() => { resumed.send("progress", '{"stage":"loading","progress":5}', "12"); resumed.send("progress", '{"stage":"translation","progress":50}', "13"); resumed.send("done"); });
    expect(result.current.events).toHaveLength(2);
    expect(result.current.latest?.progress).toBe(50);
    expect(result.current.state).toBe("finished");
    expect(resumed.readyState).toBe(FakeEventSource.CLOSED);
    unmount();
  });
  it("checks authentication when an initial EventSource response closes", async () => {
    auth.api.mockResolvedValue({ status: "running" });
    const { result, unmount } = renderHook(() => useJobStream("job-2"), { wrapper });
    const initial = FakeEventSource.instances[0];
    act(() => { initial.close(); initial.onerror?.(); });
    expect(result.current.state).toBe("reconnecting");
    await waitFor(() => expect(auth.api).toHaveBeenCalledWith("/api/jobs/job-2"));
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(2), { timeout: 2000 });
    unmount();
    expect(FakeEventSource.instances[1].readyState).toBe(FakeEventSource.CLOSED);
  });
});
