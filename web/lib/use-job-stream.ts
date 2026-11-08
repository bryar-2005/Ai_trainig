"use client";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, ApiError, refreshSession } from "@/lib/api";
import type { JobEvent } from "@/lib/types";

type StreamState = "connecting" | "live" | "reconnecting" | "finished";
type Snapshot = { jobId: string; events: JobEvent[]; state: StreamState };

/** Durable event IDs support both native reconnects and reconnection after session refresh. */
export function useJobStream(jobId: string, enabled = true) {
  const client = useQueryClient();
  const [snapshot, setSnapshot] = useState<Snapshot>({ jobId, events: [], state: "connecting" });
  const current: Snapshot = snapshot.jobId === jobId ? snapshot : { jobId, events: [], state: "connecting" };
  useEffect(() => {
    if (!enabled) return;
    let stopped = false; let source: EventSource | null = null; let timer: ReturnType<typeof setTimeout> | undefined;
    let invalidateTimer: ReturnType<typeof setTimeout> | undefined; let lastId = ""; const seen = new Set<string>();
    const update = (fn: (value: Snapshot) => Snapshot) => setSnapshot((value) => fn(value.jobId === jobId ? value : { jobId, events: [], state: "connecting" }));
    const status = (state: StreamState) => update((value) => ({ ...value, state }));
    const invalidate = () => { void client.invalidateQueries({ queryKey: ["job", jobId] }); };
    function schedule(delay: number) { clearTimeout(timer); timer = setTimeout(connect, delay); }
    function connect() {
      if (stopped) return;
      const cursor = lastId ? `?after=${encodeURIComponent(lastId)}` : "";
      const stream = new EventSource(`/api/jobs/${jobId}/stream${cursor}`, { withCredentials: true });
      source = stream;
      stream.onopen = () => { if (!stopped) status("live"); };
      stream.onerror = () => {
        if (stopped || source !== stream) return;
        status("reconnecting");
        // Network interruptions normally reconnect natively, preserving Last-Event-ID.
        // A 401 closes EventSource permanently, so use the API client to refresh
        // and explicitly reconnect. Other terminal client errors stop retrying.
        if (stream.readyState === EventSource.CLOSED) {
          void api(`/api/jobs/${jobId}`).then(() => { if (!stopped) schedule(500); }).catch((error: unknown) => {
            if (stopped) return;
            if (error instanceof ApiError && [401, 403, 404].includes(error.status)) { status("finished"); return; }
            schedule(3000);
          });
        }
      };
      stream.addEventListener("progress", (message: MessageEvent<string>) => {
        try {
          if (message.lastEventId && seen.has(message.lastEventId)) return;
          const event = JSON.parse(message.data) as JobEvent;
          if (message.lastEventId) { seen.add(message.lastEventId); lastId = message.lastEventId; }
          update((value) => ({ ...value, events: [...value.events.slice(-499), event] }));
          if (!invalidateTimer) invalidateTimer = setTimeout(() => { invalidateTimer = undefined; invalidate(); }, 200);
        } catch { /* A malformed event must not break the stream. */ }
      });
      stream.addEventListener("done", () => {
        stream.close(); if (stopped) return;
        clearTimeout(invalidateTimer); status("finished"); invalidate();
        void client.invalidateQueries({ queryKey: ["jobs"] }); void client.invalidateQueries({ queryKey: ["dashboard"] });
        void client.invalidateQueries({ queryKey: ["feedback-exports"] }); void client.invalidateQueries({ queryKey: ["models"] });
      });
      stream.addEventListener("auth_expired", () => {
        stream.close(); if (stopped) return; status("reconnecting");
        void refreshSession().then((ok) => { if (!stopped && ok) connect(); else if (!stopped) window.dispatchEvent(new Event("session-expired")); });
      });
    }
    connect();
    return () => { stopped = true; clearTimeout(timer); clearTimeout(invalidateTimer); source?.close(); };
  }, [client, enabled, jobId]);
  return { events: current.events, state: current.state, latest: current.events.at(-1) };
}
