"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, FilePlus2, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { useAppearance, useConfirm } from "@/components/providers";
import { ErrorNotice, Spinner } from "@/components/ui";
import { api, download } from "@/lib/api";
import { formatDate } from "@/lib/utils";

type Snapshot = { id: string; job_id: string; cue_count: number; direction: string | null; created_at: string; expires_at: string; sha256: string };
type Exports = { items: Snapshot[]; nightly_schedule_enabled: boolean; retention_days: number };

export function FeedbackExports() {
  const { t } = useAppearance(); const confirm = useConfirm(); const client = useQueryClient(); const router = useRouter();
  const query = useQuery({ queryKey: ["feedback-exports"], queryFn: () => api<Exports>("/api/feedback/exports"), staleTime: 0 });
  const create = useMutation({ mutationFn: () => api<{ job_id: string }>("/api/feedback/exports", { method: "POST", body: {} }), onSuccess: ({ job_id }) => router.push(`/jobs/${job_id}`), onError: (error) => toast.error(error.message) });
  const remove = useMutation({ mutationFn: (id: string) => api<void>(`/api/feedback/exports/${id}`, { method: "DELETE" }), onSuccess: () => { void client.invalidateQueries({ queryKey: ["feedback-exports"] }); }, onError: (error) => toast.error(error.message) });
  return <section className="panel mt-6 p-5">
    <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="section-title">{t("retainedFeedback")}</h2><p className="mt-1 text-xs leading-5 text-muted">{t("retainedFeedbackCopy")}</p></div><button className="btn-secondary !text-xs" disabled={create.isPending} onClick={() => create.mutate()}>{create.isPending ? <Spinner /> : <FilePlus2 size={15} />}{t("startFeedbackJob")}</button></div>
    {query.error && <div className="mt-4"><ErrorNotice message={query.error.message} /></div>}
    {query.data && <p className="mt-3 text-[11px] leading-5 text-muted">{query.data.nightly_schedule_enabled ? t("nightlyEnabled") : t("nightlyDisabled")} {t("retentionDays")}: {query.data.retention_days}.</p>}
    {query.data && !query.data.items.length ? <p className="mt-4 text-xs text-muted">{t("noFeedbackExports")}</p> : <div className="mt-3 divide-y divide-line">{query.data?.items.map((item) => <div key={item.id} className="flex flex-wrap items-center justify-between gap-3 py-3"><div><p className="text-xs font-semibold">{item.cue_count} {t("approvedHumanEdits")}{item.direction && ` · ${item.direction}`}</p><p className="mt-1 text-[11px] text-muted">{formatDate(item.created_at)} · Expires {formatDate(item.expires_at)} · SHA-256 {item.sha256.slice(0, 12)}…</p></div><div className="flex gap-2"><button className="btn-secondary !min-h-8 !text-xs" onClick={async () => { try { await download(`/api/feedback/exports/${item.id}/download`, "approved-feedback.jsonl"); } catch (error) { toast.error(error instanceof Error ? error.message : "Download failed"); } }}><Download size={14} />{t("download")}</button><button className="btn-secondary !min-h-8 !text-xs" disabled={remove.isPending} onClick={async () => { if (await confirm({ title: t("deleteFeedbackConfirm"), description: t("retainedFeedbackCopy"), confirmLabel: t("deleteFeedback"), destructive: true })) remove.mutate(item.id); }}><Trash2 size={14} />{t("deleteFeedback")}</button></div></div>)}</div>}
  </section>;
}
