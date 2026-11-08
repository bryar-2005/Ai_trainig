"use client";
import { ArrowLeft, Mail } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { Brand } from "@/components/app-shell";
import { useAppearance } from "@/components/providers";
import { ErrorNotice, Spinner } from "@/components/ui";
import { api } from "@/lib/api";

export default function ForgotPasswordPage() {
  const { t } = useAppearance(); const [email, setEmail] = useState(""); const [sent, setSent] = useState(false); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  async function submit(event: React.FormEvent) { event.preventDefault(); try { setBusy(true); setError(""); await api("/api/auth/forgot-password", { method: "POST", body: { email }, retryAuth: false }); setSent(true); } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not request a reset."); } finally { setBusy(false); } }
  return <main className="mx-auto flex min-h-screen max-w-md flex-col px-6 py-8"><Brand /><div className="flex flex-1 flex-col justify-center"><span className="grid h-12 w-12 place-items-center rounded-xl bg-accent-soft text-accent"><Mail size={22} /></span><h1 className="mt-5 text-3xl font-semibold tracking-tight">{t("resetTitle")}</h1><p className="mt-2 text-sm leading-6 text-muted">{t("resetRequestCopy")}</p>{sent ? <div className="mt-6 rounded-lg bg-accent-soft p-4 text-sm leading-6 text-accent">{t("resetSent")}</div> : <form className="mt-6 space-y-4" onSubmit={submit}>{error && <ErrorNotice message={error} />}<label className="block"><span className="field-label">{t("email")}</span><input className="field" type="email" required autoComplete="email" dir="ltr" value={email} onChange={(e) => setEmail(e.target.value)} /></label><button className="btn-primary w-full" disabled={busy} type="submit">{busy ? <Spinner /> : t("sendReset")}</button></form>}<Link href="/login" className="mt-6 inline-flex items-center gap-2 text-xs font-semibold text-accent"><ArrowLeft size={14} />{t("signIn")}</Link></div></main>;
}
