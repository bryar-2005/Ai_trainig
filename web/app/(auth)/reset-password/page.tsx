"use client";
import { KeyRound } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Brand } from "@/components/app-shell";
import { useAppearance, useSession } from "@/components/providers";
import { ErrorNotice, Spinner } from "@/components/ui";
import { api } from "@/lib/api";

export default function ResetPasswordPage() {
  const { t } = useAppearance(); const router = useRouter(); const { setUser } = useSession(); const [token, setToken] = useState(""); const [loaded, setLoaded] = useState(false); const [password, setPassword] = useState(""); const [confirm, setConfirm] = useState(""); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const initialized = useRef(false);
  useEffect(() => { if (initialized.current) return; initialized.current = true; const value = new URLSearchParams(window.location.search).get("token") ?? ""; setToken(value); setLoaded(true); window.history.replaceState(null, "", window.location.pathname); }, []);
  async function submit(event: React.FormEvent) { event.preventDefault(); if (password !== confirm) { setError(t("passwordMismatch")); return; } try { setBusy(true); setError(""); await api("/api/auth/reset-password", { method: "POST", body: { token, new_password: password }, retryAuth: false }); setUser(null); toast.success("Password reset. Please sign in."); router.replace("/login"); } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not reset password."); } finally { setBusy(false); } }
  return <main className="mx-auto flex min-h-screen max-w-md flex-col px-6 py-8"><Brand /><div className="flex flex-1 flex-col justify-center"><span className="grid h-12 w-12 place-items-center rounded-xl bg-accent-soft text-accent"><KeyRound size={22} /></span><h1 className="mt-5 text-3xl font-semibold tracking-tight">{t("choosePassword")}</h1><p className="mt-2 text-sm leading-6 text-muted">{t("passwordResetCopy")}</p>{error && <div className="mt-5"><ErrorNotice message={error} /></div>}{loaded && !token ? <p className="mt-5 text-sm text-muted">{t("missingResetToken")} <Link className="subtle-link" href="/forgot-password">{t("requestNewLink")}</Link></p> : <form className="mt-6 space-y-4" onSubmit={submit}><label className="block"><span className="field-label">{t("newPassword")}</span><input className="field" type="password" minLength={12} maxLength={128} required autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} /></label><label className="block"><span className="field-label">{t("confirmPassword")}</span><input className="field" type="password" minLength={12} maxLength={128} required autoComplete="new-password" value={confirm} onChange={(e) => setConfirm(e.target.value)} /></label><button className="btn-primary w-full" type="submit" disabled={busy || !token}>{busy ? <Spinner /> : t("resetPassword")}</button></form>}</div></main>;
}
