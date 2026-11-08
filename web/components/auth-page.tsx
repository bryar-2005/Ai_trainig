"use client";

import { ArrowRight, Film, Languages, LockKeyhole } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Brand } from "@/components/app-shell";
import { useAppearance, useSession } from "@/components/providers";
import { ErrorNotice, Spinner } from "@/components/ui";
import { api } from "@/lib/api";
import type { User } from "@/lib/types";
import { safeNext } from "@/lib/utils";

export function AuthPage({ mode }: { mode: "login" | "register" }) {
  const { t, language, setLanguage } = useAppearance();
  const { user, pending, setUser } = useSession();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const register = mode === "register";
  useEffect(() => { if (!pending && user) router.replace(safeNext(new URLSearchParams(window.location.search).get("next"))); }, [pending, user, router]);
  async function submit(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setError("");
    try {
      const user = await api<User>(`/api/auth/${mode}`, { method: "POST", body: register ? { email, password, display_name: name } : { email, password }, retryAuth: false });
      setUser(user);
      toast.success(register ? t("openWorkspace") : t("welcome"));
      const next = safeNext(new URLSearchParams(window.location.search).get("next"));
      router.replace(next); router.refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not sign in."); }
    finally { setBusy(false); }
  }
  return <main className="min-h-screen bg-surface lg:grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]"><div className="flex min-h-screen flex-col px-6 py-7 sm:px-12 lg:px-16"><div className="flex items-center justify-between"><Brand /><button type="button" className="icon-button" onClick={() => setLanguage(language === "en" ? "ckb" : "en")} aria-label={t("language")}><Languages size={18} /></button></div><div className="mx-auto flex w-full max-w-sm flex-1 flex-col justify-center py-16"><p className="eyebrow mb-3">SORANI STUDIO / {register ? t("register") : t("welcome")}</p><h1 className="text-[32px] font-semibold tracking-[-.045em]">{register ? t("register") : t("signIn")}</h1><p className="mt-2 text-sm leading-6 text-muted">{register ? t("accountCopy") : t("signinCopy")}</p><form onSubmit={submit} className="mt-8 space-y-4">{error && <ErrorNotice message={error} />}{register && <label className="block"><span className="field-label">{t("displayName")}</span><input className="field" required maxLength={120} autoComplete="name" value={name} onChange={(e) => setName(e.target.value)} /></label>}<label className="block"><span className="field-label">{t("email")}</span><input className="field" type="email" required autoComplete="email" dir="ltr" value={email} onChange={(e) => setEmail(e.target.value)} /></label><label className="block"><span className="field-label">{t("password")}</span><input className="field" type="password" aria-label={t("password")} required minLength={register ? 12 : 1} maxLength={128} autoComplete={register ? "new-password" : "current-password"} value={password} onChange={(e) => setPassword(e.target.value)} />{register && <span className="mt-1.5 block text-[11px] text-muted">{t("passwordLength")}</span>}</label>{!register && <div className="text-end"><Link href="/forgot-password" className="text-xs font-semibold text-accent">{t("forgotPassword")}</Link></div>}<button type="submit" disabled={busy} className="btn-primary w-full !py-3">{busy ? <Spinner /> : <>{register ? t("register") : t("signIn")} <ArrowRight size={16} /></>}</button></form><p className="mt-6 text-center text-xs text-muted">{register ? t("alreadyAccount") : t("needAccount")} <Link className="subtle-link" href={register ? "/login" : "/register"}>{register ? t("signIn") : t("register")}</Link></p><p className="mt-8 flex items-center justify-center gap-2 text-[11px] text-muted"><LockKeyhole size={13} /> {t("privateFiles")}</p></div><p className="text-xs text-muted">© Sorani Studio <span className="mx-2">·</span> Built for stories worth sharing.</p></div><aside className="film-grain relative hidden min-h-screen flex-col justify-end overflow-hidden p-14 text-white lg:flex"><div className="absolute end-14 top-12 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[.18em] text-white/70"><Film size={16} /> {t("authSideTag")}</div><div className="relative max-w-xl"><p className="mb-5 font-mono text-xs text-[#b5d7bf]">01:42:08.000 ─────── 01:42:11.500</p><p className="text-4xl font-medium leading-tight tracking-[-.045em]">“Some stories stay<br />with us.”</p><p dir="rtl" className="font-arabic mt-5 text-3xl text-[#c9e4ce]">هەندێک چیرۆک لەگەڵمان دەمێننەوە.</p><div className="mt-9 h-px w-full bg-white/25" /><p className="mt-5 text-xs leading-6 text-white/60">{t("authSideCopy")}</p></div></aside></main>;
}
