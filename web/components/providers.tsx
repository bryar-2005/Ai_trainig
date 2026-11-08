"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { QueryClient, QueryClientProvider, useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { Toaster } from "sonner";
import { currentUser } from "@/lib/api";
import { clearDrafts } from "@/lib/draft-store";
import { ckb, en, type Key } from "@/lib/i18n";
import type { User } from "@/lib/types";

type Appearance = { language: "en" | "ckb"; setLanguage: (lang: "en" | "ckb") => void; theme: "light" | "dark"; setTheme: (theme: "light" | "dark") => void; t: (key: Key) => string };
const AppearanceContext = createContext<Appearance | null>(null);
type Session = { user: User | null; pending: boolean; error: string | null; reload: () => Promise<User | null>; setUser: (user: User | null) => void };
const SessionContext = createContext<Session | null>(null);
type Confirmation = { title: string; description: string; confirmLabel: string; destructive?: boolean };
const ConfirmContext = createContext<((options: Confirmation) => Promise<boolean>) | null>(null);
function subscribeAppearance(callback: () => void) { window.addEventListener("storage", callback); window.addEventListener("appearance-changed", callback); return () => { window.removeEventListener("storage", callback); window.removeEventListener("appearance-changed", callback); }; }
function languageSnapshot(): "en" | "ckb" { return localStorage.getItem("sorani-language") === "ckb" ? "ckb" : "en"; }
function themeSnapshot(): "light" | "dark" { return localStorage.getItem("sorani-theme") === "dark" ? "dark" : "light"; }

function Contexts({ children }: { children: React.ReactNode }) {
  const language = useSyncExternalStore<"en" | "ckb">(subscribeAppearance, languageSnapshot, () => "en");
  const theme = useSyncExternalStore<"light" | "dark">(subscribeAppearance, themeSnapshot, () => "light");
  const [user, setUserState] = useState<User | null>(null);
  const [pending, setPending] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0); const identity = useRef<string | null>(null); const client = useQueryClient();
  const acceptUser = useCallback((value: User | null) => { const next = value?.id ?? null; if (identity.current !== next) { client.clear(); clearDrafts(); } identity.current = next; setUserState(value); }, [client]);
  useEffect(() => {
    document.documentElement.lang = language;
    document.documentElement.dir = language === "ckb" ? "rtl" : "ltr";
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [language, theme]);
  const setLanguage = useCallback((value: "en" | "ckb") => { localStorage.setItem("sorani-language", value); window.dispatchEvent(new Event("appearance-changed")); }, []);
  const setTheme = useCallback((value: "light" | "dark") => { localStorage.setItem("sorani-theme", value); window.dispatchEvent(new Event("appearance-changed")); }, []);
  const setUser = useCallback((value: User | null) => { generation.current++; acceptUser(value); setError(null); setPending(false); }, [acceptUser]);
  const reload = useCallback(async () => {
    const request = ++generation.current;
    try { const value = await currentUser(); if (generation.current === request) { acceptUser(value); setError(null); } return value; }
    catch { if (generation.current === request) setError("The API could not be reached."); return null; }
    finally { if (generation.current === request) setPending(false); }
  }, [acceptUser]);
  useEffect(() => {
    let cancelled = false; const request = ++generation.current;
    currentUser().then((value) => { if (!cancelled && generation.current === request) { acceptUser(value); setError(null); } }).catch(() => { if (!cancelled && generation.current === request) setError("The API could not be reached."); }).finally(() => { if (!cancelled && generation.current === request) setPending(false); });
    const expired = () => setUser(null);
    window.addEventListener("session-expired", expired);
    return () => { cancelled = true; window.removeEventListener("session-expired", expired); };
  }, [acceptUser, setUser]);
  const appearance = useMemo(() => ({ language, setLanguage, theme, setTheme, t: (key: Key) => (language === "ckb" ? ckb : en)[key] }), [language, setLanguage, theme, setTheme]);
  const session = useMemo(() => ({ user, pending, error, reload, setUser }), [user, pending, error, reload, setUser]);
  return <AppearanceContext.Provider value={appearance}><SessionContext.Provider value={session}><ConfirmationProvider>{children}</ConfirmationProvider><Toaster position="bottom-right" richColors /></SessionContext.Provider></AppearanceContext.Provider>;
}

function ConfirmationProvider({ children }: { children: React.ReactNode }) {
  const { t } = useAppearance(); const [options, setOptions] = useState<Confirmation | null>(null);
  const resolver = useRef<((accepted: boolean) => void) | null>(null);
  const confirm = useCallback((value: Confirmation) => new Promise<boolean>((resolve) => {
    resolver.current?.(false); resolver.current = resolve; setOptions(value);
  }), []);
  function finish(accepted: boolean) { const current = resolver.current; resolver.current = null; setOptions(null); current?.(accepted); }
  useEffect(() => () => { resolver.current?.(false); resolver.current = null; }, []);
  return <ConfirmContext.Provider value={confirm}>{children}<Dialog.Root open={!!options} onOpenChange={(open) => { if (!open) finish(false); }}><Dialog.Portal><Dialog.Overlay className="fixed inset-0 z-[70] bg-[#14231d]/55 backdrop-blur-sm" /><Dialog.Content className="fixed start-1/2 top-1/2 z-[70] w-[min(94vw,28rem)] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-line bg-surface p-6 text-ink shadow-2xl rtl:translate-x-1/2"><Dialog.Title className="text-lg font-semibold">{options?.title}</Dialog.Title><Dialog.Description className="mt-2 text-sm leading-6 text-muted">{options?.description}</Dialog.Description><div className="mt-6 flex justify-end gap-2"><button type="button" className="btn-secondary" onClick={() => finish(false)}>{t("cancel")}</button><button type="button" className={options?.destructive ? "btn-primary !bg-red-700 hover:!bg-red-800" : "btn-primary"} onClick={() => finish(true)}>{options?.confirmLabel}</button></div></Dialog.Content></Dialog.Portal></Dialog.Root></ConfirmContext.Provider>;
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 10_000, refetchOnWindowFocus: false } } }));
  return <QueryClientProvider client={client}><Contexts>{children}</Contexts></QueryClientProvider>;
}
export function useAppearance() { const value = useContext(AppearanceContext); if (!value) throw new Error("Appearance provider missing"); return value; }
export function useSession() { const value = useContext(SessionContext); if (!value) throw new Error("Session provider missing"); return value; }
export function useConfirm() { const value = useContext(ConfirmContext); if (!value) throw new Error("Confirmation provider missing"); return value; }
