"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, ArrowLeft, BookOpenText, ChevronDown, CircleHelp, Clapperboard, FileCheck2, FileOutput, FolderKanban, Languages, LayoutDashboard, ListVideo, LogOut, Menu, Moon, PanelLeftClose, PanelLeftOpen, Plus, Settings, Sparkles, Sun, Video, X } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { useAppearance, useSession } from "@/components/providers";
import { ErrorNotice, Spinner } from "@/components/ui";
import { api } from "@/lib/api";
import type { Key } from "@/lib/i18n";
import type { Dashboard, Job, Project } from "@/lib/types";
import { useJobStream } from "@/lib/use-job-stream";
import { cn } from "@/lib/utils";

const mainNav = [
  { href: "/dashboard", label: "dashboard", icon: LayoutDashboard },
  { href: "/projects", label: "projects", icon: FolderKanban },
  { href: "/jobs", label: "jobs", icon: Activity },
  { href: "/models", label: "models", icon: Sparkles },
  { href: "/settings", label: "settings", icon: Settings },
] as const;
const projectNav = [
  { suffix: "", label: "overview", icon: LayoutDashboard },
  { suffix: "/editor", label: "editor", icon: ListVideo },
  { suffix: "/preview", label: "preview", icon: Video },
  { suffix: "/glossary", label: "glossary", icon: BookOpenText },
  { suffix: "/qa", label: "qa", icon: FileCheck2 },
  { suffix: "/export", label: "export", icon: FileOutput },
] as const;

export function Brand({ compact = false, light = false }: { compact?: boolean; light?: boolean }) {
  return <Link href="/" aria-label="Sorani Studio home" className={cn("inline-flex items-center gap-2.5", light && "text-white")}><span className={cn("grid h-9 w-9 shrink-0 place-items-center rounded-[10px] text-white", light ? "bg-white/15" : "bg-[#26775b]")}><Clapperboard size={19} strokeWidth={1.8} /></span>{!compact && <span className="leading-tight"><span className="block text-[15px] font-bold tracking-[-.045em]">sorani<span className="font-normal"> studio</span></span><span className={cn("mt-0.5 block text-[9px] font-semibold tracking-[.15em]", light ? "text-white/60" : "text-muted")}>SUBTITLES, WITH CONTEXT</span></span>}</Link>;
}

function Protected({ children }: { children: React.ReactNode }) {
  const { user, pending, error, reload } = useSession();
  const router = useRouter();
  const pathname = usePathname();
  useEffect(() => { if (!pending && !error && !user) router.replace(`/login?next=${encodeURIComponent(pathname)}`); }, [pending, error, user, pathname, router]);
  if (error) return <main className="mx-auto max-w-lg p-10"><ErrorNotice message={error} retry={() => void reload()} /></main>;
  if (pending || !user) return <div className="grid min-h-screen place-items-center text-accent"><Spinner /></div>;
  return children;
}

function ShellContent({ children }: { children: React.ReactNode }) {
  const { t, language, setLanguage, theme, setTheme } = useAppearance();
  const { user } = useSession();
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [mobile, setMobile] = useState(false);
  const [jobsOpen, setJobsOpen] = useState(false);
  const sidebarRef = useRef<HTMLElement>(null); const compact = collapsed && !mobile;
  useEffect(() => {
    if (!mobile) return;
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow; document.body.style.overflow = "hidden";
    const focusable = () => [...(sidebarRef.current?.querySelectorAll<HTMLElement>('a[href], button:not([disabled])') ?? [])].filter((element) => element.getClientRects().length > 0);
    focusable()[0]?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.preventDefault(); setMobile(false); return; }
      if (event.key !== "Tab") return;
      const items = focusable(); if (!items.length) return;
      const first = items[0], last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("keydown", onKey); document.body.style.overflow = previousOverflow; previous?.focus(); };
  }, [mobile]);
  const match = /^\/projects\/([0-9a-f-]{36})(?:\/|$)/i.exec(pathname);
  const projectId = match?.[1];
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api<Project>(`/api/projects/${projectId}`), enabled: !!projectId });
  const summary = useQuery({ queryKey: ["dashboard"], queryFn: () => api<Dashboard>("/api/dashboard/summary"), refetchInterval: 15_000 });
  const toggleCollapse = () => setCollapsed((value) => !value);
  const logout = async () => {
    try { await api("/api/auth/logout", { method: "POST" }); window.location.replace("/login"); }
    catch (error) { toast.error(error instanceof Error ? error.message : "Could not sign out"); }
  };
  return <div className="min-h-screen lg:flex">
    {mobile && <button aria-label="Close navigation" className="fixed inset-0 z-30 bg-black/40 lg:hidden" onClick={() => setMobile(false)} />}
    <aside id="workspace-sidebar" ref={sidebarRef} role={mobile ? "dialog" : undefined} aria-modal={mobile || undefined} aria-label={mobile ? "Navigation" : undefined} className={cn("fixed inset-y-0 start-0 z-40 w-[250px] flex-col border-e border-line bg-[var(--sidebar)] transition-[width,transform] lg:sticky lg:top-0 lg:flex lg:h-screen lg:translate-x-0 rtl:lg:translate-x-0", mobile ? "flex translate-x-0" : "hidden", collapsed && "lg:w-[76px]")}>
      <div className="flex h-[75px] items-center justify-between border-b border-line px-5"><Brand compact={compact} /><button onClick={() => setMobile(false)} className="icon-button lg:hidden" aria-label="Close navigation"><X size={18} /></button></div>
      <div className="flex-1 overflow-y-auto px-3 py-6">
        {!compact && <p className="eyebrow px-3 pb-3">{t("workspace")}</p>}
        <nav className="space-y-1" aria-label="Main navigation">{mainNav.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || (href !== "/dashboard" && pathname.startsWith(`${href}/`));
          return <Link key={href} href={href} onClick={() => setMobile(false)} title={t(label)} className={cn("flex h-10 items-center gap-3 rounded-lg px-3 text-[13px] font-medium transition", active ? "bg-accent-soft font-semibold text-accent" : "text-muted hover:bg-[var(--canvas)] hover:text-ink", collapsed && "lg:justify-center lg:px-0")}><Icon size={18} strokeWidth={1.8} /><span className={cn(collapsed && "lg:hidden")}>{t(label)}</span>{href === "/jobs" && !!summary.data?.active_jobs.length && !compact && <span className="ms-auto rounded-full bg-accent px-1.5 text-[10px] text-white">{summary.data.active_jobs.length}</span>}</Link>;
        })}</nav>
        {!compact && <Link href="/projects/new" onClick={() => setMobile(false)} className="mt-5 flex h-10 items-center justify-center gap-2 rounded-lg border border-dashed border-[#b9d6c4] bg-[#f7fbf8] text-xs font-semibold text-[#347a5d] dark:bg-accent-soft"><Plus size={16} />{t("newProject")}</Link>}
        {projectId && !compact && <div className="mt-8"><p className="eyebrow px-3 pb-3">{t("projectMenu")}</p><div className="mb-2 truncate px-3 text-xs font-semibold text-ink">{project.data?.name ?? "…"}</div><nav className="space-y-1" aria-label="Project navigation">{projectNav.map(({ suffix, label, icon: Icon }) => { const href = `/projects/${projectId}${suffix}`; return <Link key={href} href={href} onClick={() => setMobile(false)} className={cn("flex h-9 items-center gap-3 rounded-lg px-3 text-xs font-medium text-muted hover:bg-[var(--canvas)] hover:text-ink", pathname === href && "bg-accent-soft text-accent")}><Icon size={16} />{t(label)}</Link>; })}</nav></div>}
      </div>
      <div className="border-t border-line p-3"><a href="https://github.com/bryar-2005/Ai_trainig" target="_blank" rel="noreferrer" title={t("help")} className={cn("flex h-10 items-center gap-3 rounded-lg px-3 text-xs text-muted hover:bg-[var(--canvas)]", collapsed && "lg:justify-center lg:px-0")}><CircleHelp size={18} />{!compact && t("help")}</a><button type="button" onClick={toggleCollapse} className="hidden w-full items-center gap-3 rounded-lg px-3 py-2 text-xs text-muted hover:bg-[var(--canvas)] lg:flex" aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>{collapsed ? <PanelLeftOpen size={18} /> : <><PanelLeftClose size={18} />Collapse sidebar</>}</button></div>
    </aside>
    <div className="min-w-0 flex-1" inert={mobile || undefined}><header className="sticky top-0 z-20 flex h-[75px] items-center justify-between border-b border-line bg-[var(--surface)]/95 px-5 backdrop-blur sm:px-8"><div className="flex min-w-0 items-center gap-3"><button onClick={() => setMobile(true)} className="icon-button lg:hidden" aria-label="Open navigation" aria-expanded={mobile} aria-controls="workspace-sidebar"><Menu size={20} /></button>{projectId && <Link href="/projects" className="icon-button hidden sm:inline-flex" aria-label="Back to projects"><ArrowLeft size={17} /></Link>}<div className="truncate text-xs font-medium text-muted"><span className="hidden sm:inline">{t("workspace")} <span className="mx-2 text-line">/</span></span><span className="font-semibold text-ink">{projectId ? project.data?.name ?? t("projects") : t((mainNav.find((item) => pathname.startsWith(item.href))?.label ?? "dashboard") as Key)}</span></div></div>
      <div className="flex items-center gap-1 sm:gap-2"><button type="button" className="icon-button" onClick={() => setLanguage(language === "en" ? "ckb" : "en")} title={t("language")} aria-label={t("language")}><Languages size={18} /></button><button type="button" className="icon-button" onClick={() => setTheme(theme === "light" ? "dark" : "light")} title={t("theme")} aria-label={t("theme")}>{theme === "light" ? <Moon size={18} /> : <Sun size={18} />}</button><details className="relative" onToggle={(event) => setJobsOpen(event.currentTarget.open)}><summary className="icon-button relative cursor-pointer list-none" aria-label={t("activeJobs")}><Activity size={18} />{!!summary.data?.active_jobs.length && <span className="absolute end-1.5 top-1.5 h-2 w-2 rounded-full bg-[#ee9b59] ring-2 ring-[var(--surface)]" />}</summary><div className="absolute end-0 top-12 w-72 rounded-xl border border-line bg-surface p-3 shadow-xl"><div className="flex items-center justify-between px-1 py-2"><span className="text-xs font-semibold">{t("activeJobs")}</span><Link href="/jobs" className="text-[11px] font-semibold text-accent">{t("viewAll")}</Link></div>{summary.data?.active_jobs.length ? summary.data.active_jobs.map((job) => <LiveJob key={job.id} job={job} enabled={jobsOpen} />) : <p className="px-1 py-3 text-xs text-muted">{t("noJobs")}</p>}</div></details><div className="mx-2 hidden h-7 w-px bg-line sm:block" /><details className="relative"><summary className="flex cursor-pointer list-none items-center gap-2 rounded-lg p-1.5 hover:bg-[var(--canvas)]"><span className="grid h-8 w-8 place-items-center rounded-full bg-[#e8d8c6] text-xs font-bold text-[#715641]">{user?.display_name.slice(0, 1).toUpperCase()}</span><span className="hidden max-w-24 truncate text-xs font-semibold sm:block">{user?.display_name}</span><ChevronDown className="hidden text-muted sm:block" size={14} /></summary><div className="absolute end-0 top-12 w-52 rounded-xl border border-line bg-surface p-2 shadow-xl"><div className="border-b border-line px-3 py-2"><p className="truncate text-xs font-semibold">{user?.display_name}</p><p className="truncate text-[11px] text-muted">{user?.email}</p></div><Link href="/settings" className="mt-1 flex items-center gap-2 rounded-lg px-3 py-2 text-xs hover:bg-[var(--canvas)]"><Settings size={15} />{t("settings")}</Link><button type="button" onClick={() => void logout()} className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-start text-xs hover:bg-[var(--canvas)]"><LogOut size={15} />{t("signOut")}</button></div></details></div></header>
      <main className={cn("mx-auto w-full max-w-[1600px] px-5 py-7 sm:px-8 lg:px-10", pathname.endsWith("/editor") && "max-w-none")}>{children}</main>
    </div>
  </div>;
}
function LiveJob({ job, enabled }: { job: Job; enabled: boolean }) {
  const { latest, state } = useJobStream(job.id, enabled);
  const progress = latest?.progress ?? job.progress;
  return <Link href={`/jobs/${job.id}`} className="mt-2 block rounded-lg border border-line p-3 hover:bg-[var(--canvas)]"><div className="flex justify-between text-xs font-semibold"><span className="capitalize">{job.kind.replaceAll("_", " ")}</span><span>{Math.round(progress)}%</span></div><div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[var(--canvas)]"><div className="h-full bg-accent" style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} /></div><p className="mt-2 text-[10px] text-muted">{latest?.message ?? job.stage} · {state}</p></Link>;
}
export function AppShell({ children }: { children: React.ReactNode }) { return <Protected><ShellContent>{children}</ShellContent></Protected>; }
