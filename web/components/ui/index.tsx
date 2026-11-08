"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { AlertCircle, ArrowUpRight, LoaderCircle, X } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";
import { useAppearance } from "@/components/providers";
import type { Key } from "@/lib/i18n";
import { cn } from "@/lib/utils";

export function PageHeader({ eyebrow, title, description, action }: { eyebrow?: string; title: string; description?: string; action?: ReactNode }) {
  return <div className="mb-7 flex flex-wrap items-end justify-between gap-4"><div>{eyebrow && <p className="eyebrow mb-2">{eyebrow}</p>}<h1 className="page-title">{title}</h1>{description && <p className="page-copy">{description}</p>}</div>{action}</div>;
}
export function Status({ value }: { value: string }) {
  const { t } = useAppearance();
  const palette: Record<string, string> = { completed: "bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300", approved: "bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300", translating: "bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300", running: "bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300", needs_review: "bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300", edited: "bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300", failed: "bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-300" };
  return <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold", palette[value] ?? "bg-[var(--canvas)] text-muted")}><span className="h-1.5 w-1.5 rounded-full bg-current" />{t(value as Key) || value.replaceAll("_", " ")}</span>;
}
export function EmptyState({ icon, title, description, action }: { icon: ReactNode; title: string; description?: string; action?: ReactNode }) {
  return <div className="flex min-h-48 flex-col items-center justify-center px-6 py-10 text-center"><div className="mb-4 grid h-12 w-12 place-items-center rounded-2xl bg-accent-soft text-accent">{icon}</div><p className="text-sm font-semibold">{title}</p>{description && <p className="mt-1 max-w-sm text-xs leading-5 text-muted">{description}</p>}{action && <div className="mt-4">{action}</div>}</div>;
}
export function ErrorNotice({ message, retry }: { message: string; retry?: () => void }) {
  return <div role="alert" className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/30 dark:text-red-200"><AlertCircle size={17} className="mt-0.5 shrink-0" /><div className="flex-1">{message}</div>{retry && <button type="button" onClick={retry} className="font-semibold underline">Retry</button>}</div>;
}
export function Spinner({ className }: { className?: string }) { return <LoaderCircle aria-label="Loading" className={cn("animate-spin", className)} size={18} />; }
export function InlineLink({ href, children }: { href: string; children: ReactNode }) { return <Link href={href} className="inline-flex items-center gap-1 text-xs font-semibold text-accent hover:opacity-75">{children}<ArrowUpRight size={13} /></Link>; }
export function Modal({ open, onOpenChange, title, description, children }: { open: boolean; onOpenChange: (open: boolean) => void; title: string; description?: string; children: ReactNode }) {
  return <Dialog.Root open={open} onOpenChange={onOpenChange}><Dialog.Portal><Dialog.Overlay className="fixed inset-0 z-50 bg-[#14231d]/55 backdrop-blur-sm" /><Dialog.Content className="fixed start-1/2 top-1/2 z-50 max-h-[90vh] w-[min(94vw,32rem)] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-2xl border border-line bg-surface p-6 text-ink shadow-2xl rtl:translate-x-1/2"><div className="mb-5 flex items-start justify-between gap-4"><div><Dialog.Title className="text-lg font-semibold tracking-tight">{title}</Dialog.Title>{description && <Dialog.Description className="mt-1 text-sm leading-6 text-muted">{description}</Dialog.Description>}</div><Dialog.Close className="icon-button" aria-label="Close"><X size={18} /></Dialog.Close></div>{children}</Dialog.Content></Dialog.Portal></Dialog.Root>;
}
