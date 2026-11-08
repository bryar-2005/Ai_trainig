import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
export const cn = (...values: ClassValue[]) => twMerge(clsx(values));
export const fmtNumber = (value: number | null | undefined, digits = 0) => value == null ? "—" : new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
export const formatDate = (value: string) => new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(new Date(value));
export function timecode(ms: number): string {
  const n = Math.max(0, Math.floor(ms));
  const h = Math.floor(n / 3_600_000), m = Math.floor(n / 60_000) % 60, s = Math.floor(n / 1000) % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(n % 1000).padStart(3, "0")}`;
}
export function parseTimecode(value: string): number | null {
  const match = /^(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})$/.exec(value.trim());
  if (!match) return null;
  const [, h, m, s, ms] = match.map(Number);
  if (m > 59 || s > 59) return null;
  return ((h * 60 + m) * 60 + s) * 1000 + ms;
}
export const fileSize = (n: number) => n < 1024 ? `${n} B` : n < 1024 ** 2 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 ** 2).toFixed(1)} MB`;
export const directionLabel = (direction: string) => direction === "en-ckb" ? "English → کوردی" : "کوردی → English";
export function safeNext(value: string | null): string {
  if (!value || !value.startsWith("/") || value.startsWith("//") || /[\\\u0000-\u001f\u007f]/.test(value)) return "/dashboard";
  try { return new URL(value, "https://workspace.invalid").origin === "https://workspace.invalid" ? value : "/dashboard"; }
  catch { return "/dashboard"; }
}
