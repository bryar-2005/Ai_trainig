import type { ReactNode } from "react";
import { api } from "@/lib/api";
import type { Cue, Direction, Page } from "@/lib/types";

/** Isolate LTR words and numerals in displayed Sorani text without changing stored text. */
export function BidiText({ text, direction }: { text: string; direction: "ltr" | "rtl" }) {
  if (direction === "ltr") return <span dir="ltr" className="whitespace-pre-wrap">{text}</span>;
  const pieces = text.split(/([A-Za-z0-9][A-Za-z0-9\-_.:/@%+]*)/g);
  return <span dir="rtl" className="font-arabic whitespace-pre-wrap">{pieces.map((piece, index): ReactNode => /^[A-Za-z0-9]/.test(piece) ? <bdi dir="ltr" key={index}>{piece}</bdi> : piece)}</span>;
}
export const sourceDir = (direction: Direction) => direction === "en-ckb" ? "ltr" : "rtl";
export const targetDir = (direction: Direction) => direction === "en-ckb" ? "rtl" : "ltr";
export function visibleCharacters(text: string): number {
  // Match backend QA: grapheme clusters, including spaces but excluding
  // line breaks and direction/control marks; preserve joiners within clusters.
  const clean = text.replace(/[\p{Cc}\p{Cf}\p{Zl}\p{Zp}]/gu, (char) => char === "\u200c" || char === "\u200d" ? char : "");
  if (typeof Intl.Segmenter === "function") {
    return [...new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(clean)].filter(({ segment }) => !/^\p{Cf}+$/u.test(segment)).length;
  }
  return [...clean].filter((char) => !/^[\p{Cf}\p{M}]$/u.test(char)).length;
}
export const readingSpeed = (text: string, start: number, end: number) => end > start ? visibleCharacters(text) / ((end - start) / 1000) : 0;

/** Page through authorized cues; never assume a project fits in one response. */
export async function allCues(projectId: string): Promise<Cue[]> {
  const result: Cue[] = [];
  let page = 1; let total = Infinity;
  while (result.length < total) {
    const chunk = await api<Page<Cue>>(`/api/projects/${projectId}/cues?page=${page}&page_size=500`);
    result.push(...chunk.items); total = chunk.total;
    if (!chunk.items.length) break;
    page++;
  }
  return result;
}
