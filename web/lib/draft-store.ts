// Unsaved editor text stays only in this tab's memory. A failed save survives
// client-side navigation back to the editor, but never a logout/account change.
const projects = new Map<string, Map<string, string>>();

export function pendingDrafts(projectId: string): Record<string, string> {
  return Object.fromEntries(projects.get(projectId) ?? []);
}
export function rememberDraft(projectId: string, cueId: string, text: string): void {
  const entries = projects.get(projectId) ?? new Map<string, string>();
  entries.set(cueId, text); projects.set(projectId, entries);
}
export function forgetDraft(projectId: string, cueId: string, expected: string): void {
  const entries = projects.get(projectId);
  if (entries?.get(cueId) !== expected) return;
  entries.delete(cueId); if (!entries.size) projects.delete(projectId);
}
export function clearDrafts(): void { projects.clear(); }
