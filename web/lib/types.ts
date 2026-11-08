export type Direction = "en-ckb" | "ckb-en";
export type ProjectStatus = "draft" | "translating" | "needs_review" | "completed" | "failed";
export type CueStatus = "unreviewed" | "edited" | "approved";
export interface User { id: string; email: string; display_name: string; role: "admin" | "reviewer" | "user"; preferences: Record<string, unknown>; }
export interface Media { id: string; kind: "subtitle" | "video"; original_name: string; size_bytes: number; content_type: string; sha256: string; }
export interface Project {
  id: string; name: string; description: string; direction: Direction; status: ProjectStatus;
  model_version_id: string | null; target_cps: number; hard_cps: number; max_chars: number;
  max_lines: number; register: "auto" | "formal" | "informal"; soften_profanity: boolean;
  profanity_replacements: { source: string; replacement: string }[];
  created_at: string; updated_at: string; media?: Media[]; cue_count?: number; average_confidence?: number | null;
}
export interface Cue {
  id: string; project_id: string; cue_index: number; start_ms: number; end_ms: number;
  original_start_ms: number; original_end_ms: number; source_text: string; normalized_source_text: string;
  speaker: string | null; speaker_labels: string[]; sound_effects: string[]; source_cue_ids: number[];
  sentence_source_cue_ids: number[]; machine_text: string; human_text: string | null; effective_text: string;
  model_version_id: string | null; confidence_score: number | null; cps: number; line_lengths: number[];
  glossary_mismatches: string[]; qa_warnings: string[]; flagged: boolean; status: CueStatus; revision: number;
}
export interface Revision { id: string; revision: number; reason: string; before: Record<string, unknown>; after: Record<string, unknown>; created_at: string; }
export interface GlossaryTerm { id: string; project_id: string; english: string; sorani: string; category: string; locked: boolean; origin: string; }
export interface Job { id: string; project_id: string | null; kind: string; status: "queued" | "running" | "cancel_requested" | "cancelled" | "completed" | "failed"; stage: string; progress: number; result: Record<string, unknown> | null; error: string | null; created_at: string; started_at: string | null; finished_at: string | null; }
export interface JobEvent { stage: string; progress: number; message: string; created_at: string; eta_seconds: number | null; }
export interface ModelVersion { id: string; name: string; direction: Direction; kind: string; owner_id: string | null; metrics: Record<string, number | null>; metadata: Record<string, unknown>; active: boolean; created_at: string; }
export interface Page<T> { items: T[]; total: number; page: number; page_size: number; }
export interface Dashboard { total_projects: number; cue_count: number; average_confidence: number | null; pending_review: number; recent_projects: Project[]; active_jobs: Job[]; }
export interface ProjectStats { cue_count: number; edited_count: number; edit_rate: number; approved_count: number; translated_count: number; average_confidence: number | null; confidence_histogram: number[]; unscored_count: number; cps_violations: number; hard_cps_violations: number; flagged_count: number; }
export interface ProjectActivity { id: string; cue_id: string; cue_index: number; action: string; revision: number; actor: string; created_at: string; }
export interface QA { cue_count: number; flagged_count: number; warning_counts: Record<string, number>; items: Cue[]; confidence_note: string; }
export interface FeedbackStats { approved_edits: number; directions: Record<Direction, { approved_edits: number; films: number; film_disjoint_training_possible: boolean }>; language_id_verified: boolean; }
