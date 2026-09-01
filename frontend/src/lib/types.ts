/** Shared API + UI types for the BookTalk frontend. */

export type BookStatus =
  | "parsing"
  | "structuring"
  | "chunking"
  | "contextualizing"
  | "summarizing"
  | "embedding"
  | "ready"
  | "failed";

export interface BookSummary {
  book_id: string;
  title: string;
  pages: number;
  status: BookStatus;
  progress: number;
  created_at: string;
}

export interface BookStatusResponse {
  status: BookStatus;
  progress: number;
  error?: string;
}

export interface TocChapter {
  idx: number;
  title: string;
  page_start: number;
  page_end: number;
  /**
   * The chapter's generated synopsis (SPEC §14.5). Absent — not null — for a
   * chapter the backend never summarized, so `chapter.synopsis` is the only
   * check the UI needs before offering to expand a row.
   */
  synopsis?: string;
  /**
   * The chapter's own opening prose, straight from the book. This is what the
   * reader prints on a chapter page; absent when the chapter has no indexed
   * body text.
   */
  opening?: string;
}

export interface TocResponse {
  title: string;
  chapters: TocChapter[];
}

export interface HealthResponse {
  ok: boolean;
  openai_configured: boolean;
}

export interface AsrResponse {
  text: string;
  duration_ms: number;
}

export interface UploadResponse {
  book_id: string;
  status: BookStatus;
}

export interface Source {
  chunk_id: string;
  chapter_idx: number;
  chapter_title: string;
  section_title: string | null;
  pages: [number, number];
  preview: string;
  /** The passage in full; `preview` is the first 200 characters of it. */
  text?: string;
}

/** A passage a citation points at, for the book to turn to and show. */
export interface CitedPassage {
  chapterIdx: number;
  pages: [number, number];
  text: string;
  chunkId: string;
}

export interface ChatTimings {
  rewrite_ms?: number;
  retrieve_ms?: number;
  first_token_ms?: number;
  first_audio_ms?: number;
  total_ms?: number;
}

/** One answer-sentence → source-chunks mapping, from the `citations` SSE
 * event's `map` array (SPEC §14.7). `span` holds character offsets into the
 * final answer — the server's sentence splitter is the only authority on
 * sentence boundaries, so the client highlights by slicing, never by
 * re-splitting. */
export interface CitationItem {
  sentence_idx: number;
  span?: [number, number];
  chunk_ids: string[];
}

export type ChatStage = "rewriting" | "retrieving" | "generating" | "speaking";

/** A question/answer exchange as tracked by the UI. */
export interface Exchange {
  id: string;
  turnId: string | null;
  question: string;
  /** How the question was asked; voice questions carry a small mic mark. */
  via: "voice" | "text";
  /** Post-rewrite question, when the server reformulated it. */
  rewritten: string | null;
  stage: ChatStage | null;
  /** Streaming word segments; cleared once `answer` lands. */
  segments: { id: number; text: string }[];
  answer: string | null;
  sources: Source[] | null;
  /** Normalized top retrieval score from the sources event; an absolute
   * quantity only when the cross-encoder ran (else 1.0). Parsed and kept, but
   * nothing renders it: it proved unusable as a grounding signal — 0.0017 for
   * a correctly answered question, 0.9953 for a subject the book never
   * mentions — so the warning above the answer counts traced sentences
   * instead. */
  confidence: number | null;
  citations: CitationItem[] | null;
  timings: ChatTimings | null;
  status: "streaming" | "done" | "stopped" | "error";
  error: string | null;
}

