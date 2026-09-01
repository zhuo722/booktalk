/** Typed fetch wrappers for the BookTalk API contract (SPEC §6). */

import type {
  AsrResponse,
  BookStatusResponse,
  BookSummary,
  HealthResponse,
  TocResponse,
  UploadResponse,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function toError(res: Response): Promise<ApiError> {
  let message = `The server couldn't answer (HTTP ${res.status}).`;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") message = body.detail;
    else if (typeof body?.message === "string") message = body.message;
  } catch {
    /* non-JSON error body; keep the status message */
  }
  return new ApiError(res.status, message);
}

/** Fifteen seconds is far longer than any of these endpoints needs on
 * localhost, and short enough that a stalled request still reads as a failure
 * rather than as work in progress. */
const DEFAULT_TIMEOUT_MS = 15_000;
/** Transcription waits on OpenAI, and a long clip legitimately takes seconds. */
const ASR_TIMEOUT_MS = 60_000;
/** Upload returns as soon as ingestion is queued, but the file itself still
 * has to be written and preflighted first. */
const UPLOAD_TIMEOUT_MS = 120_000;

async function request<T>(
  url: string,
  init?: RequestInit,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  // Without this, a dev-server proxy that accepts the connection and never
  // answers — which is what a stopped backend looks like — leaves the promise
  // pending forever, and the UI spins with nothing to report.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    let res: Response;
    try {
      res = await fetch(url, { ...init, signal: controller.signal });
    } catch {
      throw controller.signal.aborted
        ? new ApiError(0, "The server didn't answer in time.")
        : new ApiError(0, "Can't reach the BookTalk server.");
    }
    if (!res.ok) throw await toError(res);
    if (res.status === 204) return undefined as T;
    // Headers can arrive long before the body does, so the deadline has to
    // cover the read as well.
    try {
      return (await res.json()) as T;
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw controller.signal.aborted
        ? new ApiError(0, "The server didn't answer in time.")
        : new ApiError(res.status, "The server's reply was unreadable.");
    }
  } finally {
    clearTimeout(timer);
  }
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export async function listBooks(): Promise<BookSummary[]> {
  const data = await request<{ books: BookSummary[] }>("/api/books");
  return data.books;
}

export function uploadBook(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file, file.name);
  return request<UploadResponse>(
    "/api/books",
    { method: "POST", body: form },
    UPLOAD_TIMEOUT_MS,
  );
}

export function getBookStatus(bookId: string): Promise<BookStatusResponse> {
  return request<BookStatusResponse>(
    `/api/books/${encodeURIComponent(bookId)}/status`,
  );
}

/**
 * The book's chapter list. Chapters that were summarized during ingestion
 * (SPEC §14.5) carry a `synopsis`; the key is omitted for the rest, so the
 * shape is per-chapter rather than all-or-nothing.
 */
export function getToc(bookId: string): Promise<TocResponse> {
  return request<TocResponse>(`/api/books/${encodeURIComponent(bookId)}/toc`);
}

export function deleteBook(bookId: string): Promise<void> {
  return request<void>(`/api/books/${encodeURIComponent(bookId)}`, {
    method: "DELETE",
  });
}

export function transcribe(
  audio: Blob,
  filename: string,
  bookId?: string | null,
): Promise<AsrResponse> {
  const form = new FormData();
  form.append("audio", audio, filename);
  // With a book_id the server biases the ASR decoder with the book's own
  // vocabulary and repairs mistranscribed entity names (SPEC §14.9).
  if (bookId) form.append("book_id", bookId);
  return request<AsrResponse>(
    "/api/asr",
    { method: "POST", body: form },
    ASR_TIMEOUT_MS,
  );
}

/** Report an interrupted spoken turn so the server stops synthesizing and
 * truncates stored history to what was actually heard (SPEC §14.8).
 * Fire-and-forget: cancellation must never block the next question. */
export function cancelTurn(turnId: string, playedSeq: number): void {
  void fetch(`/api/chat/turns/${encodeURIComponent(turnId)}/cancel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ played_seq: playedSeq }),
    keepalive: true,
  }).catch(() => undefined);
}
