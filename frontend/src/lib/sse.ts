/**
 * POST-based server-sent-events reader.
 *
 * The wire parsing (chunk boundaries mid-event, multi-line `data:`, comments)
 * is delegated to eventsource-parser's TransformStream; this module owns the
 * POST + AbortController contract and the byte→event plumbing.
 */

import { EventSourceParserStream } from "eventsource-parser/stream";
import { ApiError } from "./api";

export interface SSEEvent {
  /** Event name; "message" when the stream did not name one. */
  event: string;
  /** Raw data payload (joined multi-line data). */
  data: string;
}

export type SSEEventHandler = (event: SSEEvent) => void;

/** How long to wait for the answer stream's response headers. Retrieval and
 * the first token happen well inside this on a working backend. */
const CONNECT_TIMEOUT_MS = 20_000;

/**
 * Consume a byte stream of `text/event-stream` data, invoking `onEvent` per
 * complete event. Resolves when the stream ends or the signal aborts; no
 * events are delivered after abort.
 */
export async function readSSEStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: SSEEventHandler,
  signal?: AbortSignal,
): Promise<void> {
  const events = stream
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new EventSourceParserStream());
  const reader = events.getReader();

  const onAbort = () => {
    void reader.cancel().catch(() => undefined);
  };
  if (signal) {
    if (signal.aborted) {
      onAbort();
      return;
    }
    signal.addEventListener("abort", onAbort, { once: true });
  }

  try {
    for (;;) {
      let result: ReadableStreamReadResult<{ data: string; event?: string }>;
      try {
        result = await reader.read();
      } catch (err) {
        if (signal?.aborted) return;
        throw err;
      }
      if (result.done || signal?.aborted) return;
      onEvent({ event: result.value.event ?? "message", data: result.value.data });
    }
  } finally {
    signal?.removeEventListener("abort", onAbort);
    reader.releaseLock();
  }
}

/**
 * POST JSON to `url` and stream the SSE response. Rejects with ApiError on a
 * non-2xx response; resolves quietly when aborted via `signal`.
 */
export async function postSSE(
  url: string,
  body: unknown,
  onEvent: SSEEventHandler,
  signal?: AbortSignal,
): Promise<void> {
  if (signal?.aborted) return;
  // The answer stream itself may legitimately run for a minute, so only the
  // wait for response headers gets a deadline. Without one, a dev-server proxy
  // in front of a stopped backend accepts the connection and never replies,
  // and the question sits in its pending state with nothing to report.
  const controller = new AbortController();
  const forwardAbort = () => controller.abort();
  signal?.addEventListener("abort", forwardAbort);
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, CONNECT_TIMEOUT_MS);

  try {
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (err) {
      if (signal?.aborted) return;
      if (timedOut) throw new ApiError(0, "The server didn't answer in time.");
      throw err instanceof ApiError
        ? err
        : new ApiError(0, "Can't reach the BookTalk server.");
    } finally {
      clearTimeout(timer);
    }

    if (!res.ok) {
      let message = `The server couldn't answer (HTTP ${res.status}).`;
      try {
        const detail = (await res.json()) as { detail?: string };
        if (typeof detail?.detail === "string") message = detail.detail;
      } catch {
        /* keep default */
      }
      throw new ApiError(res.status, message);
    }
    if (!res.body) throw new ApiError(0, "The server sent an empty stream.");

    await readSSEStream(res.body, onEvent, signal);
  } finally {
    signal?.removeEventListener("abort", forwardAbort);
  }
}
