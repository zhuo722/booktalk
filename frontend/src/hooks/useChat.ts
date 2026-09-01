/**
 * Chat state machine: one active SSE turn at a time, per-book threads,
 * first-class cancellation (turn_id aware) and audio hand-off to AudioQueue.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { AudioQueue } from "../lib/audioQueue";
import { postSSE } from "../lib/sse";
import { ApiError, cancelTurn } from "../lib/api";
import type {
  ChatStage,
  ChatTimings,
  CitationItem,
  Exchange,
  Source,
} from "../lib/types";

interface ActiveTurn {
  controller: AbortController;
  bookId: string;
  exchangeId: string;
  turnId: string | null;
}

export interface ChatApi {
  threads: Record<string, Exchange[]>;
  /** Book id with a live SSE stream, if any. */
  streamingBookId: string | null;
  /** Exchange currently audible through the speakers, if any. */
  speaking: { bookId: string; exchangeId: string } | null;
  ask: (bookId: string, question: string, via: "voice" | "text") => void;
  retry: (bookId: string, exchangeId: string) => void;
  /** Abort the live stream and silence playback immediately. */
  stopAll: () => void;
  clearThread: (bookId: string) => void;
}

let exchangeCounter = 0;

function newExchange(question: string, via: "voice" | "text"): Exchange {
  return {
    id: `x${++exchangeCounter}-${Date.now()}`,
    turnId: null,
    question,
    via,
    rewritten: null,
    stage: null,
    segments: [],
    answer: null,
    sources: null,
    confidence: null,
    citations: null,
    timings: null,
    status: "streaming",
    error: null,
  };
}

export function useChat(
  audioQueue: AudioQueue,
  onError: (message: string) => void,
): ChatApi {
  const [threads, setThreads] = useState<Record<string, Exchange[]>>({});
  const [streamingBookId, setStreamingBookId] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState<{
    bookId: string;
    exchangeId: string;
  } | null>(null);

  const activeRef = useRef<ActiveTurn | null>(null);
  const deadTurnsRef = useRef(new Set<string>());
  const sessionIdsRef = useRef(new Map<string, string>());
  const lastAudibleRef = useRef<{ bookId: string; exchangeId: string } | null>(
    null,
  );
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  useEffect(() => {
    audioQueue.onPlayingChange = (playing) => {
      setSpeaking(playing ? lastAudibleRef.current : null);
    };
    return () => {
      audioQueue.onPlayingChange = null;
    };
  }, [audioQueue]);

  const patchExchange = useCallback(
    (bookId: string, exchangeId: string, patch: (x: Exchange) => Exchange) => {
      setThreads((prev) => {
        const thread = prev[bookId];
        if (!thread) return prev;
        return {
          ...prev,
          [bookId]: thread.map((x) => (x.id === exchangeId ? patch(x) : x)),
        };
      });
    },
    [],
  );

  const finalizePartial = useCallback(
    (bookId: string, exchangeId: string) => {
      patchExchange(bookId, exchangeId, (x) => {
        if (x.status !== "streaming") return x;
        const partial = x.segments.map((s) => s.text).join("");
        return {
          ...x,
          status: "stopped",
          stage: null,
          answer: x.answer ?? (partial || null),
          segments: [],
        };
      });
    },
    [patchExchange],
  );

  const stopAll = useCallback(() => {
    const active = activeRef.current;
    if (active) {
      if (active.turnId) {
        deadTurnsRef.current.add(active.turnId);
        // Read playedSeq BEFORE audioQueue.stop() resets it: the server
        // truncates stored history to the sentences actually heard.
        cancelTurn(active.turnId, audioQueue.playedSeq);
      }
      active.controller.abort();
      finalizePartial(active.bookId, active.exchangeId);
      activeRef.current = null;
      setStreamingBookId(null);
    }
    audioQueue.stop();
  }, [audioQueue, finalizePartial]);

  const runStream = useCallback(
    (bookId: string, exchangeId: string, question: string) => {
      const controller = new AbortController();
      const turn: ActiveTurn = { controller, bookId, exchangeId, turnId: null };
      activeRef.current = turn;
      setStreamingBookId(bookId);
      lastAudibleRef.current = { bookId, exchangeId };

      let sessionId = sessionIdsRef.current.get(bookId);
      if (!sessionId) {
        sessionId =
          typeof crypto.randomUUID === "function"
            ? crypto.randomUUID()
            : `s-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        sessionIdsRef.current.set(bookId, sessionId);
      }

      let segmentId = 0;
      let sawDone = false;
      const patch = (fn: (x: Exchange) => Exchange) =>
        patchExchange(bookId, exchangeId, fn);

      const onEvent = ({ event, data }: { event: string; data: string }) => {
        let payload: Record<string, unknown>;
        try {
          payload = JSON.parse(data) as Record<string, unknown>;
        } catch {
          return;
        }
        const turnId =
          typeof payload.turn_id === "string" ? payload.turn_id : null;
        if (turnId) {
          if (deadTurnsRef.current.has(turnId)) return; // late chunk of a cancelled turn
          if (!turn.turnId) {
            turn.turnId = turnId;
            patch((x) => ({ ...x, turnId }));
          }
        }

        switch (event) {
          case "status":
            patch((x) => ({ ...x, stage: payload.stage as ChatStage }));
            break;
          case "question": {
            const text = typeof payload.text === "string" ? payload.text : "";
            patch((x) => ({
              ...x,
              rewritten: text && text !== x.question ? text : x.rewritten,
            }));
            break;
          }
          case "sources":
            patch((x) => ({
              ...x,
              sources: (payload.sources as Source[]) ?? [],
              confidence:
                typeof payload.confidence === "number" ? payload.confidence : null,
            }));
            break;
          case "delta": {
            const text = typeof payload.text === "string" ? payload.text : "";
            if (!text) break;
            const seg = { id: segmentId++, text };
            patch((x) => ({ ...x, segments: [...x.segments, seg] }));
            break;
          }
          case "audio":
            void audioQueue.enqueue({
              seq: Number(payload.seq ?? 0),
              format: typeof payload.format === "string" ? payload.format : "pcm",
              b64: typeof payload.b64 === "string" ? payload.b64 : "",
              sampleRate:
                typeof payload.sample_rate === "number"
                  ? payload.sample_rate
                  : undefined,
            });
            break;
          case "done": {
            sawDone = true;
            // No further audio events will arrive: release any segments
            // buffered behind a skipped seq (a failed TTS sentence).
            audioQueue.finish();
            const answer =
              typeof payload.answer === "string" ? payload.answer : null;
            const timings = (payload.timings as ChatTimings) ?? null;
            patch((x) => ({
              ...x,
              status: "done",
              stage: null,
              answer: answer ?? x.segments.map((s) => s.text).join(""),
              segments: [],
              timings,
            }));
            break;
          }
          case "citations":
            patch((x) => ({
              ...x,
              citations: (payload.map as CitationItem[]) ?? null,
            }));
            break;
          case "error": {
            const message =
              typeof payload.message === "string"
                ? payload.message
                : "The server reported an error.";
            sawDone = true; // terminal; don't overwrite with a generic error
            patch((x) => ({
              ...x,
              status: "error",
              stage: null,
              error: message,
              segments: [],
              answer:
                x.answer ?? (x.segments.map((s) => s.text).join("") || null),
            }));
            break;
          }
        }
      };

      postSSE(
        "/api/chat",
        { book_id: bookId, session_id: sessionId, question },
        onEvent,
        controller.signal,
      )
        .then(() => {
          if (controller.signal.aborted) return;
          if (!sawDone) {
            patch((x) =>
              x.status === "streaming"
                ? {
                    ...x,
                    status: "error",
                    stage: null,
                    error: "The answer stream ended early.",
                    answer: x.segments.map((s) => s.text).join("") || null,
                    segments: [],
                  }
                : x,
            );
          }
        })
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          const message =
            err instanceof ApiError
              ? err.message
              : "The connection was interrupted.";
          patch((x) => ({
            ...x,
            status: "error",
            stage: null,
            error: message,
            segments: [],
          }));
          onErrorRef.current(message);
        })
        .finally(() => {
          if (activeRef.current === turn) {
            activeRef.current = null;
            setStreamingBookId(null);
          }
        });
    },
    [audioQueue, patchExchange],
  );

  const ask = useCallback(
    (bookId: string, question: string, via: "voice" | "text") => {
      const trimmed = question.trim();
      if (!trimmed) return;
      stopAll();
      audioQueue.start(); // inside the user gesture (Safari/iOS)
      const exchange = newExchange(trimmed, via);
      setThreads((prev) => ({
        ...prev,
        [bookId]: [...(prev[bookId] ?? []), exchange],
      }));
      runStream(bookId, exchange.id, trimmed);
    },
    [audioQueue, runStream, stopAll],
  );

  const retry = useCallback(
    (bookId: string, exchangeId: string) => {
      const thread = threads[bookId];
      const exchange = thread?.find((x) => x.id === exchangeId);
      if (!exchange) return;
      stopAll();
      audioQueue.start();
      patchExchange(bookId, exchangeId, (x) => ({
        ...newExchange(x.question, x.via),
        id: x.id,
      }));
      runStream(bookId, exchangeId, exchange.question);
    },
    [audioQueue, patchExchange, runStream, stopAll, threads],
  );

  const clearThread = useCallback((bookId: string) => {
    setThreads((prev) => {
      if (!prev[bookId]) return prev;
      const next = { ...prev };
      delete next[bookId];
      return next;
    });
  }, []);

  return { threads, streamingBookId, speaking, ask, retry, stopAll, clearThread };
}
