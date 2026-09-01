import { afterEach, describe, expect, it, vi } from "vitest";
import { postSSE, readSSEStream } from "../sse";
import type { SSEEvent } from "../sse";

const enc = new TextEncoder();

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(enc.encode(chunk));
      controller.close();
    },
  });
}

describe("readSSEStream", () => {
  it("parses named events and joins multi-line data", async () => {
    const events: SSEEvent[] = [];
    await readSSEStream(
      streamOf([
        ": keep-alive comment\n" +
          "event: sources\n" +
          'data: {"sources":\n' +
          'data: [1,2]}\n' +
          "\n" +
          'data: {"plain":true}\n\n',
      ]),
      (e) => events.push(e),
    );
    expect(events).toEqual([
      { event: "sources", data: '{"sources":\n[1,2]}' },
      { event: "message", data: '{"plain":true}' },
    ]);
  });

  it("is robust to chunk boundaries mid-event and mid-field", async () => {
    const wire =
      'event: delta\ndata: {"text":"the ferry timetable"}\n\n' +
      'event: done\ndata: {"answer":"ok"}\n\n';
    // Split at awkward byte offsets: inside "event:", inside the JSON, inside "\n\n".
    const cuts = [3, 11, 20, wire.indexOf("done") + 2, wire.length - 1];
    const chunks: string[] = [];
    let prev = 0;
    for (const cut of cuts) {
      chunks.push(wire.slice(prev, cut));
      prev = cut;
    }
    chunks.push(wire.slice(prev));

    const events: SSEEvent[] = [];
    await readSSEStream(streamOf(chunks), (e) => events.push(e));
    expect(events).toEqual([
      { event: "delta", data: '{"text":"the ferry timetable"}' },
      { event: "done", data: '{"answer":"ok"}' },
    ]);
  });

  it("stops delivering events after abort and cancels the source", async () => {
    let ctrl!: ReadableStreamDefaultController<Uint8Array>;
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        ctrl = c;
      },
      cancel() {
        cancelled = true;
      },
    });

    const events: SSEEvent[] = [];
    const abort = new AbortController();
    const finished = readSSEStream(stream, (e) => events.push(e), abort.signal);

    ctrl.enqueue(enc.encode('event: delta\ndata: {"text":"one"}\n\n'));
    await vi.waitFor(() => expect(events).toHaveLength(1));

    abort.abort();
    await finished; // resolves quietly on abort

    try {
      ctrl.enqueue(enc.encode('event: delta\ndata: {"text":"two"}\n\n'));
    } catch {
      /* the pipeline was cancelled — enqueue may throw, which is fine */
    }
    await new Promise((r) => setTimeout(r, 10));

    expect(events).toHaveLength(1);
    expect(cancelled).toBe(true);
  });
});

describe("postSSE", () => {
  const realFetch = globalThis.fetch;
  afterEach(() => {
    globalThis.fetch = realFetch;
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("gives up waiting for response headers that never arrive", async () => {
    vi.useFakeTimers();
    globalThis.fetch = vi.fn((_url: unknown, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () =>
          reject(new DOMException("aborted", "AbortError")),
        );
      });
    }) as unknown as typeof fetch;

    const pending = postSSE("/api/chat", { question: "hi" }, () => undefined);
    const assertion = expect(pending).rejects.toMatchObject({
      name: "ApiError",
      message: "The server didn't answer in time.",
    });
    await vi.advanceTimersByTimeAsync(20_000);
    await assertion;
  });

  it("does not cut off a stream that has already started", async () => {
    vi.useFakeTimers();
    let ctrl!: ReadableStreamDefaultController<Uint8Array>;
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        ctrl = c;
      },
    });
    globalThis.fetch = vi.fn(() =>
      Promise.resolve(new Response(stream, { status: 200 })),
    ) as unknown as typeof fetch;

    const events: SSEEvent[] = [];
    const finished = postSSE("/api/chat", {}, (e) => events.push(e));

    // Well past the header deadline: a long answer must keep streaming.
    await vi.advanceTimersByTimeAsync(60_000);
    ctrl.enqueue(enc.encode('event: delta\ndata: {"text":"still here"}\n\n'));
    await vi.waitFor(() => expect(events).toHaveLength(1));
    ctrl.close();
    await finished;

    expect(events[0]).toEqual({
      event: "delta",
      data: '{"text":"still here"}',
    });
  });
});
