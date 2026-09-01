/**
 * A stopped backend must surface as an error, not as work in progress.
 *
 * The failure these cover was real: with the API server down, the dev-server
 * proxy accepted the connection and never answered, `fetch` never settled, and
 * the mic button spun forever with nothing to report.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, getHealth, transcribe } from "../api";

const realFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = realFetch;
  vi.useRealTimers();
  vi.restoreAllMocks();
});

/** A fetch that hangs until its signal aborts — a proxy in front of a dead
 * server, which is not the same thing as a refused connection. */
function hangingFetch(): typeof fetch {
  return vi.fn((_url: unknown, init?: RequestInit) => {
    return new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => {
        reject(new DOMException("The operation was aborted.", "AbortError"));
      });
    });
  }) as unknown as typeof fetch;
}

describe("request timeouts", () => {
  it("gives up on a request that never answers", async () => {
    vi.useFakeTimers();
    globalThis.fetch = hangingFetch();

    const pending = getHealth();
    const assertion = expect(pending).rejects.toMatchObject({
      name: "ApiError",
      status: 0,
      message: "The server didn't answer in time.",
    });
    await vi.advanceTimersByTimeAsync(15_000);
    await assertion;
  });

  it("holds a transcription open far longer than an ordinary call", async () => {
    vi.useFakeTimers();
    globalThis.fetch = hangingFetch();

    const pending = transcribe(new Blob(["x"]), "clip.webm", "book-1");
    const assertion = expect(pending).rejects.toBeInstanceOf(ApiError);
    // Still waiting where a plain GET would already have given up.
    await vi.advanceTimersByTimeAsync(15_000);
    let settled = false;
    void pending.then(
      () => (settled = true),
      () => (settled = true),
    );
    await vi.advanceTimersByTimeAsync(0);
    expect(settled).toBe(false);

    await vi.advanceTimersByTimeAsync(45_000);
    await assertion;
  });

  it("reports an unreachable server distinctly from a slow one", async () => {
    globalThis.fetch = vi.fn(() =>
      Promise.reject(new TypeError("Failed to fetch")),
    ) as unknown as typeof fetch;

    await expect(getHealth()).rejects.toMatchObject({
      status: 0,
      message: "Can't reach the BookTalk server.",
    });
  });

  it("clears its timer once the body is read", async () => {
    const clear = vi.spyOn(globalThis, "clearTimeout");
    globalThis.fetch = vi.fn(() =>
      Promise.resolve(
        new Response(JSON.stringify({ ok: true, openai_configured: true }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    ) as unknown as typeof fetch;

    await expect(getHealth()).resolves.toEqual({
      ok: true,
      openai_configured: true,
    });
    expect(clear).toHaveBeenCalled();
  });
});
