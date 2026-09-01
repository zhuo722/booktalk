/**
 * Push-to-talk microphone recorder.
 *
 * MediaRecorder with audio/webm;codecs=opus, falling back to audio/mp4
 * (Safari). Exposes the input AudioContext + source node so a visualizer can
 * tap the live signal. The API is shaped so a hands-free endpointer (VAD)
 * could later wrap `start`/`stop` without changing callers.
 */

export interface Recording {
  /** Input-side context; hand to a Visualizer together with `source`. */
  ctx: AudioContext;
  source: MediaStreamAudioSourceNode;
  mimeType: string;
  /** Stop and resolve the captured audio. */
  stop(): Promise<{ blob: Blob; filename: string; durationMs: number }>;
  /** Abandon the take: stop tracks, discard audio. */
  cancel(): void;
}

const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];

function pickMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  return MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m)) ?? "";
}

function extensionFor(mime: string): string {
  if (mime.includes("mp4")) return "m4a";
  if (mime.includes("ogg")) return "ogg";
  return "webm";
}

export function isMicPermissionError(err: unknown): boolean {
  return (
    err instanceof DOMException &&
    (err.name === "NotAllowedError" || err.name === "SecurityError")
  );
}

export function isNoMicError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "NotFoundError";
}

export async function startRecording(): Promise<Recording> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  type Ctor = typeof AudioContext;
  const Ctx: Ctor =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext: Ctor }).webkitAudioContext;
  const ctx = new Ctx();
  const source = ctx.createMediaStreamSource(stream);

  const mimeType = pickMimeType();
  const recorder = mimeType
    ? new MediaRecorder(stream, { mimeType })
    : new MediaRecorder(stream);
  const chunks: Blob[] = [];
  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };
  const startedAt = performance.now();
  recorder.start(250);

  let finished = false;
  const teardown = () => {
    finished = true;
    stream.getTracks().forEach((t) => t.stop());
    void ctx.close().catch(() => undefined);
  };

  return {
    ctx,
    source,
    mimeType: recorder.mimeType || mimeType || "audio/webm",
    stop() {
      return new Promise((resolve, reject) => {
        if (finished) {
          reject(new Error("Recording already finished."));
          return;
        }
        recorder.onstop = () => {
          const type = recorder.mimeType || mimeType || "audio/webm";
          teardown();
          resolve({
            blob: new Blob(chunks, { type }),
            filename: `question.${extensionFor(type)}`,
            durationMs: Math.round(performance.now() - startedAt),
          });
        };
        recorder.onerror = () => {
          teardown();
          reject(new Error("Recording failed."));
        };
        recorder.stop();
      });
    },
    cancel() {
      if (finished) return;
      recorder.onstop = null;
      try {
        recorder.stop();
      } catch {
        /* already inactive */
      }
      teardown();
    },
  };
}
