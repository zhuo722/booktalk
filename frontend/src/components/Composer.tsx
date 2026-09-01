import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { ApiError, transcribe } from "../lib/api";
import type { Recording } from "../lib/recorder";
import {
  isMicPermissionError,
  isNoMicError,
  startRecording,
} from "../lib/recorder";
import { IconClose, IconSend } from "./Icons";
import { MicButton } from "./MicButton";
import type { MicState } from "./MicButton";
import { Visualizer } from "./Visualizer";

/** Why the composer is unavailable; null when a ready book is selected. */
export type ComposerBlock = "empty-library" | "ingesting" | "failed" | null;

interface ComposerProps {
  block: ComposerBlock;
  /** Selected book — sent with ASR so the server can bias the decoder with
   * the book's own vocabulary and repair entity names. */
  bookId: string | null;
  onAsk: (question: string, via: "voice" | "text") => void;
  onError: (message: string) => void;
  /** Create/resume the playback AudioContext inside the user gesture. */
  primeAudio: () => void;
}

const BLOCK_COPY: Record<Exclude<ComposerBlock, null>, string> = {
  "empty-library": "Add a book to start asking",
  ingesting: "This book is still being indexed…",
  failed: "This book couldn't be indexed",
};

function formatClock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function Composer({
  block,
  bookId,
  onAsk,
  onError,
  primeAudio,
}: ComposerProps) {
  const disabled = block !== null;
  const [text, setText] = useState("");
  const [micState, setMicState] = useState<MicState>("idle");
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [micNote, setMicNote] = useState<string | null>(null);
  const recordingRef = useRef<Recording | null>(null);
  const [recording, setRecording] = useState<Recording | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Recording clock.
  useEffect(() => {
    if (micState !== "recording") {
      setElapsed(0);
      return;
    }
    const started = Date.now();
    const t = setInterval(() => setElapsed((Date.now() - started) / 1000), 250);
    return () => clearInterval(t);
  }, [micState]);

  // Abandon a live take on unmount, and remember that we are gone: a
  // transcription that resolves afterwards would otherwise ask its question
  // on a screen the reader has already left.
  //
  // The flag is re-armed on mount, not just cleared on unmount. StrictMode
  // mounts, unmounts and remounts every component in development, so a
  // cleanup-only version leaves the composer permanently marked dead: the
  // transcript comes back, the early return drops it without asking anything,
  // and the mic never leaves its busy state.
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      recordingRef.current?.cancel();
      recordingRef.current = null;
    };
  }, []);

  const submitText = (e: FormEvent) => {
    e.preventDefault();
    const q = text.trim();
    if (!q || disabled) return;
    setText("");
    onAsk(q, "text");
  };

  const handleMic = async () => {
    // A live recording must always be stoppable, even if the composer became
    // blocked mid-take (book deleted, ingest failed) — otherwise the mic
    // stream and its AudioContext are never released.
    if (disabled && micState !== "recording") return;
    if (micState === "idle") {
      primeAudio(); // playback context needs this gesture (Safari/iOS)
      try {
        const rec = await startRecording();
        recordingRef.current = rec;
        setRecording(rec);
        setMicNote(null);
        setMicState("recording");
      } catch (err) {
        if (isMicPermissionError(err)) {
          setMicNote(
            "Microphone access is blocked. Allow the microphone for this site from the icon beside the address bar, then try again — or type your question below.",
          );
        } else if (isNoMicError(err)) {
          setMicNote(
            "No microphone was found. Plug one in, or type your question below.",
          );
        } else {
          onError("The microphone couldn't be started.");
        }
      }
      return;
    }
    if (micState === "recording") {
      const rec = recordingRef.current;
      if (!rec) return;
      recordingRef.current = null;
      setRecording(null);
      setMicState("busy");
      setLevel(0);
      try {
        const { blob, filename, durationMs } = await rec.stop();
        if (durationMs < 400) {
          onError("That was too short to transcribe — hold the mic a moment longer.");
          return;
        }
        const { text: transcript } = await transcribe(blob, filename, bookId);
        if (!alive.current) return;
        onAsk(transcript, "voice");
      } catch (err) {
        if (!alive.current) return;
        onError(
          err instanceof ApiError && err.status === 422
            ? "Nothing intelligible was heard — try again, a little closer to the mic."
            : err instanceof ApiError
              ? err.message
              : "Transcription failed.",
        );
      } finally {
        // Unconditionally: setting state on an unmounted component is a no-op
        // in React 18, and gating this is how the button got stuck busy with
        // no way back.
        setMicState("idle");
      }
    }
  };

  return (
    <footer className="composer">
      <div className="composer__bar">
        <form className="composer__box" onSubmit={submitText}>
          {/* The mic stands on its own, left of the field: it is a different
              way to ask, not a control of the text box. */}
          <MicButton
            state={micState}
            level={level}
            disabled={disabled && micState !== "recording"}
            onClick={() => void handleMic()}
          />

          <div className="composer__field">
            {micState === "recording" && recording ? (
              <div className="composer__rec">
                <span className="rec-label">Listening</span>
                <Visualizer
                  ctx={recording.ctx}
                  source={recording.source}
                  className="viz--input"
                  onLevel={(l) => setLevel(Math.round(l * 50) / 50)}
                />
                <span className="rec-time">{formatClock(elapsed)}</span>
              </div>
            ) : (
              <>
                <input
                  ref={inputRef}
                  id="composer-input"
                  className="composer__input"
                  type="text"
                  value={text}
                  placeholder={
                    block !== null
                      ? BLOCK_COPY[block]
                      : micState === "busy"
                        ? "Transcribing your question…"
                        : "Ask about this book…"
                  }
                  disabled={disabled || micState === "busy"}
                  onChange={(e) => setText(e.target.value)}
                  aria-label="Type a question about the book"
                />
                <button
                  type="submit"
                  className="composer__send"
                  disabled={disabled || text.trim().length === 0}
                  aria-label="Send question"
                >
                  <IconSend size={15} />
                </button>
              </>
            )}
          </div>
        </form>

        {micNote && (
          <div className="mic-note" role="alert">
            <p>{micNote}</p>
            <button
              type="button"
              className="icon-btn icon-btn--err"
              aria-label="Dismiss microphone notice"
              onClick={() => setMicNote(null)}
            >
              <IconClose size={14} />
            </button>
          </div>
        )}
      </div>
    </footer>
  );
}
