import type { CSSProperties } from "react";
import { IconMic, IconStop } from "./Icons";

export type MicState = "idle" | "recording" | "busy";

interface MicButtonProps {
  state: MicState;
  /** Live analyser RMS (0..~1) while recording; drives the level ring. */
  level: number;
  disabled?: boolean;
  onClick: () => void;
}

const LABELS: Record<MicState, string> = {
  idle: "Ask by voice",
  recording: "Stop recording",
  busy: "Transcribing",
};

export function MicButton({ state, level, disabled, onClick }: MicButtonProps) {
  const className = [
    "mic",
    state === "recording" && "mic--recording",
    state === "busy" && "mic--busy",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      type="button"
      className={className}
      aria-label={LABELS[state]}
      aria-pressed={state === "recording"}
      disabled={disabled || state === "busy"}
      onClick={onClick}
      style={{ "--mic-level": Math.min(1, level * 2.4) } as CSSProperties}
    >
      {state === "recording" && (
        <>
          <span className="mic__ring mic__ring--pulse" aria-hidden="true" />
          <span className="mic__ring mic__ring--pulse2" aria-hidden="true" />
          <span className="mic__ring mic__ring--level" aria-hidden="true" />
        </>
      )}
      {state === "recording" ? <IconStop size={15} /> : <IconMic size={18} />}
    </button>
  );
}
