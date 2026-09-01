import { useEffect, useRef, useState } from "react";

/**
 * The one stat surface in the Studio: a pastel gradient tile carrying a
 * caps-mono stage label, a display numeral, and a sub-line. Text on the
 * gradient is always --ink at ≥15px semibold (≥10.6:1 at every stop).
 *
 * The numeral counts up when the value changes — state feedback, not
 * decoration, and skipped entirely under prefers-reduced-motion.
 */

const COUNT_MS = 480;

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

function useCountUp(value: number): number {
  const [shown, setShown] = useState(value);
  const fromRef = useRef(value);

  useEffect(() => {
    const from = fromRef.current;
    if (from === value) return;
    if (prefersReducedMotion() || typeof requestAnimationFrame === "undefined") {
      fromRef.current = value;
      setShown(value);
      return;
    }
    const started = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const p = Math.min(1, (now - started) / COUNT_MS);
      const eased = 1 - Math.pow(1 - p, 3);
      const next = Math.round(from + (value - from) * eased);
      setShown(next);
      fromRef.current = next;
      if (p < 1) raf = requestAnimationFrame(tick);
      else fromRef.current = value;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);

  return shown;
}

interface StatTileProps {
  /** Caps-mono stage label, e.g. "EMBEDDING". */
  label: string;
  /** Whole-number percentage, 0–100. */
  value: number;
  sub: string;
  tone?: "warm" | "cool";
}

export function StatTile({ label, value, sub, tone = "warm" }: StatTileProps) {
  const shown = useCountUp(value);
  return (
    <div className={`stat-tile stat-tile--${tone}`}>
      <p className="stat-tile__label">{label}</p>
      <p className="stat-tile__value">
        {shown}
        <span className="stat-tile__unit">%</span>
      </p>
      <p className="stat-tile__sub">{sub}</p>
    </div>
  );
}
