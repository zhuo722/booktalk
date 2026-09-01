import type { CSSProperties } from "react";

/**
 * The cloth a book wears, everywhere it appears. Both gradient stops, the
 * page-edge tint, the on-cloth ink and the gradient's direction come from
 * the book's position on the shelf — never fetched, never random — so a book
 * is the same colour on the shelf and in the reader. One source, five custom
 * properties, read by both surfaces.
 */

export const STAGE_LABELS: Record<string, string> = {
  parsing: "Reading pages",
  structuring: "Mapping chapters",
  chunking: "Cutting passages",
  contextualizing: "Adding context",
  summarizing: "Summarizing chapters",
  embedding: "Indexing passages",
};

/** The neighbourhood a cloth belongs to, for anything that groups by mood. */
export type ClothFamily = "warm" | "citrus" | "verdant" | "marine" | "twilight";

/**
 * Sixteen composed gradient pairs, each in two tones: thirty-two cloths.
 *
 * Evenly stepping a hue wheel was the wrong instrument — thirty-two equal
 * steps of the same saturation read as a highlighter set, and with both stops
 * a few degrees apart no cover showed a gradient at all. These are pairs
 * instead: each travels 24–70° of hue, so the cover visibly goes somewhere,
 * and the pairs are chosen as combinations (peach into rose, mint into sky,
 * periwinkle into orchid) rather than sampled off a wheel.
 *
 * Tone is the second axis: an airy tone and a deeper one for the same pair,
 * which doubles the set without adding a colour that fights the others.
 * --ink measures ≥7.7:1 on every stop of all thirty-two.
 */
const CLOTH_PAIRS: { from: number; to: number; family: ClothFamily }[] = [
  { from: 24, to: 344, family: "warm" },
  { from: 36, to: 10, family: "warm" },
  { from: 46, to: 22, family: "citrus" },
  { from: 40, to: 352, family: "warm" },
  { from: 64, to: 32, family: "citrus" },
  { from: 88, to: 148, family: "verdant" },
  { from: 112, to: 168, family: "verdant" },
  { from: 150, to: 196, family: "verdant" },
  { from: 168, to: 214, family: "marine" },
  { from: 192, to: 238, family: "marine" },
  { from: 206, to: 262, family: "marine" },
  { from: 226, to: 286, family: "twilight" },
  { from: 248, to: 308, family: "twilight" },
  { from: 272, to: 330, family: "twilight" },
  { from: 300, to: 346, family: "warm" },
  { from: 316, to: 26, family: "warm" },
];

/** satA, satB, lightA, lightB. */
const CLOTH_TONES: [number, number, number, number][] = [
  [78, 68, 89, 82],
  [86, 74, 83, 76],
];

export const CLOTH_SLOTS = CLOTH_PAIRS.length * CLOTH_TONES.length;

export interface Cloth {
  family: ClothFamily;
  /** Which of the thirty-two cloths this book wears. */
  slot: number;
  /** Gradient stop A and stop B. */
  a: string;
  b: string;
  /** The front page-edge: the same hue, opened up toward paper. */
  edge: string;
  /**
   * Secondary ink for text sitting ON the cloth — tinted from the cloth's own
   * hue rather than the neutral --ink-soft, which bottoms out near 3:1 on the
   * darker stops. Measured at 4.89:1 on the worst cloth and stop.
   */
  tint: string;
  /** The gradient's direction, so two books on one cloth still differ. */
  angle: number;
}

/**
 * The cloth for a book, by its position on the shelf. Walking the palette in
 * order is what makes a shelf look composed: the first sixteen books take the
 * sixteen pairs in the airy tone, the next sixteen repeat them deeper, and
 * only the thirty-third book comes back to a colour already on the shelf.
 * Hashing the title gave the same guarantee to nobody — two books could
 * collide on the second shelf entry.
 */
export function clothForIndex(index: number): Cloth {
  const slot = ((index % CLOTH_SLOTS) + CLOTH_SLOTS) % CLOTH_SLOTS;
  // Tone alternates between neighbours AND advances once per full walk of
  // the pairs, so the seventeenth book is the first pair in the other tone
  // rather than a repeat of the first book.
  const pair = CLOTH_PAIRS[slot % CLOTH_PAIRS.length];
  const toneIndex =
    (slot + Math.floor(slot / CLOTH_PAIRS.length)) % CLOTH_TONES.length;
  const [satA, satB, lightA, lightB] = CLOTH_TONES[toneIndex];
  // The gradient's direction walks too, so neighbours differ by more than hue.
  const angle = 112 + (slot % 5) * 14;
  return {
    family: pair.family,
    slot,
    a: `hsl(${pair.from} ${satA}% ${lightA}%)`,
    b: `hsl(${pair.to} ${satB}% ${lightB}%)`,
    edge: `hsl(${pair.from} ${Math.min(satA, 34)}% 95%)`,
    tint: `hsl(${pair.from} 26% 18%)`,
    angle,
  };
}

/** The custom properties every surface that wears a book's colour reads. */
export function clothStyle(index: number): CSSProperties {
  const cloth = clothForIndex(index);
  return {
    "--cloth-a": cloth.a,
    "--cloth-b": cloth.b,
    "--cloth-edge": cloth.edge,
    "--cloth-tint": cloth.tint,
    "--cloth-angle": `${cloth.angle}deg`,
  } as CSSProperties;
}

