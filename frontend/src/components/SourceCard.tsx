import { useState } from "react";
import type { CitedPassage, Source } from "../lib/types";

export function pagesLabel(pages: [number, number]): string {
  const [start, end] = pages;
  return start === end ? `p. ${start}` : `pp. ${start}–${end}`;
}

export function sourceLabel(source: Source): string {
  return `Ch. ${source.chapter_idx} · ${pagesLabel(source.pages)}`;
}

interface SourceCardProps {
  source: Source;
  index: number;
  /** True while the matching answer sentence is hovered. */
  lit: boolean;
  onHover: (chunkId: string | null) => void;
  /** Turn the book beside the conversation to this passage. */
  onOpenChapter: (passage: CitedPassage) => void;
}

/**
 * A cited passage as a white margin-note card. The same card serves the
 * right margin, where there is room for one, and the sources disclosure
 * under the answer, where there is not.
 */
export function SourceCard({
  source,
  index,
  lit,
  onHover,
  onOpenChapter,
}: SourceCardProps) {
  const [open, setOpen] = useState(false);
  const full = source.text || source.preview;
  // Offered only when there is more to show than the card already prints.
  const long = full.length > source.preview.length || source.preview.length > 160;
  const clamp = !open;

  return (
    <li
      className={lit ? "note note--lit" : "note"}
      onMouseEnter={() => onHover(source.chunk_id)}
      onMouseLeave={() => onHover(null)}
    >
      <p className="note__ref">
        <span className="note__num">{index + 1}</span>
        <button
          type="button"
          className="note__where note__where--go"
          onClick={() =>
            onOpenChapter({
              chapterIdx: source.chapter_idx,
              pages: source.pages,
              text: source.text || source.preview,
              chunkId: source.chunk_id,
            })
          }
          title="Show this passage on the book"
        >
          {sourceLabel(source)}
        </button>
      </p>
      <p className="note__chapter">
        {source.chapter_title}
        {source.section_title ? ` — ${source.section_title}` : ""}
      </p>
      <blockquote
        className={clamp ? "note__excerpt note__excerpt--clamped" : "note__excerpt"}
      >
        <mark className="cite-wash">{open ? full : source.preview}</mark>
      </blockquote>
      {long && (
        <button
          type="button"
          className="note__more"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        >
          {open ? "Less" : "Full passage"}
        </button>
      )}
    </li>
  );
}
