import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type {
  BookSummary,
  CitedPassage,
  TocChapter,
  TocResponse,
} from "../lib/types";
import { clothStyle } from "./cloth";
import { IconArrowLeft, IconArrowRight } from "./Icons";

interface BookPanelProps {
  book: BookSummary;
  /** The book's position in the library, so its cloth matches the shelf. */
  index: number;
  toc: TocResponse | null;
  /** Where the book was on the shelf, so it can fly from there. */
  from: DOMRect | null;
  onArrived: () => void;
  /** A passage the reader asked to see, or null for the chapter's opening. */
  cited: CitedPassage | null;
  /** The pager moved, so any citation on show no longer applies. */
  onChapter: () => void;
}

/** How long the book takes to cross from the shelf. */
const FLIGHT_MS = 620;

function pad(n: number): string {
  return n < 10 ? `0${n}` : String(n);
}

/**
 * One book, open. The volume flies in from the shelf, and one page at a time
 * lifts out of it — the chapter the reader picked. Which chapter that is
 * lives under the book, not on it: a strip of numbers stays reachable at
 * twelve chapters and at ninety, where a fan of leaves stops being clickable
 * after the first few.
 */
export function BookPanel({
  book,
  index,
  toc,
  from,
  onArrived,
  cited,
  onChapter,
}: BookPanelProps) {
  const tomeRef = useRef<HTMLDivElement>(null);
  const coverRef = useRef<HTMLDivElement>(null);
  const stripRef = useRef<HTMLDivElement>(null);
  const [landed, setLanded] = useState(from === null);
  const [localIndex, setLocalIndex] = useState(0);
  // A citation names a chapter; find where that chapter sits in the list.
  const requested =
    cited === null
      ? -1
      : (toc?.chapters.findIndex((c) => c.idx === cited.chapterIdx) ?? -1);
  const chapterIndex = requested >= 0 ? requested : localIndex;
  const setChapterIndex = (next: number) => {
    setLocalIndex(next);
    onChapter();
  };
  const chapters = toc?.chapters ?? [];
  const chapter: TocChapter | undefined = chapters[chapterIndex];
  // What the page prints: the cited passage when one was asked for, and the
  // chapter's own opening otherwise.
  const showing = requested >= 0 ? cited : null;
  const body = showing
    ? showing.text
    : (chapter?.opening ?? chapter?.synopsis ?? null);

  // FLIP, driven by the Web Animations API rather than a transition: the
  // panel renders where it belongs, and the book is animated from the rect
  // it occupied on the shelf to that place. A transition released inside
  // requestAnimationFrame lost the release whenever React re-rendered
  // mid-flight; an animation object survives that, and can be cancelled.
  useLayoutEffect(() => {
    if (!from) return;

    // Scheduled first and run at most once: a flight that cannot start —
    // no node, no measurable destination — must still open the page.
    let done = false;
    const land = () => {
      if (done) return;
      done = true;
      setLanded(true);
      onArrived();
    };
    const fallback = window.setTimeout(land, FLIGHT_MS + 200);
    const cleanup = () => window.clearTimeout(fallback);

    const node = coverRef.current;
    if (!node) return cleanup;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      land();
      return cleanup;
    }

    // Cancel before measuring: React mounts effects twice in development,
    // and a surviving flight parks the element at its starting keyframe,
    // where its rect is the shelf's rather than the destination's.
    node.getAnimations().forEach((a) => a.cancel());
    const to = node.getBoundingClientRect();
    if (!to.width || !to.height) return cleanup;

    // The box grows from its own top-left, so on the first frame the element
    // has the shelf book's size at the reader's position: the offset is the
    // distance between those two corners.
    const dx = from.left - to.left;
    const dy = from.top - to.top;

    // The book is a wide bar on the shelf and an upright cover here, so the
    // flight changes its shape as well as its place. A transform could only
    // do that by stretching the title with it; animating the box lets the
    // title reflow as the cover turns.
    const flight = node.animate(
      [
        {
          width: `${from.width}px`,
          height: `${from.height}px`,
          transform: `translate(${dx}px, ${dy}px)`,
        },
        { width: `${to.width}px`, height: `${to.height}px`, transform: "none" },
      ],
      { duration: FLIGHT_MS, easing: "cubic-bezier(.4,0,.2,1)", fill: "backwards" },
    );
    flight.finished.then(land).catch(() => undefined);
    return cleanup;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [from]);

  // Nothing to wait for when the reader was reached without a flight.
  useEffect(() => {
    if (!from) setLanded(true);
  }, [from]);

  // Keep the picked number in view when it is reached by keyboard.
  useEffect(() => {
    const strip = stripRef.current;
    const active = strip?.querySelector<HTMLElement>('[aria-current="true"]');
    active?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [chapterIndex]);

  const step = (delta: number) => {
    setChapterIndex(
      Math.min(chapters.length - 1, Math.max(0, chapterIndex + delta)),
    );
  };

  return (
    <section className="reader__book" aria-label={`${book.title}, chapters`}>
      <div className="stage">
        <div
          className={landed ? "tome tome--open" : "tome"}
          ref={tomeRef}
          style={clothStyle(index)}
        >
          {/* The cover carries the title across. Without it the book arrived
              blank and the type vanished at the hand-off, which read as a
              cut rather than a move. */}
          <div className="tome__block" ref={coverRef} aria-hidden="true">
            <span className="tome__cover">
              <span className="tome__cover-title">{book.title}</span>
            </span>
            <span className="tome__pages" />
          </div>

          {landed && chapter && (
            // Held back until the book has landed: the page is a child of the
            // flying element, so drawing it early sent a half-transparent
            // chapter drifting across the conversation. Keyed on the chapter,
            // so picking another one plays the lift again from the book
            // rather than cross-fading text in place.
            <article
              className={showing ? "page page--cited" : "page"}
              key={showing ? showing.chunkId : `ch-${chapter.idx}`}
            >
              {/* Turned to the passage the citation points at — its own
                  pages and its own words — rather than to the top of the
                  chapter it happens to sit in. The pager turns it back. */}
              <p className="page__folio">
                Chapter {chapter.idx}
                {(() => {
                  const pages = showing ? showing.pages : null;
                  const start = pages ? pages[0] : chapter.page_start;
                  const end = pages ? pages[1] : chapter.page_end;
                  if (start == null) return null;
                  return (
                    <span>
                      {" · "}
                      {start === end ? `p. ${start}` : `pp. ${start}–${end}`}
                    </span>
                  );
                })()}
              </p>
              <h2 className="page__title">{chapter.title}</h2>
              {body ? (
                <p className={showing ? "page__body page__body--cited" : "page__body"}>
                  {body}
                </p>
              ) : (
                <p className="page__body page__body--empty">
                  This chapter has no indexed text.
                </p>
              )}
            </article>
          )}
        </div>
      </div>

      {chapters.length > 0 && (
        <nav className="pager" aria-label="Chapters">
          <button
            type="button"
            className="icon-btn"
            aria-label="Previous chapter"
            disabled={chapterIndex === 0}
            onClick={() => step(-1)}
          >
            <IconArrowLeft size={16} />
          </button>

          <div className="pager__strip" ref={stripRef}>
            {chapters.map((c, i) => (
              <button
                key={c.idx}
                type="button"
                className={
                  i === chapterIndex ? "pager__num pager__num--on" : "pager__num"
                }
                aria-current={i === chapterIndex}
                aria-label={`Chapter ${c.idx}: ${c.title}`}
                onClick={() => setChapterIndex(i)}
              >
                {pad(c.idx)}
              </button>
            ))}
          </div>

          <button
            type="button"
            className="icon-btn"
            aria-label="Next chapter"
            disabled={chapterIndex >= chapters.length - 1}
            onClick={() => step(1)}
          >
            <IconArrowRight size={16} />
          </button>

          <p className="pager__count" aria-hidden="true">
            {pad(chapterIndex + 1)}
            <span> / {pad(chapters.length)}</span>
          </p>
        </nav>
      )}

      {/* Two different states, said differently: a book whose contents are
          still loading is not a book without chapters. */}
      {!toc && <p className="stage__empty">Opening the book…</p>}
      {toc && chapters.length === 0 && (
        <p className="stage__empty">No chapters were detected in this book.</p>
      )}
    </section>
  );
}
