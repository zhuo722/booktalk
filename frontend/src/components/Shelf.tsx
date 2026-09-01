import { useCallback, useEffect, useRef, useState } from "react";
import { Popover } from "radix-ui";
import type { BookSummary, TocResponse } from "../lib/types";
import { HeroPanel } from "./Hero";
import { ShelfAddRow, ShelfBook, ShelfGhostRow } from "./ShelfBook";
import { ShelfRail } from "./ShelfRail";
import { IconHelp, IconMark } from "./Icons";

interface ShelfProps {
  books: BookSummary[];
  loaded: boolean;
  tocs: Record<string, TocResponse>;
  onOpen: (bookId: string, from: DOMRect | null) => void;
  /** The cover's rect in the reader, for the book's flight back to its row. */
  returnFlight: { bookId: string; rect: DOMRect } | null;
  onReturned: () => void;
  /** Where the shelf was scrolled to when it was left. */
  initialScroll: number;
  onUpload: (file: File) => void;
  onRemove: (bookId: string) => void;
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/** What the app does, and the one thing about it people ask first. */
function ShelfHelp() {
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button type="button" className="shelf-help" aria-label="About BookTalk">
          <IconHelp size={17} />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          className="popover"
          side="top"
          align="start"
          sideOffset={10}
          collisionPadding={12}
        >
          <h2 className="popover__title">About BookTalk</h2>
          <p>
            Drop in a PDF. BookTalk maps its chapters, indexes every passage,
            and then answers your spoken questions aloud, citing the pages it
            drew from.
          </p>
          <p>
            An answer comes from the open book and nothing else — no outside
            sources, no general knowledge filling the gaps.
          </p>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

/**
 * The landing screen: a quiet corner header, a long silence, and then the
 * books, one to a row, as a slow vertical procession of physical objects.
 * Picking one is the only way into the conversation screen.
 */
export function Shelf({
  books,
  loaded,
  tocs,
  onOpen,
  returnFlight,
  onReturned,
  initialScroll,
  onUpload,
  onRemove,
}: ShelfProps) {
  const scrollRef = useRef<HTMLElement | null>(null);
  // The returning book measures its row in its own layout effect, which runs
  // before this element's ref callback does — so the restore cannot live
  // there. It is a function instead, called by whoever needs the shelf to be
  // where the reader left it, and it runs once.
  const scrollRestored = useRef(false);
  const restoreScroll = useCallback(() => {
    if (scrollRestored.current) return;
    scrollRestored.current = true;
    const el = scrollRef.current;
    if (el && initialScroll > 0) el.scrollTop = initialScroll;
  }, [initialScroll]);

  const attachScroll = (el: HTMLElement | null) => {
    scrollRef.current = el;
    restoreScroll();
  };
  const rowRefs = useRef<(HTMLLIElement | null)[]>([]);
  const [scrolledIndex, setScrolledIndex] = useState(0);
  const [pointedIndex, setPointedIndex] = useState<number | null>(null);

  // The pointer wins over the scroll position: the rail answers the hand
  // when there is one, and the scroll port when there is not.
  const activeIndex = pointedIndex ?? scrolledIndex;

  const setPointed = useCallback((index: number, over: boolean) => {
    setPointedIndex((current) =>
      over ? index : current === index ? null : current,
    );
  }, []);

  // Which book sits nearest the middle of the scroll port.
  useEffect(() => {
    const port = scrollRef.current;
    rowRefs.current.length = books.length;
    if (!port || books.length === 0) return;

    let frame = 0;
    const measure = () => {
      frame = 0;
      const box = port.getBoundingClientRect();
      const mid = box.top + box.height / 2;
      let best = 0;
      let bestGap = Infinity;
      rowRefs.current.forEach((row, i) => {
        if (!row) return;
        const r = row.getBoundingClientRect();
        const gap = Math.abs(r.top + r.height / 2 - mid);
        if (gap < bestGap) {
          bestGap = gap;
          best = i;
        }
      });
      setScrolledIndex(best);
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };

    measure();
    port.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      if (frame) cancelAnimationFrame(frame);
      port.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [books.length]);

  const goTo = useCallback((index: number) => {
    rowRefs.current[index]?.scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "center",
    });
  }, []);

  return (
    <main
      className={returnFlight ? "shelf shelf--returning" : "shelf"}
      ref={attachScroll}
    >
      <a className="skip-link" href="#shelf-stack">
        Skip to the books
      </a>

      <header className="shelf__head">
        <div className="brand brand--static">
          <span className="brand__mark" aria-hidden="true">
            <IconMark size={26} />
          </span>
          <span className="brand__words">
            <h1 className="brand__name">BookTalk</h1>
            <p className="brand__tagline">Ask your books out loud</p>
          </span>
        </div>
      </header>

      <ShelfRail books={books} activeIndex={activeIndex} onGoTo={goTo} />

      {!loaded ? (
        <>
          <ul className="shelf__stack" id="shelf-stack" tabIndex={-1}>
            <ShelfGhostRow />
            <ShelfGhostRow />
          </ul>
          <p className="sr-only" role="status">
            Opening your shelf.
          </p>
        </>
      ) : books.length === 0 ? (
        <div className="shelf__empty" id="shelf-stack" tabIndex={-1}>
          <HeroPanel onUpload={onUpload} />
        </div>
      ) : (
        <ul className="shelf__stack" id="shelf-stack" tabIndex={-1}>
          {books.map((book: BookSummary, i: number) => (
            <ShelfBook
              key={book.book_id}
              book={book}
              chapterCount={tocs[book.book_id]?.chapters.length ?? null}
              index={i}
              returnFrom={
                returnFlight?.bookId === book.book_id ? returnFlight.rect : null
              }
              onBeforeMeasure={restoreScroll}
              onReturned={onReturned}
              onOpen={(from) => onOpen(book.book_id, from)}
              onRemove={() => onRemove(book.book_id)}
              onPoint={(over) => setPointed(i, over)}
              rowRef={(el) => {
                rowRefs.current[i] = el;
              }}
            />
          ))}
          <ShelfAddRow onUpload={onUpload} />
        </ul>
      )}

      <ShelfHelp />
    </main>
  );
}
