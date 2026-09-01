import type { ReactNode } from "react";
import type { BookSummary, CitedPassage, TocResponse } from "../lib/types";
import { isIngesting } from "../hooks/useLibrary";
import { BookPanel } from "./BookPanel";
import { IconMark } from "./Icons";

interface ReaderProps {
  book: BookSummary | null;
  /** The book's position in the library — its cloth comes from there. */
  index: number;
  toc: TocResponse | null;
  /** The book's rect on the shelf, for the flight in. Null after it lands. */
  flightFrom: DOMRect | null;
  onLanded: () => void;
  onBack: () => void;
  /** True while the reader is being left, so its panels can go out. */
  leaving: boolean;
  /** A passage the reader asked to see, or null for the chapter's opening. */
  cited: CitedPassage | null;
  /** The pager moved: whatever citation was showing no longer applies. */
  onChapter: () => void;
  /** The conversation column: thread, states, and the composer dock. */
  children: ReactNode;
}

function StatusChip({
  book,
  chapterCount,
}: {
  book: BookSummary;
  chapterCount: number | null;
}) {
  if (book.status === "ready") {
    return (
      <span className="studio__meta">
        {chapterCount !== null && `${chapterCount} ch · `}
        {book.pages} pp
      </span>
    );
  }
  if (book.status === "failed") {
    return <span className="studio__meta studio__meta--err">Not indexed</span>;
  }
  if (isIngesting(book.status)) {
    return (
      <span className="studio__meta studio__meta--live">
        Indexing {Math.round(book.progress * 100)}%
      </span>
    );
  }
  return null;
}

/**
 * One book, open. The volume flies in from the shelf and splays into a fan of
 * chapter pages on the left; the conversation about it runs on the right.
 */
export function Reader({
  book,
  index,
  toc,
  flightFrom,
  onLanded,
  onBack,
  leaving,
  cited,
  onChapter,
  children,
}: ReaderProps) {
  const chapterCount = toc?.chapters.length ?? null;

  return (
    // While the book is still travelling, everything except it fades in —
    // the shelf fades its own surroundings out the same way, so across the
    // two screens only the cover moves.
    <div
      className={[
        "reader",
        flightFrom && "reader--arriving",
        leaving && "reader--leaving",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <a className="skip-link" href="#composer-input">
        Skip to the question box
      </a>

      {/* The app's name is the way home, as it is on the shelf — the same
          mark, in the same corner, at the same size. Nothing is glued to the
          top edge: the bar keeps the canvas margin every other surface keeps. */}
      <header className="reader__head">
        <button
          type="button"
          className="brand"
          onClick={onBack}
          aria-label="All books"
        >
          <span className="brand__mark" aria-hidden="true">
            <IconMark size={26} />
          </span>
          <span className="brand__words">
            <span className="brand__name">BookTalk</span>
            <span className="brand__tagline">Ask your books out loud</span>
          </span>
        </button>

        {book && (
          <div className="reader__now">
            <h1 className="reader__title">{book.title}</h1>
            <StatusChip book={book} chapterCount={chapterCount} />
          </div>
        )}
      </header>

      {book && (
        <BookPanel
          book={book}
          index={index}
          cited={cited}
          onChapter={onChapter}
          toc={toc}
          from={flightFrom}
          onArrived={onLanded}
        />
      )}

      <main className="studio reader__talk">{children}</main>
    </div>
  );
}
