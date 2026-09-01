import type { BookSummary } from "../lib/types";

interface ShelfRailProps {
  books: BookSummary[];
  /** Index under the pointer, or nearest the middle of the scroll port. */
  activeIndex: number;
  onGoTo: (index: number) => void;
}

/**
 * The fixed index at the far left: one tick per book, standing in for both
 * the shelf's length and the reader's place in it. A single book has nothing
 * to index, so the rail is taken out of the accessibility tree and out of the
 * tab order rather than offering a control that goes nowhere.
 */
export function ShelfRail({ books, activeIndex, onGoTo }: ShelfRailProps) {
  if (books.length === 0) return null;
  const inert = books.length === 1;

  return (
    <nav
      className="shelf-rail"
      aria-label="Books on the shelf"
      aria-hidden={inert || undefined}
    >
      {books.map((book, i) => (
        <button
          key={book.book_id}
          type="button"
          className="shelf-rail__tick"
          aria-label={`Scroll to ${book.title}`}
          aria-current={i === activeIndex ? "true" : undefined}
          tabIndex={inert ? -1 : undefined}
          onClick={() => onGoTo(i)}
        />
      ))}
    </nav>
  );
}
