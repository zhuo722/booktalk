import { useLayoutEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import type { BookSummary } from "../lib/types";
import { isIngesting } from "../hooks/useLibrary";
import { STAGE_LABELS, clothStyle } from "./cloth";
import { IconCheck, IconClose, IconPlus, IconTrash } from "./Icons";
import { Tip } from "./Tip";

/**
 * One row of the shelf: a book rendered as a volume lying flat, seen from
 * slightly above. The pose is pure CSS 3D — the row carries the perspective,
 * the volume carries the rotateX, and the object is built from a top face
 * (the cloth, the type, the mark), a front page-edge standing at 90° to it,
 * and the ground shadow the face casts into the table plane.
 *
 * BookSummary has no author field, so the slot the reference gives to the
 * author carries the book's own measurements in the same words the library
 * rail uses — "6 ch · 25 pp", "reading pages 12%", "not indexed" — set in
 * mono caps. Inventing an author would be the only alternative.
 */

function metaLine(book: BookSummary, chapterCount: number | null): string {
  if (isIngesting(book.status)) {
    const label = STAGE_LABELS[book.status] ?? book.status;
    return `${label} ${Math.round(book.progress * 100)}%`;
  }
  if (book.status === "failed") return "Not indexed";
  return chapterCount !== null
    ? `${chapterCount} ch · ${book.pages} pp`
    : `${book.pages} pp`;
}

interface ShelfBookProps {
  book: BookSummary;
  /** Position on the shelf — the palette is walked in order, not hashed. */
  index: number;
  /** Where this book's cover was in the reader, when returning from it. */
  returnFrom: DOMRect | null;
  onReturned: () => void;
  /** Put the shelf back where it was left, before this row is measured. */
  onBeforeMeasure: () => void;
  chapterCount: number | null;
  /** Opens the book, handing over the volume's rect so the reader can fly
   * it in from where it stood rather than cutting to a new screen. */
  onOpen: (from: DOMRect | null) => void;
  onRemove: () => void;
  onPoint: (over: boolean) => void;
  rowRef: (el: HTMLLIElement | null) => void;
}

export function ShelfBook({
  book,
  index,
  returnFrom,
  onReturned,
  onBeforeMeasure,
  chapterCount,
  onOpen,
  onRemove,
  onPoint,
  rowRef,
}: ShelfBookProps) {
  const [confirming, setConfirming] = useState(false);
  // Opening happens here, not on the next screen: the book turns square,
  // its leaves spread, and only then does the reader take over the flight
  // from wherever the volume has come to rest.
  const [opening, setOpening] = useState(false);
  const volumeRef = useRef<HTMLSpanElement>(null);

  // Coming back: the same object, the same morph, run the other way — from
  // the upright cover in the reader to this row on the shelf.
  useLayoutEffect(() => {
    const node = volumeRef.current;
    if (!node || !returnFrom) return;
    const finish = () => onReturned();
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      finish();
      return;
    }
    // The shelf has to be scrolled to where the reader left it before this
    // row can say where it is.
    onBeforeMeasure();
    node.getAnimations().forEach((a) => a.cancel());
    // Measure the book flat. Its resting rect is the bounding box of a
    // volume tilted 46°, which is 103px tall for a book that is 148px tall —
    // ending the animation on that number left the book squashed. The
    // transform comes off for one measurement and goes straight back.
    const previous = node.style.transform;
    node.style.transform = "none";
    const flat = node.getBoundingClientRect();
    node.style.transform = previous;
    if (!flat.width || !flat.height) {
      finish();
      return;
    }
    const dx =
      returnFrom.left +
      returnFrom.width / 2 -
      (flat.left + returnFrom.width / 2);
    const dy =
      returnFrom.top +
      returnFrom.height / 2 -
      (flat.top + returnFrom.height / 2);
    const flight = node.animate(
      [
        {
          width: `${returnFrom.width}px`,
          height: `${returnFrom.height}px`,
          transform: `translate(${dx}px, ${dy}px) rotateX(0deg) rotateZ(0deg)`,
        },
        {
          width: `${flat.width}px`,
          height: `${flat.height}px`,
          transform:
            "translate(0px, 0px) rotateX(var(--shelf-tilt)) rotateZ(var(--shelf-lean))",
        },
      ],
      { duration: 620, easing: "cubic-bezier(.4,0,.2,1)", fill: "backwards" },
    );
    let done = false;
    const land = () => {
      if (done) return;
      done = true;
      finish();
    };
    flight.finished.then(land).catch(() => undefined);
    const fallback = window.setTimeout(land, 820);
    return () => window.clearTimeout(fallback);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [returnFrom]);
  const ingesting = isIngesting(book.status);
  const failed = book.status === "failed";
  const pct = Math.round(book.progress * 100);
  const meta = metaLine(book, chapterCount);

  const className = [
    "shelf-book",
    opening && "shelf-book--opening",
    ingesting && "shelf-book--working",
    failed && "shelf-book--failed",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <li
      className="shelf-row"
      ref={rowRef}
      onPointerEnter={() => onPoint(true)}
      onPointerLeave={() => onPoint(false)}
    >
      <button
        type="button"
        className={className}
        style={clothStyle(index)}
        /* The visible strings, in the order a reader meets them, so the
           accessible name contains the visible label (WCAG 2.5.3). */
        aria-label={`Open ${book.title} · ${meta}`}
        onClick={(e) => {
          const volume = e.currentTarget.querySelector(".shelf-book__volume");
          if (!volume || opening) return;
          setOpening(true);
          const spread = window.matchMedia("(prefers-reduced-motion: reduce)")
            .matches
            ? 0
            : 280;
          // Hand over the rect the volume actually occupies at the end of the
          // spread, so the flight starts from where the book is, not from
          // where it was before it opened.
          window.setTimeout(() => {
            onOpen(volume.getBoundingClientRect());
          }, spread);
        }}
        onFocus={() => onPoint(true)}
        onBlur={() => onPoint(false)}
      >
        <span className="shelf-book__volume" ref={volumeRef}>
          {/* Three leaves that only exist while the book is opening: they
              spread out of the block, and the reader picks the movement up
              from there. */}
          <span className="shelf-book__leaves" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span className="shelf-book__face">
            <span className="shelf-book__cloth" aria-hidden="true" />
            <span className="shelf-book__title">{book.title}</span>
            <span className="shelf-book__meta">{meta}</span>
          </span>

          {ingesting ? (
            <span
              className="shelf-book__edge"
              role="progressbar"
              aria-valuenow={pct}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`Indexing ${book.title}`}
            >
              <span
                className="shelf-book__progress"
                style={{ width: `${pct}%` }}
              />
            </span>
          ) : (
            <span className="shelf-book__edge" aria-hidden="true" />
          )}
        </span>
      </button>

      {/* Removing is confirmed where it is asked: the bin turns red under
          the pointer, and clicking it opens a tick and a cross beside it
          rather than moving the decision somewhere else on the row. */}
      <div
        className={
          confirming ? "shelf-row__bin shelf-row__bin--open" : "shelf-row__bin"
        }
      >
        {confirming ? (
          <>
            <span className="shelf-row__ask">Remove?</span>
            <button
              type="button"
              className="icon-btn icon-btn--danger"
              aria-label={`Remove ${book.title}`}
              onClick={() => {
                setConfirming(false);
                onRemove();
              }}
            >
              <IconCheck size={15} />
            </button>
            <button
              type="button"
              className="icon-btn"
              aria-label="Keep this book"
              onClick={() => setConfirming(false)}
            >
              <IconClose size={15} />
            </button>
          </>
        ) : (
          <Tip label="Remove from library">
            <button
              type="button"
              className="icon-btn shelf-row__delete"
              aria-label={`Remove ${book.title}`}
              onClick={() => setConfirming(true)}
            >
              <IconTrash size={15} />
            </button>
          </Tip>
        )}
      </div>

      {failed && (
        <div className="shelf-row__foot">
          <p className="shelf-row__error">
            Indexing failed — remove and re-upload.
          </p>
        </div>
      )}
    </li>
  );
}

/**
 * The last row of the stack: the same volume in the same pose, dashed and
 * empty, waiting for a book. It is both the click target and the drop target.
 */
export function ShelfAddRow({ onUpload }: { onUpload: (file: File) => void }) {
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onUpload(file);
  };

  return (
    <li className="shelf-row">
      <button
        type="button"
        className={
          dragging ? "shelf-book shelf-book--add is-drag" : "shelf-book shelf-book--add"
        }
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <span className="shelf-book__volume">
          <span className="shelf-book__face">
            <span className="shelf-book__cloth" aria-hidden="true" />
            <span className="shelf-book__add">
              <span className="shelf-book__add-plus" aria-hidden="true">
                <IconPlus size={15} />
              </span>
              <span className="shelf-book__add-label">Add a book</span>
              <span className="shelf-book__add-hint">
                drop a PDF or click · up to 200 MB
              </span>
            </span>
          </span>
          <span className="shelf-book__edge" aria-hidden="true" />
        </span>
      </button>
      <input
        ref={fileRef}
        type="file"
        accept="application/pdf,.pdf"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onUpload(file);
          e.target.value = "";
        }}
      />
    </li>
  );
}

/** Two volumes with no colour and no type, held while the library loads. */
export function ShelfGhostRow() {
  return (
    <li className="shelf-row" aria-hidden="true">
      <div className="shelf-book shelf-book--ghost">
        <span className="shelf-book__volume">
          <span className="shelf-book__face">
            <span className="shelf-book__cloth" />
          </span>
          <span className="shelf-book__edge" />
        </span>
      </div>
    </li>
  );
}
