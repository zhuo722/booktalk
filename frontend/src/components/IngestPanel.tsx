import type { BookSummary, BookStatus } from "../lib/types";
import { StatTile } from "./StatTile";

const STAGES = [
  { key: "parsing", label: "Read", sub: "Extracting the text page by page" },
  {
    key: "structuring",
    label: "Map",
    sub: "Finding chapter and section headings",
  },
  {
    key: "chunking",
    label: "Cut",
    sub: "Cutting the text into retrievable passages",
  },
  {
    key: "contextualizing",
    label: "Situate",
    sub: "Writing where-in-the-book context for each passage",
  },
  {
    key: "summarizing",
    label: "Distill",
    sub: "Summarizing every chapter for thematic questions",
  },
  {
    key: "embedding",
    label: "Index",
    sub: "Embedding every passage for search",
  },
] as const;

interface IngestPanelProps {
  book: BookSummary;
}

/** Main-pane view while the selected book is being ingested. */
export function IngestPanel({ book }: IngestPanelProps) {
  const currentIdx = STAGES.findIndex((s) => s.key === book.status);
  const pct = Math.round(book.progress * 100);
  const failed = book.status === "failed";
  const stage = STAGES[currentIdx] ?? STAGES[0];

  if (failed) {
    return (
      <div className="studio__body studio__body--center">
        <section className="ingest" aria-label={`Preparing ${book.title}`}>
          <h2 className="ingest__title">This book couldn't be prepared.</h2>
          <p className="ingest__error" role="alert">
            Indexing failed — the file may be malformed or image-only. Remove it
            from the library and try another PDF.
          </p>
        </section>
      </div>
    );
  }

  return (
    <div className="studio__body studio__body--center">
      <section className="ingest" aria-label={`Preparing ${book.title}`}>
        <h2 className="ingest__title">Preparing this book</h2>
        <p className="ingest__sub">
          Getting the book ready to answer — this runs once per upload.
        </p>

        <StatTile
          label={statusLabel(book.status)}
          value={pct}
          sub={
            book.pages > 0 ? `${stage.sub} · ${book.pages} pages` : stage.sub
          }
        />

        <ol
          className="steps"
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Overall indexing progress"
        >
          {STAGES.map((s, i) => {
            const state =
              i < currentIdx ? "done" : i === currentIdx ? "current" : "todo";
            return (
              <li key={s.key} className={`step step--${state}`}>
                <span className="step__track" aria-hidden="true" />
                <span className="step__label">{s.label}</span>
              </li>
            );
          })}
        </ol>
        <p className="steps__live sr-only" aria-live="polite">
          {statusLabel(book.status)} {pct}%
        </p>
      </section>
    </div>
  );
}

function statusLabel(status: BookStatus): string {
  return status.toUpperCase();
}
