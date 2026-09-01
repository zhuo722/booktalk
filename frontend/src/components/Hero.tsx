import { useRef, useState } from "react";
import type { DragEvent } from "react";
import { IconUpload } from "./Icons";

interface HeroProps {
  onUpload: (file: File) => void;
}

/**
 * Empty-library state: one display line, the pipeline in a sentence, a drop
 * target. The panel is shared — the Shelf centres it in its own quiet when
 * there is nothing on the shelf yet, and the studio wraps it in its scroll
 * region. One component, so the two screens cannot drift apart.
 */
export function HeroPanel({ onUpload }: HeroProps) {
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onUpload(file);
  };

  return (
    <section
      className="hero"
      aria-label="Upload a book"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
        <span className="hero__accent" aria-hidden="true" />
      <h2 className="hero__display">
        Every book can talk.
        <br />
        Give it a voice.
      </h2>
      <p className="hero__lead">
        BookTalk maps a PDF's chapters, indexes every passage, and answers your
        spoken questions aloud — citing the pages it drew from.
      </p>

      <div className={dragging ? "dropzone dropzone--drag" : "dropzone"}>
        <span className="dropzone__wash" aria-hidden="true" />
        <div className="dropzone__inner">
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => fileRef.current?.click()}
          >
            <IconUpload size={15} />
            Choose a PDF
          </button>
          <p className="dropzone__hint">
            or drop one anywhere on this panel · up to 200 MB
          </p>
        </div>
      </div>

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
    </section>
  );
}

/** The studio's empty-library screen: the shared panel in the scroll region. */
export function Hero({ onUpload }: HeroProps) {
  return (
    <div className="studio__body studio__body--center">
      <HeroPanel onUpload={onUpload} />
    </div>
  );
}
