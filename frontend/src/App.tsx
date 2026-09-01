import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type { CitedPassage } from "./lib/types";
import { Tooltip } from "radix-ui";
import { AudioQueue } from "./lib/audioQueue";
import { getHealth } from "./lib/api";
import { useChat } from "./hooks/useChat";
import { isIngesting, useLibrary } from "./hooks/useLibrary";
import { Composer } from "./components/Composer";
import { Conversation } from "./components/Conversation";
import { Hero } from "./components/Hero";
import { IngestPanel } from "./components/IngestPanel";
import { Shelf } from "./components/Shelf";
import { Reader } from "./components/Reader";
import { Toasts } from "./components/Toast";
import type { ToastItem } from "./components/Toast";

let toastCounter = 0;

/** Two screens, no router: the shelf people land on, and one book's studio. */
type View = "shelf" | "chat";

export default function App() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [keyMissing, setKeyMissing] = useState(false);
  const [view, setView] = useState<View>("shelf");

  const pushToast = useCallback((message: string) => {
    setToasts((prev) => {
      if (prev.some((t) => t.message === message)) return prev;
      return [...prev, { id: ++toastCounter, message }];
    });
  }, []);

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const audioQueue = useMemo(() => new AudioQueue(), []);
  const chat = useChat(audioQueue, pushToast);

  const chatRef = useRef(chat);
  chatRef.current = chat;
  const onRemoved = useCallback((bookId: string) => {
    chatRef.current.clearThread(bookId);
  }, []);

  const library = useLibrary(pushToast, onRemoved);
  const { activeBook, activeBookId } = library;
  const activeBookIdRef = useRef(activeBookId);
  activeBookIdRef.current = activeBookId;

  // Switching books ends the running turn, silences playback, and drops the
  // passage the previous book was turned to — chapter numbers mean nothing
  // across books, so it would have opened an unrelated chapter.
  useEffect(() => {
    chatRef.current.stopAll();
    setCited(null);
  }, [activeBookId]);

  // One health probe at startup.
  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((h) => {
        if (!cancelled) setKeyMissing(!h.openai_configured);
      })
      .catch(() => {
        if (!cancelled) {
          pushToast(
            "Can't reach the BookTalk server — start the backend or `npm run mock` on port 8000.",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [pushToast]);

  // Opening a book is the only way into the studio; leaving it also ends the
  // spoken answer, so nothing keeps talking from a screen the user has left.
  const { selectBook } = library;
  // Where the book was standing when it was clicked, so the reader can fly it
  // in from exactly there rather than cutting to a new screen.
  const [flightFrom, setFlightFrom] = useState<DOMRect | null>(null);
  // Where the shelf was left, so returning lands on the same row rather
  // than at the top of the stack.
  const shelfScrollRef = useRef(0);
  const openBook = useCallback(
    (bookId: string, from?: DOMRect | null) => {
      const shelf = document.querySelector(".shelf");
      shelfScrollRef.current = shelf ? shelf.scrollTop : 0;
      selectBook(bookId);
      setFlightFrom(from ?? null);
      setView("chat");
    },
    [selectBook],
  );

  // Where the cover is standing when the reader is left, so the shelf can
  // fly the same object back into its row instead of cutting to a list.
  const [returnFlight, setReturnFlight] = useState<{
    bookId: string;
    rect: DOMRect;
  } | null>(null);

  const [leavingReader, setLeavingReader] = useState(false);
  // What the book beside the conversation is showing. A cited passage sets
  // the whole thing — chapter, pages and the passage's own words — because
  // "the chapter it came from" is not where the answer came from. The pager
  // under the book clears it back to the chapter's opening.
  const [cited, setCited] = useState<CitedPassage | null>(null);

  const backToShelf = useCallback(() => {
    chatRef.current.stopAll();
    const cover = document.querySelector(".tome__block");
    const id = activeBookIdRef.current;
    setReturnFlight(
      cover && id ? { bookId: id, rect: cover.getBoundingClientRect() } : null,
    );
    setFlightFrom(null);
    // The shelf takes over at once, so the book starts morphing back
    // immediately; the reader stays up as a departing layer above it for as
    // long as its own exit takes, which is what makes the two simultaneous
    // rather than one after the other.
    setLeavingReader(true);
    setView("shelf");
    window.setTimeout(() => setLeavingReader(false), 300);
  }, []);

  // Removing the last book from the studio leaves nothing to be in: the
  // shelf is the only screen that can stand with an empty library.
  useEffect(() => {
    if (view === "chat" && library.loaded && !activeBookId) setView("shelf");
  }, [view, library.loaded, activeBookId]);

  const ready = activeBook?.status === "ready";
  const thread = activeBookId ? (chat.threads[activeBookId] ?? []) : [];

  const ask = useCallback(
    (question: string, via: "voice" | "text") => {
      if (!activeBookId) return;
      chatRef.current.ask(activeBookId, question, via);
    },
    [activeBookId],
  );

  // Keeps the composer anchored to the bottom in transitional states.
  let mainContent = <div className="studio__body" aria-hidden="true" />;
  if (library.loaded) {
    if (library.books.length === 0) {
      mainContent = <Hero onUpload={library.upload} />;
    } else if (activeBook && (isIngesting(activeBook.status) || activeBook.status === "failed")) {
      mainContent = <IngestPanel book={activeBook} />;
    } else if (ready && activeBookId && activeBook) {
      mainContent = (
        <Conversation
          thread={thread}
          book={activeBook}
          chapterCount={library.tocs[activeBookId]?.chapters.length ?? null}
          speakingExchangeId={
            chat.speaking?.bookId === activeBookId
              ? chat.speaking.exchangeId
              : null
          }
          audioQueue={audioQueue}
          onAsk={(q) => ask(q, "text")}
          onRetry={(id) => chat.retry(activeBookId, id)}
          onStop={chat.stopAll}
          onOpenChapter={setCited}
        />
      );
    }
  }

  /* Three colour fields drifting behind both screens. The shelf runs them at
     full strength; the reader turns them down to the calm wash the reading
     surface is designed on, so moving between the screens is a change of
     attention rather than a change of place. */
  const ground = (
    <div
      className="app-ground"
      style={{ "--ground-strength": view === "shelf" ? 1 : 0.28 } as CSSProperties}
      aria-hidden="true"
    >
      <span />
      <span />
      <span />
    </div>
  );

  if (view === "shelf") {
    return (
      <Tooltip.Provider delayDuration={350}>
        {ground}
        {/* The reader on its way out, above the shelf that is already
            bringing the book home. It is scenery at this point: no pointer,
            no book panel — that object is on the shelf now. */}
        {leavingReader && activeBook && (
          <Reader
            book={activeBook}
            index={library.books.findIndex((b) => b.book_id === activeBookId)}
            toc={activeBookId ? (library.tocs[activeBookId] ?? null) : null}
            flightFrom={null}
            leaving
            cited={cited}
            onChapter={() => undefined}
            onLanded={() => undefined}
            onBack={() => undefined}
          >
            <div className="studio__body" aria-hidden="true" />
          </Reader>
        )}
        <Shelf
          books={library.books}
          loaded={library.loaded}
          tocs={library.tocs}
          returnFlight={returnFlight}
          onReturned={() => setReturnFlight(null)}
          initialScroll={shelfScrollRef.current}
          onOpen={openBook}
          onUpload={(f) => void library.upload(f)}
          onRemove={(id) => void library.remove(id)}
        />
        <Toasts toasts={toasts} onDismiss={dismissToast} />
      </Tooltip.Provider>
    );
  }

  return (
    <Tooltip.Provider delayDuration={350}>
      {ground}
      <Reader
        book={activeBook}
        index={library.books.findIndex((b) => b.book_id === activeBookId)}
        toc={activeBookId ? (library.tocs[activeBookId] ?? null) : null}
        flightFrom={flightFrom}
        leaving={leavingReader}
        cited={cited}
        onChapter={() => setCited(null)}
        onLanded={() => setFlightFrom(null)}
        onBack={backToShelf}
      >
        {mainContent}
        {/* The dock floats over the scroll region: the conversation runs
            underneath the frosted composer rather than stopping above it. */}
        <div className="studio__dock">
          {keyMissing && (
            <p className="notice notice--floating">
              The server is running without an OpenAI key — uploads work, but
              transcription and answers will fail.
            </p>
          )}
          <Composer
            block={
              ready
                ? null
                : activeBook?.status === "failed"
                  ? "failed"
                  : activeBook
                    ? "ingesting"
                    : "empty-library"
            }
            bookId={activeBook?.book_id ?? null}
            onAsk={ask}
            onError={pushToast}
            primeAudio={() => audioQueue.start()}
          />
        </div>
      </Reader>
      <Toasts toasts={toasts} onDismiss={dismissToast} />
    </Tooltip.Provider>
  );
}
