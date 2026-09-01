/**
 * Library state: book list, uploads, 1s ingestion polling, TOC cache.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  deleteBook as apiDeleteBook,
  getBookStatus,
  getToc,
  listBooks,
  uploadBook,
} from "../lib/api";
import type { BookSummary, TocResponse } from "../lib/types";

// Every non-terminal status the backend can report (see its ingestor stages);
// missing one here silently stops the status polling mid-ingest.
const INGESTING = new Set([
  "parsing",
  "structuring",
  "chunking",
  "contextualizing",
  "summarizing",
  "embedding",
]);

export function isIngesting(status: string): boolean {
  return INGESTING.has(status);
}

export interface LibraryApi {
  books: BookSummary[];
  loaded: boolean;
  activeBookId: string | null;
  activeBook: BookSummary | null;
  tocs: Record<string, TocResponse>;
  selectBook: (bookId: string | null) => void;
  upload: (file: File) => Promise<void>;
  remove: (bookId: string) => Promise<void>;
}

export function useLibrary(
  onError: (message: string) => void,
  onRemoved: (bookId: string) => void,
): LibraryApi {
  const [books, setBooks] = useState<BookSummary[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [activeBookId, setActiveBookId] = useState<string | null>(null);
  const [tocs, setTocs] = useState<Record<string, TocResponse>>({});

  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;
  const booksRef = useRef(books);
  booksRef.current = books;
  const tocsRef = useRef(tocs);
  tocsRef.current = tocs;

  const refresh = useCallback(async () => {
    const list = await listBooks();
    setBooks(list);
    return list;
  }, []);

  // Initial load; auto-select the most recent ready book.
  useEffect(() => {
    let cancelled = false;
    refresh()
      .then((list) => {
        if (cancelled) return;
        setLoaded(true);
        const first = list.find((b) => b.status === "ready") ?? list[0];
        if (first) setActiveBookId(first.book_id);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setLoaded(true);
        onErrorRef.current(
          err instanceof ApiError ? err.message : "Can't load the library.",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [refresh]);

  // Fetch TOCs for ready books we haven't seen yet.
  useEffect(() => {
    for (const book of books) {
      if (book.status !== "ready" || tocsRef.current[book.book_id]) continue;
      getToc(book.book_id)
        .then((toc) => setTocs((prev) => ({ ...prev, [book.book_id]: toc })))
        .catch(() => {
          /* TOC is enhancement; the card still works */
        });
    }
  }, [books]);

  // Poll /status every 1s while any book is ingesting.
  useEffect(() => {
    if (!books.some((b) => isIngesting(b.status))) return;
    const timer = setInterval(() => {
      const ingesting = booksRef.current.filter((b) => isIngesting(b.status));
      void Promise.all(
        ingesting.map(async (book) => {
          try {
            const s = await getBookStatus(book.book_id);
            setBooks((prev) =>
              prev.map((b) =>
                b.book_id === book.book_id
                  ? { ...b, status: s.status, progress: s.progress }
                  : b,
              ),
            );
            if (s.status === "ready") await refresh();
            if (s.status === "failed" && s.error) onErrorRef.current(s.error);
          } catch {
            /* transient poll failure; next tick retries */
          }
        }),
      );
    }, 1000);
    return () => clearInterval(timer);
  }, [books, refresh]);

  const upload = useCallback(
    async (file: File) => {
      if (!/\.pdf$/i.test(file.name)) {
        onErrorRef.current("Only PDF files are supported.");
        return;
      }
      try {
        const res = await uploadBook(file);
        const list = await refresh();
        const uploaded = list.find((b) => b.book_id === res.book_id);
        if (uploaded) setActiveBookId(uploaded.book_id);
      } catch (err) {
        onErrorRef.current(
          err instanceof ApiError ? err.message : "Upload failed.",
        );
      }
    },
    [refresh],
  );

  const remove = useCallback(
    async (bookId: string) => {
      try {
        await apiDeleteBook(bookId);
      } catch (err) {
        onErrorRef.current(
          err instanceof ApiError ? err.message : "Couldn't remove the book.",
        );
        return;
      }
      onRemoved(bookId);
      setTocs((prev) => {
        const next = { ...prev };
        delete next[bookId];
        return next;
      });
      setBooks((prev) => {
        const next = prev.filter((b) => b.book_id !== bookId);
        setActiveBookId((current) =>
          current === bookId ? (next[0]?.book_id ?? null) : current,
        );
        return next;
      });
    },
    [onRemoved],
  );

  const activeBook = books.find((b) => b.book_id === activeBookId) ?? null;

  return {
    books,
    loaded,
    activeBookId,
    activeBook,
    tocs,
    selectBook: setActiveBookId,
    upload,
    remove,
  };
}
