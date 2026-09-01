import { useState } from "react";
import type { ReactNode } from "react";
import { Collapsible } from "radix-ui";
import { StickToBottom } from "use-stick-to-bottom";
import type { AudioQueue } from "../lib/audioQueue";
import type {
  BookSummary,
  ChatStage,
  CitedPassage,
  Exchange,
  Source,
} from "../lib/types";
import {
  IconArrowRight,
  IconCheck,
  IconChevronDown,
  IconCopy,
  IconMic,
  IconRetry,
  IconStop,
} from "./Icons";
import { SourceCard, sourceLabel } from "./SourceCard";
import { Tip } from "./Tip";
import { Visualizer } from "./Visualizer";

const STAGE_COPY: Record<ChatStage, string> = {
  rewriting: "Rephrasing the question",
  retrieving: "Searching the book",
  generating: "Composing from passages",
  speaking: "Speaking",
};

const SUGGESTIONS = [
  "What is Chapter 1 about?",
  "Summarize the whole book in a few sentences.",
  "What kind of reader is this book for?",
];

/* ---- pieces ------------------------------------------------------- */

/** Exchange ids are minted as `x{n}-{epoch ms}`; recover the ask time. */
function askedAt(exchangeId: string): string | null {
  const ms = Number(exchangeId.split("-").pop());
  if (!Number.isFinite(ms) || ms < 1e12) return null;
  return new Date(ms).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function StageRow({ stage }: { stage: ChatStage | null }) {
  return (
    <>
      <p className="stage-row" aria-live="polite">
        {stage ? STAGE_COPY[stage] : "Thinking"}
        <span className="stage-dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
      </p>
      <div className="skel" aria-hidden="true">
        <i />
        <i />
        <i />
      </div>
    </>
  );
}

function StreamingAnswer({ exchange }: { exchange: Exchange }) {
  return (
    <p className="answer">
      {exchange.segments.map((seg) => (
        <span key={seg.id} className="word">
          {seg.text}
        </span>
      ))}
      <span className="cursor" aria-hidden="true" />
    </p>
  );
}

function FinalAnswer({
  exchange,
  hoveredChunk,
  onHoverChunk,
}: {
  exchange: Exchange;
  hoveredChunk: string | null;
  onHoverChunk: (chunkId: string | null) => void;
}) {
  const { answer, citations } = exchange;
  if (!answer) return null;

  // Highlight by slicing at the server-provided character spans — the
  // server's splitter is the only authority on sentence boundaries, and a
  // client-side re-split disagreed with it on decimals and abbreviations.
  const spans = (citations ?? [])
    .filter(
      (c) =>
        Array.isArray(c.span) &&
        c.span.length === 2 &&
        c.span[0] >= 0 &&
        c.span[1] > c.span[0] &&
        c.span[1] <= answer.length &&
        c.chunk_ids?.length > 0,
    )
    .sort((a, b) => a.span![0] - b.span![0]);

  if (spans.length === 0) {
    return <p className="answer">{answer}</p>;
  }

  const parts: ReactNode[] = [];
  let cursor = 0;
  spans.forEach((c, i) => {
    const [start, end] = c.span!;
    if (start < cursor) return; // overlapping span: keep the prose intact
    if (start > cursor) {
      parts.push(<span key={`gap-${i}`}>{answer.slice(cursor, start)}</span>);
    }
    const lit = hoveredChunk !== null && c.chunk_ids.includes(hoveredChunk);
    parts.push(
      <span
        key={`cite-${i}`}
        className={lit ? "sentence sentence--lit" : "sentence"}
        onMouseEnter={() => onHoverChunk(c.chunk_ids[0])}
        onMouseLeave={() => onHoverChunk(null)}
      >
        {answer.slice(start, end)}
      </span>,
    );
    cursor = end;
  });
  if (cursor < answer.length) {
    parts.push(<span key="tail">{answer.slice(cursor)}</span>);
  }
  return <p className="answer">{parts}</p>;
}

/**
 * One disclosure for the retrieval trace. Wide viewports get compact rows
 * (the excerpts already live in the margin); narrow viewports, where the
 * margin is hidden, get the same margin-note cards inline.
 */
function Sources({
  sources,
  hoveredChunk,
  onHoverChunk,
  onOpenChapter,
}: {
  sources: Source[];
  hoveredChunk: string | null;
  onHoverChunk: (chunkId: string | null) => void;
  onOpenChapter: (passage: CitedPassage) => void;
}) {
  return (
    <Collapsible.Root className="sources">
      <Collapsible.Trigger className="sources__toggle">
        Used {sources.length} {sources.length === 1 ? "source" : "sources"}
        <IconChevronDown size={13} />
      </Collapsible.Trigger>
      <Collapsible.Content className="sources__content">
        <ul className="sources__cards">
          {sources.map((s, i) => (
            <SourceCard
              key={s.chunk_id}
              source={s}
              index={i}
              lit={hoveredChunk === s.chunk_id}
              onHover={onHoverChunk}
              onOpenChapter={onOpenChapter}
            />
          ))}
        </ul>
        <ul className="sources__trace">
          {sources.map((s, i) => (
            <li key={s.chunk_id} className="trace-row">
              <span className="trace-row__num">{i + 1}</span>
              <span className="trace-row__where">{sourceLabel(s)}</span>
              <span className="trace-row__title">
                {s.chapter_title}
                {s.section_title ? ` — ${s.section_title}` : ""}
              </span>
              <span className="trace-row__id">{s.chunk_id}</span>
            </li>
          ))}
        </ul>
      </Collapsible.Content>
    </Collapsible.Root>
  );
}

function timingPills(exchange: Exchange): string[] {
  const t = exchange.timings;
  if (!t) return exchange.status === "stopped" ? ["stopped early"] : [];
  const parts: string[] = [];
  if (typeof t.first_audio_ms === "number") {
    parts.push(`first audio ${(t.first_audio_ms / 1000).toFixed(1)} s`);
  }
  if (typeof t.total_ms === "number") {
    parts.push(`total ${(t.total_ms / 1000).toFixed(1)} s`);
  }
  return parts;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Tip label={copied ? "Copied" : "Copy answer"}>
      <button
        type="button"
        className="icon-btn"
        aria-label="Copy answer"
        onClick={() => {
          void navigator.clipboard?.writeText(text).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1600);
          });
        }}
      >
        {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
      </button>
    </Tip>
  );
}

/* ---- one exchange ------------------------------------------------- */

interface ExchangeViewProps {
  exchange: Exchange;
  isLast: boolean;
  audible: boolean;
  audioQueue: AudioQueue;
  onRetry: () => void;
  onStop: () => void;
  onOpenChapter: (passage: CitedPassage) => void;
}

function ExchangeView({
  exchange,
  isLast,
  audible,
  audioQueue,
  onRetry,
  onStop,
  onOpenChapter,
}: ExchangeViewProps) {
  const [hoveredChunk, setHoveredChunk] = useState<string | null>(null);
  const streaming = exchange.status === "streaming";
  const hasText = exchange.segments.length > 0 || exchange.answer !== null;
  const pills = timingPills(exchange);
  const time = askedAt(exchange.id);
  const sources = exchange.sources ?? [];
  const showSources = sources.length > 0 && !streaming;

  return (
    <article
      className={isLast ? "exchange exchange--last" : "exchange"}
      aria-label={`Question: ${exchange.question}`}
    >
      <div className="exchange__head">
        {/* A long question is a long shape: the pill is right for one line
            and wrong for six, so the row squares off past a threshold the
            text itself decides. */}
        <h3
          className={
            exchange.question.length > 88 ? "qrow qrow--long" : "qrow"
          }
        >
          {exchange.via === "voice" && (
            <>
              <IconMic size={13} />
              <span className="sr-only">Asked by voice.</span>
            </>
          )}
          <span className="qrow__text">{exchange.question}</span>
          {time && <time className="qrow__time">{time}</time>}
        </h3>
        {exchange.rewritten && (
          <p className="qrow__rewritten">Read as “{exchange.rewritten}”</p>
        )}
      </div>

      <div className="exchange__main">
        <div className="answer-card">
          {/* Grounding, measured on the answer rather than on the query. The
              retrieval score cannot carry it: on a real book it read 0.0017
              for "what happens in chapter 2", answered correctly from four
              cited passages, and 0.9953 for a subject the book never
              mentions — wrong in both directions. How many of the answer's
              own sentences could be traced back to a passage separates the
              two cleanly. A refusal is short and cites nothing by design, so
              the length test keeps it out of this. */}
          {!streaming &&
            exchange.citations !== null &&
            exchange.citations.length === 0 &&
            (exchange.answer?.length ?? 0) > 80 && (
              <p className="lowconf" role="status">
                None of this answer could be traced to a passage — check it
                against the book.
              </p>
            )}
          {streaming && !hasText && <StageRow stage={exchange.stage} />}
          {streaming && exchange.segments.length > 0 && (
            <StreamingAnswer exchange={exchange} />
          )}
          {!streaming && (
            <FinalAnswer
              exchange={exchange}
              hoveredChunk={hoveredChunk}
              onHoverChunk={setHoveredChunk}
            />
          )}

          {(streaming || audible) && (
            <div className="speaking-row">
              {audible && (
                <>
                  <Visualizer
                    ctx={audioQueue.context}
                    source={audioQueue.outputNode}
                    className="viz--speaking"
                  />
                  <span className="speaking-row__label">Speaking</span>
                </>
              )}
              <button type="button" className="btn btn--stop" onClick={onStop}>
                <IconStop size={12} />
                Stop
              </button>
            </div>
          )}

          {exchange.status === "error" && (
            <div className="answer-error" role="alert">
              <span>{exchange.error ?? "Something went wrong."}</span>
              <button
                type="button"
                className="btn btn--quiet btn--sm"
                onClick={onRetry}
              >
                <IconRetry size={13} />
                Retry
              </button>
            </div>
          )}

          {(pills.length > 0 || showSources || exchange.answer) && (
            <div className="answer-card__footer">
              {(pills.length > 0 || exchange.answer) && (
                <div className="answer-card__meta">
                  <span className="stat-pills">
                    {pills.map((p) => (
                      <span key={p} className="stat-pill">
                        {p}
                      </span>
                    ))}
                  </span>
                  {exchange.answer && <CopyButton text={exchange.answer} />}
                </div>
              )}
              {showSources && (
                <Sources
                  sources={sources}
                  hoveredChunk={hoveredChunk}
                  onHoverChunk={setHoveredChunk}
                  onOpenChapter={onOpenChapter}
                />
              )}
            </div>
          )}
        </div>
      </div>

      <aside className="exchange__notes" aria-label="Sources from the book">
        {showSources && (
          <ul>
            {sources.map((s, i) => (
              <SourceCard
                key={s.chunk_id}
                source={s}
                index={i}
                lit={hoveredChunk === s.chunk_id}
                onHover={setHoveredChunk}
                onOpenChapter={onOpenChapter}
              />
            ))}
          </ul>
        )}
      </aside>
    </article>
  );
}

/* ---- conversation ------------------------------------------------- */

interface ConversationProps {
  thread: Exchange[];
  book: BookSummary;
  chapterCount: number | null;
  speakingExchangeId: string | null;
  audioQueue: AudioQueue;
  onAsk: (question: string) => void;
  onRetry: (exchangeId: string) => void;
  onStop: () => void;
  /** Open a cited passage's chapter on the book beside the conversation. */
  onOpenChapter: (passage: CitedPassage) => void;
}

export function Conversation({
  onOpenChapter,
  thread,
  book,
  chapterCount,
  speakingExchangeId,
  audioQueue,
  onAsk,
  onRetry,
  onStop,
}: ConversationProps) {
  if (thread.length === 0) {
    return (
      <div className="studio__body studio__body--center">
        <section className="starter" aria-label="Suggested questions">
          <div className="ready-card">
            <span className="ready-card__check" aria-hidden="true">
              <IconCheck size={13} />
            </span>
            <span className="ready-card__text">Indexed and ready</span>
            <span className="ready-card__meta">
              {chapterCount !== null && `${chapterCount} ch · `}
              {book.pages} pp
            </span>
          </div>

          <h2 className="starter__title">Ask this book something</h2>
          <p className="starter__lead">
            Answers come only from its pages — spoken aloud, with chapter and
            page citations.
          </p>
          <div className="starter__list">
            {SUGGESTIONS.map((q) => (
              <button
                key={q}
                type="button"
                className="suggestion"
                onClick={() => onAsk(q)}
              >
                <span className="suggestion__edge" aria-hidden="true" />
                <span className="suggestion__text">{q}</span>
                <IconArrowRight size={16} />
              </button>
            ))}
          </div>
        </section>
      </div>
    );
  }

  return (
    <StickToBottom className="thread" resize="smooth" initial="instant">
      <StickToBottom.Content className="thread__inner">
        {thread.map((x, i) => (
          <ExchangeView
            key={x.id}
            exchange={x}
            isLast={i === thread.length - 1}
            audible={speakingExchangeId === x.id}
            audioQueue={audioQueue}
            onRetry={() => onRetry(x.id)}
            onStop={onStop}
            onOpenChapter={onOpenChapter}
          />
        ))}
      </StickToBottom.Content>
    </StickToBottom>
  );
}
