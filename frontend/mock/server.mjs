/**
 * BookTalk mock backend — plain node http on :8000.
 *
 * Implements the full SPEC §6 contract with canned data plus the revised SSE
 * stream (turn_id on every event, base64 PCM audio segments, a trailing
 * `citations` event), so the UI is fully drivable without the real backend
 * or an API key:  npm run mock
 */

import http from "node:http";
import { randomUUID } from "node:crypto";

const PORT = 8000;
const SAMPLE_RATE = 24000;

/* ---------------- canned library ---------------- */

/**
 * Chapters carry a `synopsis` only when the real backend summarized them
 * (SPEC §14.5), so two of these deliberately have none: the TOC has to render
 * a mixed list, not an all-or-nothing one.
 *
 * `opening` is a different thing and is always present here: the chapter's
 * own first passage, lifted from the book rather than generated, which is
 * what the reader prints on a chapter page. The real endpoint sends up to
 * 900 characters of it and omits the key only for a chapter with no indexed
 * body text; without it here the reader fell back to the synopsis — a blurb
 * where the book's own words belong — and printed its empty state for the
 * two chapters that have no synopsis either.
 */
const MERIDIAN_TOC = {
  title: "A Field Guide to the Meridian Archipelago",
  chapters: [
    [
      "The Shape of the Archipelago",
      9,
      26,
      "Introduces the twelve islands as a single working system rather than a " +
        "scatter of landmasses: the ferry timetable, the prevailing weather, " +
        "and the credit that moves between harbours are what hold them together.",
      "Twelve islands, two tide races, and one improbable ferry timetable: the " +
        "archipelago is best read as a network rather than a place, and this " +
        "guide keeps returning to that claim. Nothing here is more than four " +
        "hours from anything else, provided the boat runs, and the whole of " +
        "the argument that follows sits in that provision. Draw the islands as " +
        "dots on a chart and they look like debris. Draw the crossings between " +
        "them and a shape appears — one that has more to do with debt, weather " +
        "and slack water than with geology.",
    ],
    [
      "Veldra and the Northern Approaches",
      27,
      40,
      "The northern channel and the pilots who read it. Veldra's harbour dues " +
        "financed the first survey, and the chapter follows that money into the " +
        "chartmaking that follows.",
      "The approach to Veldra is the one every pilot on the northern run learns " +
        "first and describes last, because there is no describing it: the " +
        "channel bends twice inside a mile and the second bend is invisible " +
        "until the first is behind you. Pilots were licensed here before they " +
        "were licensed anywhere else in the archipelago, and the harbour dues " +
        "that paid for the licence paid, eventually, for the survey that made " +
        "it unnecessary.",
    ],
    [
      "Iskay and the Chartmakers' Sound",
      41,
      58,
      "Edda Voss's 1911 survey fixes the channel names that older charts had " +
        "left to rumour, and her notebooks become the book's recurring source.",
      "Edda Voss came to Iskay in the March of 1911 with a hired ketch, a " +
        "sounding line and no commission from anybody, and left eighteen months " +
        "later having named half the water in this book. The Sound had been " +
        "charted before — twice, badly — and the older sheets disagreed about " +
        "which of the two northern channels carried the deeper water. Voss's " +
        "answer was to take depths at every cable's length and let the ferry " +
        "masters argue with the numbers rather than with her.",
    ],
    [
      "Thornholm's Orchard Terraces",
      59,
      76,
      null,
      "Thornholm grows apples on ground that should not hold soil, and the " +
        "terraces that make it possible are older than any record of who cut " +
        "them. Forty-one steps climb the southern face of the island; the " +
        "lowest is a wall of dressed stone two courses high, and the highest " +
        "is a scrape of turf that yields in three years out of five. The " +
        "orchards are the only cargo on the archipelago that has never once " +
        "moved by any route other than the ferry.",
    ],
    [
      "Calder's Rest and the Ferry Guilds",
      77,
      92,
      "How the guilds set a timetable no single island could have imposed, and " +
        "what happened to the crossings they chose to drop.",
      "A timetable is a political document. The one pinned inside the guild " +
        "hall at Calder's Rest in 1887 is the first that treats the twelve " +
        "islands as a single circuit, and it did what no island council could " +
        "have done alone: it decided which harbours were worth an hour of a " +
        "boat's day and which were not. Two of the crossings it dropped that " +
        "winter have never been reinstated, and both of the settlements they " +
        "served were empty within a generation.",
    ],
    [
      "Brindle Key: Salt, Rope, and Credit",
      93,
      101,
      "A single-industry harbour: salt paid for rope, rope paid for boats, and " +
        "the ledger that recorded both outlived the trade itself.",
      "Brindle Key made salt, and salt made everything else on Brindle Key. " +
        "The pans on the eastern shore ran from April to September; the rope " +
        "walk behind them ran all year on the credit the pans were expected to " +
        "earn, and the boats that carried both were mortgaged against the rope. " +
        "It was a chain of promises, each one secured on the next, and it held " +
        "for ninety years — which is to say it held until a single bad season " +
        "arrived at the wrong end of it.",
    ],
    [
      "Palewater and the Redrawn Shoals",
      102,
      121,
      "The 1954 storms move the outer shoals, and the survey has to be redone " +
        "against Voss's notebooks half a century after she closed them.",
      "The storms of February 1954 moved the outer shoals off Palewater by " +
        "more than a ship's length, and the charts were wrong from the morning " +
        "they stopped. What is remarkable is not that the ground moved — it " +
        "had moved before — but that the office charged with redrawing it " +
        "began, as Voss had begun, from the ferry masters' logs rather than " +
        "from the Admiralty's own soundings. The islands are surveyed by the " +
        "people who cross them.",
    ],
    [
      "Ossmere's Winter Markets",
      122,
      140,
      null,
      "For eleven weeks a year Ossmere is the busiest harbour in the " +
        "archipelago, and for the other forty-one it is among the quietest. " +
        "The winter markets began as an accident of the timetable — the only " +
        "island with a scheduled crossing in January was the only island where " +
        "anything could be sold in January — and hardened into an institution " +
        "that now outranks the timetable that produced it. Boats are chartered " +
        "for the market that would not otherwise leave harbour at all.",
    ],
    [
      "Skerrin and the Tide Races",
      141,
      162,
      "Tidal streams strong enough to set the day's working hours, and the " +
        "crossing that only exists for ninety minutes either side of slack water.",
      "Between Skerrin and the mainland the flood runs at seven knots on a " +
        "spring tide, which is faster than the ferry, and the crossing " +
        "therefore does not exist for most of the day. It exists for ninety " +
        "minutes either side of slack water, twice, and the island's working " +
        "hours are arranged around those windows rather than around the sun. " +
        "Ask a Skerrin fisherman what time it is and the answer will be a " +
        "state of the tide.",
    ],
    [
      "Dunmoor Bank, Half-Submerged",
      163,
      181,
      "A bank that is land at low water and hazard at high, and the century of " +
        "argument over which of those it is on a chart.",
      "Dunmoor Bank is land for four hours in every twelve and a hazard for " +
        "the other eight, and a chart has to choose. The 1868 sheet drew it as " +
        "an island with a name; the 1911 survey drew it as a drying shoal with " +
        "a depth figure; the current sheet draws it as both, in two colours, " +
        "which satisfies nobody. The argument matters because a name implies " +
        "an owner, and for most of a century the bank's owner was whichever " +
        "island's court had heard the case most recently.",
    ],
    [
      "Levant Shoal and the Light Keepers",
      182,
      204,
      "Four generations of keepers, the wicks they trimmed, and the automation " +
        "that ended the posting in a single season.",
      "The light on Levant Shoal was kept by one family for four generations " +
        "and by nobody at all after the autumn of 1971. The keepers' logs " +
        "survive in full — weather at six-hour intervals, the state of the " +
        "wicks, the vessels sighted and the vessels heard but not seen — and " +
        "they are the closest thing the archipelago has to a continuous record " +
        "of its own weather. The last entry notes the arrival of the engineers " +
        "and the number of crates they brought.",
    ],
    [
      "Greyhaven: The Timetable's End",
      205,
      232,
      "The last scheduled crossing, and what the archipelago becomes once the " +
        "network that defined it stops running.",
      "The last scheduled crossing to Greyhaven left the mainland at 06:40 on " +
        "a Tuesday in November and carried nine passengers, two of whom were " +
        "photographers. There was no ceremony, because a timetable does not " +
        "end on a date; it thins, and then one of the thin services is the " +
        "last one, and the fact is only established afterwards. What the " +
        "archipelago becomes without a network is the question this chapter " +
        "cannot answer and the islands are currently answering.",
    ],
  ].map(([title, page_start, page_end, synopsis, opening], i) => ({
    idx: i + 1,
    title,
    page_start,
    page_end,
    ...(synopsis ? { synopsis } : {}),
    ...(opening ? { opening } : {}),
  })),
};

const books = new Map();
books.set("bk_meridian", {
  book_id: "bk_meridian",
  title: MERIDIAN_TOC.title,
  pages: 232,
  status: "ready",
  progress: 1,
  created_at: "2026-08-14T09:23:41Z",
  toc: MERIDIAN_TOC,
});

function genericToc(title, pages) {
  const names = [
    "Opening bearings",
    "First crossings",
    "What the charts missed",
    "The middle passage",
    "Weather and debts",
    "A working harbour",
    "The long detour",
    "Landfall, twice",
  ];
  // An uploaded book gets openings too: the reader prints one on every
  // chapter page, so a mock book without them would open on an empty page.
  const openings = [
    "Every crossing in this book begins the same way — a bearing taken from " +
      "somewhere the compass can be trusted, and a note of the hour. What " +
      "follows is an attempt to hold both of those steady while the ground " +
      "underneath them moves.",
    "The first crossings were made without charts and recorded afterwards " +
      "from memory, which is why the earliest accounts agree about the " +
      "weather and about nothing else. Four of them survive in full.",
    "What a chart leaves out is a decision, not an oversight. This chapter " +
      "is about three such decisions and the two hundred years of argument " +
      "that each of them started.",
    "Halfway is a place here, not a fraction: a stretch of open water with " +
      "no shelter on either side of it, where a boat is committed from the " +
      "moment it leaves the last headland.",
    "Weather and debt arrive together in the records, and for the same " +
      "reason — a season that keeps the boats in harbour is a season nobody " +
      "can pay for, and the ledgers read like a barometer kept in arrears.",
    "A working harbour is a place where more leaves than arrives. By that " +
      "measure there have only ever been three of them here, and two are " +
      "now anchorages with a name and a bench.",
    "The detour was longer than the route it replaced by eleven miles and " +
      "safer by a margin nobody was able to put a number on, which is the " +
      "whole of the difficulty in a sentence.",
    "The end of a passage is made twice: once when the land is sighted and " +
      "again, some hours later, when the boat is actually tied up. This " +
      "chapter is about the interval between those two landfalls.",
  ];
  const per = Math.max(2, Math.floor(pages / names.length));
  return {
    title,
    chapters: names.map((t, i) => ({
      idx: i + 1,
      title: t,
      page_start: i * per + 1,
      page_end: i === names.length - 1 ? pages : (i + 1) * per,
      opening: openings[i],
    })),
  };
}

/* ---------------- canned answer ---------------- */

const ANSWER =
  "The book treats the archipelago less as a place than as a network: twelve islands stitched together by ferry lines, weather, and shared debts. " +
  "Chapter 3 follows the cartographer Edda Voss, whose 1911 survey fixed the channel names that older charts had left to rumour (Chapter 3). " +
  "Chapter 7 returns to her notebooks when the outer shoals are redrawn after the 1954 storms (Chapter 7). " +
  "The short version: geography set the stage, but the ferry timetable wrote the plot.";

// Each source carries its passage twice, exactly as the real one does: `text`
// is the whole retrieved chunk and `preview` is its first PREVIEW_CHARS
// characters, sliced rather than written, mid-word cut and all. Without
// `text` the card's "Full passage" control opened the preview it was already
// printing, and pressing a source's reference turned the book to a
// 200-character fragment of the passage instead of the passage.
const PREVIEW_CHARS = 200;

const SOURCES = [
  {
    chunk_id: "c03-017",
    chapter_idx: 3,
    chapter_title: "Iskay and the Chartmakers' Sound",
    section_title: "The Voss survey",
    pages: [41, 44],
    text:
      "Voss worked the sound in a hired ketch through the spring of 1911, taking depths at every cable's length. Her fair copies, inked in two colours, fixed the names of channels that older charts had left to rumour. She had no commission and no assistant, and the notebooks show her working the same six miles of water eleven times between April and July, waiting each time for the state of tide that would let her read the bottom. The names she wrote down were mostly not hers: she asked the ferry masters what they called each channel and recorded the answer, which is why the sheet that resulted was accepted by the people who had to use it and why it is still, with two corrections, the sheet in use today.",
  },
  {
    chunk_id: "c07-052",
    chapter_idx: 7,
    chapter_title: "Palewater and the Redrawn Shoals",
    section_title: "After the storms",
    pages: [102, 105],
    text:
      "The 1954 storms moved the outer shoals by more than a ship's length, and the Admiralty asked for new sheets. The drafting office began, as Voss had, from the ferry masters' logs. It was not sentiment: the logs were the only continuous record of where boats had actually passed and where they had touched, and they covered the ground the Admiralty's own soundings had last visited in 1911. The new sheet took nineteen months and disagreed with Voss's in four places, three of which were the storms' doing and one of which, the drafting office was careful to note, had been an error of hers all along.",
  },
  {
    chunk_id: "c01-002",
    chapter_idx: 1,
    chapter_title: "The Shape of the Archipelago",
    section_title: null,
    pages: [9, 12],
    text:
      "Twelve islands, two tide races, and one improbable ferry timetable: the archipelago is best read as a network rather than a place, and the guide keeps returning to that claim. The islands have almost nothing in common geologically and rather a lot in common financially, and the second of those facts is the one that explains the shape of a working week here. A crossing that runs pays for a crossing that does not; a harbour that ships apples underwrites a harbour that ships nothing at all. Take the timetable away and what remains is twelve unrelated places within sight of one another.",
  },
].map((source) => ({ ...source, preview: source.text.slice(0, PREVIEW_CHARS) }));

// Mirrors the backend's citations event (SPEC §14.7):
// {"map": [{sentence_idx, span, chunk_ids}]} — span holds character offsets
// into the final answer, computed here from ANSWER itself so the client can
// highlight by slicing exactly as it does against the real server.
const SENTENCE_SPANS = (() => {
  const spans = [];
  const re = /[^.!?…]+[.!?…]+["”')\]]*\s*/g;
  let m;
  while ((m = re.exec(ANSWER)) !== null) {
    const trimmed = m[0].replace(/\s+$/, "");
    spans.push([m.index, m.index + trimmed.length]);
  }
  return spans;
})();

const CITATIONS = [
  { sentence_idx: 0, span: SENTENCE_SPANS[0], chunk_ids: ["c01-002"] },
  { sentence_idx: 1, span: SENTENCE_SPANS[1], chunk_ids: ["c03-017"] },
  { sentence_idx: 2, span: SENTENCE_SPANS[2], chunk_ids: ["c07-052"] },
  { sentence_idx: 3, span: SENTENCE_SPANS[3], chunk_ids: ["c01-002"] },
].filter((c) => Array.isArray(c.span));

/* ---------------- PCM helpers ---------------- */

/** Soft two-partial tone with fade in/out, s16le mono base64. */
function toneSegment(freq, seconds) {
  const n = Math.floor(SAMPLE_RATE * seconds);
  const buf = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE;
    const env = Math.min(1, i / 800, (n - i) / 1600);
    const v =
      0.11 * env * Math.sin(2 * Math.PI * freq * t) +
      0.05 * env * Math.sin(2 * Math.PI * freq * 1.5 * t);
    buf.writeInt16LE(Math.round(v * 32767), i * 2);
  }
  return buf.toString("base64");
}

const AUDIO_SEGMENTS = [toneSegment(220, 0.9), toneSegment(196, 1.1)];

/* ---------------- http plumbing ---------------- */

const sessions = new Map(); // session_id -> exchange count

function cors(res) {
  res.setHeader("Access-Control-Allow-Origin", "http://localhost:5173");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type,Accept");
}

function json(res, status, body) {
  cors(res);
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

function publicBook(b) {
  const { toc, timer, ...rest } = b;
  return rest;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function readBody(req, limit = 1024 * 1024) {
  return new Promise((resolve) => {
    let head = Buffer.alloc(0);
    req.on("data", (chunk) => {
      if (head.length < limit) head = Buffer.concat([head, chunk]);
    });
    req.on("end", () => resolve(head));
    req.on("error", () => resolve(head));
  });
}

/* ---------------- ingestion simulation ---------------- */

function startIngestion(book) {
  // Mirrors the real ingestor's stage order — including contextualizing and
  // summarizing, which the UI must survive (a missing stage once killed the
  // status polling mid-ingest).
  const steps = [
    ["parsing", 0.08, 500],
    ["parsing", 0.15, 600],
    ["structuring", 0.3, 900],
    ["chunking", 0.35, 900],
    ["contextualizing", 0.45, 800],
    ["contextualizing", 0.55, 800],
    ["summarizing", 0.65, 800],
    ["embedding", 0.75, 700],
    ["embedding", 0.88, 700],
    ["ready", 1, 600],
  ];
  let i = 0;
  const tick = () => {
    if (!books.has(book.book_id)) return;
    const [status, progress, delay] = steps[i];
    book.status = status;
    book.progress = progress;
    i += 1;
    if (i < steps.length) book.timer = setTimeout(tick, delay);
  };
  book.timer = setTimeout(tick, 400);
}

/* ---------------- SSE chat ---------------- */

async function handleChat(req, res) {
  const raw = await readBody(req);
  let body = {};
  try {
    body = JSON.parse(raw.toString("utf8") || "{}");
  } catch {
    /* fall through to validation */
  }
  const { book_id: bookId, session_id: sessionId, question } = body;
  const book = books.get(bookId);
  if (!book) return json(res, 404, { detail: "Unknown book." });
  if (book.status !== "ready")
    return json(res, 409, { detail: "The book is still being indexed." });
  if (!question || !question.trim())
    return json(res, 422, { detail: "The question is empty." });

  cors(res);
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  let closed = false;
  req.on("close", () => {
    closed = true;
  });

  const turnId = `t_${randomUUID().slice(0, 8)}`;
  const send = (event, data) => {
    if (closed) return;
    res.write(`event: ${event}\ndata: ${JSON.stringify({ turn_id: turnId, ...data })}\n\n`);
  };

  const history = sessions.get(sessionId) ?? 0;
  sessions.set(sessionId, history + 1);

  if (history > 0) {
    send("status", { stage: "rewriting" });
    await sleep(420);
    send("question", {
      text: `${question.replace(/[.?!\s]+$/, "")} — in the context of the Voss survey?`,
    });
  } else {
    send("question", { text: question });
  }

  send("status", { stage: "retrieving" });
  await sleep(360);
  // `confidence` is still part of the contract and still worth sending, but it
  // is no longer what the UI warns from: the banner counts how many sentences
  // the citations traced, so a high number here does not suppress it and a low
  // one does not raise it.
  send("sources", { sources: SOURCES, confidence: 0.97 });
  send("status", { stage: "generating" });
  await sleep(300);

  const words = ANSWER.match(/\S+\s*/g) ?? [ANSWER];
  let sentAudio0 = false;
  for (let i = 0; i < words.length; i++) {
    if (closed) return;
    send("delta", { text: words[i] });
    // first sentence finished → first TTS segment (sentence-pipelined)
    if (!sentAudio0 && /[.!?]\s*$/.test(words[i]) && i > 3) {
      sentAudio0 = true;
      send("status", { stage: "speaking" });
      send("audio", {
        seq: 0,
        format: "pcm",
        sample_rate: SAMPLE_RATE,
        b64: AUDIO_SEGMENTS[0],
      });
    }
    await sleep(28);
  }
  if (closed) return;

  send("audio", {
    seq: 1,
    format: "pcm",
    sample_rate: SAMPLE_RATE,
    b64: AUDIO_SEGMENTS[1],
  });
  await sleep(220);

  send("done", {
    answer: ANSWER,
    timings: {
      rewrite_ms: history > 0 ? 412 : 0,
      retrieve_ms: 187,
      first_token_ms: 934,
      first_audio_ms: 2861,
      total_ms: 6420,
    },
  });
  send("citations", { map: CITATIONS });
  if (!closed) res.end();
}

/* ---------------- router ---------------- */

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);
  const path = url.pathname;

  if (req.method === "OPTIONS") {
    cors(res);
    res.writeHead(204);
    return res.end();
  }

  if (req.method === "GET" && path === "/api/health") {
    return json(res, 200, { ok: true, openai_configured: true });
  }

  if (req.method === "GET" && path === "/api/books") {
    return json(res, 200, {
      books: [...books.values()].map(publicBook),
    });
  }

  if (req.method === "POST" && path === "/api/books") {
    const head = await readBody(req, 64 * 1024);
    const match = head.toString("latin1").match(/filename="([^"]+)"/);
    const filename = match ? match[1] : "uploaded.pdf";
    const title = filename.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ");
    const pages = 190 + Math.floor(Math.random() * 60);
    const book = {
      book_id: `bk_${randomUUID().slice(0, 8)}`,
      title,
      pages,
      status: "parsing",
      progress: 0.02,
      created_at: new Date().toISOString(),
      toc: genericToc(title, pages),
    };
    books.set(book.book_id, book);
    startIngestion(book);
    return json(res, 202, { book_id: book.book_id, status: "parsing" });
  }

  const statusMatch = path.match(/^\/api\/books\/([^/]+)\/status$/);
  if (req.method === "GET" && statusMatch) {
    const book = books.get(statusMatch[1]);
    if (!book) return json(res, 404, { detail: "Unknown book." });
    return json(res, 200, { status: book.status, progress: book.progress });
  }

  const tocMatch = path.match(/^\/api\/books\/([^/]+)\/toc$/);
  if (req.method === "GET" && tocMatch) {
    const book = books.get(tocMatch[1]);
    if (!book) return json(res, 404, { detail: "Unknown book." });
    return json(res, 200, book.toc);
  }

  const delMatch = path.match(/^\/api\/books\/([^/]+)$/);
  if (req.method === "DELETE" && delMatch) {
    const book = books.get(delMatch[1]);
    if (!book) return json(res, 404, { detail: "Unknown book." });
    if (book.timer) clearTimeout(book.timer);
    books.delete(book.book_id);
    cors(res);
    res.writeHead(204);
    return res.end();
  }

  if (req.method === "POST" && path === "/api/asr") {
    await readBody(req);
    await sleep(700);
    return json(res, 200, {
      text: "What does the book say about the 1911 survey of the channels?",
      duration_ms: 2350,
    });
  }

  if (req.method === "POST" && path === "/api/chat") {
    return handleChat(req, res);
  }

  return json(res, 404, { detail: "Not found." });
});

server.listen(PORT, () => {
  console.log(`BookTalk mock backend listening on http://localhost:${PORT}`);
  console.log("Seeded: 1 ready book. Uploads simulate the full ingestion arc.");
});
