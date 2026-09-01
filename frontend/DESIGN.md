# BookTalk design brief v3 — "Studio Light"

Third direction, per client feedback on v2: **no dark surfaces anywhere** — the dark
shelf rail is retired. The whole app now follows the client's dashboard reference image:
cool light canvas, floating white cards with large radii and very soft shadows, ONE
decisive blue theme color, soft pastel gradient tiles, and **frosted-glass (毛玻璃)
overlay surfaces**. Typography stays (Literata / Instrument Sans / Spline Sans Mono).

Note on skill conflicts: the client explicitly requested glassmorphism and a stronger
theme color. Per impeccable's own "the brief wins" rule, this overrides minimalist-ui's
glass ban — but glass is applied only to surfaces that genuinely overlay content
(header, rail, composer, toasts, sheet), never as decoration on inline cards, and the
craft-floor contrast floors remain non-negotiable on every glass surface.

## Tokens (replaces v2's shelf/spine tokens; the rest carries over)

```css
--canvas: #E9EBF2;                 /* page ground, slightly cooler than v2 */
--canvas-glow-a: radial-gradient(...#FFD9C7 at ~8% opacity);  /* two static ambient   */
--canvas-glow-b: radial-gradient(...#BFD8FF at ~10% opacity); /* washes on <body>, so */
                                   /* glass surfaces visibly frost over something      */
--card: #FFFFFF;
--ink: #171A20;  --ink-soft: #5F6672;  --line: #E2E5EB;  --line-strong: #CCD2DC;
--blue: #3D63F2;  --blue-deep: #2C4BD8;  --blue-soft: #DCE7FF;
--live: #F0604D;  --live-strong: #D93C23;  --live-ink: #B8301E;   /* recording only */
--grad-warm: linear-gradient(135deg,#FFD9C7,#F6B9D2);
--grad-cool: linear-gradient(135deg,#BFD8FF,#A3E7DB);
--glass: rgba(255,255,255,0.72);
--glass-blur: saturate(1.4) blur(18px);
--glass-line: rgba(255,255,255,0.55);   /* 1px top edge on glass surfaces */
--radius-card: 20px; --radius-inner: 12px; --radius-pill: 999px;
--shadow-card: 0 8px 28px rgba(23,26,32,0.07);
--shadow-lift: 0 14px 36px rgba(23,26,32,0.12);
--err: #B3402F; --err-bg: #FBE9E4;
```

Rules: text on glass only when the glass alpha ≥0.7 over the light canvas (measure the
effective contrast; ink ≥4.5:1 must hold with the blur OFF too — the @supports fallback
is solid `rgba(255,255,255,0.92)`). Coral stays recording-only. No gradient text.

## The theme color, used decisively

v2 under-used the blue; v3 makes it the app's voice, exactly as the reference does:
- Primary actions filled --blue with white glyphs (mic idle, send when armed).
- The SELECTED book card gets a --blue selection ring + a small blue check pill
  ("Sep"-pill energy from the reference); selected TOC row = blue text + blue-soft pill.
- Ingest progress bars, the stage meter, the equalizer/waveform in speaking state,
  links, focus rings, "Used N sources" trigger: all --blue.
- Timings/stat pills keep white ground with mono ink, blue numerals allowed.

## Surfaces

- **Canvas**: --canvas with the two ambient pastel glows fixed behind everything
  (position: fixed, pointer-events: none, no animation — operate.md still bans ambient
  motion). This is what makes frosted surfaces read as glass instead of gray.
- **Header bar** (new): slim sticky bar across the top of the studio column — book
  title, chapter/page meta pill, health/status. `background: var(--glass);
  backdrop-filter: var(--glass-blur)`; hairline bottom --line at 60%; content scrolls
  under it.
- **Left rail, light**: a frosted glass panel (--glass + blur, 1px --glass-line inner
  top edge, --shadow-card) floating 12px inside the canvas edge, radius-card. Inside:
  - **Book cards replace spines**: each book = white inner card, radius-inner, with a
    small COVER TILE on the left (44×58px, radius 8px): a deterministic pastel gradient
    hashed from the title — hue family alternates warm/cool — with the book's initial
    letter in Literata white/ink (pick per contrast). Title (Literata, 2-line clamp),
    meta line in mono ("12 ch · 226 pp"). Selected card: blue ring + check pill.
    Ingesting: cover tile at 60% opacity, thin blue progress underline, stage label in
    mono. Failed: err tint label.
  - "Add a book" = dashed light slot with a blue plus.
  - **TOC**: light rows, mono zero-padded numerals in --ink-soft, blue-soft pill on the
    active chapter, synopsis disclosure rows keep working (restyle to light).
- **Studio column**: unchanged structure (62rem column), cards white radius-card.
- **Composer**: frosted glass bar (--glass + blur), sticky at the bottom, floating 12px
  from edges with radius-card and --shadow-lift — the conversation scrolls under it.
  Input inside stays a solid white field (text never sits on raw glass).
- **Stat tiles** (ingest): unchanged v2 behavior, gradients from the two families.
- **Empty-state hero**: white card, headline, drop region washed --grad-cool at 12%;
  add a slim --grad-warm accent bar or corner wash so the hero isn't colorless.
- **Citations/sidenotes, sources trace, low-confidence banner**: keep structure and
  tokens (white mini-cards, blue mono indices, --blue-soft highlight, err banner).
- **Toasts + mobile sheet**: frosted glass (--glass + blur), radius-card.

## Motion & states

All v2 rules stand: 120/180/240ms state-only motion, no load choreography,
prefers-reduced-motion honored, every control keeps default/hover/focus/active/
disabled/loading/error states. Card hover lifts with --shadow-lift. Selection ring
transitions 180ms. Nothing else animates.

## Verification bar (unchanged and binding)

craft-floor Verify list; redesign-existing-projects audit against the finished work;
contrast measured for every shipped pairing INCLUDING text-on-glass with the fallback
background; `npm run build` zero TS errors; `npx vitest run` green; browser surfaces
(::selection, caret, scrollbars, focus ring) re-themed to the light palette only —
no dark scrollbar remnants from the v2 shelf.
