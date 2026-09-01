# BookTalk design brief — the Shelf page (landing / book chooser)

A new **first screen** that sits in front of the existing conversation screen: the user
lands on a shelf of their books, picks one, and only then enters the conversation. Its
skeleton copies press.stripe.com; its palette is the light "Studio Light" system from
DESIGN.md v3. Nothing about the conversation screen's behavior changes.

## The reference skeleton (measured from press.stripe.com, 1366px viewport)

Read this as the structure to reproduce, not as pixels to match:

| Element | What it is there |
|---|---|
| `main > container` | one column, full-bleed |
| `header` | top-left only: `h1` wordmark (serif, ~17.6px) над `h2` tagline (serif italic, ~16px); side padding ~66px; then **~229px of empty space** before the first book |
| `.PressHomepageProductList` | a vertical stack, one book per row, ~166px row height, no gaps — the books read as a physical pile seen at a slight angle |
| `button.PressHomepageBook` | the whole row is the hit target: `display:flex; align-items:center; padding: 0 96px`; the row's background is that book's own cloth colour; inside, `.PressHomepageBook__text` carries `h2` title + `h3` author |
| `canvas.PressHomepageCanvas` | a WebGL layer drawing the 3D volumes over those rows (we replace this with CSS 3D — see below) |
| `.PressHomepageMenu__bookList` | a fixed rail at the far left, ~34px wide: one small tick `button` per book, acting as index + scroll position |
| bottom-left | a single small `?` affordance |

The essential rhythm: **enormous quiet, then a slow vertical procession of one book at a
time, each one a physical object rather than a card.** Copy that. Copy the left tick
rail. Copy the whisper-quiet header. Do not copy the dark ground.

## Palette (light — from DESIGN.md v3, unchanged)

Ground `--canvas #E9EBF2` with the two fixed ambient pastel glows behind everything;
`--ink #171A20`, `--ink-soft #5F6672`, `--blue #3D63F2`, `--blue-soft #DCE7FF`,
`--line #E2E5EB`, glass tokens as defined. Book "cloth" colours are the light pastel
families already used for cover tiles (warm `#FFD9C7→#F6B9D2`, cool `#BFD8FF→#A3E7DB`),
plus two more light neutrals (sand `#F0E6D2`, mist `#DDE4EC`) so a shelf of six books
does not repeat a colour: hue chosen deterministically by hashing the book title, exactly
as the cover tiles already do.

## The book object (CSS 3D, no WebGL)

Each book is a volume lying flat, seen slightly from above — the Stripe Press pose:

- Row: full-bleed, `min-height: 168px`, centred content, no gap between rows.
- The volume: a `~560px × 108px` block, `perspective: 1400px` on the row and
  `transform: rotateX(46deg)` on the block, so we see the **top face** (the cover) and a
  thin **front edge** (the pages). Build it from two stacked elements plus a shadow:
  - top face: the book's pastel gradient + a 1px inner highlight along its upper edge;
  - front edge: 10–12px tall, a lighter tint of the same hue with 3 hairline "page"
    lines, so the object reads as paper not plastic;
  - ground shadow: `0 18px 30px rgba(23,26,32,.14)`, tightening on hover.
- Text sits ON the top face, laid out like a real spine: **author left (mono, uppercase,
  0.09em tracking, `--ink-soft`), title centre (Literata, `--ink`), a small round
  BookTalk mark right**. Long titles clamp to one line with an ellipsis.
- Ink on every cloth must clear 4.5:1; if a hashed gradient cannot, lighten the gradient
  rather than darkening the ink (measure both gradient stops).

## Interaction

- **Hover / focus**: the volume lifts (`translateY(-6px)`, shadow grows, 180ms) and the
  matching tick in the left rail goes `--blue`. Nothing else moves — no tilt, no scale
  bounce.
- **Click / Enter**: that book becomes the active book and the app switches to the
  conversation screen. A ready book goes straight in; a still-ingesting book goes in too
  (the conversation screen already renders its own ingest panel); a failed book opens the
  conversation screen's failed state.
- **Left tick rail**: fixed, vertically centred, ~34px wide, one 14×2px tick per book in
  `--line-strong`; the tick under the pointer/scroll position is `--blue` and 20px wide.
  Clicking a tick scrolls its book into view (`scrollIntoView({behavior:'smooth'})`,
  suppressed under `prefers-reduced-motion`). Every tick is a real `button` with an
  `aria-label` naming its book; the rail is `aria-hidden` only if a book count of 1.
- **Scroll**: plain document scroll. No parallax, no scroll-jacking, no reveal
  animations — operate.md still bans load choreography.

## Page furniture

- **Header** (top-left, no bar, no glass): `BookTalk` wordmark in Literata, and under it
  the italic tagline `Ask your books out loud`. Side padding 64px, top 24px.
- **Empty shelf** (no books yet): the existing Hero/upload content, centred in the same
  quiet — headline, one line of explanation, and the drop region. The shelf and the
  upload affordance are the same screen; when books exist, the upload slot becomes the
  **last row of the stack**: a dashed light volume in the same pose reading
  "Add a book · drop a PDF or click", with the same hover lift.
- **Ingesting books** appear in the stack immediately: cloth at 60% opacity, a `--blue`
  progress underline along the front edge, and the stage label replacing the author slot
  in mono ("SITUATING 47%"). Polling and stage vocabulary already work — reuse
  `useLibrary`.
- **Remove**: a small ghost icon-button revealed on hover at the row's right margin,
  outside the volume, with the existing confirm behavior.
- **Bottom-left `?`**: a small glass round button opening a plain popover: what the app
  does, and the note that answers come only from the selected book.

## Routing between the two screens

No router dependency. `App.tsx` holds `view: "shelf" | "chat"`:
- initial view is `"shelf"`;
- choosing a book sets the active book and `view = "chat"`;
- the conversation screen gains a **back affordance** in its header — the BookTalk
  wordmark becomes a button ("← All books") returning to `"shelf"` — and going back
  must call the existing `stopAll()` so a spoken answer does not keep playing;
- selecting a different book already stops playback (App's existing effect); keep that.
- The chat screen's left rail keeps working exactly as it is — the shelf is an
  additional entry point, not a replacement for it.

## Verification bar (binding)

craft-floor Verify list; the redesign-existing-projects audit run against the finished
work; contrast measured for ink on every generated cloth colour (both gradient stops)
and for text over glass with the no-blur fallback; `npm run build` zero TS errors;
`npx vitest run` green; keyboard path complete (Tab reaches every book and tick, Enter
opens, focus ring visible on the volume itself); 1366 / 1024 / 900 / 375px widths — on
narrow screens the rail hides, the volume narrows to the viewport minus 32px, and the
rotateX flattens toward 30deg so the type stays legible.
