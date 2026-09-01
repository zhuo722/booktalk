# AI workflow

How BookTalk was actually built. This is a process record, not a methodology pitch — where
the process failed, that is written down too, because a workflow you only report the wins
of is not evidence of anything.

The primary sources are the artifacts, and they are all in this repository:
[SPEC.md](../SPEC.md) written before the code, [RESEARCH.md](RESEARCH.md) with the raw
agent output behind it, the test suite, and the eval's per-query CSVs. The history is
delivered as a single commit, so this file carries the order of the work rather than
pointing at it.

The build proper ran in ten steps:

```
 1  the technical specification, written before any application code
 2  three open-source design skills read in full, then adopted as binding
 3  a five-agent research pass, folded into the spec as §14
 4  backend v1, built against the spec
 5  frontend v1, built against the same spec, in parallel and without coordination
 6  the frontend rebuilt on a new visual direction
 7  backend v2: structure ladder, FlashRank, citations, PCM voice, the eval harness
 8  migrate-on-start ordering — a bug only a live run could find
 9  an explicit base_url for the OpenAI SDK — the second such bug
10  the real six-arm ablation, with its fixture, golden set and results committed
```

Five smaller rounds followed the same pattern — chapter synopses, the citations wire fix,
the documentation, a three-lens adversarial review with its fixes, a test-count correction —
and are described in §5. After those came a different kind of work altogether, with its own
sections: the interactive phase (§7) and the final audit (§8).

Read top to bottom, that order says the thing worth saying: **specification, then
research, then implementation, then two bugs that only a live run could find, then
measurement.** Nothing was built before it was specified, and nothing was claimed before it
was measured.

---

## 1. Spec-first, and the spec was binding

[SPEC.md](../SPEC.md) came first, before a single line of application code, and it is a
contract rather than a sketch: the file tree, the module
responsibilities, the SQLite DDL, the `AIClient` Protocol signature, the SSE event
sequence, the token budgets, and a per-phase definition of done. Its own opening line says
implementation agents must follow it exactly and prefer the simplest solution where it is
silent.

The load-bearing decision was **fixing the API contract (§6) and the SSE event shape (§5.2)
up front**. That is what made the backend and the frontend buildable in parallel with zero
coordination between them: the frontend agent never needed to ask the backend agent
anything, because the questions were already answered in writing. `frontend/mock/server.mjs`
exists for the same reason — it implements §6 with canned data, so the UI was fully
drivable before the backend it talks to existed.

The cost of that approach is visible too. The v2 research pass (SPEC §14) overrode enough
of the original design that the real module tree no longer matches the tree drawn in
SPEC §3: `ingest/` gained `toc.py`, `headings.py`, `markdown.py`, `vocab.py`, `enrich.py`,
`prompts.py`; `speech/` gained `pcm.py`, `turns.py`, `phonetics.py`; `rag/` gained
`citations.py`. TTS went from mp3 to raw PCM, and citations went from inline markers to
post-hoc attribution.
Rather than rewrite that section into a lie about what was planned when, SPEC §14 states
which section each revision overrides and the tree is left as the historical record.

The gap widened again in the interactive phase (§7 below): the frontend SPEC §7 drew —
`Shell`, `Library`, `Toc`, `BookCard`, a 300px left rail — no longer exists, and `Shelf`,
`Reader` and the flight between them are nowhere in the spec because nobody had thought of
them when it was written. The same treatment applies: SPEC §3 and SPEC §7 now open with a
note saying which parts the shipped code left behind and where the current shape is written
down ([ARCHITECTURE.md](ARCHITECTURE.md), [DESIGN.md](../frontend/DESIGN.md),
[DESIGN-SHELF.md](../frontend/DESIGN-SHELF.md)). A specification that is quietly edited
to match the code stops being evidence of anything.

## 2. A role split between planning and execution

Two tiers of model, doing different jobs:

- A **planner model** (Claude Fable 5) wrote SPEC.md, the design briefs, and the agent
  contracts, and reviewed at every phase boundary rather than continuously. It also
  carried out the frontend rebuild (step 6) itself, because visual judgement was the
  bottleneck there rather than throughput.
- **Executor agents** (Claude Opus 5) implemented against those contracts —
  backend v1, frontend v1, the v2 upgrade, the integration suite, the eval harness.

The division is deliberate: writing the contract and satisfying it are different skills,
and an agent that authors its own acceptance criteria will always meet them. Review at
phase boundaries rather than per-file kept the reviewer looking at whether the *phase*
achieved its purpose, which is a different question from whether each diff is locally
reasonable.

## 3. External input, reviewed rather than trusted

Two channels, both leaving artifacts in the repository.

**Design skills adopted from GitHub.** Three open-source skill packages were read in full,
then adopted: `minimalist-ui` and `redesign-existing-projects` from
[Leonxlnx/taste-skill](https://github.com/Leonxlnx/taste-skill) (MIT), and
`impeccable-craft` from [pbakaus/impeccable](https://github.com/pbakaus/impeccable)
(Apache-2.0). Provenance, licenses and exact changes are recorded in
[`.claude/skills/README.md`](../.claude/skills/README.md), and only instruction files were
installed — no scripts, no hooks, no executable content from repositories we do not
control. SPEC §7 was then amended to make those skills the *binding authority* on
frontend quality, overriding the spec's own token choices where they conflicted (its
original body font, Inter, was struck out by that rule). Adopting external taste and then
letting it overrule your own earlier decision is the part that makes the adoption real
rather than decorative.

**A five-agent research pass.** Before finalizing the design, five agents ran in parallel —
RAG products, PDF parsing, voice pipelines, retrieval and eval, frontend — with
instructions to **read actual source and verify claims** rather than answer from memory.
Roughly 500k tokens, 309 tool calls, ~16 minutes wall clock; full output preserved at
[`docs/research/raw-recommendations.txt`](research/raw-recommendations.txt) and the
synthesis at [`docs/RESEARCH.md`](RESEARCH.md).

What makes it more than a literature review is that recommendations were *benchmarked on
this machine* before adoption: FlashRank's model measured at 5 MB and ~7.6 ms per 20
passages with a resolving cp39 wheel; pymupdf4llm compared against raw `get_text()` on a
real 244-page book (129 ligature characters and 68 hyphen line-breaks → 0 and 0); Docling's
`HybridChunker` read as source, and `semchunk` verified installable on Python 3.9. Live
GitHub data was checked rather than assumed — Verba turned out to be archived, chatbot-ui
and vocode roughly two years stale — and **popularity was explicitly not treated as
fitness**.

The surviving recommendations became SPEC §14, which is why the research step
touches the spec rather than the code. And the rejections were written down with reasons —
LangChain, ragas/deepeval, HyDE, late chunking, full RAPTOR, semantic chunking as the
primary path, Tailwind/shadcn, browser VAD, wavesurfer.js — because a design is also the
set of things it decided not to do, and an undocumented rejection is indistinguishable from
an oversight.

## 4. Structured accountability, with a mandatory deviations field

Every agent returned a machine-checked structured report, and the schema included a
**required `deviations` array**. Not optional, not free-text at the bottom: a field the
agent had to fill or fail validation.

This exists because the default failure mode of a capable agent is a confident summary that
quietly omits the three places it did something other than what was asked. Making deviation
a first-class output changes the incentive — reporting a departure is compliance, hiding
one is a schema violation.

Reported deviations were then **reviewed, not rubber-stamped**. Some were accepted and
folded into the spec (the v2 module tree above is largely accepted deviation). Some were
rejected and sent back. The distinction only exists because the field forced the
conversation to happen at all.

## 5. Verification culture — and the bugs that got through anyway

The rule was: each agent verifies its own work, and the orchestrator re-verifies
independently. Agents ran their own tests; the orchestrator ran them again, read the diffs,
and checked claims against artifacts rather than against summaries.

**It was not sufficient.** Both of the following passed the 363 green tests the suite held
at the time (375 today) and an independent re-verification, and were then found within
minutes of running the application end to end with a real API key:

- **Migrate-on-start ordering.** `init_db` created tables, indexes and column
  migrations in a single ordered pass, so `CREATE INDEX idx_chunks_kind` ran *before*
  `ALTER TABLE chunks ADD COLUMN kind`. Invisible to every test, because every test starts
  from a fresh temporary database where the column is in the `CREATE TABLE`. It only bites a
  database created **before** the v2 schema — i.e. exactly the developer machine that had
  been running v1 all week. Fixed by splitting the pass into tables → migrations → indexes,
  and pinned by `tests/unit/test_db_migration.py`, which builds a v1-shaped `chunks` table
  and asserts both the columns and the index exist afterwards.
- **An empty `OPENAI_BASE_URL` poisoning the SDK.** `.env.example` ships
  `OPENAI_BASE_URL=` as a documented empty override. dotenv loads that as `""`, the app
  passed `base_url=None`, and the OpenAI SDK then fell back to reading the environment
  variable itself — where it found the empty string and built broken request URLs. What
  made it *hard* rather than obvious is that the code is full of well-intentioned
  per-chunk fallbacks: a failed contextual-enrichment call degrades to the static
  breadcrumb, a failed TTS call degrades one sentence to text-only, a failed reranker
  degrades to fusion order. Every one of those did its job, so the app kept working while
  quietly doing less. Fixed by normalizing empty to `None` in `config.py` and **always**
  passing an explicit `base_url` to the SDK.

Two honest notes on that second one. First, on what guards it — and this paragraph is worth
reading as an example of its own subject. Its first draft said the fix was guarded by a code
invariant and a comment at the call site, **not by a regression test of its own**, because
the failure only reproduces against a live SDK. That was already wrong when it was written:
the two tests now in `tests/unit/test_client_params.py` — a blank `OPENAI_BASE_URL=` in the
environment never reaching the SDK, and a configured one being used — had landed one round
earlier. The honest version is narrower than either claim: those tests assert
on the base URL of the constructed client, so they pin the boundary the bug crossed, not the
broken requests it produced. The claim was written from the state of the code as it was
remembered rather than as the tree had it, and nothing checked it — the same failure mode
§4 exists to catch in implementation agents, arriving in the documentation instead.
Second, the general lesson is the uncomfortable one: *graceful degradation hides bugs from
tests*. Every fallback in this codebase is defensible in isolation and collectively they
made a total misconfiguration look like a slightly worse day.

**A third of the same kind, found while preparing the demo — and the clearest example in
this file.** Holding the mic produced nothing: no answer, no error, the button spinning
forever. The composer keeps an `alive` ref so a transcription that resolves after the reader
has been left does not ask its question on a screen nobody is looking at. The cleanup set it
to `false`; nothing set it back to `true`. React's StrictMode mounts, unmounts and remounts
every component in development, so the flag was false from the first render on: `/api/asr`
returned 200, the early return dropped the transcript, and the `finally` clause — gated on
the same flag — never took the mic out of its busy state. Every layer reported success. What
named it was the server log: `POST /api/asr 200 OK` with no `POST /api/chat` behind it, over
and over. **When every step's own logs say "fine", look at what the next step never
received.** The same run also showed that a stopped backend left the UI loading forever
rather than erroring, because no fetch carried a deadline; both fixes have tests, and the
manual checklist in [TESTING.md](TESTING.md) now says to watch the question reach the thread
rather than only to watch the transcription start.

Three further bugs were caught earlier and never reached the shipped code at all, by the
integration-test agent — which was briefed to **attack** the pipeline rather than confirm
it. Adversarial framing in the agent's contract found defects that the same agent, asked to
"write integration tests", would have written passing tests around.

**A third gap, found while writing this documentation.** The backend emits the citation map
as `{"map": [{"sentence_idx", "chunk_ids"}]}` per SPEC §14.7; the client reads
`payload.citations` with a `{sentence, chunk_id}` shape, which is what `mock/server.mjs`
sends. So per-sentence sidenotes render perfectly against the mock and silently render
nothing against the real backend — and because the client's fallback is un-annotated prose,
nothing looks broken. It survived 363 tests for the same structural reason as the two bugs
above: **the mock is the frontend's oracle, and it was written from the same reading of the
spec as the client it validates.** A contract needs one implementation and one
*independently derived* consumer, not two readings by the same author. The documentation
pass reported it rather than fixing it (its scope was docs); the orchestrator then fixed
the client to read the server's `{"map": [{sentence_idx, chunk_ids}]}` shape and corrected
the mock to mirror the real wire contract — the mock now copies the server, it no longer
copies the client.

**And it drifted again, in exactly the way the paragraph above predicts.** The interactive
phase (§7) taught the backend two new things — a TOC chapter carries its `opening` prose,
and a source carries its passage in full as `text` rather than only a 200-character
`preview` — because the reader's page and its "Full passage" control needed them. The mock
was not updated for either, so both features were dead against the offline demo path while
looking entirely healthy against the real server: the chapter page fell back to the
generated synopsis, and "Full passage" opened the preview it was already printing. Nothing
failed, because nothing checks that the mock still implements the contract it claims to.
That is a *second* instance of the same structural defect, and the fix in the previous
paragraph — make the mock copy the server — was never going to prevent it. Only the shared
fixture in "what I would do differently" would.

## 6. Prompting efficiency

The pattern was **one detailed contract per agent**, not iterative correction. Each
implementation agent received a brief that already contained the interfaces it had to
satisfy, the constants it had to use, the definition of done, and the report schema it had
to return — so the expensive back-and-forth happened once, in writing, before any tokens
were spent generating code. Specification is cheaper than correction, and it leaves an
artifact.

Mid-flight scope updates were sent **by message to a running agent exactly once** across
the contract-driven phases — everything up to and including the documentation pass. That
number is a deliberate metric: every additional mid-flight message is evidence that the
contract was underspecified, and the discipline of writing the contract well enough to
avoid them is the same discipline that produced SPEC.md. It is also a metric that only
means something where there is a contract to be measured against; §7 is the phase where
there was not, and it does not report one.

## 7. The interactive phase: seventeen rounds with nothing to check against

Four hours of work in seventeen rounds, and almost none of the apparatus in §1–§6 applied.
There was no contract to satisfy, because the acceptance criterion was what a person sees on
a screen, and that cannot be written down in advance with enough precision to be checked.
The loop was: run it, watch it, measure what looked wrong, fix that one thing. One round per
fix.

**What was built.** The dark rail of the earlier design was retired for one light world with
blue as its only voice and coral kept for recording, across 44 verified contrast pairings. A
Shelf went in front of the conversation — one book per full-bleed row, each a CSS-3D volume
lying flat, a fixed tick rail with scroll-spy. The reader was then made to *be* the book
rather than a pane beside it, which deleted `Shell`, `Library`, `Toc` and `BookCard` and 65
CSS blocks orphaned by their removal. The very next round threw half of that away: the fan
of one leaf per chapter became a single page lifted out of the book with the chapters as a
scrolling strip of numbers — worth recording because it is the kind of thing only building
the thing tells you, that *a fan stops being clickable after the first few leaves and a
strip works at ninety.* The rest of the phase was small corrections of exactly the sort this
kind of work generates: spacing, alignment, timing, a colour walk, a misplaced button.

**What the backend changed, the UI asked for.** Two API additions in the whole phase, both
because a screen needed something the frozen contract did not carry: the TOC learned to send
each chapter's opening prose, because the reader's page should print the book's own words
rather than a generated blurb; and sources learned to carry their passage in full rather
than a 200-character preview, because the "Full passage" control had nothing to open. That
is an argument for building the screen *before* freezing the contract, not after. (A third
backend change came from using the app rather than from the UI: a hinted ASR decoder that
hears nothing sometimes hands back its own vocabulary prompt, and the app was answering a
question nobody had asked. The fix checks the transcript against the prompt's opening words
and retries without the hint — and a token-overlap test was tried first and rejected real
questions, because a short book's vocabulary contains most of the ordinary words in it.)

**Eleven rounds on one movement.** A book leaving the shelf and arriving in the reader,
settling at a 620ms flight with a 150ms hand-off. Eleven is an embarrassing number until you
read what each round fixed, at which point it becomes a list of things that were *measurably*
wrong rather than matters of taste: a flight that never moved, because StrictMode's second
mount measured the destination while the element was still parked at its starting keyframe
and the distance came out near zero (cancel first, measure second — the flight then read dx
332px, scale 1.38); a shape that snapped at the hand-off, because a transform cannot turn a
wide bar into an upright cover, so the flight animates the cover's own box instead; an offset
measured against the wrong box, since the element grows from its own top-left; a return
flight that landed squashed, because the resting rect of a volume tilted 46° is 103px tall
for a book that is 148px tall; and a shelf exit cut off mid-move, the other books given 520ms
to slide away on a screen that was replaced after 110ms.

None of this was reachable by testing. The offline suite stayed green throughout and has no
opinion about any of it. What replaced testing was *measurement* — reading back computed
values instead of trusting the eye. Alignment was fixed by checking that the answer cards and
the mic both start at x=697, and that the chapter strip and the question box both end at
bottom 838 with height 52; the two screens' wordmarks were reconciled by finding that an
`h1`'s line-height was 22.4px where the reader's `span` inherited 24.6px. "It looks fine" was
never accepted as a result, and that is the only discipline from the earlier phases that
survived intact into this one.

The phase also produced the correction that matters most to a reader of this app: **the
low-relevance banner was firing on good answers.** It read the retrieval confidence score,
and on a real book that score is wrong in both directions — 0.0017 for "what happens in
chapter 2", answered correctly from four cited passages, against 0.9953 for a subject the
book never mentions. The warning now counts how many of the answer's own sentences the
post-hoc attribution could trace back to a passage, and shows only when a substantial
answer (>80 characters) traced none; a refusal is short and cites nothing by design, so the
length test keeps it out. The score is still computed and still travels on the `sources`
event — it just no longer decides anything the reader sees. It is a small change with an
uncomfortable implication: the number had been shipped as a quality signal, and it had
never been checked against a case where its answer was known.

## 8. The final audit: 61 agents, four lenses, and a verifier briefed to refute

The last pass before freezing was an audit rather than a review: **61 agents across four
lenses**, each finding then handed to an independent verifier briefed to *refute* it. **46
findings were confirmed and 11 refuted** — roughly one in five did not survive the second
look, which is the whole reason for the second look. An agent asked to find problems will
find the number of problems it expects to find; the refutation step is what keeps that
appetite from becoming the record. The confirmed code findings were fixed in one pass; the
confirmed documentation findings are what this file, TESTING.md, the SPEC notes and the mock
have been corrected against.

What an audit catches that neither tests nor a person clicking around does:

- **A stylesheet selector that never matched.** Five rule sets styled `.shelf__help` while
  the button renders `.shelf-help`. The help control had therefore sat out both screen
  transitions since it was written, and its return animation had never run once — invisible
  to a build, invisible to a test, and invisible to the eye because absence of motion looks
  like a design decision.
- **A citation left in state across books.** The passage the reader had been turned to
  survived a book switch, so opening the next book turned it to an unrelated chapter of a
  different work — chapter numbers mean nothing across books.
- **A transcription resolving after its screen was gone.** A recording that came back after
  the reader had been left asked its question anyway, starting a spoken answer over the
  shelf.
- **A palette that reached half its colours.** The cloth walk advanced the hue pair but not
  the tone, so only 16 of the 32 composed cloths were ever used and the seventeenth book
  wore the first one's cover. Nobody had ever put seventeen books on the shelf.
- **A comment with two numbers exactly reversed.** The confidence figures in a code comment
  were stated backwards — the kind of defect that no test can see, that costs nothing at
  run time, and that every future reader believes.

Plus the subtraction an audit is actually for: ~130 lines of CSS for a rail, a sheet and a
header that no longer exist, a duplicate progress rule whose stale copy won, two orphaned
design tokens, four dead exports, a playback level loop running every frame for a callback
nobody set, and an unbalanced brace that every build had been reporting into a log nobody
read (a browser recovers from it by silently swallowing the rule after it).

The pattern in that list is worth naming: almost everything in it was written during the
phase that had no contract to check against. The audit is what stood in for the contract,
after the fact and at a cost of 61 agents — which is a defensible way to end a project and
a poor way to run one.

---

## What I would do differently

- **Test the contract, not the implementations of it.** One shared fixture of recorded SSE
  frames *and REST payloads*, asserted against by both the Python server and the TypeScript
  client, would have caught the citations mismatch — and it would have caught the next one,
  which duly arrived: the mock went on serving TOC chapters and sources that the real
  backend had since grown two fields past (§5). Two hand-written implementations of one
  spec will always drift, and telling one of them to copy the other only fixes the drift
  that has already happened.
- **Run the app end to end earlier, and on a dirty database.** Both shipped bugs needed a
  live key and pre-existing state. A green unit suite says the parts work; it says nothing
  about the machine the software actually runs on. That should have been a phase gate, not
  a final step.
- **Budget an audit of the fallbacks.** Every `except: log and degrade` in this codebase
  should have to answer "what does the operator see when this fires?" At least one of them
  currently answers "nothing", which is how a broken base URL became a slow day instead of
  an error.
- **Pick a bigger eval fixture, or two.** 25 pages saturates recall@5 in every arm
  including the naive baseline, which means the harness cannot distinguish the top three
  configurations at all. The measurement was honest; the instrument was too small, and that
  is a planning failure rather than a result.
- **Put the interactive work before the contract, not after it.** §7 is seventeen rounds
  of a phase the process in §1–§6 does not describe, and its two API changes both came from
  a screen discovering that the frozen contract could not answer it. A spec written after a
  week of prototype screens would have specified different things; the parallel-build
  advantage that a fixed contract buys is real, but it was bought partly with fields the
  API had to grow later.
- **Say up front how a movement will be judged.** The eleven flight rounds are not eleven
  changes of mind — every one fixed a number that was demonstrably wrong — but nothing in
  the process said in advance what "landed correctly" meant, so each fault had to be
  noticed before it could be measured. A short written acceptance list for a transition
  (starts where the click was, changes shape rather than scaling, page held until landing,
  degrades to an instant cut) costs one paragraph and would have collapsed several of
  those rounds into one.
