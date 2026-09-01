---
name: impeccable-craft
description: Frontend quality floor and polish process for BookTalk UI work. Use when building, refining, auditing, or polishing any frontend surface — components, layouts, states, typography, color, motion. Contains the craft floor (verify checks + banned defaults), Operate/Read-mode product-UI rules, and the polish pass procedure.
---

# Impeccable craft rules (curated for BookTalk)

Three playbooks curated from the Impeccable design skill (see `.claude/skills/README.md`
for source and license). References to Impeccable's own scripts, hooks, DESIGN.md/PRODUCT.md
workflow, and `context.mjs`/`critique-storage.mjs` do NOT apply in this project — skip them
and apply the design rules directly.

Read in this order, for these purposes:

1. [reference/craft-floor.md](reference/craft-floor.md) — always, immediately before editing
   UI code. The "Verify" list is a set of checks on the built result; the "Refuse" list names
   default patterns that require justification from the brief.
2. [reference/operate.md](reference/operate.md) — BookTalk is exactly this surface type:
   an Operate (task tool: upload, ask, listen) + Read (book content, sources) product UI.
   Its rules override brand-surface instincts: restrained color, quick 150–250ms transitions,
   no decorative motion, every control with all its states, consistent component vocabulary.
3. [reference/polish.md](reference/polish.md) — for the final polish pass: triage order
   (broken tasks → missing states → drift → visual inconsistency → cleanup), whole-path
   verification with mouse/keyboard/touch, and the finishing source-diff cleanup.

Project design direction (binding, from the BookTalk brief): warm paper/editorial aesthetic,
serif display type for book titles and headings, restrained single accent, light theme only.
Where a generic rule conflicts with this committed direction, the direction wins.
