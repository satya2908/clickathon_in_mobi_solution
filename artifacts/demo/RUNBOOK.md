# Demo runbook — 2 to 3 minutes

Team Low Cardinality · InMobi track.

The track's suggested demo is: *a metric drops → the system runs → the drill-down lights up → a
plain-English diagnosis → optionally a follow-up question in chat.* This follows that order.

The one thing worth protecting on camera: the diagnosis must visibly come **out of the pipeline**,
not out of a person. Judges are explicit that a hand-written diagnosis without a matching trace
scores nothing. So the trace goes on screen, and the run id and case id stay visible.

---

## Before you hit record

State to be in:

- ClickHouse holds **history only** — Jun 1 to Jul 5, with the *original* dimension tables. The
  release has not been seen.
- `cases`, `runs`, `case_steps` are empty except for one investigation of **Jul 5**, a normal day.
  That is what the console opens on, so the first thing on camera is the system finding nothing
  wrong. It makes the contrast do the work.
- The console is up with the ingest control switched on.

```bash
# console, host-side
cd web
INGEST_ENABLED=true \
VERDICT_BIN=../.venv/bin/verdict \
VERDICT_INGEST_ROOT=/Users/sohham.seal/Desktop/clickathon \
npx next dev -p 3100
```

Open `http://localhost:3100`. Have the release path on your clipboard:

```
/Users/sohham.seal/Desktop/clickathon/unseen_data
```

Close every other tab. The ingest takes a few minutes of wall clock — plan to cut or speed that
section, and do not narrate over dead air.

---

## The script

**0:00 — What it is.** One sentence over the console showing Jul 5, quiet.

> "This is Verdict. It watches ten ad-tech metrics across a dimension lattice in ClickHouse, and
> when one of them moves it works out which segment is responsible. Right now it is looking at
> July 5th and it is telling me nothing is wrong, which is the correct answer for that day."

**0:15 — Point it at the unseen release.** Click **Ingest release**, paste the path, hit Run.

> "The unseen bundle just landed. I have not looked at it. I am going to point the system at the
> folder and let it run — it appends the events, notices the release reissued its dimension
> tables and rebuilds against them, then investigates every window the batch covers."

*(cut / speed up the wait)*

**0:35 — Reload. The drill-down lights up.**

> "Five days ingested, five windows investigated. July 8th is the one to look at."

Open the July 8 case. Land on `fill_rate`, `os_version=iOS 17.5`.

**0:50 — The diagnosis.** Read the narrative panel.

> "Fill rate for iOS 17.5 at 0.478 against an expected 0.791 — down 39.6%. And it says why it
> believes that: removing iOS 17.5 returns the parent to expectation, 99% of the movement is
> accounted for, and the fault is spread across the whole segment rather than hiding in one
> corner of it."

**1:15 — The part that is actually hard.** This is the bit that separates the submission; give it
time.

> "The expected value there is *not* historical, and this is the interesting part. The release
> regenerated its dimension attributes against the same IDs — app_00000 was ecommerce tier 3
> before, it is news tier 2 now. So 'publisher_tier=tier_3' before the boundary and after it are
> two different groups of apps, and comparing them is meaningless.
>
> The system checked. 42% of the grid stopped agreeing with its own history the day the release
> landed, against 0.2% the day before. So it refused the historical baseline and compared iOS
> 17.5 against its sibling OS versions in the same window instead — no history needed.
>
> Earlier versions of this blamed publisher_tier=tier_3 at −24%, confidently and wrongly. That
> is the failure mode the whole check exists to prevent."

**1:45 — The trace.** Switch to the trace tab / waterfall.

> "Every step is recorded — what it did, why it did it, what came back. Same trace id goes to
> OpenTelemetry and into ClickStack, and every case links to it."

Scroll the tree. Land on the audit step and let the "why" be readable.

**2:10 — ClickHouse is doing the work.**

> "The drill-down is ClickHouse queries, not the model. Fourteen thousand cells tested in about
> two and a half seconds against a lattice of rollups. The LLM writes the sentence at the end and
> every number in it is verified against the computed figures before it is shown — if it invents
> one, the sentence is thrown away and the template is used."

**2:25 — Follow-up in chat.** Open Verdict.AI, ask something like:

> "Which countries did the iOS 17.5 drop hit hardest?"

Let it run the MCP query and answer.

**2:50 — Close.**

> "Alert to answer, with the reasoning attached."

---

## Things not to do on camera

- Do not open the terminal to run the investigation. The point is that the console drives it.
- Do not skip the trace. It is the mandatory artifact.
- Do not claim the numbers are historical baselines on July 8. They are sibling medians, and the
  distinction is the strongest thing in the submission.
- Do not say "real time". The ingest is a batch and the runbook is honest about that.

## If something breaks

- **Ingest button errors** — check `INGEST_ENABLED=true` is set and the path is under
  `VERDICT_INGEST_ROOT`. The error message says which.
- **Console shows no runs** — the reload happens on the client; hit refresh again.
- **Verdict.AI cannot reach ClickHouse** — the MCP server binds on 8001; confirm the container is
  up. Skip the chat section rather than debugging it on camera.
- **Worst case** — the five persisted runs and the full diagnosis are already committed at
  `artifacts/unseen/DIAGNOSIS.md`. Walk through that instead.
