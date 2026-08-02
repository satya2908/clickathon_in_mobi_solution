# Verdict

An automated root-cause analyst for ad-tech metrics, built on ClickHouse.

When a metric moves, Verdict finds the segment responsible, proves the claim by removing that
segment and showing the parent metric returns to normal, publishes what it ruled out and why,
and attaches a confidence score you can audit line by line. A language model writes the prose.
It does not decide anything: switch it off and every number in every case file is identical.

Built for the ClickHouse Click-a-thon 2026, InMobi "Automated Root-Cause Analyst" track.

---

## The problem with the obvious approach

Rank every segment by how far it moved, report the worst one. This fails in three ways that
matter, and all three are present in the hackathon dataset.

**It reports passengers as drivers.** When Android 15 fill rate collapses, every device model
that skews Android 15 also drops. Galaxy S23 shows a large, statistically significant decline.
It is not the cause; it is downstream of the cause. Ranking cannot tell the two apart because
both look identical from the top.

**It cannot see interactions.** One incident here only exists at the intersection of APAC and
iOS 18.1. Viewed one dimension at a time, APAC looks mildly off and iOS 18.1 looks mildly off,
and neither clears a sensible threshold. The cause is invisible to any one-dimensional scan.

**It cannot see compensating pairs.** Another incident moves eCPM down for one ad format and
up for another by almost exactly the offsetting amount. Every aggregate stays flat. A detector
watching totals sees a perfectly healthy system.

Verdict addresses these with an explain-away test rather than a ranking, two complementary
detectors rather than one, and a published ledger of what it cleared rather than silence.

---

## Quick start

You need a ClickHouse Cloud service and the hackathon dataset.

```bash
git clone https://github.com/satya2908/clickathon_in_mobi_solution
cd clickathon_in_mobi_solution
cp .env.example .env        # fill in CLICKHOUSE_HOST / CLICKHOUSE_PASSWORD
```

### With Docker

```bash
docker compose up -d                          # whole stack: console, chat, tracing, MCP
docker compose exec verdict verdict schema apply
docker compose exec verdict verdict load
docker compose exec verdict verdict investigate --start 2026-07-05 --hours 24
```

Then open the console. Everything else is reachable from it.

| Surface | Where | What it is |
|---|---|---|
| Console | <http://localhost:3000> | cases, evidence, investigation traces |
| Verdict.AI | bottom-right of the console | chat, with a SQL tool on the same tables |
| LibreChat | <http://localhost:3080> | the same chat, full window |
| MCP server | <http://localhost:8001/sse> | the SQL tool, on its own, for any MCP client |
| HyperDX | hosted, with ClickHouse Cloud | every investigation as a distributed trace |

Ports are overridable — `WEB_PORT`, `LIBRECHAT_PORT`, `MCP_PORT`, `HYPERDX_UI_PORT`.

Traces go to the collector, which writes them to ClickHouse, and the HyperDX that ships with
ClickHouse Cloud reads them from there — so the default needs no extra container. To run
ClickStack locally instead, start the profile and repoint the console at it:

```bash
docker compose --profile selfhosted up -d
NEXT_PUBLIC_HYPERDX_URL=http://localhost:8080     # in .env, then rebuild web
```

### Without Docker

```bash
uv venv && uv pip install -e ".[dev]"
set -a && source .env && set +a
verdict config check                          # validates config, touches no network
verdict schema apply
verdict load --data-dir ../hackathon_dataaset/InMobi/data
verdict investigate --start 2026-07-05 --hours 24
```

`verdict investigate` takes `--grain 5m|1h|1d`, `--metric` (repeatable), `--no-llm` to force
template prose, and `--no-persist` to investigate without writing anything.

---

## What a run does

Five stages, one span each, all of them recorded. `verdict investigate` on a 24-hour window
takes a few seconds and writes a case file per finding.

**Detect.** Two detectors that fail in different directions, which is the point of having two.
The *temporal* detector compares each cell against its own history — a robust baseline over
prior weeks, with overdispersion estimated rather than assumed, because ad traffic is nowhere
near Poisson and a binomial test on it reports everything as significant. It catches main
effects and needs history. The *structural* detector compares each cell against its siblings
in the same bucket, needs no history at all, and is the one that sees compensating pairs and
interaction-only incidents. A *confirmatory* pass re-tests what either one proposes.

Multiple testing is real here: a 1-way and 2-way lattice over this dataset is tens of
thousands of simultaneous tests per metric per bucket, so p-values go through
Benjamini-Hochberg. Cells too small to resolve the effect are not tested and not silently
dropped — they go to the coverage ledger with the reason and the smallest effect they could
have detected.

**Localize.** Candidates are not ranked. Each is subjected to counterfactuals that can fail:

- *Sufficiency* — remove the candidate from the parent. Does the parent return to
  expectation? This is the explain-away test, and it is what separates a driver from a
  passenger, because removing Galaxy S23 does not repair Android 15.
- *Minimality* — remove the candidate's guiltiest child. Does the candidate still deviate? If
  not, the answer is the child, and reporting the parent is a true statement that is too broad
  to act on.
- *Maximality* — do the candidate's siblings move too? If they do, the cause is the parent.
- *Holdout* — split the window in half. Does the effect reproduce in both? An effect present
  in one half is a spike with a story attached.

**Score.** Confidence is a weighted reading of five components — significance, sufficiency,
minimality, stability, separation — each published with the number, the weight, and a sentence
saying what it measured. A component that could not be tested scores nothing and says so
rather than defaulting to a value; the weights renormalise around it. There is no model in
this path. The same inputs give the same score.

**Narrate.** A language model writes the prose from a bundle of already-computed figures.
Every number in the draft is checked against that bundle, and a draft containing a figure that
is not in it is rejected and replaced with the template. Prose is presentation, never
evidence: `--no-llm` changes the wording and nothing else.

**Publish.** Case, candidates (including every cleared one and why), per-step trace, and
coverage gaps go to ClickHouse, and the trace also goes to HyperDX over OTLP.

### What the corpus currently produces

From 9M events, an eight-run history: 58 cases, 2,373 candidates evaluated, 3,277 recorded
steps, 26,913 coverage-ledger rows.

The candidate outcomes are the interesting part, because they are mostly refusals — 1,434
rejected as too narrow, 370 as too broad, 335 moving the wrong way, 28 that did not reproduce
across the holdout, 19 immaterial. Fourteen survived everything and were accused. A detector
that accepted its first plausible answer would have reported any of the other 2,359.

Forty-four cases are published as *unlocalized*: something moved, nothing survived the
counterfactuals, and saying so is the honest result. Of the narratives, 21 are model-written
and numerically verified; the rest fell back to the template, mostly against free-tier quota
limits, which is exactly the degradation the fallback exists for.

---

## The console

`http://localhost:3000`. Four things you can do with a case:

- **Trace** — every step of the investigation as a tree or a waterfall, each node carrying
  what it did, why it did it, what it concluded, and the SQL it ran. The waterfall is drawn
  from recorded start offsets, so the gaps are real gaps.
- **Evidence** — the confidence breakdown component by component, and the full candidate table
  with each rejection and its reason.
- **Narrative** — the prose, with its impact figures alongside it.
- **Actions** — optional. Off by default; see below.

Cases carry a priority derived from impact and confidence, and impact is left explicitly
unpriced for count metrics rather than being rounded to zero dollars.

### Actions

A toggle, default off. When enabled, each case is sent to a remediation service twice: once to
draft steps, and once more, in a separate session, to review the draft against the same
evidence and drop what it cannot support. Both passes are advisory and labelled as such. The
pipeline's verdicts do not depend on them.

---

## Verdict.AI and MCP

The console has a chat bubble backed by LibreChat. It has one tool: SQL against the same
tables the detectors read, over the official ClickHouse MCP server. It is attached by default
and pinned in the composer, because a chat that quietly stops consulting the database and
answers from memory is worse than no chat.

The server is told about the schema before its first query — that rollups hold counters and
never rates, how the `combo`/`key_a`/`key_b` keying works, which tables hold pipeline output,
and that the coverage ledger exists and a segment missing from `cases` was not necessarily
innocent. It is also told to show its SQL, so anything it says can be checked.

One protocol note worth recording, because it cost an afternoon. Everything else in this
project reaches Gemini through its OpenAI-compatible endpoint, so the pipeline and the chat
share one wire format and the provider can be changed by editing a URL. That does not survive
tool use: Gemini attaches a `thought_signature` to every function call and rejects the
following turn unless it is echoed back, and the field rides in `extra_content`, which has no
place in the OpenAI schema and is dropped in translation. Identical requests, measured
directly against the API, return 200 with the signature and 400 without it. No model setting
avoids it. So the chat spec alone uses LibreChat's native Google provider, which has somewhere
to put the field.

### From other MCP clients

The server is published on `localhost:8001` and speaks SSE, so anything that speaks MCP can
use it. For Cursor, `.cursor/mcp.json` is in the repository and needs no edit:

```json
{ "mcpServers": { "verdict-clickhouse": { "url": "http://localhost:8001/sse" } } }
```

---

## Tracing

Every run emits OpenTelemetry spans to the collector, which writes them to ClickHouse, where
HyperDX renders them. Cases store their `trace_id`, so the console deep-links each case to its
own trace — with the time range already set, because HyperDX otherwise opens on Live Tail and
finds nothing, which reads as a lost trace rather than as the wrong hour.

The same steps are also written to `case_steps` as first-class rows, and the console reads
those. The investigation view therefore works whether or not anyone is running the
observability stack: HyperDX is where you go to see one run against everything else happening
at the time, not a dependency of being able to read a case.

---

## Configuration

Behaviour lives in `config/*.yaml`; credentials live in the environment. The YAML holds
`${VAR}` placeholders resolved at load time, so the same file is valid unchanged as a local
file, a Docker bind mount, or a Kubernetes ConfigMap, with secrets arriving separately.

| Placeholder | Meaning |
|---|---|
| `${VAR}` | required, startup fails naming the missing variable |
| `${VAR:-default}` | falls back when unset **or** empty |
| `${VAR-default}` | falls back only when unset |

`verdict config check` validates everything without connecting anywhere, and doubles as the
container healthcheck so a broken ConfigMap shows up as an unhealthy container rather than as
a run that dies partway through.

Two settings are worth knowing about before you touch them:

- **`retention.enforce`** defaults to `false`. The day counts describe production intent, but a
  historical dataset is by definition older than a 7-day raw TTL, so enabling this on an
  analysis corpus instructs ClickHouse to delete all of it on the first background merge.
- **`llm.enabled`** can be `false` at any time. Prose degrades to a template; no number changes.

---

## Data model

Nothing stores a metric. Rollups store additive counters and every metric is divided out at
read time. This is not a stylistic preference: a stored fill rate cannot be re-aggregated,
because averaging hourly fill rates only reproduces the daily fill rate if every hour carried
identical traffic, which never happens. Storing counters makes a rollup row mean the same
thing at every grain and in every combination.

Rollups hold a **1-way and 2-way lattice** in long form — `(bucket, combo, key_a, key_b,
requests, fills, impressions, clicks, revenue)`, where `combo` names the keying
(`region`, `region|os_version`, or `__all__`). The alternative, one row per full dimension
tuple, was measured on the real dataset and rejected: it compresses 9M events to 7.9M rows at
hourly grain, a pointless 1.14x, because dimension cardinality is high relative to event
volume. The lattice gives 6.4x hourly and 150x daily. The cost is that three-way interactions
are out of reach, which is a stated limit rather than a hidden one.

All 46 combinations are produced by a single `ARRAY JOIN` inside one materialized view. Using
46 separate views would be 46 chances for the streaming path and the backfill path to define a
bucket boundary differently, and any such disagreement appears in the data as a step change
indistinguishable from a real incident.

Grains chain 5m → 1h → 1d, each a `SummingMergeTree` with its own retention. Chaining views
off a summing table is safe because a view sees pre-merge blocks, and a sum of partial sums is
the total.

### Which slices are legal

`advertiser_id` is empty on unfilled requests, so `vertical` and `campaign_type` exist only on
filled events. Slicing **fill rate** by `campaign_type` therefore returns 1.0 for every value:
the denominator has quietly become "filled requests". A naive scan reports that as a clean and
confident finding, and it is entirely an artefact.

Rather than hand-listing the illegal pairs, each dimension records the funnel stage at which it
becomes known and each metric records the stage of its denominator population. A slice is legal
only when every row the metric counts actually carries the dimension. Run `verdict config
matrix` to see the derived result and the reason behind each refusal.

---

## Verification at load time

Two failure modes here are quiet enough to corrupt every downstream conclusion while every
command still reports success, so the loader checks for both:

- A dimension CSV that is really a **Git LFS pointer stub**. It parses as valid CSV, the
  dictionary loads with three entries, every lookup returns `''`, and the entire lattice
  collapses to one empty-string segment. Every metric still computes; every answer is wrong.
- **Rollups disagreeing with the facts.** Each grain must reproduce the raw totals exactly, and
  each one-way combo must independently sum to the grand total. A lattice that is not a
  partition of the data produces statistics that are internally consistent and wrong.

Every glossary metric is also computed twice, once from raw events and once from the rollup,
and the two must agree. That makes the published formulas testable rather than aspirational.

---

## Testing against known answers

Accuracy on real data is unmeasurable, because nobody knows the true cause. `verdict inject`
plants synthetic incidents in a copy of the corpus and writes the answer key beside it, which
turns "does this look right" into a score.

The catalogue covers the three shapes that defeat ranking — a main effect with passengers
downstream of it, an interaction visible only at an intersection, and a compensating pair that
leaves every total flat — plus a high-cardinality segment, a slow drift that no
window-versus-window comparison can see, and a **clean** window with nothing planted in it at
all. That last one is the one that matters most: a detector is only as good as its willingness
to return nothing, and the clean case is the only test that can catch an invented incident.

The suite is 460 tests over the statistics, the counterfactuals, the confidence scoring, the
schema, and the narration guard, and it runs in about four seconds without touching the
network.

```bash
pytest -q
```

---

## Layout

```
src/verdict/
  detect.py structural.py   the two detectors, and dispersion estimation
  localize.py               the counterfactuals: sufficiency, minimality, maximality, holdout
  confidence.py             the five scored components
  narrate.py llm.py         prose, and the numeric guard that can reject it
  stats.py                  baselines, intervals, corrections
  inject.py injectcat.py    synthetic incidents and their answer key
  schema.py load.py db.py   ClickHouse DDL, loading, verification
  pipeline.py cli.py        stage orchestration and the command surface
  trace.py store.py         spans, and everything that gets persisted
web/                        the console (Next.js)
config/                     behaviour as YAML; also the LibreChat and collector config
```

---

## Licence

MIT. See [LICENSE](LICENSE).
