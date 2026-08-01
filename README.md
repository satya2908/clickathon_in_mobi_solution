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
docker compose up -d                          # starts ClickStack, builds Verdict
docker compose exec verdict verdict schema apply
docker compose exec verdict verdict load
```

HyperDX is then at <http://localhost:8080>, where each investigation appears as a trace.

### Without Docker

```bash
uv venv && uv pip install -e ".[dev]"
set -a && source .env && set +a
verdict config check                          # validates config, touches no network
verdict schema apply
verdict load --data-dir ../hackathon_dataaset/InMobi/data
```

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

## Licence

MIT. See [LICENSE](LICENSE).
