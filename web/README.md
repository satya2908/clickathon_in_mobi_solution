# Verdict console

Dashboard and observability UI for [verdict](../clickathon_in_mobi_solution) — the automated
autonomous investigator agent for ad-tech metrics on ClickHouse.

Currently a **UI mockup**: the layout, interactions and vocabulary are real, the numbers come
from `lib/data.ts` rather than from ClickHouse.

## Running

```bash
npm install
npm run dev
```

Then open <http://localhost:3131> (or whatever port `next dev` reports).

## Design system

There is no Tailwind and no component library. The visual language is the one from
`panacea-nucodegraph/web`: CSS custom properties in `app/globals.css`, short class names
(`.tbl`, `.btn`, `.badge`, `.seg`, `.panelbox`), 1px borders instead of shadows, 13px system
sans with monospace for anything machine-readable. Light is the default; the toggle in the top
bar switches to the same dark palette the reference console uses.

Two constraints hold it together:

- **Six type sizes** — 10, 11, 12, 13, 15, 24. Ten is for uppercase labels, 24 for KPI numbers.
- **Four hues** — indigo for interactive and selected, then red, amber and green. Everything else
  is greyscale, so a colour never means two things. `localized` is the ordinary verdict and gets
  no colour at all; the kinds that need a second look are the ones that stand out.

## Layout

| Region | What it shows |
| --- | --- |
| Top bar | Tenant, the window and grain that scope every number, the health chip, and links out. Nothing here is disabled: a control with no backend behind it was deleted rather than greyed out, and the window and grain are static text because they are labels, not controls. |
| KPI tiles | Open cases split by verdict kind, revenue at risk, mean confidence against the publish threshold, coverage gaps. Four numbers, one line of scope each. |
| Parent series | Observed vs expected with the ±4.5% band and the incident onset marked. Sized to a glance, not to a report. |
| Control strip | Verdict-kind filter chips with counts, plus a free-text filter over metric and segment. |
| Case table | One 30px row per row of `cases` — twelve columns, no sentences, sortable by priority, effect, confidence or impact. |
| Case panel | Opens on click. The header is the title plus three pills — how far off, what it costs, how much to trust it. Trace on the left selects **one** node into the right, with a `?` drawer for why each stage exists; evidence and narrative are separate tabs. |

There is no nav: the console is one page.

Nothing is stated in two places. The window and grain are in the top bar because they scope every
number on every screen; the run id is in the status bar because that is a property of the data,
not of any one case. Detector, mode, p and φ used to sit in the panel header and now live only
where they are evidence — in the `temporal:*` and `localize:*` spans, and in the confidence
breakdown on the Evidence tab.

Health lives in the top bar rather than in a panel on the page: only a partial run or a failed
accuracy check qualifies, and the chip disappears entirely when everything passed. Anything
already visible as a badge on a case row is not an alert.

Cases are deep-linkable: opening one pushes the case-id prefix to the URL hash, so
`/#8f31a0c47b2e` opens straight into that case and the browser Back button closes the panel.

## Wiring it to real data

Everything the UI renders is typed in `lib/types.ts` against the verdict result schema, so the
field names already match the columns:

- `Case` → `cases`, one row per verdict
- `Candidate` → `case_candidates`, the exoneration ledger
- `Step` → `case_steps`, which is also one OpenTelemetry span in HyperDX
- `CoverageGap` → `coverage_ledger`
- `Run` → `runs`

Replacing the mock means swapping the exports in `lib/data.ts` for queries against those tables.
Two things in the UI are derived rather than stored and would need to stay derived:

- **Priority** (`lib/format.ts`) is `|impact.revenue| × confidence` bucketed into P0–P3, so a
  large number nobody can stand behind does not outrank a proven small one.
- **`traceFor()`** builds the span tree from each case's own numbers. Against a real backend it
  would read `case_steps` by `case_id` instead.
- **`clearedOf()`** counts only `cleared`, `immaterial`, `wrong_direction` and
  `did_not_reproduce` as exonerated. `too_broad`, `too_narrow` and `partial` are part of the
  answer, and counting them as ruled out overstates the one number this product cannot afford to
  overstate.

`confidence` is computed from `confidence_json` rather than stored, so the total in the table can
never disagree with the component breakdown in the panel. Unscored components contribute zero
instead of being renormalised away — a check that could not run is missing evidence, not neutral
evidence.

`Step` carries only its `result`. What a stage does and why it exists are the same eleven
sentences on every case, so they live once in `STEP_METHOD` and are read deliberately from the
Method drawer instead of being restated on all 23 spans of every trace.
