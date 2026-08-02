# Diagnosis — unseen incident dataset

**Team Low Cardinality** · ClickHouse Click-a-thon 2026 · InMobi *Automated Root-Cause Analyst*

Produced by the pipeline. Every number below is read back out of ClickHouse from what the run persisted, and every one carries the query that returns it.

## The diagnosis

On **2026-07-08**, fill rate for **os_version=iOS 17.5** fell to **0.47773** against an expectation of **0.79118** — a **-39.6%** move, confidence **0.67**.

The expectation is not historical. This release reissued its dimension tables, so a segment label names one group of entities before the boundary and a different group after it, and the audit rejected segment-level history for these windows. The figure above is the unweighted median fill rate across the sibling levels of `os_version` inside the same window, which needs no history at all.

Case `75f902835e2055c284de0a6c3d6e0b08` · run `62b0021401f84b11a29eec93a13a1c01` · trace `9b04a3357220fe9a33fc2128a1555508`

### What the system wrote

*(llm, verified against the computed figures)*

Fill rate for os_version=iOS 17.5 fell 39.6% in the window 2026-07-08 00:00 to 2026-07-09 00:00, from an expected 79.1% to an observed 47.8%.

## What moved
- os_version=iOS 17.5: observed 47.8%, expected 79.2%, relative effect -39.7%, flagged by the siblings detector

## Why this segment
- sufficiency: held, because parent moved -0.059793 and with os_version=iOS 17.5 removed it moves +0.000488359, so 99% of the change is accounted for by this segment
- minimality: held, because os_version=iOS 17.5 sits -0.313451 from the sibling norm of 0.79118 and with its worst child (os_version=iOS 17.5 AND region=APAC) removed, 98% of that gap is still there, so the child is not the whole story
- maximality: held, because 100% of os_version=iOS 17.5's traffic sits in sub-segments that moved with it, across 6 way(s) of splitting it, meaning the fault is spread across the segment rather than hiding in one corner of it

## What could not be tested
- ad_format=rewarded: sits -4.7% from its siblings, under the 5.0% this metric needs to be worth reporting
- ad_format=interstitial: sits +0.0% from its siblings, under the 5.0% this metric needs to be worth reporting
- ad_format=banner: no rollup cell covers the overlap between this segment and the accused, so no prediction could be made for it
- ad_format=native: sits +2.1% from its siblings, under the 5.0% this metric needs to be worth reporting
- ad_format=video: no rollup cell covers the overlap between this segment and the accused, so no prediction could be made for it
- category=entertainment: sits -1.8% from its siblings, under the 5.0% this metric needs to be worth reporting
- category=ecommerce: sits +3.2% from its siblings, under the 5.0% this metric needs to be worth reporting
- category=utility: sits +1.1% from its siblings, under the 5.0% this metric needs to be worth reporting

## Confidence
- 66.5%
- significance score 25.0% (weight 29.4%)
- sufficiency score 99.2% (weight 35.3%)
- minimality score 98.5% (weight 17.6%)
- stability score 0.0% (weight 0.0%)
- separation score 38.5% (weight 17.6%)

## The numbers, and how to reproduce them

### The incident, per day, straight from the events

```sql
SELECT toDate(event_time) AS day,
       round(sumIf(is_filled, os = 'iOS 17.5') / nullIf(countIf(os = 'iOS 17.5'), 0), 5) AS ios_17_5,
       round(sumIf(is_filled, os != 'iOS 17.5') / nullIf(countIf(os != 'iOS 17.5'), 0), 5) AS everything_else,
       countIf(os = 'iOS 17.5') AS requests
FROM (SELECT event_time, is_filled,
             dictGet('dict_geo_device', 'os_version', geo_device_id) AS os
      FROM ad_events WHERE event_time >= '2026-07-01')
GROUP BY day ORDER BY day;
```

| day | iOS 17.5 | everything else | requests | gap |
|---|---|---|---|---|
| 2026-07-01 | 0.78570 | 0.78428 | 55,380 | +0.2% |
| 2026-07-02 | 0.78428 | 0.78344 | 54,259 | +0.1% |
| 2026-07-03 | 0.78486 | 0.78413 | 53,290 | +0.1% |
| 2026-07-04 | 0.78388 | 0.78390 | 44,402 | -0.0% |
| 2026-07-05 | 0.78421 | 0.78529 | 45,711 | -0.1% |
| 2026-07-06 | 0.79454 | 0.79259 | 58,093 | +0.2% |
| 2026-07-07 | 0.79691 | 0.79350 | 58,170 | +0.4% |
| 2026-07-08 | 0.47773 | 0.79167 | 58,395 | -39.7% |
| 2026-07-09 | 0.47664 | 0.79308 | 57,247 | -39.9% |
| 2026-07-10 | 0.79559 | 0.79294 | 55,609 | +0.3% |

Flat within half a percent through Jul 7, collapsed on Jul 8 and Jul 9, recovered on Jul 10. The system reported it on exactly those two days and stayed silent on the third.

### Why segment-level history was refused

```sql
SELECT toDate(event_time) AS day, count() AS requests,
       round(sum(is_filled) / count(), 5) AS fill_rate
FROM ad_events WHERE event_time >= '2026-07-01' GROUP BY day ORDER BY day;
```

| day | requests | platform fill rate |
|---|---|---|
| 2026-07-01 | 288,553 | 0.78456 |
| 2026-07-02 | 284,319 | 0.78360 |
| 2026-07-03 | 275,480 | 0.78427 |
| 2026-07-04 | 232,726 | 0.78390 |
| 2026-07-05 | 239,194 | 0.78509 |
| 2026-07-06 | 303,115 | 0.79296 |
| 2026-07-07 | 303,589 | 0.79415 |
| 2026-07-08 | 304,116 | 0.73139 |
| 2026-07-09 | 298,552 | 0.73241 |
| 2026-07-10 | 290,628 | 0.79345 |

The platform total runs straight through the Jul 5 → Jul 6 corpus boundary with no step, because relabelling which entities carry which attribute cannot move a total. Below it, 42.4% of the grid stopped agreeing with its own history on Jul 6. That gap is the whole argument for keeping the aggregate's history while refusing every segment's.

## Every case the system published

### 2026-07-06 — 12 case(s)

`run_id = 424382dacb9a4809beae37193739ce04` · 10 cells tested · 30.69s · status `complete`

| metric | segment | verdict | observed | expected | change | confidence |
|---|---|---|---|---|---|---|
| ecpm | ad_format=native | localized | 1.80567 | 2.49982 | -27.8% | 0.50 |
| rpr | region=NAM | localized | 0.00281 | 0.00124 | +127.1% | 0.50 |
| ecpm | region=NAM | localized | 3.62685 | 1.59359 | +127.6% | 0.49 |
| requests | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 41576.00000 | 7592.28347 | +447.6% | 0.00 |
| impressions | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 32389.00000 | 5942.97192 | +445.0% | 0.00 |
| fills | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 33052.00000 | 6073.39903 | +444.2% | 0.00 |
| clicks | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 363.00000 | 74.50804 | +387.2% | 0.00 |
| requests | category=gaming AND publisher_tier=tier_2 | undecomposed | 29475.00000 | 63277.67387 | -53.4% | 0.00 |
| ctr | category=news AND region=NAM | unlocalized | 0.01245 | 0.00816 | +52.6% | 0.00 |
| clicks | category=gaming AND publisher_tier=tier_2 | undecomposed | 257.00000 | 533.18605 | -51.8% | 0.00 |
| impressions | category=gaming AND publisher_tier=tier_2 | undecomposed | 23493.00000 | 48678.48814 | -51.7% | 0.00 |
| fills | category=gaming AND publisher_tier=tier_2 | undecomposed | 23993.00000 | 49570.18328 | -51.6% | 0.00 |

### 2026-07-07 — 12 case(s)

`run_id = e02af081733f41c5924f6918d44d7b62` · 10 cells tested · 34.14s · status `complete`

| metric | segment | verdict | observed | expected | change | confidence |
|---|---|---|---|---|---|---|
| rpr | ad_format=native | localized | 0.00142 | 0.00192 | -26.4% | 0.54 |
| rpr | region=NAM | localized | 0.00282 | 0.00123 | +128.8% | 0.50 |
| ecpm | ad_format=native | localized | 1.80388 | 2.51153 | -28.2% | 0.49 |
| ecpm | region=NAM | localized | 3.62700 | 1.58751 | +128.5% | 0.49 |
| clicks | country=ID AND os_version=iOS 17.5 | undecomposed | 308.00000 | 20.23768 | +1421.9% | 0.00 |
| requests | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 41385.00000 | 7699.69976 | +437.5% | 0.00 |
| impressions | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 32283.00000 | 6022.45941 | +436.0% | 0.00 |
| fills | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 32902.00000 | 6147.23624 | +435.2% | 0.00 |
| clicks | category=gaming AND publisher_tier=tier_2 | undecomposed | 234.00000 | 582.54000 | -59.8% | 0.00 |
| requests | category=gaming AND publisher_tier=tier_2 | undecomposed | 29751.00000 | 63017.02993 | -52.8% | 0.00 |
| fills | category=gaming AND publisher_tier=tier_2 | undecomposed | 24257.00000 | 51025.98660 | -52.5% | 0.00 |
| impressions | category=gaming AND publisher_tier=tier_2 | undecomposed | 23809.00000 | 50019.48022 | -52.4% | 0.00 |

### 2026-07-08 — 13 case(s)

`run_id = 62b0021401f84b11a29eec93a13a1c01` · 10 cells tested · 35.77s · status `complete`

| metric | segment | verdict | observed | expected | change | confidence |
|---|---|---|---|---|---|---|
| fill_rate | os_version=iOS 17.5 | localized | 0.47773 | 0.79118 | -39.6% | 0.67 |
| rpr | ad_format=native | localized | 0.00131 | 0.00179 | -26.8% | 0.54 |
| rpr | region=NAM | localized | 0.00271 | 0.00106 | +155.7% | 0.50 |
| ecpm | region=NAM | localized | 3.63631 | 1.59577 | +127.9% | 0.49 |
| clicks | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 229.00000 | 31.76171 | +621.0% | 0.00 |
| requests | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 41770.00000 | 7586.86392 | +450.6% | 0.00 |
| fills | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 19964.00000 | 3724.89609 | +436.0% | 0.00 |
| impressions | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 19546.00000 | 3655.32431 | +434.7% | 0.00 |
| requests | category=gaming AND publisher_tier=tier_2 | undecomposed | 29879.00000 | 65429.53331 | -54.3% | 0.00 |
| fills | category=gaming AND publisher_tier=tier_2 | undecomposed | 22341.00000 | 47411.60375 | -52.9% | 0.00 |
| impressions | category=gaming AND publisher_tier=tier_2 | undecomposed | 21871.00000 | 46106.51088 | -52.6% | 0.00 |
| clicks | category=gaming AND publisher_tier=tier_2 | undecomposed | 250.00000 | 476.96825 | -47.6% | 0.00 |
| fill_rate | country=PH AND device_model=iPhone 13 | unlocalized | 0.79014 | 0.71996 | +9.7% | 0.00 |

### 2026-07-09 — 13 case(s)

`run_id = 47adb54dd6fe42728a656ee3c2f5d8d9` · 10 cells tested · 0.00s · status `running`

| metric | segment | verdict | observed | expected | change | confidence |
|---|---|---|---|---|---|---|
| fill_rate | os_version=iOS 17.5 | localized | 0.47664 | 0.79335 | -39.9% | 0.67 |
| rpr | os_version=iOS 17.5 | localized | 0.00084 | 0.00174 | -51.8% | 0.66 |
| rpr | region=NAM | localized | 0.00258 | 0.00100 | +156.6% | 0.53 |
| ecpm | region=NAM | localized | 3.44987 | 1.51094 | +128.3% | 0.53 |
| clicks | country=ID AND os_version=iOS 17.5 | undecomposed | 193.00000 | 4.87243 | +3861.1% | 0.00 |
| requests | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 40891.00000 | 7500.53261 | +445.2% | 0.00 |
| impressions | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 19054.00000 | 3787.70515 | +403.0% | 0.00 |
| fills | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 19457.00000 | 3870.02993 | +402.8% | 0.00 |
| requests | device_model=Pixel 7 AND os_version=Android 13 | undecomposed | 3293.00000 | 13998.75226 | -76.5% | 0.00 |
| fills | device_model=Pixel 7 AND os_version=Android 13 | undecomposed | 2637.00000 | 11106.58534 | -76.3% | 0.00 |
| impressions | device_model=Pixel 7 AND os_version=Android 13 | undecomposed | 2590.00000 | 10837.75747 | -76.1% | 0.00 |
| clicks | category=gaming AND publisher_tier=tier_2 | undecomposed | 248.00000 | 423.95652 | -41.5% | 0.00 |
| fill_rate | country=PH AND device_model=iPhone 13 | unlocalized | 0.78891 | 0.73171 | +7.8% | 0.00 |

### 2026-07-09 — 13 case(s)

`run_id = 47adb54dd6fe42728a656ee3c2f5d8d9` · 10 cells tested · 36.16s · status `complete`

| metric | segment | verdict | observed | expected | change | confidence |
|---|---|---|---|---|---|---|
| fill_rate | os_version=iOS 17.5 | localized | 0.47664 | 0.79335 | -39.9% | 0.67 |
| rpr | os_version=iOS 17.5 | localized | 0.00084 | 0.00174 | -51.8% | 0.66 |
| rpr | region=NAM | localized | 0.00258 | 0.00100 | +156.6% | 0.53 |
| ecpm | region=NAM | localized | 3.44987 | 1.51094 | +128.3% | 0.53 |
| clicks | country=ID AND os_version=iOS 17.5 | undecomposed | 193.00000 | 4.87243 | +3861.1% | 0.00 |
| requests | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 40891.00000 | 7500.53261 | +445.2% | 0.00 |
| impressions | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 19054.00000 | 3787.70515 | +403.0% | 0.00 |
| fills | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 19457.00000 | 3870.02993 | +402.8% | 0.00 |
| requests | device_model=Pixel 7 AND os_version=Android 13 | undecomposed | 3293.00000 | 13998.75226 | -76.5% | 0.00 |
| fills | device_model=Pixel 7 AND os_version=Android 13 | undecomposed | 2637.00000 | 11106.58534 | -76.3% | 0.00 |
| impressions | device_model=Pixel 7 AND os_version=Android 13 | undecomposed | 2590.00000 | 10837.75747 | -76.1% | 0.00 |
| clicks | category=gaming AND publisher_tier=tier_2 | undecomposed | 248.00000 | 423.95652 | -41.5% | 0.00 |
| fill_rate | country=PH AND device_model=iPhone 13 | unlocalized | 0.78891 | 0.73171 | +7.8% | 0.00 |

### 2026-07-10 — 10 case(s)

`run_id = 02ffde24fd6043d9b543a385d6d075c3` · 10 cells tested · 0.00s · status `running`

| metric | segment | verdict | observed | expected | change | confidence |
|---|---|---|---|---|---|---|
| ecpm | region=NAM | localized | 3.44609 | 1.51084 | +128.1% | 0.54 |
| rpr | region=NAM | localized | 0.00268 | 0.00117 | +128.4% | 0.53 |
| clicks | country=ID AND os_version=iOS 17.5 | undecomposed | 291.00000 | 11.44598 | +2442.4% | 0.00 |
| requests | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 39569.00000 | 7397.04024 | +434.9% | 0.00 |
| fills | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 31468.00000 | 5928.36233 | +430.8% | 0.00 |
| impressions | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 30862.00000 | 5816.82529 | +430.6% | 0.00 |
| fills | device_model=Pixel 7 AND os_version=Android 13 | undecomposed | 2553.00000 | 10702.14324 | -76.1% | 0.00 |
| impressions | device_model=Pixel 7 AND os_version=Android 13 | undecomposed | 2495.00000 | 10452.12970 | -76.1% | 0.00 |
| requests | category=gaming AND publisher_tier=tier_2 | undecomposed | 28568.00000 | 59968.43883 | -52.4% | 0.00 |
| clicks | category=gaming AND publisher_tier=tier_2 | undecomposed | 223.00000 | 439.14286 | -49.2% | 0.00 |

### 2026-07-10 — 10 case(s)

`run_id = 02ffde24fd6043d9b543a385d6d075c3` · 10 cells tested · 34.52s · status `complete`

| metric | segment | verdict | observed | expected | change | confidence |
|---|---|---|---|---|---|---|
| ecpm | region=NAM | localized | 3.44609 | 1.51084 | +128.1% | 0.54 |
| rpr | region=NAM | localized | 0.00268 | 0.00117 | +128.4% | 0.53 |
| clicks | country=ID AND os_version=iOS 17.5 | undecomposed | 291.00000 | 11.44598 | +2442.4% | 0.00 |
| requests | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 39569.00000 | 7397.04024 | +434.9% | 0.00 |
| fills | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 31468.00000 | 5928.36233 | +430.8% | 0.00 |
| impressions | device_model=iPhone 14 AND os_version=iOS 17.5 | undecomposed | 30862.00000 | 5816.82529 | +430.6% | 0.00 |
| fills | device_model=Pixel 7 AND os_version=Android 13 | undecomposed | 2553.00000 | 10702.14324 | -76.1% | 0.00 |
| impressions | device_model=Pixel 7 AND os_version=Android 13 | undecomposed | 2495.00000 | 10452.12970 | -76.1% | 0.00 |
| requests | category=gaming AND publisher_tier=tier_2 | undecomposed | 28568.00000 | 59968.43883 | -52.4% | 0.00 |
| clicks | category=gaming AND publisher_tier=tier_2 | undecomposed | 223.00000 | 439.14286 | -49.2% | 0.00 |

## The trace

Every step the system took to reach the headline verdict, as persisted in `case_steps` for case `75f902835e2055c284de0a6c3d6e0b08`. Same tree the console renders, and the same trace id (`9b04a3357220fe9a33fc2128a1555508`) carried into OpenTelemetry.

```sql
SELECT name, kind, offset_ms, duration_ms, what, why, result
FROM case_steps WHERE case_id = '75f902835e2055c284de0a6c3d6e0b08'
ORDER BY offset_ms, step_id;
```

11 steps.

- **investigate** *(pipeline, +68ms, 8585ms)*
  - what: Investigating 2026-07-08 00:00 to 2026-07-09 00:00 at 1h grain
  - why: One root span per run, so every stage below shares a trace id and the case can link a reader to the whole investigation rather than to one stage of it.
  - **audit** *(statistics, +627ms, 1644ms)*
    - what: Checked whether the baseline still describes this population
    - why: A baseline drawn from a population that has since changed produces confident, internally consistent, wrong answers, and nothing downstream can detect that. A calibrated baseline disagrees with a few percent of a recent window; one describing the wrong population disagrees with most of it.
    - result: Baseline rejected: 42.1% of all tested cells are flagged on a recent window, against a bar of 10%.
  - **detect** *(detector, +2272ms, 124ms)*
    - what: Scanned 10 metric(s) over 2026-07-08 00:00 to 2026-07-09 00:00 at 1h
    - why: Segment-level history failed its audit, so segments are compared only against their siblings in the same window, which needs no baseline. The platform aggregate keeps its own history: relabelling entities rearranges segments and cannot move a total, and a total has no siblings to be compared against instead.
    - result: 10 cells tested, 159 structural anomalies, 1090 could not be tested
    - **temporal:fill_rate:__all__** *(detector, +2295ms, 0ms)*
      - what: Compared every __all__ cell of Fill rate in 2026-07-08 00:00 to 2026-07-09 00:00 against the same window in each of the previous 4 weeks.
      - why: Weekly alignment holds weekday and hour-of-day constant, so ordinary seasonality cannot masquerade as a change.
      - result: 1 cell(s) deviated beyond 5% at p<0.01 after correction; 0 untestable; dispersion 1.67x.
    - **structural:fill_rate:country|device_model** *(detector, +2301ms, 0ms)*
      - what: Fitted an additive row-and-column model to Fill rate across the 16x8 country by device_model grid, using median polish on log values, and measured what each cell had left over.
      - why: An anomaly confined to one intersection barely moves either dimension on its own, and a pair moving in opposite directions leaves every total unchanged. Neither is visible to a scan that compares against the past.
      - result: 11 cell(s) beyond 5.0 standard errors, with the table scattering 1.4x wider than pure sampling error predicts.
    - **structural:fill_rate:device_model|region** *(detector, +2304ms, 0ms)*
      - what: Fitted an additive row-and-column model to Fill rate across the 8x5 device_model by region grid, using median polish on log values, and measured what each cell had left over.
      - why: An anomaly confined to one intersection barely moves either dimension on its own, and a pair moving in opposite directions leaves every total unchanged. Neither is visible to a scan that compares against the past.
      - result: 5 cell(s) beyond 5.0 standard errors, with the table scattering 1.3x wider than pure sampling error predicts.
  - **correct** *(statistics, +2397ms, 0ms)*
    - what: Benjamini-Hochberg at alpha=0.01 over 10 tests
    - why: At an uncorrected threshold, one cell in a hundred crosses it by chance, which across this lattice means dozens of confident findings in data where nothing happened.
    - result: 3 finding(s) survived
  - **localize:fill_rate** *(localizer, +6039ms, 2ms)*
    - what: Localizing the fall in fill_rate
    - why: The detector says a metric moved; it does not say where. Localization removes each candidate in turn and asks whether the parent returns to expectation.
    - result: accused os_version=iOS 17.5
  - **narrate** *(llm, +6042ms, 2611ms)*
    - what: Phrasing the pre-computed claims as prose
    - why: The model is given claim tuples and forbidden from computing. Every number it writes is checked against the bundle afterwards, and a draft containing an unsupported figure is discarded in favour of the template.
    - result: llm, verified=True
  - **confidence** *(scoring, +6042ms, 0ms)*
    - what: Scoring the verdict across its graded components
    - why: A verdict with no confidence attached forces an operator to treat a marginal result and an overwhelming one identically.
    - result: 0.67 from 4/5 components
    - **siblings:fill_rate** *(localizer, +6042ms, 0ms)*
      - what: Compared every level of 7 dimension(s) against the median level of its own dimension, inside Fill rate's own window, and removed each in turn to see which one brings the total back to that median.
      - why: The baseline audit rejected this run's history, so every counterfactual against the past is asking about a population that no longer exists. Siblings in the same window need no history and are robust to one level having collapsed.
      - result: os_version=iOS 17.5 reads 0.477729 against a sibling median of 0.79118, and removing it accounts for 99% of the gap between the total (0.731385) and that median.

### Confidence, decomposed

```json
{
  "caveat": "1 of 5 components could not be evaluated (stability), so their weight was removed and the rest renormalised. This score rests on 85% of the evidence a complete case carries.",
  "components": [
    {
      "detail": "p = 0 is 300.0 decades of evidence, 100% of the way from the floor at p = 0.1 to full credit at p = 1e-06. The finding did not survive the false-discovery correction over its family, so this component is capped at 0.25 whatever its nominal p-value.",
      "name": "significance",
      "score": 0.25,
      "state": "scored",
      "weight": 0.2941176470588235
    },
    {
      "detail": "Parent moved -0.059793. With os_version=iOS 17.5 removed it moves +0.000488359, so 99% of the change is accounted for by this segment.",
      "name": "sufficiency",
      "score": 0.9918325060982671,
      "state": "scored",
      "weight": 0.3529411764705882
    },
    {
      "detail": "os_version=iOS 17.5 sits -0.313451 from the sibling norm of 0.79118. With its worst child (os_version=iOS 17.5 AND region=APAC) removed, 98% of that gap is still there, so the child is not the whole story.",
      "name": "minimality",
      "score": 0.9846599964703091,
      "state": "scored",
      "weight": 0.1764705882352941
    },
    {
      "detail": "The window was never split, either because holdout testing is switched off or because this candidate was not the one carried forward to it.",
      "name": "stability",
      "score": 0.0,
      "state": "unknown",
      "weight": 0.0
    },
    {
      "detail": "The closest rival, device_model=iPhone 14, reaches 0.244 against the accused segment's 0.396 on relative deviation -- a relative margin of 39% over 10 candidate(s) still in contention.",
      "name": "separation",
      "score": 0.3852540123607207,
      "state": "scored",
      "weight": 0.1764705882352941
    }
  ],
  "components_scored": 4,
  "components_total": 5,
  "publishable": true,
  "value": 0.665337
}
```
