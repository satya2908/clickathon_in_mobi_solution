"""Schema generation.

DDL is generated from the metric registry rather than hand-written, so the rollup lattice can
never drift out of step with the declared dimensions. Adding a dimension to metrics.yaml
extends the lattice; hand-written DDL would have silently left a hole that looks like a
dimension the analyst simply never found anything in.

Storage shape
-------------
One rollup table per time grain, holding a 1-way and 2-way *lattice* in long form:

    (bucket, combo, key_a, key_b, requests, fills, impressions, clicks, revenue)

``combo`` names which dimensions a row is keyed by ('region', 'region|os_version', or
'__all__' for the grand total). This was chosen over a full-grain rollup after measuring both
on the real dataset: full-grain gives only 1.14x compression at hourly grain here because
dimension cardinality is high relative to event volume, while the lattice gives 6.4x hourly
and 150x daily. The lattice cannot answer 3-way questions, which is a known and stated limit.

Only additive counters are stored. Every metric is divided out at read time, so a rollup row
means the same thing however it is later aggregated.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .config import Config
from .metrics import MetricRegistry

TOTAL_COMBO = "__all__"

# How each lattice dimension is resolved from a raw ad_events row. Dictionaries keep the fact
# table narrow (three id columns instead of nine denormalized strings) while making the join
# a hash lookup rather than a join.
_DIM_SOURCE_SQL = {
    "ad_format": "ad_format",
    "category": "dictGet('dict_apps', 'category', app_id)",
    "publisher_tier": "dictGet('dict_apps', 'publisher_tier', app_id)",
    "vertical": "dictGet('dict_advertisers', 'vertical', advertiser_id)",
    "campaign_type": "dictGet('dict_advertisers', 'campaign_type', advertiser_id)",
    "region": "dictGet('dict_geo_device', 'region', geo_device_id)",
    "country": "dictGet('dict_geo_device', 'country', geo_device_id)",
    "device_model": "dictGet('dict_geo_device', 'device_model', geo_device_id)",
    "os_version": "dictGet('dict_geo_device', 'os_version', geo_device_id)",
}

GRAINS = {
    "5m": ("rollup_5m", "toStartOfFiveMinutes"),
    "1h": ("rollup_1h", "toStartOfHour"),
    "1d": ("rollup_1d", "toStartOfDay"),
}

# How deep the lattice goes at each grain. This is a statistical-power decision, not a storage
# one, though it happens to save an order of magnitude of storage as well.
#
# Detecting a 10-point drop in a 0.785 fill rate at alpha=0.01 with 80% power needs about 454
# requests in the window and again in the baseline. Measured against this dataset -- 9M events
# over 35 days, dimension cardinalities 16/8/8/7/7/5/5/3/3 giving 1 + 62 + 1647 = 1710 cells --
# the traffic available per cell is:
#
#     grain   buckets   global    1-way (country)   2-way (country|vertical)
#     5m       10,080      893                 56                        0.5
#     1h          840   10,714                670                         96
#     1d           35  257,143             16,071                      2,300
#
# So a 2-way cell at 5-minute grain carries about half a request. Materializing those 1647
# combos costs 1647/1710 = 96% of the rows in the table to produce cells that every detector
# must reject as underpowered, and it is measured: 'country|vertical' at 5m has a median of 1
# request per cell and 100% of cells below 30. The 5-minute tier therefore keeps only the grand
# total and the one-way cells, which is what fast top-line alerting actually reads; two-way
# localization happens at hourly and daily grain, where the cells can support a test.
LATTICE_DEPTH = {"5m": 1, "1h": 2, "1d": 2}

# Where each grain's rows come from. "events" means a materialized view reads ad_events
# directly; anything else names a rollup table that a view reads to build this one.
#
# Hourly reads events rather than chaining off rollup_5m because the 5-minute tier is shallow
# and a coarser grain cannot recover two-way cells its source never stored. Daily chains off
# hourly, since both are full depth.
#
# This mapping is also what makes backfill safe, and that is not a detail. A materialized view
# fires on every insert into its source, including inserts made by the backfill itself. So
# backfilling rollup_1h automatically fills rollup_1d through mv_1h_to_1d, and a backfill
# statement for rollup_1d would insert the same rows a second time. That bug is invisible in
# any per-table check -- both tables look populated and internally consistent -- and shows up
# only as a daily grain carrying exactly twice the traffic of the facts.
ROLLUP_SOURCE = {"5m": "events", "1h": "events", "1d": "rollup_1h"}


@dataclass(frozen=True)
class Statement:
    name: str
    sql: str


def combos(dims: list[str], max_depth: int = 2) -> list[tuple[str, str | None]]:
    """The lattice: the grand total, every dimension alone, and every unordered pair.

    ``max_depth`` of 1 stops after the single dimensions. See ``LATTICE_DEPTH`` for why a grain
    would want that.
    """
    out: list[tuple[str, str | None]] = [(TOTAL_COMBO, None)]
    if max_depth >= 1:
        out.extend((d, None) for d in dims)
    if max_depth >= 2:
        out.extend((a, b) for a, b in combinations(dims, 2))
    return out


def combo_name(a: str, b: str | None) -> str:
    return a if b is None else f"{a}|{b}"


def _combo_array(dims: list[str], max_depth: int = 2) -> str:
    """The array of (combo, key_a, key_b) tuples fanned out per event.

    One ARRAY JOIN replaces what would otherwise be one materialized view per combo, which
    matters for more than tidiness: with 1710 views the streaming path and the backfill path
    are 1710 chances to define a bucket boundary slightly differently, and any disagreement
    shows up in the data as a step change that looks exactly like a real incident.
    """
    rows = []
    for a, b in combos(dims, max_depth):
        if a == TOTAL_COMBO:
            rows.append("    ('__all__', '', '')")
        elif b is None:
            rows.append(f"    ('{a}', CAST({_DIM_SOURCE_SQL[a]} AS String), '')")
        else:
            rows.append(
                f"    ('{a}|{b}', CAST({_DIM_SOURCE_SQL[a]} AS String), "
                f"CAST({_DIM_SOURCE_SQL[b]} AS String))"
            )
    return "[\n" + ",\n".join(rows) + "\n  ]"


def rollup_select_from_events(
    dims: list[str], bucket_fn: str, where: str = "", max_depth: int = 2
) -> str:
    """The one definition of how raw events become rollup rows.

    Used verbatim by both the materialized view and the historical backfill so the two cannot
    disagree at the handover boundary.
    """
    clause = f"\nWHERE {where}" if where else ""
    return f"""SELECT
  {bucket_fn}(event_time)             AS bucket,
  c.1                                 AS combo,
  c.2                                 AS key_a,
  c.3                                 AS key_b,
  count()                             AS requests,
  sum(is_filled)                      AS fills,
  sum(is_impression)                  AS impressions,
  sum(is_click)                       AS clicks,
  sum(revenue)                        AS revenue
FROM ad_events
ARRAY JOIN {_combo_array(dims, max_depth)} AS c{clause}
GROUP BY bucket, combo, key_a, key_b"""


def rollup_select_from_rollup(source: str, bucket_fn: str, where: str = "") -> str:
    """Coarser grains re-aggregate the finer table.

    Safe on a SummingMergeTree source even though a view sees pre-merge blocks: the blocks are
    partial sums, and a sum of partial sums is the total.
    """
    clause = f"\nWHERE {where}" if where else ""
    return f"""SELECT
  {bucket_fn}(bucket)   AS bucket,
  combo,
  key_a,
  key_b,
  sum(requests)         AS requests,
  sum(fills)            AS fills,
  sum(impressions)      AS impressions,
  sum(clicks)           AS clicks,
  sum(revenue)          AS revenue
FROM {source}{clause}
GROUP BY bucket, combo, key_a, key_b"""


def _ttl(days: int, column: str, enforce: bool) -> str:
    return f"\nTTL {column} + INTERVAL {days} DAY DELETE" if enforce else ""


def dimension_ddl() -> list[Statement]:
    stmts = [
        Statement(
            "dim_apps",
            """CREATE TABLE IF NOT EXISTS dim_apps (
  app_id          String,
  category        LowCardinality(String),
  publisher_tier  LowCardinality(String)
) ENGINE = MergeTree ORDER BY app_id""",
        ),
        Statement(
            "dim_advertisers",
            """CREATE TABLE IF NOT EXISTS dim_advertisers (
  advertiser_id  String,
  vertical       LowCardinality(String),
  campaign_type  LowCardinality(String)
) ENGINE = MergeTree ORDER BY advertiser_id""",
        ),
        Statement(
            "dim_geo_device",
            """CREATE TABLE IF NOT EXISTS dim_geo_device (
  geo_device_id  String,
  region         LowCardinality(String),
  country        LowCardinality(String),
  device_model   LowCardinality(String),
  os_version     LowCardinality(String)
) ENGINE = MergeTree ORDER BY geo_device_id""",
        ),
    ]
    # LIFETIME(0) pins the dictionary: these are static reference tables for the run, and a
    # background reload mid-investigation would mean two queries in one case file silently
    # resolved the same id differently.
    stmts += [
        Statement(
            "dict_apps",
            """CREATE DICTIONARY IF NOT EXISTS dict_apps (
  app_id          String,
  category        String,
  publisher_tier  String
)
PRIMARY KEY app_id
SOURCE(CLICKHOUSE(TABLE 'dim_apps'))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0)""",
        ),
        Statement(
            "dict_advertisers",
            """CREATE DICTIONARY IF NOT EXISTS dict_advertisers (
  advertiser_id  String,
  vertical       String,
  campaign_type  String
)
PRIMARY KEY advertiser_id
SOURCE(CLICKHOUSE(TABLE 'dim_advertisers'))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0)""",
        ),
        Statement(
            "dict_geo_device",
            """CREATE DICTIONARY IF NOT EXISTS dict_geo_device (
  geo_device_id  String,
  region         String,
  country        String,
  device_model   String,
  os_version     String
)
PRIMARY KEY geo_device_id
SOURCE(CLICKHOUSE(TABLE 'dim_geo_device'))
LAYOUT(COMPLEX_KEY_HASHED())
LIFETIME(0)""",
        ),
    ]
    return stmts


def fact_ddl(cfg: Config) -> list[Statement]:
    ttl = _ttl(cfg.retention.raw_events_days, "event_time", cfg.retention.enforce)
    return [
        Statement(
            "ad_events",
            f"""CREATE TABLE IF NOT EXISTS ad_events (
  event_time     DateTime64(3, 'UTC') CODEC(Delta(8), ZSTD(1)),
  app_id         LowCardinality(String),
  geo_device_id  LowCardinality(String),
  advertiser_id  LowCardinality(String),
  ad_format      LowCardinality(String),
  is_filled      UInt8  CODEC(ZSTD(1)),
  is_impression  UInt8  CODEC(ZSTD(1)),
  is_click       UInt8  CODEC(ZSTD(1)),
  revenue        Float64 CODEC(ZSTD(1))
)
ENGINE = MergeTree
PARTITION BY toDate(event_time)
ORDER BY (event_time, app_id){ttl}""",
        )
    ]


def rollup_ddl(cfg: Config) -> list[Statement]:
    stmts: list[Statement] = []
    ttl_days = {
        "5m": cfg.retention.rollup_5m_days,
        "1h": cfg.retention.rollup_1h_days,
        "1d": cfg.retention.rollup_1d_days,
    }
    for grain, (table, _) in GRAINS.items():
        partition = "toDate(bucket)" if grain == "5m" else "toYYYYMM(bucket)"
        ttl = _ttl(ttl_days[grain], "bucket", cfg.retention.enforce)
        stmts.append(
            Statement(
                table,
                f"""CREATE TABLE IF NOT EXISTS {table} (
  bucket       DateTime('UTC') CODEC(DoubleDelta, ZSTD(1)),
  combo        LowCardinality(String),
  key_a        LowCardinality(String),
  key_b        LowCardinality(String),
  requests     UInt64  CODEC(T64, ZSTD(1)),
  fills        UInt64  CODEC(T64, ZSTD(1)),
  impressions  UInt64  CODEC(T64, ZSTD(1)),
  clicks       UInt64  CODEC(T64, ZSTD(1)),
  revenue      Float64 CODEC(ZSTD(1))
)
ENGINE = SummingMergeTree((requests, fills, impressions, clicks, revenue))
PARTITION BY {partition}
ORDER BY (combo, bucket, key_a, key_b){ttl}""",
            )
        )
    return stmts


def view_name(grain: str) -> str:
    source = ROLLUP_SOURCE[grain]
    return f"mv_events_to_{grain}" if source == "events" else f"mv_{source[7:]}_to_{grain}"


def view_ddl(dims: list[str]) -> list[Statement]:
    """Materialized views for the live path, one per grain, derived from ``ROLLUP_SOURCE``.

    These cover events arriving after load. History is backfilled explicitly with the same
    SELECT, because a view only ever fires on new inserts.
    """
    stmts = []
    for grain, (table, bucket_fn) in GRAINS.items():
        source = ROLLUP_SOURCE[grain]
        select = (
            rollup_select_from_events(dims, bucket_fn, max_depth=LATTICE_DEPTH[grain])
            if source == "events"
            else rollup_select_from_rollup(source, bucket_fn)
        )
        name = view_name(grain)
        stmts.append(
            Statement(name, f"CREATE MATERIALIZED VIEW IF NOT EXISTS {name} TO {table} AS\n{select}")
        )
    return stmts


def results_ddl(cfg: Config) -> list[Statement]:
    """Tables holding what the analyst concluded.

    Cases are kept far longer than the events that produced them: a closed case is a few
    kilobytes, and recognising that today's incident is a repeat of one from eight months ago
    is worth more than the raw rows ever were.
    """
    ttl = _ttl(cfg.retention.cases_days, "detected_at", cfg.retention.enforce)
    return [
        Statement(
            "cases",
            f"""CREATE TABLE IF NOT EXISTS cases (
  case_id         String,
  run_id          String,
  detected_at     DateTime('UTC'),
  metric          LowCardinality(String),
  grain           LowCardinality(String),
  window_start    DateTime('UTC'),
  window_end      DateTime('UTC'),
  direction       LowCardinality(String),
  observed        Float64,
  expected        Float64,
  relative_effect Float64,
  p_value         Float64,
  dispersion      Float64,
  verdict_kind    LowCardinality(String),
  segment         String,
  segment_json    String,
  confidence      Float64,
  confidence_json String,
  gates_json      String,
  impact_json     String,
  narrative       String,
  narrative_source LowCardinality(String),
  -- What the model did, kept queryable rather than only logged. The system's central claim is
  -- that a figure it cannot verify never reaches a reader, and these columns are what turn that
  -- from an assertion into something a sceptic can check: filter on narrative_verified = 0 and
  -- narrative_rejected holds the exact literals the model wrote that the evidence did not
  -- support. A guardrail whose firings are not recorded cannot be audited.
  narrative_model LowCardinality(String),
  narrative_verified UInt8,
  narrative_rejected Array(String),
  narrative_prompt_tokens UInt32,
  narrative_completion_tokens UInt32,
  narrative_latency_ms UInt32,
  fingerprint     String,
  trace_id        String,
  recurrence_of   String,
  -- How the verdict was reached, as opposed to what it says. A reader deciding how much
  -- weight to give a case needs to know whether it came from a comparison against history
  -- or against siblings, whether localization could run the removal test at all, and how
  -- wide a sweep the finding survived.
  detector        LowCardinality(String),
  mode            LowCardinality(String),
  cells_tested    UInt32
)
ENGINE = ReplacingMergeTree(detected_at)
PARTITION BY toYYYYMM(detected_at)
ORDER BY (case_id){ttl}""",
        ),
        Statement(
            "case_candidates",
            """CREATE TABLE IF NOT EXISTS case_candidates (
  case_id        String,
  candidate      String,
  candidate_json String,
  depth          UInt8,
  observed       Float64,
  expected       Float64,
  predicted      Float64,
  residual       Float64,
  sufficiency    Float64,
  minimality     Float64,
  maximality     Float64,
  holdout        Float64,
  p_value        Float64,
  status         LowCardinality(String),
  reason         String
)
ENGINE = MergeTree
ORDER BY (case_id, status, candidate)""",
        ),
        Statement(
            "case_steps",
            """CREATE TABLE IF NOT EXISTS case_steps (
  case_id     String,
  step_id     String,
  parent_id   String,
  ordinal     UInt16,
  name        String,
  kind        LowCardinality(String),
  what        String,
  why         String,
  result      String,
  sql         String,
  duration_ms UInt32,
  offset_ms   UInt32,
  span_id     String
)
ENGINE = MergeTree
ORDER BY (case_id, ordinal)""",
        ),
        Statement(
            "case_recommendations",
            # What to do about a case, as opposed to what happened -- the only table here whose
            # contents are model-written rather than computed, which is why the provenance sits
            # beside the advice: which models produced it, how many candidates the first pass
            # drafted, and how many survived independent review. A reader can see the filter
            # working, and a summary that kept two of six says more about the advice than any
            # confidence label the model could assign itself.
            #
            # Replacing, keyed on the case, because regenerating advice supersedes it rather
            # than adding to it. Generating costs roughly two minutes of model time per case,
            # so this is also the cache that stops a UI toggle from paying that twice.
            """CREATE TABLE IF NOT EXISTS case_recommendations (
  case_id          String,
  generated_at     DateTime,
  status           LowCardinality(String),
  summary          String,
  drafted          UInt16,
  kept             UInt16,
  recommendations  String,
  generation_model LowCardinality(String),
  validation_model LowCardinality(String),
  job_id           String,
  error            String
)
ENGINE = ReplacingMergeTree(generated_at)
ORDER BY case_id""",
        ),
        Statement(
            "coverage_ledger",
            """CREATE TABLE IF NOT EXISTS coverage_ledger (
  run_id       String,
  metric       LowCardinality(String),
  grain        LowCardinality(String),
  window_start DateTime('UTC'),
  combo        LowCardinality(String),
  key_a        String,
  key_b        String,
  denominator  UInt64,
  required     UInt64,
  reason       LowCardinality(String),
  resolvable_effect Float64 DEFAULT -1
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(window_start)
ORDER BY (run_id, metric, combo, key_a, key_b)""",
        ),
        Statement(
            "feedback",
            """CREATE TABLE IF NOT EXISTS feedback (
  case_id     String,
  labelled_at DateTime('UTC') DEFAULT now(),
  label       LowCardinality(String),
  note        String,
  operator    String
)
ENGINE = ReplacingMergeTree(labelled_at)
ORDER BY (case_id)""",
        ),
        Statement(
            "runs",
            """CREATE TABLE IF NOT EXISTS runs (
  run_id      String,
  started_at  DateTime('UTC'),
  finished_at DateTime('UTC'),
  config_json String,
  git_sha     String,
  trace_id    String,
  cases_found UInt32,
  status      LowCardinality(String),
  note        String,
  -- Measured, not derived. started_at and finished_at are second-resolution, which cannot
  -- express a run that takes two seconds without rounding it to two or three.
  duration_ms UInt32 DEFAULT 0
)
ENGINE = ReplacingMergeTree(started_at)
ORDER BY (run_id)""",
        ),
    ]


def migration_statements() -> list[Statement]:
    """Additive changes to tables that may already exist.

    ``CREATE TABLE IF NOT EXISTS`` is a no-op against a table created by an earlier version, so a
    column added to the DDL above never reaches a database that has already been set up. Rather
    than requiring a drop -- which on a populated instance means re-loading nine million rows to
    gain one column -- each additive change is also expressed as an idempotent ALTER.

    Keep every statement in here safe to run repeatedly and safe to run against a table that
    already has the final shape, because it runs on every apply.
    """
    return [
        Statement(
            "migrate_coverage_resolvable_effect",
            "ALTER TABLE coverage_ledger ADD COLUMN IF NOT EXISTS resolvable_effect Float64 DEFAULT -1",
        ),
        # AFTER duration_ms so a migrated table still matches its own DDL. Note that this is
        # tidiness, not safety: `ClickHouse.insert` names its columns explicitly and the driver
        # binds by name, so appending at the end would be harmless. The `cases` table proves it
        # -- nine ALTERs without AFTER have left its physical order three columns adrift of
        # CASE_COLUMNS, and every value still lands correctly.
        #
        # The real hazard is any write that bypasses that column list. A bare
        # `INSERT INTO ... VALUES`, or `insert_arrow`, which takes no column names, does bind by
        # position and would misfile every value on a table whose order has drifted.
        Statement(
            "migrate_steps_offset_ms",
            "ALTER TABLE case_steps ADD COLUMN IF NOT EXISTS offset_ms UInt32 DEFAULT 0 AFTER duration_ms",
        ),
        # Last in the DDL too, so appending it here keeps the positional insert aligned.
        Statement(
            "migrate_runs_duration_ms",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS duration_ms UInt32 DEFAULT 0",
        ),
        *(
            Statement(f"migrate_cases_{name.split()[0]}", f"ALTER TABLE cases ADD COLUMN IF NOT EXISTS {name}")
            for name in (
                "narrative_model LowCardinality(String) DEFAULT ''",
                "narrative_verified UInt8 DEFAULT 0",
                "narrative_rejected Array(String) DEFAULT []",
                "narrative_prompt_tokens UInt32 DEFAULT 0",
                "narrative_completion_tokens UInt32 DEFAULT 0",
                "narrative_latency_ms UInt32 DEFAULT 0",
                "detector LowCardinality(String) DEFAULT ''",
                "mode LowCardinality(String) DEFAULT ''",
                "cells_tested UInt32 DEFAULT 0",
            )
        ),
    ]


def all_statements(cfg: Config, registry: MetricRegistry) -> list[Statement]:
    dims = registry.lattice_dimensions
    unknown = [d for d in dims if d not in _DIM_SOURCE_SQL]
    if unknown:
        raise ValueError(
            f"Lattice dimension(s) {unknown} have no source expression in schema.py. "
            "Add them to _DIM_SOURCE_SQL, or the rollup would silently omit them."
        )
    return (
        dimension_ddl()
        + fact_ddl(cfg)
        + rollup_ddl(cfg)
        + view_ddl(dims)
        + results_ddl(cfg)
        + migration_statements()
    )


def backfill_statements(dims: list[str], where: str = "") -> list[Statement]:
    """Populate rollups from already-loaded history, mirroring the view chain exactly.

    Only the grains fed directly from ``ad_events`` get a statement. The rest are filled as a
    side effect: a materialized view fires on inserts made by this backfill just as it does on
    live traffic, so writing rollup_1h also writes rollup_1d, and issuing a statement for the
    daily grain too would insert every one of those rows a second time.

    Each statement uses the same SELECT and the same depth as the view that would have produced
    those rows live, so a table's contents do not depend on whether the events arrived before
    or after the views existed.
    """
    stmts = []
    for grain, (table, bucket_fn) in GRAINS.items():
        if ROLLUP_SOURCE[grain] != "events":
            continue
        select = rollup_select_from_events(dims, bucket_fn, where, max_depth=LATTICE_DEPTH[grain])
        stmts.append(Statement(f"backfill_{grain}", f"INSERT INTO {table}\n{select}"))
    return stmts


def cascaded_grains() -> list[str]:
    """Grains that a backfill fills indirectly, through another grain's materialized view."""
    return [g for g in GRAINS if ROLLUP_SOURCE[g] != "events"]
