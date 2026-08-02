import 'server-only';

import { iso, rows } from './clickhouse';
import type {
  Candidate,
  Case,
  CheckState,
  Component,
  CoverageGap,
  Detector,
  Direction,
  Grain,
  LocalizationMode,
  Metric,
  NarrativeSource,
  Point,
  Run,
  Step,
  StepKind,
  VerdictKind,
} from './types';

/** Coverage gaps are per (run, metric, grain, window), and a wide sweep can leave thousands.
 *  The table is evidence of what could not be tested, not a work queue, so the biggest
 *  denominators -- the cells that came closest to being testable -- are the informative end. */
const COVERAGE_PER_CASE = 100;

const num = (v: unknown, fallback = 0) => {
  const n = typeof v === 'number' ? v : Number.parseFloat(String(v ?? ''));
  return Number.isFinite(n) ? n : fallback;
};

function parse<T>(raw: unknown, fallback: T): T {
  if (typeof raw !== 'string' || !raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

// ---------------------------------------------------------------- runs

interface RunRow {
  run_id: string;
  started_at: string;
  finished_at: string;
  status: string;
  cases_found: number;
  git_sha: string;
  trace_id: string;
  note: string;
  duration_ms: number;
}

export async function getRuns(limit = 20): Promise<Run[]> {
  const raw = await rows<RunRow>(
    `SELECT run_id, started_at, finished_at, status, cases_found, git_sha, trace_id, note,
            duration_ms
     FROM runs ORDER BY started_at DESC LIMIT {limit:UInt32}`,
    { limit },
  );
  return raw.map(r => {
    const started = iso(r.started_at);
    const finished = iso(r.finished_at);
    return {
      run_id: r.run_id,
      started_at: started,
      finished_at: finished,
      status: (r.status || 'complete') as Run['status'],
      cases_found: num(r.cases_found),
      git_sha: r.git_sha ?? '',
      trace_id: r.trace_id ?? '',
      note: r.note ?? '',
      // The measured elapsed time when the run recorded one. Runs written before that column
      // existed fall back to the timestamp difference, which is whole seconds and so reads 2s
      // for anything between 1.5 and 2.5.
      duration_ms:
        num(r.duration_ms) || Math.max(0, Date.parse(finished) - Date.parse(started)) || 0,
    };
  });
}

// ---------------------------------------------------------------- steps

interface StepRow {
  case_id: string;
  step_id: string;
  parent_id: string;
  ordinal: number;
  name: string;
  kind: string;
  what: string;
  why: string;
  result: string;
  sql: string;
  duration_ms: number;
  offset_ms: number;
  span_id: string;
}

const KINDS: StepKind[] = ['step', 'detector', 'statistics', 'localizer', 'scoring', 'llm', 'query', 'pipeline'];
const asKind = (k: string): StepKind => (KINDS.includes(k as StepKind) ? (k as StepKind) : 'step');

/** Rebuilds the span tree from `parent_id`. Steps whose parent is missing from the row set
 *  are attached at the top rather than dropped, so a partially stored trace still renders
 *  everything it has instead of silently losing a subtree. */
function buildTree(steps: StepRow[]): Step | null {
  if (!steps.length) return null;

  const nodes = new Map<string, Step>();
  for (const s of steps) {
    nodes.set(s.step_id, {
      step_id: s.step_id,
      parent_id: s.parent_id,
      span_id: s.span_id,
      ordinal: num(s.ordinal),
      name: s.name,
      kind: asKind(s.kind),
      what: s.what ?? '',
      why: s.why ?? '',
      result: s.result ?? '',
      sql: s.sql || undefined,
      duration_ms: num(s.duration_ms),
      offset_ms: num(s.offset_ms),
    });
  }

  const roots: Step[] = [];
  for (const node of nodes.values()) {
    const parent = node.parent_id ? nodes.get(node.parent_id) : undefined;
    if (parent) (parent.children ??= []).push(node);
    else roots.push(node);
  }
  for (const node of nodes.values()) node.children?.sort((a, b) => a.ordinal - b.ordinal);
  roots.sort((a, b) => a.ordinal - b.ordinal);

  if (roots.length === 1) return roots[0];

  // More than one root means the run predates the single enclosing span. Present them under
  // a synthetic parent rather than showing only the first fragment.
  return {
    step_id: 'synthetic-root',
    parent_id: '',
    span_id: '',
    ordinal: 0,
    name: 'investigation',
    kind: 'pipeline',
    what: `${roots.length} top-level stages`,
    why: 'This run stored its stages as separate roots, so they are grouped here for reading.',
    result: '',
    // The fragments share no clock, so the synthetic parent spans from the earliest start to
    // the latest finish rather than summing durations, which would double-count any overlap.
    duration_ms: Math.max(...roots.map(r => r.offset_ms + r.duration_ms)) - Math.min(...roots.map(r => r.offset_ms)),
    offset_ms: Math.min(...roots.map(r => r.offset_ms)),
    children: roots,
  };
}

async function getTraces(caseIds: string[]): Promise<Map<string, Step>> {
  const out = new Map<string, Step>();
  if (!caseIds.length) return out;

  const raw = await rows<StepRow>(
    `SELECT case_id, step_id, parent_id, ordinal, name, kind, what, why, result, sql,
            duration_ms, offset_ms, span_id
     FROM case_steps WHERE case_id IN {ids:Array(String)} ORDER BY case_id, ordinal`,
    { ids: caseIds },
  );

  const byCase = new Map<string, StepRow[]>();
  for (const r of raw) {
    const list = byCase.get(r.case_id);
    if (list) list.push(r);
    else byCase.set(r.case_id, [r]);
  }
  for (const [id, steps] of byCase) {
    const tree = buildTree(steps);
    if (tree) out.set(id, tree);
  }
  return out;
}

// ---------------------------------------------------------------- candidates & coverage

interface CandidateRow {
  case_id: string;
  candidate: string;
  depth: number;
  observed: number;
  expected: number;
  predicted: number;
  residual: number;
  sufficiency: number;
  minimality: number;
  maximality: number;
  holdout: number;
  status: string;
  reason: string;
}

async function getCandidates(caseIds: string[]): Promise<Map<string, Candidate[]>> {
  const out = new Map<string, Candidate[]>();
  if (!caseIds.length) return out;

  const raw = await rows<CandidateRow>(
    `SELECT case_id, candidate, depth, observed, expected, predicted, residual,
            sufficiency, minimality, maximality, holdout, status, reason
     FROM case_candidates WHERE case_id IN {ids:Array(String)}
     -- Accused first, then by how much of the parent each would explain: the reading order
     -- of the exoneration ledger is "here is the answer, here is what it beat".
     ORDER BY case_id, status = 'accused' DESC, sufficiency DESC`,
    { ids: caseIds },
  );

  for (const r of raw) {
    const c: Candidate = {
      candidate: r.candidate,
      depth: num(r.depth),
      observed: num(r.observed),
      expected: num(r.expected),
      predicted: num(r.predicted),
      residual: num(r.residual),
      sufficiency: num(r.sufficiency),
      minimality: num(r.minimality),
      maximality: num(r.maximality),
      holdout: num(r.holdout),
      status: (r.status || 'considered') as Candidate['status'],
      reason: r.reason ?? '',
    };
    const list = out.get(r.case_id);
    if (list) list.push(c);
    else out.set(r.case_id, [c]);
  }
  return out;
}

interface CoverageRow {
  metric: string;
  grain: string;
  window_start: string;
  combo: string;
  key_a: string;
  key_b: string;
  denominator: number;
  required: number;
  reason: string;
  resolvable_effect: number;
}

/** Keyed by metric and window rather than by case, because that is how the ledger is written:
 *  a cell that could not be tested belongs to the sweep, not to whichever finding survived it. */
async function getCoverage(runId: string): Promise<Map<string, CoverageGap[]>> {
  const out = new Map<string, CoverageGap[]>();
  const raw = await rows<CoverageRow>(
    `SELECT metric, grain, window_start, combo, key_a, key_b, denominator, required, reason,
            resolvable_effect
     FROM coverage_ledger WHERE run_id = {run:String}
     ORDER BY metric, denominator DESC`,
    { run: runId },
  );

  for (const r of raw) {
    const key = `${r.metric}|${r.grain}|${iso(r.window_start)}`;
    const list = out.get(key) ?? [];
    if (list.length >= COVERAGE_PER_CASE) continue;
    list.push({
      combo: r.combo,
      key_a: r.key_a ?? '',
      key_b: r.key_b ?? '',
      denominator: num(r.denominator),
      required: num(r.required),
      reason: r.reason ?? '',
      resolvable_effect: num(r.resolvable_effect, -1),
    });
    out.set(key, list);
  }
  return out;
}

// ---------------------------------------------------------------- cases

interface CaseRow {
  case_id: string;
  run_id: string;
  detected_at: string;
  metric: string;
  grain: string;
  window_start: string;
  window_end: string;
  direction: string;
  observed: number;
  expected: number;
  relative_effect: number;
  p_value: number;
  dispersion: number;
  verdict_kind: string;
  segment: string;
  segment_json: string;
  confidence: number;
  confidence_json: string;
  gates_json: string;
  impact_json: string;
  narrative: string;
  narrative_source: string;
  narrative_model: string;
  narrative_verified: number;
  narrative_rejected: string[];
  fingerprint: string;
  trace_id: string;
  recurrence_of: string;
  detector: string;
  mode: string;
  cells_tested: number;
}

const GATES = ['sufficiency', 'minimality', 'maximality', 'holdout'] as const;

/** The engine writes `{checks: {name: {state, score, detail}}}`; the table wants a flat state
 *  per gate. A gate the engine never wrote is `unknown`, which is a real and different answer
 *  from `fail` -- it means the check could not run, not that it ran and the candidate lost. */
function readGates(raw: string): Record<(typeof GATES)[number], CheckState> {
  const parsed = parse<{ checks?: Record<string, { state?: string }> }>(raw, {});
  const checks = parsed.checks ?? {};
  const out = {} as Record<(typeof GATES)[number], CheckState>;
  for (const g of GATES) {
    const state = checks[g]?.state;
    out[g] = state === 'pass' || state === 'fail' ? state : 'unknown';
  }
  return out;
}

/** `state: 'scored'` is the engine's way of saying the component was actually measured.
 *  Anything else -- withheld, unknown -- must not contribute to the weighted sum, which is
 *  why it is carried as a boolean rather than collapsed into a zero score. */
function readComponents(raw: string): Component[] {
  const parsed = parse<{ components?: { name?: string; score?: number; weight?: number; state?: string; detail?: string }[] }>(
    raw,
    {},
  );
  return (parsed.components ?? []).map(c => ({
    name: (c.name ?? 'significance') as Component['name'],
    score: num(c.score),
    weight: num(c.weight),
    scored: c.state === 'scored',
    detail: c.detail ?? '',
  }));
}

function readImpact(raw: string): Case['impact_json'] {
  const p = parse<{
    units?: number;
    unit?: string;
    revenue?: number | null;
    direct?: boolean;
    basis?: string[] | string;
  }>(raw, {});
  return {
    units: num(p.units),
    unit: p.unit ?? '',
    // Deliberately not `num()`, which returns 0 for null. The engine writes null whenever a
    // count metric could not be converted to revenue, and flattening that to zero told the
    // console every such case was worth nothing -- which put the entire board in the lowest
    // priority bucket and reported a revenue-at-risk total that quietly excluded them.
    revenue: typeof p.revenue === 'number' && Number.isFinite(p.revenue) ? p.revenue : null,
    direct: Boolean(p.direct),
    basis: Array.isArray(p.basis) ? p.basis : p.basis ? [p.basis] : [],
  };
}

export async function getCases(runId: string): Promise<Case[]> {
  const raw = await rows<CaseRow>(
    `SELECT case_id, run_id, detected_at, metric, grain, window_start, window_end, direction,
            observed, expected, relative_effect, p_value, dispersion, verdict_kind, segment,
            segment_json, confidence, confidence_json, gates_json, impact_json, narrative,
            narrative_source, narrative_model, narrative_verified, narrative_rejected,
            fingerprint, trace_id, recurrence_of, detector, mode, cells_tested
     FROM cases WHERE run_id = {run:String} ORDER BY confidence DESC, p_value ASC`,
    { run: runId },
  );
  if (!raw.length) return [];

  const ids = raw.map(r => r.case_id);
  const [candidates, coverage, traces] = await Promise.all([
    getCandidates(ids),
    getCoverage(runId),
    getTraces(ids),
  ]);

  return raw.map(r => {
    const windowStart = iso(r.window_start);
    const grain = (r.grain || '1h') as Grain;
    return {
      case_id: r.case_id,
      run_id: r.run_id,
      detected_at: iso(r.detected_at),
      metric: r.metric as Metric,
      grain,
      window_start: windowStart,
      window_end: iso(r.window_end),
      direction: (r.direction || 'flat') as Direction,
      observed: num(r.observed),
      expected: num(r.expected),
      relative_effect: num(r.relative_effect),
      p_value: num(r.p_value),
      dispersion: num(r.dispersion),
      verdict_kind: (r.verdict_kind || 'localized') as VerdictKind,
      segment: r.segment || 'all traffic',
      segment_json: parse<Record<string, string>>(r.segment_json, {}),
      confidence: num(r.confidence),
      confidence_json: readComponents(r.confidence_json),
      gates_json: readGates(r.gates_json),
      impact_json: readImpact(r.impact_json),
      narrative: r.narrative ?? '',
      narrative_source: (r.narrative_source || 'template') as NarrativeSource,
      unsupported: r.narrative_rejected ?? [],
      narrative_verified: num(r.narrative_verified) === 1,
      fingerprint: r.fingerprint ?? '',
      trace_id: r.trace_id ?? '',
      recurrence_of: r.recurrence_of ?? '',
      detector: (r.detector || 'temporal') as Detector,
      mode: (r.mode || 'explain_away') as LocalizationMode,
      candidates: candidates.get(r.case_id) ?? [],
      coverage: coverage.get(`${r.metric}|${grain}|${windowStart}`) ?? [],
      cells_tested: num(r.cells_tested),
      llm_model: r.narrative_model ?? '',
      trace: traces.get(r.case_id) ?? null,
    };
  });
}

// ---------------------------------------------------------------- series

/** Numerator and denominator over the stored counters, matching `config/metrics.yaml`. A
 *  ratio is always sum/sum over the bucket, never a mean of per-row ratios, because the
 *  latter does not survive aggregation and would put a different number on the chart than
 *  the one the detector tested. */
const FORMULA: Record<Metric, { num: keyof Counters; den: keyof Counters | null; scale: number }> = {
  requests: { num: 'requests', den: null, scale: 1 },
  fills: { num: 'fills', den: null, scale: 1 },
  fill_rate: { num: 'fills', den: 'requests', scale: 1 },
  impressions: { num: 'impressions', den: null, scale: 1 },
  render_rate: { num: 'impressions', den: 'fills', scale: 1 },
  clicks: { num: 'clicks', den: null, scale: 1 },
  ctr: { num: 'clicks', den: 'impressions', scale: 1 },
  revenue: { num: 'revenue', den: null, scale: 1 },
  ecpm: { num: 'revenue', den: 'impressions', scale: 1000 },
  rpr: { num: 'revenue', den: 'requests', scale: 1 },
};

export const CHART_METRICS: Metric[] = ['fill_rate', 'revenue', 'ecpm', 'ctr', 'requests', 'render_rate'];

export const METRIC_LABEL: Record<Metric, string> = {
  requests: 'Requests / hour',
  fills: 'Fills / hour',
  fill_rate: 'Fill rate',
  impressions: 'Impressions / hour',
  render_rate: 'Render rate',
  clicks: 'Clicks / hour',
  ctr: 'CTR',
  revenue: 'Revenue / hour',
  ecpm: 'eCPM',
  rpr: 'Revenue per request',
};

interface Counters {
  requests: number;
  fills: number;
  impressions: number;
  clicks: number;
  revenue: number;
}

interface BucketRow extends Counters {
  ts: string;
}

const valueOf = (m: Metric, c: Counters | undefined): number | null => {
  if (!c) return null;
  const f = FORMULA[m];
  const numerator = num(c[f.num]);
  if (!f.den) return numerator;
  const denominator = num(c[f.den]);
  return denominator > 0 ? (numerator / denominator) * f.scale : null;
};

const median = (xs: number[]) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

/** Baseline weeks to look back over, matching the detector. */
const WEEKS = 6;
const HOUR_MS = 3_600_000;
const WEEK_MS = 7 * 24 * HOUR_MS;

export interface Series {
  metric: Metric;
  label: string;
  points: Point[];
  /** Index range where observed left the expected band, or -1 when it never did. Derived
   *  from the data rather than from a case, so the highlight cannot claim an incident the
   *  series does not show. */
  from: number;
  to: number;
  effect: number;
  weeks: number;
}

export async function getSeries(metric: Metric, startIso: string, endIso: string): Promise<Series> {
  const start = Date.parse(startIso);
  const end = Date.parse(endIso);
  const empty: Series = { metric, label: METRIC_LABEL[metric], points: [], from: -1, to: -1, effect: 0, weeks: 0 };
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return empty;

  // One scan covers the window and every baseline week behind it. Splitting them would run
  // seven queries to answer one question.
  const from = new Date(start - WEEKS * WEEK_MS).toISOString().slice(0, 19).replace('T', ' ');
  const to = new Date(end).toISOString().slice(0, 19).replace('T', ' ');

  const raw = await rows<BucketRow>(
    // Aliased to `ts`, not to `bucket`: an alias that shadows the column it is derived from
    // makes the WHERE clause compare a String against a DateTime and the whole scan fails.
    `SELECT toString(bucket) AS ts,
            sum(requests) AS requests, sum(fills) AS fills, sum(impressions) AS impressions,
            sum(clicks) AS clicks, sum(revenue) AS revenue
     FROM rollup_1h
     WHERE combo = '__all__'
       AND bucket >= parseDateTimeBestEffort({from:String})
       AND bucket <  parseDateTimeBestEffort({to:String})
     GROUP BY bucket ORDER BY bucket`,
    { from, to },
  );
  if (!raw.length) return empty;

  const byTime = new Map<number, Counters>();
  for (const r of raw) byTime.set(Date.parse(iso(r.ts)), r);

  const points: Point[] = [];
  let weeksSeen = 0;

  for (let t = start; t < end; t += HOUR_MS) {
    const observed = valueOf(metric, byTime.get(t));
    if (observed === null) continue;

    // Same weekday, same hour, prior weeks: the comparison that keeps a Saturday from being
    // reported as an incident every Saturday.
    const history: number[] = [];
    for (let w = 1; w <= WEEKS; w++) {
      const v = valueOf(metric, byTime.get(t - w * WEEK_MS));
      if (v !== null) history.push(v);
    }
    weeksSeen = Math.max(weeksSeen, history.length);

    const expected = median(history) ?? observed;
    // Robust spread from the baseline itself. A fixed percentage band looks the same on a
    // metric that never moves and one that swings 30% a day, which is exactly the
    // distinction a reader needs the band to draw.
    const mad = median(history.map(h => Math.abs(h - expected))) ?? 0;
    const sigma = mad * 1.4826;
    const pad = sigma > 0 ? 2 * sigma : Math.abs(expected) * 0.03;

    points.push({
      t: new Date(t).toISOString(),
      observed,
      expected,
      lo: expected - pad,
      hi: expected + pad,
    });
  }

  const outside = points.map((p, i) => (p.observed < p.lo || p.observed > p.hi ? i : -1)).filter(i => i >= 0);
  const fromIdx = outside.length ? outside[0] : -1;
  const toIdx = outside.length ? outside[outside.length - 1] : -1;

  let effect = 0;
  if (fromIdx >= 0) {
    const span = points.slice(fromIdx, toIdx + 1);
    const obs = span.reduce((s, p) => s + p.observed, 0);
    const exp = span.reduce((s, p) => s + p.expected, 0);
    effect = exp !== 0 ? (obs - exp) / exp : 0;
  }

  return { metric, label: METRIC_LABEL[metric], points, from: fromIdx, to: toIdx, effect, weeks: weeksSeen };
}

// ---------------------------------------------------------------- dashboard

export interface Dashboard {
  run: Run | null;
  runs: Run[];
  cases: Case[];
  series: Series[];
  spans: number;
  /** True when the database answered but had nothing in it, as distinct from a failed read. */
  empty: boolean;
}

/** One call for the whole page. The console shows a single run at a time, because mixing
 *  two sweeps into one case list would put findings from different windows and different
 *  code versions in the same table with nothing to tell them apart. */
export async function getDashboard(runId?: string): Promise<Dashboard> {
  const runList = await getRuns();
  // Prefer a run that actually produced cases: the most recent sweep is frequently a
  // single-metric re-run, and landing the console on an empty one looks like a broken
  // deployment rather than a quiet night.
  const run = runId
    ? (runList.find(r => r.run_id === runId) ?? null)
    : (runList.find(r => r.cases_found > 0) ?? runList[0] ?? null);

  if (!run) return { run: null, runs: runList, cases: [], series: [], spans: 0, empty: true };

  const cases = await getCases(run.run_id);
  const window = cases[0];

  const [series, spanCount] = await Promise.all([
    window
      ? Promise.all(CHART_METRICS.map(m => getSeries(m, window.window_start, window.window_end)))
      : Promise.resolve([] as Series[]),
    rows<{ n: string }>(`SELECT count() AS n FROM default.otel_traces WHERE ServiceName = 'verdict'`),
  ]);

  return {
    run,
    runs: runList,
    cases,
    series: series.filter(s => s.points.length > 0),
    spans: num(spanCount[0]?.n),
    empty: cases.length === 0,
  };
}
