/** Mirrors the `verdict` result schema: cases, case_candidates, case_steps,
 *  coverage_ledger, runs. Field names are the column names so a real backend can
 *  be dropped in without renaming anything in the UI. */

export type VerdictKind = 'localized' | 'unlocalized' | 'undecomposed' | 'no_data';
export type Direction = 'rise' | 'fall' | 'flat';
export type Detector = 'temporal' | 'structural';
export type Grain = '5m' | '1h' | '1d';
export type LocalizationMode = 'explain_away' | 'structural_only' | 'no_data';
export type NarrativeSource = 'llm' | 'template';
export type RunStatus = 'running' | 'complete' | 'partial';
export type CheckState = 'pass' | 'fail' | 'unknown';

export type CandidateStatus =
  | 'accused'
  | 'cleared'
  | 'partial'
  | 'considered'
  | 'too_broad'
  | 'too_narrow'
  | 'wrong_direction'
  | 'immaterial'
  | 'did_not_reproduce';

export type StepKind = 'step' | 'detector' | 'statistics' | 'localizer' | 'scoring' | 'llm' | 'query' | 'pipeline';

export type Metric =
  | 'requests'
  | 'fills'
  | 'fill_rate'
  | 'impressions'
  | 'render_rate'
  | 'clicks'
  | 'ctr'
  | 'revenue'
  | 'ecpm'
  | 'rpr';

/** One row of `case_candidates` — the exoneration ledger. */
export interface Candidate {
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
  status: CandidateStatus;
  reason: string;
}

/** One weighted component of `confidence_json`. */
export interface Component {
  name: 'significance' | 'sufficiency' | 'minimality' | 'stability' | 'separation';
  score: number;
  weight: number;
  scored: boolean;
  detail: string;
}

/** One row of `coverage_ledger` — what the run could not resolve, published. */
export interface CoverageGap {
  combo: string;
  key_a: string;
  key_b: string;
  denominator: number;
  required: number;
  reason: string;
  resolvable_effect: number;
}

/** One row of `case_steps`, which is also one OpenTelemetry span in HyperDX.
 *
 *  `what`, `why` and `result` are written by the engine as the step runs, `why` before the
 *  result is known. They are the three questions a reader asks of any stage, so they are
 *  stored rather than reconstructed. */
export interface Step {
  step_id: string;
  parent_id: string;
  span_id: string;
  ordinal: number;
  name: string;
  kind: StepKind;
  what: string;
  why: string;
  result: string;
  sql?: string;
  duration_ms: number;
  /** Milliseconds from the start of the run to the start of this step. Gives the waterfall a
   *  position for each bar; without it the layout could only be inferred from sibling order,
   *  which assumes the run never waited on anything. */
  offset_ms: number;
  children?: Step[];
  /** Nesting level, filled in when the tree is flattened for the timeline. */
  depth?: number;
}

/** One surviving remediation, after independent review. Mirrors the service's schema. */
export interface Recommendation {
  title: string;
  action: string;
  rationale: string;
  expected_benefit: string;
  validation_step: string;
  risk: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  confidence: 'high' | 'medium' | 'low';
  evidence: string[];
}

/** The advice for one case, with the provenance that makes it auditable: which models wrote
 *  it, how many candidates the first pass drafted, and how many survived the second. */
export interface RecommendationSet {
  case_id: string;
  generated_at: string;
  status: 'completed' | 'failed';
  summary: string;
  /** First-pass count. `drafted - recommendations.length` is what review removed. */
  drafted: number;
  recommendations: Recommendation[];
  generation_model: string;
  validation_model: string;
  job_id: string;
  error: string;
}

/** One row of `cases`, plus the joined evidence a reviewer needs on the page. */
export interface Case {
  case_id: string;
  run_id: string;
  detected_at: string;
  metric: Metric;
  grain: Grain;
  window_start: string;
  window_end: string;
  direction: Direction;
  observed: number;
  expected: number;
  relative_effect: number;
  p_value: number;
  dispersion: number;
  verdict_kind: VerdictKind;
  segment: string;
  segment_json: Record<string, string>;
  confidence: number;
  confidence_json: Component[];
  /** The engine's publication decision. A score above the numeric threshold is not enough
   * when significance or too many confidence components could not be measured. */
  publishable: boolean;
  confidence_caveat: string;
  gates_json: Record<'sufficiency' | 'minimality' | 'maximality' | 'holdout', CheckState>;
  /** `revenue` is null whenever the metric is a count rather than money and no conversion to
   *  revenue was defensible. That is not the same as zero, and the UI must not round it to
   *  zero: an unquantified impact is unknown, and a case ranked as harmless because nobody
   *  priced it is the exact failure this system exists to avoid. `units` always carries the
   *  measured move, in whatever `unit` the metric counts. */
  impact_json: { units: number; unit: string; revenue: number | null; direct: boolean; basis: string[] };
  narrative: string;
  narrative_source: NarrativeSource;
  /** `narrative_rejected`: figures the model wrote that were not in the evidence bundle.
   *  Non-empty means the draft was discarded and the template published instead. */
  unsupported: string[];
  narrative_verified: boolean;
  fingerprint: string;
  trace_id: string;
  recurrence_of: string;
  detector: Detector;
  mode: LocalizationMode;
  candidates: Candidate[];
  /** The complete count for this metric/grain/window. `coverage` below contains only the
   * highest-volume rows retained for drill-down. */
  coverage_total: number;
  coverage: CoverageGap[];
  cells_tested: number;
  llm_model: string;
  /** Root of the stored span tree. Absent when the case predates step persistence. */
  trace: Step | null;
}

export interface Run {
  run_id: string;
  started_at: string;
  finished_at: string;
  status: RunStatus;
  cases_found: number;
  git_sha: string;
  trace_id: string;
  note: string;
  duration_ms: number;
}

/** A failed self-check. Derived from the run and from Verdict's own accuracy — never
 *  from a case, because anything visible on a case row is not news. */
export interface Health {
  level: 'd' | 'w';
  what: string;
  detail: string;
  where: string;
}

export interface Point {
  t: string;
  observed: number;
  expected: number;
  lo: number;
  hi: number;
  baseline_weeks_seen: number;
  baseline_weeks_used: number;
}
