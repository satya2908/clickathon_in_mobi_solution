import type { Candidate, Direction, Metric, StepKind, VerdictKind } from './types';

/** Proportions and ratios read as percentages; counts and money do not. */
const RATE: Metric[] = ['fill_rate', 'render_rate', 'ctr'];

export const isRate = (m: Metric) => RATE.includes(m);

export const pct = (x: number, digits = 1) =>
  `${x > 0 ? '+' : x < 0 ? '\u2212' : ''}${Math.abs(x * 100).toFixed(digits)}%`;

export const rate = (x: number) => `${(x * 100).toFixed(2)}%`;

export const money = (x: number) => {
  const a = Math.abs(x);
  const s = a >= 1_000_000 ? `${(a / 1_000_000).toFixed(2)}M` : a >= 1000 ? `${(a / 1000).toFixed(1)}k` : a.toFixed(2);
  return `${x < 0 ? '\u2212' : ''}$${s}`;
};

export const count = (n: number) => {
  const a = Math.abs(n);
  if (a >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (a >= 1000) return `${(n / 1000).toFixed(1)}k`;
  // Expected values are model output and arrive fractional, so an unrounded template
  // printed a forecast of 173 clicks as 173.00008342316 and overran the column. Counts of
  // things are whole; only the small ones keep a decimal, where it carries information.
  return a >= 10 ? `${Math.round(n)}` : `${Number(n.toFixed(1))}`;
};

export const metricValue = (m: Metric, v: number) => {
  if (isRate(m)) return rate(v);
  if (m === 'revenue') return money(v);
  if (m === 'ecpm') return `$${v.toFixed(2)}`;
  if (m === 'rpr') return `$${v.toFixed(4)}`;
  return count(v);
};

/** Tick values a person would have chosen: multiples of 1, 2 or 5 inside the data
 *  range. Slicing the range into four equal parts produced axes labelled 57.42% and
 *  66.50%, which are arithmetic rather than reference points. */
export function ticks(min: number, max: number, target = 4): { values: number[]; step: number } {
  const raw = (max - min) / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const values: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) values.push(v);
  return { values, step };
}

/** Axis labels carry only the digits the step can distinguish, so a 10-point step
 *  reads 60% rather than 60.00%. */
export const axisValue = (m: Metric, v: number, step: number) => {
  if (isRate(m)) return `${(v * 100).toFixed(Math.max(0, -Math.floor(Math.log10(step * 100) + 1e-9)))}%`;
  return metricValue(m, v);
};

/** p is compared against a fixed 0.01 threshold, so the exponent is what matters.
 *  ASCII minus, deliberately: this value gets pasted into queries and notebooks, and
 *  U+2212 inside an exponent does not parse. */
export const pval = (p: number) => (p === 0 ? '0' : p < 0.001 ? p.toExponential(1) : p.toFixed(4));

export const ms = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(2)}s` : `${Math.round(n)}ms`);

export const clock = (iso: string) => iso.slice(11, 16);
export const day = (iso: string) => iso.slice(5, 10);
export const stamp = (iso: string) => `${iso.slice(5, 10)} ${iso.slice(11, 16)}`;

export const ARROW: Record<Direction, string> = { fall: '\u2193', rise: '\u2191', flat: '\u2192' };

export const KIND_LABEL: Record<VerdictKind, string> = {
  localized: 'localized',
  unlocalized: 'unlocalized',
  undecomposed: 'undecomposed',
  no_data: 'no data',
};

/** `localized` is the ordinary outcome, so it gets no colour — the two that need a
 *  second look do. Indigo is reserved for interactive and selected state. */
export const KIND_BADGE: Record<VerdictKind, string> = {
  localized: 'badge',
  unlocalized: 'badge w',
  undecomposed: 'badge w',
  no_data: 'badge q',
};

/** Bar segments under OPEN CASES, in the same order as the filter chips. */
export const KIND_FILL: Record<VerdictKind, string> = {
  localized: 'var(--tx2)',
  unlocalized: 'var(--warn)',
  undecomposed: 'var(--warn)',
  no_data: 'var(--line2)',
};

/** Three colours, not seven. The localizer is the argument, the queries are the
 *  footnotes, everything else is scaffolding — and all three are tokens, so they
 *  survive the dark theme. Raw hex here was invisible on a dark background. */
export const STEP_COLOR: Record<StepKind, string> = {
  localizer: 'var(--acc)',
  query: 'var(--tx3)',
  step: 'var(--line2)',
  pipeline: 'var(--line2)',
  detector: 'var(--line2)',
  statistics: 'var(--line2)',
  scoring: 'var(--line2)',
  llm: 'var(--line2)',
};

/** Genuinely exonerated. `too_broad`, `too_narrow` and `partial` are *parts of the
 *  answer*, and counting them as ruled out overstates the one number this product
 *  cannot afford to overstate. */
const EXONERATED: Candidate['status'][] = ['cleared', 'immaterial', 'wrong_direction', 'did_not_reproduce'];

export const clearedOf = (candidates: Candidate[]) =>
  `${candidates.filter(c => EXONERATED.includes(c.status)).length} / ${candidates.length}`;

/** Priority is not a stored column: it is impact ranked by confidence, so a big number nobody
 *  can stand behind does not outrank a proven small one.
 *
 *  Revenue is null for count metrics that could not be priced. Treating that as zero -- which
 *  is what `Math.abs(null)` quietly does -- ranked every unpriced case as harmless and put
 *  the whole board in the bottom bucket. Unknown is not zero, so those rank on confidence
 *  alone, and are capped below the top bucket: a finding nobody has sized should not outrank
 *  one that has been measured in money. */
export const priority = (revenue: number | null, confidence: number): 0 | 1 | 2 | 3 => {
  if (revenue == null || !Number.isFinite(revenue)) {
    if (confidence >= 0.75) return 1;
    if (confidence >= PUBLISHABLE) return 2;
    return 3;
  }
  const weighted = Math.abs(revenue) * confidence;
  if (weighted >= 12_000) return 0;
  if (weighted >= 3_000) return 1;
  if (weighted >= 700) return 2;
  return 3;
};

/** Duplicated from lib/data rather than imported, to keep this module free of that cycle. */
const PUBLISHABLE = 0.5;

/** Impact, in whatever terms it was actually established. Money when the case was priced;
 *  otherwise the raw move with its unit, so the row says "20.4k fills" rather than "$0.00". */
export const impact = (i: { units: number; unit: string; revenue: number | null }): string =>
  i.revenue != null && Number.isFinite(i.revenue) ? money(i.revenue) : `${count(Math.round(i.units))} ${i.unit}`;
