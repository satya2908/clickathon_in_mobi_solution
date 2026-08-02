import type { Case, Health, Run, Step, VerdictKind } from './types';

export const P_THRESHOLD = 0.01;

export const KINDS: VerdictKind[] = ['localized', 'unlocalized', 'undecomposed', 'no_data'];

/** Depth-first, carrying nesting level. The timeline lays every span against one clock and so
 *  has no structure of its own to indent by; without the level recorded here, a flat list of
 *  forty spans loses the containment the tree makes obvious. */
export const flatten = (s: Step, out: Step[] = [], depth = 0): Step[] => {
  out.push({ ...s, depth });
  s.children?.forEach(c => flatten(c, out, depth + 1));
  return out;
};

export interface Kpi {
  cases: number;
  published: number;
  byKind: Record<VerdictKind, number>;
  meanConfidence: number;
  revenueAtRisk: number;
  /** Cases whose impact could not be converted to revenue. Reported alongside the total,
   *  because a sum that silently excludes them is a floor presented as a figure. */
  unpriced: number;
  cellsTested: number;
  coverageGaps: number;
  llmVerified: number;
  spans: number;
}

/** Case figures are folds over the rows returned. Coverage is run-level because gaps belong
 *  to the sweep and can exist for metrics that produced no case at all. */
export function kpiOf(cases: Case[], spans: number, coverageGaps: number): Kpi {
  const byKind = { localized: 0, unlocalized: 0, undecomposed: 0, no_data: 0 } as Record<VerdictKind, number>;
  for (const c of cases) byKind[c.verdict_kind] = (byKind[c.verdict_kind] ?? 0) + 1;

  return {
    cases: cases.length,
    published: cases.filter(c => c.publishable).length,
    byKind,
    meanConfidence: cases.length ? cases.reduce((s, c) => s + c.confidence, 0) / cases.length : 0,
    // Losses only. Netting a recovered segment against a broken one would report a quiet
    // hour for a night in which one thing broke and another improved.
    // `Math.min(0, null)` is 0, so an unpriced case used to vanish into the total without
    // trace. They are excluded deliberately and counted instead.
    revenueAtRisk: cases.reduce((s, c) => (c.impact_json.revenue != null ? s + Math.min(0, c.impact_json.revenue) : s), 0),
    unpriced: cases.filter(c => c.impact_json.revenue == null).length,
    cellsTested: cases[0]?.cells_tested ?? 0,
    coverageGaps,
    llmVerified: cases.filter(c => c.narrative_source === 'llm' && c.narrative_verified).length,
    spans,
  };
}

/** Only two classes of item qualify: the run did not finish clean, and a draft was rejected
 *  by the numeric guard. Anything already visible as a badge on a case row is not an alert --
 *  it is the same fact restated in a second place. */
export function healthOf(run: Run | null, cases: Case[]): Health[] {
  const out: Health[] = [];

  if (run && run.status !== 'complete') {
    out.push({
      level: run.status === 'partial' ? 'd' : 'w',
      what: `run ${run.run_id.slice(0, 8)} ${run.status}`,
      detail: run.note || 'no detail recorded',
      where: 'Runs',
    });
  }

  const rejected = cases.filter(c => c.unsupported.length > 0);
  if (rejected.length) {
    out.push({
      level: 'w',
      what: `${rejected.length} draft${rejected.length > 1 ? 's' : ''} rejected by the numeric guard`,
      detail: rejected
        .slice(0, 3)
        .map(c => `${c.metric} · ${c.unsupported.join(', ')}`)
        .join(' · '),
      where: 'Narrative',
    });
  }

  const borrowed = cases.filter(c => c.confidence_json.some(k => !k.scored && k.name === 'significance'));
  if (borrowed.length) {
    out.push({
      level: 'w',
      what: `${borrowed.length} case${borrowed.length > 1 ? 's' : ''} with significance unscored`,
      detail: 'evidence measured on a different cell than the one accused, so it was withheld',
      where: 'Evidence',
    });
  }

  return out;
}
