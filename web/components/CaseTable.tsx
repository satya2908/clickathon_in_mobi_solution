'use client';

import { SortIcon } from './icons';
import { Segment } from './Segment';
import { ARROW, clearedOf, impact, KIND_BADGE, KIND_LABEL, metricValue, pct, priority } from '@/lib/format';
import type { Case } from '@/lib/types';

export type Sort = 'priority' | 'effect' | 'confidence' | 'impact';

/** Every heading here is a term of art, and several are terms this system invented. A reader
 *  who cannot tell `Expected` from `Observed`, or who reads `Cleared` as "resolved" rather
 *  than "exonerated", will misread the whole table -- so each one says what it means. */
const HINT: Record<string, string> = {
  Pri: 'Impact weighted by confidence, so a large number nobody can stand behind does not outrank a proven small one. P0 is the most urgent.',
  Metric: 'The metric that moved. The arrow is the direction of the move.',
  Effect: 'How far the accused segment moved, relative to what was expected of it.',
  Segment:
    'The slice being accused. Multiple pills mean the intersection of them moved, and neither dimension moved on its own.',
  Verdict:
    'localized: a segment was named and the removal test held. unlocalized: real movement, no segment explains it. undecomposed: a lead that failed its breadth checks. no data: too little traffic to decompose.',
  Observed: 'What the accused segment actually read over the window.',
  Expected:
    'The detector baseline over aligned prior weeks: trimmed pooled counters for counts and proportions, or a log-space median for revenue and continuous ratios.',
  Conf:
    'A deterministic weighted score, not a model output. Publication also requires enough components and a measured significance check; the UI uses the engine’s stored decision.',
  Impact:
    'Estimated revenue effect. Some are reached through a chain of estimates rather than measured directly; the basis is on the case.',
  Cleared:
    'Candidates tested and genuinely exonerated, over candidates considered. Says what was ruled out, not only what was found.',
  Gaps: 'Cells that could not be tested at all — too little traffic, or no usable baseline. Published rather than dropped, because an untested segment is not an innocent one.',
  Flags: 'Anything about this case that should be read before the verdict is trusted.',
};

/** At most one flag per row, by severity. Two badges is a tie the eye has to break;
 *  the overflow count says a second exists without competing for the scan. */
function flagsOf(c: Case): { label: string; cls: string }[] {
  const flags: { label: string; cls: string }[] = [];
  if (c.narrative_source === 'template' && c.unsupported.length) flags.push({ label: 'guard fail', cls: 'badge d' });
  if (!c.publishable) flags.push({ label: 'withheld', cls: 'badge w' });
  if (c.recurrence_of) flags.push({ label: 'recurrence', cls: 'badge w' });
  return flags;
}

/** Why this cell reads in fills rather than dollars, said on the cell rather than in a
 *  footnote nobody reaches. */
const impactHint = (c: Case): string =>
  c.impact_json.revenue == null
    ? `No defensible conversion to revenue for ${c.metric}, so the move is reported in ${c.impact_json.unit}.`
    : c.impact_json.direct
      ? 'Measured directly.'
      : `Reached through a chain of estimates: ${c.impact_json.basis.join(' -> ') || 'see the case'}.`;

// Segment is the zero: it absorbs whatever the fixed columns leave, because it is the one
// cell whose length is not bounded. The rest are sized to their widest real value -- impact
// at 88px was rendering "20.1k im..." next to 290px of empty segment column.
const COLS: { w: number; r?: boolean }[] = [
  { w: 3 },
  { w: 34 },
  { w: 120 },
  { w: 88, r: true },
  { w: 0 },
  { w: 106 },
  { w: 82, r: true },
  { w: 94, r: true },
  { w: 96 },
  { w: 152, r: true },
  { w: 74, r: true },
  { w: 48, r: true },
  { w: 158 },
];

export function CaseTable({
  cases,
  openId,
  sort,
  onSort,
  onOpen,
}: {
  cases: Case[];
  openId: string | null;
  sort: Sort;
  onSort: (s: Sort) => void;
  onOpen: (id: string) => void;
}) {
  const Th = ({ label, k, r }: { label: string; k?: Sort; r?: boolean }) => (
    <th className={r ? 'r' : undefined} aria-sort={k === sort ? 'ascending' : undefined} title={HINT[label]}>
      {k ? (
        <button onClick={() => onSort(k)}>
          {label}
          <SortIcon on={k === sort} />
        </button>
      ) : (
        <span className="hint">{label}</span>
      )}
    </th>
  );

  return (
    <div className="tblbox">
      <table className="tbl">
        <colgroup>
          {COLS.map((c, i) => (
            <col key={i} style={c.w ? { width: c.w } : undefined} />
          ))}
        </colgroup>
        <thead>
          <tr>
            <th />
            <Th label="Pri" k="priority" />
            <Th label="Metric" />
            <Th label="Effect" k="effect" r />
            <Th label="Segment" />
            <Th label="Verdict" />
            <Th label="Observed" r />
            <Th label="Expected" r />
            <Th label="Conf" k="confidence" />
            <Th label="Impact" k="impact" r />
            <Th label="Cleared" r />
            <Th label="Gaps" r />
            <Th label="Flags" />
          </tr>
        </thead>
        <tbody>
          {cases.map(c => {
            const p = priority(c.impact_json.revenue, c.confidence);
            const flags = flagsOf(c);
            const named = Object.keys(c.segment_json).length > 0;
            const publishable = c.publishable;
            return (
              <tr
                key={c.case_id}
                tabIndex={0}
                aria-selected={c.case_id === openId}
                onClick={() => onOpen(c.case_id)}
                onKeyDown={e => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onOpen(c.case_id);
                  }
                }}
              >
                <td className={`spine p${p}`} />
                <td className="m">P{p}</td>
                <td className="m strong">
                  <span className={`arrow ${c.direction}`}>{ARROW[c.direction]}</span> {c.metric}
                </td>
                <td className={`r num ${c.direction}`}>{pct(c.relative_effect)}</td>
                <td>{named ? <Segment label={c.segment} max={2} /> : <span className="dim2">—</span>}</td>
                <td>
                  <span className={KIND_BADGE[c.verdict_kind]}>{KIND_LABEL[c.verdict_kind]}</span>
                </td>
                <td className="m r">{metricValue(c.metric, c.observed)}</td>
                <td className="m r dim2">{metricValue(c.metric, c.expected)}</td>
                <td className="m strong">
                  {c.confidence.toFixed(2)}
                  <span className="cbar">
                    <i className={publishable ? '' : 'low'} style={{ width: `${c.confidence * 100}%` }} />
                  </span>
                </td>
                <td className={`r num ${c.impact_json.units < 0 ? 'fall' : 'rise'}`} title={impactHint(c)}>
                  {impact(c.impact_json)}
                </td>
                <td className="m r">{clearedOf(c.candidates)}</td>
                <td
                  className="m r"
                  style={{ color: c.coverage_total ? 'var(--warn)' : 'var(--tx3)' }}
                  title={
                    c.coverage_total > c.coverage.length
                      ? `${c.coverage_total.toLocaleString()} total; the case panel shows the ${c.coverage.length} highest-volume rows`
                      : undefined
                  }
                >
                  {c.coverage_total}
                </td>
                <td title={flags.map(f => f.label).join(', ')}>
                  {flags[0] && <span className={flags[0].cls}>{flags[0].label}</span>}
                  {flags.length > 1 && (
                    <span className="dim2" style={{ fontSize: 10 }}>
                      {' '}
                      +{flags.length - 1}
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {cases.length === 0 && <div className="empty">no cases match this filter</div>}
    </div>
  );
}
