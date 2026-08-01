'use client';

import { SortIcon } from './icons';
import { PUBLISH_THRESHOLD } from '@/lib/data';
import { ARROW, clearedOf, KIND_BADGE, KIND_LABEL, metricValue, money, pct, priority } from '@/lib/format';
import type { Case } from '@/lib/types';

export type Sort = 'priority' | 'effect' | 'confidence' | 'impact';

/** At most one flag per row, by severity. Two badges is a tie the eye has to break;
 *  the overflow count says a second exists without competing for the scan. */
function flagsOf(c: Case): { label: string; cls: string }[] {
  const flags: { label: string; cls: string }[] = [];
  if (c.narrative_source === 'template' && c.unsupported.length) flags.push({ label: 'guard fail', cls: 'badge d' });
  if (c.confidence < PUBLISH_THRESHOLD) flags.push({ label: 'below publish', cls: 'badge w' });
  if (c.recurrence_of) flags.push({ label: 'recurrence', cls: 'badge w' });
  return flags;
}

const COLS: { w: number; r?: boolean }[] = [
  { w: 3 },
  { w: 34 },
  { w: 120 },
  { w: 68, r: true },
  { w: 0 },
  { w: 106 },
  { w: 82, r: true },
  { w: 82, r: true },
  { w: 92 },
  { w: 88, r: true },
  { w: 74, r: true },
  { w: 48, r: true },
  { w: 118 },
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
    <th className={r ? 'r' : undefined} aria-sort={k === sort ? 'ascending' : undefined}>
      {k ? (
        <button onClick={() => onSort(k)}>
          {label}
          <SortIcon on={k === sort} />
        </button>
      ) : (
        label
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
            const publishable = c.confidence >= PUBLISH_THRESHOLD;
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
                <td className="m strong" title={c.segment}>
                  {named ? c.segment : <span className="dim2">—</span>}
                </td>
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
                <td className={`r num ${c.impact_json.revenue < 0 ? 'fall' : 'rise'}`}>{money(c.impact_json.revenue)}</td>
                <td className="m r">{clearedOf(c.candidates)}</td>
                <td className="m r" style={{ color: c.coverage.length ? 'var(--warn)' : 'var(--tx3)' }}>
                  {c.coverage.length}
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
