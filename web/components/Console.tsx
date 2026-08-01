'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { CasePanel } from './CasePanel';
import { CaseTable, type Sort } from './CaseTable';
import { MetricChart } from './MetricChart';
import { SearchIcon } from './icons';
import { TopBar } from './TopBar';
import { healthOf, KINDS, kpiOf, PUBLISH_THRESHOLD } from '@/lib/data';
import { KIND_FILL, KIND_LABEL, money, priority } from '@/lib/format';
import type { Series } from '@/lib/queries';
import type { Case, Run, VerdictKind } from '@/lib/types';

const SORTERS: Record<Sort, (a: Case, b: Case) => number> = {
  priority: (a, b) =>
    priority(a.impact_json.revenue, a.confidence) - priority(b.impact_json.revenue, b.confidence) ||
    a.impact_json.revenue - b.impact_json.revenue,
  effect: (a, b) => Math.abs(b.relative_effect) - Math.abs(a.relative_effect),
  confidence: (a, b) => b.confidence - a.confidence,
  impact: (a, b) => a.impact_json.revenue - b.impact_json.revenue,
};

interface Props {
  run: Run | null;
  runs: Run[];
  cases: Case[];
  series: Series[];
  spans: number;
  empty: boolean;
}

/** Shown when the database is reachable but has no cases. Distinct from a failed read,
 *  which logs server-side and degrades that one section: a console that renders an empty
 *  table for both leaves the reader unable to tell "nothing broke" from "I am broken". */
function Empty({ runs }: { runs: Run[] }) {
  return (
    <div className="wrap">
      <div className="panelbox" style={{ padding: 28 }}>
        <div className="hd" style={{ marginBottom: 10 }}>
          No cases to show
        </div>
        <p className="dim" style={{ maxWidth: 620, lineHeight: 1.6, margin: '0 0 14px' }}>
          {runs.length
            ? 'The most recent runs completed without publishing a case. Either the windows were quiet, or the sweep has not been pointed at a window containing an incident.'
            : 'No runs have been recorded yet. The console reads what the engine writes, so it stays empty until an investigation has been persisted.'}
        </p>
        <div className="sql">verdict investigate --start 2026-06-23T00:00:00 --hours 48</div>
      </div>
    </div>
  );
}

export function Console({ run, runs, cases, series, spans, empty }: Props) {
  const [kind, setKind] = useState<VerdictKind | null>(null);
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState<Sort>('priority');
  const [openId, setOpenId] = useState<string | null>(null);
  // Opening a case pushes a history entry, so Back closes the panel instead of
  // leaving the console. Only pop what we pushed.
  const pushed = useRef(false);

  const kpi = useMemo(() => kpiOf(cases, spans), [cases, spans]);
  const health = useMemo(() => healthOf(run, cases), [run, cases]);

  useEffect(() => {
    const sync = () => {
      const h = window.location.hash.slice(1);
      setOpenId(h ? (cases.find(c => c.case_id.startsWith(h))?.case_id ?? null) : null);
    };
    sync();
    window.addEventListener('hashchange', sync);
    return () => window.removeEventListener('hashchange', sync);
  }, [cases]);

  const open = (id: string) => {
    pushed.current = true;
    window.location.hash = id.slice(0, 12);
  };
  const close = () => {
    if (pushed.current) {
      pushed.current = false;
      window.history.back();
    } else {
      window.history.replaceState(null, '', window.location.pathname);
      setOpenId(null);
    }
  };

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return cases
      .filter(c => (!kind || c.verdict_kind === kind) && (!q || `${c.metric} ${c.segment}`.toLowerCase().includes(q)))
      .sort(SORTERS[sort]);
  }, [cases, kind, query, sort]);

  const openCase = openId ? cases.find(c => c.case_id === openId) : null;
  const window0 = cases[0];

  return (
    <div className="app">
      <TopBar
        health={health}
        run={run}
        windowStart={window0?.window_start ?? run?.started_at ?? ''}
        windowEnd={window0?.window_end ?? run?.finished_at ?? ''}
        grain={window0?.grain ?? '1h'}
      />

      <div className="body">
        <div className="scroll">
          {empty ? (
            <Empty runs={runs} />
          ) : (
            <div className="wrap">
              <div className="kpis">
                <div className="kpi">
                  <span className="hd">Open cases</span>
                  <span className="v">{kpi.cases}</span>
                  <span className="split" title={KINDS.map(k => `${kpi.byKind[k]} ${KIND_LABEL[k]}`).join(' · ')}>
                    {KINDS.map(k => (
                      <i key={k} style={{ width: `${(kpi.byKind[k] / kpi.cases) * 100}%`, background: KIND_FILL[k] }} />
                    ))}
                  </span>
                </div>

                <div className="kpi">
                  <span className="hd">Revenue at risk</span>
                  <span className="v fall">{money(kpi.revenueAtRisk)}</span>
                  <span className="def">losses only · not netted</span>
                </div>

                <div className="kpi">
                  <span className="hd">Mean confidence</span>
                  <span className="v">{kpi.meanConfidence.toFixed(2)}</span>
                  <span className="def">
                    {kpi.published} / {kpi.cases} above {PUBLISH_THRESHOLD.toFixed(2)}
                  </span>
                </div>

                <div className="kpi">
                  <span className="hd">Coverage gaps</span>
                  <span className="v">{kpi.coverageGaps.toLocaleString()}</span>
                  <span className="def">cells untestable at {window0?.grain ?? '1h'}</span>
                </div>
              </div>

              {series.length > 0 && <MetricChart series={series} />}

              <div className="strip">
                <div className="fchips" role="group" aria-label="Filter by verdict">
                  <button className={`fchip${kind === null ? ' on' : ''}`} aria-pressed={kind === null} onClick={() => setKind(null)}>
                    All <span className="n">{kpi.cases}</span>
                  </button>
                  {KINDS.filter(k => kpi.byKind[k] > 0).map(k => (
                    <button
                      key={k}
                      className={`fchip${kind === k ? ' on' : ''}`}
                      aria-pressed={kind === k}
                      onClick={() => setKind(kind === k ? null : k)}
                    >
                      <span className="sw" style={{ background: KIND_FILL[k] }} />
                      {KIND_LABEL[k]} <span className="n">{kpi.byKind[k]}</span>
                    </button>
                  ))}
                </div>

                <div className="row sp" style={{ gap: 6, width: 232 }}>
                  <span className="dim2" style={{ display: 'inline-flex' }}>
                    <SearchIcon />
                  </span>
                  <input
                    className="inp"
                    placeholder="filter metric or segment…"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    aria-label="Filter cases"
                  />
                </div>
              </div>

              <CaseTable cases={rows} openId={openId} sort={sort} onSort={setSort} onOpen={open} />
            </div>
          )}
        </div>
      </div>

      <div className="status">
        <span>{run ? run.run_id.slice(0, 8) : 'no run'}</span>
        <span>{kpi.cellsTested.toLocaleString()} cells</span>
        <span>{kpi.spans.toLocaleString()} spans</span>
        <span>{kpi.llmVerified} narratives verified</span>
        <span>
          {rows.length} of {kpi.cases} shown
        </span>
        <span className="sp">{run ? `${run.finished_at.slice(11, 16)} UTC` : ''}</span>
      </div>

      {openCase && <CasePanel key={openCase.case_id} c={openCase} onClose={close} />}
    </div>
  );
}
