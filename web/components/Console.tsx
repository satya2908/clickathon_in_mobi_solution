'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { CasePanel } from './CasePanel';
import { CaseTable, type Sort } from './CaseTable';
import { IngestPanel } from './IngestPanel';
import { MetricChart } from './MetricChart';
import { SearchIcon } from './icons';
import { TopBar } from './TopBar';
import { healthOf, KINDS, kpiOf } from '@/lib/data';
import { KIND_FILL, KIND_LABEL, money, priority } from '@/lib/format';
import type { Series } from '@/lib/queries';
import type { Case, RecommendationSet, Run, VerdictKind } from '@/lib/types';

/** Unpriced cases sort last within their bucket rather than as zero. A case nobody could
 *  convert to revenue is of unknown size, and sorting it among the genuinely small ones
 *  hides it exactly where a reader has stopped looking. */
const byRevenue = (a: Case, b: Case) => {
  const x = a.impact_json.revenue;
  const y = b.impact_json.revenue;
  if (x == null && y == null) return b.confidence - a.confidence;
  if (x == null) return 1;
  if (y == null) return -1;
  return x - y;
};

const SORTERS: Record<Sort, (a: Case, b: Case) => number> = {
  priority: (a, b) =>
    priority(a.impact_json.revenue, a.confidence) - priority(b.impact_json.revenue, b.confidence) || byRevenue(a, b),
  effect: (a, b) => Math.abs(b.relative_effect) - Math.abs(a.relative_effect),
  confidence: (a, b) => b.confidence - a.confidence,
  impact: byRevenue,
};

interface Props {
  run: Run | null;
  runs: Run[];
  cases: Case[];
  series: Series[];
  spans: number;
  coverageGaps: number;
  recommendationsEnabled: boolean;
  ingestEnabled: boolean;
  empty: boolean;
}

/** Shown only when the database answered successfully but has no cases. Failed reads throw
 *  into the route error boundary, so they can never render as a clean empty run. */
function Empty({ runs, coverageGaps }: { runs: Run[]; coverageGaps: number }) {
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
        {coverageGaps > 0 && (
          <p className="dim" style={{ maxWidth: 620, lineHeight: 1.6, margin: '0 0 14px' }}>
            The selected run also recorded {coverageGaps.toLocaleString()} cells it could not
            test. No case is required for a coverage gap to exist.
          </p>
        )}
        <div className="sql">verdict investigate --start 2026-06-23T00:00:00 --hours 48</div>
      </div>
    </div>
  );
}

/** Priority is not a stored column. It is impact ranked by confidence, bucketed, so the
 *  filter has to describe what the buckets mean -- "P0" is otherwise just a colour. */
const PRIORITIES = [0, 1, 2, 3] as const;

const PRI_HINT: Record<number, string> = {
  0: 'Large impact and high confidence. Look at these first.',
  1: 'Material impact, or a large one held back by a weaker verdict.',
  2: 'Small but proven, or large and speculative.',
  3: 'Marginal on both counts. Kept so the ledger is complete.',
};

const PRI_FILL = ['var(--err)', 'var(--warn)', 'var(--tx2)', 'var(--line2)'];

const AI_HINT =
  'Off by default. When on, each case gains an Actions tab: one model drafts remediations from ' +
  'the case evidence, then a second reviews the draft independently and deletes anything the ' +
  'evidence does not support. Proposals, not findings — nothing here is verified against the ' +
  'numbers the way the narrative is.';

const KIND_HINT: Record<VerdictKind, string> = {
  localized: 'A segment was named and removing it returned the parent to its expected band.',
  unlocalized: 'The movement is real but no segment explains it — the signature of a change upstream of the auction.',
  undecomposed: 'A segment was named but failed its breadth checks. A lead, not a verdict.',
  no_data: 'Too little traffic to decompose. Published so it is not silently dropped.',
};

export function Console({
  run,
  runs,
  cases,
  series,
  spans,
  coverageGaps,
  recommendationsEnabled,
  ingestEnabled,
  empty,
}: Props) {
  const [kind, setKind] = useState<VerdictKind | null>(null);
  const [pri, setPri] = useState<number | null>(null);
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState<Sort>('priority');
  const [openId, setOpenId] = useState<string | null>(null);
  // Opening a case pushes a history entry, so Back closes the panel instead of
  // leaving the console. Only pop what we pushed.
  const pushed = useRef(false);

  const kpi = useMemo(() => kpiOf(cases, spans, coverageGaps), [cases, spans, coverageGaps]);
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

  // Advice is generated on demand, one case at a time, and cached in ClickHouse. Sequential
  // rather than parallel: each case is two full agent turns, and firing ten at once buys
  // nothing but a rate limit and a bill.
  const [recsOn, setRecsOn] = useState(false);
  const [recs, setRecs] = useState<Map<string, RecommendationSet>>(new Map());
  const [generating, setGenerating] = useState<string | null>(null);
  const [pending, setPending] = useState(0);
  // Holds the run the queue was started for, so switching runs restarts it and a re-render
  // does not.
  const queued = useRef<string | null>(null);

  const store = (set: RecommendationSet) => setRecs(prev => new Map(prev).set(set.case_id, set));

  async function generateFor(caseId: string, force: boolean) {
    if (!recommendationsEnabled || !run) return;
    setGenerating(caseId);
    try {
      const res = await fetch('/api/recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run: run.run_id, case_id: caseId, force }),
      });
      const body = await res.json();
      if (body.set) store(body.set);
    } catch (err) {
      store({
        case_id: caseId,
        generated_at: new Date().toISOString(),
        status: 'failed',
        summary: '',
        drafted: 0,
        recommendations: [],
        generation_model: '',
        validation_model: '',
        job_id: '',
        error: (err as Error).message,
      });
    } finally {
      setGenerating(null);
    }
  }

  // Switching the toggle on loads what exists and then generates the rest, one at a time,
  // so nobody has to open each case and press a button. Only the missing ones: a case that
  // already has advice is never regenerated, because each one costs about two minutes of
  // model time and the results do not change unless the case does.
  //
  // Sequential on purpose. Firing seven at once buys a rate limit, and doing them in order
  // means the queue can be abandoned part-way with everything finished so far kept.
  useEffect(() => {
    if (!recommendationsEnabled || !recsOn || !run) return;
    if (queued.current === run.run_id) return;
    queued.current = run.run_id;

    let cancelled = false;
    (async () => {
      let missing: string[] = [];
      try {
        const res = await fetch(`/api/recommendations?run=${encodeURIComponent(run.run_id)}`);
        const body = (await res.json()) as { sets?: Record<string, RecommendationSet>; missing?: string[] };
        if (cancelled) return;
        if (body.sets) setRecs(new Map(Object.entries(body.sets)));
        missing = body.missing ?? [];
      } catch {
        return;
      }

      setPending(missing.length);
      for (const caseId of missing) {
        if (cancelled) return;
        await generateFor(caseId, false);
        if (cancelled) return;
        setPending(n => Math.max(0, n - 1));
      }
    })();

    // Turning the toggle off abandons the queue. Anything already generated is in ClickHouse
    // and comes straight back if it is switched on again -- which requires clearing the guard
    // here, or the second switch-on matches the run it already ran for and returns without
    // reloading anything, leaving the panel permanently empty.
    return () => {
      cancelled = true;
      queued.current = null;
      setPending(0);
    };
  }, [recommendationsEnabled, recsOn, run]);

  const recsReady = useMemo(
    () => [...recs.values()].filter(s => s.status === 'completed').length,
    [recs],
  );

  const byPri = useMemo(() => {
    const counts: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0 };
    for (const c of cases) counts[priority(c.impact_json.revenue, c.confidence)]++;
    return counts;
  }, [cases]);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return cases
      .filter(
        c =>
          (!kind || c.verdict_kind === kind) &&
          (pri === null || priority(c.impact_json.revenue, c.confidence) === pri) &&
          (!q || `${c.metric} ${c.segment}`.toLowerCase().includes(q)),
      )
      .sort(SORTERS[sort]);
  }, [cases, kind, pri, query, sort]);

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
            <Empty runs={runs} coverageGaps={coverageGaps} />
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

                <div
                  className="kpi"
                  title={
                    kpi.unpriced
                      ? `Losses only, never netted against recoveries. ${kpi.unpriced} of ${kpi.cases} cases measure a count that could not be converted to revenue and are excluded, so this is a floor.`
                      : 'Losses only, never netted against recoveries: a quiet total would hide an hour in which one thing broke and another improved.'
                  }
                >
                  <span className="hd">Revenue at risk</span>
                  <span className="v fall">{money(kpi.revenueAtRisk)}</span>
                  <span className="def">
                    losses only · not netted
                    {kpi.unpriced > 0 && <span style={{ color: 'var(--warn)' }}> · {kpi.unpriced} unpriced</span>}
                  </span>
                </div>

                <div className="kpi">
                  <span className="hd">Mean confidence</span>
                  <span className="v">{kpi.meanConfidence.toFixed(2)}</span>
                  <span className="def">
                    {kpi.published} / {kpi.cases} engine-publishable
                  </span>
                </div>

                <div className="kpi">
                  <span className="hd">Coverage gaps</span>
                  <span className="v">{kpi.coverageGaps.toLocaleString()}</span>
                  <span className="def">all untestable cells in this run</span>
                </div>
              </div>

              {series.length > 0 && <MetricChart series={series} />}

              <div className="strip">
                <div className="fchips" role="group" aria-label="Filter by priority">
                  <button
                    className={`fchip${pri === null ? ' on' : ''}`}
                    aria-pressed={pri === null}
                    onClick={() => setPri(null)}
                    title="Every priority"
                  >
                    All <span className="n">{kpi.cases}</span>
                  </button>
                  {PRIORITIES.filter(p => byPri[p] > 0).map(p => (
                    <button
                      key={p}
                      className={`fchip pchip${pri === p ? ' on' : ''}`}
                      aria-pressed={pri === p}
                      onClick={() => setPri(pri === p ? null : p)}
                      title={PRI_HINT[p]}
                    >
                      <span className="dot" style={{ background: PRI_FILL[p] }} />P{p} <span className="n">{byPri[p]}</span>
                    </button>
                  ))}
                </div>

                <div className="fchips" role="group" aria-label="Filter by verdict" style={{ marginLeft: 4 }}>
                  {KINDS.filter(k => kpi.byKind[k] > 0).map(k => (
                    <button
                      key={k}
                      className={`fchip${kind === k ? ' on' : ''}`}
                      aria-pressed={kind === k}
                      onClick={() => setKind(kind === k ? null : k)}
                      title={KIND_HINT[k]}
                    >
                      <span className="sw" style={{ background: KIND_FILL[k] }} />
                      {KIND_LABEL[k]} <span className="n">{kpi.byKind[k]}</span>
                    </button>
                  ))}
                </div>

                {/* Everything after this sits on the right: the filters describe what you are
                    looking at, these act on it. */}
                <div className="push" />

                {recommendationsEnabled && (
                  /* Off by default. Everything else on this page is measured; this is a model
                     proposing actions, and opting into that should be a decision rather than
                     something a reader discovers already switched on. */
                  <label className="aitog" title={AI_HINT}>
                    <input
                      type="checkbox"
                      checked={recsOn}
                      onChange={e => setRecsOn(e.target.checked)}
                      aria-label="AI recommendations"
                    />
                    <span className="track">
                      <span className="knob" />
                    </span>
                    <span className="lbl">
                      AI Recommendations
                      {recsOn && pending > 0 && (
                        <span className="n busy" title={`${pending} case(s) still to generate`}>
                          <span className="spin xs" />
                          {pending}
                        </span>
                      )}
                      {recsOn && pending === 0 && recsReady > 0 && <span className="n">{recsReady}</span>}
                    </span>
                  </label>
                )}

                {ingestEnabled && <IngestPanel />}

                <div className="row" style={{ gap: 6, width: 232 }}>
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
        {/* Beside the cell count deliberately: the two together are the claim, and either one
            alone invites the wrong question. */}
        <span title="Temporal segment-and-metric tests; structural sibling-grid tests are separate">
          {kpi.cellsTested.toLocaleString()} temporal tests
        </span>
        {run && run.duration_ms > 0 && (
          <span title="Wall clock for the whole run: detection, localization and persistence">
            {run.duration_ms < 1000
              ? `${run.duration_ms} ms`
              : `${(run.duration_ms / 1000).toFixed(1)}s`}
          </span>
        )}
        <span>{kpi.spans.toLocaleString()} spans</span>
        <span>{kpi.llmVerified} narratives verified</span>
        <span>
          {rows.length} of {kpi.cases} shown
        </span>
        <span className="sp">{run ? `${run.finished_at.slice(11, 16)} UTC` : ''}</span>
      </div>

      {openCase && (
        <CasePanel
          key={openCase.case_id}
          c={openCase}
          onClose={close}
          recommendations={recs.get(openCase.case_id) ?? null}
          recsEnabled={recommendationsEnabled && recsOn}
          recsBusy={generating === openCase.case_id}
          onGenerate={force => generateFor(openCase.case_id, force)}
        />
      )}
    </div>
  );
}
