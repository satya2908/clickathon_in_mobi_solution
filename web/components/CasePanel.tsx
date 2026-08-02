'use client';

import { useEffect, useMemo, useState } from 'react';
import { CloseIcon, LinkIcon } from './icons';
import { TraceTree } from './TraceTree';
import { Narrative } from './Narrative';
import { usePanelWidth } from './usePanelWidth';
import { Recommendations } from './Recommendations';
import { Waterfall } from './Waterfall';
import { Segment } from './Segment';
import { flatten } from '@/lib/data';
import { traceUrl } from '@/lib/links';
import { ARROW, clearedOf, impact, KIND_BADGE, KIND_LABEL, metricValue, money, ms, pct } from '@/lib/format';
import type { Candidate, Case, RecommendationSet, Step } from '@/lib/types';

const STATUS_BADGE: Record<Candidate['status'], string> = {
  accused: 'badge a',
  cleared: 'badge g',
  partial: 'badge w',
  too_broad: 'badge w',
  too_narrow: 'badge w',
  wrong_direction: 'badge',
  immaterial: 'badge',
  did_not_reproduce: 'badge d',
  considered: 'badge q',
};

/** One span, in full: what the stage did, why it was worth running, and what came back.
 *  All three are written by the engine as the step executes -- `why` before the result is
 *  known, so it cannot be a rationalisation of whatever happened to be found. */
function NodeDetail({ n }: { n: Step }) {
  const failed = n.result.startsWith('failed:');
  return (
    <div className="nd">
      <div className="ndh">
        <span className="k">{n.kind}</span>
        <span className="n">{n.name}</span>
        <span className="t">
          {ms(n.duration_ms)}
          {n.span_id ? ` · span ${n.span_id.slice(0, 8)}` : ''}
        </span>
      </div>

      {n.what && (
        <div className="ndrow">
          <span className="k">What</span>
          <span className="v">{n.what}</span>
        </div>
      )}

      {n.why && (
        <div className="ndrow">
          <span className="k">Why</span>
          <span className="v dim">{n.why}</span>
        </div>
      )}

      <div className="ndrow">
        <span className="k">Result</span>
        <span className="v">
          {failed ? (
            <span className="guard">
              <span className="mono">{n.result}</span>
            </span>
          ) : (
            n.result || <span className="dim2">nothing recorded</span>
          )}
        </span>
      </div>

      {n.sql && (
        <div className="ndrow">
          <span className="k">SQL</span>
          <span className="v">
            <div className="sql">{n.sql}</div>
          </span>
        </div>
      )}
    </div>
  );
}

/** The reasoning, gathered. Each stage's `why` is written once per run but reads the same
 *  across cases, so the drawer collapses the trace to one row per distinct stage rather
 *  than repeating the same sentence down a thirty-node tree. */
function Method({ nodes }: { nodes: Step[] }) {
  const seen = new Map<string, string>();
  for (const n of nodes) {
    const stage = n.name.split(':')[0];
    if (n.why && !seen.has(stage)) seen.set(stage, n.why);
  }

  if (!seen.size) {
    return (
      <div className="nd">
        <div className="ndh">
          <span className="n">no rationale recorded</span>
        </div>
        <div className="ndrow">
          <span className="v dim">This run stored its steps without the `why` field.</span>
        </div>
      </div>
    );
  }

  return (
    <div className="nd">
      <div className="ndh">
        <span className="n">why each stage exists</span>
      </div>
      {[...seen].map(([stage, why]) => (
        <div className="ndrow" key={stage} style={{ gridTemplateColumns: '132px minmax(0, 1fr)' }}>
          <span className="k mono" style={{ textTransform: 'none', letterSpacing: 0, fontSize: 11 }}>
            {stage}
          </span>
          <span className="v dim">{why}</span>
        </div>
      ))}
    </div>
  );
}

/** A case stored before step persistence has no tree to draw. Saying so beats an empty
 *  panel that looks like a rendering failure. */
function NoTrace() {
  return (
    <div className="pane">
      <div className="empty">
        No steps were stored for this case. Re-run the investigation to capture the trace.
      </div>
    </div>
  );
}

export function CasePanel({
  c,
  onClose,
  recommendations,
  recsEnabled,
  recsBusy,
  onGenerate,
}: {
  c: Case;
  onClose: () => void;
  recommendations: RecommendationSet | null;
  recsEnabled: boolean;
  recsBusy: boolean;
  onGenerate: (force: boolean) => void;
}) {
  const root = c.trace;
  const nodes = useMemo(() => (root ? flatten(root) : []), [root]);
  const [tab, setTab] = useState<'trace' | 'evidence' | 'narrative' | 'actions'>('trace');
  // Opens on the localizer rather than the root: the root's detail is a restatement
  // of the header, so landing there wastes the first look at the panel. Keyed on step_id
  // rather than span_id, which is empty whenever tracing is switched off.
  const [sel, setSel] = useState(() => (nodes.find(n => n.name.startsWith('localize:')) ?? nodes[0])?.step_id ?? '');
  const [method, setMethod] = useState(false);
  const [view, setView] = useState<'tree' | 'timeline'>('tree');
  const { width, dragging, onPointerDown, onKeyDown, reset, min } = usePanelWidth();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);
    const { overflow } = document.body.style;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = overflow;
    };
  }, [onClose]);

  // Actions appears only when the toggle is on, so the panel keeps its shape for anyone who
  // never turns model-written advice on at all.
  const tabs = (recsEnabled
    ? (['trace', 'evidence', 'narrative', 'actions'] as const)
    : (['trace', 'evidence', 'narrative'] as const)) as readonly typeof tab[];

  useEffect(() => {
    if (!recsEnabled && tab === 'actions') setTab('trace');
  }, [recsEnabled, tab]);

  const node = nodes.find(n => n.step_id === sel) ?? nodes[0] ?? null;
  const accused = c.candidates.find(x => x.status === 'accused');
  const hyperdx = traceUrl(c.trace_id, c.detected_at);
  const unscored = c.confidence_json.filter(x => !x.scored).length;
  const publishable = c.publishable;
  const gateMark = { pass: '✓', fail: '✗', unknown: '–' } as const;
  const gateCls = { pass: 'y', fail: 'n', unknown: 'u' } as const;

  return (
    <>
      <div className="scrim" onMouseDown={onClose} />
      <div
        className={`side${dragging ? ' dragging' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-label={`Case ${c.case_id.slice(0, 8)}`}
        style={width === null ? undefined : { width }}
      >
        <div
          className="sgrip"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize panel — arrow keys, or double-click to reset"
          aria-valuenow={width ?? 0}
          aria-valuemin={min}
          tabIndex={0}
          onPointerDown={onPointerDown}
          onKeyDown={onKeyDown}
          onDoubleClick={reset}
          title="Drag to resize · double-click to reset"
        />
        <div className="shead">
          <div className="stop">
            <div style={{ minWidth: 0, flex: 1 }}>
              <h2>
                <span className={c.direction}>{ARROW[c.direction]}</span>
                <span className="mono">{c.metric}</span>
                <span className="dim2" style={{ fontWeight: 400 }}>
                  ·
                </span>
                <Segment label={c.segment} />
                <span className={KIND_BADGE[c.verdict_kind]}>{KIND_LABEL[c.verdict_kind]}</span>
                {c.recurrence_of && <span className="badge w">recurrence</span>}
              </h2>
              <div className="pills">
                <span className={`pill ${c.direction === 'fall' ? 'd' : c.direction === 'rise' ? 'g' : ''}`}>
                  <span className="k">Effect</span>
                  <span className="v">{pct(c.relative_effect)}</span>
                </span>
                <span className={`pill ${c.impact_json.units < 0 ? 'd' : 'g'}`}>
                  <span className="k">Impact</span>
                  <span className="v">{impact(c.impact_json)}</span>
                </span>
                <span className={`pill${publishable ? '' : ' w'}`}>
                  <span className="k">Confidence</span>
                  <span className="v">{c.confidence.toFixed(2)}</span>
                </span>

                <span className="pinfo sp">
                  {c.case_id.slice(0, 12)}
                  {hyperdx && (
                    <a href={hyperdx} target="_blank" rel="noreferrer">
                      <LinkIcon />
                      HyperDX
                    </a>
                  )}
                </span>
              </div>
            </div>
            <button className="btn sm sp" onClick={onClose} title="Close (Esc)" style={{ display: 'inline-flex' }}>
              <CloseIcon />
            </button>
          </div>

          <div className="stabs">
            {tabs.map(t => (
              <button
                key={t}
                className={`${tab === t ? 'on' : ''}${t === 'actions' ? ' accent' : ''}`}
                onClick={() => setTab(t)}
              >
                {t === 'trace'
                  ? `Trace ${nodes.length}`
                  : t === 'evidence'
                    ? `Evidence ${c.candidates.length}`
                    : t === 'narrative'
                      ? 'Narrative'
                      : `Actions${recommendations?.status === 'completed' ? ` ${recommendations.recommendations.length}` : ''}`}
              </button>
            ))}
          </div>
        </div>

        {tab === 'trace' && !root && <NoTrace />}

        {tab === 'trace' && root && (
          <div className="sbody">
            <div className="tcol">
              <div className="colhd">
                Trace
                {/* Two readings of one set of spans. The tree answers what contains what; the
                    timeline answers where the time went, which a nested list cannot show
                    because it gives a 2ms stage and a 2s stage the same height. */}
                <span className="vtoggle sp">
                  <button
                    className={view === 'tree' ? 'on' : ''}
                    onClick={() => setView('tree')}
                    title="Nesting: which stage ran inside which"
                  >
                    Tree
                  </button>
                  <button
                    className={view === 'timeline' ? 'on' : ''}
                    onClick={() => setView('timeline')}
                    title="Timing: every span against one clock, positioned by when it started"
                  >
                    Timeline
                  </button>
                </span>
                <span className="mono" style={{ letterSpacing: 0, textTransform: 'none', marginLeft: 8 }}>
                  {ms(root.duration_ms)}
                </span>
              </div>
              <div className="scroll">
                {view === 'tree' ? (
                  <TraceTree
                    root={root}
                    selected={sel}
                    onSelect={(s: Step) => {
                      setSel(s.step_id);
                      setMethod(false);
                    }}
                  />
                ) : (
                  <Waterfall
                    nodes={nodes}
                    selected={sel}
                    onSelect={(s: Step) => {
                      setSel(s.step_id);
                      setMethod(false);
                    }}
                  />
                )}
              </div>
              <div className="kinds">
                <span>
                  <i style={{ background: 'var(--acc)' }} /> localizer
                </span>
                <span>
                  <i style={{ background: 'var(--tx3)' }} /> query
                </span>
                <span>
                  <i style={{ background: 'var(--line2)' }} /> other
                </span>
              </div>
            </div>
            <div className="dcol">
              <div className="colhd">
                {method ? 'Method' : 'Node'}
                <span className="sp" />
                <button
                  className={method ? 'on' : ''}
                  onClick={() => setMethod(m => !m)}
                  aria-pressed={method}
                  title={method ? 'Back to node' : 'Why each stage exists'}
                >
                  {method ? '×' : '?'}
                </button>
              </div>
              <div className="scroll">{method ? <Method nodes={nodes} /> : node ? <NodeDetail n={node} /> : null}</div>
            </div>
          </div>
        )}

        {tab === 'evidence' && (
          <div className="scroll">
            <div className="pane">
              <section className="sec">
                <div className="sechd">
                  <span className="hd">Confidence {c.confidence.toFixed(2)}</span>
                  <span
                    className={publishable ? 'badge g' : 'badge w'}
                    title={c.confidence_caveat || undefined}
                  >
                    {publishable ? 'publishable' : 'withheld'}
                  </span>
                  {unscored > 0 && <span className="badge w">{unscored} of 5 unscored</span>}
                  <span className="gates sp">
                    {(Object.keys(c.gates_json) as (keyof typeof c.gates_json)[]).map((g, i) => (
                      <span key={g}>
                        {i > 0 && <span className="dim2"> · </span>}
                        {g} <b className={gateCls[c.gates_json[g]]}>{gateMark[c.gates_json[g]]}</b>
                      </span>
                    ))}
                  </span>
                </div>
                {c.confidence_caveat && (
                  <p className="dim" style={{ margin: '0 0 10px', lineHeight: 1.5 }}>
                    {c.confidence_caveat}
                  </p>
                )}
                <div className="tblbox">
                  <table className="tbl dense">
                    <colgroup>
                      <col style={{ width: 128 }} />
                      <col style={{ width: 60 }} />
                      <col style={{ width: 58 }} />
                      <col style={{ width: 88 }} />
                      <col />
                    </colgroup>
                    <thead>
                      <tr>
                        <th>Component</th>
                        <th className="r">Weight</th>
                        <th className="r">Score</th>
                        <th />
                        <th>Detail</th>
                      </tr>
                    </thead>
                    <tbody>
                      {c.confidence_json.map(k => (
                        <tr key={k.name}>
                          <td className="m strong">{k.name}</td>
                          <td className="m r">{k.weight.toFixed(2)}</td>
                          <td className="m r" style={{ color: k.scored ? 'var(--tx)' : 'var(--tx3)' }}>
                            {k.scored ? k.score.toFixed(2) : '—'}
                          </td>
                          <td>
                            <span className="compbar">
                              <i className={k.scored ? '' : 'off'} style={{ width: `${(k.scored ? k.score : 1) * 100}%` }} />
                            </span>
                          </td>
                          <td title={k.detail}>{k.detail}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="sec">
                <div className="sechd">
                  <span className="hd">Candidates</span>
                  <span className="dim2 mono" style={{ fontSize: 11 }}>
                    {clearedOf(c.candidates)} ruled out{accused ? ' · 1 accused' : ' · none accused'}
                  </span>
                </div>
                <div className="tblbox">
                  <table className="tbl dense">
                    <colgroup>
                      <col />
                      <col style={{ width: 56 }} />
                      <col style={{ width: 78 }} />
                      <col style={{ width: 88 }} />
                      <col style={{ width: 52 }} />
                      <col style={{ width: 52 }} />
                      <col style={{ width: 52 }} />
                      <col style={{ width: 112 }} />
                      <col style={{ width: 220 }} />
                    </colgroup>
                    <thead>
                      <tr>
                        <th>Candidate</th>
                        <th className="r">Depth</th>
                        <th className="r">Observed</th>
                        <th className="r">If innocent</th>
                        <th className="r">Suff</th>
                        <th className="r">Min</th>
                        <th className="r">Hold</th>
                        <th>Status</th>
                        <th>Reason</th>
                      </tr>
                    </thead>
                    <tbody>
                      {c.candidates.map(k => (
                        <tr key={k.candidate}>
                          <td className="m strong" title={k.candidate}>
                            {k.candidate}
                          </td>
                          <td className="m r">{k.depth}</td>
                          <td className="m r">{metricValue(c.metric, k.observed)}</td>
                          <td className="m r">{metricValue(c.metric, k.predicted)}</td>
                          <td className="m r">{k.sufficiency ? k.sufficiency.toFixed(2) : '—'}</td>
                          <td className="m r">{k.minimality ? k.minimality.toFixed(2) : '—'}</td>
                          <td className="m r">{k.holdout ? k.holdout.toFixed(2) : '—'}</td>
                          <td>
                            <span className={STATUS_BADGE[k.status]}>{k.status}</span>
                          </td>
                          <td title={k.reason}>{k.reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="sec">
                <div className="sechd">
                  <span className="hd">Coverage</span>
                  <span className="dim2 mono" style={{ fontSize: 11 }}>
                    {c.coverage_total
                      ? `${c.coverage_total.toLocaleString()} cells unresolved${
                          c.coverage_total > c.coverage.length
                            ? ` · showing top ${c.coverage.length}`
                            : ''
                        }`
                      : 'no gaps'}
                  </span>
                </div>
                <div className="tblbox">
                  {c.coverage_total === 0 ? (
                    <div className="empty">every cell cleared the detection floor</div>
                  ) : (
                    <table className="tbl dense">
                      <colgroup>
                        <col style={{ width: 128 }} />
                        <col />
                        <col style={{ width: 100 }} />
                        <col style={{ width: 88 }} />
                        <col style={{ width: 180 }} />
                        <col style={{ width: 110 }} />
                      </colgroup>
                      <thead>
                        <tr>
                          <th>Combo</th>
                          <th>Key</th>
                          <th className="r">Denominator</th>
                          <th className="r">Required</th>
                          <th>Reason</th>
                          <th className="r">Resolvable</th>
                        </tr>
                      </thead>
                      <tbody>
                        {c.coverage.map((g, i) => (
                          <tr key={i}>
                            <td className="m">{g.combo}</td>
                            <td className="m strong">{[g.key_a, g.key_b].filter(Boolean).join(' · ')}</td>
                            <td className="m r">{g.denominator.toLocaleString()}</td>
                            <td className="m r">{g.required.toLocaleString()}</td>
                            <td>
                              <span className="badge w">{g.reason}</span>
                            </td>
                            <td className="m r">{g.resolvable_effect < 0 ? 'unresolvable' : pct(g.resolvable_effect)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </section>
            </div>
          </div>
        )}

        {tab === 'narrative' && (
          <div className="scroll">
            <div className="pane">
              <div className="sechd">
                <span className={c.narrative_source === 'llm' ? 'badge a' : 'badge'}>{c.narrative_source}</span>
                {c.llm_model && <span className="badge">{c.llm_model}</span>}
                {c.narrative_verified ? (
                  <span className="badge g">figures verified</span>
                ) : (
                  <span className="badge w">not model-written</span>
                )}
              </div>

              {/* Prose holds a readable measure while the facts move alongside it. Stacking
                  them left a column of empty panel to the right of every line, and the
                  impact figures are what a reader checks the prose against. */}
              <div className="nargrid">
                <div className="narmain">
                  <Narrative text={c.narrative} />

                  {c.unsupported.length > 0 && (
                    <div className="guard">
                      <b>guard rejected draft</b>
                      <span>
                        {c.unsupported.length} figures absent from the evidence bundle ({c.unsupported.join(', ')}) — template published instead
                      </span>
                    </div>
                  )}
                </div>

                <aside className="naraside">
                  <div className="hd" style={{ marginBottom: 8 }}>
                    Impact
                  </div>
                  <div className="tblbox">
                    <table className="tbl dense">
                      <colgroup>
                        <col style={{ width: 78 }} />
                        <col />
                      </colgroup>
                      <tbody>
                        <tr>
                          <td className="m dim2">units</td>
                          <td className="m strong">
                            {c.impact_json.units.toLocaleString()} {c.impact_json.unit}
                          </td>
                        </tr>
                        <tr>
                          <td className="m dim2">revenue</td>
                          <td className="m strong">
                            {c.impact_json.revenue != null ? (
                              money(c.impact_json.revenue)
                            ) : (
                              <span className="dim2">not priced &mdash; {c.metric} is a count</span>
                            )}
                          </td>
                        </tr>
                        <tr>
                          <td className="m dim2">direct</td>
                          <td className="m">{String(c.impact_json.direct)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  {/* The chain, one step per line. Joined with arrows it ran past the panel
                      edge and the reader lost which multiplication produced the figure. */}
                  <div className="hd" style={{ margin: '16px 0 8px' }}>
                    Basis
                  </div>
                  {c.impact_json.basis.length ? (
                    <ol className="basis">
                      {c.impact_json.basis.map((b, i) => (
                        <li key={i}>{b}</li>
                      ))}
                    </ol>
                  ) : (
                    <p className="dim2" style={{ fontSize: 11.5, margin: 0 }}>
                      Measured directly — no chain of estimates.
                    </p>
                  )}
                </aside>
              </div>
            </div>
          </div>
        )}

        {tab === 'actions' && (
          <div className="sbody one">
            <div className="dcol">
              <div className="colhd">
                What to do about it
                <span className="sp dim2" style={{ letterSpacing: 0, textTransform: 'none' }}>
                  proposed, not verified
                </span>
              </div>
              <div className="scroll" style={{ padding: '14px 18px 24px' }}>
                {recommendations ? (
                  <Recommendations
                    set={recommendations}
                    busy={recsBusy}
                    onRegenerate={() => onGenerate(true)}
                  />
                ) : recsBusy ? (
                  <div className="recs">
                    <div className="recwait">
                      <span className="spin" />
                      <div>
                        <b>Generating, then reviewing</b>
                        <span>
                          One pass drafts remediations from the case; a second, with no sight of
                          the first pass&apos;s reasoning, deletes what the evidence cannot carry.
                          Two model turns, usually under two minutes.
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="recs">
                    <p className="recsum dim">
                      No advice has been generated for this case yet.
                    </p>
                    <button className="btn sm" onClick={() => onGenerate(false)}>
                      Generate
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
