'use client';

import type { Recommendation, RecommendationSet } from '@/lib/types';

const PRIORITY_CLASS: Record<Recommendation['priority'], string> = {
  critical: 'badge d',
  high: 'badge w',
  medium: 'badge',
  low: 'badge q',
};

/** Advice, kept visually distinct from everything else in the panel.
 *
 *  Trace, evidence and narrative are all derived from measurement -- the narrative by a model,
 *  but one whose every figure was checked against the evidence bundle before publication.
 *  This tab is different in kind: it is a model proposing actions nobody has verified, and it
 *  is marked so a reader never mistakes a suggestion for a finding. */
export function Recommendations({ set, onRegenerate, busy }: {
  set: RecommendationSet;
  onRegenerate: () => void;
  busy: boolean;
}) {
  if (set.status === 'failed') {
    return (
      <div className="recs">
        <div className="guard">
          <b>could not generate advice</b>
          <span>{set.error || 'the remediation service did not return a result'}</span>
        </div>
        <button className="btn sm" onClick={onRegenerate} disabled={busy} style={{ marginTop: 12 }}>
          {busy ? 'retrying…' : 'Try again'}
        </button>
      </div>
    );
  }

  const removed = Math.max(0, set.drafted - set.recommendations.length);

  return (
    <div className="recs">
      <div className="recbar">
        <span className="badge a">AI generated</span>
        <span className="recprov">
          {/* The filter is the story. A first pass proposes; an independent reviewer with the
              same evidence and no attachment to the draft deletes what the case cannot carry. */}
          {set.drafted > 0 ? (
            <>
              <b>{set.recommendations.length}</b> kept of <b>{set.drafted}</b> drafted
              {removed > 0 && <> · {removed} removed in review</>}
            </>
          ) : (
            <>{set.recommendations.length} proposed</>
          )}
        </span>
        <button className="btn sm sp" onClick={onRegenerate} disabled={busy}>
          {busy ? 'regenerating…' : 'Regenerate'}
        </button>
      </div>

      {set.summary && <p className="recsum">{set.summary}</p>}

      {set.recommendations.length === 0 ? (
        <div className="empty" style={{ marginTop: 14 }}>
          Review rejected every draft. For this case the evidence did not support a concrete
          action, and an empty list is the honest answer.
        </div>
      ) : (
        <ol className="reclist">
          {set.recommendations.map((r, i) => (
            <li className="rec" key={i}>
              <div className="rech">
                <span className="recn">{i + 1}</span>
                <span className="rect">{r.title}</span>
                <span className={PRIORITY_CLASS[r.priority]}>{r.priority}</span>
                <span className="badge q">{r.confidence} confidence</span>
              </div>

              <div className="recrow">
                <span className="k">Action</span>
                <span className="v">{r.action}</span>
              </div>
              <div className="recrow">
                <span className="k">Why</span>
                <span className="v dim">{r.rationale}</span>
              </div>
              {r.expected_benefit && (
                <div className="recrow">
                  <span className="k">Expected</span>
                  <span className="v dim">{r.expected_benefit}</span>
                </div>
              )}
              <div className="recrow">
                <span className="k">Verify</span>
                <span className="v">{r.validation_step}</span>
              </div>
              {r.risk && (
                <div className="recrow">
                  <span className="k">Risk</span>
                  <span className="v warn">{r.risk}</span>
                </div>
              )}
              {r.evidence.length > 0 && (
                <div className="recrow">
                  <span className="k">Evidence</span>
                  <span className="v">
                    {r.evidence.map((e, j) => (
                      <span className="recev" key={j}>
                        {e}
                      </span>
                    ))}
                  </span>
                </div>
              )}
            </li>
          ))}
        </ol>
      )}

      <p className="recfoot">
        Written by {set.generation_model || 'a model'} and reviewed by{' '}
        {set.validation_model || 'a second pass'}. Unlike the narrative, these are not checked
        against the evidence bundle — they are proposals, and the verify step on each one is
        there because none of them has been tested.
      </p>
    </div>
  );
}
