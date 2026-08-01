'use client';

import { STEP_COLOR } from '@/lib/format';
import type { Step } from '@/lib/types';

/** The investigation as a timeline rather than an outline.
 *
 *  The tree next to this answers "what contains what". It cannot answer "where did the time
 *  go", because a nested list gives every stage the same visual weight whether it took two
 *  milliseconds or two seconds. Laying the same spans against a shared clock makes the
 *  expensive stage the widest thing on screen, which is the one question a reviewer asks of a
 *  trace before any other.
 *
 *  Positions come from `offset_ms`, measured at span start, rather than from summing earlier
 *  siblings. The difference shows up as the gaps: time a stage spent waiting is time no child
 *  accounts for, and a derived layout would quietly close those up and report a run that was
 *  busy throughout. */
export function Waterfall({
  nodes,
  selected,
  onSelect,
}: {
  nodes: Step[];
  selected: string;
  onSelect: (s: Step) => void;
}) {
  if (!nodes.length) return null;

  // The run's own extent, not the root span's duration: the root is snapshotted while still
  // open, so a child that outlived the reading would otherwise overflow the track.
  const start = Math.min(...nodes.map(n => n.offset_ms));
  const end = Math.max(...nodes.map(n => n.offset_ms + n.duration_ms));
  const total = Math.max(1, end - start);

  // Four ticks. Six were specified for a wider column and overlapped into a smear once the
  // track was sized to fit beside the 460px trace panel.
  const ticks = Array.from({ length: 4 }, (_, i) => (total * i) / 3);

  return (
    <div className="wf">
      <div className="wf-axis">
        {ticks.map((t, i) => (
          <span key={i} className="wf-tick" style={{ left: `${(t / total) * 100}%` }}>
            {t >= 1000 ? `${(t / 1000).toFixed(1)}s` : `${Math.round(t)}ms`}
          </span>
        ))}
      </div>

      <div className="wf-rows">
        {nodes.map(n => {
          const failed = n.result.startsWith('failed:');
          const left = ((n.offset_ms - start) / total) * 100;
          // A floor on width, because a sub-millisecond span is still a span that ran and a
          // zero-width bar reads as one that did not.
          const width = Math.max(0.4, (n.duration_ms / total) * 100);
          return (
            <button
              key={n.step_id}
              className={`wf-row${n.step_id === selected ? ' on' : ''}`}
              onClick={() => onSelect(n)}
              aria-current={n.step_id === selected}
              title={`${n.name} · started ${n.offset_ms}ms in · ran ${n.duration_ms}ms`}
            >
              <span className="wf-name" style={{ paddingLeft: (n.depth ?? 0) * 10 }}>
                <span className="wf-dot" style={{ background: STEP_COLOR[n.kind] }} />
                {n.name}
              </span>
              <span className="wf-track">
                <i
                  className="wf-bar"
                  style={{
                    left: `${left}%`,
                    width: `${Math.min(width, 100 - left)}%`,
                    background: failed ? 'var(--err)' : STEP_COLOR[n.kind],
                  }}
                />
              </span>
              <span className="wf-ms">{n.duration_ms}ms</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
