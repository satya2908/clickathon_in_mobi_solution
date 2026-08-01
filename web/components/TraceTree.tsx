'use client';

import { useState } from 'react';
import { ChevronIcon } from './icons';
import { STEP_COLOR } from '@/lib/format';
import type { Step } from '@/lib/types';

/** The tree is a picker, not a document. Selecting a span fills the right column;
 *  nothing expands in place, so the list never reflows under the cursor. */
export function TraceTree({ root, selected, onSelect }: { root: Step; selected: string; onSelect: (s: Step) => void }) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const toggle = (id: string) =>
    setCollapsed(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const render = (s: Step, depth: number, total: number): React.ReactNode => {
    const kids = s.children ?? [];
    const open = !collapsed.has(s.step_id);
    // span_id is empty whenever tracing is off, so identity keys off step_id, which the
    // engine always writes.
    const failed = s.result.startsWith('failed:');
    return (
      <div key={s.step_id}>
        <button
          className={`span${s.step_id === selected ? ' on' : ''}${failed ? ' err' : ''}`}
          onClick={() => onSelect(s)}
          aria-current={s.step_id === selected}
          title={s.what || s.name}
        >
          <span className="sind" style={{ width: depth * 14 }} />
          <span
            className={`stw${open ? ' open' : ''}`}
            onClick={e => {
              if (!kids.length) return;
              e.stopPropagation();
              toggle(s.step_id);
            }}
            role={kids.length ? 'button' : undefined}
            aria-label={kids.length ? (open ? 'Collapse' : 'Expand') : undefined}
          >
            {kids.length > 0 && <ChevronIcon size={9} />}
          </span>
          <span className="sdot" style={{ background: STEP_COLOR[s.kind] }} />
          <span className="sname">{s.name}</span>
          {/* Neutral, because it encodes duration. The dot already encodes kind, and a
              bar tinted by kind made the scaffolding stages nearly invisible. */}
          <span className="sbar">
            <i
              style={{
                width: `${Math.max(3, total > 0 ? (s.duration_ms / total) * 100 : 3)}%`,
                background: failed ? 'var(--err)' : 'var(--tx3)',
              }}
            />
          </span>
          <span className="sms">{s.duration_ms}ms</span>
        </button>
        {open && kids.map(k => render(k, depth + 1, total))}
      </div>
    );
  };

  return <div className="tree">{render(root, 0, root.duration_ms)}</div>;
}
