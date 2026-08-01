'use client';

import { useEffect, useRef, useState } from 'react';
import { AlertIcon } from './icons';
import type { Health, Run } from '@/lib/types';

/** Renders nothing when every check passed. An indicator that is always lit teaches
 *  people to stop looking at it, which is exactly what the old always-on alerts
 *  panel did. A popover rather than a modal: two rows do not warrant a scrim. */
export function HealthChip({ health, run }: { health: Health[]; run: Run | null }) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    const onDown = (e: MouseEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('mousedown', onDown);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('mousedown', onDown);
    };
  }, [open]);

  if (health.length === 0) return null;
  const worst = health.some(h => h.level === 'd') ? '' : ' w';

  return (
    <div className="hwrap" ref={wrap}>
      <button
        className={`chip${worst}`}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        aria-haspopup="dialog"
        title={`${health.length} checks failed`}
      >
        <AlertIcon size={12} />
        <b>{health.length}</b>
      </button>

      {open && (
        <div className="pop" role="dialog" aria-label="Failed checks">
          {health.map(h => (
            <div className="popr" key={h.what}>
              <span className={`ic ${h.level}`}>
                <AlertIcon size={13} />
              </span>
              <span className="bd">
                <span className="t1">{h.what}</span>
                <span className="t2">{h.detail}</span>
              </span>
              <span className="go">{h.where}</span>
            </div>
          ))}
          <div className="popf">
            {health.length} check{health.length > 1 ? 's' : ''} failed{run ? ` · run ${run.run_id.slice(0, 8)}` : ''}
          </div>
        </div>
      )}
    </div>
  );
}
