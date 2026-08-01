'use client';

import { HealthChip } from './HealthChip';
import { ThemeToggle } from './ThemeToggle';
import { hyperdxUrl } from '@/lib/links';
import { stamp } from '@/lib/format';
import type { Grain, Health, Run } from '@/lib/types';

/** Everything here either works or is plainly static context. The window and grain
 *  are labels rather than controls: they scope every number on the page, and reading
 *  them as text is more honest than a dropdown that cannot drop. */
export function TopBar({
  health,
  run,
  windowStart,
  windowEnd,
  grain,
}: {
  health: Health[];
  run: Run | null;
  windowStart: string;
  windowEnd: string;
  grain: Grain;
}) {
  return (
    <div className="top">
      <span className="logo">
        <span className="mark">V</span>
        Verdict
      </span>
      <span className="vr" />
      <span className="mono dim2" style={{ fontSize: 11 }}>
        inmobi | glance
      </span>
      <span className="vr" />
      <span className="mono dim" style={{ fontSize: 11 }}>
        {windowStart && windowEnd ? `${stamp(windowStart)} → ${stamp(windowEnd)} UTC · ${grain}` : 'no window'}
      </span>

      <HealthChip health={health} run={run} />

      <div className="row sp" style={{ gap: 8 }}>
        <a className="btn sm" href={hyperdxUrl()} target="_blank" rel="noreferrer" style={{ textDecoration: 'none' }}>
          HyperDX
        </a>
        <ThemeToggle />
        <span className="av" title="Verdict console">
          V
        </span>
      </div>
    </div>
  );
}
