'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { axisValue, metricValue, pct, stamp, ticks } from '@/lib/format';
import type { Series } from '@/lib/queries';

// Floor, not the height. The chart grows into whatever the page has spare, which on a
// 1080p window is most of it -- 132px of plot under a seven-row table left the band and the
// line inside a strip too shallow to read a deviation off.
const H_MIN = 132;
const PAD = { t: 8, r: 12, b: 20, l: 50 };

const path = (pts: { x: number; y: number }[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');

export function MetricChart({ series }: { series: Series[] }) {
  const [which, setWhich] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const shape = series[Math.min(which, series.length - 1)];
  const metric = shape.metric;
  // Rendering the SVG at the container's true pixel width keeps the viewBox 1:1.
  // A fixed viewBox with preserveAspectRatio="none" stretched every axis label.
  const [w, setW] = useState(1200);
  const [H, setH] = useState(H_MIN);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => {
      setW(Math.max(320, Math.round(e.contentRect.width)));
      setH(Math.max(H_MIN, Math.round(e.contentRect.height)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const { points, geo } = useMemo(() => {
    const points = shape.points;
    const lo = Math.min(...points.map(p => Math.min(p.observed, p.lo)));
    const hi = Math.max(...points.map(p => Math.max(p.observed, p.hi)));
    const span = hi - lo || Math.abs(hi) || 1;
    const min = lo - span * 0.2;
    const max = hi + span * 0.12;
    const x = (i: number) => PAD.l + (i / Math.max(1, points.length - 1)) * (w - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - (v - min) / (max - min)) * (H - PAD.t - PAD.b);
    return { points, geo: { x, y, min, max } };
  }, [shape, w, H]);

  const obs = points.map((p, i) => ({ x: geo.x(i), y: geo.y(p.observed) }));
  const exp = points.map((p, i) => ({ x: geo.x(i), y: geo.y(p.expected) }));
  const bandTop = points.map((p, i) => ({ x: geo.x(i), y: geo.y(p.hi) }));
  const bandBottom = points.map((p, i) => ({ x: geo.x(i), y: geo.y(p.lo) })).reverse();
  const axis = ticks(geo.min, geo.max);

  // -1 means the observed line never left the band, in which case there is no onset to
  // mark and drawing one would invent an incident the series does not show.
  const flagged = shape.from >= 0 && shape.to >= shape.from;
  const onset = flagged ? points[shape.from] : null;
  const cursor = hover ?? points.length - 1;
  const at = points[cursor];
  const delta = at.expected !== 0 ? (at.observed - at.expected) / at.expected : 0;
  const baselineLabel = at.baseline_weeks_seen
    ? `${at.baseline_weeks_used}${
        at.baseline_weeks_used === at.baseline_weeks_seen
          ? ''
          : ` of ${at.baseline_weeks_seen}`
      } aligned baseline wk used`
    : 'no usable baseline';

  return (
    <div className="panelbox chartbox">
      <div className="pbhead">
        <span className="hd">Parent series</span>
        <div className="seg" role="group" aria-label="Metric">
          {series.map((s, i) => (
            <button key={s.metric} className={i === which ? 'on' : ''} aria-pressed={i === which} onClick={() => setWhich(i)}>
              {s.metric}
            </button>
          ))}
        </div>
      </div>

      <div className="readout">
        <span className="mono dim2" style={{ fontSize: 11 }}>
          {stamp(at.t)}
        </span>
        <span className="big">{metricValue(metric, at.observed)}</span>
        <span className={`num ${delta < 0 ? 'fall' : delta > 0 ? 'rise' : ''}`}>{pct(delta)}</span>
        <span className="mono dim2 sp" style={{ fontSize: 11 }}>
          expected {metricValue(metric, at.expected)}
        </span>
        <span className="clegend">
          <span>
            <i style={{ borderColor: 'var(--acc)' }} /> observed
          </span>
          <span>
            <i style={{ borderColor: 'var(--tx3)', borderTopStyle: 'dashed' }} /> expected
          </span>
          <span>
            <i style={{ borderColor: 'var(--tx3)', opacity: 0.35 }} /> robust historical spread ·{' '}
            {baselineLabel}
          </span>
        </span>
      </div>

      <div className="chartw" ref={box}>
        <svg width={w} height={H} viewBox={`0 0 ${w} ${H}`}>
          {axis.values.map(v => (
            <g key={v}>
              <line className="gridline" x1={PAD.l} x2={w - PAD.r} y1={geo.y(v)} y2={geo.y(v)} />
              <text className="axistx" x={PAD.l - 8} y={geo.y(v) + 3} textAnchor="end">
                {axisValue(metric, v, axis.step)}
              </text>
            </g>
          ))}

          {flagged && (
            <>
              <rect
                className="anomfill"
                x={geo.x(shape.from)}
                y={PAD.t}
                width={Math.max(1, geo.x(shape.to) - geo.x(shape.from))}
                height={H - PAD.t - PAD.b}
              />
              <line className="anomedge" x1={geo.x(shape.from)} x2={geo.x(shape.from)} y1={PAD.t} y2={H - PAD.b} />
            </>
          )}

          <path className="bandfill" d={`${path(bandTop)} ${path(bandBottom).replace('M', 'L')} Z`} />
          <path className="expline" d={path(exp)} />
          <path className="obsline" d={path(obs)} />

          {flagged && onset && (
            <circle cx={geo.x(shape.from)} cy={geo.y(onset.observed)} r={3} fill="var(--err)" stroke="var(--bg)" strokeWidth={1.4} />
          )}
          <line className="gridline" x1={obs[cursor].x} x2={obs[cursor].x} y1={PAD.t} y2={H - PAD.b} />
          <circle cx={obs[cursor].x} cy={obs[cursor].y} r={3} fill="var(--acc)" stroke="var(--bg)" strokeWidth={1.4} />

          {points.map((p, i) =>
            i % 8 === 0 ? (
              <text key={i} className="axistx" x={geo.x(i)} y={H - 6} textAnchor="middle">
                {p.t.slice(11, 16)}
              </text>
            ) : null,
          )}

          {points.map((_, i) => (
            <rect
              key={i}
              x={geo.x(i) - (w - PAD.l - PAD.r) / points.length / 2}
              y={PAD.t}
              width={(w - PAD.l - PAD.r) / points.length}
              height={H - PAD.t - PAD.b}
              fill="transparent"
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
          ))}
        </svg>
      </div>

      <div className="cfoot">
        {flagged && onset ? (
          <>
            <span className="badge d">onset {stamp(onset.t)}</span>
            <span className="mono dim2" style={{ fontSize: 11 }}>
              parent effect {pct(shape.effect)} over {shape.to - shape.from + 1}h outside the historical band
            </span>
          </>
        ) : (
          <span className="mono dim2" style={{ fontSize: 11 }}>
            {shape.label} stayed inside its expected band for the whole window
          </span>
        )}
      </div>
    </div>
  );
}
