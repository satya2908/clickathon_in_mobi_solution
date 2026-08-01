import type { CSSProperties } from 'react';

/** A segment label as pills rather than as a sentence.
 *
 *  `publisher_tier=tier_1 AND category=gaming` is four pieces of information wearing the
 *  costume of prose: two dimensions, two values, and a conjunction. At table density the eye
 *  has to parse the `=` and the `AND` before it can find the part that varies between rows,
 *  which is the value. Splitting them lets the dimension recede and the value carry the weight.
 *
 *  The conjunction becomes a separator rather than a word. It is the same operator on every
 *  multi-dimension segment in the system -- an intersection, always -- so spelling it out
 *  costs three characters per row to say something that is never in question. */
export function Segment({ label, max = 0, style }: { label: string; max?: number; style?: CSSProperties }) {
  if (!label || label === 'all traffic') {
    return (
      <span className="seg-pills" style={style}>
        <span className="seg-all">all traffic</span>
      </span>
    );
  }

  const parts = label.split(' AND ');
  const shown = max > 0 ? parts.slice(0, max) : parts;
  const hidden = parts.length - shown.length;

  return (
    // The full string stays in the title: it is what gets pasted into a query, and the pills
    // are not selectable as one run of text.
    <span className="seg-pills" title={label} style={style}>
      {shown.map((part, i) => {
        const eq = part.indexOf('=');
        const dim = eq < 0 ? '' : part.slice(0, eq);
        const value = eq < 0 ? part : part.slice(eq + 1);
        return (
          <span className="seg-pill" key={`${part}-${i}`}>
            {dim && <span className="seg-k">{dim}</span>}
            <span className="seg-v">{value}</span>
          </span>
        );
      })}
      {hidden > 0 && <span className="seg-more">+{hidden}</span>}
    </span>
  );
}
