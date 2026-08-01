import { Fragment } from 'react';

/** The narrative, as the engine actually wrote it.
 *
 *  `template_narration` composes four sections joined by blank lines -- what moved, what the
 *  counterfactual showed, what was cleared, what could not be scored. HTML collapses those
 *  breaks, so the whole thing arrived as one seventeen-line block and the structure the
 *  generator went to the trouble of producing was invisible.
 *
 *  Figures are set in mono so they can be picked out without reading the sentence around
 *  them. This is presentation only: no number is reformatted, rounded, or reordered, because
 *  the text was passed by a verifier that checked every figure against the evidence bundle
 *  and changing one here would invalidate that check. */

// Deliberately conservative. Matches a signed decimal with optional exponent, percent, or
// thousands separators, but only as a whole token -- so `2026-07-05` and `p=2.50e-02` keep
// their shape rather than being carved into pieces.
const FIGURE = /(?<![\w.-])(?:[+\u2212-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?%?)(?![\w-])/g;

function withFigures(text: string, keyPrefix: string) {
  const out: React.ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(FIGURE)) {
    const at = m.index ?? 0;
    if (at > last) out.push(text.slice(last, at));
    out.push(
      <span className="fig" key={`${keyPrefix}-${at}`}>
        {m[0]}
      </span>,
    );
    last = at + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Narrative({ text }: { text: string }) {
  if (!text.trim()) return <p className="narbody dim2">No narrative was written for this case.</p>;

  // Split on blank lines; a single newline inside a section is a wrap, not a break.
  const paragraphs = text
    .split(/\n\s*\n/)
    .map(p => p.replace(/\s*\n\s*/g, ' ').trim())
    .filter(Boolean);

  return (
    <div className="narbody">
      {paragraphs.map((p, i) => (
        <Fragment key={i}>
          {/* The first paragraph is the verdict itself; the rest are its support. */}
          <p className={i === 0 ? 'lead' : undefined}>{withFigures(p, String(i))}</p>
        </Fragment>
      ))}
    </div>
  );
}
