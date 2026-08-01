import { Fragment } from 'react';

/** The narrative, as the engine actually wrote it.
 *
 *  The generator emits a small, fixed markup: a lead sentence, then `## ` sections of `- `
 *  bullets. Nothing else is permitted, so this is a five-case parser rather than a markdown
 *  dependency, and anything unrecognised falls through as a paragraph instead of vanishing.
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

type Block =
  | { kind: 'lead'; text: string }
  | { kind: 'para'; text: string }
  | { kind: 'heading'; text: string }
  | { kind: 'list'; items: string[] };

function parse(text: string): Block[] {
  const blocks: Block[] = [];
  let list: string[] | null = null;
  let seenLead = false;

  const flush = () => {
    if (list?.length) blocks.push({ kind: 'list', items: list });
    list = null;
  };

  for (const raw of text.split('\n')) {
    const line = raw.trim();
    if (!line) {
      flush();
      continue;
    }
    if (line.startsWith('## ')) {
      flush();
      blocks.push({ kind: 'heading', text: line.slice(3).trim() });
      continue;
    }
    if (line.startsWith('- ')) {
      (list ??= []).push(line.slice(2).trim());
      continue;
    }
    flush();
    // The first ordinary line is the verdict; later ones are a model that ignored the shape,
    // which still has to render rather than disappear.
    blocks.push({ kind: seenLead ? 'para' : 'lead', text: line });
    seenLead = true;
  }
  flush();
  return blocks;
}

export function Narrative({ text }: { text: string }) {
  if (!text.trim()) return <div className="narbody dim2">No narrative was written for this case.</div>;

  return (
    <div className="narbody">
      {parse(text).map((b, i) => (
        <Fragment key={i}>
          {b.kind === 'heading' && <h4 className="narh">{b.text}</h4>}
          {b.kind === 'lead' && <p className="lead">{withFigures(b.text, String(i))}</p>}
          {b.kind === 'para' && <p>{withFigures(b.text, String(i))}</p>}
          {b.kind === 'list' && (
            <ul className="narlist">
              {b.items.map((item, j) => (
                <li key={j}>{withFigures(item, `${i}-${j}`)}</li>
              ))}
            </ul>
          )}
        </Fragment>
      ))}
    </div>
  );
}
