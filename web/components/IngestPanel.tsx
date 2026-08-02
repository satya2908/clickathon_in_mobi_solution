'use client';

import { useState } from 'react';

/** Point the system at a release without leaving the console.
 *
 *  This runs the same `verdict ingest` the terminal runs -- append the events, refresh the
 *  dimensions if the release reissued them, verify the batch reads back at every grain, then
 *  investigate each window it covers. There is no second ingest path here, deliberately: a
 *  browser-only shortcut that skipped the dimension refresh would attribute a whole batch to
 *  whatever the dictionaries last held, which is a bug this project has already had once.
 *
 *  The page is not refreshed automatically when it finishes. A run that has just landed changes
 *  every number on the screen, and reloading out from under someone mid-read is worse than
 *  asking. */

interface Result {
  ok: boolean;
  path: string;
  exitCode: number | null;
  durationMs: number;
  output: string;
  error?: string;
}

function UploadIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path
        d="M8 11V3m0 0L4.5 6.5M8 3l3.5 3.5M2.5 11.5v1a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-1"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function seconds(ms: number): string {
  return ms >= 60_000 ? `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s` : `${(ms / 1000).toFixed(1)}s`;
}

export function IngestPanel() {
  const [open, setOpen] = useState(false);
  const [path, setPath] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);

  async function run() {
    if (!path.trim() || busy) return;
    setBusy(true);
    setResult(null);
    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: path.trim() }),
      });
      const body = await res.json();
      setResult(
        res.ok
          ? body
          : { ok: false, path, exitCode: null, durationMs: 0, output: body.output ?? '', error: body.error ?? `HTTP ${res.status}` },
      );
    } catch (err) {
      setResult({ ok: false, path, exitCode: null, durationMs: 0, output: '', error: (err as Error).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="ingest">
      <button
        className={`fchip ingbtn${open ? ' open' : ''}`}
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        title="Append a new batch of events and investigate every window it covers, using the same command the terminal runs."
      >
        <UploadIcon />
        Ingest data
      </button>

      {open && (
        <div className="ingpop" role="dialog" aria-label="Ingest data">
          <div className="ingrow">
            <input
              className="inp grow"
              placeholder="/path/to/release (folder with ad_events.parquet)"
              value={path}
              spellCheck={false}
              disabled={busy}
              onChange={e => setPath(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && run()}
              aria-label="Path to the release"
            />
            <button className="fchip go" onClick={run} disabled={busy || !path.trim()}>
              {busy ? <span className="spin xs" /> : null}
              {busy ? 'Running' : 'Run'}
            </button>
          </div>

          <p className="ingnote">
            Appends the events, refreshes dimensions if the release reissued them, then
            investigates each window it covers. Minutes, not seconds — leave this open.
          </p>

          {result && (
            <div className={`ingres${result.ok ? ' ok' : ' bad'}`}>
              <div className="ingtop">
                <strong>{result.ok ? 'Ingest complete' : 'Ingest failed'}</strong>
                {result.durationMs > 0 && <span className="dim2">{seconds(result.durationMs)}</span>}
              </div>
              {result.error && <p className="ingerr">{result.error}</p>}
              {result.output && <pre className="ingout">{result.output.slice(-4000)}</pre>}
              {result.ok && (
                <button className="fchip on" onClick={() => window.location.reload()}>
                  Reload to see the new run
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
