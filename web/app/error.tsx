'use client';

import { useEffect } from 'react';

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <main className="app">
      <div className="body">
        <div className="scroll">
          <div className="wrap">
            <div className="panelbox" role="alert" style={{ padding: 28 }}>
              <div className="hd" style={{ marginBottom: 10 }}>
                Verdict data could not be loaded
              </div>
              <p className="dim" style={{ maxWidth: 620, lineHeight: 1.6, margin: '0 0 14px' }}>
                The console could not read ClickHouse. This is a data-source failure, not a
                clean run with no cases.
              </p>
              <button className="btn sm" onClick={reset}>
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
