import { NextResponse } from 'next/server';

import { ingestEnabled } from '@/lib/features';
import { IngestError, runIngest } from '@/lib/ingest';

// A release is appended, the dimensions are refreshed if it reissued them, and every window it
// covers is then investigated. On the unseen bundle that is minutes, not seconds, and Node's
// default route timeout would kill it partway through a load.
export const maxDuration = 900;
export const dynamic = 'force-dynamic';

/** POST runs `verdict ingest` against a path and reports what happened.
 *
 *  Synchronous on purpose. A job queue would be the right shape for a service that many people
 *  use at once, and the wrong one here: the console has a single operator watching a single
 *  release land, and the thing they need is the command's own output, which is the same output
 *  they would have seen in a terminal. */
export async function POST(req: Request) {
  if (!ingestEnabled()) {
    return NextResponse.json(
      { error: 'Ingest from the console is disabled for this deployment' },
      { status: 503, headers: { 'Cache-Control': 'no-store' } },
    );
  }

  let body: { path?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'body must be JSON' }, { status: 400 });
  }

  let result;
  try {
    result = await runIngest(body.path ?? '');
  } catch (err) {
    if (err instanceof IngestError) {
      return NextResponse.json({ error: err.message }, { status: err.status });
    }
    return NextResponse.json({ error: (err as Error).message }, { status: 400 });
  }

  return NextResponse.json(result, {
    status: result.ok ? 200 : 502,
    headers: { 'Cache-Control': 'no-store' },
  });
}
