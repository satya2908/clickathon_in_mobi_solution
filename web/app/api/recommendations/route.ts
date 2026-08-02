import { NextResponse } from 'next/server';

import { recommendationsEnabled } from '@/lib/features';
import { getCases } from '@/lib/queries';
import { generate, load, save } from '@/lib/recommend';

// Two agent turns per case, roughly two minutes. Node's default route timeout would kill this
// mid-generation and leave the spend with nothing to show for it.
export const maxDuration = 800;
export const dynamic = 'force-dynamic';

function disabled() {
  return NextResponse.json(
    { error: 'AI recommendations are disabled for this deployment' },
    { status: 503, headers: { 'Cache-Control': 'no-store' } },
  );
}

/** GET returns whatever advice already exists, without generating any.
 *
 *  The toggle needs this: switching it on must show cached results immediately rather than
 *  regenerating work that has already been paid for. */
export async function GET(req: Request) {
  if (!recommendationsEnabled()) return disabled();

  const runId = new URL(req.url).searchParams.get('run') ?? '';
  if (!runId) return NextResponse.json({ error: 'run is required' }, { status: 400 });

  const cases = await getCases(runId);
  const have = await load(cases.map(c => c.case_id));
  return NextResponse.json({
    sets: Object.fromEntries(have),
    missing: cases.filter(c => !have.has(c.case_id)).map(c => c.case_id),
  });
}

/** POST generates advice for cases that do not have it.
 *
 *  One case per request, chosen by the client, so the browser can show progress and the work
 *  is resumable: a page closed halfway through keeps everything already generated. Doing the
 *  whole run in one request would mean a single timeout discards ten minutes of model time.
 *
 *  A failure is stored rather than thrown away. An error row is a case somebody tried and
 *  could not generate advice for, which is different from one nobody has tried, and only the
 *  first of those should be retried automatically. */
export async function POST(req: Request) {
  if (!recommendationsEnabled()) return disabled();

  let body: { run?: string; case_id?: string; force?: boolean };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'body must be JSON' }, { status: 400 });
  }

  const { run, case_id: caseId, force } = body;
  if (!run || !caseId) return NextResponse.json({ error: 'run and case_id are required' }, { status: 400 });

  const cases = await getCases(run);
  const target = cases.find(c => c.case_id === caseId);
  if (!target) return NextResponse.json({ error: `case ${caseId} is not in run ${run}` }, { status: 404 });

  if (!force) {
    const existing = await load([caseId]);
    const hit = existing.get(caseId);
    if (hit && hit.status === 'completed') return NextResponse.json({ set: hit, cached: true });
  }

  try {
    const set = await generate(target);
    await save(set);
    return NextResponse.json({ set, cached: false });
  } catch (err) {
    const message = (err as Error).message.slice(0, 1000);
    const failed = {
      case_id: caseId,
      generated_at: new Date().toISOString(),
      status: 'failed' as const,
      summary: '',
      drafted: 0,
      recommendations: [],
      generation_model: '',
      validation_model: '',
      job_id: '',
      error: message,
    };
    // Best effort: if the database is also unreachable, the response still carries the reason.
    await save(failed).catch(() => {});
    return NextResponse.json({ set: failed, cached: false, error: message }, { status: 502 });
  }
}
