import 'server-only';

import { chWrite, rows } from './clickhouse';
import type { Case, Recommendation, RecommendationSet } from './types';

/** Two-pass remediation advice for a diagnosed case.
 *
 *  The remediation service runs a generation pass and then an independent validation pass with
 *  a separate session, giving the reviewer the first draft and the original evidence and asking
 *  it to delete anything the case does not support. That second pass is the point: a single
 *  model asked for remediations will happily return six, of which two are restatements of the
 *  verdict and one is actively harmful. The reviewer is instructed that an empty list beats
 *  unsupported advice, and it does cut items -- one run dropped a pause-the-inventory
 *  suggestion as "mechanism-unsupported and potentially harmful".
 *
 *  Both counts are kept so a reader can see the filter working. "Three of four kept" is a more
 *  useful signal about advice quality than any confidence label a model assigns itself. */

const SERVICE = process.env.RECOMMEND_URL || 'http://localhost:8157';
const SERVICE_TOKEN = process.env.RECOMMEND_API_TOKEN?.trim() || '';

// Generation and validation are two full agent turns; a completed run measured about 110
// seconds. The ceiling is generous because timing out on work that is nearly done wastes the
// whole spend, and the caller polls rather than blocks.
const TIMEOUT_MS = Number(process.env.RECOMMEND_TIMEOUT_MS || 360_000);
const POLL_MS = 3_000;

function serviceHeaders(json = false): Record<string, string> {
  const headers: Record<string, string> = {};
  if (json) headers['Content-Type'] = 'application/json';
  if (SERVICE_TOKEN) headers.Authorization = `Bearer ${SERVICE_TOKEN}`;
  return headers;
}

interface JobRecord {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  draft_recommendations?: number;
  generation_model?: string;
  validation_model?: string;
  error?: string;
  result?: { summary?: string; recommendations?: Recommendation[] } | null;
}

/** What the model is allowed to see. Deliberately not the whole row: the narrative is already
 *  a model artefact and feeding it back invites the second pass to agree with the first, and
 *  the trace is thousands of spans of no remediation value. Everything here is a computed
 *  fact, which is what the prompt tells the model to treat as authoritative. */
function digest(c: Case) {
  return {
    metric: c.metric,
    segment: c.segment || 'all traffic',
    verdict: c.verdict_kind,
    direction: c.direction,
    window: { start: c.window_start, end: c.window_end, grain: c.grain },
    observed: c.observed,
    expected: c.expected,
    relative_effect: c.relative_effect,
    p_value: c.p_value,
    confidence: c.confidence,
    confidence_components: c.confidence_json.map(x => ({
      name: x.name,
      scored: x.scored,
      score: x.score,
      weight: x.weight,
      why: x.detail,
    })),
    gates: c.gates_json,
    impact: c.impact_json,
    detector: c.detector,
    localization_mode: c.mode,
    // Both halves of the ledger. What was ruled out constrains remediation as much as what was
    // accused: advice that targets an exonerated segment is advice against the evidence.
    accused: c.candidates.filter(x => x.status === 'accused').map(x => x.candidate),
    exonerated: c.candidates.filter(x => x.status === 'cleared').map(x => x.candidate),
    other_candidates: c.candidates
      .filter(x => x.status !== 'accused' && x.status !== 'cleared')
      .slice(0, 12)
      .map(x => ({ candidate: x.candidate, status: x.status })),
    untestable_cells: c.coverage_total,
  };
}

const CONTEXT = `This is an ad-tech supply/demand funnel: requests -> fills -> impressions -> clicks, with
revenue accruing on impressions. A "segment" is a slice of traffic by dimension. The verdict,
statistics, confidence components and exoneration ledger are computed deterministically and are
authoritative -- do not recompute or dispute them. Recommend operational or configuration
changes an ad-ops or engineering team could actually make. Where the verdict is "unlocalized",
no segment explains the movement, which usually points to a change upstream of the auction;
say so rather than inventing a segment to blame.`;

async function post(path: string, body: unknown): Promise<Response> {
  return fetch(`${SERVICE}${path}`, {
    method: 'POST',
    headers: serviceHeaders(true),
    body: JSON.stringify(body),
    cache: 'no-store',
  });
}

function connectionHint(): string {
  let host: string;
  try {
    host = new URL(SERVICE).hostname;
  } catch {
    return 'RECOMMEND_URL must be a valid absolute URL.';
  }
  if (host === 'cursor-cli-agent') {
    return 'Start the optional service with `./stack.sh up --with-ai`.';
  }
  if (host === 'host.docker.internal') {
    return (
      'A loopback-only SSH forward is not reachable from Docker; bind the forward to an ' +
      'address visible to the container or use the shared Docker network.'
    );
  }
  if (host === 'localhost' || host === '127.0.0.1') {
    return 'Start the remediation service on the host; localhost is only valid for host-side development.';
  }
  return 'Check the configured address, service health, and network route.';
}

/** Runs both passes and returns the validated set. Throws with a readable message; the caller
 *  turns that into a stored `failed` row so a broken run is visible rather than looking like
 *  a case nobody has generated advice for yet. */
export async function generate(c: Case): Promise<RecommendationSet> {
  let accepted: Response;
  try {
    accepted = await post('/v1/remediations', {
      case_id: c.case_id,
      case_data: digest(c),
      additional_context: CONTEXT,
      max_recommendations: 5,
    });
  } catch (err) {
    throw new Error(
      `cannot reach the remediation service at ${SERVICE} (${(err as Error).message}). ` +
        connectionHint(),
    );
  }
  if (!accepted.ok) {
    throw new Error(`remediation service refused the case (${accepted.status}): ${(await accepted.text()).slice(0, 300)}`);
  }
  const { job_id } = (await accepted.json()) as { job_id: string };

  const deadline = Date.now() + TIMEOUT_MS;
  let job: JobRecord | null = null;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, POLL_MS));
    const res = await fetch(`${SERVICE}/v1/remediations/${job_id}`, {
      headers: serviceHeaders(),
      cache: 'no-store',
    });
    if (!res.ok) continue;
    job = (await res.json()) as JobRecord;
    if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') break;
  }

  if (!job) throw new Error('remediation service never reported a status');
  if (job.status !== 'completed') {
    throw new Error(job.error || `remediation job ended as ${job.status}`);
  }

  return {
    case_id: c.case_id,
    generated_at: new Date().toISOString(),
    status: 'completed',
    summary: job.result?.summary ?? '',
    drafted: job.draft_recommendations ?? 0,
    recommendations: job.result?.recommendations ?? [],
    generation_model: job.generation_model ?? '',
    validation_model: job.validation_model ?? '',
    job_id,
    error: '',
  };
}

const chDateTime = (iso: string) => iso.replace('T', ' ').replace(/\.\d+/, '').replace('Z', '');

export async function save(set: RecommendationSet): Promise<void> {
  await chWrite().insert({
    table: 'case_recommendations',
    format: 'JSONEachRow',
    values: [
      {
        case_id: set.case_id,
        generated_at: chDateTime(set.generated_at),
        status: set.status,
        summary: set.summary,
        drafted: set.drafted,
        kept: set.recommendations.length,
        recommendations: JSON.stringify(set.recommendations),
        generation_model: set.generation_model,
        validation_model: set.validation_model,
        job_id: set.job_id,
        error: set.error,
      },
    ],
  });
}

interface Row {
  case_id: string;
  generated_at: string;
  status: string;
  summary: string;
  drafted: number;
  kept: number;
  recommendations: string;
  generation_model: string;
  validation_model: string;
  job_id: string;
  error: string;
}

/** `FINAL` because the table is Replacing and a regenerated set must not be read alongside the
 *  one it replaced. The row count here is one per case, so the cost is irrelevant. */
export async function load(caseIds: string[]): Promise<Map<string, RecommendationSet>> {
  const out = new Map<string, RecommendationSet>();
  if (!caseIds.length) return out;

  const raw = await rows<Row>(
    `SELECT case_id, toString(generated_at) AS generated_at, status, summary, drafted, kept,
            recommendations, generation_model, validation_model, job_id, error
     FROM case_recommendations FINAL
     WHERE case_id IN {ids:Array(String)}`,
    { ids: caseIds },
  );

  for (const r of raw) {
    let parsed: Recommendation[] = [];
    try {
      const j = JSON.parse(r.recommendations || '[]');
      if (Array.isArray(j)) parsed = j;
    } catch {
      // A row whose payload will not parse is reported as a failure rather than as no advice:
      // silence here is indistinguishable from never having run.
      parsed = [];
    }
    out.set(r.case_id, {
      case_id: r.case_id,
      generated_at: r.generated_at,
      status: r.status === 'completed' ? 'completed' : 'failed',
      summary: r.summary,
      drafted: Number(r.drafted) || 0,
      recommendations: parsed,
      generation_model: r.generation_model,
      validation_model: r.validation_model,
      job_id: r.job_id,
      error: r.error,
    });
  }
  return out;
}
