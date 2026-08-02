import 'server-only';

import { spawn } from 'node:child_process';
import { existsSync, realpathSync, statSync } from 'node:fs';
import path from 'node:path';

/** Drives `verdict ingest` from the console, so a fresh release can be pointed at the system
 *  without dropping to a terminal.
 *
 *  The engine already does the whole loop in one command -- append events, refresh dimensions if
 *  the release reissued them, verify the batch is readable at every grain, then investigate each
 *  window it covers. This is a thin wrapper around that, deliberately: a second implementation of
 *  the ingest sequence living in the web tier is the kind of thing that drifts from the real one
 *  and then diagnoses a corpus nobody else can reproduce.
 *
 *  Three constraints, because this endpoint takes a filesystem path from a browser and runs a
 *  program on it.
 *
 *  It is off unless switched on. Absent, false, or malformed values leave it unavailable, the
 *  same way the recommendations gate behaves.
 *
 *  Arguments are passed as an array and never through a shell, so a path cannot carry a second
 *  command in it.
 *
 *  The resolved real path must sit inside a configured root. Resolved, so that `..` and symlinks
 *  are followed before the check rather than after, which is the usual way containment tests are
 *  defeated. */

/** The repository root. `next dev` runs from `web/`, the built server from the image's app dir. */
function repoRoot(): string {
  const cwd = process.cwd();
  return path.basename(cwd) === 'web' ? path.resolve(cwd, '..') : cwd;
}

function verdictBin(): string {
  return process.env.VERDICT_BIN || path.join(repoRoot(), '.venv', 'bin', 'verdict');
}

/** Paths outside this are refused. Defaults to the parent of the repository, so a release
 *  unpacked beside the checkout works without configuration while `/etc` does not. */
function ingestRoot(): string {
  return process.env.VERDICT_INGEST_ROOT || path.resolve(repoRoot(), '..');
}

function timeoutMs(): number {
  return Number(process.env.INGEST_TIMEOUT_MS ?? 900_000);
}

export interface IngestResult {
  ok: boolean;
  path: string;
  exitCode: number | null;
  durationMs: number;
  /** The tail of the command's combined output, which is what the operator needs to see. */
  output: string;
  error?: string;
}

export class IngestError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

/** Resolve a user-supplied path, or explain why it cannot be used. */
export function resolveTarget(raw: string): string {
  const trimmed = (raw ?? '').trim();
  if (!trimmed) throw new IngestError('A path is required', 400);

  const root = ingestRoot();
  const absolute = path.isAbsolute(trimmed) ? trimmed : path.resolve(root, trimmed);
  if (!existsSync(absolute)) throw new IngestError(`No such path: ${absolute}`, 400);

  // realpath before the containment test, so a symlink pointing out of the root is caught.
  const real = realpathSync(absolute);
  const realRoot = realpathSync(root);
  if (real !== realRoot && !real.startsWith(realRoot + path.sep)) {
    throw new IngestError(`Path is outside the permitted root (${realRoot})`, 403);
  }

  const stats = statSync(real);
  if (stats.isDirectory()) {
    if (!existsSync(path.join(real, 'ad_events.parquet'))) {
      throw new IngestError(`${real} does not contain ad_events.parquet`, 400);
    }
  } else if (!real.endsWith('.parquet')) {
    throw new IngestError('Expected a directory or a .parquet file', 400);
  }

  return real;
}

/** Where the engine's ingest endpoint lives, when the CLI is not on this machine.
 *
 *  Set in Compose, absent for host-side `next dev`. The two paths do the same thing -- both end
 *  up running `verdict ingest` -- and differ only in which container the process starts in. */
function ingestUrl(): string {
  return (process.env.VERDICT_INGEST_URL ?? '').trim();
}

/** Run the ingest, wherever the engine happens to be.
 *
 *  The path is validated on whichever side will open it. Over HTTP that is the engine container,
 *  which is the only one with the release mounted -- checking containment here against a
 *  filesystem that does not hold the file would reject every valid path and, worse, could pass
 *  one that resolves somewhere else over there. `serve.Runner.resolve` applies the same rules at
 *  the other end. */
export async function runIngest(raw: string): Promise<IngestResult> {
  const url = ingestUrl();
  return url ? postIngest(url, (raw ?? '').trim()) : spawnIngest(resolveTarget(raw));
}

/** Hand the job to the engine container and wait for its answer. */
async function postIngest(url: string, target: string): Promise<IngestResult> {
  const started = Date.now();
  const abort = AbortSignal.timeout(timeoutMs());
  try {
    const res = await fetch(`${url.replace(/\/$/, '')}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: target }),
      signal: abort,
      cache: 'no-store',
    });
    const body = await res.json();
    if (res.ok) return body as IngestResult;
    return {
      ok: false,
      path: target,
      exitCode: body?.exitCode ?? null,
      durationMs: body?.durationMs ?? Date.now() - started,
      output: body?.output ?? '',
      error: body?.error ?? `Ingest endpoint returned ${res.status}`,
    };
  } catch (err) {
    return {
      ok: false,
      path: target,
      exitCode: null,
      durationMs: Date.now() - started,
      output: '',
      error: `Could not reach the ingest endpoint at ${url}: ${(err as Error).message}`,
    };
  }
}

/** Run the CLI here, which is the host-side development path. */
function spawnIngest(target: string): Promise<IngestResult> {
  const bin = verdictBin();
  if (!existsSync(bin)) {
    return Promise.resolve({
      ok: false,
      path: target,
      exitCode: null,
      durationMs: 0,
      output: '',
      error: `The verdict CLI is not at ${bin}. Set VERDICT_BIN to its location.`,
    });
  }

  const started = Date.now();
  return new Promise<IngestResult>(resolve => {
    const child = spawn(bin, ['ingest', target], {
      cwd: repoRoot(),
      env: process.env,
      // No shell. A path is data here, never something that can carry a command.
      shell: false,
    });

    const chunks: string[] = [];
    let size = 0;
    const collect = (buf: Buffer) => {
      const text = buf.toString();
      size += text.length;
      chunks.push(text);
      // Keep the tail bounded; a verbose run should not be able to exhaust memory.
      while (size > 64_000 && chunks.length > 1) size -= chunks.shift()!.length;
    };
    child.stdout.on('data', collect);
    child.stderr.on('data', collect);

    const killer = setTimeout(() => child.kill('SIGKILL'), timeoutMs());

    const finish = (exitCode: number | null, error?: string) => {
      clearTimeout(killer);
      resolve({
        ok: exitCode === 0 && !error,
        path: target,
        exitCode,
        durationMs: Date.now() - started,
        output: chunks.join('').slice(-64_000),
        error,
      });
    };

    child.on('error', err => finish(null, err.message));
    child.on('close', code =>
      finish(code, code === 0 ? undefined : `verdict ingest exited with code ${code}`),
    );
  });
}
