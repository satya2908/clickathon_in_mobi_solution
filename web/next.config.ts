import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { NextConfig } from 'next';

/** Reads the repo-root `.env` into the process, mirroring `read_dotenv` on the Python side.
 *
 *  Next only looks for env files beside `package.json`, which would mean a second copy of
 *  the ClickHouse credentials living in `web/`. Two copies of a connection string drift,
 *  and the drift shows up as a console reading a different database than the engine wrote
 *  to -- which looks like missing data, not like misconfiguration.
 *
 *  Anything already in the environment wins, so Docker and CI keep control. */
function loadRootEnv() {
  // In the container the app is copied without its parent, so the root file is simply
  // absent and the real environment is authoritative.
  for (const path of [resolve(process.cwd(), '../.env'), resolve(process.cwd(), '.env')]) {
    let text: string;
    try {
      text = readFileSync(path, 'utf8');
    } catch {
      continue;
    }
    for (const raw of text.split('\n')) {
      const line = raw.trim();
      if (!line || line.startsWith('#')) continue;
      const eq = line.indexOf('=');
      if (eq < 0) continue;
      const key = line.slice(0, eq).replace(/^export\s+/, '').trim();
      if (!key || key in process.env) continue;
      process.env[key] = line
        .slice(eq + 1)
        .trim()
        .replace(/^(['"])(.*)\1$/, '$2');
    }
  }
}

loadRootEnv();

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Self-contained server bundle, so the image does not need node_modules at runtime.
  output: 'standalone',
};

export default nextConfig;
