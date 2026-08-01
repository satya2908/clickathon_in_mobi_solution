#!/usr/bin/env node
/** Point every `resolved` URL in package-lock.json at public npm.
 *
 *  npm stamps the lockfile with whichever registry the installing machine was configured for.
 *  On a corporate network that is an internal Artifactory, and a lockfile full of internal
 *  URLs only installs on the laptop that produced it -- `npm ci` inside a build container
 *  fails on a certificate it has no reason to trust.
 *
 *  Pinning the registry in .npmrc is the obvious fix and does not work here: this machine
 *  cannot reach registry.npmjs.org directly, because the proxy that provides the Artifactory
 *  mirror also intercepts TLS. So the install goes through whatever the developer's global
 *  config says, and the artifact that gets committed is normalised afterwards.
 *
 *  Safe rather than a workaround: `integrity` is a hash of the tarball contents, not of where
 *  it came from. `npm ci` still verifies it received byte-for-byte the package the lockfile
 *  pinned. Versions stay exact; only the host changes.
 *
 *  Runs automatically after `npm install` via the `postinstall` hook.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const PUBLIC = 'https://registry.npmjs.org';
const lockfile = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'package-lock.json');

let text;
try {
  text = readFileSync(lockfile, 'utf8');
} catch {
  // No lockfile yet (a bare `npm install` in a fresh clone). Nothing to normalise.
  process.exit(0);
}

// Matches `<registry>/<any path>/<package>/-/<file>.tgz`, keeping the trailing package path
// that every npm-compatible registry shares and replacing only the host and mirror prefix.
const rewritten = text.replace(
  /"resolved": "https?:\/\/[^"]*?\/((?:@[^/"]+\/)?[^/"]+\/-\/[^"]+)"/g,
  `"resolved": "${PUBLIC}/$1"`,
);

if (rewritten !== text) {
  writeFileSync(lockfile, rewritten);
  const n = (rewritten.match(/registry\.npmjs\.org/g) ?? []).length;
  console.log(`package-lock.json: ${n} resolved URLs pointed at public npm`);
}
