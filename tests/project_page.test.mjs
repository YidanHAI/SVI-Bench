import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const docsRoot = path.join(repositoryRoot, 'docs');

async function read(relativePath) {
  return fsp.readFile(path.join(repositoryRoot, relativePath), 'utf8');
}

test('project page exposes the benchmark, results, workflow, and citation sections', async () => {
  const html = await read('docs/index.html');
  for (const id of ['benchmark', 'protocol', 'scoring', 'results', 'alignment', 'interactflow', 'reproduce', 'citation']) {
    assert.match(html, new RegExp(`id=["']${id}["']`));
  }
  assert.match(html, /53\.61/);
  assert.match(html, /min\(D1<sub>i<\/sub>, D2<sub>i<\/sub>\)/);
  assert.match(html, /No RTSP conversion/);
});

test('every local project-page asset reference resolves inside docs', async () => {
  const html = await read('docs/index.html');
  const references = [...html.matchAll(/(?:src|href)=["']([^"']+)["']/g)]
    .map((match) => match[1])
    .filter((reference) => !reference.startsWith('#') && !/^https?:\/\//.test(reference));

  assert.ok(references.length >= 8);
  for (const reference of references) {
    const cleanPath = reference.split('#', 1)[0].split('?', 1)[0];
    const resolved = path.resolve(docsRoot, cleanPath);
    assert.ok(resolved.startsWith(`${docsRoot}${path.sep}`), `reference escapes docs: ${reference}`);
    await fsp.access(resolved);
  }
});

test('project page keeps formal leaderboard values and diagnostic labels distinct', async () => {
  const script = await read('docs/app.js');
  const expected = [
    ['JoyAI-VL-Interaction', '53.61', '39.67'],
    ['MOSS-VL-Realtime', '25.44', '26.33'],
    ['MiniCPM-O-4.5-9B', '21.33', '20.44'],
    ['Doubao Seed 2.1 Pro', '21.22', '29.44'],
    ['Mage-VL', '19.5', '21.78'],
  ];

  for (const [name, overall, withoutD3] of expected) {
    assert.ok(script.includes(`name: "${name}"`));
    assert.ok(script.includes(`overall: ${overall}`));
    assert.ok(script.includes(`withoutD3: ${withoutD3}`));
  }
  assert.match(script, /Diagnostic only: this view does not replace or rerank the official leaderboard/);
});

test('project page contains no private endpoint or credential literal', async () => {
  const contents = await Promise.all([
    read('docs/index.html'),
    read('docs/styles.css'),
    read('docs/app.js'),
  ]);
  const joined = contents.join('\n');

  assert.doesNotMatch(joined, /hubrouter\.jd\.com/i);
  assert.doesNotMatch(joined, /180\.184\.148\.170|10\.119\.100\.169/);
  assert.doesNotMatch(joined, /(?:sk-|ghp_)[A-Za-z0-9_-]{16,}/);
});

test('project page is self-contained for branch-based GitHub Pages deployment', async () => {
  await fsp.access(path.join(docsRoot, 'index.html'));
  await fsp.access(path.join(docsRoot, '.nojekyll'));
  await fsp.access(path.join(docsRoot, 'robots.txt'));
  await fsp.access(path.join(docsRoot, 'sitemap.xml'));
});
