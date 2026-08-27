import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { promoteTaskDirectory } from '../scripts/artifact_promotion.mjs';

async function fixture() {
  const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'artifact-promotion-test-'));
  const source = path.join(root, 'scratch', 'task');
  const destination = path.join(root, 'output', 'task');
  await fsp.mkdir(source, { recursive: true });
  await fsp.mkdir(path.dirname(destination), { recursive: true });
  await fsp.writeFile(path.join(source, 'task.mp4'), 'validated-video-bytes');
  await fsp.writeFile(path.join(source, 'summary.json'), '{}');
  return { root, source, destination };
}

test('copy promotion verifies and atomically publishes a task directory', async (t) => {
  const { root, source, destination } = await fixture();
  t.after(() => fsp.rm(root, { recursive: true, force: true }));

  const result = await promoteTaskDirectory({
    sourceDir: source,
    destinationDir: destination,
    relativeVideoPath: 'task.mp4',
    forceCopy: true,
    validateVideo: async (videoPath) => ({
      ok: (await fsp.readFile(videoPath, 'utf8')) === 'validated-video-bytes',
      decode_ok: true,
      probe: { duration_s: 1 },
    }),
  });

  assert.equal(result.method, 'copy-verify-rename');
  assert.match(result.video_sha256, /^[a-f0-9]{64}$/);
  await assert.rejects(fsp.stat(source), { code: 'ENOENT' });
  assert.equal(await fsp.readFile(path.join(destination, 'task.mp4'), 'utf8'), 'validated-video-bytes');
});

test('failed copied-video validation never publishes the destination', async (t) => {
  const { root, source, destination } = await fixture();
  t.after(() => fsp.rm(root, { recursive: true, force: true }));

  await assert.rejects(
    promoteTaskDirectory({
      sourceDir: source,
      destinationDir: destination,
      relativeVideoPath: 'task.mp4',
      forceCopy: true,
      validateVideo: async () => ({ ok: false, error: 'decode error' }),
    }),
    /decode error/,
  );
  assert.equal(await fsp.readFile(path.join(source, 'task.mp4'), 'utf8'), 'validated-video-bytes');
  await assert.rejects(fsp.stat(destination), { code: 'ENOENT' });
});
