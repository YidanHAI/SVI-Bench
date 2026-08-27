import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

import {
  jpegDimensions,
  normalizeJpegDataUrlToReference,
} from '../scripts/jpeg_frame_normalizer.mjs';

function jpeg(width, height, color) {
  const result = spawnSync('ffmpeg', [
    '-loglevel', 'error',
    '-f', 'lavfi',
    '-i', `color=c=${color}:s=${width}x${height},format=yuv444p,format=yuv420p`,
    '-frames:v', '1',
    '-f', 'image2pipe',
    '-vcodec', 'mjpeg',
    'pipe:1',
  ]);
  assert.equal(result.status, 0, result.stderr.toString('utf8'));
  return result.stdout;
}

test('Mage frames are normalized to the session reference-frame size', async () => {
  const source = jpeg(80, 40, 'red');
  const reference = jpeg(32, 24, 'blue');
  const result = await normalizeJpegDataUrlToReference(
    `data:image/jpeg;base64,${source.toString('base64')}`,
    `data:image/jpeg;base64,${reference.toString('base64')}`,
  );

  assert.equal(result.normalized, true);
  assert.deepEqual(result.sourceSize, { width: 80, height: 40 });
  assert.deepEqual(result.targetSize, { width: 32, height: 24 });
  const output = Buffer.from(result.imageUrl.split(',', 2)[1], 'base64');
  assert.deepEqual(jpegDimensions(output), { width: 32, height: 24 });
});

test('same-sized Query frames are passed through byte-for-byte', async () => {
  const source = jpeg(32, 24, 'green');
  const dataUrl = `data:image/jpeg;base64,${source.toString('base64')}`;
  const result = await normalizeJpegDataUrlToReference(dataUrl, dataUrl);

  assert.equal(result.normalized, false);
  assert.equal(result.imageUrl, dataUrl);
});

test('odd source dimensions never exceed the even reference canvas', async () => {
  const source = jpeg(359, 638, 'red');
  const reference = jpeg(358, 640, 'blue');
  const result = await normalizeJpegDataUrlToReference(
    `data:image/jpeg;base64,${source.toString('base64')}`,
    `data:image/jpeg;base64,${reference.toString('base64')}`,
  );

  assert.equal(result.normalized, true);
  const output = Buffer.from(result.imageUrl.split(',', 2)[1], 'base64');
  assert.deepEqual(jpegDimensions(output), { width: 358, height: 640 });
});

test('normalization preserves an odd-width reference canvas', async () => {
  const source = jpeg(358, 640, 'red');
  const reference = jpeg(359, 640, 'blue');
  const result = await normalizeJpegDataUrlToReference(
    `data:image/jpeg;base64,${source.toString('base64')}`,
    `data:image/jpeg;base64,${reference.toString('base64')}`,
  );

  assert.equal(result.normalized, true);
  const output = Buffer.from(result.imageUrl.split(',', 2)[1], 'base64');
  assert.deepEqual(jpegDimensions(output), { width: 359, height: 640 });
});
