import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveTaskWallWindow } from '../scripts/task_wall_budget.mjs';

test('a worker restart preserves the original task deadline', () => {
  const first = resolveTaskWallWindow({
    maxTaskWallS: 3600,
    nowMs: Date.parse('2026-08-21T00:00:00.000Z'),
  });
  const resumed = resolveTaskWallWindow({
    persistedStartedAt: first.startedAt,
    maxTaskWallS: 3600,
    nowMs: Date.parse('2026-08-21T00:40:00.000Z'),
  });

  assert.equal(resumed.startedAt, first.startedAt);
  assert.equal(resumed.deadlineAt, first.deadlineAt);
  assert.equal(resumed.remainingMs, 20 * 60 * 1000);
  assert.equal(resumed.expired, false);
});

test('a resumed task is expired once its cumulative hour is consumed', () => {
  const window = resolveTaskWallWindow({
    persistedStartedAt: '2026-08-21T00:00:00.000Z',
    maxTaskWallS: 3600,
    nowMs: Date.parse('2026-08-21T01:00:00.001Z'),
  });

  assert.equal(window.remainingMs, 0);
  assert.equal(window.expired, true);
});

test('a new recovery pass receives a fresh wall window', () => {
  const window = resolveTaskWallWindow({
    persistedStartedAt: '',
    maxTaskWallS: 3600,
    nowMs: Date.parse('2026-08-21T02:00:00.000Z'),
  });

  assert.equal(window.startedAt, '2026-08-21T02:00:00.000Z');
  assert.equal(window.deadlineAt, '2026-08-21T03:00:00.000Z');
});
