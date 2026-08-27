import assert from 'node:assert/strict';
import test from 'node:test';

import { buildRecordingAttemptAudit } from '../scripts/recording_attempt_audit.mjs';

const tasks = [{ id: 'T1' }];
const failedState = {
  results: {},
  failures: { T1: { status: 'failed', attempts: 12 } },
  attempts: { T1: 12 },
  recovery_batch_count: 1,
  recovery_batches: [{ batch_index: 1, tasks: [{ task_id: 'T1' }] }],
};

test('a configured second recovery batch remains bounded and auditable', () => {
  const audit = buildRecordingAttemptAudit({
    tasks,
    state: failedState,
    eventsPath: '/path/that/does/not/exist',
    maxRetriesPerBatch: 5,
    maxRecoveryBatches: 2,
  });

  assert.equal(audit.recovery_batch_count, 1);
  assert.equal(audit.additional_recovery_batch_allowed, true);
  assert.equal(audit.policy.max_failed_task_recovery_batches, 2);
});

test('no recovery batch is allowed after the configured limit', () => {
  const audit = buildRecordingAttemptAudit({
    tasks,
    state: {
      ...failedState,
      recovery_batch_count: 2,
      recovery_batches: [
        { batch_index: 1, tasks: [{ task_id: 'T1' }] },
        { batch_index: 2, tasks: [{ task_id: 'T1' }] },
      ],
    },
    eventsPath: '/path/that/does/not/exist',
    maxRetriesPerBatch: 5,
    maxRecoveryBatches: 2,
  });

  assert.equal(audit.additional_recovery_batch_allowed, false);
});

test('a global later batch does not consume another task recovery budget', () => {
  const audit = buildRecordingAttemptAudit({
    tasks,
    state: {
      ...failedState,
      recovery_batch_count: 2,
      recovery_batches: [
        { batch_index: 1, tasks: [{ task_id: 'T1' }] },
        { batch_index: 2, tasks: [{ task_id: 'another-task' }] },
      ],
    },
    eventsPath: '/path/that/does/not/exist',
    maxRetriesPerBatch: 5,
    maxRecoveryBatches: 2,
  });

  assert.equal(audit.tasks[0].recovery_batch_count, 1);
  assert.equal(audit.additional_recovery_batch_allowed, true);
});
