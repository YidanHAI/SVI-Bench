import fs from 'node:fs';

function readJsonl(filePath) {
  try {
    return fs.readFileSync(filePath, 'utf8').split(/\r?\n/).filter(Boolean).flatMap((line) => {
      try { return [JSON.parse(line)]; } catch { return []; }
    });
  } catch {
    return [];
  }
}

function taskStatus(state, taskId) {
  if (state.results?.[taskId]?.status === 'ok') return 'ok';
  if (state.failures?.[taskId]?.status === 'failed') return 'failed';
  if (state.current_task?.task_id === taskId) return 'running';
  return 'pending';
}

export function buildRecordingAttemptAudit({
  tasks,
  state,
  eventsPath,
  maxRetriesPerBatch = 5,
  maxRecoveryBatches = 1,
}) {
  const events = readJsonl(eventsPath);
  const failuresByTask = new Map();
  const completionsByTask = new Map();
  const exhaustedByTask = new Map();
  const recoveryByTask = new Map();
  const infrastructureFailures = [];

  const append = (mapping, taskId, record) => {
    if (!taskId) return;
    const records = mapping.get(taskId) || [];
    records.push(record);
    mapping.set(taskId, records);
  };

  for (const event of events) {
    if (event.event === 'task_attempt_failed') {
      append(failuresByTask, event.task_id, {
        timestamp: event.timestamp,
        attempt: Number(event.attempt || 0),
        batch_attempt_limit: Number(event.max_attempts || 0),
        reason: String(event.error || ''),
      });
    } else if (event.event === 'task_complete') {
      append(completionsByTask, event.task_id, {
        timestamp: event.timestamp,
        attempt: Number(event.attempt || 0),
      });
    } else if (event.event === 'task_failed_exhausted') {
      append(exhaustedByTask, event.task_id, {
        timestamp: event.timestamp,
        cumulative_attempts: Number(event.attempts || 0),
        reason: String(event.error || ''),
      });
    } else if (event.event === 'failed_task_backfill_scheduled') {
      const used = Number(event.attempts_already_used || 0);
      const allowed = Number(event.additional_attempts_allowed || 0);
      append(recoveryByTask, event.task_id, {
        timestamp: event.timestamp,
        attempt_start: used + 1,
        attempt_end: used + allowed,
        attempts_allowed: allowed,
      });
    } else if (event.event === 'route_recovery_failed') {
      infrastructureFailures.push({
        timestamp: event.timestamp,
        reason: String(event.reason || ''),
        error: String(event.error || ''),
      });
    }
  }

  const hasRecordedRecovery = [...recoveryByTask.values()].some((records) => records.length > 0)
    || Object.values(state.failure_history || {}).some((records) => (
      Array.isArray(records) && records.length > 0
    ));
  const recoveryBatchCount = Math.max(
    Number(state.recovery_batch_count || 0),
    hasRecordedRecovery ? 1 : 0,
  );

  const taskAudits = tasks.map((task, index) => {
    const taskId = task.id;
    const attemptFailures = failuresByTask.get(taskId) || [];
    const completions = completionsByTask.get(taskId) || [];
    const exhaustedEvents = exhaustedByTask.get(taskId) || [];
    const recoveryBatches = recoveryByTask.get(taskId) || [];
    const observedAttempts = [
      Number(state.attempts?.[taskId] || 0),
      Number(state.results?.[taskId]?.attempts || 0),
      Number(state.failures?.[taskId]?.attempts || 0),
      Number(state.current_task?.task_id === taskId ? state.current_task.attempt : 0),
      ...attemptFailures.map((item) => item.attempt),
      ...completions.map((item) => item.attempt),
    ];
    const cumulativeAttempts = Math.max(0, ...observedAttempts);
    const currentLimit = Number(state.retry_attempt_limits?.[taskId] || 0)
      || Number(state.current_task?.task_id === taskId ? state.current_task.max_attempts : 0)
      || Math.max(maxRetriesPerBatch + 1, cumulativeAttempts);
    const taskRecoveryBatchCount = Array.isArray(state.recovery_batches)
      ? state.recovery_batches.filter((batch) => (
        Array.isArray(batch?.tasks)
        && batch.tasks.some((item) => item?.task_id === taskId)
      )).length
      : recoveryBatches.length;
    const passStartedAtMs = Date.parse(state.task_wall_pass_started_at?.[taskId] || '');
    const maxTaskWallS = Number(state.max_task_wall_s || 0);
    const taskWallWindowOpen = !Number.isFinite(passStartedAtMs)
      || maxTaskWallS <= 0
      || Date.now() < passStartedAtMs + maxTaskWallS * 1000;
    return {
      task_index: index + 1,
      task_id: taskId,
      status: taskStatus(state, taskId),
      cumulative_attempts: cumulativeAttempts,
      current_cumulative_attempt_limit: currentLimit,
      initial_batch: {
        attempt_start: 1,
        attempt_end: maxRetriesPerBatch + 1,
        max_retries: maxRetriesPerBatch,
      },
      recovery_batches: recoveryBatches,
      recovery_batch_count: taskRecoveryBatchCount,
      existing_recovery_continuation_available: currentLimit > cumulativeAttempts
        && taskWallWindowOpen,
      additional_recovery_batch_allowed: taskRecoveryBatchCount < maxRecoveryBatches,
      attempt_failures: attemptFailures,
      completion_events: completions,
      exhausted_events: exhaustedEvents,
    };
  });

  return {
    generated_at: new Date().toISOString(),
    policy: {
      max_retries_per_batch: maxRetriesPerBatch,
      attempts_per_batch: maxRetriesPerBatch + 1,
      max_failed_task_recovery_batches: maxRecoveryBatches,
      automatic_limit_growth: false,
    },
    recovery_batch_count: recoveryBatchCount,
    additional_recovery_batch_allowed: taskAudits.some((task) => (
      task.status === 'failed'
      && (
        task.existing_recovery_continuation_available
        || task.additional_recovery_batch_allowed
      )
    )),
    tasks: taskAudits,
    infrastructure_failures: infrastructureFailures,
  };
}
