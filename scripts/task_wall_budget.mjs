function parseEpochMs(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : NaN;
  if (typeof value !== 'string' || !value.trim()) return NaN;
  return Date.parse(value);
}

export function resolveTaskWallWindow({
  persistedStartedAt = '',
  maxTaskWallS,
  nowMs = Date.now(),
}) {
  if (!Number.isFinite(nowMs)) throw new Error('nowMs must be finite');
  if (!Number.isInteger(maxTaskWallS) || maxTaskWallS <= 0) {
    throw new Error('maxTaskWallS must be a positive integer');
  }
  const persistedMs = parseEpochMs(persistedStartedAt);
  const startedAtMs = Number.isFinite(persistedMs) && persistedMs <= nowMs
    ? persistedMs
    : nowMs;
  const deadlineMs = startedAtMs + maxTaskWallS * 1000;
  return {
    startedAtMs,
    startedAt: new Date(startedAtMs).toISOString(),
    deadlineMs,
    deadlineAt: new Date(deadlineMs).toISOString(),
    remainingMs: Math.max(0, deadlineMs - nowMs),
    expired: nowMs >= deadlineMs,
  };
}
