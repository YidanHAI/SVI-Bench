import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function run(command, args, { stderrLimit = 4 * 1024 * 1024 } = {}) {
  return new Promise((resolve) => {
    const proc = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (chunk) => {
      stdout = `${stdout}${chunk}`.slice(-stderrLimit);
    });
    proc.stderr.on('data', (chunk) => {
      stderr = `${stderr}${chunk}`.slice(-stderrLimit);
    });
    proc.once('error', (error) => resolve({ code: null, signal: null, stdout, stderr, error }));
    proc.once('close', (code, signal) => resolve({ code, signal, stdout, stderr, error: null }));
  });
}

function parseMarkerSignalStats(text) {
  const rows = [];
  let current = null;
  for (const line of String(text || '').split(/\r?\n/)) {
    const frameMatch = line.match(/^frame:\s*(\d+).*\bpts_time:([0-9.+-]+)/);
    if (frameMatch) {
      current = {
        frame: Number(frameMatch[1]),
        pts_s: Number(frameMatch[2]),
      };
      continue;
    }
    const valueMatch = line.match(/^lavfi\.signalstats\.VAVG=([0-9.+-]+)/);
    if (!valueMatch || !current) continue;
    rows.push({ ...current, v_avg: Number(valueMatch[1]) });
    current = null;
  }
  return rows.filter((row) => Number.isFinite(row.pts_s) && Number.isFinite(row.v_avg));
}

export async function calibrateWebmClockFromVisualMarker({
  input,
  markerRemovedEventOffsetS,
  videoWidth,
  videoHeight,
  markerSize = 48,
  markerX = 0,
  markerY = null,
  markerVMax = 80,
  clearVMin = 105,
  minimumMarkerRunS = 0.2,
  scanPaddingS = 2,
  ffmpegBin = 'ffmpeg',
}) {
  const removedOffsetS = Number(markerRemovedEventOffsetS);
  const width = Number(videoWidth);
  const height = Number(videoHeight);
  const size = Number(markerSize);
  const x = Number(markerX);
  const y = markerY == null ? height - size : Number(markerY);
  if (
    !input
    || !Number.isFinite(removedOffsetS)
    || !Number.isInteger(width)
    || !Number.isInteger(height)
    || !Number.isInteger(size)
    || !Number.isInteger(x)
    || !Number.isInteger(y)
    || size < 8
    || x < 0
    || y < 0
    || x + size > width
    || y + size > height
  ) {
    return { ok: false, error: 'Invalid WebM visual-clock calibration inputs' };
  }

  const scanDurationS = Math.max(1, removedOffsetS + Number(scanPaddingS));
  const filter = [
    `crop=${size}:${size}:${x}:${y}`,
    'signalstats',
    'metadata=print:key=lavfi.signalstats.VAVG:file=-',
  ].join(',');
  const result = await run(ffmpegBin, [
    '-v', 'error',
    '-i', input,
    '-t', scanDurationS.toFixed(3),
    '-vf', filter,
    '-an',
    '-f', 'null',
    '-',
  ]);
  if (result.code !== 0) {
    return {
      ok: false,
      error: result.error?.message
        || `Unable to scan WebM clock marker: ${result.stderr.trim().slice(-1000)}`,
    };
  }

  const rows = parseMarkerSignalStats(result.stdout);
  if (rows.length < 2) {
    return { ok: false, error: 'WebM clock-marker scan produced too few video frames' };
  }
  const frameSteps = rows.slice(1).map((row, index) => row.pts_s - rows[index].pts_s)
    .filter((value) => value > 0 && value < 1)
    .sort((a, b) => a - b);
  const frameStepS = frameSteps.length
    ? frameSteps[Math.floor(frameSteps.length / 2)]
    : 0.04;
  const runs = [];
  let active = null;
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    if (row.v_avg <= markerVMax) {
      if (!active || row.pts_s - active.last_pts_s > Math.max(0.12, frameStepS * 2.5)) {
        if (active) runs.push(active);
        active = {
          start_index: index,
          end_index: index,
          start_pts_s: row.pts_s,
          end_pts_s: row.pts_s,
          last_pts_s: row.pts_s,
          frames: 1,
        };
      } else {
        active.end_index = index;
        active.end_pts_s = row.pts_s;
        active.last_pts_s = row.pts_s;
        active.frames += 1;
      }
    } else if (active) {
      runs.push(active);
      active = null;
    }
  }
  if (active) runs.push(active);
  const markerRuns = runs.filter((run) => (
    run.end_pts_s - run.start_pts_s + frameStepS >= minimumMarkerRunS
  ));
  const markerRun = markerRuns[markerRuns.length - 1];
  if (!markerRun) {
    return {
      ok: false,
      error: 'The pre-playback visual clock marker was not found in the WebM',
      frames_scanned: rows.length,
      scan_duration_s: scanDurationS,
    };
  }
  const transition = rows.slice(markerRun.end_index + 1)
    .find((row) => row.v_avg >= clearVMin);
  if (!transition) {
    return {
      ok: false,
      error: 'The visual clock marker never disappeared before the WebM scan ended',
      frames_scanned: rows.length,
      scan_duration_s: scanDurationS,
      marker_run: markerRun,
    };
  }

  const eventClockMinusWebmPtsS = removedOffsetS - transition.pts_s;
  if (eventClockMinusWebmPtsS < -0.25 || eventClockMinusWebmPtsS > 30) {
    return {
      ok: false,
      error: `Implausible event/WebM clock offset: ${eventClockMinusWebmPtsS.toFixed(3)}s`,
      marker_transition_pts_s: transition.pts_s,
      marker_removed_event_offset_s: removedOffsetS,
    };
  }
  return {
    ok: true,
    protocol: 'pre_playback_visual_marker_v1',
    event_clock_minus_webm_pts_s: eventClockMinusWebmPtsS,
    marker_removed_event_offset_s: removedOffsetS,
    marker_transition_pts_s: transition.pts_s,
    marker_run: {
      start_pts_s: markerRun.start_pts_s,
      end_pts_s: markerRun.end_pts_s,
      frames: markerRun.frames,
    },
    marker_region: { x, y, size },
    frame_step_s: frameStepS,
    frames_scanned: rows.length,
    scan_duration_s: scanDurationS,
  };
}

export function mergeIntervals(intervals, minDurationS = 0.001) {
  const sorted = intervals
    .map((item) => ({
      ...item,
      start_s: Number(item.start_s),
      end_s: Number(item.end_s),
    }))
    .filter((item) => (
      Number.isFinite(item.start_s)
      && Number.isFinite(item.end_s)
      && item.end_s - item.start_s >= minDurationS
    ))
    .sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s);
  const merged = [];
  for (const item of sorted) {
    const previous = merged[merged.length - 1];
    if (!previous || item.start_s > previous.end_s + 0.001) {
      merged.push({ ...item, reasons: item.reasons || [item.reason].filter(Boolean) });
      continue;
    }
    previous.end_s = Math.max(previous.end_s, item.end_s);
    previous.reasons = [...new Set([
      ...(previous.reasons || []),
      ...(item.reasons || [item.reason].filter(Boolean)),
    ])];
  }
  return merged;
}

export function deriveOutageIntervalsFromEvents(eventRows) {
  const intervals = [];
  let active = null;
  let lastProgress = null;

  const begin = (startMs, reason, sourcePositionS = null) => {
    if (active) {
      if (startMs < active.start_ms) active.start_ms = startMs;
      if (reason && !active.reasons.includes(reason)) active.reasons.push(reason);
      return;
    }
    active = {
      start_ms: startMs,
      start_source_position_s: sourcePositionS,
      reasons: reason ? [reason] : [],
    };
  };
  const end = (endMs, reason) => {
    if (!active || endMs <= active.start_ms) return;
    intervals.push({
      start_s: active.start_ms / 1000,
      end_s: endMs / 1000,
      start_source_position_s: active.start_source_position_s,
      reasons: [...new Set([...active.reasons, reason].filter(Boolean))],
    });
    active = null;
  };

  for (const event of eventRows) {
    const tMs = Number(event.t_ms);
    if (!Number.isFinite(tMs)) continue;
    const sourcePositionS = Number(event.source_position_s);
    if (event.type === 'local_rtsp_relay_progress' && Number.isFinite(sourcePositionS)) {
      lastProgress = { t_ms: tMs, source_position_s: sourcePositionS };
      continue;
    }
    if (
      event.type === 'local_rtsp_relay_failed'
      || event.type === 'local_rtsp_relay_output_stalled'
    ) {
      let startMs = tMs;
      if (lastProgress && Number.isFinite(sourcePositionS)) {
        const sourceDeltaMs = Math.max(0, sourcePositionS - lastProgress.source_position_s) * 1000;
        startMs = Math.min(tMs, lastProgress.t_ms + sourceDeltaMs);
      }
      begin(startMs, event.type, Number.isFinite(sourcePositionS) ? sourcePositionS : null);
      continue;
    }
    if (event.type === 'capture_stream_disconnected') {
      begin(tMs, event.type, Number.isFinite(sourcePositionS) ? sourcePositionS : null);
      continue;
    }
    if (
      event.type === 'local_rtsp_publisher_timeline_resumed'
      || event.type === 'local_rtsp_relay_resumed'
      || event.type === 'capture_stream_reconnect_ready'
    ) {
      end(tMs, event.type);
    }
  }
  return mergeIntervals(intervals);
}

export function readJsonl(filePath) {
  return fs.readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

export function buildKeepSegments(trimStartS, trimEndS, outageIntervals) {
  const start = Math.max(0, Number(trimStartS));
  const end = Math.max(start, Number(trimEndS));
  const outages = mergeIntervals(outageIntervals)
    .map((item) => ({
      ...item,
      start_s: Math.max(start, item.start_s),
      end_s: Math.min(end, item.end_s),
    }))
    .filter((item) => item.end_s - item.start_s > 0.001);
  const segments = [];
  let cursor = start;
  for (const outage of outages) {
    if (outage.start_s - cursor > 0.01) {
      segments.push({ start_s: cursor, end_s: outage.start_s });
    }
    cursor = Math.max(cursor, outage.end_s);
  }
  if (end - cursor > 0.01) segments.push({ start_s: cursor, end_s: end });
  return { segments, outages };
}

export async function renderTaskVideo({
  input,
  output,
  trimStartS,
  trimEndS,
  outageIntervals = [],
  ffmpegBin = 'ffmpeg',
}) {
  const { segments, outages } = buildKeepSegments(trimStartS, trimEndS, outageIntervals);
  if (!segments.length) {
    return { ok: false, error: 'No playable interval remains after outage removal', segments, outages };
  }
  const tempOutput = `${output}.tmp-${process.pid}-${Date.now()}.mp4`;
  const filters = segments.map((segment, index) => (
    `[0:v]trim=start=${segment.start_s.toFixed(6)}:end=${segment.end_s.toFixed(6)},`
    + `setpts=PTS-STARTPTS[v${index}]`
  ));
  const concatInputs = segments.map((_, index) => `[v${index}]`).join('');
  filters.push(`${concatInputs}concat=n=${segments.length}:v=1:a=0[outv]`);
  const result = await run(ffmpegBin, [
    '-y', '-i', input,
    '-filter_complex', filters.join(';'),
    '-map', '[outv]',
    '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '8',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    tempOutput,
  ]);
  if (result.code !== 0) {
    await fsp.unlink(tempOutput).catch(() => {});
    const stderrTail = result.stderr.trim().split('\n').slice(-40).join('\n');
    return {
      ok: false,
      error: result.error?.message
        || `ffmpeg task render failed (code=${result.code}, signal=${result.signal || 'none'}): ${stderrTail}`,
      segments,
      outages,
    };
  }
  try {
    const handle = await fsp.open(tempOutput, 'r');
    try {
      await handle.sync();
    } finally {
      await handle.close();
    }
  } catch (error) {
    await fsp.unlink(tempOutput).catch(() => {});
    return {
      ok: false,
      error: `ffmpeg task render fsync failed: ${error.message}`,
      segments,
      outages,
    };
  }
  await fsp.rename(tempOutput, output);
  return {
    ok: true,
    segments,
    outages,
    removed_duration_s: outages.reduce((sum, item) => sum + item.end_s - item.start_s, 0),
    output_duration_s: segments.reduce((sum, item) => sum + item.end_s - item.start_s, 0),
  };
}

function parseIntervals(stderr, prefix) {
  const regex = new RegExp(
    `${prefix}_start:(-?[0-9.]+)\\s+${prefix}_end:(-?[0-9.]+)\\s+${prefix}_duration:([0-9.]+)`,
    'g',
  );
  return [...stderr.matchAll(regex)].map((match) => ({
    start_s: Number(match[1]),
    end_s: Number(match[2]),
    duration_s: Number(match[3]),
  }));
}

export async function probeVideo(videoPath, ffprobeBin = 'ffprobe') {
  const result = await run(ffprobeBin, [
    '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height,codec_name:format=duration,size',
    '-of', 'json', videoPath,
  ]);
  if (result.code !== 0) {
    throw new Error(result.error?.message || result.stderr.trim() || `ffprobe failed for ${videoPath}`);
  }
  const data = JSON.parse(result.stdout);
  const stream = data.streams?.[0] || {};
  return {
    duration_s: Number(data.format?.duration),
    size_bytes: Number(data.format?.size),
    width: Number(stream.width),
    height: Number(stream.height),
    codec_name: stream.codec_name || '',
  };
}

export async function validateVideoDecode({
  videoPath,
  ffmpegBin = 'ffmpeg',
  ffprobeBin = 'ffprobe',
  decodeAttempts = 3,
  retryDelayMs = 1000,
}) {
  const attempts = Math.max(1, Math.trunc(Number(decodeAttempts) || 1));
  let latest = null;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    let probe = null;
    try {
      probe = await probeVideo(videoPath, ffprobeBin);
    } catch (error) {
      latest = {
        ok: false,
        decode_ok: false,
        probe: null,
        error: error.message,
        attempts_used: attempt,
      };
    }
    if (probe) {
      const decode = await run(ffmpegBin, [
        '-hide_banner', '-v', 'error', '-xerror', '-i', videoPath,
        '-map', '0:v:0', '-an', '-f', 'null', '-',
      ]);
      const error = decode.code === 0
        ? ''
        : decode.error?.message || decode.stderr.split('\n').filter(Boolean).slice(-8).join('\n');
      latest = {
        ok: decode.code === 0,
        decode_ok: decode.code === 0,
        probe,
        error,
        attempts_used: attempt,
      };
      if (latest.ok) return latest;
    }
    if (attempt < attempts) await sleep(retryDelayMs);
  }
  return latest;
}

export function calculateMaximumTaskVideoDurationS({
  minimumDurationS,
  responseWaitRequired,
  responseTimeoutS,
  postResponseRecordingS,
  maxExtraS,
}) {
  const responseTailS = responseWaitRequired
    ? responseTimeoutS + postResponseRecordingS
    : 0;
  return minimumDurationS + responseTailS + maxExtraS;
}

export async function validateTaskVideo({
  videoPath,
  expectedDurationS,
  minimumDurationS = null,
  maximumDurationS = null,
  queryEvents = [],
  expectedQueries = [],
  ffmpegBin = 'ffmpeg',
  ffprobeBin = 'ffprobe',
  durationToleranceS = 1.5,
  queryTimeToleranceS = 0.5,
  maxBlackDurationS = 2,
  requiredRecordedUntilS = null,
  requiredRecordedUntilToleranceS = 0.25,
}) {
  const errors = [];
  const warnings = [];
  let probe = null;
  try {
    probe = await probeVideo(videoPath, ffprobeBin);
  } catch (error) {
    return { ok: false, errors: [error.message], warnings, probe: null };
  }
  const durationErrorS = Math.abs(probe.duration_s - expectedDurationS);
  if (!Number.isFinite(probe.duration_s) || durationErrorS > durationToleranceS) {
    errors.push(
      `task video duration ${probe.duration_s.toFixed(3)}s differs from expected `
      + `${expectedDurationS.toFixed(3)}s by ${durationErrorS.toFixed(3)}s`,
    );
  }
  if (Number.isFinite(minimumDurationS) && probe.duration_s < minimumDurationS - durationToleranceS) {
    errors.push(
      `task video duration ${probe.duration_s.toFixed(3)}s is shorter than minimum `
      + `${minimumDurationS.toFixed(3)}s`,
    );
  }
  if (Number.isFinite(maximumDurationS) && probe.duration_s > maximumDurationS + durationToleranceS) {
    errors.push(
      `task video duration ${probe.duration_s.toFixed(3)}s exceeds maximum `
      + `${maximumDurationS.toFixed(3)}s`,
    );
  }
  if (
    Number.isFinite(requiredRecordedUntilS)
    && probe.duration_s < requiredRecordedUntilS - requiredRecordedUntilToleranceS
  ) {
    errors.push(
      `task video duration ${probe.duration_s.toFixed(3)}s does not cover required content through `
      + `${requiredRecordedUntilS.toFixed(3)}s, including the post-response recording tail`,
    );
  }
  if (queryEvents.length !== expectedQueries.length) {
    errors.push(`query event count ${queryEvents.length} differs from expected ${expectedQueries.length}`);
  }
  for (let index = 0; index < Math.min(queryEvents.length, expectedQueries.length); index += 1) {
    const actual = Number(queryEvents[index].actual_query_video_time_s);
    const expected = Number(expectedQueries[index].query_time_s);
    const drift = Math.abs(actual - expected);
    if (!Number.isFinite(actual) || !Number.isFinite(expected) || drift > queryTimeToleranceS) {
      errors.push(
        `query ${expectedQueries[index].id || index + 1} timing drift ${drift.toFixed(3)}s `
        + `(actual=${actual}, expected=${expected})`,
      );
    }
  }

  const decode = await run(ffmpegBin, [
    '-hide_banner', '-v', 'error', '-xerror', '-i', videoPath,
    '-map', '0:v:0', '-an', '-f', 'null', '-',
  ]);
  if (decode.code !== 0) {
    errors.push(decode.error?.message || decode.stderr.split('\n').slice(-8).join('\n'));
  }
  return {
    ok: errors.length === 0,
    errors,
    warnings,
    probe,
    expected_duration_s: expectedDurationS,
    minimum_duration_s: minimumDurationS,
    maximum_duration_s: maximumDurationS,
    required_recorded_until_s: requiredRecordedUntilS,
    required_recorded_until_tolerance_s: requiredRecordedUntilToleranceS,
    duration_tolerance_s: durationToleranceS,
    duration_error_s: durationErrorS,
    decode_ok: decode.code === 0,
  };
}

export async function replaceFile(source, target) {
  await fsp.rename(source, target).catch(async () => {
    await fsp.copyFile(source, target);
    await fsp.unlink(source).catch(() => {});
  });
}

export function siblingFfprobe(ffmpegBin) {
  return ffmpegBin.includes(path.sep)
    ? path.join(path.dirname(ffmpegBin), 'ffprobe')
    : 'ffprobe';
}
