#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadRecordingCampaignConfig } from './recording_config.mjs';
import { getVlmProfile, loadVlmRegistry } from './vlm_profiles.mjs';
import { probeVideo, validateVideoDecode } from './video_quality.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RESPONSE_POLICY = 'wait_then_keep_as_capability_result';
const RESPONSE_PROTOCOL = 'strict_protocol_non_deferred_non_echo_response_v4';

function parseArgs(argv) {
  const args = {
    config: 'config/recording_campaign.json',
    runId: '',
    report: '',
    allowIncomplete: false,
    fullDecode: false,
    decodeConcurrency: 4,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      index += 1;
      if (index >= argv.length) throw new Error(`Missing value for ${arg}`);
      return argv[index];
    };
    if (arg === '--config') args.config = next();
    else if (arg === '--run-id') args.runId = next();
    else if (arg === '--report') args.report = next();
    else if (arg === '--allow-incomplete') args.allowIncomplete = true;
    else if (arg === '--full-decode') args.fullDecode = true;
    else if (arg === '--decode-concurrency') args.decodeConcurrency = Number(next());
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!Number.isInteger(args.decodeConcurrency) || args.decodeConcurrency < 1 || args.decodeConcurrency > 16) {
    throw new Error('--decode-concurrency must be an integer from 1 through 16');
  }
  args.config = path.resolve(ROOT, args.config);
  return args;
}

function readJson(filePath) {
  try { return JSON.parse(fs.readFileSync(filePath, 'utf8')); } catch { return null; }
}

function safeName(value) {
  return String(value).replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^_+|_+$/g, '') || 'task';
}

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function sha256Text(value) {
  return crypto.createHash('sha256').update(String(value)).digest('hex');
}

function sameText(left, right) {
  return String(left || '') === String(right || '');
}

function realPathInside(filePath, directory) {
  try {
    const realFile = fs.realpathSync(filePath);
    const realDirectory = fs.realpathSync(directory);
    const relative = path.relative(realDirectory, realFile);
    return relative && !relative.startsWith('..') && !path.isAbsolute(relative);
  } catch {
    return false;
  }
}

async function mapLimit(items, concurrency, action) {
  const results = new Array(items.length);
  let cursor = 0;
  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      results[index] = await action(items[index], index);
    }
  });
  await Promise.all(workers);
  return results;
}

const args = parseArgs(process.argv.slice(2));
const loaded = loadRecordingCampaignConfig(args.config);
const config = loaded.raw;
const current = readJson(path.join(loaded.outputRoot, 'current.json'));
const runId = args.runId || String(current?.campaign_id || '');
if (!runId) throw new Error('No recording campaign run is available');
const runRoot = path.join(loaded.outputRoot, 'runs', runId);
const taskPath = path.join(runRoot, 'tasks.jsonl');
if (!fs.existsSync(taskPath)) throw new Error(`Run task manifest is missing: ${taskPath}`);
const tasks = fs.readFileSync(taskPath, 'utf8')
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));
const enabledModels = config.models.filter((model) => model.enabled === true);
const registry = loadVlmRegistry(path.resolve(ROOT, config.vlm_registry));
const systemPrompt = fs.readFileSync(path.join(ROOT, 'config', 'joyai_system_prompt.txt'), 'utf8').trim();
const systemPromptSha256 = sha256Text(systemPrompt);
const queryDriftToleranceS = 0.5;

const work = [];
for (const model of enabledModels) {
  const profile = getVlmProfile(registry, model.vlm_profile);
  const modelDir = path.join(runRoot, 'models', model.id);
  const supervisor = readJson(path.join(modelDir, 'supervisor_state.json'));
  const modelErrors = [];
  if (!supervisor) modelErrors.push('supervisor_state.json is missing or invalid');
  if (supervisor?.system_prompt_sha256 !== systemPromptSha256) {
    modelErrors.push('supervisor system prompt hash mismatch');
  }
  for (let index = 0; index < tasks.length; index += 1) {
    work.push({ model, profile, modelDir, supervisor, modelErrors, task: tasks[index], index });
  }
}

const checked = await mapLimit(work, args.fullDecode ? args.decodeConcurrency : 8, async ({
  model,
  profile,
  modelDir,
  supervisor,
  task,
  index,
}) => {
  const errors = [];
  const taskDir = path.join(modelDir, `${String(index + 1).padStart(3, '0')}_${safeName(task.id)}`);
  const result = supervisor?.results?.[task.id] || null;
  if (result?.status !== 'ok') errors.push('current supervisor has no validated status=ok result');
  const summary = readJson(path.join(taskDir, 'summary.json'));
  const validation = readJson(path.join(taskDir, 'capture_validation.json'));
  if (!summary) errors.push('summary.json is missing or invalid');
  if (!validation) errors.push('capture_validation.json is missing or invalid');
  if (summary?.status !== 'ok') errors.push(`summary status is ${summary?.status || 'missing'}`);
  if (summary?.task?.id !== task.id || validation?.task_id !== task.id) {
    errors.push('task identity mismatch');
  }
  if (summary?.vlm_backend_identity?.ok !== true || validation?.model_identity_ok !== true) {
    errors.push('target model identity was not verified');
  }
  if (validation?.system_prompt_sha256 !== systemPromptSha256) {
    errors.push('task system prompt hash mismatch');
  }
  const contract = validation?.evaluation_contract || {};
  if (contract.model_request_id !== profile.model) errors.push('model request id mismatch');
  if (contract.input_transport !== profile.input_transport) errors.push('input transport mismatch');
  if (
    model.runner === 'joyai_scaffold'
    && contract.interaction_scaffold !== 'joyai-official-live-adapter'
  ) errors.push('JoyAI scaffold contract is missing');
  if (profile.input_transport === 'native-video-realtime') {
    if (contract.realtime_protocol !== profile.realtime_protocol) errors.push('realtime protocol mismatch');
    if (contract.realtime_query_mode !== profile.realtime_query_mode) errors.push('realtime Query mode mismatch');
    if (validation?.native_realtime_audit?.ok !== true) errors.push('native realtime audit failed');
  } else if (validation?.frame_stream_audit?.ok !== true) {
    errors.push('frame stream audit failed');
  }
  if (summary?.recording_start_barrier?.ok !== true) errors.push('recording start barrier failed');
  if (summary?.recording_start_barrier?.webui_same_session_model_ready !== true) {
    errors.push('model was not ready in the recording WebUI session before upload');
  }
  if (summary?.recording_start_barrier?.target_video_uploaded_after_model_ready !== true) {
    errors.push('target video upload did not follow model readiness');
  }
  if (summary?.query_delivery_attestation?.ok !== true) errors.push('Query once-delivery audit failed');
  if (summary?.query_delivery_attestation?.replay_count !== 0) errors.push('Query replay was observed');
  if (summary?.query_delivery_attestation?.cache_into_later_frames !== false) {
    errors.push('Query was cached into later frames');
  }
  const expectedQueries = Array.isArray(task.queries) ? task.queries : [];
  const actualQueries = Array.isArray(summary?.query_events) ? summary.query_events : [];
  if (actualQueries.length !== expectedQueries.length) {
    errors.push(`Query round count is ${actualQueries.length}/${expectedQueries.length}`);
  }
  for (let queryIndex = 0; queryIndex < Math.min(expectedQueries.length, actualQueries.length); queryIndex += 1) {
    const expected = expectedQueries[queryIndex];
    const actual = actualQueries[queryIndex];
    if (actual.id !== expected.id || !sameText(actual.query, expected.query)) {
      errors.push(`Query ${queryIndex + 1} identity or text mismatch`);
    }
    const configuredDrift = Math.abs(Number(actual.query_time_s) - Number(expected.query_time_s));
    const actualDrift = Math.abs(Number(actual.actual_query_video_time_s) - Number(expected.query_time_s));
    if (!Number.isFinite(configuredDrift) || configuredDrift > 0.001) {
      errors.push(`Query ${expected.id} configured time mismatch`);
    }
    if (!Number.isFinite(actualDrift) || actualDrift > queryDriftToleranceS) {
      errors.push(`Query ${expected.id} video-time drift exceeds ${queryDriftToleranceS}s`);
    }
    if (actual.video_ready_before_query !== true) errors.push(`Query ${expected.id} preceded video readiness`);
    if (actual.query_frame_queue?.acknowledged !== true) errors.push(`Query ${expected.id} frame queue was not acknowledged`);
  }
  if (validation?.upstream_query_audit?.ok !== true) errors.push('upstream Query audit failed');
  if (Number(validation?.upstream_query_audit?.explicit_provider_error_count || 0) !== 0) {
    errors.push('an explicit provider error was accepted');
  }
  const outcome = validation?.model_response_outcome || summary?.model_response_outcome || {};
  const acceptedCapabilityTimeout = outcome.policy === RESPONSE_POLICY
    && outcome.response_semantics === RESPONSE_PROTOCOL
    && outcome.timeout_is_recording_failure === false
    && ['timeout', 'partial_timeout'].includes(outcome.status);
  if (outcome.response_semantics !== RESPONSE_PROTOCOL) errors.push('response outcome semantics mismatch');
  if (!acceptedCapabilityTimeout && !['complete', 'not_required'].includes(outcome.status)) {
    errors.push(`model response outcome is not accepted: ${outcome.status || 'missing'}`);
  }
  if (summary?.provider_infrastructure_failure) errors.push('provider infrastructure failure was recorded');
  if (Number(summary?.stream_reconnect_count || 0) !== 0) errors.push('stream reconnect occurred');
  if (summary?.task_video_render?.outages?.length) errors.push('recorded task video contains an outage');
  if (summary?.video_quality?.ok !== true || summary?.video_quality?.decode_ok !== true) {
    errors.push('capture-time video validation failed');
  }
  const taskMp4 = String(result?.task_mp4 || summary?.files?.task_mp4 || '');
  if (path.basename(taskMp4) !== `${safeName(task.id)}_task.mp4`) errors.push('task MP4 filename mismatch');
  if (!taskMp4 || !fs.existsSync(taskMp4)) errors.push('task MP4 is missing');
  else if (!realPathInside(taskMp4, taskDir)) errors.push('task MP4 is outside its canonical task directory');
  let probe = null;
  let fullDecode = null;
  if (taskMp4 && fs.existsSync(taskMp4)) {
    try {
      probe = await probeVideo(taskMp4);
      if (!(probe.duration_s > 0 && probe.size_bytes > 0 && probe.width > 0 && probe.height > 0)) {
        errors.push('ffprobe returned invalid metadata');
      }
      const lastQueryS = Math.max(0, ...expectedQueries.map((query) => Number(query.query_time_s) || 0));
      const minimumDurationS = Number(summary?.minimum_task_video_duration_s)
        || Math.max(Number(task.duration_s) || 0, lastQueryS + 1);
      if (probe.duration_s + 1.5 < minimumDurationS) {
        errors.push(`task MP4 is too short: ${probe.duration_s.toFixed(3)}s < ${minimumDurationS.toFixed(3)}s`);
      }
    } catch (error) {
      errors.push(`ffprobe failed: ${error.message}`);
    }
    if (args.fullDecode) {
      fullDecode = await validateVideoDecode({
        videoPath: taskMp4,
        decodeAttempts: 1,
        retryDelayMs: 0,
      });
      if (fullDecode?.ok !== true) {
        errors.push(`full decode failed: ${fullDecode?.error || 'unknown error'}`);
      }
    }
  }
  return {
    model_id: model.id,
    task_index: index + 1,
    task_id: task.id,
    ok: errors.length === 0,
    errors,
    task_mp4: taskMp4,
    probe,
    full_decode: fullDecode,
  };
});

const models = enabledModels.map((model) => {
  const modelDir = path.join(runRoot, 'models', model.id);
  const supervisor = readJson(path.join(modelDir, 'supervisor_state.json'));
  const taskResults = checked.filter((item) => item.model_id === model.id);
  const errors = [];
  if (!supervisor) errors.push('supervisor state is unavailable');
  if (supervisor?.status !== 'complete') errors.push(`supervisor status is ${supervisor?.status || 'missing'}`);
  return {
    id: model.id,
    output: modelDir,
    real_output: fs.existsSync(modelDir) ? fs.realpathSync(modelDir) : '',
    ok: errors.length === 0 && taskResults.every((item) => item.ok),
    errors,
    tasks_ok: taskResults.filter((item) => item.ok).length,
    tasks_total: tasks.length,
    current_task: supervisor?.current_task || null,
    task_results: taskResults,
  };
});

const report = {
  schema_version: 1,
  status: models.length === 4 && models.every((model) => model.ok) ? 'complete' : 'incomplete',
  checked_at: new Date().toISOString(),
  campaign_id: runId,
  campaign_config: args.config,
  campaign_config_sha256: sha256File(args.config),
  task_manifest: taskPath,
  task_manifest_sha256: sha256File(taskPath),
  system_prompt_sha256: systemPromptSha256,
  full_decode_requested: args.fullDecode,
  decode_concurrency: args.fullDecode ? args.decodeConcurrency : null,
  expected_models: 4,
  models_ok: models.filter((model) => model.ok).length,
  models_total: models.length,
  task_videos_ok: models.reduce((total, model) => total + model.tasks_ok, 0),
  task_videos_total: models.length * tasks.length,
  models,
};
const reportPath = args.report
  ? path.resolve(ROOT, args.report)
  : path.join(runRoot, 'acceptance', 'recording_verification.json');
await fsp.mkdir(path.dirname(reportPath), { recursive: true });
const temporary = `${reportPath}.tmp.${process.pid}`;
await fsp.writeFile(temporary, `${JSON.stringify(report, null, 2)}\n`, { mode: 0o600 });
await fsp.rename(temporary, reportPath);
console.log(JSON.stringify({
  status: report.status,
  campaign_id: runId,
  models_ok: report.models_ok,
  models_total: report.models_total,
  task_videos_ok: report.task_videos_ok,
  task_videos_total: report.task_videos_total,
  per_model: Object.fromEntries(models.map((model) => [model.id, `${model.tasks_ok}/${model.tasks_total}`])),
  report: reportPath,
}, null, 2));
if (report.status !== 'complete' && !args.allowIncomplete) process.exitCode = 1;
