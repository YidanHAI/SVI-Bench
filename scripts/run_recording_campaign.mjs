#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  getVlmProfile,
  loadVlmRegistry,
  safeProfileSnapshot,
} from './vlm_profiles.mjs';
import {
  DEFAULT_RECORDING_CAMPAIGN_CONFIG,
  loadRecordingCampaignConfig,
  normalizeRecordingWebUrl,
  recordingWebUrl,
} from './recording_config.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const EVALUATION_PROTOCOL = 'joyai-formal-query-once-frame-stream-v1';
const DEFAULT_CONFIG = DEFAULT_RECORDING_CAMPAIGN_CONFIG;

function printHelp() {
  console.log(`Usage:
  node scripts/run_recording_campaign.mjs run [options]
  node scripts/run_recording_campaign.mjs add [options]
  node scripts/run_recording_campaign.mjs dry-run [options]
  node scripts/run_recording_campaign.mjs status [options]
  node scripts/run_recording_campaign.mjs stop [options]

Options:
  --config FILE       Campaign config; default config/recording_campaign.json
  --models IDS        Comma-separated campaign model ids; default all enabled models
  --task-ids IDS      Comma-separated task ids for a bounded subset run
  --run-id ID         Optional human-readable run directory id
  --resume-current    Resume the current run and reuse only per-task validated MP4s
  --resume-run-id ID  Resume a specific existing run directory
                      With add, target a specific live or stopped run; default current
  --allow-webui-migration
                      Resume on the canonical configured WebUI while preserving validated MP4s
  --log-path FILE     Launcher log recorded in campaign state
  --json              Machine-readable status output

The WebUI, network route, model list, and non-secret runtime policy come from the campaign config.
Secrets are read only from the environment or the private launcher credential file.`);
}

function parseArgs(argv) {
  const command = argv[0] && !argv[0].startsWith('-') ? argv[0] : 'dry-run';
  const args = {
    command,
    config: DEFAULT_CONFIG,
    models: '',
    taskIds: '',
    runId: '',
    resumeCurrent: false,
    resumeRunId: '',
    allowWebUiMigration: false,
    logPath: process.env.RECORDING_CAMPAIGN_LOG || '',
    json: false,
  };
  const rest = argv[0] === command ? argv.slice(1) : argv;
  for (let index = 0; index < rest.length; index += 1) {
    const arg = rest[index];
    const next = () => {
      if (index + 1 >= rest.length) throw new Error(`Missing value for ${arg}`);
      index += 1;
      return rest[index];
    };
    if (arg === '--config') args.config = next();
    else if (arg.startsWith('--config=')) args.config = arg.slice('--config='.length);
    else if (arg === '--models') args.models = next();
    else if (arg.startsWith('--models=')) args.models = arg.slice('--models='.length);
    else if (arg === '--task-ids') args.taskIds = next();
    else if (arg.startsWith('--task-ids=')) args.taskIds = arg.slice('--task-ids='.length);
    else if (arg === '--run-id') args.runId = next();
    else if (arg.startsWith('--run-id=')) args.runId = arg.slice('--run-id='.length);
    else if (arg === '--resume-current') args.resumeCurrent = true;
    else if (arg === '--resume-run-id') args.resumeRunId = next();
    else if (arg.startsWith('--resume-run-id=')) {
      args.resumeRunId = arg.slice('--resume-run-id='.length);
    }
    else if (arg === '--allow-webui-migration') args.allowWebUiMigration = true;
    else if (arg === '--log-path') args.logPath = next();
    else if (arg.startsWith('--log-path=')) args.logPath = arg.slice('--log-path='.length);
    else if (arg === '--json') args.json = true;
    else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    } else throw new Error(`Unknown option: ${arg}`);
  }
  if (!['run', 'add', 'dry-run', 'status', 'stop'].includes(args.command)) {
    throw new Error(`Unknown command: ${args.command}`);
  }
  args.config = path.resolve(ROOT, args.config);
  args.logPath = args.logPath ? path.resolve(ROOT, args.logPath) : '';
  args.taskIds = [...new Set(args.taskIds.split(',').map((item) => item.trim()).filter(Boolean))];
  args.runId = String(args.runId || '').trim();
  args.resumeRunId = String(args.resumeRunId || '').trim();
  if ((args.resumeCurrent || args.resumeRunId) && args.taskIds.length) {
    throw new Error('resume options cannot be combined with --task-ids');
  }
  if ((args.resumeCurrent || args.resumeRunId) && args.runId) {
    throw new Error('resume options cannot be combined with --run-id');
  }
  if (args.resumeCurrent && args.resumeRunId) {
    throw new Error('--resume-current cannot be combined with --resume-run-id');
  }
  if (args.allowWebUiMigration && !args.resumeCurrent && !args.resumeRunId) {
    throw new Error('--allow-webui-migration requires a resume option');
  }
  if (args.command === 'add') {
    if (!args.models.trim()) throw new Error('add requires --models IDS');
    if (args.runId) throw new Error('add cannot be combined with --run-id');
    if (args.taskIds.length) throw new Error('add cannot be combined with --task-ids');
    if (args.allowWebUiMigration) {
      throw new Error('add cannot migrate a live run to another WebUI');
    }
    if (!args.resumeRunId) args.resumeCurrent = true;
  }
  if (args.runId && !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/.test(args.runId)) {
    throw new Error('--run-id must be filesystem-safe and at most 80 characters');
  }
  if (args.resumeRunId && !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/.test(args.resumeRunId)) {
    throw new Error('--resume-run-id must be filesystem-safe and at most 80 characters');
  }
  for (const taskId of args.taskIds) {
    if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(taskId)) {
      throw new Error(`Invalid --task-ids value: ${taskId}`);
    }
  }
  return args;
}

function readJson(filePath) {
  try { return JSON.parse(fs.readFileSync(filePath, 'utf8')); } catch { return null; }
}

function sameJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

async function writeJsonAtomic(filePath, value) {
  await fsp.mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.tmp-${process.pid}`;
  await fsp.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 });
  await fsp.rename(temporary, filePath);
}

function sha256Bytes(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function sha256File(filePath) {
  return sha256Bytes(fs.readFileSync(filePath));
}

function createRunId(requested = '') {
  if (requested) return requested;
  const stamp = new Date().toISOString()
    .replace(/[-:]/g, '')
    .replace(/\.\d{3}Z$/, 'Z');
  return `run_${stamp}_${process.pid}`;
}

function requestRecordingWebUiHome({ webui, username, password, timeoutMs = 30000 }) {
  return new Promise((resolve, reject) => {
    const url = new URL(webui.url);
    const authorization = Buffer.from(`${username}:${password}`).toString('base64');
    const req = https.request({
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port || 443,
      path: `${url.pathname}${url.search}`,
      method: 'GET',
      headers: { authorization: `Basic ${authorization}` },
      rejectUnauthorized: webui.tlsRejectUnauthorized,
      timeout: timeoutMs,
    }, (res) => {
      const chunks = [];
      let bytes = 0;
      const maxBytes = 2 * 1024 * 1024;
      res.on('data', (chunk) => {
        bytes += chunk.length;
        if (bytes <= maxBytes) chunks.push(chunk);
      });
      res.once('end', () => resolve({
        statusCode: Number(res.statusCode || 0),
        contentType: String(res.headers['content-type'] || ''),
        body: Buffer.concat(chunks).toString('utf8'),
        truncated: bytes > maxBytes,
      }));
    });
    req.once('timeout', () => req.destroy(new Error(`request timed out after ${timeoutMs}ms`)));
    req.once('error', reject);
    req.end();
  });
}

async function verifyRecordingWebUi(campaign) {
  let response;
  try {
    response = await requestRecordingWebUiHome({
      webui: campaign.webui,
      username: process.env[campaign.webui.usernameEnv],
      password: process.env[campaign.webui.passwordEnv],
    });
  } catch (error) {
    const detail = /wrong version number/i.test(String(error?.message || ''))
      ? ' TLS negotiation was rejected; the port may be serving plaintext HTTP or an unrelated service.'
      : '';
    throw new Error(
      `Recording WebUI preflight failed for ${campaign.webUrl}: ${error.message}.${detail}`,
    );
  }
  const title = response.body.match(/<title>\s*([^<]+?)\s*<\/title>/i)?.[1]?.trim() || '';
  const expectedTitle = title.toLowerCase() === campaign.webui.expectedTitle.toLowerCase();
  const hasUploadUi = campaign.webui.identityMarkers.every((marker) => response.body.includes(marker));
  if (response.statusCode !== 200 || !expectedTitle || !hasUploadUi || response.truncated) {
    throw new Error(
      `Recording WebUI identity check failed: expected authenticated `
      + `${campaign.webui.expectedTitle} upload UI at ${campaign.webUrl}, observed status=${response.statusCode}, `
      + `content_type=${response.contentType || '(missing)'}, title=${JSON.stringify(title || '(missing)')}`,
    );
  }
  return {
    ok: true,
    checked_at: new Date().toISOString(),
    web_url: campaign.webUrl,
    status_code: response.statusCode,
    title,
    local_upload_ui: true,
  };
}

function resolveProjectPath(value, field) {
  const resolved = path.resolve(ROOT, String(value || ''));
  if (!String(value || '').trim()) throw new Error(`Campaign config is missing ${field}`);
  return resolved;
}

function positiveInteger(value, field, { min = 0, max = Number.MAX_SAFE_INTEGER } = {}) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
    throw new Error(`${field} must be an integer from ${min} through ${max}`);
  }
  return parsed;
}

function loadCampaign(args) {
  const loaded = loadRecordingCampaignConfig(args.config);
  const raw = loaded.raw;
  if (raw.evaluation_protocol !== EVALUATION_PROTOCOL) {
    throw new Error(`Campaign evaluation_protocol must be exactly ${EVALUATION_PROTOCOL}`);
  }
  const registryPath = resolveProjectPath(raw.vlm_registry, 'vlm_registry');
  const registry = loadVlmRegistry(registryPath);
  const source = {
    manifest: resolveProjectPath(raw.task_source?.manifest, 'task_source.manifest'),
    xlsx: raw.task_source?.xlsx
      ? resolveProjectPath(raw.task_source.xlsx, 'task_source.xlsx')
      : '',
    sheet: String(raw.task_source?.sheet || '').trim(),
    videoDir: resolveProjectPath(raw.task_source?.video_dir, 'task_source.video_dir'),
  };
  source.taskIds = args.taskIds;
  if (!fs.existsSync(source.manifest) || !fs.statSync(source.manifest).isFile()) {
    throw new Error(`Fixed task manifest does not exist: ${source.manifest}`);
  }
  if (source.xlsx && (!fs.existsSync(source.xlsx) || !fs.statSync(source.xlsx).isFile())) {
    throw new Error(`Workbook does not exist: ${source.xlsx}`);
  }
  if (!fs.existsSync(source.videoDir) || !fs.statSync(source.videoDir).isDirectory()) {
    throw new Error(`Video directory does not exist: ${source.videoDir}`);
  }
  const defaults = {
    attemptWorkRoot: path.resolve(String(
      raw.defaults?.attempt_work_root || path.join(os.tmpdir(), 'joyvl-recording'),
    )),
    maxParallelModels: positiveInteger(
      raw.defaults?.max_parallel_models ?? 1,
      'max_parallel_models',
      { min: 1 },
    ),
    maxTaskRetries: positiveInteger(raw.defaults?.max_task_retries ?? 5, 'max_task_retries', { max: 5 }),
    maxFailedTaskRecoveryBatches: positiveInteger(
      raw.defaults?.max_failed_task_recovery_batches ?? 1,
      'max_failed_task_recovery_batches',
      { min: 1, max: 3 },
    ),
    maxTaskWallS: positiveInteger(
      raw.defaults?.max_task_wall_s ?? 3600,
      'max_task_wall_s',
      { min: 300 },
    ),
    maxInfrastructureRestarts: positiveInteger(
      raw.defaults?.max_infrastructure_restarts ?? 5,
      'max_infrastructure_restarts',
      { max: 5 },
    ),
    maxProviderQueryStartDriftS: positiveInteger(
      raw.defaults?.max_provider_query_start_drift_s ?? 5,
      'max_provider_query_start_drift_s',
      { min: 1, max: 60 },
    ),
    progressIntervalS: positiveInteger(raw.defaults?.progress_interval_s ?? 1800, 'progress_interval_s', { min: 30 }),
    phaseRestartBackoffS: positiveInteger(raw.defaults?.phase_restart_backoff_s ?? 60, 'phase_restart_backoff_s', { min: 1 }),
  };
  const frameScheduler = {
    maxInFlight: positiveInteger(
      raw.frame_scheduler?.max_in_flight,
      'frame_scheduler.max_in_flight',
      { min: 1, max: 1 },
    ),
    busyPolicy: String(raw.frame_scheduler?.busy_policy || ''),
    queueCapacity: positiveInteger(
      raw.frame_scheduler?.queue_capacity,
      'frame_scheduler.queue_capacity',
      { max: 0 },
    ),
    processIntervalS: Number(raw.frame_scheduler?.process_interval_s),
    framesPerBatch: positiveInteger(
      raw.frame_scheduler?.frames_per_batch,
      'frame_scheduler.frames_per_batch',
      { min: 1, max: 1 },
    ),
  };
  if (frameScheduler.busyPolicy !== 'skip') {
    throw new Error('frame_scheduler.busy_policy must be skip');
  }
  if (frameScheduler.processIntervalS !== 1) {
    throw new Error('frame_scheduler.process_interval_s must be exactly 1');
  }
  const upstreamRequestPolicy = {
    sdkMaxRetries: positiveInteger(
      raw.upstream_request_policy?.sdk_max_retries,
      'upstream_request_policy.sdk_max_retries',
      { max: 0 },
    ),
    nonQueryTimeoutS: positiveInteger(
      raw.upstream_request_policy?.non_query_timeout_s,
      'upstream_request_policy.non_query_timeout_s',
      { min: 1, max: 180 },
    ),
    queryTimeoutS: positiveInteger(
      raw.upstream_request_policy?.query_timeout_s,
      'upstream_request_policy.query_timeout_s',
      { min: 1, max: 180 },
    ),
    warmupTimeoutS: positiveInteger(
      raw.upstream_request_policy?.warmup_timeout_s,
      'upstream_request_policy.warmup_timeout_s',
      { min: 1, max: 180 },
    ),
  };
  if (upstreamRequestPolicy.nonQueryTimeoutS > upstreamRequestPolicy.queryTimeoutS) {
    throw new Error(
      'upstream_request_policy.non_query_timeout_s must not exceed query_timeout_s',
    );
  }
  if (!Array.isArray(raw.models) || !raw.models.length) throw new Error('Campaign config models must be non-empty');
  const ids = new Set();
  const ports = new Set();
  const models = raw.models.map((model) => {
    const id = String(model.id || '').trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(id)) throw new Error(`Invalid campaign model id: ${id}`);
    if (ids.has(id)) throw new Error(`Duplicate campaign model id: ${id}`);
    ids.add(id);
    if (!['adapter', 'prompt_proxy', 'joyai_scaffold'].includes(model.runner)) {
      throw new Error(
        `Model ${id} runner must be adapter, prompt_proxy, or joyai_scaffold under ${EVALUATION_PROTOCOL}`,
      );
    }
    const profile = getVlmProfile(registry, String(model.vlm_profile || ''));
    if (model.runner === 'adapter' && profile.route !== 'joyai_adapter') {
      throw new Error(`Model ${id} adapter runner requires a joyai_adapter profile`);
    }
    if (model.runner === 'prompt_proxy' && profile.route !== 'direct') {
      throw new Error(`Model ${id} prompt_proxy runner requires a direct profile`);
    }
    if (model.runner === 'joyai_scaffold') {
      if (profile.route !== 'direct') {
        throw new Error(`Model ${id} joyai_scaffold runner requires a direct upstream profile`);
      }
      if (profile.interaction_scaffold !== 'joyai-official-live-adapter') {
        throw new Error(
          `Model ${id} joyai_scaffold runner requires interaction_scaffold=joyai-official-live-adapter`,
        );
      }
    }
    let scaffoldDeployment = null;
    if (model.scaffold_deployment != null) {
      if (!['prompt_proxy', 'joyai_scaffold'].includes(model.runner)) {
        throw new Error(
          `Model ${id} scaffold_deployment requires runner=prompt_proxy or joyai_scaffold`,
        );
      }
      const apiBase = String(model.scaffold_deployment.api_base || '').replace(/\/+$/, '');
      const accessTokenEnv = String(
        model.scaffold_deployment.access_token_env || '',
      ).trim();
      if (!apiBase || !['http:', 'https:'].includes(new URL(apiBase).protocol)) {
        throw new Error(`Model ${id} scaffold_deployment.api_base must use HTTP or HTTPS`);
      }
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(accessTokenEnv)) {
        throw new Error(`Model ${id} scaffold_deployment.access_token_env is invalid`);
      }
      scaffoldDeployment = {
        apiBase,
        accessTokenEnv,
        auditPath: resolveProjectPath(
          model.scaffold_deployment.audit_path,
          `${id}.scaffold_deployment.audit_path`,
        ),
      };
    }
    let proxyPort = null;
    if (
      ['prompt_proxy', 'joyai_scaffold'].includes(model.runner)
      && !scaffoldDeployment
    ) {
      proxyPort = positiveInteger(model.proxy_port, `${id}.proxy_port`, { min: 1024, max: 65535 });
      if (ports.has(proxyPort)) throw new Error(`Duplicate prompt-proxy port: ${proxyPort}`);
      ports.add(proxyPort);
      if (!profile.api_key_env) throw new Error(`Direct profile ${profile.id} is missing api_key_env`);
    }
    if (model.persist_user_query === true) {
      throw new Error(
        `Model ${id} requests persist_user_query=true, but every Query must be delivered exactly once`,
      );
    }
    const nativeStreamingModePolicy = String(
      model.native_streaming_mode_policy || 'proactive',
    ).trim();
    if (![
      'proactive',
      'interactive-after-query',
    ].includes(nativeStreamingModePolicy)) {
      throw new Error(
        `Model ${id} has unsupported native_streaming_mode_policy=${nativeStreamingModePolicy}`,
      );
    }
    if (
      nativeStreamingModePolicy !== 'proactive'
      && profile.realtime_protocol !== 'joyai-http-session-v1'
    ) {
      throw new Error(
        `Model ${id} interactive streaming requires realtime_protocol=joyai-http-session-v1`,
      );
    }
    const nativeSystemPromptTransport = String(
      model.native_system_prompt_transport || 'system-role',
    ).trim();
    if (!['system-role', 'inline-user-query'].includes(nativeSystemPromptTransport)) {
      throw new Error(
        `Model ${id} has unsupported native_system_prompt_transport=${nativeSystemPromptTransport}`,
      );
    }
    const nativeFrameClock = String(model.native_frame_clock || 'inbound-turn').trim();
    if (!['inbound-turn', 'wall-media'].includes(nativeFrameClock)) {
      throw new Error(`Model ${id} has unsupported native_frame_clock=${nativeFrameClock}`);
    }
    if (
      (nativeSystemPromptTransport !== 'system-role' || nativeFrameClock !== 'inbound-turn')
      && profile.realtime_protocol !== 'joyai-http-session-v1'
    ) {
      throw new Error(
        `Model ${id} native prompt/clock compatibility requires joyai-http-session-v1`,
      );
    }
    let nativeQueryAudioManifest = '';
    if (profile.realtime_query_mode === 'input-audio-once') {
      nativeQueryAudioManifest = resolveProjectPath(
        model.native_query_audio_manifest,
        `${id}.native_query_audio_manifest`,
      );
      if (!fs.statSync(nativeQueryAudioManifest).isFile()) {
        throw new Error(`Model ${id} native_query_audio_manifest must point to a file`);
      }
    } else if (model.native_query_audio_manifest) {
      throw new Error(
        `Model ${id} native_query_audio_manifest requires realtime_query_mode=input-audio-once`,
      );
    }
    return {
      id,
      enabled: model.enabled !== false,
      runner: model.runner,
      profile,
      proxyPort,
      scaffoldDeployment,
      nativeStreamingModePolicy,
      nativeSystemPromptTransport,
      nativeFrameClock,
      nativeQueryAudioManifest,
      disableThinking: model.disable_thinking === true,
      continueAfterTaskFailure: model.continue_after_task_failure !== false,
      blockedReason: String(model.blocked_reason || '').trim(),
    };
  });
  const requested = args.models.split(',').map((item) => item.trim()).filter(Boolean);
  const selected = requested.length
    ? requested.map((id) => {
      const model = models.find((item) => item.id === id);
      if (!model) throw new Error(`Unknown campaign model ${id}; available=${models.map((item) => item.id).join(',')}`);
      return model;
    })
    : models.filter((model) => model.enabled);
  if (!selected.length) throw new Error('No campaign models selected');
  if (new Set(selected.map((model) => model.id)).size !== selected.length) {
    throw new Error('The --models selection contains duplicates');
  }
  for (const model of selected) {
    if (model.profile.input_transport === 'native-video-batch') {
      throw new Error(
        `Model ${model.id} requires native-video-batch input (${model.profile.native_video_schema}), `
        + `which is forbidden under ${EVALUATION_PROTOCOL}. Select its JoyAI-compatible `
        + `frame-stream profile${model.blockedReason ? `: ${model.blockedReason}` : ''}`,
      );
    }
    if (
      model.runner === 'prompt_proxy'
      && model.profile.input_transport !== 'native-video-realtime'
    ) {
      throw new Error(
        `Model ${model.id} prompt_proxy requires native-video-realtime input`,
      );
    }
    if (model.runner === 'joyai_scaffold' && model.profile.input_transport !== 'stateful-frame-stream') {
      throw new Error(
        `Model ${model.id} cannot use joyai_scaffold with input_transport=${model.profile.input_transport}`,
      );
    }
    if (model.runner === 'adapter' && model.profile.input_transport !== 'stateful-frame-stream') {
      throw new Error(
        `Model ${model.id} cannot use the JoyAI adapter with input_transport=${model.profile.input_transport}`,
      );
    }
  }
  return {
    configPath: loaded.path,
    configSha256: sha256File(loaded.path),
    evaluationProtocol: EVALUATION_PROTOCOL,
    webUrl: loaded.webui.url,
    webui: loaded.webui,
    network: loaded.network,
    outputRoot: loaded.outputRoot,
    registry,
    source,
    defaults,
    frameScheduler,
    upstreamRequestPolicy,
    models: selected,
  };
}

function requiredEnvironment(campaign) {
  const names = new Set([
    campaign.webui.usernameEnv,
    campaign.webui.passwordEnv,
  ]);
  for (const model of campaign.models) {
    if (model.scaffoldDeployment) {
      names.add(model.scaffoldDeployment.accessTokenEnv);
    } else if (model.profile.api_key_env) {
      names.add(model.profile.api_key_env);
    }
  }
  return [...names].map((name) => ({ name, set: Boolean(process.env[name]) }));
}

function findExecutable(command) {
  if (command.includes(path.sep)) {
    try { fs.accessSync(command, fs.constants.X_OK); return path.resolve(command); } catch { return ''; }
  }
  for (const directory of String(process.env.PATH || '').split(path.delimiter)) {
    if (!directory) continue;
    const candidate = path.join(directory, command);
    try { fs.accessSync(candidate, fs.constants.X_OK); return candidate; } catch {}
  }
  return '';
}

function requiredExecutables(campaign) {
  const commands = ['python3', 'ffmpeg', 'ffprobe', process.execPath];
  if (campaign.network.mode === 'wireguard') commands.push('ip', 'wg');
  if (campaign.models.some((model) => (
    model.runner === 'prompt_proxy'
    || (model.runner === 'joyai_scaffold' && !model.scaffoldDeployment)
  ))) {
    commands.push(process.env.CLOUDFLARED_BIN || 'cloudflared');
  }
  return [...new Set(commands)].map((command) => ({ command, path: findExecutable(command) }));
}

function runChildCollect(command, argv) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, argv, { cwd: ROOT, env: process.env, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.once('error', reject);
    child.once('exit', (code, signal) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(
        `${command} exited with code ${code ?? 'null'} signal ${signal || ''}: ${(stderr || stdout).slice(-2000)}`,
      ));
    });
  });
}

async function verifyRecordingNetwork(campaign) {
  if (campaign.network.mode === 'direct') {
    return {
      ok: true,
      checked_at: new Date().toISOString(),
      mode: 'direct',
      target_host: new URL(campaign.webUrl).hostname,
    };
  }

  const targetHost = new URL(campaign.webUrl).hostname;
  const ipBin = findExecutable('ip') || 'ip';
  const wgBin = findExecutable('wg') || 'wg';
  const route = await runChildCollect(ipBin, ['route', 'get', targetHost]);
  const routeText = route.stdout.trim();
  const expectedDevice = campaign.network.interface;
  const routeMatches = new RegExp(`(?:^|\\s)dev\\s+${expectedDevice.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?:\\s|$)`)
    .test(routeText);
  if (!routeMatches) {
    throw new Error(
      `Recording WebUI ${targetHost} is not routed through configured WireGuard interface `
      + `${expectedDevice}: ${routeText || '(no route output)'}`,
    );
  }

  const handshakes = await runChildCollect(
    wgBin,
    ['show', expectedDevice, 'latest-handshakes'],
  );
  const timestamps = handshakes.stdout.split(/\r?\n/)
    .map((line) => Number(line.trim().split(/\s+/).at(-1)))
    .filter((value) => Number.isFinite(value) && value > 0);
  if (!timestamps.length) {
    throw new Error(`WireGuard interface ${expectedDevice} has no completed peer handshake`);
  }
  const latestHandshakeEpochS = Math.max(...timestamps);
  const handshakeAgeS = Math.max(0, Math.floor(Date.now() / 1000) - latestHandshakeEpochS);
  if (handshakeAgeS > campaign.network.maxHandshakeAgeS) {
    throw new Error(
      `WireGuard interface ${expectedDevice} latest handshake is ${handshakeAgeS}s old; `
      + `limit=${campaign.network.maxHandshakeAgeS}s`,
    );
  }
  return {
    ok: true,
    checked_at: new Date().toISOString(),
    mode: 'wireguard',
    interface: expectedDevice,
    target_host: targetHost,
    route: routeText,
    latest_handshake_epoch_s: latestHandshakeEpochS,
    handshake_age_s: handshakeAgeS,
    max_handshake_age_s: campaign.network.maxHandshakeAgeS,
  };
}

function readTasks(taskPath) {
  return fs.readFileSync(taskPath, 'utf8').split(/\r?\n/).filter(Boolean).map((line, index) => {
    try { return JSON.parse(line); } catch (error) { throw new Error(`Invalid task JSON at line ${index + 1}: ${error.message}`); }
  });
}

async function buildTaskManifest(campaign, directory) {
  await fsp.mkdir(directory, { recursive: true });
  const taskPath = path.join(directory, 'tasks.jsonl');
  const reportPath = path.join(directory, 'tasks.report.json');
  const sourceTasks = readTasks(campaign.source.manifest);
  const sourceById = new Map(sourceTasks.map((task) => [String(task.id || ''), task]));
  const requestedIds = campaign.source.taskIds;
  const missingIds = requestedIds.filter((taskId) => !sourceById.has(taskId));
  if (missingIds.length) {
    throw new Error(`Requested task ids are absent from the fixed manifest: ${missingIds.join(', ')}`);
  }
  const tasks = requestedIds.length
    ? requestedIds.map((taskId) => sourceById.get(taskId))
    : sourceTasks;
  if (!tasks.length) throw new Error('Fixed task manifest contains no selected tasks');
  await fsp.writeFile(taskPath, `${tasks.map((task) => JSON.stringify(task)).join('\n')}\n`);
  const report = {
    source: campaign.source.manifest,
    selected_rows: tasks.length,
    tasks_written: tasks.length,
    missing_count: 0,
    multi_round_tasks: tasks.filter((task) => Array.isArray(task.queries) && task.queries.length > 1).length,
  };
  await writeJsonAtomic(reportPath, report);
  const ids = new Set();
  for (const task of tasks) {
    if (!task.id || ids.has(task.id)) throw new Error(`Task id is missing or duplicated: ${task.id || '(missing)'}`);
    ids.add(task.id);
    const sourcePath = path.resolve(String(task.local_video_path || ''));
    if (!fs.existsSync(sourcePath) || !fs.statSync(sourcePath).isFile()) {
      throw new Error(`Task ${task.id} local video is missing: ${sourcePath}`);
    }
    if (!Array.isArray(task.queries) || !task.queries.length) {
      throw new Error(`Task ${task.id} has no Query rounds`);
    }
    const queryTexts = task.queries.map((round) => String(round.query || '').trim());
    if (new Set(queryTexts).size !== queryTexts.length) {
      throw new Error(
        `Task ${task.id} repeats identical Query text across rounds; `
        + 'the frame-stream proxy cannot distinguish that from WebUI Query replay',
      );
    }
  }
  return { taskPath, reportPath, tasks, report };
}

function sourceSignatures(tasks) {
  const paths = [...new Set(tasks.map((task) => path.resolve(task.local_video_path)))].sort();
  return paths.map((filePath) => {
    const stat = fs.statSync(filePath);
    return { path: filePath, bytes: stat.size, mtime_ms: Math.round(stat.mtimeMs) };
  });
}

function campaignProvenance(campaign, built) {
  const taskSha256 = sha256File(built.taskPath);
  const sourceTaskManifestSha256 = sha256File(campaign.source.manifest);
  const workbookSha256 = campaign.source.xlsx ? sha256File(campaign.source.xlsx) : '';
  const sources = sourceSignatures(built.tasks);
  const profileSnapshots = campaign.models.map((model) => ({
    campaign_id: model.id,
    runner: model.runner,
    profile: safeProfileSnapshot(model.profile),
    proxy_port: model.proxyPort,
    scaffold_deployment: model.scaffoldDeployment ? {
      api_base: model.scaffoldDeployment.apiBase,
      access_token_env: model.scaffoldDeployment.accessTokenEnv,
      audit_path: model.scaffoldDeployment.auditPath,
    } : null,
    query_delivery: 'once_per_round',
    native_streaming_mode_policy: model.nativeStreamingModePolicy,
    native_system_prompt_transport: model.nativeSystemPromptTransport,
    native_frame_clock: model.nativeFrameClock,
    disable_thinking: model.disableThinking,
  }));
  return {
    taskSha256,
    sourceTaskManifestSha256,
    workbookSha256,
    sources,
    profileSnapshots,
  };
}

async function materializePlan(campaign, built, provenance, runId) {
  const root = path.join(campaign.outputRoot, 'runs', runId);
  if (fs.existsSync(root) && fs.readdirSync(root).length > 0) {
    throw new Error(`Run directory already exists; use --resume-current or another --run-id: ${root}`);
  }
  await fsp.mkdir(root, { recursive: true });
  const taskPath = path.join(root, 'tasks.jsonl');
  const reportPath = path.join(root, 'tasks.report.json');
  const bytes = fs.readFileSync(built.taskPath);
  if (!fs.existsSync(taskPath)) await fsp.writeFile(taskPath, bytes);
  const report = { ...built.report, out: taskPath, report: reportPath };
  await writeJsonAtomic(reportPath, report);
  const modelPlans = campaign.models.map((model) => ({
    ...model,
    outputDir: path.join(root, 'models', model.id),
  }));
  const spec = {
    version: 2,
    run_id: runId,
    campaign_id: runId,
    created_at: new Date().toISOString(),
    fixed_web_url: campaign.webUrl,
    network: campaign.network,
    input_mode: 'upload',
    frame_scheduler: campaign.frameScheduler,
    upstream_request_policy: campaign.upstreamRequestPolicy,
    config_path: campaign.configPath,
    config_sha256: campaign.configSha256,
    workbook: campaign.source.xlsx,
    workbook_sha256: provenance.workbookSha256,
    source_task_manifest: campaign.source.manifest,
    source_task_manifest_sha256: provenance.sourceTaskManifestSha256,
    requested_task_ids: campaign.source.taskIds,
    task_manifest: taskPath,
    task_manifest_sha256: provenance.taskSha256,
    tasks_total: built.tasks.length,
    unique_videos: provenance.sources.length,
    source_video_signatures: provenance.sources,
    hashes_are_provenance_only: true,
    max_task_retries: campaign.defaults.maxTaskRetries,
    max_failed_task_recovery_batches: campaign.defaults.maxFailedTaskRecoveryBatches,
    max_task_wall_s: campaign.defaults.maxTaskWallS,
    attempt_work_root: campaign.defaults.attemptWorkRoot,
    max_parallel_models: Math.min(campaign.defaults.maxParallelModels, modelPlans.length),
    max_provider_query_start_drift_s: campaign.defaults.maxProviderQueryStartDriftS,
    model_order: modelPlans.map((model) => ({
      id: model.id,
      runner: model.runner,
      profile: safeProfileSnapshot(model.profile),
      scaffold_deployment: model.scaffoldDeployment ? {
        api_base: model.scaffoldDeployment.apiBase,
        access_token_env: model.scaffoldDeployment.accessTokenEnv,
        audit_path: model.scaffoldDeployment.auditPath,
      } : null,
      output: model.outputDir,
    })),
  };
  await writeJsonAtomic(path.join(root, 'campaign_spec.json'), spec);
  return { runId, root, taskPath, reportPath, models: modelPlans, spec };
}

function modelCommand(campaign, plan, model) {
  if (model.runner === 'adapter') {
    return {
      command: process.execPath,
      argv: [
        path.join(ROOT, 'scripts', 'run_recording_job.mjs'),
        '--mode', 'upload',
        '--profile', model.profile.id,
        '--vlm-config', campaign.registry.path,
        '--tasks', plan.taskPath,
        '--out-root', model.outputDir,
        '--',
        '--web-url', campaign.webUrl,
        '--process-interval-s', String(campaign.frameScheduler.processIntervalS),
        '--frames-per-batch', String(campaign.frameScheduler.framesPerBatch),
        '--task-retries', String(campaign.defaults.maxTaskRetries),
        '--retry-gap-s', '20',
        '--health-retries', '20',
        '--health-interval-s', '5',
        '--stream-ready-timeout-s', '180',
        '--video-upload-timeout-s', '900',
        '--post-video-response-timeout-s', '180',
        '--stream-start-retries', '1',
        '--stream-start-retry-gap-s', '5',
        '--attempt-timeout-margin-s', '600',
        '--fresh-browser-per-attempt',
      ],
    };
  }
  const argv = [
    path.join(ROOT, 'scripts', 'supervise_scaffold_recording.mjs'),
    '--tasks', plan.taskPath,
    '--out', model.outputDir,
    '--attempt-work-root', campaign.defaults.attemptWorkRoot,
    '--web-url', campaign.webUrl,
    '--profile-id', model.runner === 'joyai_scaffold'
      ? `${model.id}-joyai-scaffold`
      : `${model.id}-system-prompt`,
    '--model', model.profile.model,
    '--upstream-api-base', model.profile.api_base,
    '--upstream-protocol', model.profile.upstream_protocol,
    '--input-transport', model.profile.input_transport,
    '--api-key-env', model.profile.api_key_env,
    '--process-interval-s', String(campaign.frameScheduler.processIntervalS),
    '--frames-per-batch', String(campaign.frameScheduler.framesPerBatch),
    '--max-retries', String(campaign.defaults.maxTaskRetries),
    '--max-failed-task-recovery-batches', String(
      campaign.defaults.maxFailedTaskRecoveryBatches,
    ),
    '--max-task-wall-s', String(campaign.defaults.maxTaskWallS),
    '--max-provider-query-start-drift-s', String(
      campaign.defaults.maxProviderQueryStartDriftS,
    ),
    '--upstream-sdk-max-retries', String(campaign.upstreamRequestPolicy.sdkMaxRetries),
    '--non-query-request-timeout-s', String(
      campaign.upstreamRequestPolicy.nonQueryTimeoutS,
    ),
    '--query-request-timeout-s', String(campaign.upstreamRequestPolicy.queryTimeoutS),
    '--provider-warmup-timeout-s', String(campaign.upstreamRequestPolicy.warmupTimeoutS),
    '--progress-interval-s', String(campaign.defaults.progressIntervalS),
  ];
  if (model.runner === 'joyai_scaffold') {
    argv.push('--interaction-scaffold', 'joyai-official-live-adapter');
  }
  if (model.profile.input_transport === 'native-video-realtime') {
    argv.push(
      '--native-video-schema', model.profile.native_video_schema,
      '--realtime-protocol', model.profile.realtime_protocol,
      '--realtime-api-base', model.profile.realtime_api_base,
      '--realtime-query-mode', model.profile.realtime_query_mode,
      '--native-streaming-mode-policy', model.nativeStreamingModePolicy,
      '--native-system-prompt-transport', model.nativeSystemPromptTransport,
      '--native-frame-clock', model.nativeFrameClock,
      '--max-realtime-session-s', String(model.profile.max_realtime_session_s),
    );
    if (model.nativeQueryAudioManifest) {
      argv.push('--native-query-audio-manifest', model.nativeQueryAudioManifest);
    }
  }
  if (model.scaffoldDeployment) {
    argv.push(
      '--deployed-adapter-api-base', model.scaffoldDeployment.apiBase,
      '--deployed-adapter-access-token-env', model.scaffoldDeployment.accessTokenEnv,
      '--deployed-adapter-audit-path', model.scaffoldDeployment.auditPath,
    );
  } else {
    argv.push('--proxy-port', String(model.proxyPort));
  }
  for (const alias of model.profile.backend_aliases) argv.push('--backend-alias', alias);
  if (model.disableThinking) argv.push('--disable-thinking');
  if (model.continueAfterTaskFailure) argv.push('--continue-after-task-failure');
  return { command: process.execPath, argv };
}

function quoteArg(value) {
  const text = String(value);
  return /^[A-Za-z0-9_./:=@+-]+$/.test(text) ? text : `'${text.replaceAll("'", "'\\''")}'`;
}

function summaryPathFor(model) {
  return model.runner === 'adapter'
    ? path.join(model.outputDir, model.profile.id, 'run_summary.json')
    : path.join(model.outputDir, 'run_summary.json');
}

function inspectModel(
  model,
  tasksTotal,
  maxTaskRetries,
  maxFailedTaskRecoveryBatches = 1,
) {
  const summaryPath = summaryPathFor(model);
  const summary = readJson(summaryPath);
  const rows = Array.isArray(summary?.task_results) ? summary.task_results : [];
  const okRows = rows.filter((row) => ['ok', 'skipped'].includes(row.status));
  const failedRows = rows.filter((row) => !['ok', 'skipped'].includes(row.status));
  const filesOk = okRows.every((row) => {
    const video = String(row.task_mp4 || '');
    return video && fs.existsSync(video) && fs.statSync(video).size > 0;
  });
  const complete = summary?.status === 'complete'
    && rows.length === tasksTotal
    && failedRows.length === 0
    && filesOk;
  const summaryTerminalTaskFailures = rows.length === tasksTotal
    && failedRows.length > 0
    && failedRows.every((row) => Number(row.attempts || 0) >= maxTaskRetries + 1);
  const supervisor = ['prompt_proxy', 'joyai_scaffold'].includes(model.runner)
    ? readJson(path.join(model.outputDir, 'supervisor_state.json'))
    : null;
  const supervisorResolvedTasks = new Set([
    ...Object.keys(supervisor?.results || {}),
    ...Object.keys(supervisor?.failures || {}),
  ]).size;
  const interruptedTaskResumeAvailable = Boolean(
    supervisor
    && supervisorResolvedTasks < tasksTotal
    && ['stopped', 'running'].includes(String(supervisor.status || ''))
  );
  const terminalTaskFailures = summaryTerminalTaskFailures
    && !interruptedTaskResumeAvailable;
  const recoveryBatchCount = Math.max(
    Number(summary?.attempt_audit?.recovery_batch_count || 0),
    Number(supervisor?.recovery_batch_count || 0),
  );
  const failedSupervisorTasks = Object.entries(supervisor?.failures || {})
    .filter(([, failure]) => failure?.status === 'failed')
    .map(([taskId]) => taskId);
  const taskWallWindowOpen = (taskId) => {
    const startedAtMs = Date.parse(supervisor?.task_wall_pass_started_at?.[taskId] || '');
    const maxTaskWallS = Number(supervisor?.max_task_wall_s || 0);
    return !Number.isFinite(startedAtMs)
      || maxTaskWallS <= 0
      || Date.now() < startedAtMs + maxTaskWallS * 1000;
  };
  const recoveryBatchCountForTask = (taskId) => (
    Array.isArray(supervisor?.recovery_batches)
      ? supervisor.recovery_batches.filter((batch) => (
        Array.isArray(batch?.tasks)
        && batch.tasks.some((task) => task?.task_id === taskId)
      )).length
      : 0
  );
  const existingRecoveryContinuationAvailable = failedSupervisorTasks.length > 0
    && failedSupervisorTasks.every((taskId) => (
      Number(supervisor?.retry_attempt_limits?.[taskId] || 0)
      > Number(supervisor?.attempts?.[taskId] || 0)
      && taskWallWindowOpen(taskId)
    ));
  const taskRecoveryAvailable = failedSupervisorTasks.length > 0
    && failedSupervisorTasks.every((taskId) => (
      (
        Number(supervisor?.retry_attempt_limits?.[taskId] || 0)
        > Number(supervisor?.attempts?.[taskId] || 0)
        && taskWallWindowOpen(taskId)
      )
      || recoveryBatchCountForTask(taskId) < maxFailedTaskRecoveryBatches
    ));
  return {
    status: complete ? 'complete' : (summary?.status || 'pending'),
    complete,
    terminal_task_failures: terminalTaskFailures,
    interrupted_task_resume_available: interruptedTaskResumeAvailable,
    failed_task_recovery_available: Boolean(
      ['prompt_proxy', 'joyai_scaffold'].includes(model.runner)
      && (
        taskRecoveryAvailable
        || (
          failedSupervisorTasks.length === 0
          && recoveryBatchCount < maxFailedTaskRecoveryBatches
        )
      )
    ),
    failed_task_recovery_continuation_available: existingRecoveryContinuationAvailable,
    tasks_total: tasksTotal,
    tasks_ok: okRows.length,
    tasks_failed: failedRows.length,
    current_task: supervisor?.current_task || null,
    summary_path: summaryPath,
    last_error: complete ? '' : (summary?.last_error || supervisor?.last_error || ''),
  };
}

function pidMatches(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    if (!processExists(pid)) return false;
    const commandLine = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8');
    return commandLine.includes('run_recording_campaign.mjs');
  } catch { return false; }
}

function processExists(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function acquireLock(lockPath) {
  const attempt = () => {
    const fd = fs.openSync(lockPath, 'wx', 0o600);
    fs.writeFileSync(fd, `${JSON.stringify({ pid: process.pid, started_at: new Date().toISOString() })}\n`);
    return fd;
  };
  try { return attempt(); } catch (error) {
    if (error.code !== 'EEXIST') throw error;
    const old = readJson(lockPath);
    if (pidMatches(Number(old?.pid))) throw new Error(`A recording campaign is already running with pid ${old.pid}`);
    fs.unlinkSync(lockPath);
    return attempt();
  }
}

function waitChild(child) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    child.once('error', (error) => finish({ code: 1, signal: '', error: error.message }));
    child.once('exit', (code, signal) => finish({ code: code ?? 1, signal: signal || '', error: '' }));
  });
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

function dynamicModelDirectory(runRoot) {
  return path.join(runRoot, 'dynamic_models');
}

function dynamicModelStatePath(runRoot, modelId) {
  return path.join(dynamicModelDirectory(runRoot), `${modelId}.json`);
}

function acquireDynamicModelLock(runRoot, modelId) {
  const directory = dynamicModelDirectory(runRoot);
  fs.mkdirSync(directory, { recursive: true });
  const lockPath = path.join(directory, `${modelId}.lock`);
  const attempt = () => {
    const fd = fs.openSync(lockPath, 'wx', 0o600);
    fs.writeFileSync(fd, `${JSON.stringify({
      pid: process.pid,
      model_id: modelId,
      started_at: new Date().toISOString(),
    })}\n`);
    return { fd, lockPath };
  };
  try {
    return attempt();
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
    const old = readJson(lockPath);
    if (pidMatches(Number(old?.pid))) {
      throw new Error(`Model ${modelId} already has a dynamic worker with pid ${old.pid}`);
    }
    fs.unlinkSync(lockPath);
    return attempt();
  }
}

async function releaseDynamicModelLock(lock) {
  if (!lock) return;
  try { fs.closeSync(lock.fd); } catch {}
  await fsp.unlink(lock.lockPath).catch(() => {});
}

function readDynamicModelStates(
  runRoot,
  maxTaskRetries = 5,
  maxFailedTaskRecoveryBatches = 1,
) {
  const directory = dynamicModelDirectory(runRoot);
  if (!fs.existsSync(directory)) return {};
  const states = {};
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
    const state = readJson(path.join(directory, entry.name));
    if (!state?.model_id) continue;
    const workerLive = ['preparing', 'running', 'retry_wait'].includes(state.status)
      && pidMatches(Number(state.pid));
    const inspection = state.output && state.runner
      ? inspectModel(
        { runner: state.runner, outputDir: state.output },
        Number(state.tasks_total || 0),
        maxTaskRetries,
        maxFailedTaskRecoveryBatches,
      )
      : null;
    states[state.model_id] = {
      ...state,
      ...(inspection || {}),
      // The dynamic worker lifecycle is authoritative while its supervisor is live.
      // inspectModel supplies fresh task counts/current_task between worker exits.
      status: inspection?.complete ? 'complete' : state.status,
      live: workerLive,
      dynamic: true,
    };
  }
  return states;
}

async function prepareCampaign(campaign, args, { persistent }) {
  const temporary = await fsp.mkdtemp(path.join(persistent ? campaign.outputRoot : os.tmpdir(), '.recording-plan-'));
  try {
    const built = await buildTaskManifest(campaign, temporary);
    const provenance = campaignProvenance(campaign, built);
    const runId = createRunId(args.runId);
    if (!persistent) {
      const root = path.join(campaign.outputRoot, 'runs', runId);
      return {
        runId,
        provenance,
        built,
        root,
        taskPath: path.join(root, 'tasks.jsonl'),
        models: campaign.models.map((model) => ({ ...model, outputDir: path.join(root, 'models', model.id) })),
      };
    }
    return {
      provenance,
      built,
      ...(await materializePlan(campaign, built, provenance, runId)),
    };
  } finally {
    if (!persistent) await fsp.rm(temporary, { recursive: true, force: true });
    else await fsp.rm(temporary, { recursive: true, force: true });
  }
}

function prepareCurrentResumePlan(
  campaign,
  current,
  { allowWebUiMigration = false, allowRunning = false } = {},
) {
  if (!current?.state_path) throw new Error('No current campaign is available to resume');
  if (!allowRunning && pidMatches(Number(current.pid))) {
    throw new Error(`Current campaign is still running with pid ${current.pid}`);
  }
  const statePath = path.resolve(String(current.state_path));
  const relativeStatePath = path.relative(campaign.outputRoot, statePath);
  if (!relativeStatePath || relativeStatePath.startsWith('..') || path.isAbsolute(relativeStatePath)) {
    throw new Error('Current campaign state is outside the configured output root');
  }
  const previous = readJson(statePath);
  if (!previous?.campaign_id) throw new Error(`Unable to read current campaign state: ${statePath}`);
  if (!allowRunning && pidMatches(Number(previous.pid))) {
    throw new Error(`Requested campaign is still running with pid ${previous.pid}`);
  }
  const previousWebUrl = normalizeRecordingWebUrl(previous.fixed_web_url);
  const configuredWebUrl = normalizeRecordingWebUrl(campaign.webUrl);
  const webUiChanged = previousWebUrl !== configuredWebUrl;
  if (webUiChanged && !allowWebUiMigration) {
    throw new Error(
      'Current campaign used a different recording WebUI; pass --allow-webui-migration '
      + 'to preserve validated task outputs and audit the route change',
    );
  }
  if (
    webUiChanged
    && configuredWebUrl !== normalizeRecordingWebUrl(recordingWebUrl())
  ) {
    throw new Error('WebUI migration is only allowed toward the canonical campaign WebUI');
  }
  if (previous.input_mode !== 'upload' || previous.evaluation_protocol !== EVALUATION_PROTOCOL) {
    throw new Error('Current campaign protocol is incompatible with formal upload recording');
  }

  const root = path.dirname(statePath);
  const spec = readJson(path.join(root, 'campaign_spec.json'));
  if (!spec || spec.campaign_id !== previous.campaign_id) {
    throw new Error(`Current campaign spec is missing or inconsistent: ${root}`);
  }
  if (!sameJson(spec.upstream_request_policy || null, campaign.upstreamRequestPolicy)) {
    throw new Error(
      'Current campaign used a different upstream request policy; start a new run',
    );
  }
  const taskPath = path.resolve(String(spec.task_manifest || previous.task_manifest || ''));
  if (!fs.existsSync(taskPath)) throw new Error('Current run task manifest is missing');
  const tasks = readTasks(taskPath);
  const plannedModels = Array.isArray(spec.model_order) ? spec.model_order : [];
  const models = campaign.models.map((model) => {
    const planned = plannedModels.find((item) => item.id === model.id);
    if (planned && planned.runner !== model.runner) {
      throw new Error(`Model runner changed since the current run started: ${model.id}`);
    }
    const outputDir = planned
      ? path.resolve(String(planned.output || ''))
      : path.join(root, 'models', model.id);
    const relativeOutput = path.relative(root, outputDir);
    if (!relativeOutput || relativeOutput.startsWith('..') || path.isAbsolute(relativeOutput)) {
      throw new Error(`Model output is outside the current campaign root: ${model.id}`);
    }
    return { ...model, outputDir };
  });
  const reportPath = path.join(root, 'tasks.report.json');
  const report = readJson(reportPath) || { tasks_written: tasks.length, missing_count: 0 };
  return {
    runId: previous.campaign_id,
    provenance: {
      taskSha256: sha256File(taskPath),
      workbookSha256: spec.workbook_sha256,
      sources: spec.source_video_signatures || [],
    },
    built: { taskPath, reportPath, tasks, report },
    root,
    taskPath,
    reportPath,
    models,
    spec,
    webUiMigration: webUiChanged ? {
      from: previousWebUrl,
      to: configuredWebUrl,
    } : null,
    resumedCurrent: true,
  };
}

async function dryRun(args) {
  const campaign = loadCampaign(args);
  const resumeRequested = Boolean(args.resumeCurrent || args.resumeRunId);
  const current = args.resumeCurrent
    ? readJson(path.join(campaign.outputRoot, 'current.json'))
    : args.resumeRunId
    ? {
      pid: null,
      state_path: path.join(
        campaign.outputRoot,
        'runs',
        args.resumeRunId,
        'campaign_state.json',
      ),
    }
    : null;
  const plan = resumeRequested
    ? prepareCurrentResumePlan(campaign, current, {
      allowWebUiMigration: args.allowWebUiMigration,
      allowRunning: true,
    })
    : await prepareCampaign(campaign, args, { persistent: false });
  const environment = requiredEnvironment(campaign);
  const executables = requiredExecutables(campaign);
  const commands = plan.models.map((model) => {
    const command = modelCommand(campaign, plan, model);
    return {
      model: model.id,
      output: model.outputDir,
      command: [command.command, ...command.argv].map(quoteArg).join(' '),
    };
  });
  const launchReady = environment.every((item) => item.set)
    && executables.every((item) => item.path);
  console.log(JSON.stringify({
    status: resumeRequested ? 'dry_run_resume' : 'dry_run',
    launch_ready: launchReady,
    fixed_web_url: campaign.webUrl,
    network: campaign.network,
    input_mode: 'upload',
    evaluation_protocol: campaign.evaluationProtocol,
    frame_scheduler: campaign.frameScheduler,
    upstream_request_policy: campaign.upstreamRequestPolicy,
    run_id: plan.runId,
    campaign_id: plan.runId,
    output_root: plan.root,
    workbook: campaign.source.xlsx,
    requested_task_ids: campaign.source.taskIds,
    tasks_total: plan.built.tasks.length,
    unique_videos: plan.provenance.sources.length,
    multi_round_tasks: plan.built.report.multi_round_tasks,
    missing_mappings: plan.built.report.missing_count,
    max_task_retries: campaign.defaults.maxTaskRetries,
    max_failed_task_recovery_batches: campaign.defaults.maxFailedTaskRecoveryBatches,
    max_parallel_models: Math.min(campaign.defaults.maxParallelModels, campaign.models.length),
    environment,
    executables,
    model_order: campaign.models.map((model) => model.id),
    commands,
  }, null, 2));
  if (!launchReady) process.exitCode = 2;
}

async function addModels(args) {
  const campaign = loadCampaign(args);
  await fsp.mkdir(campaign.outputRoot, { recursive: true });
  const pointer = args.resumeRunId
    ? {
      pid: null,
      state_path: path.join(
        campaign.outputRoot,
        'runs',
        args.resumeRunId,
        'campaign_state.json',
      ),
    }
    : readJson(path.join(campaign.outputRoot, 'current.json'));
  const plan = prepareCurrentResumePlan(campaign, pointer, { allowRunning: true });
  const parentStatePath = path.join(plan.root, 'campaign_state.json');
  const parentState = readJson(parentStatePath);
  const parentLive = pidMatches(Number(parentState?.pid));

  for (const model of plan.models) {
    const inspection = inspectModel(
      model,
      plan.built.tasks.length,
      campaign.defaults.maxTaskRetries,
      campaign.defaults.maxFailedTaskRecoveryBatches,
    );
    if (
      parentLive
      && Array.isArray(parentState?.model_order)
      && parentState.model_order.includes(model.id)
      && !inspection.complete
    ) {
      throw new Error(
        `Model ${model.id} is already scheduled by live campaign pid ${parentState.pid}`,
      );
    }
    const activeChildPid = Number(parentState?.current_child_pids?.[model.id] || 0);
    if (processExists(activeChildPid)) {
      throw new Error(`Model ${model.id} is already running with child pid ${activeChildPid}`);
    }
  }

  const environment = requiredEnvironment(campaign);
  const missingEnvironment = environment.filter((item) => !item.set).map((item) => item.name);
  if (missingEnvironment.length) {
    throw new Error(`Missing required environment variables: ${missingEnvironment.join(', ')}`);
  }
  const missingExecutables = requiredExecutables(campaign)
    .filter((item) => !item.path)
    .map((item) => item.command);
  if (missingExecutables.length) {
    throw new Error(`Missing required executables: ${missingExecutables.join(', ')}`);
  }
  const networkPreflight = await verifyRecordingNetwork(campaign);
  const webUiPreflight = await verifyRecordingWebUi(campaign);

  let stopping = false;
  const children = new Map();
  const terminate = () => {
    stopping = true;
    for (const child of children.values()) {
      if (!child?.pid) continue;
      try { child.kill('SIGTERM'); } catch {}
    }
  };
  process.once('SIGINT', terminate);
  process.once('SIGTERM', terminate);

  const runOne = async (model) => {
    const lock = acquireDynamicModelLock(plan.root, model.id);
    const workerStatePath = dynamicModelStatePath(plan.root, model.id);
    const previous = readJson(workerStatePath);
    const previousInfrastructureStarts = Number(
      previous?.infrastructure_starts_total
      ?? previous?.infrastructure_starts
      ?? 0,
    );
    let workerState = {
      version: 1,
      kind: 'dynamic_model_worker',
      campaign_id: plan.runId,
      model_id: model.id,
      model: model.profile.model,
      runner: model.runner,
      status: 'preparing',
      live: true,
      dynamic: true,
      pid: process.pid,
      child_pid: null,
      parent_campaign_pid: Number(parentState?.pid || 0) || null,
      started_at: previous?.started_at || new Date().toISOString(),
      updated_at: new Date().toISOString(),
      finished_at: null,
      fixed_web_url: campaign.webUrl,
      evaluation_protocol: campaign.evaluationProtocol,
      task_manifest: plan.taskPath,
      tasks_total: plan.built.tasks.length,
      output: model.outputDir,
      profile: safeProfileSnapshot(model.profile),
      config_path: campaign.configPath,
      config_sha256: campaign.configSha256,
      frame_scheduler: campaign.frameScheduler,
      network_preflight: networkPreflight,
      webui_preflight: webUiPreflight,
      infrastructure_starts: previousInfrastructureStarts,
      infrastructure_starts_total: previousInfrastructureStarts,
      manager_infrastructure_starts: 0,
      log_path: args.logPath,
      last_error: '',
    };
    const persist = async (extra = {}) => {
      const inspection = inspectModel(
        model,
        plan.built.tasks.length,
        campaign.defaults.maxTaskRetries,
        campaign.defaults.maxFailedTaskRecoveryBatches,
      );
      workerState = {
        ...workerState,
        ...inspection,
        ...extra,
        updated_at: new Date().toISOString(),
      };
      await writeJsonAtomic(workerStatePath, workerState);
    };

    try {
      let inspection = inspectModel(
        model,
        plan.built.tasks.length,
        campaign.defaults.maxTaskRetries,
        campaign.defaults.maxFailedTaskRecoveryBatches,
      );
      if (inspection.complete) {
        await persist({
          status: 'complete',
          live: false,
          finished_at: new Date().toISOString(),
          last_error: '',
        });
        return;
      }
      await persist({ status: 'running' });
      while (!stopping && !inspection.complete) {
        const retryFailedTasks = Boolean(
          inspection.terminal_task_failures
          && inspection.failed_task_recovery_available,
        );
        if (inspection.terminal_task_failures && !retryFailedTasks) {
          await persist({
            status: 'incomplete',
            live: false,
            finished_at: new Date().toISOString(),
            last_error: inspection.last_error
              || 'One or more tasks exhausted the bounded retry budget',
          });
          return;
        }
        const starts = Number(workerState.manager_infrastructure_starts || 0);
        if (starts >= campaign.defaults.maxInfrastructureRestarts + 1) {
          await persist({
            status: 'failed',
            live: false,
            finished_at: new Date().toISOString(),
            last_error: `Exceeded ${campaign.defaults.maxInfrastructureRestarts} infrastructure restarts in this manager`,
          });
          return;
        }
        const command = modelCommand(campaign, plan, model);
        if (retryFailedTasks) command.argv.push('--retry-failed-tasks');
        const child = spawn(command.command, command.argv, {
          cwd: ROOT,
          env: process.env,
          stdio: 'inherit',
        });
        children.set(model.id, child);
        const totalStarts = Number(workerState.infrastructure_starts_total || 0) + 1;
        await persist({
          status: 'running',
          child_pid: child.pid || null,
          infrastructure_starts: totalStarts,
          infrastructure_starts_total: totalStarts,
          manager_infrastructure_starts: starts + 1,
          retrying_failed_tasks: retryFailedTasks,
          command: [command.command, ...command.argv].map(quoteArg).join(' '),
        });
        const exit = await waitChild(child);
        children.delete(model.id);
        inspection = inspectModel(
          model,
          plan.built.tasks.length,
          campaign.defaults.maxTaskRetries,
          campaign.defaults.maxFailedTaskRecoveryBatches,
        );
        await persist({
          child_pid: null,
          last_exit_code: exit.code,
          last_exit_signal: exit.signal,
          last_spawn_error: exit.error || '',
        });
        if (!inspection.complete && !inspection.terminal_task_failures && !stopping) {
          await persist({ status: 'retry_wait' });
          await sleep(campaign.defaults.phaseRestartBackoffS * 1000);
        }
      }
      if (stopping) {
        await persist({
          status: 'stopped',
          live: false,
          child_pid: null,
          finished_at: new Date().toISOString(),
          last_error: 'Dynamic model worker stopped by signal',
        });
      } else {
        await persist({
          status: 'complete',
          live: false,
          child_pid: null,
          finished_at: new Date().toISOString(),
          last_error: '',
        });
      }
    } catch (error) {
      await persist({
        status: stopping ? 'stopped' : 'failed',
        live: false,
        child_pid: null,
        finished_at: new Date().toISOString(),
        last_error: error.message || String(error),
      }).catch(() => {});
      throw error;
    } finally {
      children.delete(model.id);
      await releaseDynamicModelLock(lock);
    }
  };

  try {
    console.log(
      `Dynamically adding ${plan.models.length} model(s) to ${plan.runId}: `
      + plan.models.map((model) => model.id).join(', '),
    );
    console.log('Dynamic additions have no campaign-level model concurrency cap.');
    const results = await Promise.allSettled(plan.models.map((model) => runOne(model)));
    const failures = results
      .map((result, index) => ({ result, model: plan.models[index].id }))
      .filter(({ result }) => result.status === 'rejected');
    if (failures.length) {
      throw new Error(failures.map(({ model, result }) => (
        `${model}: ${result.reason?.message || String(result.reason)}`
      )).join('; '));
    }
  } finally {
    terminate();
    process.removeListener('SIGINT', terminate);
    process.removeListener('SIGTERM', terminate);
  }
}

async function runCampaign(args) {
  const campaign = loadCampaign(args);
  await fsp.mkdir(campaign.outputRoot, { recursive: true });
  const lockPath = path.join(campaign.outputRoot, '.campaign.lock');
  const currentPath = path.join(campaign.outputRoot, 'current.json');
  const resumeRequested = Boolean(args.resumeCurrent || args.resumeRunId);
  const resumePointer = args.resumeCurrent
    ? readJson(currentPath)
    : args.resumeRunId
    ? {
      pid: null,
      state_path: path.join(
        campaign.outputRoot,
        'runs',
        args.resumeRunId,
        'campaign_state.json',
      ),
    }
    : null;
  const lockFd = acquireLock(lockPath);
  const children = new Map();
  let stopping = false;
  let state = null;
  let statePath = '';
  const startedAt = new Date().toISOString();
  const writeCurrent = async (extra = {}) => writeJsonAtomic(currentPath, {
    pid: process.pid,
    status: state?.status || 'preparing',
    campaign_id: state?.campaign_id || '',
    state_path: statePath,
    log_path: args.logPath,
    config_path: campaign.configPath,
    fixed_web_url: campaign.webUrl,
    input_mode: 'upload',
    started_at: state?.started_at || startedAt,
    updated_at: new Date().toISOString(),
    ...extra,
  });
  const terminate = () => {
    stopping = true;
    for (const child of children.values()) {
      if (child?.pid) {
        try { child.kill('SIGTERM'); } catch {}
      }
    }
  };
  process.once('SIGINT', terminate);
  process.once('SIGTERM', terminate);

  try {
    if (!resumeRequested) await writeCurrent();
    const environment = requiredEnvironment(campaign);
    const missingEnvironment = environment.filter((item) => !item.set).map((item) => item.name);
    if (missingEnvironment.length) {
      throw new Error(`Missing required environment variables: ${missingEnvironment.join(', ')}`);
    }
    const missingExecutables = requiredExecutables(campaign).filter((item) => !item.path).map((item) => item.command);
    if (missingExecutables.length) throw new Error(`Missing required executables: ${missingExecutables.join(', ')}`);

    const networkPreflight = await verifyRecordingNetwork(campaign);
    const webUiPreflight = await verifyRecordingWebUi(campaign);

    const plan = resumeRequested
      ? prepareCurrentResumePlan(campaign, resumePointer, {
        allowWebUiMigration: args.allowWebUiMigration,
      })
      : await prepareCampaign(campaign, args, { persistent: true });
    statePath = path.join(plan.root, 'campaign_state.json');
    const previous = readJson(statePath);
    state = {
      version: 2,
      run_id: plan.runId,
      campaign_id: plan.runId,
      status: 'running',
      pid: process.pid,
      started_at: previous?.started_at || startedAt,
      updated_at: new Date().toISOString(),
      finished_at: null,
      fixed_web_url: campaign.webUrl,
      input_mode: 'upload',
      evaluation_protocol: campaign.evaluationProtocol,
      network_preflight: networkPreflight,
      webui_preflight: webUiPreflight,
      webui_migrations: [
        ...(Array.isArray(previous?.webui_migrations) ? previous.webui_migrations : []),
        ...(plan.webUiMigration ? [{
          ...plan.webUiMigration,
          migrated_at: new Date().toISOString(),
          explicit: true,
        }] : []),
      ],
      task_manifest: plan.taskPath,
      task_manifest_sha256: plan.provenance.taskSha256,
      tasks_total: plan.built.tasks.length,
      unique_videos: plan.provenance.sources.length,
      hashes_are_provenance_only: true,
      model_order: plan.models.map((model) => model.id),
      max_parallel_models: Math.min(campaign.defaults.maxParallelModels, plan.models.length),
      max_failed_task_recovery_batches: campaign.defaults.maxFailedTaskRecoveryBatches,
      resumed_current: args.resumeCurrent,
      resumed_run_id: args.resumeRunId || '',
      current_model: '',
      current_child_pid: null,
      current_models: [],
      current_child_pids: {},
      infrastructure_starts: previous?.infrastructure_starts || {},
      models: previous?.models || {},
      last_error: '',
      log_path: args.logPath,
    };
    const refresh = () => {
      for (const model of plan.models) {
        const inspection = inspectModel(
          model,
          state.tasks_total,
          campaign.defaults.maxTaskRetries,
          campaign.defaults.maxFailedTaskRecoveryBatches,
        );
        if (children.has(model.id) && !inspection.complete) {
          inspection.status = 'running';
        }
        state.models[model.id] = {
          ...(state.models[model.id] || {}),
          ...inspection,
          runner: model.runner,
          model: model.profile.model,
          output: model.outputDir,
          infrastructure_starts: Number(state.infrastructure_starts[model.id] || 0),
        };
      }
      state.current_models = [...children.keys()];
      state.current_child_pids = Object.fromEntries(
        [...children.entries()].map(([id, child]) => [id, child.pid || null]),
      );
      state.current_model = state.current_models[0] || '';
      state.current_child_pid = state.current_model
        ? state.current_child_pids[state.current_model]
        : null;
    };
    let persistChain = Promise.resolve();
    const persist = () => {
      const write = async () => {
        refresh();
        state.updated_at = new Date().toISOString();
        await writeJsonAtomic(statePath, state);
        await writeCurrent();
      };
      persistChain = persistChain.then(write, write);
      return persistChain;
    };
    await persist();
    const timer = setInterval(() => persist().catch(() => {}), 30000);
    timer.unref();

    try {
      const runModel = async (model) => {
        let inspection = inspectModel(
          model,
          state.tasks_total,
          campaign.defaults.maxTaskRetries,
          campaign.defaults.maxFailedTaskRecoveryBatches,
        );
        if (inspection.complete) {
          await persist();
          return;
        }
        while (!stopping && !inspection.complete) {
          const retryFailedTasks = Boolean(
            inspection.terminal_task_failures
            && inspection.failed_task_recovery_available,
          );
          if (inspection.terminal_task_failures && !retryFailedTasks) {
            state.models[model.id].status = 'incomplete';
            state.models[model.id].last_error ||= 'One or more tasks exhausted the bounded retry budget';
            break;
          }
          const starts = Number(state.infrastructure_starts[model.id] || 0);
          if (starts >= campaign.defaults.maxInfrastructureRestarts + 1) {
            state.models[model.id].status = 'failed';
            state.models[model.id].last_error = `Exceeded ${campaign.defaults.maxInfrastructureRestarts} infrastructure restarts`;
            break;
          }
          const command = modelCommand(campaign, plan, model);
          if (retryFailedTasks) command.argv.push('--retry-failed-tasks');
          state.infrastructure_starts[model.id] = starts + 1;
          state.models[model.id] = {
            ...(state.models[model.id] || {}),
            status: 'running',
            infrastructure_starts: starts + 1,
            retrying_failed_tasks: retryFailedTasks,
            command: [command.command, ...command.argv].map(quoteArg).join(' '),
          };
          const child = spawn(command.command, command.argv, {
            cwd: ROOT,
            env: process.env,
            stdio: 'inherit',
          });
          children.set(model.id, child);
          await persist();
          const exit = await waitChild(child);
          children.delete(model.id);
          inspection = inspectModel(
            model,
            state.tasks_total,
            campaign.defaults.maxTaskRetries,
            campaign.defaults.maxFailedTaskRecoveryBatches,
          );
          state.models[model.id] = {
            ...(state.models[model.id] || {}),
            ...inspection,
            last_exit_code: exit.code,
            last_exit_signal: exit.signal,
            last_spawn_error: exit.error || '',
          };
          await persist();
          if (!inspection.complete && !inspection.terminal_task_failures && !stopping) {
            await sleep(campaign.defaults.phaseRestartBackoffS * 1000);
          }
        }
        await persist();
      };

      let nextModelIndex = 0;
      const workerErrors = [];
      const runWorker = async () => {
        while (!stopping) {
          const index = nextModelIndex;
          nextModelIndex += 1;
          if (index >= plan.models.length) return;
          const model = plan.models[index];
          try {
            await runModel(model);
          } catch (error) {
            const activeChild = children.get(model.id);
            if (activeChild?.pid) {
              try { activeChild.kill('SIGTERM'); } catch {}
            }
            children.delete(model.id);
            workerErrors.push({ model: model.id, error });
            state.models[model.id] = {
              ...(state.models[model.id] || {}),
              status: 'failed',
              last_error: error.message || String(error),
            };
            await persist().catch(() => {});
          }
        }
      };
      const workerCount = Math.min(campaign.defaults.maxParallelModels, plan.models.length);
      await Promise.allSettled(Array.from({ length: workerCount }, () => runWorker()));
      if (workerErrors.length) {
        throw new Error(workerErrors
          .map(({ model, error }) => `${model}: ${error.message || String(error)}`)
          .join('; '));
      }
      refresh();
      state.current_model = '';
      state.current_child_pid = null;
      state.current_models = [];
      state.current_child_pids = {};
      if (stopping) {
        state.status = 'stopped';
        state.last_error = 'Campaign stopped by signal';
      } else {
        const incomplete = Object.values(state.models).filter((model) => !model.complete);
        state.status = incomplete.length ? 'complete_with_errors' : 'complete';
        state.last_error = incomplete.length
          ? `Incomplete models: ${incomplete.map((model) => model.model).join(', ')}`
          : '';
      }
      state.finished_at = new Date().toISOString();
      await persist();
      if (state.status === 'complete_with_errors') process.exitCode = 1;
    } finally {
      clearInterval(timer);
    }
  } catch (error) {
    const wasStopping = stopping;
    if (!wasStopping) terminate();
    if (state) {
      state.status = wasStopping ? 'stopped' : 'failed';
      state.last_error = error.message;
      state.current_child_pid = null;
      state.current_models = [];
      state.current_child_pids = {};
      state.finished_at = new Date().toISOString();
      if (statePath) await writeJsonAtomic(statePath, state).catch(() => {});
    }
    if (!(resumeRequested && !state)) {
      await writeCurrent({ status: wasStopping ? 'stopped' : 'failed', error: error.message }).catch(() => {});
    }
    throw error;
  } finally {
    process.removeListener('SIGINT', terminate);
    process.removeListener('SIGTERM', terminate);
    try { fs.closeSync(lockFd); } catch {}
    await fsp.unlink(lockPath).catch(() => {});
  }
}

function campaignOutputRoot(args) {
  const campaign = loadCampaign(args);
  return { campaign, outputRoot: campaign.outputRoot };
}

function currentState(args) {
  const { campaign, outputRoot } = campaignOutputRoot(args);
  const requestedStatePath = args.resumeRunId
    ? path.join(outputRoot, 'runs', args.resumeRunId, 'campaign_state.json')
    : '';
  const requestedState = requestedStatePath ? readJson(requestedStatePath) : null;
  const current = requestedStatePath
    ? {
      pid: requestedState?.pid || null,
      status: requestedState?.status || 'unknown',
      campaign_id: requestedState?.campaign_id || args.resumeRunId,
      state_path: requestedStatePath,
      log_path: requestedState?.log_path || '',
    }
    : readJson(path.join(outputRoot, 'current.json'));
  const state = requestedStatePath
    ? requestedState
    : current?.state_path ? readJson(current.state_path) : null;
  return { campaign, outputRoot, current, state };
}

async function status(args) {
  const { campaign, outputRoot, current, state } = currentState(args);
  if (!current) {
    console.log(args.json ? JSON.stringify({ status: 'not_started', output_root: outputRoot }, null, 2) : 'No recording campaign has been started.');
    return;
  }
  const mainLive = pidMatches(Number(current.pid));
  const runRoot = current.state_path ? path.dirname(path.resolve(current.state_path)) : '';
  const dynamicModels = runRoot
    ? readDynamicModelStates(
      runRoot,
      campaign.defaults.maxTaskRetries,
      campaign.defaults.maxFailedTaskRecoveryBatches,
    )
    : {};
  const dynamicWorkerPids = [...new Set(Object.values(dynamicModels)
    .filter((model) => model.live)
    .map((model) => Number(model.pid))
    .filter((pid) => pid > 0))];
  const live = mainLive || dynamicWorkerPids.length > 0;
  const baseStatus = state?.status || current.status || (mainLive ? 'preparing' : 'unknown');
  const payload = {
    status: dynamicWorkerPids.length > 0 ? 'running' : baseStatus,
    live,
    pid: current.pid,
    campaign_id: state?.campaign_id || current.campaign_id || '',
    fixed_web_url: campaign.webUrl,
    input_mode: 'upload',
    tasks_total: state?.tasks_total ?? null,
    max_parallel_models: state?.max_parallel_models ?? 1,
    current_model: state?.current_model || '',
    current_child_pid: state?.current_child_pid || null,
    current_models: state?.current_models || (state?.current_model ? [state.current_model] : []),
    current_child_pids: state?.current_child_pids || (
      state?.current_model && state?.current_child_pid
        ? { [state.current_model]: state.current_child_pid }
        : {}
    ),
    models: { ...(state?.models || {}), ...dynamicModels },
    dynamic_worker_pids: dynamicWorkerPids,
    state_path: current.state_path || '',
    log_path: current.log_path || '',
    last_error: state?.last_error || current.error || '',
  };
  if (args.json) {
    console.log(JSON.stringify(payload, null, 2));
    return;
  }
  console.log(`Campaign: ${payload.campaign_id || '(preparing)'}  status=${payload.status}  pid=${payload.pid}  live=${payload.live}`);
  console.log(`WebUI: ${campaign.webUrl}  mode=upload  tasks=${payload.tasks_total ?? 'preparing'}`);
  for (const [id, model] of Object.entries(payload.models)) {
    const currentTask = model.current_task?.task_id ? ` current=${model.current_task.task_id}` : '';
    console.log(`${id.padEnd(27)} ${String(model.status || 'pending').padEnd(20)} ${model.tasks_ok || 0}/${model.tasks_total || payload.tasks_total || 0}${currentTask}`);
  }
  if (payload.dynamic_worker_pids.length) {
    console.log(`Dynamic workers: ${payload.dynamic_worker_pids.join(', ')}`);
  }
  if (payload.state_path) console.log(`State: ${payload.state_path}`);
  if (payload.log_path) console.log(`Log:   ${payload.log_path}`);
  if (payload.last_error) console.log(`Error: ${payload.last_error}`);
}

async function stop(args) {
  const { campaign, current, state } = currentState(args);
  const runRoot = current?.state_path ? path.dirname(path.resolve(current.state_path)) : '';
  const dynamicModels = runRoot ? readDynamicModelStates(
    runRoot,
    campaign.defaults.maxTaskRetries,
    campaign.defaults.maxFailedTaskRecoveryBatches,
  ) : {};
  const pids = new Set();
  const requestedModels = new Set(
    String(args.models || '').split(',').map((item) => item.trim()).filter(Boolean),
  );
  if (!requestedModels.size) {
    const mainPid = Number(current?.pid || 0);
    if (pidMatches(mainPid)) pids.add(mainPid);
  } else {
    const parentModels = new Set([
      ...(Array.isArray(state?.current_models) ? state.current_models : []),
      state?.current_model || '',
    ].filter(Boolean));
    if ([...requestedModels].some((modelId) => parentModels.has(modelId))) {
      const mainPid = Number(current?.pid || 0);
      if (pidMatches(mainPid)) pids.add(mainPid);
    }
  }
  for (const [modelId, model] of Object.entries(dynamicModels)) {
    if (requestedModels.size && !requestedModels.has(modelId)) continue;
    const pid = Number(model.pid || 0);
    if (model.live && pidMatches(pid)) pids.add(pid);
  }
  if (!pids.size) {
    console.log(requestedModels.size
      ? `No live dynamic worker was found for model(s): ${[...requestedModels].join(',')}.`
      : 'No live recording campaign process was found.');
    return;
  }
  for (const pid of pids) process.kill(pid, 'SIGTERM');
  console.log(`Stop requested for recording campaign pid(s): ${[...pids].join(', ')}.`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.command === 'dry-run') await dryRun(args);
  else if (args.command === 'add') await addModels(args);
  else if (args.command === 'run') await runCampaign(args);
  else if (args.command === 'status') await status(args);
  else if (args.command === 'stop') await stop(args);
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
