#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promoteTaskDirectory, relocateDirectory } from './artifact_promotion.mjs';
import { resolveTaskWallWindow } from './task_wall_budget.mjs';
import { buildRecordingAttemptAudit } from './recording_attempt_audit.mjs';
import {
  loadRecordingCampaignConfig,
  normalizeRecordingWebUrl,
} from './recording_config.mjs';
import { probeVideo, validateVideoDecode } from './video_quality.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_API_KEY_ENV = 'MODELBEST_API_KEY';
const DEFAULT_MAX_PROVIDER_QUERY_START_DRIFT_S = 5;
const DEFAULT_MAX_FAILED_TASK_RECOVERY_BATCHES = 1;
const CURRENT_RECORDING_PROTOCOL = 'webui_session_ready_then_single_pass_upload_v2';
const JOYAI_FRAME_STREAM_PROTOCOL = 'joyai-formal-query-once-frame-stream-v1';
const MODEL_RESPONSE_TIMEOUT_POLICY = 'wait_then_keep_as_capability_result';
const MODEL_RESPONSE_TIMEOUT_S = 180;
const MODEL_RESPONSE_OUTCOME_PROTOCOL = 'strict_protocol_non_deferred_non_echo_response_v4';
const JOYAI_OFFICIAL_SCAFFOLD = 'joyai-official-live-adapter';
const DEFAULT_UPSTREAM_SDK_MAX_RETRIES = 0;
const DEFAULT_NON_QUERY_REQUEST_TIMEOUT_S = 30;
const DEFAULT_QUERY_REQUEST_TIMEOUT_S = 180;
const DEFAULT_PROVIDER_WARMUP_TIMEOUT_S = 180;
const DEFAULT_MAX_TASK_WALL_S = 3600;
const RECORDING_CAMPAIGN = loadRecordingCampaignConfig();
const DEFAULT_WEB_URL = RECORDING_CAMPAIGN.webui.url;
const DEFAULT_WEB_TLS_REJECT_UNAUTHORIZED = RECORDING_CAMPAIGN.webui.tlsRejectUnauthorized;

function printHelp() {
  console.log(`Usage:
  node scripts/supervise_scaffold_recording.mjs [options]

Options:
  --tasks FILE                 Required task JSONL
  --out DIR                    Required stable output root
  --attempt-work-root DIR      Local scratch root for raw recording attempts;
                               default /tmp/joyvl-recording
  --web-url URL                Recording WebUI URL from the campaign config
  --web-username USER          WebUI Basic Auth username
  --profile-id ID              Required ephemeral profile id
  --model ID                   Required upstream request model
  --upstream-api-base URL      Required OpenAI-compatible upstream base
  --upstream-protocol NAME     openai-chat
  --input-transport NAME       Required model input transport from config/vlm_models.json
  --interaction-scaffold NAME joyai-official-live-adapter for the pinned JoyAI webinfer stack
  --process-interval-s N       WebUI frame interval; formal protocol requires 1
  --frames-per-batch N         Frames per analyzer tick; formal protocol requires 1
  --native-video-schema NAME   Exact provider-native video input schema
  --realtime-protocol NAME     Native realtime wire protocol
  --realtime-api-base URL      Native realtime WebSocket endpoint
  --realtime-query-mode NAME   Native realtime text-Query delivery mode
  --native-streaming-mode-policy NAME
                               proactive or interactive-after-query;
                               default proactive
  --native-system-prompt-transport NAME
                               system-role or inline-user-query; default system-role
  --native-frame-clock NAME     inbound-turn or wall-media; default inbound-turn
  --max-realtime-session-s N   Provider session lifetime limit
  --max-provider-query-start-drift-s N
                               Maximum absolute drift from the WebUI Query time; default ${DEFAULT_MAX_PROVIDER_QUERY_START_DRIFT_S}
  --upstream-sdk-max-retries N SDK retries inside one frame request; formal protocol requires 0
  --non-query-request-timeout-s N
                               Hard deadline for an ordinary frame request; default ${DEFAULT_NON_QUERY_REQUEST_TIMEOUT_S}
  --query-request-timeout-s N  Hard deadline for a Query-bearing request; default ${DEFAULT_QUERY_REQUEST_TIMEOUT_S}
  --provider-warmup-timeout-s N
                               Hard deadline for provider warmup; default ${DEFAULT_PROVIDER_WARMUP_TIMEOUT_S}
  --api-key-env NAME           Environment variable containing the upstream key; default ${DEFAULT_API_KEY_ENV}
  --backend-alias ID           Accepted response model identity; repeatable
  --allow-upstream-migration   Resume an existing model run through a new API gateway
  --disable-thinking           Ask hybrid-reasoning models to use non-thinking mode
  --continue-after-task-failure
                               Record an exhausted task as failed and continue later tasks
  --retry-failed-tasks         Give tasks already marked failed one new retry budget
  --max-failed-task-recovery-batches N
                               Maximum bounded failed-task recovery batches; default ${DEFAULT_MAX_FAILED_TASK_RECOVERY_BATCHES}
  --max-retries N              Retries after the first attempt; default 5
  --max-task-wall-s N          Maximum cumulative wall time for one task pass;
                               default ${DEFAULT_MAX_TASK_WALL_S}
  --upstream-blocked-backoff-s N
                               Wait before rechecking a quota-paused upstream; default 300
  --progress-interval-s N      Progress heartbeat interval; default 1800
  --proxy-port N               Local prompt proxy port; default 18070
  --deployed-adapter-api-base URL
                               Adapter URL as seen by the recording WebUI
  --deployed-adapter-access-token-env NAME
                               Environment variable holding the deployed adapter token
  --deployed-adapter-audit-path FILE
                               Shared JSONL audit path written by the deployed adapter
  --cloudflared PATH           cloudflared binary; default cloudflared
  --retain-last-failed-artifacts
                               Keep only the latest failed raw capture for debugging

JOYVL_WEB_PASSWORD and the selected upstream API-key environment variable are required. Optional
NAME_2 ... NAME_9 or NAME_POOL values form an ordered failover pool for upstream safety limits.
No secret is written to the output directory or passed on a child command line.`);
}

function parseArgs(argv) {
  const args = {
    tasks: '',
    out: '',
    attemptWorkRoot: process.env.JOYVL_ATTEMPT_WORK_ROOT
      || path.join(os.tmpdir(), 'joyvl-recording'),
    webUrl: DEFAULT_WEB_URL,
    webUsername: process.env.JOYVL_WEB_USERNAME || '',
    profileId: '',
    model: '',
    upstreamApiBase: '',
    upstreamProtocol: 'openai-chat',
    inputTransport: '',
    interactionScaffold: '',
    processIntervalS: 1,
    framesPerBatch: 1,
    nativeVideoSchema: '',
    realtimeProtocol: '',
    realtimeApiBase: '',
    realtimeQueryMode: '',
    nativeQueryAudioManifest: '',
    nativeStreamingModePolicy: 'proactive',
    nativeSystemPromptTransport: 'system-role',
    nativeFrameClock: 'inbound-turn',
    maxRealtimeSessionS: 0,
    maxProviderQueryStartDriftS: DEFAULT_MAX_PROVIDER_QUERY_START_DRIFT_S,
    upstreamSdkMaxRetries: DEFAULT_UPSTREAM_SDK_MAX_RETRIES,
    nonQueryRequestTimeoutS: DEFAULT_NON_QUERY_REQUEST_TIMEOUT_S,
    queryRequestTimeoutS: DEFAULT_QUERY_REQUEST_TIMEOUT_S,
    providerWarmupTimeoutS: DEFAULT_PROVIDER_WARMUP_TIMEOUT_S,
    apiKeyEnv: DEFAULT_API_KEY_ENV,
    backendAliases: [],
    allowUpstreamMigration: false,
    disableThinking: false,
    continueAfterTaskFailure: false,
    retryFailedTasks: false,
    maxFailedTaskRecoveryBatches: DEFAULT_MAX_FAILED_TASK_RECOVERY_BATCHES,
    maxRetries: 5,
    maxTaskWallS: DEFAULT_MAX_TASK_WALL_S,
    upstreamBlockedBackoffS: 300,
    progressIntervalS: 1800,
    proxyPort: 18070,
    deployedAdapterApiBase: '',
    deployedAdapterAccessTokenEnv: '',
    deployedAdapterAuditPath: '',
    cloudflared: process.env.CLOUDFLARED_BIN || 'cloudflared',
    retainLastFailedArtifacts: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      if (index + 1 >= argv.length) throw new Error(`Missing value for ${arg}`);
      index += 1;
      return argv[index];
    };
    if (arg === '--tasks') args.tasks = next();
    else if (arg.startsWith('--tasks=')) args.tasks = arg.slice('--tasks='.length);
    else if (arg === '--out') args.out = next();
    else if (arg.startsWith('--out=')) args.out = arg.slice('--out='.length);
    else if (arg === '--attempt-work-root') args.attemptWorkRoot = next();
    else if (arg.startsWith('--attempt-work-root=')) {
      args.attemptWorkRoot = arg.slice('--attempt-work-root='.length);
    }
    else if (arg === '--web-url') args.webUrl = next();
    else if (arg.startsWith('--web-url=')) args.webUrl = arg.slice('--web-url='.length);
    else if (arg === '--web-username') args.webUsername = next();
    else if (arg.startsWith('--web-username=')) args.webUsername = arg.slice('--web-username='.length);
    else if (arg === '--profile-id') args.profileId = next();
    else if (arg.startsWith('--profile-id=')) args.profileId = arg.slice('--profile-id='.length);
    else if (arg === '--model') args.model = next();
    else if (arg.startsWith('--model=')) args.model = arg.slice('--model='.length);
    else if (arg === '--upstream-api-base') args.upstreamApiBase = next();
    else if (arg.startsWith('--upstream-api-base=')) {
      args.upstreamApiBase = arg.slice('--upstream-api-base='.length);
    } else if (arg === '--api-key-env') args.apiKeyEnv = next();
    else if (arg === '--upstream-protocol') args.upstreamProtocol = next();
    else if (arg.startsWith('--upstream-protocol=')) {
      args.upstreamProtocol = arg.slice('--upstream-protocol='.length);
    }
    else if (arg === '--input-transport') args.inputTransport = next();
    else if (arg.startsWith('--input-transport=')) {
      args.inputTransport = arg.slice('--input-transport='.length);
    }
    else if (arg === '--interaction-scaffold') args.interactionScaffold = next();
    else if (arg.startsWith('--interaction-scaffold=')) {
      args.interactionScaffold = arg.slice('--interaction-scaffold='.length);
    }
    else if (arg === '--process-interval-s') args.processIntervalS = Number(next());
    else if (arg.startsWith('--process-interval-s=')) {
      args.processIntervalS = Number(arg.slice('--process-interval-s='.length));
    }
    else if (arg === '--frames-per-batch') args.framesPerBatch = Number(next());
    else if (arg.startsWith('--frames-per-batch=')) {
      args.framesPerBatch = Number(arg.slice('--frames-per-batch='.length));
    }
    else if (arg === '--native-video-schema') args.nativeVideoSchema = next();
    else if (arg.startsWith('--native-video-schema=')) {
      args.nativeVideoSchema = arg.slice('--native-video-schema='.length);
    }
    else if (arg === '--realtime-protocol') args.realtimeProtocol = next();
    else if (arg.startsWith('--realtime-protocol=')) {
      args.realtimeProtocol = arg.slice('--realtime-protocol='.length);
    }
    else if (arg === '--realtime-api-base') args.realtimeApiBase = next();
    else if (arg.startsWith('--realtime-api-base=')) {
      args.realtimeApiBase = arg.slice('--realtime-api-base='.length);
    }
    else if (arg === '--realtime-query-mode') args.realtimeQueryMode = next();
    else if (arg.startsWith('--realtime-query-mode=')) {
      args.realtimeQueryMode = arg.slice('--realtime-query-mode='.length);
    }
    else if (arg === '--native-query-audio-manifest') {
      args.nativeQueryAudioManifest = next();
    }
    else if (arg.startsWith('--native-query-audio-manifest=')) {
      args.nativeQueryAudioManifest = arg.slice('--native-query-audio-manifest='.length);
    }
    else if (arg === '--native-streaming-mode-policy') {
      args.nativeStreamingModePolicy = next();
    }
    else if (arg.startsWith('--native-streaming-mode-policy=')) {
      args.nativeStreamingModePolicy = arg.slice('--native-streaming-mode-policy='.length);
    }
    else if (arg === '--native-system-prompt-transport') {
      args.nativeSystemPromptTransport = next();
    }
    else if (arg.startsWith('--native-system-prompt-transport=')) {
      args.nativeSystemPromptTransport = arg.slice('--native-system-prompt-transport='.length);
    }
    else if (arg === '--native-frame-clock') args.nativeFrameClock = next();
    else if (arg.startsWith('--native-frame-clock=')) {
      args.nativeFrameClock = arg.slice('--native-frame-clock='.length);
    }
    else if (arg === '--max-realtime-session-s') args.maxRealtimeSessionS = Number(next());
    else if (arg.startsWith('--max-realtime-session-s=')) {
      args.maxRealtimeSessionS = Number(arg.slice('--max-realtime-session-s='.length));
    }
    else if (arg === '--max-provider-query-start-drift-s') {
      args.maxProviderQueryStartDriftS = Number(next());
    }
    else if (arg.startsWith('--max-provider-query-start-drift-s=')) {
      args.maxProviderQueryStartDriftS = Number(
        arg.slice('--max-provider-query-start-drift-s='.length),
      );
    }
    else if (arg === '--upstream-sdk-max-retries') {
      args.upstreamSdkMaxRetries = Number(next());
    }
    else if (arg.startsWith('--upstream-sdk-max-retries=')) {
      args.upstreamSdkMaxRetries = Number(arg.slice('--upstream-sdk-max-retries='.length));
    }
    else if (arg === '--non-query-request-timeout-s') {
      args.nonQueryRequestTimeoutS = Number(next());
    }
    else if (arg.startsWith('--non-query-request-timeout-s=')) {
      args.nonQueryRequestTimeoutS = Number(arg.slice('--non-query-request-timeout-s='.length));
    }
    else if (arg === '--query-request-timeout-s') {
      args.queryRequestTimeoutS = Number(next());
    }
    else if (arg.startsWith('--query-request-timeout-s=')) {
      args.queryRequestTimeoutS = Number(arg.slice('--query-request-timeout-s='.length));
    }
    else if (arg === '--provider-warmup-timeout-s') {
      args.providerWarmupTimeoutS = Number(next());
    }
    else if (arg.startsWith('--provider-warmup-timeout-s=')) {
      args.providerWarmupTimeoutS = Number(arg.slice('--provider-warmup-timeout-s='.length));
    }
    else if (arg.startsWith('--api-key-env=')) args.apiKeyEnv = arg.slice('--api-key-env='.length);
    else if (arg === '--backend-alias') args.backendAliases.push(next());
    else if (arg.startsWith('--backend-alias=')) {
      args.backendAliases.push(arg.slice('--backend-alias='.length));
    }
    else if (arg === '--allow-upstream-migration') args.allowUpstreamMigration = true;
    else if (arg === '--disable-thinking') args.disableThinking = true;
    else if (arg === '--persist-user-query') {
      throw new Error('--persist-user-query is forbidden; each Query must be delivered exactly once');
    }
    else if (arg === '--continue-after-task-failure') args.continueAfterTaskFailure = true;
    else if (arg === '--retry-failed-tasks') args.retryFailedTasks = true;
    else if (arg === '--max-failed-task-recovery-batches') {
      args.maxFailedTaskRecoveryBatches = Number(next());
    }
    else if (arg.startsWith('--max-failed-task-recovery-batches=')) {
      args.maxFailedTaskRecoveryBatches = Number(
        arg.slice('--max-failed-task-recovery-batches='.length),
      );
    }
    else if (arg === '--max-retries') args.maxRetries = Number(next());
    else if (arg.startsWith('--max-retries=')) args.maxRetries = Number(arg.slice('--max-retries='.length));
    else if (arg === '--max-task-wall-s') args.maxTaskWallS = Number(next());
    else if (arg.startsWith('--max-task-wall-s=')) {
      args.maxTaskWallS = Number(arg.slice('--max-task-wall-s='.length));
    }
    else if (arg === '--upstream-blocked-backoff-s') args.upstreamBlockedBackoffS = Number(next());
    else if (arg.startsWith('--upstream-blocked-backoff-s=')) {
      args.upstreamBlockedBackoffS = Number(arg.slice('--upstream-blocked-backoff-s='.length));
    }
    else if (arg === '--progress-interval-s') args.progressIntervalS = Number(next());
    else if (arg.startsWith('--progress-interval-s=')) {
      args.progressIntervalS = Number(arg.slice('--progress-interval-s='.length));
    } else if (arg === '--proxy-port') args.proxyPort = Number(next());
    else if (arg.startsWith('--proxy-port=')) args.proxyPort = Number(arg.slice('--proxy-port='.length));
    else if (arg === '--deployed-adapter-api-base') args.deployedAdapterApiBase = next();
    else if (arg.startsWith('--deployed-adapter-api-base=')) {
      args.deployedAdapterApiBase = arg.slice('--deployed-adapter-api-base='.length);
    }
    else if (arg === '--deployed-adapter-access-token-env') {
      args.deployedAdapterAccessTokenEnv = next();
    }
    else if (arg.startsWith('--deployed-adapter-access-token-env=')) {
      args.deployedAdapterAccessTokenEnv = arg.slice(
        '--deployed-adapter-access-token-env='.length,
      );
    }
    else if (arg === '--deployed-adapter-audit-path') args.deployedAdapterAuditPath = next();
    else if (arg.startsWith('--deployed-adapter-audit-path=')) {
      args.deployedAdapterAuditPath = arg.slice('--deployed-adapter-audit-path='.length);
    }
    else if (arg === '--cloudflared') args.cloudflared = next();
    else if (arg.startsWith('--cloudflared=')) args.cloudflared = arg.slice('--cloudflared='.length);
    else if (arg === '--retain-last-failed-artifacts') args.retainLastFailedArtifacts = true;
    else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    } else throw new Error(`Unknown option: ${arg}`);
  }
  for (const [option, value] of [
    ['--tasks', args.tasks],
    ['--out', args.out],
    ['--profile-id', args.profileId],
    ['--model', args.model],
    ['--upstream-api-base', args.upstreamApiBase],
  ]) {
    if (!String(value || '').trim()) throw new Error(`${option} is required`);
  }
  if (!Number.isInteger(args.maxRetries) || args.maxRetries < 0 || args.maxRetries > 5) {
    throw new Error('--max-retries must be an integer from 0 through 5');
  }
  if (
    !Number.isInteger(args.maxFailedTaskRecoveryBatches)
    || args.maxFailedTaskRecoveryBatches < 1
    || args.maxFailedTaskRecoveryBatches > 3
  ) {
    throw new Error('--max-failed-task-recovery-batches must be an integer from 1 through 3');
  }
  if (!String(args.attemptWorkRoot || '').trim()) {
    throw new Error('--attempt-work-root must not be empty');
  }
  args.attemptWorkRoot = path.resolve(args.attemptWorkRoot);
  if (!Number.isInteger(args.maxTaskWallS) || args.maxTaskWallS < 300) {
    throw new Error('--max-task-wall-s must be an integer of at least 300 seconds');
  }
  if (!Number.isFinite(args.upstreamBlockedBackoffS) || args.upstreamBlockedBackoffS < 30) {
    throw new Error('--upstream-blocked-backoff-s must be at least 30');
  }
  if (!Number.isFinite(args.progressIntervalS) || args.progressIntervalS < 60) {
    throw new Error('--progress-interval-s must be at least 60');
  }
  if (!Number.isInteger(args.proxyPort) || args.proxyPort <= 0 || args.proxyPort > 65535) {
    throw new Error('--proxy-port is invalid');
  }
  if (
    !Number.isFinite(args.maxProviderQueryStartDriftS)
    || args.maxProviderQueryStartDriftS <= 0
    || args.maxProviderQueryStartDriftS > 60
  ) {
    throw new Error('--max-provider-query-start-drift-s must be greater than 0 and at most 60');
  }
  if (args.upstreamSdkMaxRetries !== 0) {
    throw new Error('--upstream-sdk-max-retries must be exactly 0 for formal recordings');
  }
  for (const [name, value] of [
    ['--non-query-request-timeout-s', args.nonQueryRequestTimeoutS],
    ['--query-request-timeout-s', args.queryRequestTimeoutS],
    ['--provider-warmup-timeout-s', args.providerWarmupTimeoutS],
  ]) {
    if (!Number.isInteger(value) || value < 1 || value > MODEL_RESPONSE_TIMEOUT_S) {
      throw new Error(`${name} must be an integer from 1 through ${MODEL_RESPONSE_TIMEOUT_S}`);
    }
  }
  if (args.nonQueryRequestTimeoutS > args.queryRequestTimeoutS) {
    throw new Error('--non-query-request-timeout-s must not exceed --query-request-timeout-s');
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(args.profileId)) {
    throw new Error('--profile-id must be filesystem-safe');
  }
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(args.apiKeyEnv)) {
    throw new Error('--api-key-env must be an environment variable name');
  }
  if (args.upstreamProtocol !== 'openai-chat') {
    throw new Error('--upstream-protocol must be openai-chat');
  }
  if (!args.inputTransport) {
    throw new Error('--input-transport is required; do not infer or downgrade a model input modality');
  }
  if (args.processIntervalS !== 1 || args.framesPerBatch !== 1) {
    throw new Error(
      'Formal frame scheduling requires --process-interval-s 1 and --frames-per-batch 1',
    );
  }
  if (!['image-frame-batch', 'stateful-frame-stream', 'native-video-realtime'].includes(args.inputTransport)) {
    throw new Error(
      `The managed proxy runner cannot serve input transport ${args.inputTransport}; `
      + 'native-video-batch cannot be used for Query-once realtime monitoring',
    );
  }
  if (args.interactionScaffold) {
    if (args.interactionScaffold !== JOYAI_OFFICIAL_SCAFFOLD) {
      throw new Error(`Unsupported --interaction-scaffold: ${args.interactionScaffold}`);
    }
    if (args.inputTransport !== 'stateful-frame-stream') {
      throw new Error(
        `${JOYAI_OFFICIAL_SCAFFOLD} requires --input-transport stateful-frame-stream`,
      );
    }
    if (args.upstreamProtocol !== 'openai-chat') {
      throw new Error(`${JOYAI_OFFICIAL_SCAFFOLD} currently requires --upstream-protocol openai-chat`);
    }
  } else if (args.inputTransport === 'stateful-frame-stream') {
    throw new Error('stateful-frame-stream requires --interaction-scaffold');
  }
  const deployedAdapterFields = [
    args.deployedAdapterApiBase,
    args.deployedAdapterAccessTokenEnv,
    args.deployedAdapterAuditPath,
  ];
  const deployedAdapterFieldCount = deployedAdapterFields.filter(Boolean).length;
  if (deployedAdapterFieldCount && deployedAdapterFieldCount !== deployedAdapterFields.length) {
    throw new Error('All deployed-adapter options must be provided together');
  }
  if (deployedAdapterFieldCount) {
    const deployedJoyAiScaffold = (
      args.interactionScaffold === JOYAI_OFFICIAL_SCAFFOLD
      && args.inputTransport === 'stateful-frame-stream'
    );
    const deployedNativeRealtime = (
      !args.interactionScaffold
      && args.inputTransport === 'native-video-realtime'
    );
    if (!deployedJoyAiScaffold && !deployedNativeRealtime) {
      throw new Error(
        'A deployed adapter must provide either the official JoyAI frame-stream scaffold '
        + 'or a native-video-realtime transport',
      );
    }
    args.deployedAdapterApiBase = args.deployedAdapterApiBase.replace(/\/+$/, '');
    const deployedUrl = new URL(args.deployedAdapterApiBase);
    if (!['http:', 'https:'].includes(deployedUrl.protocol)) {
      throw new Error('--deployed-adapter-api-base must use HTTP or HTTPS');
    }
    if (deployedUrl.username || deployedUrl.password) {
      throw new Error('--deployed-adapter-api-base must not embed credentials');
    }
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(args.deployedAdapterAccessTokenEnv)) {
      throw new Error('--deployed-adapter-access-token-env is invalid');
    }
    args.deployedAdapterAuditPath = path.resolve(ROOT, args.deployedAdapterAuditPath);
  }
  if (args.inputTransport === 'native-video-realtime') {
    if (!args.nativeVideoSchema) throw new Error('--native-video-schema is required');
    if (![
      'modelbest-video-full-duplex-v1',
      'jd-responses-causal-video-chunks-v1',
      'ark-responses-causal-video-chunks-v1',
      'joyai-http-session-v1',
    ].includes(args.realtimeProtocol)) {
      throw new Error(`Unsupported --realtime-protocol: ${args.realtimeProtocol || '(missing)'}`);
    }
    const expectedQueryModes = args.realtimeProtocol === 'modelbest-video-full-duplex-v1'
      ? ['session-instruction', 'input-audio-once']
      : ['input-text-once'];
    if (!expectedQueryModes.includes(args.realtimeQueryMode)) {
      throw new Error(
        `--realtime-query-mode must be ${expectedQueryModes.join(' or ')} `
        + `for ${args.realtimeProtocol}`,
      );
    }
    if (args.realtimeQueryMode === 'input-audio-once') {
      if (!args.nativeQueryAudioManifest) {
        throw new Error('--native-query-audio-manifest is required for input-audio-once');
      }
      args.nativeQueryAudioManifest = path.resolve(ROOT, args.nativeQueryAudioManifest);
      if (!fs.statSync(args.nativeQueryAudioManifest).isFile()) {
        throw new Error('--native-query-audio-manifest must point to a file');
      }
    } else if (args.nativeQueryAudioManifest) {
      throw new Error('--native-query-audio-manifest requires input-audio-once');
    }
    const realtimeUrl = new URL(args.realtimeApiBase);
    const validRealtimeSchemes = args.realtimeProtocol === 'modelbest-video-full-duplex-v1'
      ? ['ws:', 'wss:']
      : ['http:', 'https:'];
    if (!validRealtimeSchemes.includes(realtimeUrl.protocol)) {
      throw new Error(`--realtime-api-base must use ${validRealtimeSchemes.join(' or ')}`);
    }
    if (realtimeUrl.username || realtimeUrl.password) {
      throw new Error('--realtime-api-base must not embed credentials');
    }
    if (!Number.isFinite(args.maxRealtimeSessionS) || args.maxRealtimeSessionS <= 0) {
      throw new Error('--max-realtime-session-s must be positive');
    }
    if (![
      'proactive',
      'interactive-after-query',
    ].includes(args.nativeStreamingModePolicy)) {
      throw new Error(
        `Unsupported --native-streaming-mode-policy: ${args.nativeStreamingModePolicy}`,
      );
    }
    if (!['system-role', 'inline-user-query'].includes(args.nativeSystemPromptTransport)) {
      throw new Error(
        `Unsupported --native-system-prompt-transport: ${args.nativeSystemPromptTransport}`,
      );
    }
    if (!['inbound-turn', 'wall-media'].includes(args.nativeFrameClock)) {
      throw new Error(`Unsupported --native-frame-clock: ${args.nativeFrameClock}`);
    }
    if (
      (args.nativeSystemPromptTransport !== 'system-role'
        || args.nativeFrameClock !== 'inbound-turn')
      && args.realtimeProtocol !== 'joyai-http-session-v1'
    ) {
      throw new Error('Native prompt/clock compatibility requires joyai-http-session-v1');
    }
    if (
      args.nativeStreamingModePolicy !== 'proactive'
      && args.realtimeProtocol !== 'joyai-http-session-v1'
    ) {
      throw new Error(
        '--native-streaming-mode-policy interactive modes require joyai-http-session-v1',
      );
    }
  } else if (
    args.nativeVideoSchema
    || args.realtimeProtocol
    || args.realtimeApiBase
    || args.realtimeQueryMode
    || args.nativeQueryAudioManifest
    || args.maxRealtimeSessionS
  ) {
    throw new Error('Realtime-only options require --input-transport native-video-realtime');
  }
  args.upstreamApiBase = String(args.upstreamApiBase || '').replace(/\/+$/, '');
  const upstreamUrl = new URL(args.upstreamApiBase);
  if (!['http:', 'https:'].includes(upstreamUrl.protocol)) {
    throw new Error('--upstream-api-base must use HTTP or HTTPS');
  }
  if (upstreamUrl.username || upstreamUrl.password) {
    throw new Error('--upstream-api-base must not embed credentials');
  }
  args.backendAliases = [...new Set([args.model, ...args.backendAliases]
    .map((value) => String(value || '').trim()).filter(Boolean))];
  args.webUrl = normalizeRecordingWebUrl(args.webUrl);
  if (args.webUrl !== DEFAULT_WEB_URL) {
    throw new Error(`Recording WebUI must match the selected campaign config: ${DEFAULT_WEB_URL}`);
  }
  args.tasks = path.resolve(ROOT, args.tasks);
  args.out = path.resolve(ROOT, args.out);
  return args;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const RETRYABLE_REMOVE_CODES = new Set(['EBUSY', 'ENOTEMPTY', 'EPERM', 'EACCES']);

async function removePathWithRetries(target, {
  attempts = 30,
  delayMs = 1000,
  ignoreFailure = false,
} = {}) {
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      await fsp.rm(target, { recursive: true, force: true });
      return true;
    } catch (error) {
      if (!RETRYABLE_REMOVE_CODES.has(error?.code) || attempt >= attempts) {
        if (ignoreFailure) return false;
        throw error;
      }
      await sleep(delayMs);
    }
  }
  return false;
}

function safeName(value) {
  return String(value).replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^_+|_+$/g, '') || 'task';
}

function timestamp() {
  return new Date().toISOString();
}

function sha256(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function canonicalJson(value) {
  if (Array.isArray(value)) return value.map(canonicalJson);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, canonicalJson(value[key])]),
    );
  }
  return value;
}

function sameJson(left, right) {
  return JSON.stringify(canonicalJson(left)) === JSON.stringify(canonicalJson(right));
}

function normalizedEvaluationContract(contract, deployedDefaults = null) {
  if (!contract || typeof contract !== 'object') return contract;
  const normalized = structuredClone(contract);
  if (normalized.input_transport !== 'native-video-realtime') {
    delete normalized.native_streaming_mode_policy;
    delete normalized.native_system_prompt_transport;
    delete normalized.native_frame_clock;
    // Deployment topology is operational provenance, not an evaluation semantic.
    delete normalized.scaffold_deployment;
    return normalized;
  }
  const sameDeployedAdapter = Boolean(
    normalized.scaffold_deployment
    && deployedDefaults?.scaffold_deployment
    && sameJson(
      normalized.scaffold_deployment,
      deployedDefaults.scaffold_deployment,
    )
  );
  if (!Object.hasOwn(normalized, 'native_streaming_mode_policy')) {
    normalized.native_streaming_mode_policy = sameDeployedAdapter
      ? deployedDefaults.native_streaming_mode_policy
      : 'proactive';
  }
  if (!Object.hasOwn(normalized, 'native_system_prompt_transport')) {
    normalized.native_system_prompt_transport = sameDeployedAdapter
      ? deployedDefaults.native_system_prompt_transport
      : 'system-role';
  }
  if (!Object.hasOwn(normalized, 'native_frame_clock')) {
    normalized.native_frame_clock = sameDeployedAdapter
      ? deployedDefaults.native_frame_clock
      : 'inbound-turn';
  }
  delete normalized.scaffold_deployment;
  return normalized;
}

function compatibleEvaluationContract(previous, current) {
  const normalizedCurrent = normalizedEvaluationContract(current);
  const normalizedPrevious = normalizedEvaluationContract(previous, normalizedCurrent);
  if (sameJson(normalizedPrevious, normalizedCurrent)) return true;
  if (
    normalizedPrevious?.input_transport !== 'native-video-realtime'
    || normalizedCurrent?.input_transport !== 'native-video-realtime'
    || normalizedPrevious?.native_frame_clock !== 'inbound-turn'
    || normalizedCurrent?.native_frame_clock !== 'wall-media'
  ) return false;
  return sameJson(
    { ...normalizedPrevious, native_frame_clock: 'wall-media' },
    normalizedCurrent,
  );
}

function sourceVideoSnapshot(task) {
  const sourcePath = path.resolve(String(task.local_video_path || ''));
  const stat = fs.statSync(sourcePath);
  return {
    path: sourcePath,
    size_bytes: stat.size,
    mtime_ms: Math.round(stat.mtimeMs),
  };
}

function buildProviderQueryTiming({
  taskSummary,
  acceptanceEvents,
  providerQueryEvents,
  submissionEvents,
  maxAbsoluteDriftS,
}) {
  const taskStartedAtMs = Date.parse(String(taskSummary?.started_at || ''));
  const queryEvents = Array.isArray(taskSummary?.query_events) ? taskSummary.query_events : [];
  const submissionsByRequestId = new Map(submissionEvents
    .filter((event) => String(event.request_id || ''))
    .map((event) => [String(event.request_id), event]));
  const deliveredByHash = new Map();
  for (const event of providerQueryEvents) {
    const hash = String(event.user_query_sha256 || '');
    const records = deliveredByHash.get(hash) || [];
    records.push(event);
    deliveredByHash.set(hash, records);
  }
  const acceptedByHash = new Map();
  for (const event of acceptanceEvents) {
    const hash = String(event.user_query_sha256 || '');
    const records = acceptedByHash.get(hash) || [];
    records.push(event);
    acceptedByHash.set(hash, records);
  }
  const hashOffsets = new Map();
  const rounds = queryEvents.map((event, index) => {
    const hash = sha256(String(event.query || ''));
    const occurrence = hashOffsets.get(hash) || 0;
    hashOffsets.set(hash, occurrence + 1);
    const accepted = acceptedByHash.get(hash)?.[occurrence] || null;
    const delivered = deliveredByHash.get(hash)?.[occurrence] || null;
    const requestId = String(delivered?.request_id || '');
    const submitted = submissionsByRequestId.get(requestId) || null;
    const expectedAtMs = Number.isFinite(taskStartedAtMs)
      && Number.isFinite(Number(event.query_sent_offset_s))
      ? taskStartedAtMs + Number(event.query_sent_offset_s) * 1000
      : Number.NaN;
    const submittedAtMs = Date.parse(String(submitted?.timestamp || ''));
    const acceptedAtMs = Date.parse(String(accepted?.timestamp || ''));
    const acceptanceDriftS = Number.isFinite(expectedAtMs) && Number.isFinite(acceptedAtMs)
      ? (acceptedAtMs - expectedAtMs) / 1000
      : null;
    const providerSubmissionDelayS = Number.isFinite(expectedAtMs) && Number.isFinite(submittedAtMs)
      ? (submittedAtMs - expectedAtMs) / 1000
      : null;
    const queuedQueryFrame = (
      accepted?.query_frame_queue_policy === 'fifo-query-time-frame'
      || delivered?.query_frame_queue_policy === 'fifo-query-time-frame'
    );
    const auditedQueueDelayS = Number.isFinite(Number(delivered?.query_frame_queue_delay_ms))
      ? Number(delivered.query_frame_queue_delay_ms) / 1000
      : null;
    return {
      id: String(event.id || `R${index + 1}`),
      query_index: Number(event.query_index || index + 1),
      user_query_sha256: hash,
      request_id: requestId,
      expected_query_sent_at: Number.isFinite(expectedAtMs)
        ? new Date(expectedAtMs).toISOString()
        : '',
      queue_accepted_at: Number.isFinite(acceptedAtMs)
        ? new Date(acceptedAtMs).toISOString()
        : '',
      queue_acceptance_drift_s: acceptanceDriftS,
      provider_submission_started_at: Number.isFinite(submittedAtMs)
        ? new Date(submittedAtMs).toISOString()
        : '',
      provider_submission_delay_s: providerSubmissionDelayS,
      drift_s: acceptanceDriftS,
      scheduler_queue_delay_s: queuedQueryFrame ? auditedQueueDelayS : 0,
      latency_origin: 'webui_query_dispatch',
      query_frame_binding: queuedQueryFrame ? 'query-time-displayed-frame' : 'webui-ingress-frame',
      scheduling_policy: queuedQueryFrame ? 'fifo-query-time-frame' : 'legacy-direct-dispatch',
      upstream_submission_observed: Boolean(submitted),
      max_absolute_drift_s: maxAbsoluteDriftS,
      ok: acceptanceDriftS != null
        && Math.abs(acceptanceDriftS) <= maxAbsoluteDriftS,
    };
  });
  return {
    policy: rounds.some((round) => round.scheduling_policy === 'fifo-query-time-frame')
      ? 'webui-query-time-origin-with-fifo-provider-submission'
      : 'provider_query_submission_aligned_to_webui_query',
    ok: rounds.every((round) => round.ok),
    expected_rounds: queryEvents.length,
    timed_rounds: rounds.filter((round) => round.queue_acceptance_drift_s != null).length,
    max_absolute_drift_s: maxAbsoluteDriftS,
    rounds,
  };
}

function readTasks(taskPath) {
  return fs.readFileSync(taskPath, 'utf8').split(/\r?\n/).filter((line) => line.trim()).map((line, index) => {
    const task = JSON.parse(line);
    if (!String(task.id || '').trim()) throw new Error(`Task line ${index + 1} has no id`);
    return task;
  });
}

function resolveUpstreamKeys(args) {
  const candidates = [];
  const add = (value) => {
    const key = String(value || '').trim();
    if (key && !candidates.includes(key)) candidates.push(key);
  };
  add(process.env[args.apiKeyEnv]);
  for (let slot = 2; slot <= 9; slot += 1) add(process.env[`${args.apiKeyEnv}_${slot}`]);
  for (const value of String(process.env[`${args.apiKeyEnv}_POOL`] || '').split(/[\s,;]+/)) add(value);
  if (candidates.length) return candidates;
  throw new Error(`${args.apiKeyEnv} (or a numbered/pool variant) is required`);
}

function replaceAllSecrets(value, secrets) {
  let text = String(value || '');
  for (const secret of secrets) {
    if (secret) text = text.split(secret).join('[REDACTED]');
  }
  return text
    .replace(/([?&](?:api_key|password|secret|access_token|token)=)[^&\s]+/gi, '$1[REDACTED]')
    .replace(/:\/\/[^/@\s]+@/g, '://[REDACTED]@');
}

function tlsRejectUnauthorizedFor(urlString) {
  const url = new URL(urlString);
  const webUrl = new URL(DEFAULT_WEB_URL);
  const protocol = url.protocol === 'wss:' ? 'https:'
    : url.protocol === 'ws:' ? 'http:' : url.protocol;
  const port = url.port || (protocol === 'https:' ? '443' : '80');
  const webPort = webUrl.port || (webUrl.protocol === 'https:' ? '443' : '80');
  if (protocol === webUrl.protocol && url.hostname === webUrl.hostname && port === webPort) {
    return DEFAULT_WEB_TLS_REJECT_UNAUTHORIZED;
  }
  return true;
}

function isTemporarilyBlockedUpstream(error) {
  const message = String(error?.message || error || '');
  return /SetLimitExceeded|RateLimitExceeded|QuotaExceeded|AccountOverdue|TooManyRequests/i.test(message)
    || /model service has been paused/i.test(message);
}

function isKeySafetyLimit(error) {
  const message = String(error?.message || error || '');
  return /SetLimitExceeded|AccountOverdue|QuotaExceeded|model service has been paused|安全体验模式/i.test(message);
}

async function writeJsonAtomic(filePath, payload) {
  await fsp.mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.tmp-${process.pid}`;
  await fsp.writeFile(temporary, `${JSON.stringify(payload, null, 2)}\n`);
  await fsp.rename(temporary, filePath);
}

async function appendJsonl(filePath, payload) {
  await fsp.mkdir(path.dirname(filePath), { recursive: true });
  await fsp.appendFile(filePath, `${JSON.stringify(payload)}\n`);
}

function request(urlString, {
  timeoutMs = 30000,
  bearer = '',
  basicUsername = '',
  basicPassword = '',
  method = 'GET',
  payload = null,
  requestHeaders = {},
} = {}) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlString);
    const transport = url.protocol === 'https:' ? https : http;
    const headers = { ...requestHeaders };
    const body = payload == null ? null : Buffer.from(JSON.stringify(payload));
    if (bearer) headers.authorization = `Bearer ${bearer}`;
    if (basicUsername || basicPassword) {
      headers.authorization = `Basic ${Buffer.from(`${basicUsername}:${basicPassword}`).toString('base64')}`;
    }
    if (body) {
      headers['content-type'] = 'application/json';
      headers['content-length'] = String(body.length);
    }
    const req = transport.request({
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: `${url.pathname}${url.search}`,
      method,
      headers,
      rejectUnauthorized: tlsRejectUnauthorizedFor(urlString),
      timeout: timeoutMs,
    }, (res) => {
      const chunks = [];
      let length = 0;
      res.on('data', (chunk) => {
        length += chunk.length;
        if (length <= 4 * 1024 * 1024) chunks.push(chunk);
      });
      res.on('end', () => {
        const body = Buffer.concat(chunks).toString('utf8');
        if (res.statusCode >= 200 && res.statusCode < 300) {
          let json = null;
          try { json = body ? JSON.parse(body) : {}; } catch { json = null; }
          resolve({ statusCode: res.statusCode, body, json });
        } else reject(new Error(`HTTP ${res.statusCode}: ${body.slice(0, 500)}`));
      });
    });
    req.once('timeout', () => req.destroy(new Error(`Request timed out after ${timeoutMs}ms`)));
    req.once('error', reject);
    if (body) req.write(body);
    req.end();
  });
}

function childExit(child) {
  if (child.exitCode != null || child.signalCode) {
    return Promise.resolve({ code: child.exitCode, signal: child.signalCode || '' });
  }
  return new Promise((resolve) => child.once('exit', (code, signal) => resolve({
    code: code ?? 1,
    signal: signal || '',
  })));
}

async function terminateChild(child, { processGroup = false, graceMs = 10000 } = {}) {
  if (!child || child.exitCode != null || child.signalCode) return;
  const send = (signal) => {
    try {
      if (processGroup && child.pid) process.kill(-child.pid, signal);
      else child.kill(signal);
    } catch {
      // The process may have exited between the state check and kill.
    }
  };
  send('SIGTERM');
  const exited = await Promise.race([
    childExit(child).then(() => true),
    sleep(graceMs).then(() => false),
  ]);
  if (!exited) {
    send('SIGKILL');
    await childExit(child).catch(() => {});
  }
}

function forwardLines(stream, prefix, onLine = null) {
  let pending = '';
  stream.setEncoding('utf8');
  stream.on('data', (chunk) => {
    pending += chunk;
    const lines = pending.split(/\r?\n/);
    pending = lines.pop() || '';
    for (const line of lines) {
      if (line) console.log(`${prefix}${line}`);
      if (onLine) onLine(line);
    }
  });
  stream.on('end', () => {
    if (pending) {
      console.log(`${prefix}${pending}`);
      if (onLine) onLine(pending);
    }
  });
}

function acquireLock(lockPath) {
  const tryOpen = () => {
    const fd = fs.openSync(lockPath, 'wx', 0o600);
    fs.writeFileSync(fd, `${process.pid}\n`);
    return fd;
  };
  try {
    return tryOpen();
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
    const oldPidText = fs.readFileSync(lockPath, 'utf8').trim();
    const oldPid = Number(oldPidText);
    if (!oldPidText || !Number.isInteger(oldPid) || oldPid <= 0) {
      fs.unlinkSync(lockPath);
      return tryOpen();
    }
    try {
      process.kill(oldPid, 0);
      throw new Error(`Another recording supervisor is active with pid ${oldPid}`);
    } catch (pidError) {
      if (pidError.code !== 'ESRCH') throw pidError;
      fs.unlinkSync(lockPath);
      return tryOpen();
    }
  }
}

function rewritePaths(value, fromPath, toPath) {
  if (typeof value === 'string') return value.startsWith(fromPath) ? `${toPath}${value.slice(fromPath.length)}` : value;
  if (Array.isArray(value)) return value.map((item) => rewritePaths(item, fromPath, toPath));
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, rewritePaths(item, fromPath, toPath)]));
  }
  return value;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const joyaiScaffold = args.interactionScaffold === JOYAI_OFFICIAL_SCAFFOLD;
  const deployedAdapter = Boolean(args.deployedAdapterApiBase);
  if (!fs.existsSync(args.tasks)) throw new Error(`Task manifest does not exist: ${args.tasks}`);
  const webPassword = process.env.JOYVL_WEB_PASSWORD || '';
  if (!webPassword) throw new Error('JOYVL_WEB_PASSWORD is required');
  const upstreamKeys = deployedAdapter ? [] : resolveUpstreamKeys(args);
  let upstreamKeyIndex = 0;
  const blockedUpstreamKeySlots = new Set();
  const accessToken = deployedAdapter
    ? String(process.env[args.deployedAdapterAccessTokenEnv] || '')
    : crypto.randomBytes(32).toString('hex');
  if (!accessToken) {
    throw new Error(`${args.deployedAdapterAccessTokenEnv} is required`);
  }
  const secrets = [webPassword, ...upstreamKeys, accessToken].filter(Boolean);
  const tasksText = fs.readFileSync(args.tasks, 'utf8');
  const tasks = readTasks(args.tasks);
  const taskManifestSha256 = sha256(tasksText);
  const promptPath = path.join(ROOT, 'config', 'joyai_system_prompt.txt');
  const systemPromptFileBytes = fs.readFileSync(promptPath);
  const systemPromptFileSha256 = sha256(systemPromptFileBytes);
  const systemPrompt = systemPromptFileBytes.toString('utf8').trim();
  const systemPromptSha256 = sha256(systemPrompt);
  const queryAudioManifestSha256 = args.nativeQueryAudioManifest
    ? sha256(fs.readFileSync(args.nativeQueryAudioManifest))
    : '';
  const auditPath = deployedAdapter
    ? args.deployedAdapterAuditPath
    : path.join(args.out, 'prompt_proxy_audit.jsonl');
  const eventsPath = path.join(args.out, 'supervisor_events.jsonl');
  const progressPath = path.join(args.out, 'progress_checks.jsonl');
  const statePath = path.join(args.out, 'supervisor_state.json');
  const summaryPath = path.join(args.out, 'run_summary.json');
  const runtimeDir = path.join(args.out, '.runtime');
  const workDir = path.join(
    args.attemptWorkRoot,
    `${safeName(path.basename(args.out))}-${sha256(path.resolve(args.out)).slice(0, 12)}`,
  );
  const failedAttemptsDir = path.join(args.out, 'failed_attempts');
  const lastFailedArtifactsDir = path.join(args.out, 'last_failed_artifacts');
  const lockPath = path.join(args.out, '.supervisor.lock');
  await fsp.mkdir(args.out, { recursive: true });
  const lockFd = acquireLock(lockPath);

  const previousState = (() => {
    try { return JSON.parse(fs.readFileSync(statePath, 'utf8')); } catch { return null; }
  })();
  const evaluationContract = {
    schema: 2,
    model_request_id: args.model,
    upstream_protocol: args.upstreamProtocol,
    input_transport: args.inputTransport,
    interaction_scaffold: args.interactionScaffold,
    native_video_schema: args.nativeVideoSchema,
    realtime_protocol: args.realtimeProtocol,
    realtime_api_base: args.realtimeApiBase,
    realtime_query_mode: args.realtimeQueryMode,
    native_query_audio_manifest: args.nativeQueryAudioManifest,
    native_query_audio_manifest_sha256: queryAudioManifestSha256,
    native_streaming_mode_policy: args.nativeStreamingModePolicy,
    native_system_prompt_transport: args.nativeSystemPromptTransport,
    native_frame_clock: args.nativeFrameClock,
    max_realtime_session_s: args.maxRealtimeSessionS,
    max_provider_query_start_drift_s: args.maxProviderQueryStartDriftS,
    upstream_request_policy: {
      sdk_max_retries: args.upstreamSdkMaxRetries,
      non_query_timeout_s: args.nonQueryRequestTimeoutS,
      query_timeout_s: args.queryRequestTimeoutS,
      warmup_timeout_s: args.providerWarmupTimeoutS,
      timeout_classification: 'model_capability',
      timeout_transport: 'audited_synthetic_silence_http_200',
      rate_limit_and_transport_classification: 'retryable_infrastructure',
      infrastructure_error_transport: 'non_retryable_http_424',
    },
    thinking_disabled: args.disableThinking,
    scaffold_deployment: deployedAdapter ? {
      api_base: args.deployedAdapterApiBase,
      access_token_env: args.deployedAdapterAccessTokenEnv,
      audit_path: args.deployedAdapterAuditPath,
    } : null,
  };
  const previousRecordingProtocol = previousState
    ? (previousState.recording_protocol || 'legacy_pre_api_ready_single_pass')
    : CURRENT_RECORDING_PROTOCOL;
  const recordingProtocolChanged = Boolean(
    previousState && previousRecordingProtocol !== CURRENT_RECORDING_PROTOCOL,
  );
  const evaluationContractChanged = Boolean(
    previousState
    && !compatibleEvaluationContract(
      previousState.evaluation_contract || null,
      evaluationContract,
    ),
  );
  const evaluationContractMigrated = Boolean(
    previousState
    && !sameJson(previousState.evaluation_contract || null, evaluationContract)
    && compatibleEvaluationContract(previousState.evaluation_contract || null, evaluationContract),
  );
  const systemPromptChanged = Boolean(
    previousState?.system_prompt_sha256
    && previousState.system_prompt_sha256 !== systemPromptSha256,
  );
  const resetAllTaskState = recordingProtocolChanged
    || evaluationContractChanged
    || systemPromptChanged;
  const taskContracts = Object.fromEntries(tasks.map((task) => [task.id, task]));
  const sourceVideoSnapshots = Object.fromEntries(
    tasks.map((task) => [task.id, sourceVideoSnapshot(task)]),
  );
  const previousTaskContracts = previousState?.task_contracts || {};
  const previousSourceSnapshots = previousState?.source_video_snapshots || {};
  const previousHasPerTaskContracts = Object.keys(previousTaskContracts).length > 0
    && Object.keys(previousSourceSnapshots).length > 0;
  const manifestChanged = Boolean(
    previousState?.task_manifest_sha256
    && previousState.task_manifest_sha256 !== taskManifestSha256,
  );
  const changedTaskIds = new Set(previousState ? tasks
    .filter((task) => (
      resetAllTaskState
      || (!previousHasPerTaskContracts && manifestChanged)
      || (previousHasPerTaskContracts && (
        !sameJson(previousTaskContracts[task.id], task)
        || !sameJson(previousSourceSnapshots[task.id], sourceVideoSnapshots[task.id])
      ))
    ))
    .map((task) => task.id) : []);
  const priorRecoveryBatchCount = resetAllTaskState ? 0 : Math.max(
    Number(previousState?.recovery_batch_count || 0),
    Object.values(previousState?.failure_history || {}).some((records) => (
      Array.isArray(records) && records.length > 0
    )) ? 1 : 0,
  );
  const previousUpstreamApiBase = previousState?.upstream_api_base || '';
  const upstreamApiChanged = Boolean(
    previousUpstreamApiBase && previousUpstreamApiBase !== args.upstreamApiBase,
  );
  const migrationAt = timestamp();
  const upstreamApiHistory = Array.isArray(previousState?.upstream_api_history)
    ? structuredClone(previousState.upstream_api_history)
    : [];
  if (!upstreamApiHistory.length && previousUpstreamApiBase) {
    upstreamApiHistory.push({
      api_base: previousUpstreamApiBase,
      model_request_id: previousState?.model_request_id || args.model,
      started_at: previousState?.started_at || null,
      ended_at: upstreamApiChanged ? migrationAt : null,
      source: 'initial',
    });
  }
  if (upstreamApiChanged) {
    const previousEntry = upstreamApiHistory[upstreamApiHistory.length - 1];
    if (previousEntry?.api_base === previousUpstreamApiBase && !previousEntry.ended_at) {
      previousEntry.ended_at = migrationAt;
    }
    upstreamApiHistory.push({
      api_base: args.upstreamApiBase,
      model_request_id: args.model,
      started_at: migrationAt,
      ended_at: null,
      source: args.allowUpstreamMigration ? 'explicit_migration' : 'configuration_update',
    });
  } else if (!upstreamApiHistory.length) {
    upstreamApiHistory.push({
      api_base: args.upstreamApiBase,
      model_request_id: args.model,
      started_at: previousState?.started_at || migrationAt,
      ended_at: null,
      source: 'initial',
    });
  }

  let protocolMigrationArchive = null;
  const protocolMigrations = Array.isArray(previousState?.protocol_migrations)
    ? structuredClone(previousState.protocol_migrations)
    : [];
  if (recordingProtocolChanged) {
    const archiveDir = path.join(args.out, 'protocol_history');
    const archiveStamp = migrationAt.replace(/[^0-9A-Za-z]+/g, '_').replace(/^_+|_+$/g, '');
    await fsp.mkdir(archiveDir, { recursive: true });
    const archivedState = path.join(
      archiveDir,
      `supervisor_state_${safeName(previousRecordingProtocol)}_${archiveStamp}.json`,
    );
    const archivedSummary = path.join(
      archiveDir,
      `run_summary_${safeName(previousRecordingProtocol)}_${archiveStamp}.json`,
    );
    const archivedEvents = path.join(
      archiveDir,
      `supervisor_events_${safeName(previousRecordingProtocol)}_${archiveStamp}.jsonl`,
    );
    const archivedProgress = path.join(
      archiveDir,
      `progress_checks_${safeName(previousRecordingProtocol)}_${archiveStamp}.jsonl`,
    );
    await fsp.copyFile(statePath, archivedState);
    if (fs.existsSync(summaryPath)) await fsp.copyFile(summaryPath, archivedSummary);
    if (fs.existsSync(eventsPath)) await fsp.rename(eventsPath, archivedEvents);
    if (fs.existsSync(progressPath)) await fsp.rename(progressPath, archivedProgress);
    protocolMigrationArchive = {
      migrated_at: migrationAt,
      from_protocol: previousRecordingProtocol,
      to_protocol: CURRENT_RECORDING_PROTOCOL,
      archived_state: archivedState,
      archived_summary: fs.existsSync(archivedSummary) ? archivedSummary : '',
      archived_events: fs.existsSync(archivedEvents) ? archivedEvents : '',
      archived_progress: fs.existsSync(archivedProgress) ? archivedProgress : '',
      preserved_success_count: Object.values(previousState.results || {})
        .filter((item) => item?.status === 'ok').length,
      reset_failed_task_ids: Object.entries(previousState.failures || {})
        .filter(([, failure]) => failure?.status === 'failed')
        .map(([taskId]) => taskId)
        .sort(),
    };
    protocolMigrations.push(protocolMigrationArchive);
  }

  const state = {
    version: 1,
    status: 'running',
    pid: process.pid,
    started_at: previousState?.started_at || timestamp(),
    updated_at: timestamp(),
    finished_at: null,
    tasks_path: args.tasks,
    task_manifest_sha256: taskManifestSha256,
    task_contracts: taskContracts,
    source_video_snapshots: sourceVideoSnapshots,
    tasks_total: tasks.length,
    profile_id: args.profileId,
    model_request_id: args.model,
    upstream_api_base: args.upstreamApiBase,
    upstream_protocol: args.upstreamProtocol,
    input_transport: args.inputTransport,
    interaction_scaffold: args.interactionScaffold,
    evaluation_contract: evaluationContract,
    evaluation_protocol: ['image-frame-batch', 'stateful-frame-stream'].includes(args.inputTransport)
      ? JOYAI_FRAME_STREAM_PROTOCOL
      : args.realtimeProtocol,
    native_video_schema: args.nativeVideoSchema,
    realtime_protocol: args.realtimeProtocol,
    realtime_api_base: args.realtimeApiBase,
    realtime_query_mode: args.realtimeQueryMode,
    native_query_audio_manifest: args.nativeQueryAudioManifest,
    native_query_audio_manifest_sha256: queryAudioManifestSha256,
    native_streaming_mode_policy: args.nativeStreamingModePolicy,
    max_realtime_session_s: args.maxRealtimeSessionS,
    max_provider_query_start_drift_s: args.maxProviderQueryStartDriftS,
    upstream_api_history: upstreamApiHistory,
    recording_protocol: CURRENT_RECORDING_PROTOCOL,
    protocol_migrations: protocolMigrations,
    expected_backend_aliases: args.backendAliases,
    thinking_disabled: args.disableThinking,
    persistent_user_query: false,
    continue_after_task_failure: args.continueAfterTaskFailure,
    retry_failed_tasks: args.retryFailedTasks,
    max_failed_task_recovery_batches: args.maxFailedTaskRecoveryBatches,
    max_task_wall_s: args.maxTaskWallS,
    system_prompt_source_path: promptPath,
    system_prompt_file_sha256: systemPromptFileSha256,
    system_prompt_sha256: systemPromptSha256,
    retries_allowed: args.maxRetries,
    upstream_key_pool_size: upstreamKeys.length,
    active_upstream_key_slot: deployedAdapter ? null : upstreamKeyIndex + 1,
    upstream_key_rotations: Number(previousState?.upstream_key_rotations || 0),
    blocked_upstream_key_slots: [],
    upstream_blocked: null,
    current_task: null,
    tunnel_restarts: Number(previousState?.tunnel_restarts || 0),
    attempts: resetAllTaskState ? {} : structuredClone(previousState?.attempts || {}),
    results: resetAllTaskState ? {} : structuredClone(previousState?.results || {}),
    failures: resetAllTaskState ? {} : structuredClone(previousState?.failures || {}),
    failure_history: resetAllTaskState ? {} : structuredClone(previousState?.failure_history || {}),
    retry_attempt_limits: resetAllTaskState
      ? {}
      : structuredClone(previousState?.retry_attempt_limits || {}),
    task_wall_pass_started_at: resetAllTaskState
      ? {}
      : structuredClone(previousState?.task_wall_pass_started_at || {}),
    recovery_batch_count: priorRecoveryBatchCount,
    recovery_batches: !resetAllTaskState && Array.isArray(previousState?.recovery_batches)
      ? structuredClone(previousState.recovery_batches)
      : [],
    last_error: previousState?.last_error || '',
  };
  for (const taskId of changedTaskIds) {
    delete state.attempts[taskId];
    delete state.results[taskId];
    delete state.failures[taskId];
    delete state.failure_history[taskId];
    delete state.retry_attempt_limits[taskId];
    delete state.task_wall_pass_started_at[taskId];
  }
  const retryFailedTaskIds = new Set(args.retryFailedTasks
    ? Object.entries(state.failures)
      .filter(([, failure]) => failure?.status === 'failed')
      .map(([taskId]) => taskId)
    : []);
  const recoveryBatchCountForTask = (taskId) => state.recovery_batches.filter((batch) => (
    Array.isArray(batch?.tasks)
    && batch.tasks.some((task) => task?.task_id === taskId)
  )).length;
  const taskWallWindowOpen = (taskId) => {
    const startedAtMs = Date.parse(state.task_wall_pass_started_at[taskId] || '');
    return !Number.isFinite(startedAtMs)
      || Date.now() < startedAtMs + args.maxTaskWallS * 1000;
  };
  const continuedRecoveryTaskIds = new Set([...retryFailedTaskIds].filter((taskId) => (
    Number(state.retry_attempt_limits[taskId] || 0) > Number(state.attempts[taskId] || 0)
    && taskWallWindowOpen(taskId)
  )));
  const newRecoveryTaskIds = new Set([...retryFailedTaskIds].filter(
    (taskId) => !continuedRecoveryTaskIds.has(taskId),
  ));
  const recoveryLimitReachedTaskIds = [...newRecoveryTaskIds].filter(
    (taskId) => recoveryBatchCountForTask(taskId) >= args.maxFailedTaskRecoveryBatches,
  );
  if (recoveryLimitReachedTaskIds.length) {
    try { fs.closeSync(lockFd); } catch {}
    await fsp.unlink(lockPath).catch(() => {});
    throw new Error(
      `Failed-task recovery batch limit reached (${args.maxFailedTaskRecoveryBatches}) `
      + `for ${recoveryLimitReachedTaskIds.join(', ')}; `
      + 'refusing to expand those cumulative attempt limits again',
    );
  }
  if (newRecoveryTaskIds.size) {
    for (const taskId of newRecoveryTaskIds) {
      const batchIndex = recoveryBatchCountForTask(taskId) + 1;
      let batch = state.recovery_batches.find((item) => item?.batch_index === batchIndex);
      if (!batch) {
        batch = {
          batch_index: batchIndex,
          started_at: timestamp(),
          max_retries_per_task: args.maxRetries,
          attempts_per_task: args.maxRetries + 1,
          tasks: [],
        };
        state.recovery_batches.push(batch);
      }
      const attemptsAlreadyUsed = Number(state.attempts[taskId] || 0);
      batch.tasks.push({
        task_id: taskId,
        attempt_start: attemptsAlreadyUsed + 1,
        attempt_end: attemptsAlreadyUsed + args.maxRetries + 1,
        enrolled_at: timestamp(),
      });
      state.recovery_batch_count = Math.max(state.recovery_batch_count, batchIndex);
    }
  }
  for (const taskId of retryFailedTaskIds) {
    const failure = state.failures[taskId];
    const history = Array.isArray(state.failure_history[taskId])
      ? state.failure_history[taskId]
      : [];
    if (!history.some((item) => item?.failed_at === failure?.failed_at)) {
      history.push(structuredClone(failure));
    }
    state.failure_history[taskId] = history;
    if (newRecoveryTaskIds.has(taskId)) {
      state.retry_attempt_limits[taskId] = Number(state.attempts[taskId] || 0)
        + args.maxRetries + 1;
      delete state.task_wall_pass_started_at[taskId];
    }
    delete state.failures[taskId];
  }
  let stopping = false;
  let proxyChild = null;
  let tunnelChild = null;
  let tunnelUrl = '';
  let verifiedTunnelUrl = '';
  let captureChild = null;
  const activeAdapterApiBase = () => (
    deployedAdapter ? args.deployedAdapterApiBase : `${tunnelUrl}/v1`
  );
  const activeAdapterControlApiBase = () => (
    deployedAdapter
      ? args.deployedAdapterApiBase
      : `http://127.0.0.1:${args.proxyPort}/v1`
  );

  const logEvent = async (event, details = {}) => {
    const record = { timestamp: timestamp(), event, ...details };
    await appendJsonl(eventsPath, record);
    console.log(`[supervisor] ${record.timestamp} ${event}${details.task_id ? ` task=${details.task_id}` : ''}`);
  };
  if (protocolMigrationArchive) {
    await logEvent('recording_protocol_migrated', protocolMigrationArchive);
  }
  if (
    previousState
    && (
      evaluationContractChanged
      || evaluationContractMigrated
      || systemPromptChanged
      || changedTaskIds.size
    )
  ) {
    await logEvent('recording_configuration_adjusted', {
      evaluation_contract_changed: evaluationContractChanged,
      evaluation_contract_compatibly_migrated: evaluationContractMigrated,
      system_prompt_changed: systemPromptChanged,
      reset_task_ids: [...changedTaskIds].sort(),
      preserved_task_ids: tasks
        .map((task) => task.id)
        .filter((taskId) => !changedTaskIds.has(taskId))
        .sort(),
    });
  }

  const canonicalDir = (task, index) => path.join(
    args.out,
    `${String(index + 1).padStart(3, '0')}_${safeName(task.id)}`,
  );

  const readCanonical = (task, index) => {
    const dir = canonicalDir(task, index);
    try {
      const summary = JSON.parse(fs.readFileSync(path.join(dir, 'summary.json'), 'utf8'));
      const validation = JSON.parse(fs.readFileSync(path.join(dir, 'capture_validation.json'), 'utf8'));
      const videoPath = summary?.files?.task_mp4;
      const valid = summary?.status === 'ok'
        && summary?.task?.id === task.id
        && sameJson(validation?.task_contract, task)
        && sameJson(validation?.source_video_snapshot, sourceVideoSnapshot(task))
        && compatibleEvaluationContract(validation?.evaluation_contract, evaluationContract)
        && summary?.model_response_outcome?.response_semantics
          === MODEL_RESPONSE_OUTCOME_PROTOCOL
        && summary?.vlm_backend_identity?.ok === true
        && validation?.system_prompt_sha256 === systemPromptSha256
        && validation?.model_identity_ok === true
        && (
          args.inputTransport !== 'native-video-realtime'
          || validation?.native_realtime_audit?.ok === true
        )
        && videoPath
        && fs.existsSync(videoPath)
        && fs.statSync(videoPath).size > 0;
      return valid ? { dir, summary, validation, videoPath } : null;
    } catch {
      return null;
    }
  };

  const reconcileCompleted = () => {
    for (let index = 0; index < tasks.length; index += 1) {
      const existing = readCanonical(tasks[index], index);
      if (existing) {
        state.results[tasks[index].id] = {
          status: 'ok',
          task_index: index + 1,
          task_id: tasks[index].id,
          task_dir: existing.dir,
          task_mp4: existing.videoPath,
          finished_at: existing.summary.finished_at || existing.validation.validated_at,
          attempts: Number(state.attempts[tasks[index].id] || existing.validation.attempt || 1),
          model_identity: existing.summary.vlm_backend_identity.observed_models || [],
        };
        delete state.failures[tasks[index].id];
        delete state.retry_attempt_limits[tasks[index].id];
        delete state.task_wall_pass_started_at[tasks[index].id];
      } else if (state.results[tasks[index].id]?.status === 'ok') {
        delete state.results[tasks[index].id];
      }
    }
  };

  const buildSummary = () => {
    const taskResults = tasks.map((task, index) => {
      const result = state.results[task.id];
      const failure = state.failures[task.id];
      return result || failure || {
        task_index: index + 1,
        task_id: task.id,
        status: state.current_task?.task_id === task.id ? 'running' : 'pending',
        attempts: Number(state.attempts[task.id] || 0),
        task_dir: '',
        task_mp4: '',
      };
    });
    const counts = taskResults.reduce((accumulator, item) => {
      accumulator[item.status] = (accumulator[item.status] || 0) + 1;
      return accumulator;
    }, {});
    const attemptAudit = buildRecordingAttemptAudit({
      tasks,
      state,
      eventsPath,
      maxRetriesPerBatch: args.maxRetries,
      maxRecoveryBatches: args.maxFailedTaskRecoveryBatches,
    });
    return {
      status: state.status,
      started_at: state.started_at,
      updated_at: state.updated_at,
      finished_at: state.finished_at,
      output_dir: args.out,
      tasks_path: args.tasks,
      task_manifest_sha256: taskManifestSha256,
      task_contracts: state.task_contracts,
      source_video_snapshots: state.source_video_snapshots,
      profile_id: args.profileId,
      model_request_id: args.model,
      upstream_api_base: args.upstreamApiBase,
      upstream_protocol: args.upstreamProtocol,
      input_transport: args.inputTransport,
      evaluation_protocol: state.evaluation_protocol,
      native_video_schema: args.nativeVideoSchema,
      realtime_protocol: args.realtimeProtocol,
      realtime_api_base: args.realtimeApiBase,
      realtime_query_mode: args.realtimeQueryMode,
      native_query_audio_manifest: args.nativeQueryAudioManifest,
      native_query_audio_manifest_sha256: queryAudioManifestSha256,
      max_realtime_session_s: args.maxRealtimeSessionS,
      max_provider_query_start_drift_s: args.maxProviderQueryStartDriftS,
      upstream_api_history: state.upstream_api_history,
      recording_protocol: state.recording_protocol,
      protocol_migrations: state.protocol_migrations,
      expected_backend_aliases: args.backendAliases,
      thinking_disabled: args.disableThinking,
      scaffold_deployment: evaluationContract.scaffold_deployment,
      persistent_user_query: false,
      continue_after_task_failure: args.continueAfterTaskFailure,
      retry_failed_tasks: args.retryFailedTasks,
      max_task_wall_s: args.maxTaskWallS,
      task_wall_pass_started_at: state.task_wall_pass_started_at,
      system_prompt_source_path: promptPath,
      system_prompt_file_sha256: systemPromptFileSha256,
      system_prompt_sha256: systemPromptSha256,
      retries_allowed: args.maxRetries,
      upstream_key_pool_size: state.upstream_key_pool_size,
      active_upstream_key_slot: state.active_upstream_key_slot,
      upstream_key_rotations: state.upstream_key_rotations,
      blocked_upstream_key_slots: state.blocked_upstream_key_slots,
      upstream_blocked: state.upstream_blocked,
      tunnel_restarts: state.tunnel_restarts,
      current_task: state.current_task,
      tasks_total: tasks.length,
      counts,
      task_results: taskResults,
      attempt_audit: attemptAudit,
      last_error: state.last_error,
    };
  };

  const persistState = async () => {
    state.updated_at = timestamp();
    await writeJsonAtomic(statePath, state);
    await writeJsonAtomic(summaryPath, buildSummary());
  };

  if (upstreamApiChanged) {
    await appendJsonl(eventsPath, {
      timestamp: migrationAt,
      event: 'upstream_api_migrated',
      from: previousUpstreamApiBase,
      to: args.upstreamApiBase,
      model_request_id: args.model,
      explicit: true,
    });
  }

  const writeProgress = async (source) => {
    reconcileCompleted();
    const completed = Object.values(state.results).filter((item) => item.status === 'ok').length;
    await appendJsonl(progressPath, {
      timestamp: timestamp(),
      source,
      supervisor_pid: process.pid,
      status: state.status,
      completed,
      total: tasks.length,
      current_task: state.current_task,
      tunnel_restarts: state.tunnel_restarts,
      active_upstream_key_slot: state.active_upstream_key_slot,
      upstream_key_rotations: state.upstream_key_rotations,
      last_error: state.last_error,
    });
    await persistState();
    console.log(`[progress] ${completed}/${tasks.length} complete; current=${state.current_task?.task_id || 'none'}`);
  };

  const ensureProxy = async () => {
    if (deployedAdapter) return;
    if (proxyChild && proxyChild.exitCode == null && !proxyChild.signalCode) return;
    const nativeRealtime = args.inputTransport === 'native-video-realtime';
    if (!joyaiScaffold && !nativeRealtime) {
      throw new Error('The release supervisor supports only the configured formal runners');
    }
    const proxyScript = joyaiScaffold
      ? 'joyai_scaffold_adapter.py'
      : args.realtimeProtocol === 'joyai-http-session-v1'
        ? 'joyai_native_http_session_proxy.mjs'
        : 'native_video_realtime_proxy.mjs';
    const expectedProxyProtocol = args.upstreamProtocol;
    const proxyExecutable = joyaiScaffold
      ? (process.env.PYTHON_BIN || 'python3')
      : process.execPath;
    proxyChild = spawn(proxyExecutable, [path.join(ROOT, 'scripts', proxyScript)], {
      cwd: ROOT,
      env: {
        ...process.env,
        UPSTREAM_API_BASE: args.upstreamApiBase,
        UPSTREAM_API_KEY: upstreamKeys[upstreamKeyIndex],
        PROMPT_PROXY_UPSTREAM_PROTOCOL: args.upstreamProtocol,
        PROMPT_PROXY_INPUT_TRANSPORT: args.inputTransport,
        PROMPT_PROXY_DISABLE_THINKING: args.disableThinking ? '1' : '0',
        PROMPT_PROXY_NORMALIZE_ACTION_TAGS: '1',
        PROMPT_PROXY_ADVERTISED_MODEL: args.model,
        PROMPT_PROXY_ACCESS_TOKEN: accessToken,
        PROMPT_PROXY_HOST: '127.0.0.1',
        PROMPT_PROXY_PORT: String(args.proxyPort),
        PROMPT_PROXY_AUDIT_PATH: auditPath,
        JOYAI_SYSTEM_PROMPT_FILE: promptPath,
        JOYAI_SCAFFOLD_ENABLE_SUMMARIZER: '0',
        JOYAI_SCAFFOLD_SDK_MAX_RETRIES: String(args.upstreamSdkMaxRetries),
        JOYAI_SCAFFOLD_NON_QUERY_TIMEOUT_SECONDS: String(
          args.nonQueryRequestTimeoutS,
        ),
        JOYAI_SCAFFOLD_QUERY_TIMEOUT_SECONDS: String(args.queryRequestTimeoutS),
        JOYAI_SCAFFOLD_WARMUP_TIMEOUT_SECONDS: String(args.providerWarmupTimeoutS),
        NATIVE_VIDEO_SCHEMA: args.nativeVideoSchema,
        NATIVE_REALTIME_PROTOCOL: args.realtimeProtocol,
        NATIVE_REALTIME_API_BASE: args.realtimeApiBase,
        NATIVE_REALTIME_QUERY_MODE: args.realtimeQueryMode,
        NATIVE_QUERY_AUDIO_MANIFEST: args.nativeQueryAudioManifest,
        NATIVE_STREAMING_MODE_POLICY: args.nativeStreamingModePolicy,
        NATIVE_HTTP_SYSTEM_PROMPT_TRANSPORT: args.nativeSystemPromptTransport,
        NATIVE_HTTP_FRAME_CLOCK: args.nativeFrameClock,
        NATIVE_REALTIME_MAX_SESSION_S: String(args.maxRealtimeSessionS || ''),
        NATIVE_HTTP_NON_QUERY_TIMEOUT_MS: String(args.nonQueryRequestTimeoutS * 1000),
        NATIVE_HTTP_QUERY_TIMEOUT_MS: String(args.queryRequestTimeoutS * 1000),
        NATIVE_HTTP_RESET_TIMEOUT_MS: String(args.queryRequestTimeoutS * 1000),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    forwardLines(proxyChild.stdout, '[model-proxy] ');
    forwardLines(proxyChild.stderr, '[model-proxy] ');
    for (let attempt = 1; attempt <= 30; attempt += 1) {
      if (proxyChild.exitCode != null || proxyChild.signalCode) {
        throw new Error(`Prompt proxy exited before becoming healthy (code=${proxyChild.exitCode})`);
      }
      try {
        const health = await request(`http://127.0.0.1:${args.proxyPort}/health`, { timeoutMs: 3000 });
        if (health.json?.system_prompt_sha256 !== systemPromptSha256) {
          throw new Error('Prompt proxy is serving a different system prompt');
        }
        if (health.json?.system_prompt_file_sha256 !== systemPromptFileSha256) {
          throw new Error('Prompt proxy is serving a different system prompt source file');
        }
        if (health.json?.upstream_protocol !== expectedProxyProtocol) {
          throw new Error('Model proxy is serving a different upstream protocol');
        }
        if (health.json?.input_transport !== args.inputTransport) {
          throw new Error('Prompt proxy is serving a different input transport');
        }
        if (
          joyaiScaffold
          && (
            health.json?.adapter !== JOYAI_OFFICIAL_SCAFFOLD
            || health.json?.query_context_policy !== 'official-live-adapter-history-replay'
          )
        ) {
          throw new Error('Model proxy is not the pinned official JoyAI live adapter');
        }
        if (
          joyaiScaffold
          && (
            health.json?.upstream_request_policy?.sdk_max_retries
              !== args.upstreamSdkMaxRetries
            || health.json?.upstream_request_policy?.non_query_timeout_s
              !== args.nonQueryRequestTimeoutS
            || health.json?.upstream_request_policy?.query_timeout_s
              !== args.queryRequestTimeoutS
            || health.json?.upstream_request_policy?.warmup_timeout_s
              !== args.providerWarmupTimeoutS
          )
        ) {
          throw new Error('Model proxy is serving a different upstream request policy');
        }
        if (
          !nativeRealtime
          && health.json?.evaluation_protocol !== JOYAI_FRAME_STREAM_PROTOCOL
        ) {
          throw new Error('Prompt proxy is not serving the JoyAI formal frame-stream protocol');
        }
        if (health.json?.query_delivery !== 'once_per_round') {
          throw new Error('Model proxy does not enforce once-per-round Query delivery');
        }
        if (nativeRealtime && health.json?.native_video_schema !== args.nativeVideoSchema) {
          throw new Error('Native realtime proxy is serving a different video schema');
        }
        if (nativeRealtime && health.json?.realtime_protocol !== args.realtimeProtocol) {
          throw new Error('Native realtime proxy is serving a different realtime protocol');
        }
        if (nativeRealtime && health.json?.query_transport !== args.realtimeQueryMode) {
          throw new Error('Native realtime proxy is serving a different Query transport');
        }
        if (
          args.realtimeQueryMode === 'input-audio-once'
          && (
            health.json?.query_audio_manifest_sha256 !== queryAudioManifestSha256
            || health.json?.query_audio_realtime_pacing !== true
          )
        ) {
          throw new Error('Native realtime proxy is serving a different Query audio contract');
        }
        if (
          nativeRealtime
          && health.json?.streaming_mode_policy !== args.nativeStreamingModePolicy
        ) {
          throw new Error('Native realtime proxy is serving a different streaming mode policy');
        }
        if (
          nativeRealtime
          && health.json?.system_prompt_transport !== args.nativeSystemPromptTransport
        ) {
          throw new Error('Native realtime proxy is serving a different system-prompt transport');
        }
        if (nativeRealtime && health.json?.frame_clock_policy !== args.nativeFrameClock) {
          throw new Error('Native realtime proxy is serving a different frame clock');
        }
        await logEvent('model_proxy_ready', {
          system_prompt_source_path: promptPath,
          system_prompt_file_sha256: systemPromptFileSha256,
          system_prompt_sha256: systemPromptSha256,
          upstream_protocol: expectedProxyProtocol,
          input_transport: args.inputTransport,
          evaluation_protocol: nativeRealtime
            ? args.realtimeProtocol
            : JOYAI_FRAME_STREAM_PROTOCOL,
          query_delivery: 'once_per_round',
      interaction_scaffold: args.interactionScaffold,
      evaluation_contract: state.evaluation_contract,
        });
        return;
      } catch (error) {
        if (attempt >= 30) throw error;
        await sleep(1000);
      }
    }
  };

  const stopProxy = async () => {
    if (deployedAdapter) return;
    await terminateChild(proxyChild, { graceMs: 5000 });
    proxyChild = null;
    verifiedTunnelUrl = '';
  };

  const syncKeyPoolState = () => {
    state.active_upstream_key_slot = deployedAdapter ? null : upstreamKeyIndex + 1;
    state.blocked_upstream_key_slots = [...blockedUpstreamKeySlots]
      .map((slot) => slot + 1)
      .sort((left, right) => left - right);
  };

  const rotateUpstreamKey = async (reason, error) => {
    blockedUpstreamKeySlots.add(upstreamKeyIndex);
    const fromIndex = upstreamKeyIndex;
    let nextIndex = -1;
    for (let offset = 1; offset <= upstreamKeys.length; offset += 1) {
      const candidate = (fromIndex + offset) % upstreamKeys.length;
      if (!blockedUpstreamKeySlots.has(candidate)) {
        nextIndex = candidate;
        break;
      }
    }
    syncKeyPoolState();
    if (nextIndex < 0) return false;
    upstreamKeyIndex = nextIndex;
    state.upstream_key_rotations += 1;
    state.upstream_blocked = null;
    state.last_error = '';
    syncKeyPoolState();
    await stopProxy();
    await logEvent('upstream_key_rotated', {
      reason,
      from_slot: fromIndex + 1,
      to_slot: nextIndex + 1,
      pool_size: upstreamKeys.length,
      blocked_slots: state.blocked_upstream_key_slots,
      error: replaceAllSecrets(error?.message || error, secrets),
    });
    await persistState();
    return true;
  };

  const resetUpstreamKeyPool = async (reason) => {
    blockedUpstreamKeySlots.clear();
    upstreamKeyIndex = 0;
    syncKeyPoolState();
    await stopProxy();
    await logEvent('upstream_key_pool_retry', {
      reason,
      active_slot: state.active_upstream_key_slot,
      pool_size: upstreamKeys.length,
    });
    await persistState();
  };

  const stopTunnel = async () => {
    if (deployedAdapter) return;
    await terminateChild(tunnelChild, { graceMs: 5000 });
    tunnelChild = null;
    tunnelUrl = '';
    verifiedTunnelUrl = '';
  };

  const startTunnel = async () => {
    if (deployedAdapter) {
      throw new Error('A deployed adapter does not use a Cloudflare tunnel');
    }
    await stopTunnel();
    state.tunnel_restarts += 1;
    let resolveUrl;
    let rejectUrl;
    const urlPromise = new Promise((resolve, reject) => {
      resolveUrl = resolve;
      rejectUrl = reject;
    });
    let settled = false;
    const observe = (line) => {
      const match = line.match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com/i);
      if (match && !settled) {
        settled = true;
        resolveUrl(match[0]);
      }
    };
    tunnelChild = spawn(args.cloudflared, [
      'tunnel', '--no-autoupdate', '--protocol', 'http2',
      '--url', `http://127.0.0.1:${args.proxyPort}`,
    ], {
      cwd: ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    forwardLines(tunnelChild.stdout, '[cloudflared] ', observe);
    forwardLines(tunnelChild.stderr, '[cloudflared] ', observe);
    tunnelChild.once('error', (error) => {
      if (!settled) {
        settled = true;
        rejectUrl(error);
      }
    });
    tunnelChild.once('exit', (code, signal) => {
      if (!settled) {
        settled = true;
        rejectUrl(new Error(`cloudflared exited before publishing a URL: code=${code}, signal=${signal || ''}`));
      }
    });
    tunnelUrl = await Promise.race([
      urlPromise,
      sleep(60000).then(() => { throw new Error('Timed out waiting for cloudflared quick-tunnel URL'); }),
    ]);
    for (let attempt = 1; attempt <= 30; attempt += 1) {
      try {
        const health = await request(`${tunnelUrl}/health`, { timeoutMs: 10000 });
        if (health.json?.system_prompt_sha256 !== systemPromptSha256) {
          throw new Error('Public tunnel reached an unexpected model proxy');
        }
        await logEvent('tunnel_ready', { tunnel_origin: tunnelUrl, restart: state.tunnel_restarts });
        return;
      } catch (error) {
        if (attempt >= 30) throw error;
        await sleep(2000);
      }
    }
  };

  const verifyWebUiRoute = async () => {
    const modelsUrl = `${activeAdapterApiBase().replace(/\/+$/, '')}/models`;
    const catalogResponse = await request(modelsUrl, {
      timeoutMs: 90000,
      bearer: accessToken,
    });
    const modelRows = Array.isArray(catalogResponse.json?.models)
      ? catalogResponse.json.models
      : Array.isArray(catalogResponse.json?.data) ? catalogResponse.json.data : [];
    const available = modelRows.map((item) => String(item?.id || '')).filter(Boolean);
    if (!modelRows.length) {
      throw new Error('Adapter did not return an upstream model catalog');
    }
    if (!available.includes(args.model)) {
      throw new Error(
        `WebUI model catalog does not contain requested model ${args.model}; `
        + `available=${available.join(', ') || '(none)'}`,
      );
    }

    await logEvent('adapter_route_verified', {
      model: args.model,
      route: deployedAdapter ? 'in_pod_scaffold_adapter' : 'cloudflare_scaffold_adapter',
      adapter_api_base: activeAdapterApiBase(),
      catalog_contains_requested_model: available.includes(args.model),
      available_count: available.length,
      multimodal_inference_deferred_to_capture_warmup: true,
    });
  };

  const taskWallTimeoutError = (taskId, phase) => {
    const error = new Error(
      `Task ${taskId} exceeded the ${args.maxTaskWallS}s cumulative wall limit during ${phase}`,
    );
    error.code = 'TASK_WALL_TIMEOUT';
    return error;
  };

  const assertTaskWallBudget = (taskId, deadlineMs, phase) => {
    if (deadlineMs && Date.now() >= deadlineMs) throw taskWallTimeoutError(taskId, phase);
  };

  const ensureRoute = async (reason, taskId = '', taskWallDeadlineMs = 0) => {
    let cycle = 0;
    while (!stopping) {
      assertTaskWallBudget(taskId, taskWallDeadlineMs, 'route recovery');
      cycle += 1;
      try {
        if (deployedAdapter) {
          await verifyWebUiRoute();
          state.upstream_blocked = null;
          state.last_error = '';
          await logEvent('route_ready', {
            model: args.model,
            route: 'in_pod_scaffold_adapter',
            adapter_api_base: activeAdapterApiBase(),
          });
          assertTaskWallBudget(taskId, taskWallDeadlineMs, 'route verification');
          await persistState();
          return;
        }
        await ensureProxy();
        const localHealth = await request(`http://127.0.0.1:${args.proxyPort}/health`, { timeoutMs: 5000 });
        if (localHealth.json?.system_prompt_sha256 !== systemPromptSha256) {
          throw new Error('Local prompt proxy health hash mismatch');
        }
        if (!tunnelChild || tunnelChild.exitCode != null || !tunnelUrl) await startTunnel();
        const publicHealth = await request(`${tunnelUrl}/health`, { timeoutMs: 15000 });
        if (publicHealth.json?.system_prompt_sha256 !== systemPromptSha256) {
          throw new Error('Public prompt proxy health hash mismatch');
        }
        if (verifiedTunnelUrl !== tunnelUrl) {
          await verifyWebUiRoute();
          verifiedTunnelUrl = tunnelUrl;
        }
        state.upstream_blocked = null;
        state.last_error = '';
        assertTaskWallBudget(taskId, taskWallDeadlineMs, 'route verification');
        await logEvent('route_ready', { model: args.model, tunnel_origin: tunnelUrl });
        await persistState();
        return;
      } catch (error) {
        if (error?.code === 'TASK_WALL_TIMEOUT') throw error;
        state.last_error = replaceAllSecrets(error.message, secrets);
        if (isKeySafetyLimit(error) && upstreamKeys.length > 1) {
          const rotated = await rotateUpstreamKey(reason, error);
          if (rotated) continue;
        }
        if (isTemporarilyBlockedUpstream(error)) {
          const checkedAt = timestamp();
          const retryAt = new Date(Date.now() + args.upstreamBlockedBackoffS * 1000).toISOString();
          state.upstream_blocked = {
            since: state.upstream_blocked?.since || checkedAt,
            last_checked_at: checkedAt,
            next_retry_at: retryAt,
            backoff_s: args.upstreamBlockedBackoffS,
            error: state.last_error,
          };
          await logEvent('upstream_temporarily_blocked', {
            reason,
            cycle,
            retry_at: retryAt,
            backoff_s: args.upstreamBlockedBackoffS,
            error: state.last_error,
          });
          await persistState();
          const remainingMs = taskWallDeadlineMs
            ? taskWallDeadlineMs - Date.now()
            : args.upstreamBlockedBackoffS * 1000;
          if (remainingMs <= 0) throw taskWallTimeoutError(taskId, 'upstream backoff');
          await sleep(Math.min(args.upstreamBlockedBackoffS * 1000, remainingMs));
          if (blockedUpstreamKeySlots.size >= upstreamKeys.length && upstreamKeys.length > 1) {
            await resetUpstreamKeyPool('all key slots reached a safety limit; retry after backoff');
          }
          continue;
        }
        await logEvent('route_recovery_failed', { reason, cycle, error: state.last_error });
        await stopTunnel();
        if (proxyChild && (proxyChild.exitCode != null || proxyChild.signalCode)) proxyChild = null;
        await persistState();
        const retryDelayMs = Math.min(60, cycle * 10) * 1000;
        const remainingMs = taskWallDeadlineMs
          ? taskWallDeadlineMs - Date.now()
          : retryDelayMs;
        if (remainingMs <= 0) throw taskWallTimeoutError(taskId, 'route retry backoff');
        await sleep(Math.min(retryDelayMs, remainingMs));
      }
    }
    throw new Error('Supervisor stop requested while recovering route');
  };

  const writeRuntimeFiles = async (task) => {
    await fsp.mkdir(runtimeDir, { recursive: true });
    const configPath = path.join(runtimeDir, 'vlm_models.json');
    const taskPath = path.join(runtimeDir, 'current_task.jsonl');
    await writeJsonAtomic(configPath, {
      version: 1,
      profiles: [{
        id: args.profileId,
        model: args.model,
        api_base: activeAdapterApiBase(),
        route: 'direct',
        interaction_scaffold: args.interactionScaffold,
        formal_eval: false,
        enabled: true,
        api_key_env: 'PROMPT_PROXY_ACCESS_TOKEN',
        input_transport: args.inputTransport,
        ...(args.nativeVideoSchema ? {
          native_video_schema: args.nativeVideoSchema,
          native_video_verified: true,
        } : {}),
        ...(args.realtimeProtocol ? {
          realtime_protocol: args.realtimeProtocol,
          realtime_api_base: args.realtimeApiBase,
          realtime_query_mode: args.realtimeQueryMode,
          native_streaming_mode_policy: args.nativeStreamingModePolicy,
          native_system_prompt_transport: args.nativeSystemPromptTransport,
          native_frame_clock: args.nativeFrameClock,
          max_realtime_session_s: args.maxRealtimeSessionS,
        } : {}),
        preflight: false,
        identity_check: true,
        backend_aliases: args.backendAliases,
        tags: joyaiScaffold
          ? [deployedAdapter ? 'in-pod' : 'direct', 'joyai-official-live-adapter']
          : ['direct', 'system-prompt-proxy'],
        description: joyaiScaffold
          ? deployedAdapter
            ? 'Predeployed in-Pod route through the pinned official JoyAI live adapter.'
            : 'Ephemeral external VLM route through the pinned official JoyAI live adapter.'
          : args.inputTransport === 'native-video-realtime'
          ? 'Ephemeral provider-native realtime video route with Query-once delivery.'
          : 'Ephemeral direct VLM route with exact JoyAI system-prompt injection.',
      }],
    });
    await fsp.writeFile(taskPath, `${JSON.stringify(task)}\n`);
    return { configPath, taskPath };
  };

  const auditOffset = () => {
    try { return fs.statSync(auditPath).size; } catch { return 0; }
  };

  const readAuditSince = async (offset) => {
    let buffer;
    try {
      const handle = await fsp.open(auditPath, 'r');
      const size = (await handle.stat()).size;
      buffer = Buffer.alloc(Math.max(0, size - offset));
      if (buffer.length) await handle.read(buffer, 0, buffer.length, offset);
      await handle.close();
    } catch {
      return [];
    }
    return buffer.toString('utf8').split(/\r?\n/).filter(Boolean).flatMap((line) => {
      try { return [JSON.parse(line)]; } catch { return []; }
    });
  };

  const runCaptureAttempt = async (
    task,
    index,
    attempt,
    maxAttempts,
    taskWallDeadlineMs,
  ) => {
    assertTaskWallBudget(task.id, taskWallDeadlineMs, 'capture startup');
    const taskWorkDir = path.join(
      workDir,
      `${String(index + 1).padStart(3, '0')}_${safeName(task.id)}.attempt-${attempt}`,
    );
    await removePathWithRetries(taskWorkDir);
    await fsp.mkdir(taskWorkDir, { recursive: true });
    const { configPath, taskPath } = await writeRuntimeFiles(task);
    if (!deployedAdapter) {
      const resetResponse = await request(`http://127.0.0.1:${args.proxyPort}/reset`, {
        method: 'POST',
        timeoutMs: args.inputTransport === 'native-video-realtime'
          ? args.queryRequestTimeoutS * 1000
          : 5000,
        bearer: accessToken,
        payload: {},
      });
      if (resetResponse.json?.ok !== true) {
        throw new Error('Prompt proxy conversation reset failed');
      }
    }
    const beforeAudit = auditOffset();
    const capturePath = path.join(ROOT, 'scripts', 'capture.mjs');
    const commandArgs = [
      capturePath,
      '--tasks', taskPath,
      '--out', taskWorkDir,
      '--web-url', args.webUrl,
      '--web-username', args.webUsername,
      '--vlm-config', configPath,
      '--vlm-profile', args.profileId,
      '--vlm-control-api-base', activeAdapterControlApiBase(),
      '--local-video-mode', 'upload',
      '--process-interval-s', String(args.processIntervalS),
      '--frames-per-batch', String(args.framesPerBatch),
      '--task-retries', '0',
      '--health-retries', '4',
      '--health-interval-s', '5',
      '--stream-ready-timeout-s', '180',
      '--video-upload-timeout-s', '900',
      '--post-video-response-timeout-s', String(MODEL_RESPONSE_TIMEOUT_S),
      '--stream-start-retries', '1',
      '--stream-start-retry-gap-s', '5',
      '--attempt-timeout-margin-s', '600',
      '--no-full-ui-mp4',
      '--fresh-browser-per-attempt',
    ];
    if (args.retainLastFailedArtifacts) commandArgs.push('--keep-failed-attempts');
    console.log(`[capture] starting ${task.id}, attempt ${attempt}/${maxAttempts}`);
    captureChild = spawn(process.execPath, commandArgs, {
      cwd: ROOT,
      detached: true,
      env: {
        ...process.env,
        JOYVL_WEB_PASSWORD: webPassword,
        PROMPT_PROXY_ACCESS_TOKEN: accessToken,
      },
      stdio: ['ignore', 'inherit', 'inherit'],
    });
    let routeFailureResolve;
    const routeFailure = new Promise((resolve) => { routeFailureResolve = resolve; });
    const onTunnelExit = (code, signal) => routeFailureResolve({
      reason: `cloudflared exited during capture: code=${code}, signal=${signal || ''}`,
    });
    if (!deployedAdapter) tunnelChild.once('exit', onTunnelExit);
    let healthFailures = 0;
    let healthRunning = false;
    const healthTimer = deployedAdapter ? null : setInterval(async () => {
      if (healthRunning || !tunnelUrl) return;
      healthRunning = true;
      try {
        await request(`${tunnelUrl}/health`, { timeoutMs: 15000 });
        healthFailures = 0;
      } catch (error) {
        healthFailures += 1;
        if (healthFailures >= 2) {
          routeFailureResolve({ reason: `public tunnel health failed twice: ${error.message}` });
        }
      } finally {
        healthRunning = false;
      }
    }, 60000);
    const captureExitPromise = childExit(captureChild).then((result) => ({ type: 'capture_exit', ...result }));
    let taskWallTimer = null;
    const taskWallTimeout = new Promise((resolve) => {
      const remainingMs = Math.max(1, taskWallDeadlineMs - Date.now());
      taskWallTimer = setTimeout(() => resolve({
        reason: `Task ${task.id} exceeded the ${args.maxTaskWallS}s cumulative wall limit`,
      }), remainingMs);
    });
    const winner = await Promise.race([
      captureExitPromise,
      routeFailure.then((failure) => ({ type: 'route_failure', ...failure })),
      taskWallTimeout.then((failure) => ({ type: 'task_wall_timeout', ...failure })),
    ]);
    if (taskWallTimer) clearTimeout(taskWallTimer);
    if (healthTimer) clearInterval(healthTimer);
    tunnelChild?.removeListener('exit', onTunnelExit);
    if (winner.type === 'route_failure') {
      await logEvent('capture_route_lost', { task_id: task.id, attempt, error: winner.reason });
      await terminateChild(captureChild, { processGroup: true, graceMs: 10000 });
    }
    if (winner.type === 'task_wall_timeout') {
      await logEvent('capture_task_wall_timeout', {
        task_id: task.id,
        task_index: index + 1,
        attempt,
        max_task_wall_s: args.maxTaskWallS,
        error: winner.reason,
      });
      await terminateChild(captureChild, { processGroup: true, graceMs: 10000 });
    }
    const captureExit = await captureExitPromise;
    captureChild = null;
    await sleep(500);

    const captureSummary = (() => {
      try { return JSON.parse(fs.readFileSync(path.join(taskWorkDir, 'run_summary.json'), 'utf8')); } catch { return null; }
    })();
    const taskResult = captureSummary?.task_results?.[0] || null;
    const sourceTaskDir = taskResult?.task_dir || path.join(taskWorkDir, `001_${safeName(task.id)}`);
    const taskSummaryPath = path.join(sourceTaskDir, 'summary.json');
    const taskSummary = (() => {
      try { return JSON.parse(fs.readFileSync(taskSummaryPath, 'utf8')); } catch { return null; }
    })();
    const attemptAuditEvents = await readAuditSince(beforeAudit);
    const nativeRealtime = args.inputTransport === 'native-video-realtime';
    const requestAuditEvent = nativeRealtime
      ? 'native_realtime_frame_forwarded'
      : joyaiScaffold
      ? 'joyai_scaffold_frame_received'
      : 'system_prompt_injected';
    const auditEvents = attemptAuditEvents.filter((event) => (
      event.event === requestAuditEvent
      && event.system_prompt_sha256 === systemPromptSha256
      && String(event.model || '').toLowerCase() === args.model.toLowerCase()
      && (
        nativeRealtime
        || (
          event.evaluation_protocol === JOYAI_FRAME_STREAM_PROTOCOL
          && event.video_input_count === 0
          && event.image_count === 1
          && (
            joyaiScaffold
            || (
              Array.isArray(event.forwarded_roles)
              && event.forwarded_roles[0] === 'system'
            )
          )
        )
      )
    ));
    const auditedRequestIds = new Set(auditEvents.map((event) => event.request_id).filter(Boolean));
    const acceptedModels = new Set(args.backendAliases.map((value) => value.toLowerCase()));
    const responseAuditEvents = attemptAuditEvents.filter((event) => (
      event.event === 'upstream_response_received'
      && auditedRequestIds.has(event.request_id)
      && event.ok === true
      && event.has_assistant_message === true
      && String(event.response_id || '').trim()
      && acceptedModels.has(String(event.response_model || '').toLowerCase())
      && event.system_prompt_file_sha256 === systemPromptFileSha256
      && event.system_prompt_sha256 === systemPromptSha256
    ));
    const taskQueryEvents = Array.isArray(taskSummary?.query_events)
      ? taskSummary.query_events
      : [];
    const expectedQueryHashes = taskQueryEvents.map((event) => sha256(String(event.query || '')));
    const queryFrameQueueRequired = joyaiScaffold || nativeRealtime;
    const expectedQueryEventIds = new Set(taskQueryEvents.map((event) => (
      String(event.query_frame_queue?.query_event_id || `${task.id}:${event.id || ''}`)
    )));
    const upstreamQueryAuditEvents = auditEvents.filter((event) => (
      event.user_query_present === true
    ));
    const queryAcceptanceEventName = nativeRealtime
      ? 'query_frame_enqueued'
      : joyaiScaffold
      ? 'query_frame_queued'
      : '';
    const queryAuditEvents = queryFrameQueueRequired
      ? attemptAuditEvents.filter((event) => (
        event.event === queryAcceptanceEventName
        && event.user_query_present === true
        && expectedQueryEventIds.has(String(event.query_event_id || ''))
      ))
      : upstreamQueryAuditEvents;
    const actualQueryHashes = queryAuditEvents.map((event) => (
      String(event.user_query_sha256 || '')
    ));
    const successfulResponseRequestIds = new Set(
      responseAuditEvents.map((event) => event.request_id).filter(Boolean),
    );
    const queryRequestIds = new Set(
      upstreamQueryAuditEvents.map((event) => event.request_id).filter(Boolean),
    );
    const failedResponseAuditEvents = attemptAuditEvents.filter((event) => (
      event.event === 'upstream_response_received'
      && auditedRequestIds.has(event.request_id)
      && event.ok !== true
    ));
    const failedQueryResponseAuditEvents = failedResponseAuditEvents.filter((event) => (
      queryRequestIds.has(event.request_id)
    ));
    const modelLatencyTimeoutAuditEvents = failedResponseAuditEvents.filter((event) => (
      event.failure_class === 'model_latency_timeout'
      && event.deadline_exceeded === true
      && event.retryable_infrastructure === false
    ));
    const explicitProviderErrorAuditEvents = failedResponseAuditEvents.filter((event) => (
      event.failure_class !== 'model_latency_timeout'
    ));
    const queryProviderErrorAuditEvents = failedQueryResponseAuditEvents.filter((event) => (
      event.failure_class !== 'model_latency_timeout'
    ));
    const modelResponseOutcome = taskSummary?.model_response_outcome || null;
    const timedOutQueryIds = new Set(
      Array.isArray(modelResponseOutcome?.timed_out_query_ids)
        ? modelResponseOutcome.timed_out_query_ids.map((value) => String(value))
        : [],
    );
    const timedOutQueryHashes = taskQueryEvents
      .filter((event) => timedOutQueryIds.has(String(event.id || '')))
      .map((event) => sha256(String(event.query || '')));
    const providerSessionEvents = nativeRealtime
      ? attemptAuditEvents.filter((event) => (
        event.event === 'provider_session_created'
        && event.realtime_protocol === args.realtimeProtocol
        && event.input_transport === args.inputTransport
        && event.native_video_schema === args.nativeVideoSchema
        && event.system_prompt_sha256 === systemPromptSha256
        && String(event.provider_session_id || '').trim()
      ))
      : [];
    const nativeProviderQueryEventMatches = (event) => (
      event.realtime_protocol === args.realtimeProtocol
      && event.input_transport === args.inputTransport
      && event.native_video_schema === args.nativeVideoSchema
      && event.system_prompt_sha256 === systemPromptSha256
      && event.user_query_present === true
    );
    const deliveredProviderQueryEvents = nativeRealtime
      ? attemptAuditEvents.filter((event) => (
        event.event === 'provider_query_delivered'
        && nativeProviderQueryEventMatches(event)
      ))
      : [];
    const providerQueryEvents = nativeRealtime
      ? deliveredProviderQueryEvents
      : upstreamQueryAuditEvents;
    const providerQueryHashes = providerQueryEvents
      .map((event) => String(event.user_query_sha256 || ''));
    const providerQuerySubmissionEvents = nativeRealtime
      ? attemptAuditEvents.filter((event) => (
        event.event === 'provider_query_submission_started'
        && event.realtime_protocol === args.realtimeProtocol
        && event.input_transport === args.inputTransport
        && event.native_video_schema === args.nativeVideoSchema
        && event.user_query_present === true
        && String(event.request_id || '').trim()
      ))
      : joyaiScaffold
      ? attemptAuditEvents.filter((event) => (
        event.event === 'joyai_scaffold_upstream_request'
        && event.user_query_present === true
        && String(event.request_id || '').trim()
      ))
      : upstreamQueryAuditEvents;
    const hashCounts = (values) => values.reduce((counts, value) => {
      counts.set(value, (counts.get(value) || 0) + 1);
      return counts;
    }, new Map());
    const expectedQueryHashCounts = hashCounts(expectedQueryHashes);
    const actualQueryHashCounts = hashCounts(actualQueryHashes);
    const providerQueryHashCounts = hashCounts(providerQueryHashes);
    const timedOutQueryHashCounts = hashCounts(timedOutQueryHashes);
    const remainingExpectedProviderCounts = new Map(expectedQueryHashCounts);
    const unmatchedProviderQueryHashes = providerQueryHashes.filter((hash) => {
      const remaining = Number(remainingExpectedProviderCounts.get(hash) || 0);
      if (remaining <= 0) return true;
      remainingExpectedProviderCounts.set(hash, remaining - 1);
      return false;
    });
    const remainingProviderCounts = new Map(providerQueryHashCounts);
    const queuedButNotSubmittedQueryHashes = expectedQueryHashes.filter((hash) => {
      const remaining = Number(remainingProviderCounts.get(hash) || 0);
      if (remaining <= 0) return true;
      remainingProviderCounts.set(hash, remaining - 1);
      return false;
    });
    const unresolvedTimeoutCounts = new Map(timedOutQueryHashCounts);
    const queuedBackpressureIsCapabilityTimeout = queuedButNotSubmittedQueryHashes.every((hash) => {
      const remaining = Number(unresolvedTimeoutCounts.get(hash) || 0);
      if (remaining <= 0) return false;
      unresolvedTimeoutCounts.set(hash, remaining - 1);
      return true;
    });
    const queryAuditMatches = expectedQueryHashes.length === actualQueryHashes.length
      && [...expectedQueryHashCounts].every(([hash, count]) => actualQueryHashCounts.get(hash) === count)
      && unmatchedProviderQueryHashes.length === 0
      && auditEvents.every((event) => (
        event.query_delivery === 'once_per_round'
        && event.input_transport === args.inputTransport
        && event.persistent_query_injected !== true
      ));
    const queryFrameQueueMatches = !queryFrameQueueRequired || (
      queryAuditEvents.length === expectedQueryHashes.length
      && queryAuditEvents.every((event) => (
        event.query_frame_queue_policy === 'fifo-query-time-frame'
        && String(event.query_event_id || '').trim()
        && String(event.frame_time_range || '').trim()
      ))
      && taskQueryEvents.every((event) => (
          event.query_frame_queue?.policy === 'fifo-query-time-frame'
          && event.query_frame_queue?.acknowledged === true
          && (
            event.query_frame_queue?.gate_released === true
            || event.query_frame_queue?.gate_disarmed_without_frame === true
          )
          && event.query_frame_queue?.capture_surface
            === 'displayed_video_frame_at_query_dispatch'
        ),
      )
    );
    const queryRequestsWithoutSuccessfulResponse = providerQueryEvents.filter((event) => (
      !event.request_id || !successfulResponseRequestIds.has(event.request_id)
    ));
    const modelLatencyTimeoutRequestIds = new Set(modelLatencyTimeoutAuditEvents
      .map((event) => String(event.request_id || ''))
      .filter(Boolean));
    const deliveredCapabilityTimeoutMatches = queryRequestsWithoutSuccessfulResponse.every(
      (event) => (
        modelLatencyTimeoutRequestIds.has(String(event.request_id || ''))
        && timedOutQueryHashCounts.has(String(event.user_query_sha256 || ''))
      ),
    );
    const queryResponseAuditMatches = providerQueryEvents.every((event) => (
      Boolean(event.request_id) && successfulResponseRequestIds.has(event.request_id)
    ));
    const capabilityTimeoutAccepted = Boolean(
      modelResponseOutcome?.policy === MODEL_RESPONSE_TIMEOUT_POLICY
      && modelResponseOutcome?.response_semantics === MODEL_RESPONSE_OUTCOME_PROTOCOL
      && Number(modelResponseOutcome?.timeout_s) === MODEL_RESPONSE_TIMEOUT_S
      && modelResponseOutcome?.timeout_is_recording_failure === false
      && ['timeout', 'partial_timeout'].includes(modelResponseOutcome?.status)
      && timedOutQueryIds.size > 0
      && [...timedOutQueryIds].every((queryId) => (
        taskQueryEvents.some((event) => String(event.id || '') === queryId)
      ))
      && queuedBackpressureIsCapabilityTimeout
      && deliveredCapabilityTimeoutMatches
      && queryProviderErrorAuditEvents.length === 0
    );
    const providerQueryAuditMatches = (
      unmatchedProviderQueryHashes.length === 0
      && queuedButNotSubmittedQueryHashes.length === 0
    );
    const providerQueryResolutionMatches = (
      unmatchedProviderQueryHashes.length === 0
      && (
        queuedButNotSubmittedQueryHashes.length === 0
        || capabilityTimeoutAccepted
      )
    );
    const providerQueryTiming = buildProviderQueryTiming({
      taskSummary,
      acceptanceEvents: queryAuditEvents,
      providerQueryEvents,
      submissionEvents: providerQuerySubmissionEvents,
      maxAbsoluteDriftS: args.maxProviderQueryStartDriftS,
    });
    const videoPath = taskSummary?.files?.task_mp4 || taskResult?.task_mp4 || '';
    let videoProbe = null;
    let validationError = '';
    if (!taskSummary) {
      validationError = taskResult?.error || 'capture produced no retained task summary';
    } else if (taskSummary.status !== 'ok') {
      validationError = taskSummary.error || `task summary status is ${taskSummary.status}`;
    } else if (taskResult?.status && taskResult.status !== 'ok') {
      validationError = taskResult.error || `capture task result status is ${taskResult.status}`;
    } else if (captureExit.code !== 0) {
      validationError = taskResult?.error || `capture exited with code ${captureExit.code}`;
    }
    else if (taskSummary?.task?.id !== task.id) validationError = 'task identity mismatch in summary';
    else if (taskSummary?.vlm_backend_identity?.ok !== true) validationError = 'model backend identity was not verified';
    else if (taskSummary?.vlm_inference_warmup?.ok !== true) {
      validationError = 'model API readiness inference did not complete before target-video upload';
    }
    else if (taskSummary?.recording_start_barrier?.protocol !== CURRENT_RECORDING_PROTOCOL) {
      validationError = 'single-pass recording-start protocol attestation is missing';
    }
    else if (taskSummary?.recording_start_barrier?.ok !== true) {
      validationError = 'model-ready/upload-ready/playback-start ordering was not satisfied';
    }
    else if (
      taskSummary?.recording_start_barrier?.complete_file_upload?.ok !== true
      || taskSummary.recording_start_barrier.complete_file_upload.complete_file_uploaded !== true
      || taskSummary.recording_start_barrier.complete_file_upload.local_size_bytes
        !== taskSummary.recording_start_barrier.complete_file_upload.remote_size_bytes
    ) {
      validationError = 'The WebUI did not attest a complete byte-for-byte MP4 upload';
    }
    else if (
      taskSummary?.recording_start_barrier?.model_ready_strategy !== 'webui_same_session'
      || taskSummary?.recording_start_barrier?.webui_same_session_model_ready !== true
      || !taskSummary?.recording_start_barrier?.webui_warmup_session_id
      || taskSummary.recording_start_barrier.webui_warmup_session_id
        !== taskSummary.recording_start_barrier.target_session_id
    ) {
      validationError = 'Same-session end-to-end model readiness was not verified';
    }
    else if (
      (joyaiScaffold || nativeRealtime)
      && (
        taskSummary?.recording_start_barrier?.provider_warmup_required !== true
        || taskSummary?.recording_start_barrier?.provider_warmup?.ok !== true
        || taskSummary.recording_start_barrier.provider_warmup.multimodal_input !== true
        || taskSummary.recording_start_barrier.provider_warmup.session_state_unchanged !== true
      )
    ) {
      validationError = 'real isolated multimodal provider warmup was not verified before target upload';
    }
    else if (
      nativeRealtime
      && (
        taskSummary?.recording_start_barrier?.native_warmup_state_cleared !== true
        || taskSummary?.recording_start_barrier?.warmup_session_reset?.ok !== true
        || taskSummary.recording_start_barrier.warmup_session_reset.reset_acknowledged !== true
      )
    ) {
      validationError = 'native provider warmup state was not reset before target-video upload';
    }
    else if (
      taskSummary?.recording_start_barrier?.seek_performed_after_stream_start !== false
      || taskSummary?.recording_start_barrier?.target_video_started_once !== true
    ) {
      validationError = 'target video was not recorded as one non-seekable playback pass';
    }
    else if (
      taskSummary?.process_interval_s !== 1
      || taskSummary?.frames_per_batch !== 1
    ) {
      validationError = 'frame scheduler was not the required serial 1 FPS / single-frame protocol';
    }
    else if (
      taskSummary?.recording_start_barrier?.first_query_frame_gate?.required === true
      && (
        taskSummary.recording_start_barrier.first_query_frame_gate.installed !== true
        || taskSummary.recording_start_barrier.first_query_frame_gate.released !== true
      )
    ) {
      validationError = 'the first Query was not aligned with the first target-frame inference';
    }
    else if (taskSummary?.query_delivery_attestation?.ok !== true) {
      validationError = 'capture did not attest once-per-round Query delivery';
    }
    else if (
      taskSummary?.model_response_outcome?.response_semantics
        !== MODEL_RESPONSE_OUTCOME_PROTOCOL
    ) {
      validationError = 'capture used obsolete model-response completion semantics';
    }
    else if (!auditEvents.length) {
      validationError = nativeRealtime
        ? 'no provider-native realtime video forwarding audit event was recorded'
        : joyaiScaffold
        ? 'no official JoyAI scaffold ingress audit event was recorded'
        : 'no exact JoyAI system-prompt injection audit event was recorded';
    }
    else if (!queryAuditMatches) {
      validationError = 'adapter audit did not prove exactly one Query queue acceptance per annotated round';
    }
    else if (!queryFrameQueueMatches) {
      validationError = 'adapter audit did not bind every Query to its Query-time frame through FIFO scheduling';
    }
    else if (nativeRealtime && !providerSessionEvents.length) {
      validationError = 'no provider-native realtime session creation was attested';
    }
    else if (!providerQueryResolutionMatches) {
      validationError = 'provider audit found a Query that was neither submitted once nor retained as a capability timeout';
    }
    else if (explicitProviderErrorAuditEvents.length) {
      const classes = [...new Set(
        explicitProviderErrorAuditEvents.map((event) => event.failure_class || 'unclassified'),
      )].join(',');
      validationError = `upstream inference returned an explicit provider error (${classes})`;
    }
    else if (!queryResponseAuditMatches && !capabilityTimeoutAccepted) {
      validationError = 'the unique upstream Query request did not return a successful target-model response';
    }
    else if (providerQueryTiming?.ok !== true) {
      validationError = 'provider Query scheduling audit did not preserve the annotated WebUI Query origin';
    }
    else if (!responseAuditEvents.length && !capabilityTimeoutAccepted) {
      validationError = 'no prompt-correlated target-model response identifier was recorded';
    }
    else if (!videoPath || !fs.existsSync(videoPath)) validationError = 'task MP4 is missing';
    else {
      try {
        videoProbe = await probeVideo(videoPath);
        if (!(videoProbe.duration_s > 0 && videoProbe.size_bytes > 0 && videoProbe.width > 0 && videoProbe.height > 0)) {
          validationError = 'ffprobe returned invalid task MP4 metadata';
        }
      } catch (error) {
        validationError = `task MP4 probe failed: ${error.message}`;
      }
    }
    if (winner.type === 'route_failure') validationError = winner.reason;
    if (winner.type === 'task_wall_timeout') validationError = winner.reason;
    if (validationError) {
      await writeJsonAtomic(path.join(args.out, 'last_failed_attempt.json'), {
        timestamp: timestamp(),
        task_id: task.id,
        task_index: index + 1,
        attempt,
        error: replaceAllSecrets(validationError, secrets),
        capture_exit_code: captureExit.code,
        task_status: taskSummary?.status || '',
        actual_query_video_time_s: taskSummary?.actual_query_video_time_s ?? null,
        query_events: taskSummary?.query_events || [],
        provider_query_timing: providerQueryTiming,
        vlm_response_count: taskSummary?.vlm_response_count ?? 0,
        vlm_backend_identity: taskSummary?.vlm_backend_identity || null,
        video_quality: taskSummary?.video_quality || null,
        local_video_ready_status: taskSummary?.local_video_upload?.ready_status || null,
      });
      if (args.retainLastFailedArtifacts) {
        await removePathWithRetries(lastFailedArtifactsDir, { ignoreFailure: true });
        await relocateDirectory(taskWorkDir, lastFailedArtifactsDir);
        await logEvent('last_failed_artifacts_retained', {
          task_id: task.id,
          attempt,
          path: lastFailedArtifactsDir,
        });
      } else {
        const cleaned = await removePathWithRetries(taskWorkDir, { ignoreFailure: true });
        if (!cleaned) {
          await logEvent('workdir_cleanup_deferred', { task_id: task.id, attempt, path: taskWorkDir });
        }
      }
      return {
        ok: false,
        error: replaceAllSecrets(validationError, secrets),
        routeFailure: winner.type === 'route_failure',
        taskWallTimeout: winner.type === 'task_wall_timeout',
      };
    }

    const destination = canonicalDir(task, index);
    await removePathWithRetries(destination);
    const sourceVideoPath = String(taskSummary.files?.task_mp4 || '');
    const relativeVideoPath = path.relative(sourceTaskDir, sourceVideoPath);
    const removedArtifacts = [];
    for (const key of ['ui_webm', 'ui_mp4', 'events_jsonl']) {
      const candidate = String(taskSummary.files?.[key] || '');
      const relative = candidate ? path.relative(sourceTaskDir, candidate) : '';
      if (
        candidate
        && path.resolve(candidate) !== path.resolve(sourceVideoPath)
        && relative
        && !relative.startsWith('..')
        && !path.isAbsolute(relative)
      ) {
        await fsp.unlink(candidate).catch(() => {});
        removedArtifacts.push(path.basename(candidate));
      }
    }
    let artifactPromotion;
    try {
      artifactPromotion = await promoteTaskDirectory({
        sourceDir: sourceTaskDir,
        destinationDir: destination,
        relativeVideoPath,
        validateVideo: (candidate) => validateVideoDecode({
          videoPath: candidate,
          decodeAttempts: 1,
          retryDelayMs: 0,
        }),
      });
    } catch (error) {
      const promotionError = `Task artifact promotion failed: ${error.message}`;
      await removePathWithRetries(destination, { ignoreFailure: true });
      await removePathWithRetries(taskWorkDir, { ignoreFailure: true });
      await writeJsonAtomic(path.join(args.out, 'last_failed_attempt.json'), {
        timestamp: timestamp(),
        task_id: task.id,
        task_index: index + 1,
        attempt,
        error: promotionError,
        capture_exit_code: captureExit.code,
        task_status: taskSummary?.status || '',
      });
      return {
        ok: false,
        error: promotionError,
        routeFailure: false,
        taskWallTimeout: false,
      };
    }
    const rewrittenSummary = rewritePaths(taskSummary, sourceTaskDir, destination);
    rewrittenSummary.task_index = index + 1;
    rewrittenSummary.task_total = tasks.length;
    const finalVideoPath = rewrittenSummary.files.task_mp4;
    for (const key of ['ui_webm', 'ui_mp4', 'events_jsonl']) {
      if (rewrittenSummary.files) delete rewrittenSummary.files[key];
    }
    if (artifactPromotion.validation?.probe) videoProbe = artifactPromotion.validation.probe;
    rewrittenSummary.retention = {
      policy: 'successful_task_mp4_and_compact_metadata_only',
      removed_artifacts: removedArtifacts,
      retained_artifacts: [path.basename(finalVideoPath), 'summary.json', 'capture_validation.json'],
      artifact_promotion: {
        method: artifactPromotion.method,
        copied_video_sha256: artifactPromotion.video_sha256,
        copied_video_decode_ok: artifactPromotion.validation?.decode_ok ?? null,
      },
    };
    await writeJsonAtomic(path.join(destination, 'summary.json'), rewrittenSummary);
    await writeJsonAtomic(path.join(destination, 'capture_validation.json'), {
      validated_at: timestamp(),
      task_id: task.id,
      task_index: index + 1,
      attempt,
      model_request_id: args.model,
      task_contract: task,
      source_video_snapshot: sourceVideoSnapshot(task),
      evaluation_contract: evaluationContract,
      observed_models: rewrittenSummary.vlm_backend_identity.observed_models || [],
      model_identity_ok: true,
      prompt_source_path: promptPath,
      system_prompt_file_sha256: systemPromptFileSha256,
      system_prompt_sha256: systemPromptSha256,
      prompt_audit_event_count: auditEvents.length,
      prompt_audit_first_request_id: auditEvents[0]?.request_id || '',
      prompt_audit_last_request_id: auditEvents[auditEvents.length - 1]?.request_id || '',
      query_delivery_attestation: rewrittenSummary.query_delivery_attestation,
      upstream_query_audit: {
        policy: 'once_per_round_no_replay',
        ok: queryAuditMatches
          && queryFrameQueueMatches
          && providerQueryResolutionMatches
          && (queryResponseAuditMatches || capabilityTimeoutAccepted),
        expected_rounds: expectedQueryHashes.length,
        queue_accepted_rounds: actualQueryHashes.length,
        upstream_submitted_rounds: providerQueryHashes.length,
        queued_capability_timeout_rounds: queuedButNotSubmittedQueryHashes.length,
        request_ids: providerQueryEvents.map((event) => event.request_id).filter(Boolean),
        successful_response_for_every_query: queryResponseAuditMatches,
        capability_timeout_accepted: capabilityTimeoutAccepted,
        timed_out_query_ids: [...timedOutQueryIds],
        explicit_provider_error_count: queryProviderErrorAuditEvents.length,
        model_latency_timeout_count: failedQueryResponseAuditEvents.filter((event) => (
          event.failure_class === 'model_latency_timeout'
        )).length,
        provider_query_timing: providerQueryTiming,
      },
      frame_stream_audit: nativeRealtime ? null : {
        ok: queryAuditMatches
          && queryFrameQueueMatches
          && providerQueryResolutionMatches
          && (queryResponseAuditMatches || capabilityTimeoutAccepted)
          && providerQueryTiming?.ok === true,
        evaluation_protocol: JOYAI_FRAME_STREAM_PROTOCOL,
        requests: auditEvents.length,
        all_requests_single_frame: auditEvents.every((event) => (
          event.image_count === 1 && event.video_input_count === 0
        )),
        scheduler: {
          max_in_flight: 1,
          busy_policy: 'skip',
          queue_capacity: 0,
          process_interval_s: rewrittenSummary.process_interval_s,
          frames_per_batch: rewrittenSummary.frames_per_batch,
        },
      },
      native_realtime_audit: nativeRealtime ? {
        ok: queryAuditMatches
          && queryFrameQueueMatches
          && providerQueryResolutionMatches
          && (queryResponseAuditMatches || capabilityTimeoutAccepted)
          && providerQueryTiming?.ok === true,
        realtime_protocol: args.realtimeProtocol,
        native_video_schema: args.nativeVideoSchema,
        query_transport: args.realtimeQueryMode,
        provider_session_count: providerSessionEvents.length,
        provider_session_ids: providerSessionEvents
          .map((event) => event.provider_session_id)
          .filter(Boolean),
        queue_accepted_query_count: actualQueryHashes.length,
        provider_query_hash_count: providerQueryHashes.length,
        queued_capability_timeout_count: queuedButNotSubmittedQueryHashes.length,
        every_query_reached_upstream: providerQueryAuditMatches,
        provider_query_timing: providerQueryTiming,
      } : null,
      target_response_attestation_count: responseAuditEvents.length,
      target_response_identifiers: [...new Set(responseAuditEvents.map((event) => event.response_id))],
      target_response_models: [...new Set(responseAuditEvents.map((event) => event.response_model))],
      target_response_first_request_id: responseAuditEvents[0]?.request_id || '',
      target_response_last_request_id: responseAuditEvents[responseAuditEvents.length - 1]?.request_id || '',
      model_response_outcome: modelResponseOutcome,
      model_response_outcome_protocol: MODEL_RESPONSE_OUTCOME_PROTOCOL,
      recording_start_barrier: rewrittenSummary.recording_start_barrier,
      artifact_promotion: rewrittenSummary.retention.artifact_promotion,
      video_path: finalVideoPath,
      video_probe: videoProbe,
    });
    const cleaned = await removePathWithRetries(taskWorkDir, { ignoreFailure: true });
    if (!cleaned) {
      await logEvent('workdir_cleanup_deferred', { task_id: task.id, attempt, path: taskWorkDir });
    }
    await fsp.unlink(path.join(args.out, 'last_failed_attempt.json')).catch(() => {});
    if (args.retainLastFailedArtifacts) {
      await removePathWithRetries(lastFailedArtifactsDir, { ignoreFailure: true });
    }
    return {
      ok: true,
      taskDir: destination,
      taskMp4: finalVideoPath,
      observedModels: rewrittenSummary.vlm_backend_identity.observed_models || [],
      auditEvents: auditEvents.length,
      responseAuditEvents: responseAuditEvents.length,
      responseIdentifiers: [...new Set(responseAuditEvents.map((event) => event.response_id))],
      modelResponseOutcome,
      videoProbe,
    };
  };

  const shutdown = async (status, error = '') => {
    stopping = true;
    if (error) state.last_error = replaceAllSecrets(error, secrets);
    else if (status === 'complete') state.last_error = '';
    state.status = status;
    state.current_task = null;
    state.finished_at = ['complete', 'incomplete', 'failed'].includes(status) ? timestamp() : null;
    await terminateChild(captureChild, { processGroup: true, graceMs: 10000 });
    captureChild = null;
    await stopTunnel();
    await terminateChild(proxyChild, { graceMs: 5000 });
    proxyChild = null;
    if (status === 'complete') {
      await removePathWithRetries(workDir, { ignoreFailure: true });
      await removePathWithRetries(runtimeDir, { ignoreFailure: true });
    }
    await persistState().catch(() => {});
    await writeProgress(`shutdown:${status}`).catch(() => {});
    try { fs.closeSync(lockFd); } catch {}
    await fsp.unlink(lockPath).catch(() => {});
  };

  const handleSignal = (signal) => {
    if (stopping) return;
    stopping = true;
    console.warn(`[supervisor] received ${signal}; stopping managed processes`);
    terminateChild(captureChild, { processGroup: true, graceMs: 5000 }).catch(() => {});
  };
  process.once('SIGINT', handleSignal);
  process.once('SIGTERM', handleSignal);

  const progressTimer = setInterval(() => {
    writeProgress('supervisor_timer').catch((error) => {
      console.error(`[progress] ${replaceAllSecrets(error.message, secrets)}`);
    });
  }, args.progressIntervalS * 1000);
  progressTimer.unref();

  try {
    reconcileCompleted();
    await persistState();
    await logEvent('supervisor_started', {
      pid: process.pid,
      tasks_total: tasks.length,
      already_complete: Object.keys(state.results).length,
      model: args.model,
      upstream_key_pool_size: upstreamKeys.length,
      active_upstream_key_slot: state.active_upstream_key_slot,
      system_prompt_sha256: systemPromptSha256,
    });
    for (const taskId of retryFailedTaskIds) {
      const continuation = continuedRecoveryTaskIds.has(taskId);
      const attemptsUsed = Number(state.attempts[taskId] || 0);
      const attemptLimit = Number(state.retry_attempt_limits[taskId] || attemptsUsed);
      await logEvent(continuation
        ? 'failed_task_backfill_resumed'
        : 'failed_task_backfill_scheduled', {
        task_id: taskId,
        attempts_already_used: attemptsUsed,
        additional_attempts_allowed: Math.max(0, attemptLimit - attemptsUsed),
        recovery_batch_count: state.recovery_batch_count,
        task_recovery_batch_index: recoveryBatchCountForTask(taskId),
      });
    }
    await writeProgress('supervisor_start');

    for (let index = 0; index < tasks.length; index += 1) {
      if (stopping) break;
      const task = tasks[index];
      if (readCanonical(task, index)) {
        delete state.task_wall_pass_started_at[task.id];
        await logEvent('task_skipped_existing', { task_id: task.id, task_index: index + 1 });
        continue;
      }
      const maxAttempts = Math.max(
        args.maxRetries + 1,
        Number(state.retry_attempt_limits[task.id] || 0),
      );
      const taskWallWindow = resolveTaskWallWindow({
        persistedStartedAt: state.task_wall_pass_started_at[task.id],
        maxTaskWallS: args.maxTaskWallS,
      });
      const taskWallStartedAtMs = taskWallWindow.startedAtMs;
      const taskWallDeadlineMs = taskWallWindow.deadlineMs;
      state.task_wall_pass_started_at[task.id] = taskWallWindow.startedAt;
      let succeeded = false;
      let taskWallExpired = false;
      for (let attempt = Number(state.attempts[task.id] || 0) + 1; attempt <= maxAttempts; attempt += 1) {
        if (stopping) break;
        state.current_task = {
          task_index: index + 1,
          task_id: task.id,
          attempt,
          max_attempts: maxAttempts,
          started_at: timestamp(),
          task_wall_started_at: new Date(taskWallStartedAtMs).toISOString(),
          task_wall_deadline_at: new Date(taskWallDeadlineMs).toISOString(),
          max_task_wall_s: args.maxTaskWallS,
        };
        state.last_error = '';
        await persistState();
        try {
          await ensureRoute(
            `before ${task.id} attempt ${attempt}`,
            task.id,
            taskWallDeadlineMs,
          );
        } catch (error) {
          if (error?.code !== 'TASK_WALL_TIMEOUT') throw error;
          state.last_error = error.message;
          state.attempts[task.id] = maxAttempts;
          taskWallExpired = true;
          await logEvent('task_wall_timeout', {
            task_id: task.id,
            task_index: index + 1,
            attempt,
            max_attempts: maxAttempts,
            max_task_wall_s: args.maxTaskWallS,
            phase: 'route_recovery',
            error: error.message,
          });
          await persistState();
          break;
        }
        if (stopping) break;
        // Route recovery and key failover are infrastructure work, not task attempts.
        state.attempts[task.id] = attempt;
        state.current_task.started_at = timestamp();
        await persistState();
        const result = await runCaptureAttempt(
          task,
          index,
          attempt,
          maxAttempts,
          taskWallDeadlineMs,
        );
        if (result.ok) {
          state.results[task.id] = {
            status: 'ok',
            task_index: index + 1,
            task_id: task.id,
            attempts: attempt,
            task_dir: result.taskDir,
            task_mp4: result.taskMp4,
            model_identity: result.observedModels,
            prompt_audit_event_count: result.auditEvents,
            target_response_attestation_count: result.responseAuditEvents,
            target_response_identifiers: result.responseIdentifiers,
            model_response_outcome: result.modelResponseOutcome,
            video_duration_s: result.videoProbe.duration_s,
            finished_at: timestamp(),
          };
          state.current_task = null;
          state.last_error = '';
          delete state.failures[task.id];
          delete state.retry_attempt_limits[task.id];
          delete state.task_wall_pass_started_at[task.id];
          await persistState();
          await logEvent('task_complete', {
            task_id: task.id,
            task_index: index + 1,
            attempt,
            video_duration_s: result.videoProbe.duration_s,
          });
          await writeProgress('task_complete');
          succeeded = true;
          break;
        }
        state.last_error = result.error;
        await logEvent('task_attempt_failed', {
          task_id: task.id,
          task_index: index + 1,
          attempt,
          max_attempts: maxAttempts,
          error: result.error,
        });
        await stopTunnel();
        if (result.taskWallTimeout) {
          state.attempts[task.id] = maxAttempts;
          taskWallExpired = true;
          await logEvent('task_wall_timeout', {
            task_id: task.id,
            task_index: index + 1,
            attempt,
            max_attempts: maxAttempts,
            max_task_wall_s: args.maxTaskWallS,
            phase: 'capture',
            error: result.error,
          });
        }
        await persistState();
        if (stopping) break;
        if (taskWallExpired) break;
        if (attempt < maxAttempts) await sleep(20000);
      }
      if (stopping) break;
      if (!succeeded) {
        const failureError = state.last_error || `${task.id} has no successful attempt`;
        const lastFailurePath = path.join(args.out, 'last_failed_attempt.json');
        let failureArtifact = '';
        if (fs.existsSync(lastFailurePath)) {
          await fsp.mkdir(failedAttemptsDir, { recursive: true });
          failureArtifact = path.join(
            failedAttemptsDir,
            `${String(index + 1).padStart(3, '0')}_${safeName(task.id)}_attempt_${Number(state.attempts[task.id] || 0)}.json`,
          );
          await fsp.copyFile(lastFailurePath, failureArtifact);
        }
        state.failures[task.id] = {
          status: 'failed',
          task_index: index + 1,
          task_id: task.id,
          attempts: Number(state.attempts[task.id] || 0),
          error: failureError,
          failed_at: timestamp(),
          failure_artifact: failureArtifact,
          task_dir: '',
          task_mp4: '',
        };
        delete state.retry_attempt_limits[task.id];
        delete state.task_wall_pass_started_at[task.id];
        state.current_task = null;
        await persistState();
        await logEvent('task_failed_exhausted', {
          task_id: task.id,
          task_index: index + 1,
          attempts: state.failures[task.id].attempts,
          error: failureError,
        });
        await writeProgress('task_failed_exhausted');
        if (!args.continueAfterTaskFailure) {
          throw new Error(`${task.id} failed after ${maxAttempts} attempts: ${failureError}`);
        }
      }
      if (index + 1 < tasks.length) await sleep(8000);
    }

    if (stopping) await shutdown('stopped', 'Supervisor was stopped by signal');
    else {
      reconcileCompleted();
      const completed = Object.values(state.results).filter((item) => item.status === 'ok').length;
      if (completed !== tasks.length) {
        const failedTaskIds = tasks
          .map((task) => task.id)
          .filter((taskId) => state.failures[taskId]?.status === 'failed');
        if (!args.continueAfterTaskFailure) {
          throw new Error(`Only ${completed}/${tasks.length} tasks are complete`);
        }
        await logEvent('task_pass_complete_with_failures', {
          completed,
          total: tasks.length,
          failed_task_ids: failedTaskIds,
        });
        await shutdown(
          'incomplete',
          `Completed ${completed}/${tasks.length}; failed tasks: ${failedTaskIds.join(', ')}`,
        );
      } else {
        await logEvent('all_tasks_complete', { completed, total: tasks.length });
        await shutdown('complete');
      }
    }
  } catch (error) {
    const message = replaceAllSecrets(error.stack || error.message || String(error), secrets);
    console.error(message);
    await logEvent('supervisor_failed', { error: message.split(/\r?\n/)[0] }).catch(() => {});
    await shutdown(stopping ? 'stopped' : 'failed', message);
    process.exitCode = 1;
  } finally {
    clearInterval(progressTimer);
    process.removeListener('SIGINT', handleSignal);
    process.removeListener('SIGTERM', handleSignal);
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
