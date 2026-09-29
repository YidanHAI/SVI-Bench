#!/usr/bin/env node
import { chromium } from 'playwright';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import net from 'node:net';
import http from 'node:http';
import https from 'node:https';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import WebSocket from 'ws';
import {
  calibrateWebmClockFromVisualMarker,
  calculateMaximumTaskVideoDurationS,
  renderTaskVideo,
  siblingFfprobe,
  validateTaskVideo,
  validateVideoDecode,
} from './video_quality.mjs';
import {
  DEFAULT_VLM_CONFIG,
  getVlmProfile,
  loadVlmRegistry,
  normalizeApiBase,
  profileFingerprint,
  resolveProfileApiKey,
  safeProfileSnapshot,
} from './vlm_profiles.mjs';
import {
  loadRecordingCampaignConfig,
  normalizeRecordingWebUrl,
} from './recording_config.mjs';
import {
  classifyVlmResponseText,
  isDeferredVlmText,
  isQueryEchoVlmText,
  isSubstantiveVlmText,
  parseFrameTimeRangeS,
  responseBelongsToQuery,
} from './vlm_response_protocol.mjs';

const RECORDING_CAMPAIGN = loadRecordingCampaignConfig();
const DEFAULT_WEB_URL = RECORDING_CAMPAIGN.webui.url;
const DEFAULT_WEB_TLS_REJECT_UNAUTHORIZED = RECORDING_CAMPAIGN.webui.tlsRejectUnauthorized;
const CURRENT_RECORDING_PROTOCOL = 'webui_session_ready_then_single_pass_upload_v2';
const MODEL_RESPONSE_TIMEOUT_POLICY = 'wait_then_keep_as_capability_result';
const MODEL_RESPONSE_OUTCOME_PROTOCOL = 'strict_protocol_non_deferred_non_echo_response_v4';
const JOYAI_OFFICIAL_SCAFFOLD = 'joyai-official-live-adapter';
const RECORDING_FINALIZATION_GUARD_S = 0.5;
const RECORDING_CLOCK_MARKER = Object.freeze({
  id: '__capture_recording_clock_marker',
  size: 48,
  x: 0,
  color: 'rgb(0, 255, 0)',
});
const sourceVideoValidationCache = new Map();

function parseArgs(argv) {
  const args = {
    tasks: '',
    webUrl: DEFAULT_WEB_URL,
    webUsername: process.env.JOYVL_WEB_USERNAME || '',
    webPassword: process.env.JOYVL_WEB_PASSWORD || '',
    vlmProfile: '',
    vlmConfig: DEFAULT_VLM_CONFIG,
    vlmModel: process.env.JOYVL_VLM_MODEL || '',
    vlmApiBase: process.env.JOYVL_VLM_API_BASE || '',
    vlmControlApiBase: process.env.JOYVL_VLM_CONTROL_API_BASE || '',
    vlmApiKey: process.env.JOYVL_VLM_API_KEY || '',
    vlmRoute: process.env.JOYVL_VLM_ROUTE || 'auto',
    vlmFormalEval: null,
    vlmBackendAliases: [],
    vlmPreflight: true,
    vlmIdentityCheck: true,
    vlmWarmup: true,
    vlmWarmupTimeoutS: 180,
    vlmWarmupRetries: 2,
    vlmProfileFingerprint: '',
    vlmProfileSnapshot: null,
    vlmPreflightResult: null,
    out: 'outputs/run',
    defaultDurationS: 45,
    processIntervalS: 1,
    framesPerBatch: 1,
    width: 1600,
    height: 1000,
    headless: true,
    mp4: true,
    fullUiMp4: true,
    keepFailedAttempts: false,
    freshBrowserPerAttempt: false,
    skipExisting: false,
    skipTaskIds: [],
    sourceDecodeCheck: true,
    limit: 0,
    perTaskTimeoutS: 0,
    taskGapS: 0,
    taskRetries: 0,
    retryGapS: 20,
    healthRetries: 1,
    healthIntervalS: 5,
    streamReadyTimeoutS: 60,
    videoUploadTimeoutS: 900,
    streamStartRetries: 0,
    streamStartRetryGapS: 3,
    streamReconnects: 0,
    streamReconnectTimeoutS: 60,
    uploadControlReconnects: 5,
    uploadControlReconnectTimeoutS: 10,
    attemptTimeoutS: 0,
    attemptTimeoutMarginS: 240,
    localVideoMode: 'upload',
    localRelayPublishProtocol: 'rtsp',
    localRtspRelayHost: '127.0.0.1',
    localRtspRelayPort: 8554,
    localRtmpRelayPort: 1935,
    localRtspPlaybackHost: '',
    localRtspPlaybackPort: 8554,
    localRtspRelayPrefix: 'vl_local',
    localRtspWarmupMs: 800,
    localRtspEncode: false,
    localRtspFps: 12,
    localRtspGopS: 1,
    localRtspScaleLongEdge: 0,
    localRtspCrf: 23,
    localRtspMaxrate: '',
    localRtspBufsize: '',
    localRtspPrerollS: 0,
    localRtspPostrollS: 30,
    localRtspConnectLeadS: 0,
    localRtspReconnectWarmupMs: 3000,
    videoQualityCheck: true,
    videoDurationToleranceS: 1.5,
    videoQueryTimeToleranceS: 0.5,
    videoMaxBlackS: 2,
    postVideoResponseTimeoutS: 180,
    postResponseRecordingS: 1,
    videoMaxExtraS: 5,
    ffmpegBin: 'ffmpeg',
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`Missing value for ${arg}`);
      i += 1;
      return argv[i];
    };

    if (arg === '--tasks') args.tasks = next();
    else if (arg === '--web-url') args.webUrl = next();
    else if (arg === '--web-username') args.webUsername = next();
    else if (arg === '--web-password') args.webPassword = next();
    else if (arg === '--vlm-profile') args.vlmProfile = next();
    else if (arg === '--vlm-config') args.vlmConfig = next();
    else if (arg === '--vlm-model') args.vlmModel = next();
    else if (arg === '--vlm-api-base') args.vlmApiBase = next();
    else if (arg === '--vlm-control-api-base') args.vlmControlApiBase = next();
    else if (arg === '--vlm-api-key') args.vlmApiKey = next();
    else if (arg === '--vlm-route') args.vlmRoute = next();
    else if (arg === '--vlm-backend-alias') args.vlmBackendAliases.push(next());
    else if (arg === '--no-vlm-preflight') args.vlmPreflight = false;
    else if (arg === '--no-vlm-identity-check') args.vlmIdentityCheck = false;
    else if (arg === '--no-vlm-warmup') args.vlmWarmup = false;
    else if (arg === '--vlm-warmup-timeout-s') args.vlmWarmupTimeoutS = Number(next());
    else if (arg === '--vlm-warmup-retries') args.vlmWarmupRetries = Number(next());
    else if (arg === '--out') args.out = next();
    else if (arg === '--default-duration-s') args.defaultDurationS = Number(next());
    else if (arg === '--process-interval-s') args.processIntervalS = Number(next());
    else if (arg === '--frames-per-batch') args.framesPerBatch = Number(next());
    else if (arg === '--width') args.width = Number(next());
    else if (arg === '--height') args.height = Number(next());
    else if (arg === '--limit') args.limit = Number(next());
    else if (arg === '--per-task-timeout-s') args.perTaskTimeoutS = Number(next());
    else if (arg === '--task-gap-s') args.taskGapS = Number(next());
    else if (arg === '--task-retries') args.taskRetries = Number(next());
    else if (arg === '--retry-gap-s') args.retryGapS = Number(next());
    else if (arg === '--health-retries') args.healthRetries = Number(next());
    else if (arg === '--health-interval-s') args.healthIntervalS = Number(next());
    else if (arg === '--stream-ready-timeout-s') args.streamReadyTimeoutS = Number(next());
    else if (arg === '--video-upload-timeout-s') args.videoUploadTimeoutS = Number(next());
    else if (arg === '--stream-start-retries') args.streamStartRetries = Number(next());
    else if (arg === '--stream-start-retry-gap-s') args.streamStartRetryGapS = Number(next());
    else if (arg === '--stream-reconnects') args.streamReconnects = Number(next());
    else if (arg === '--stream-reconnect-timeout-s') args.streamReconnectTimeoutS = Number(next());
    else if (arg === '--upload-control-reconnects') args.uploadControlReconnects = Number(next());
    else if (arg === '--upload-control-reconnect-timeout-s') args.uploadControlReconnectTimeoutS = Number(next());
    else if (arg === '--attempt-timeout-s') args.attemptTimeoutS = Number(next());
    else if (arg === '--attempt-timeout-margin-s') args.attemptTimeoutMarginS = Number(next());
    else if (arg === '--local-video-mode') args.localVideoMode = next();
    else if (arg === '--local-relay-publish-protocol') args.localRelayPublishProtocol = next().toLowerCase();
    else if (arg === '--local-rtsp-relay-host') args.localRtspRelayHost = next();
    else if (arg === '--local-rtsp-relay-port') args.localRtspRelayPort = Number(next());
    else if (arg === '--local-rtmp-relay-port') args.localRtmpRelayPort = Number(next());
    else if (arg === '--local-rtsp-playback-host') args.localRtspPlaybackHost = next();
    else if (arg === '--local-rtsp-playback-port') args.localRtspPlaybackPort = Number(next());
    else if (arg === '--local-rtsp-relay-prefix') args.localRtspRelayPrefix = next();
    else if (arg === '--local-rtsp-warmup-ms') args.localRtspWarmupMs = Number(next());
    else if (arg === '--local-rtsp-encode') args.localRtspEncode = true;
    else if (arg === '--local-rtsp-fps') args.localRtspFps = Number(next());
    else if (arg === '--local-rtsp-gop-s') args.localRtspGopS = Number(next());
    else if (arg === '--local-rtsp-scale-long-edge') args.localRtspScaleLongEdge = Number(next());
    else if (arg === '--local-rtsp-crf') args.localRtspCrf = Number(next());
    else if (arg === '--local-rtsp-maxrate') args.localRtspMaxrate = next();
    else if (arg === '--local-rtsp-bufsize') args.localRtspBufsize = next();
    else if (arg === '--local-rtsp-preroll-s') args.localRtspPrerollS = Number(next());
    else if (arg === '--local-rtsp-postroll-s') args.localRtspPostrollS = Number(next());
    else if (arg === '--local-rtsp-connect-lead-s') args.localRtspConnectLeadS = Number(next());
    else if (arg === '--local-rtsp-reconnect-warmup-ms') args.localRtspReconnectWarmupMs = Number(next());
    else if (arg === '--no-video-quality-check') args.videoQualityCheck = false;
    else if (arg === '--video-duration-tolerance-s') args.videoDurationToleranceS = Number(next());
    else if (arg === '--video-query-time-tolerance-s') args.videoQueryTimeToleranceS = Number(next());
    else if (arg === '--video-max-black-s') args.videoMaxBlackS = Number(next());
    else if (arg === '--post-video-response-timeout-s') args.postVideoResponseTimeoutS = Number(next());
    else if (arg === '--post-response-recording-s') args.postResponseRecordingS = Number(next());
    else if (arg === '--video-max-extra-s') args.videoMaxExtraS = Number(next());
    else if (arg === '--ffmpeg-bin') args.ffmpegBin = next();
    else if (arg === '--headed') args.headless = false;
    else if (arg === '--headless') args.headless = true;
    else if (arg === '--skip-existing') args.skipExisting = true;
    else if (arg === '--skip-task-id') args.skipTaskIds.push(next());
    else if (arg === '--no-source-decode-check') args.sourceDecodeCheck = false;
    else if (arg === '--no-mp4') args.mp4 = false;
    else if (arg === '--no-full-ui-mp4') args.fullUiMp4 = false;
    else if (arg === '--keep-failed-attempts') args.keepFailedAttempts = true;
    else if (arg === '--fresh-browser-per-attempt') args.freshBrowserPerAttempt = true;
    else if (arg === '-h' || arg === '--help') {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!args.tasks) throw new Error('Missing --tasks');
  if (args.vlmProfile) {
    const registry = loadVlmRegistry(args.vlmConfig);
    const profile = getVlmProfile(registry, args.vlmProfile);
    args.vlmConfig = registry.path;
    args.vlmModel = profile.model;
    args.vlmApiBase = profile.api_base;
    args.vlmApiKey = resolveProfileApiKey(profile, process.env, args.vlmApiKey);
    args.vlmRoute = profile.route;
    args.vlmFormalEval = profile.formal_eval;
    args.vlmBackendAliases = profile.backend_aliases;
    args.vlmPreflight = args.vlmPreflight && profile.preflight;
    args.vlmIdentityCheck = args.vlmIdentityCheck && profile.identity_check;
    args.vlmProfileFingerprint = profileFingerprint(profile);
    args.vlmProfileSnapshot = safeProfileSnapshot(profile);
    if (profile.input_transport === 'native-video-batch') {
      throw new Error(
        `VLM profile ${profile.id} requires ${profile.input_transport} input `
        + `(${profile.native_video_schema}); capture.mjs receives JPEG analyzer frames and must not `
        + 'silently downgrade a video-capable model',
      );
    }
    if (
      profile.input_transport === 'native-video-realtime'
      && (!profile.realtime_protocol || !profile.realtime_api_base || !profile.realtime_query_mode)
    ) {
      throw new Error(
        `VLM profile ${profile.id} does not fully specify its native-video-realtime adapter`,
      );
    }
    if (profile.api_key_env && !args.vlmApiKey) {
      throw new Error(
        `VLM profile ${profile.id} requires API key environment variable ${profile.api_key_env}`,
      );
    }
  }
  if (Boolean(args.vlmModel) !== Boolean(args.vlmApiBase)) {
    throw new Error('--vlm-model and --vlm-api-base must be provided together');
  }
  if (!['auto', 'joyai_adapter', 'direct'].includes(args.vlmRoute)) {
    throw new Error('--vlm-route must be auto, joyai_adapter, or direct');
  }
  if (args.vlmModel) {
    args.vlmApiBase = normalizeApiBase(args.vlmApiBase);
    args.vlmControlApiBase = normalizeApiBase(args.vlmControlApiBase || args.vlmApiBase);
    for (const [field, value] of [
      ['--vlm-api-base', args.vlmApiBase],
      ['--vlm-control-api-base', args.vlmControlApiBase],
    ]) {
      const parsed = new URL(value);
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        throw new Error(`${field} must use HTTP or HTTPS`);
      }
      if (parsed.username || parsed.password) {
        throw new Error(`${field} must not embed credentials`);
      }
    }
    if (args.vlmRoute === 'auto') {
      const url = new URL(args.vlmApiBase);
      args.vlmRoute = ['127.0.0.1', 'localhost'].includes(url.hostname) && url.port === '8070'
        ? 'joyai_adapter'
        : 'direct';
    }
    if (args.vlmFormalEval == null) args.vlmFormalEval = args.vlmRoute === 'joyai_adapter';
    args.vlmBackendAliases = [...new Set([
      args.vlmModel,
      ...args.vlmBackendAliases.flatMap((value) => String(value).split(',')),
    ].map((value) => value.trim()).filter(Boolean))];
  }
  if (!Number.isFinite(args.defaultDurationS) || args.defaultDurationS <= 0) {
    throw new Error('--default-duration-s must be positive');
  }
  if (!Number.isFinite(args.taskGapS) || args.taskGapS < 0) {
    throw new Error('--task-gap-s must be non-negative');
  }
  if (!Number.isInteger(args.taskRetries) || args.taskRetries < 0 || args.taskRetries > 5) {
    throw new Error('--task-retries must be a non-negative integer no greater than 5');
  }
  if (!Number.isFinite(args.retryGapS) || args.retryGapS < 0) {
    throw new Error('--retry-gap-s must be non-negative');
  }
  if (!Number.isInteger(args.healthRetries) || args.healthRetries < 1) {
    throw new Error('--health-retries must be a positive integer');
  }
  if (!Number.isFinite(args.healthIntervalS) || args.healthIntervalS < 0) {
    throw new Error('--health-interval-s must be non-negative');
  }
  if (!Number.isFinite(args.vlmWarmupTimeoutS) || args.vlmWarmupTimeoutS <= 0) {
    throw new Error('--vlm-warmup-timeout-s must be positive');
  }
  if (!Number.isInteger(args.vlmWarmupRetries) || args.vlmWarmupRetries < 0 || args.vlmWarmupRetries > 5) {
    throw new Error('--vlm-warmup-retries must be a non-negative integer no greater than 5');
  }
  if (!Number.isFinite(args.streamReadyTimeoutS) || args.streamReadyTimeoutS <= 0) {
    throw new Error('--stream-ready-timeout-s must be positive');
  }
  if (!Number.isFinite(args.videoUploadTimeoutS) || args.videoUploadTimeoutS <= 0) {
    throw new Error('--video-upload-timeout-s must be positive');
  }
  if (!Number.isInteger(args.streamStartRetries) || args.streamStartRetries < 0) {
    throw new Error('--stream-start-retries must be a non-negative integer');
  }
  if (!Number.isFinite(args.streamStartRetryGapS) || args.streamStartRetryGapS < 0) {
    throw new Error('--stream-start-retry-gap-s must be non-negative');
  }
  if (!Number.isInteger(args.streamReconnects) || args.streamReconnects < 0) {
    throw new Error('--stream-reconnects must be a non-negative integer');
  }
  if (!Number.isFinite(args.streamReconnectTimeoutS) || args.streamReconnectTimeoutS <= 0) {
    throw new Error('--stream-reconnect-timeout-s must be positive');
  }
  if (!Number.isInteger(args.uploadControlReconnects) || args.uploadControlReconnects < 0) {
    throw new Error('--upload-control-reconnects must be a non-negative integer');
  }
  if (!Number.isFinite(args.uploadControlReconnectTimeoutS) || args.uploadControlReconnectTimeoutS <= 0) {
    throw new Error('--upload-control-reconnect-timeout-s must be positive');
  }
  if (!Number.isFinite(args.attemptTimeoutS) || args.attemptTimeoutS < 0) {
    throw new Error('--attempt-timeout-s must be non-negative');
  }
  if (!Number.isFinite(args.attemptTimeoutMarginS) || args.attemptTimeoutMarginS < 0) {
    throw new Error('--attempt-timeout-margin-s must be non-negative');
  }
  if (!['rtsp', 'upload'].includes(args.localVideoMode)) {
    throw new Error('--local-video-mode must be "rtsp" or "upload"');
  }
  if (!['rtsp', 'rtmp'].includes(args.localRelayPublishProtocol)) {
    throw new Error('--local-relay-publish-protocol must be "rtsp" or "rtmp"');
  }
  if (!Number.isFinite(args.localRtspRelayPort) || args.localRtspRelayPort <= 0) {
    throw new Error('--local-rtsp-relay-port must be positive');
  }
  if (!Number.isFinite(args.localRtmpRelayPort) || args.localRtmpRelayPort <= 0) {
    throw new Error('--local-rtmp-relay-port must be positive');
  }
  if (!Number.isFinite(args.localRtspPlaybackPort) || args.localRtspPlaybackPort <= 0) {
    throw new Error('--local-rtsp-playback-port must be positive');
  }
  if (!Number.isFinite(args.localRtspWarmupMs) || args.localRtspWarmupMs < 0) {
    throw new Error('--local-rtsp-warmup-ms must be non-negative');
  }
  if (!Number.isFinite(args.localRtspFps) || args.localRtspFps <= 0) {
    throw new Error('--local-rtsp-fps must be positive');
  }
  if (!Number.isFinite(args.localRtspGopS) || args.localRtspGopS <= 0) {
    throw new Error('--local-rtsp-gop-s must be positive');
  }
  if (!Number.isFinite(args.localRtspScaleLongEdge) || args.localRtspScaleLongEdge < 0) {
    throw new Error('--local-rtsp-scale-long-edge must be non-negative');
  }
  if (!Number.isFinite(args.localRtspCrf) || args.localRtspCrf < 0 || args.localRtspCrf > 51) {
    throw new Error('--local-rtsp-crf must be between 0 and 51');
  }
  if (!Number.isFinite(args.localRtspPrerollS) || args.localRtspPrerollS < 0) {
    throw new Error('--local-rtsp-preroll-s must be non-negative');
  }
  if (!Number.isFinite(args.localRtspPostrollS) || args.localRtspPostrollS < 0) {
    throw new Error('--local-rtsp-postroll-s must be non-negative');
  }
  if (!Number.isFinite(args.localRtspConnectLeadS) || args.localRtspConnectLeadS < 0) {
    throw new Error('--local-rtsp-connect-lead-s must be non-negative');
  }
  if (!Number.isFinite(args.localRtspReconnectWarmupMs) || args.localRtspReconnectWarmupMs < 0) {
    throw new Error('--local-rtsp-reconnect-warmup-ms must be non-negative');
  }
  if (!Number.isFinite(args.videoDurationToleranceS) || args.videoDurationToleranceS < 0) {
    throw new Error('--video-duration-tolerance-s must be non-negative');
  }
  if (!Number.isFinite(args.videoQueryTimeToleranceS) || args.videoQueryTimeToleranceS < 0) {
    throw new Error('--video-query-time-tolerance-s must be non-negative');
  }
  if (!Number.isFinite(args.videoMaxBlackS) || args.videoMaxBlackS < 0) {
    throw new Error('--video-max-black-s must be non-negative');
  }
  if (!Number.isFinite(args.postVideoResponseTimeoutS) || args.postVideoResponseTimeoutS <= 0) {
    throw new Error('--post-video-response-timeout-s must be positive');
  }
  if (!Number.isFinite(args.postResponseRecordingS) || args.postResponseRecordingS < 0) {
    throw new Error('--post-response-recording-s must be non-negative');
  }
  if (!Number.isFinite(args.videoMaxExtraS) || args.videoMaxExtraS < 0) {
    throw new Error('--video-max-extra-s must be non-negative');
  }
  args.webUrl = normalizeRecordingWebUrl(args.webUrl);
  if (args.webUrl !== DEFAULT_WEB_URL) {
    throw new Error(`Recording WebUI must match the selected campaign config: ${DEFAULT_WEB_URL}`);
  }
  args.skipTaskIds = [...new Set(args.skipTaskIds.map((value) => String(value).trim()).filter(Boolean))];
  return args;
}

function printHelp() {
  console.log(`
Usage:
  npm run capture -- --tasks tasks.jsonl --web-url ${DEFAULT_WEB_URL} --out outputs/run_001

Options:
  --tasks PATH                 JSONL or CSV task file
  --web-url URL                WebUI URL, default ${DEFAULT_WEB_URL}
  --web-username USER          Optional WebUI HTTP Basic Auth username
  --web-password PASSWORD      Optional WebUI HTTP Basic Auth password
  --vlm-profile ID             Load a model from config/vlm_models.json
  --vlm-config PATH            Model registry path, default ${DEFAULT_VLM_CONFIG}
  --vlm-model MODEL            Switch WebUI to this VLM before video playback
  --vlm-api-base URL           OpenAI-compatible API base used with --vlm-model
  --vlm-control-api-base URL   Optional local adapter control base; defaults to --vlm-api-base
  --vlm-api-key KEY            Optional API key; prefer JOYVL_VLM_API_KEY
  --vlm-route ROUTE            auto, joyai_adapter, or direct; default auto
  --vlm-backend-alias MODEL    Allowed backend response model; repeatable
  --no-vlm-preflight           Disable strict upstream /models availability check
  --no-vlm-identity-check      Disable response backend identity verification
  --no-vlm-warmup              Disable the fail-closed inference barrier before playback
  --vlm-warmup-timeout-s N     Seconds allowed for each real visual warmup request, default 180
  --vlm-warmup-retries N       Retry a failed warmup N times before playback, default 2
  --out DIR                    Output directory, default outputs/run
  --default-duration-s N       Recording duration when task.duration_s is absent
  --process-interval-s N       Default WebUI processing interval
  --frames-per-batch N         Default WebUI frames per batch
  --limit N                    Run only first N tasks
  --task-gap-s N               Wait N seconds between tasks
  --task-retries N             Retry failed tasks N times, default 0
  --retry-gap-s N              Wait N seconds before a retry, default 20
  --health-retries N           WebUI TCP health-check attempts before each task
  --health-interval-s N        Seconds between WebUI health-check attempts
  --stream-ready-timeout-s N   Seconds to wait for streaming, default 60
  --video-upload-timeout-s N   Seconds allowed for one local-video upload, default 900
  --stream-start-retries N     Retry initial RTSP/WebRTC start N times before failing, default 0
  --stream-start-retry-gap-s N Seconds between start retries, default 3
  --stream-reconnects N        Rebuild WebRTC after mid-run ICE disconnect up to N times, default 0
  --stream-reconnect-timeout-s N
                               Seconds to wait for a reconnect to become streaming, default 60
  --upload-control-reconnects N
                               Reconnect the uploaded-video control WebSocket up to N times, default 5
  --upload-control-reconnect-timeout-s N
                               Seconds allowed for each uploaded-video control reconnect, default 10
  --attempt-timeout-s N        Hard wall timeout per task attempt; 0 derives from task duration
  --attempt-timeout-margin-s N Extra seconds for derived attempt timeout, default 240
  --local-video-mode MODE      How to feed local_video_path tasks: upload or rtsp, default upload
  --local-relay-publish-protocol P
                               Publisher protocol: rtsp or rtmp, default rtsp
  --local-rtsp-relay-host H    Relay publish host for local_video_path tasks, default 127.0.0.1
  --local-rtsp-relay-port N    RTSP publish port, default 8554
  --local-rtmp-relay-port N    RTMP publish port, default 1935
  --local-rtsp-playback-host H RTSP host passed to WebUI; defaults to the publish host
  --local-rtsp-playback-port N RTSP playback port passed to WebUI, default 8554
  --local-rtsp-relay-prefix P  RTSP path prefix, default vl_local
  --local-rtsp-warmup-ms N     Milliseconds to wait after starting ffmpeg before clicking start
  --local-rtsp-encode          Re-encode local video as low-latency H.264 instead of stream copy
  --local-rtsp-fps N           FPS for --local-rtsp-encode, default 12
  --local-rtsp-gop-s N         Keyframe interval seconds for --local-rtsp-encode, default 1
  --local-rtsp-scale-long-edge N
                               Downscale relay video so the long edge is at most N pixels, default 0/off
  --local-rtsp-crf N           H.264 CRF for relay encoding, default 23
  --local-rtsp-maxrate RATE    Optional H.264 VBV maxrate, e.g. 900k
  --local-rtsp-bufsize SIZE    Optional H.264 VBV bufsize, e.g. 1800k
  --local-rtsp-preroll-s N     Prepend N seconds of frozen first frame and align query to original t=0
  --local-rtsp-postroll-s N    Append N seconds of frozen last frame to keep relay alive, default 30
  --local-rtsp-connect-lead-s N
                               For preroll replay, delay browser RTSP/WebRTC start until N seconds before original t=0
  --local-rtsp-reconnect-warmup-ms N
                               Milliseconds to wait after forced publisher restart during WebRTC reconnect, default 3000
  --no-video-quality-check     Disable final decode/duration/black-screen validation
  --video-duration-tolerance-s N
                               Maximum task-video duration error, default 1.5
  --video-query-time-tolerance-s N
                               Maximum query dispatch timing error, default 0.5
  --video-max-black-s N        Maximum allowed continuous black screen, default 2
  --post-video-response-timeout-s N
                               Wait up to N seconds for the final at/after-end query reply, default 180
  --post-response-recording-s N
                               Keep recording N seconds after query replies are complete, default 1
  --video-max-extra-s N        Maximum unexplained task-video tail, default 5
  --ffmpeg-bin PATH            ffmpeg binary for local relay, default ffmpeg
  --skip-existing              Skip tasks with an existing status=ok summary.json
  --skip-task-id ID            Explicitly quarantine one task for this run; repeatable
  --no-source-decode-check     Disable full local source-video decode validation
  --keep-failed-attempts       Keep failed attempt directories for debugging; default deletes them after summary capture
  --fresh-browser-per-attempt  Relaunch Chromium before every task attempt for stronger isolation
  --headed                     Use visible Chromium
  --headless                   Use headless Chromium, default
  --no-mp4                     Keep only Playwright WebM
  --no-full-ui-mp4             Build task MP4 directly from WebM without a redundant full ui.mp4
`);
}

async function validateSourceVideoCached(videoPath, ffmpegBin) {
  const resolvedPath = path.resolve(videoPath);
  const stat = await fsp.stat(resolvedPath);
  const cacheKey = `${resolvedPath}:${stat.size}:${stat.mtimeMs}`;
  const cached = sourceVideoValidationCache.get(cacheKey);
  if (cached) return { ...cached, cached: true };
  const result = await validateVideoDecode({
    videoPath: resolvedPath,
    ffmpegBin,
    ffprobeBin: siblingFfprobe(ffmpegBin),
  });
  const validation = {
    ...result,
    path: resolvedPath,
    size_bytes: stat.size,
    mtime_ms: stat.mtimeMs,
    cached: false,
  };
  sourceVideoValidationCache.set(cacheKey, validation);
  return validation;
}

async function readTasks(taskPath) {
  const text = await fsp.readFile(taskPath, 'utf8');
  const ext = path.extname(taskPath).toLowerCase();
  const rows = ext === '.csv' ? parseCsv(text) : parseJsonl(text);
  return rows.map(normalizeTask);
}

function parseJsonl(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`Invalid JSONL at line ${index + 1}: ${error.message}`);
      }
    });
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;

  const pushCell = () => {
    row.push(cell);
    cell = '';
  };
  const pushRow = () => {
    if (row.length || cell) {
      pushCell();
      rows.push(row);
    }
    row = [];
  };

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (ch === '"' && next === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ',') {
      pushCell();
    } else if (ch === '\n') {
      pushRow();
    } else if (ch !== '\r') {
      cell += ch;
    }
  }
  if (cell || row.length) pushRow();
  if (!rows.length) return [];

  const header = rows[0].map((name) => name.trim());
  return rows.slice(1).filter((cells) => cells.some((c) => c.trim())).map((cells) => {
    const obj = {};
    header.forEach((name, index) => {
      obj[name] = cells[index] ?? '';
    });
    return obj;
  });
}

function parseTaskQueries(raw) {
  let queries = [];
  const rawQueries = raw.queries ?? raw.query_rounds ?? raw.prompts;
  if (Array.isArray(rawQueries)) {
    queries = rawQueries;
  } else if (typeof rawQueries === 'string' && rawQueries.trim()) {
    try {
      const parsed = JSON.parse(rawQueries);
      if (Array.isArray(parsed)) queries = parsed;
    } catch {
      queries = [];
    }
  }

  if (!queries.length) {
    queries = [{
      id: 'R1',
      query: raw.query ?? raw.prompt ?? raw.text_query ?? '',
      query_time_s: raw.query_time_s ?? raw.query_at_s ?? raw.query_second ?? raw.time_s,
    }];
  }

  return queries.map((item, index) => {
    const query = String(item.query ?? item.prompt ?? item.text_query ?? item.content ?? '').trim();
    const queryTimeS = Number(item.query_time_s ?? item.time_s ?? item.query_at_s ?? item.query_second);
    const id = String(item.id ?? item.round_id ?? item.label ?? `R${index + 1}`).trim() || `R${index + 1}`;
    if (!query) {
      throw new Error(`Query round ${id} is missing query text`);
    }
    if (!Number.isFinite(queryTimeS) || queryTimeS < 0) {
      throw new Error(`Query round ${id} has invalid query_time_s`);
    }
    return { id, query, query_time_s: queryTimeS };
  }).sort((a, b) => a.query_time_s - b.query_time_s);
}

function normalizeTask(raw) {
  const queries = parseTaskQueries(raw);
  const firstQuery = queries[0];
  const task = {
    id: String(raw.id ?? raw.task_id ?? raw.question_id ?? '').trim(),
    category: String(raw.category ?? '').trim(),
    scene: String(raw.scene ?? '').trim(),
    video_url: String(raw.video_url ?? raw.video ?? raw.url ?? raw.rtsp_url ?? '').trim(),
    local_video_path: String(raw.local_video_path ?? raw.source_video_path ?? raw.local_video ?? raw.source_video ?? '').trim(),
    original_video_url: String(raw.original_video_url ?? raw.original_rtsp_url ?? '').trim(),
    query: firstQuery.query,
    query_time_s: firstQuery.query_time_s,
    queries,
    duration_s: raw.duration_s === undefined || raw.duration_s === '' ? undefined : Number(raw.duration_s),
    local_video_duration_s: raw.local_video_duration_s === undefined || raw.local_video_duration_s === ''
      ? undefined
      : Number(raw.local_video_duration_s),
    process_interval_s: raw.process_interval_s === undefined || raw.process_interval_s === '' ? undefined : Number(raw.process_interval_s),
    frames_per_batch: raw.frames_per_batch === undefined || raw.frames_per_batch === '' ? undefined : Number(raw.frames_per_batch),
  };

  if (!task.id) throw new Error(`Task is missing id: ${JSON.stringify(raw)}`);
  if (!task.video_url && !task.local_video_path) {
    throw new Error(`Task ${task.id} is missing video_url or local_video_path`);
  }
  if (!task.query) throw new Error(`Task ${task.id} is missing query`);
  if (task.duration_s !== undefined && (!Number.isFinite(task.duration_s) || task.duration_s <= 0)) {
    throw new Error(`Task ${task.id} has invalid duration_s`);
  }
  return task;
}

function safeName(value) {
  return String(value).replace(/[^a-zA-Z0-9_.-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 120) || 'task';
}

function buildRelayEndpoints(args, task, attempt) {
  const prefix = safeName(args.localRtspRelayPrefix || 'vl_local');
  const id = safeName(task.id);
  const suffix = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const relayPath = `${prefix}_${id}_a${attempt}_${suffix}`;
  const publishProtocol = args.localRelayPublishProtocol;
  const publishPort = publishProtocol === 'rtmp'
    ? args.localRtmpRelayPort
    : args.localRtspRelayPort;
  const playbackHost = args.localRtspPlaybackHost || args.localRtspRelayHost;
  return {
    relayPath,
    publishProtocol,
    publishUrl: `${publishProtocol}://${args.localRtspRelayHost}:${publishPort}/${relayPath}`,
    playbackUrl: `rtsp://${playbackHost}:${args.localRtspPlaybackPort}/${relayPath}`,
    keepaliveUrl: `rtsp://${args.localRtspRelayHost}:${args.localRtspRelayPort}/${relayPath}`,
  };
}

function localRtspEncodingArgs(args) {
  const fps = Math.max(1, Math.round(args.localRtspFps));
  const gopFrames = Math.max(1, Math.round(fps * args.localRtspGopS));
  const encodeArgs = [
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-tune', 'zerolatency',
    '-pix_fmt', 'yuv420p',
    '-profile:v', 'baseline',
    '-g', String(gopFrames),
    '-keyint_min', String(gopFrames),
    '-sc_threshold', '0',
    '-bf', '0',
    '-refs', '1',
    '-crf', String(args.localRtspCrf),
    '-x264-params', 'repeat-headers=1:force-cfr=1',
  ];
  if (args.localRtspMaxrate) encodeArgs.push('-maxrate', args.localRtspMaxrate);
  if (args.localRtspBufsize) encodeArgs.push('-bufsize', args.localRtspBufsize);
  return { fps, gopFrames, args: encodeArgs };
}

function localRtspVideoFilter(args, { preroll = false } = {}) {
  const filters = [`fps=${Math.max(1, Math.round(args.localRtspFps))}`];
  if (preroll) {
    filters.push(
      `tpad=start_duration=${args.localRtspPrerollS}:start_mode=clone:`
      + `stop_duration=${args.localRtspPostrollS}:stop_mode=clone`,
    );
  }
  if (args.localRtspScaleLongEdge > 0) {
    const edge = Math.max(2, Math.round(args.localRtspScaleLongEdge / 2) * 2);
    filters.push(`scale='if(gte(iw\\,ih)\\,min(${edge}\\,iw)\\,-2)':'if(gte(iw\\,ih)\\,-2\\,min(${edge}\\,ih))'`);
  }
  filters.push('setpts=PTS-STARTPTS');
  return filters.join(',');
}

function replayCachePath(args, sourcePath) {
  const stat = fs.statSync(sourcePath);
  const key = [
    sourcePath,
    stat.size,
    Math.round(stat.mtimeMs),
    args.localRtspPrerollS,
    args.localRtspPostrollS,
    args.localRtspFps,
    args.localRtspGopS,
    args.localRtspScaleLongEdge,
    args.localRtspCrf,
    args.localRtspMaxrate,
    args.localRtspBufsize,
  ].join('|');
  const hash = crypto.createHash('sha256').update(key).digest('hex').slice(0, 16);
  const base = safeName(path.basename(sourcePath, path.extname(sourcePath)));
  return path.join(args.out, '_rtsp_replay_cache', `${base}_${hash}.mp4`);
}

async function runLoggedProcess(command, argv, logPath) {
  await ensureDir(path.dirname(logPath));
  const log = fs.createWriteStream(logPath, { flags: 'w' });
  return new Promise((resolve, reject) => {
    const proc = spawn(command, argv, { stdio: ['ignore', 'pipe', 'pipe'] });
    proc.stdout.pipe(log, { end: false });
    proc.stderr.pipe(log, { end: false });
    proc.on('error', (error) => {
      log.end();
      reject(error);
    });
    proc.on('close', (code, signal) => {
      log.end();
      if (code === 0) {
        resolve({ code, signal });
      } else {
        reject(new Error(`${command} exited with code ${code ?? 'null'} signal ${signal ?? 'null'}; see ${logPath}`));
      }
    });
  });
}

async function prepareLocalRtspReplay(args, task, sourcePath, taskDir, events, startedAtMs) {
  if (args.localRtspPrerollS <= 0) {
    return {
      inputPath: sourcePath,
      cachePath: '',
      logPath: '',
      prerollS: 0,
      postrollS: 0,
      preprocessed: false,
    };
  }

  const cachePath = replayCachePath(args, sourcePath);
  const logPath = path.join(taskDir, 'local_rtsp_replay_preprocess.log');
  if (fs.existsSync(cachePath) && fs.statSync(cachePath).size > 0) {
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'local_rtsp_replay_cache_hit',
      task_id: task.id,
      source_video_path: sourcePath,
      replay_input_path: cachePath,
      preroll_s: args.localRtspPrerollS,
      postroll_s: args.localRtspPostrollS,
    });
    return {
      inputPath: cachePath,
      cachePath,
      logPath,
      prerollS: args.localRtspPrerollS,
      postrollS: args.localRtspPostrollS,
      preprocessed: true,
    };
  }

  await ensureDir(path.dirname(cachePath));
  const tmpPath = `${cachePath}.tmp-${process.pid}-${Date.now()}.mp4`;
  const { args: encodeArgs } = localRtspEncodingArgs(args);
  const ffmpegArgs = [
    '-hide_banner',
    '-y',
    '-i', sourcePath,
    '-map', '0:v:0',
    '-an',
    '-vf', localRtspVideoFilter(args, { preroll: true }),
    ...encodeArgs,
    '-f', 'mp4',
    '-movflags', '+faststart',
    tmpPath,
  ];

  appendJsonl(events, {
    t_ms: monotonicMs() - startedAtMs,
    type: 'local_rtsp_replay_preprocess_start',
    task_id: task.id,
    ffmpeg: args.ffmpegBin,
    args: ffmpegArgs,
    source_video_path: sourcePath,
    replay_input_path: cachePath,
    log_path: logPath,
    preroll_s: args.localRtspPrerollS,
    postroll_s: args.localRtspPostrollS,
  });

  try {
    await runLoggedProcess(args.ffmpegBin, ffmpegArgs, logPath);
    await fsp.rename(tmpPath, cachePath);
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'local_rtsp_replay_preprocess_done',
      task_id: task.id,
      replay_input_path: cachePath,
      log_path: logPath,
      preroll_s: args.localRtspPrerollS,
      postroll_s: args.localRtspPostrollS,
    });
  } catch (error) {
    await fsp.rm(tmpPath, { force: true }).catch(() => {});
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'local_rtsp_replay_preprocess_error',
      task_id: task.id,
      error: error.message,
      source_video_path: sourcePath,
      replay_input_path: cachePath,
      log_path: logPath,
    });
    throw error;
  }

  return {
    inputPath: cachePath,
    cachePath,
    logPath,
    prerollS: args.localRtspPrerollS,
    postrollS: args.localRtspPostrollS,
    preprocessed: true,
  };
}

async function probeMediaDurationS(command, inputPath) {
  const ffprobe = path.join(path.dirname(command), 'ffprobe');
  const candidates = command.includes(path.sep) ? [ffprobe, 'ffprobe'] : ['ffprobe'];
  for (const candidate of candidates) {
    const duration = await new Promise((resolve) => {
      const proc = spawn(candidate, [
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        inputPath,
      ], { stdio: ['ignore', 'pipe', 'ignore'] });
      let stdout = '';
      proc.stdout.on('data', (chunk) => {
        stdout += chunk.toString();
      });
      proc.on('error', () => resolve(null));
      proc.on('close', (code) => {
        if (code !== 0) {
          resolve(null);
          return;
        }
        const value = Number(stdout.trim());
        resolve(Number.isFinite(value) && value > 0 ? value : null);
      });
    });
    if (duration) return duration;
  }
  return null;
}

async function probeMediaVideoCodec(command, inputPath) {
  const ffprobe = path.join(path.dirname(command), 'ffprobe');
  const candidates = command.includes(path.sep) ? [ffprobe, 'ffprobe'] : ['ffprobe'];
  for (const candidate of candidates) {
    const codec = await new Promise((resolve) => {
      const proc = spawn(candidate, [
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        inputPath,
      ], { stdio: ['ignore', 'pipe', 'ignore'] });
      let stdout = '';
      proc.stdout.on('data', (chunk) => {
        stdout += chunk.toString();
      });
      proc.on('error', () => resolve(''));
      proc.on('close', (code) => {
        resolve(code === 0 ? stdout.trim().toLowerCase() : '');
      });
    });
    if (codec) return codec;
  }
  return '';
}

function uploadCachePath(args, sourcePath) {
  const stat = fs.statSync(sourcePath);
  const key = [sourcePath, stat.size, Math.round(stat.mtimeMs), 'h264-crf18-aac'].join('|');
  const hash = crypto.createHash('sha256').update(key).digest('hex').slice(0, 16);
  const base = safeName(path.basename(sourcePath, path.extname(sourcePath)));
  return path.join(args.out, '_upload_video_cache', `${base}_${hash}.mp4`);
}

async function prepareLocalUploadVideo(args, task, sourcePath, taskDir, events, startedAtMs) {
  const sourceCodec = await probeMediaVideoCodec(args.ffmpegBin, sourcePath);
  if (!sourceCodec || sourceCodec === 'h264') {
    return {
      inputPath: sourcePath,
      sourceCodec: sourceCodec || 'unknown',
      transcoded: false,
      cachePath: '',
      logPath: '',
    };
  }

  const cachePath = uploadCachePath(args, sourcePath);
  const logPath = path.join(taskDir, 'upload_video_preprocess.log');
  if (fs.existsSync(cachePath) && fs.statSync(cachePath).size > 0) {
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'capture_upload_video_cache_hit',
      task_id: task.id,
      source_video_path: sourcePath,
      source_video_codec: sourceCodec,
      upload_video_path: cachePath,
    });
    return {
      inputPath: cachePath,
      sourceCodec,
      transcoded: true,
      cachePath,
      logPath,
    };
  }

  await ensureDir(path.dirname(cachePath));
  const tmpPath = `${cachePath}.tmp-${process.pid}-${Date.now()}.mp4`;
  const ffmpegArgs = [
    '-hide_banner',
    '-y',
    '-i', sourcePath,
    '-map', '0:v:0',
    '-map', '0:a?',
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-crf', '18',
    '-pix_fmt', 'yuv420p',
    '-c:a', 'aac',
    '-b:a', '128k',
    '-movflags', '+faststart',
    tmpPath,
  ];
  appendJsonl(events, {
    t_ms: monotonicMs() - startedAtMs,
    type: 'capture_upload_video_preprocess_start',
    task_id: task.id,
    source_video_path: sourcePath,
    source_video_codec: sourceCodec,
    upload_video_path: cachePath,
    ffmpeg: args.ffmpegBin,
    args: ffmpegArgs,
    log_path: logPath,
  });

  try {
    await runLoggedProcess(args.ffmpegBin, ffmpegArgs, logPath);
    await fsp.rename(tmpPath, cachePath);
  } catch (error) {
    await fsp.rm(tmpPath, { force: true }).catch(() => {});
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'capture_upload_video_preprocess_error',
      task_id: task.id,
      source_video_path: sourcePath,
      source_video_codec: sourceCodec,
      error: error.message,
      log_path: logPath,
    });
    throw error;
  }

  appendJsonl(events, {
    t_ms: monotonicMs() - startedAtMs,
    type: 'capture_upload_video_preprocess_done',
    task_id: task.id,
    source_video_path: sourcePath,
    source_video_codec: sourceCodec,
    upload_video_path: cachePath,
    log_path: logPath,
  });
  return {
    inputPath: cachePath,
    sourceCodec,
    transcoded: true,
    cachePath,
    logPath,
  };
}

async function startLocalRtspPublisher(args, task, relay, taskDir, events, startedAtMs) {
  const sourcePath = path.resolve(task.local_video_path);
  if (!fs.existsSync(sourcePath)) {
    throw new Error(`Local video not found for ${task.id}: ${sourcePath}`);
  }

  const replay = await prepareLocalRtspReplay(args, task, sourcePath, taskDir, events, startedAtMs);
  const replayDurationS = await probeMediaDurationS(args.ffmpegBin, replay.inputPath);
  const logPath = path.join(taskDir, 'local_rtsp_publisher.log');
  const keepaliveLogPath = path.join(taskDir, 'local_rtsp_keepalive.log');
  const log = fs.createWriteStream(logPath, { flags: 'w' });
  const keepaliveLog = fs.createWriteStream(keepaliveLogPath, { flags: 'w' });
  const initialReplayOffsetS = replay.prerollS > 0 && args.localRtspConnectLeadS > 0
    ? Math.max(0, replay.prerollS - args.localRtspConnectLeadS - args.localRtspWarmupMs / 1000)
    : 0;
  const timelineStartMs = monotonicMs() - initialReplayOffsetS * 1000;
  const controlledFrameRelay = relay.publishProtocol === 'rtmp' && replay.preprocessed;

  const publisher = {
    proc: null,
    stopped: false,
    restartCount: 0,
    restartTimer: null,
    sourcePath,
    publisherInputPath: replay.inputPath,
    replayCachePath: replay.cachePath,
    replayPreprocessLogPath: replay.logPath,
    prerollS: replay.prerollS,
    postrollS: replay.postrollS,
    replayDurationS,
    initialReplayOffsetS,
    relayPath: relay.relayPath,
    publishProtocol: relay.publishProtocol,
    publishUrl: relay.publishUrl,
    playbackUrl: relay.playbackUrl,
    keepaliveUrl: relay.keepaliveUrl,
    relayUrl: relay.publishUrl,
    logPath,
    log,
    keepaliveLogPath,
    keepaliveLog,
    keepaliveProc: null,
    keepaliveOwnerProc: null,
    timelineStartMs,
    controlledFrameRelay,
    timelineFrozenAtMs: null,
    autoFrozenForStall: false,
    lastStartMs: 0,
    forceRestartCount: 0,
    restartWaiters: [],
    ready: false,
    readyAtMs: 0,
    pendingSeekOffsetS: initialReplayOffsetS,
    lastSourcePositionS: initialReplayOffsetS,
    lastProgressAtMs: null,
    lastProgressSourcePositionS: null,
    activeOutage: null,
    outageIntervals: [],
  };

  publisher.beginOutage = (startMs, reason, sourcePositionS = null) => {
    const boundedStartMs = Math.min(monotonicMs(), Math.max(startedAtMs, startMs));
    if (publisher.activeOutage) {
      publisher.activeOutage.startMs = Math.min(publisher.activeOutage.startMs, boundedStartMs);
      if (reason && !publisher.activeOutage.reasons.includes(reason)) {
        publisher.activeOutage.reasons.push(reason);
      }
      return;
    }
    publisher.activeOutage = {
      startMs: boundedStartMs,
      startSourcePositionS: Number.isFinite(sourcePositionS) ? sourcePositionS : null,
      reasons: reason ? [reason] : [],
    };
    appendJsonl(events, {
      t_ms: boundedStartMs - startedAtMs,
      type: 'capture_output_outage_start',
      task_id: task.id,
      reason,
      source_position_s: publisher.activeOutage.startSourcePositionS,
    });
  };

  publisher.endOutage = (endMs, reason) => {
    const active = publisher.activeOutage;
    if (!active || endMs <= active.startMs) return;
    const interval = {
      startMs: active.startMs,
      endMs,
      startSourcePositionS: active.startSourcePositionS,
      reasons: [...new Set([...active.reasons, reason].filter(Boolean))],
    };
    publisher.outageIntervals.push(interval);
    publisher.activeOutage = null;
    appendJsonl(events, {
      t_ms: endMs - startedAtMs,
      type: 'capture_output_outage_end',
      task_id: task.id,
      outage_start_offset_s: (interval.startMs - startedAtMs) / 1000,
      outage_end_offset_s: (interval.endMs - startedAtMs) / 1000,
      outage_duration_s: (interval.endMs - interval.startMs) / 1000,
      reasons: interval.reasons,
    });
  };

  publisher.waitUntilReady = (timeoutMs = 30000) => {
    if (publisher.ready && publisher.proc && publisher.proc.exitCode === null) {
      return Promise.resolve({
        restart_count: publisher.restartCount,
        seek_offset_s: publisher.lastSeekOffsetS,
        pid: publisher.proc.pid,
        ready_offset_ms: publisher.readyAtMs - startedAtMs,
      });
    }
    return new Promise((resolve) => {
      let settled = false;
      const finish = (value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        resolve(value);
      };
      const timeout = setTimeout(() => finish(null), timeoutMs);
      publisher.restartWaiters.push(finish);
    });
  };

  const buildFfmpegArgs = (seekOffsetS) => {
    const ffmpegArgs = [
      '-hide_banner',
      '-loglevel', 'info',
      '-re',
    ];
    if (seekOffsetS > 0.001) {
      ffmpegArgs.push('-ss', seekOffsetS.toFixed(3));
    }
    ffmpegArgs.push(
      '-fflags', '+genpts',
      '-i', replay.inputPath,
      '-map', '0:v:0',
      '-an',
    );

    if (replay.preprocessed) {
      ffmpegArgs.push('-c:v', 'copy');
    } else if (args.localRtspEncode) {
      const { args: encodeArgs } = localRtspEncodingArgs(args);
      ffmpegArgs.push(
        '-vf', localRtspVideoFilter(args),
        ...encodeArgs,
      );
    } else {
      ffmpegArgs.push('-c:v', 'copy');
    }

    if (relay.publishProtocol === 'rtmp') {
      ffmpegArgs.push(
        '-avoid_negative_ts', 'make_zero',
        '-f', 'flv',
        relay.publishUrl,
      );
    } else {
      ffmpegArgs.push(
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        relay.publishUrl,
      );
    }
    return ffmpegArgs;
  };

  const buildPublisherInvocation = (seekOffsetS) => {
    if (!controlledFrameRelay) {
      return { command: args.ffmpegBin, argv: buildFfmpegArgs(seekOffsetS) };
    }
    const relayScript = path.join(path.dirname(fileURLToPath(import.meta.url)), 'realtime_frame_relay.mjs');
    const argv = [
      relayScript,
      '--input', replay.inputPath,
      '--output', relay.publishUrl,
      '--ffmpeg', args.ffmpegBin,
      '--seek-s', seekOffsetS.toFixed(3),
      '--fps', String(Math.max(1, Math.round(args.localRtspFps))),
      '--gop-s', String(args.localRtspGopS),
      '--crf', String(args.localRtspCrf),
    ];
    if (args.localRtspMaxrate) argv.push('--maxrate', args.localRtspMaxrate);
    if (args.localRtspBufsize) argv.push('--bufsize', args.localRtspBufsize);
    if (publisher.timelineFrozenAtMs != null) argv.push('--start-frozen');
    return { command: process.execPath, argv };
  };

  publisher.currentTimelinePositionS = () => {
    const referenceMs = publisher.timelineFrozenAtMs ?? monotonicMs();
    return Math.max(0, (referenceMs - publisher.timelineStartMs) / 1000);
  };

  publisher.stopKeepalive = () => {
    const keepaliveProc = publisher.keepaliveProc;
    publisher.keepaliveProc = null;
    publisher.keepaliveOwnerProc = null;
    if (!keepaliveProc || keepaliveProc.exitCode !== null || keepaliveProc.signalCode !== null) return;
    keepaliveProc.kill('SIGTERM');
    setTimeout(() => {
      if (keepaliveProc.exitCode === null && keepaliveProc.signalCode === null) {
        keepaliveProc.kill('SIGKILL');
      }
    }, 5000).unref();
  };

  const spawnOne = () => {
    if (publisher.stopped) return;
    if (publisher.restartTimer) {
      clearTimeout(publisher.restartTimer);
      publisher.restartTimer = null;
    }
    publisher.restartCount += 1;
    publisher.lastStartMs = monotonicMs();
    publisher.ready = false;
    publisher.stopKeepalive();
    const elapsedS = publisher.currentTimelinePositionS();
    const desiredSeekOffsetS = publisher.pendingSeekOffsetS ?? elapsedS;
    const seekOffsetS = publisher.replayDurationS && publisher.replayDurationS > 0
      ? Math.min(desiredSeekOffsetS, Math.max(0, publisher.replayDurationS - 0.5))
      : desiredSeekOffsetS;
    const invocation = buildPublisherInvocation(seekOffsetS);
    const ffmpegArgs = invocation.argv;
    publisher.lastSeekOffsetS = seekOffsetS;
    publisher.lastSourcePositionS = seekOffsetS;
    const proc = spawn(invocation.command, ffmpegArgs, {
      stdio: [controlledFrameRelay ? 'pipe' : 'ignore', 'pipe', 'pipe'],
    });
    publisher.proc = proc;
    proc.stdout.pipe(log, { end: false });
    proc.stderr.pipe(log, { end: false });
    const markReady = () => {
      if (publisher.proc !== proc || publisher.ready) return;
      publisher.ready = true;
      publisher.readyAtMs = monotonicMs();
      const resumeAfterReady = publisher.autoFrozenForStall;
      if (publisher.pendingSeekOffsetS != null) {
        const readyPositionS = Number.isFinite(publisher.lastSourcePositionS)
          ? publisher.lastSourcePositionS
          : seekOffsetS;
        publisher.timelineStartMs = publisher.readyAtMs - readyPositionS * 1000;
        if (publisher.timelineFrozenAtMs != null) {
          publisher.timelineFrozenAtMs = publisher.readyAtMs;
        }
        publisher.pendingSeekOffsetS = null;
      }
      if (publisher.autoFrozenForStall) {
        publisher.timelineFrozenAtMs = null;
        publisher.autoFrozenForStall = false;
        publisher.endOutage(publisher.readyAtMs, 'publisher ready after automatic restart');
      }
      const readyInfo = {
        restart_count: publisher.restartCount,
        seek_offset_s: seekOffsetS,
        source_position_s: publisher.lastSourcePositionS,
        pid: proc.pid,
        ready_offset_ms: publisher.readyAtMs - startedAtMs,
      };
      appendJsonl(events, {
        t_ms: publisher.readyAtMs - startedAtMs,
        type: 'local_rtsp_publisher_ready',
        task_id: task.id,
        ...readyInfo,
        publish_url: relay.publishUrl,
        playback_url: relay.playbackUrl,
      });
      const waiters = publisher.restartWaiters.splice(0);
      for (const resolve of waiters) resolve(readyInfo);
      if (resumeAfterReady && proc.stdin?.writable) {
        proc.stdin.write('resume\n');
      } else if (publisher.timelineFrozenAtMs != null && proc.stdin?.writable) {
        proc.stdin.write('freeze\n');
      }
    };
    let keepaliveStarted = false;
    const startKeepalive = () => {
      if (keepaliveStarted || publisher.proc !== proc || publisher.stopped) return;
      keepaliveStarted = true;
      const keepaliveScript = path.join(
        path.dirname(fileURLToPath(import.meta.url)),
        'rtsp_keepalive_reader.mjs',
      );
      const keepaliveProc = spawn(process.execPath, [
        keepaliveScript,
        '--url', relay.keepaliveUrl,
        '--ffmpeg', args.ffmpegBin,
        '--retry-ms', '500',
        '--connect-timeout-ms', '15000',
        '--stall-timeout-ms', '20000',
      ], { stdio: ['ignore', 'pipe', 'pipe'] });
      publisher.keepaliveProc = keepaliveProc;
      publisher.keepaliveOwnerProc = proc;
      keepaliveProc.stdout.pipe(keepaliveLog, { end: false });
      keepaliveProc.stderr.pipe(keepaliveLog, { end: false });
      let stdoutTail = '';
      keepaliveProc.stdout.on('data', (chunk) => {
        stdoutTail += chunk.toString();
        const lines = stdoutTail.split(/\r?\n/);
        stdoutTail = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          let message;
          try {
            message = JSON.parse(line);
          } catch {
            continue;
          }
          const messageAtMs = monotonicMs();
          appendJsonl(events, {
            t_ms: messageAtMs - startedAtMs,
            task_id: task.id,
            ...message,
            type: `local_rtsp_keepalive_${message.type}`,
          });
          if (publisher.proc !== proc || publisher.keepaliveProc !== keepaliveProc) continue;
          if (message.type === 'reader_ready') {
            markReady();
          } else if (message.type === 'reader_exit' && message.ready) {
            publisher.ready = false;
          }
        }
      });
      keepaliveProc.on('exit', (code, signal) => {
        if (publisher.keepaliveProc !== keepaliveProc) return;
        publisher.keepaliveProc = null;
        publisher.keepaliveOwnerProc = null;
        publisher.ready = false;
        appendJsonl(events, {
          t_ms: monotonicMs() - startedAtMs,
          type: 'local_rtsp_keepalive_exit',
          task_id: task.id,
          code,
          signal,
          url: relay.keepaliveUrl,
          stopped: publisher.stopped,
        });
        if (!publisher.stopped && publisher.proc === proc && proc.exitCode === null) {
          keepaliveStarted = false;
          setTimeout(startKeepalive, 500).unref();
        }
      });
    };
    const markPublisherOutputReady = () => {
      if (publisher.proc !== proc) return;
      startKeepalive();
    };
    let stderrTail = '';
    proc.stderr.on('data', (chunk) => {
      stderrTail = `${stderrTail}${chunk.toString()}`.slice(-4096);
      if (!controlledFrameRelay && /Output #0|frame=\s*\d+/.test(stderrTail)) {
        markPublisherOutputReady();
      }
    });
    if (controlledFrameRelay) {
      let stdoutTail = '';
      proc.stdout.on('data', (chunk) => {
        stdoutTail += chunk.toString();
        const lines = stdoutTail.split(/\r?\n/);
        stdoutTail = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          let message;
          try {
            message = JSON.parse(line);
          } catch {
            continue;
          }
          const messageAtMs = monotonicMs();
          const sourcePositionS = Number(message.source_position_s);
          if (Number.isFinite(sourcePositionS)) publisher.lastSourcePositionS = sourcePositionS;
          if (Number.isFinite(sourcePositionS) && message.type === 'relay_frozen') {
            publisher.timelineStartMs = messageAtMs - sourcePositionS * 1000;
            publisher.timelineFrozenAtMs = messageAtMs;
          } else if (Number.isFinite(sourcePositionS) && message.type === 'relay_resumed') {
            publisher.timelineStartMs = messageAtMs - sourcePositionS * 1000;
            publisher.timelineFrozenAtMs = null;
          } else if (
            Number.isFinite(sourcePositionS)
            && message.type === 'relay_progress'
            && publisher.timelineFrozenAtMs == null
          ) {
            publisher.timelineStartMs = messageAtMs - sourcePositionS * 1000;
            publisher.lastProgressAtMs = messageAtMs;
            publisher.lastProgressSourcePositionS = sourcePositionS;
          } else if (
            Number.isFinite(sourcePositionS)
            && (message.type === 'relay_output_stalled' || message.type === 'relay_failed')
          ) {
            let outageStartMs = messageAtMs;
            if (
              Number.isFinite(publisher.lastProgressAtMs)
              && Number.isFinite(publisher.lastProgressSourcePositionS)
            ) {
              outageStartMs = Math.min(
                messageAtMs,
                publisher.lastProgressAtMs
                  + Math.max(0, sourcePositionS - publisher.lastProgressSourcePositionS) * 1000,
              );
            }
            publisher.beginOutage(outageStartMs, message.type, sourcePositionS);
            publisher.timelineStartMs = messageAtMs - sourcePositionS * 1000;
            publisher.pendingSeekOffsetS = sourcePositionS;
            if (publisher.timelineFrozenAtMs == null) publisher.autoFrozenForStall = true;
            publisher.timelineFrozenAtMs = messageAtMs;
          }
          if (message.type === 'relay_resumed') {
            publisher.endOutage(messageAtMs, 'relay resumed');
          }
          appendJsonl(events, {
            t_ms: messageAtMs - startedAtMs,
            task_id: task.id,
            ...message,
            type: `local_rtsp_${message.type}`,
          });
          if (message.type === 'relay_ready') markPublisherOutputReady();
        }
      });
    }
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'local_rtsp_publisher_start',
      pid: proc.pid,
      restart_count: publisher.restartCount,
      ffmpeg: invocation.command,
      args: ffmpegArgs,
      source_video_path: sourcePath,
      publisher_input_path: replay.inputPath,
      replay_cache_path: replay.cachePath || undefined,
      preroll_s: replay.prerollS,
      postroll_s: replay.postrollS,
      replay_duration_s: publisher.replayDurationS || undefined,
      initial_replay_offset_s: publisher.initialReplayOffsetS,
      relay_path: relay.relayPath,
      publish_protocol: relay.publishProtocol,
      publish_url: relay.publishUrl,
      playback_url: relay.playbackUrl,
      relay_url: relay.publishUrl,
      log_path: logPath,
      publisher_timeline_start_offset_ms: publisher.timelineStartMs - startedAtMs,
      publisher_start_offset_ms: publisher.lastStartMs - startedAtMs,
      publisher_elapsed_s: elapsedS,
      publisher_seek_offset_s: seekOffsetS,
      controlled_frame_relay: controlledFrameRelay,
    });
    proc.on('error', (error) => {
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'local_rtsp_publisher_error',
        error: error.message,
        source_video_path: sourcePath,
        publish_url: relay.publishUrl,
        playback_url: relay.playbackUrl,
      });
    });
    proc.on('exit', (code, signal) => {
      if (publisher.proc === proc) {
        publisher.ready = false;
        publisher.stopKeepalive();
      }
      if (!publisher.stopped && publisher.pendingSeekOffsetS == null) {
        const exitPositionS = publisher.currentTimelinePositionS();
        publisher.pendingSeekOffsetS = publisher.replayDurationS && publisher.replayDurationS > 0
          ? Math.min(exitPositionS, Math.max(0, publisher.replayDurationS - 0.5))
          : exitPositionS;
      }
      if (!publisher.stopped) {
        publisher.beginOutage(
          monotonicMs(),
          `publisher exit (${signal || (code ?? 'unknown')})`,
          publisher.lastSourcePositionS,
        );
      }
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'local_rtsp_publisher_exit',
        code,
        signal,
        restart_count: publisher.restartCount,
        publish_url: relay.publishUrl,
        playback_url: relay.playbackUrl,
        stopped: publisher.stopped,
      });
      if (!publisher.stopped) {
        if (!publisher.restartTimer) {
          publisher.restartTimer = setTimeout(() => {
            publisher.restartTimer = null;
            spawnOne();
          }, 500);
        }
      }
    });
  };

  publisher.restartNow = async (reason, { resetTimeline = false } = {}) => {
    if (publisher.stopped) return null;
    publisher.forceRestartCount += 1;
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'local_rtsp_publisher_force_restart',
      task_id: task.id,
      reason,
      force_restart_count: publisher.forceRestartCount,
      restart_count: publisher.restartCount,
      publish_url: relay.publishUrl,
      playback_url: relay.playbackUrl,
    });

    if (resetTimeline) {
      publisher.pendingSeekOffsetS = publisher.initialReplayOffsetS;
    } else if (publisher.pendingSeekOffsetS == null) {
      const currentPositionS = publisher.currentTimelinePositionS();
      publisher.pendingSeekOffsetS = publisher.replayDurationS && publisher.replayDurationS > 0
        ? Math.min(currentPositionS, Math.max(0, publisher.replayDurationS - 0.5))
        : currentPositionS;
    }
    publisher.ready = false;
    const waitForReady = publisher.waitUntilReady(90000);
    if (publisher.restartTimer) {
      clearTimeout(publisher.restartTimer);
      publisher.restartTimer = null;
    }

    const proc = publisher.proc;
    if (!proc || proc.exitCode !== null || proc.signalCode !== null) {
      spawnOne();
      return waitForReady;
    }

    let exited = false;
    proc.once('exit', () => {
      exited = true;
    });
    proc.kill('SIGTERM');
    setTimeout(() => {
      if (!exited) proc.kill('SIGKILL');
    }, 1500).unref();

    return waitForReady;
  };

  publisher.freezeTimeline = (reason) => {
    if (!publisher.controlledFrameRelay) return false;
    if (publisher.timelineFrozenAtMs != null) {
      publisher.autoFrozenForStall = false;
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'local_rtsp_publisher_timeline_freeze_held',
        task_id: task.id,
        reason,
        source_position_s: publisher.currentTimelinePositionS(),
      });
      return true;
    }
    publisher.timelineFrozenAtMs = monotonicMs();
    publisher.beginOutage(publisher.timelineFrozenAtMs, reason, publisher.currentTimelinePositionS());
    if (publisher.proc?.stdin?.writable) publisher.proc.stdin.write('freeze\n');
    appendJsonl(events, {
      t_ms: publisher.timelineFrozenAtMs - startedAtMs,
      type: 'local_rtsp_publisher_timeline_frozen',
      task_id: task.id,
      reason,
      source_position_s: publisher.currentTimelinePositionS(),
    });
    return true;
  };

  publisher.resumeTimeline = (reason) => {
    if (!publisher.controlledFrameRelay || publisher.timelineFrozenAtMs == null) return 0;
    const resumedAtMs = monotonicMs();
    const frozenMs = Math.max(0, resumedAtMs - publisher.timelineFrozenAtMs);
    publisher.timelineStartMs += frozenMs;
    publisher.timelineFrozenAtMs = null;
    publisher.endOutage(resumedAtMs, reason);
    if (publisher.proc?.stdin?.writable) publisher.proc.stdin.write('resume\n');
    appendJsonl(events, {
      t_ms: resumedAtMs - startedAtMs,
      type: 'local_rtsp_publisher_timeline_resumed',
      task_id: task.id,
      reason,
      frozen_ms: frozenMs,
      source_position_s: publisher.currentTimelinePositionS(),
    });
    return frozenMs;
  };

  spawnOne();
  return publisher;
}

async function stopLocalRtspPublisher(publisher) {
  if (!publisher) return;
  publisher.stopped = true;
  publisher.stopKeepalive?.();
  if (publisher.restartTimer) {
    clearTimeout(publisher.restartTimer);
    publisher.restartTimer = null;
  }
  if (!publisher.proc || publisher.proc.killed) {
    publisher.log?.end();
    publisher.keepaliveLog?.end();
    return;
  }
  if (publisher.controlledFrameRelay && publisher.proc.stdin?.writable) {
    publisher.proc.stdin.write('stop\n');
  }
  let exited = false;
  publisher.proc.once('exit', () => {
    exited = true;
  });
  publisher.proc.kill('SIGTERM');
  await new Promise((resolve) => {
    const timeout = setTimeout(() => {
      if (!exited) publisher.proc.kill('SIGKILL');
      resolve();
    }, 3000);
    publisher.proc.once('exit', () => {
      clearTimeout(timeout);
      resolve();
    });
  });
  publisher.log?.end();
  publisher.keepaliveLog?.end();
}

async function ensureDir(dir) {
  await fsp.mkdir(dir, { recursive: true });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function tcpProbe(urlString, timeoutMs = 5000) {
  return new Promise((resolve) => {
    let url;
    try {
      url = new URL(urlString);
    } catch {
      resolve(false);
      return;
    }
    const port = Number(url.port || (url.protocol === 'https:' ? 443 : 80));
    const socket = net.createConnection({ host: url.hostname, port });
    let done = false;
    const finish = (ok) => {
      if (done) return;
      done = true;
      socket.destroy();
      resolve(ok);
    };
    socket.setTimeout(timeoutMs);
    socket.once('connect', () => finish(true));
    socket.once('timeout', () => finish(false));
    socket.once('error', () => finish(false));
  });
}

async function waitForWebUiHealthy(args, label) {
  for (let attempt = 1; attempt <= args.healthRetries; attempt += 1) {
    const ok = await tcpProbe(args.webUrl, 5000);
    if (ok) {
      if (args.healthRetries > 1) {
        console.log(`${label}: WebUI health ok on attempt ${attempt}/${args.healthRetries}`);
      }
      return;
    }
    if (attempt < args.healthRetries) {
      console.log(`${label}: WebUI health failed (${attempt}/${args.healthRetries}); retrying in ${args.healthIntervalS}s`);
      await sleep(args.healthIntervalS * 1000);
    }
  }
  throw new Error(`${label}: WebUI TCP health check failed after ${args.healthRetries} attempt(s): ${args.webUrl}`);
}

function monotonicMs() {
  return Number(process.hrtime.bigint() / 1000000n);
}

async function samplePagePerformanceClock(page, sampleCount = 5) {
  const samples = [];
  for (let index = 0; index < sampleCount; index += 1) {
    const requestedAtMs = monotonicMs();
    const pagePerformanceMs = await page.evaluate(() => performance.now());
    const receivedAtMs = monotonicMs();
    samples.push({
      requested_at_ms: requestedAtMs,
      received_at_ms: receivedAtMs,
      page_performance_ms: pagePerformanceMs,
      round_trip_ms: receivedAtMs - requestedAtMs,
      node_minus_page_performance_ms: (
        (requestedAtMs + receivedAtMs) / 2 - pagePerformanceMs
      ),
    });
  }
  return samples.sort((left, right) => left.round_trip_ms - right.round_trip_ms)[0];
}

async function installRecordingClockMarker(page) {
  return page.evaluate((marker) => new Promise((resolve) => {
    document.getElementById(marker.id)?.remove();
    const element = document.createElement('div');
    element.id = marker.id;
    Object.assign(element.style, {
      position: 'fixed',
      left: `${marker.x}px`,
      bottom: '0px',
      width: `${marker.size}px`,
      height: `${marker.size}px`,
      background: marker.color,
      border: '0',
      borderRadius: '0',
      boxShadow: 'none',
      opacity: '1',
      pointerEvents: 'none',
      zIndex: '2147483647',
    });
    document.documentElement.appendChild(element);
    requestAnimationFrame(() => requestAnimationFrame(() => resolve({
      installed: true,
      visible: Boolean(document.getElementById(marker.id)),
      page_performance_ms: performance.now(),
    })));
  }), RECORDING_CLOCK_MARKER);
}

async function removeRecordingClockMarker(page) {
  const pageClock = await samplePagePerformanceClock(page);
  const requestedAtMs = monotonicMs();
  const removal = await page.evaluate((markerId) => new Promise((resolve) => {
    requestAnimationFrame(() => {
      const element = document.getElementById(markerId);
      const removed = Boolean(element);
      element?.remove();
      const removedAtPerformanceMs = performance.now();
      requestAnimationFrame(() => resolve({
        removed,
        absent: !document.getElementById(markerId),
        removed_at_page_performance_ms: removedAtPerformanceMs,
        confirmed_at_page_performance_ms: performance.now(),
      }));
    });
  }), RECORDING_CLOCK_MARKER.id);
  const acknowledgedAtMs = monotonicMs();
  const removedAtMs = (
    removal.removed_at_page_performance_ms
    + pageClock.node_minus_page_performance_ms
  );
  return {
    ...removal,
    requested_at_ms: requestedAtMs,
    acknowledged_at_ms: acknowledgedAtMs,
    removed_at_ms: removedAtMs,
    page_clock_sample: pageClock,
  };
}

function tryJson(text) {
  if (typeof text !== 'string') return { raw: String(text) };
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

function redactSecretsForLog(value) {
  if (Array.isArray(value)) return value.map(redactSecretsForLog);
  if (typeof value === 'string') return redactUrlForLog(value);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => {
    if (/^(?:api[_-]?key|authorization|password|secret|access[_-]?token)$/i.test(key)) {
      return [key, item ? '[REDACTED]' : ''];
    }
    return [key, redactSecretsForLog(item)];
  }));
}

function appendJsonl(stream, event) {
  stream.write(`${JSON.stringify(redactSecretsForLog(event))}\n`);
}

function extractSessionIdFromUrl(url) {
  try {
    return new URL(url).searchParams.get('session_id') || '';
  } catch {
    return '';
  }
}

function buildWebSocketUrl(webUrl, sessionId) {
  const url = new URL(webUrl);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.pathname = '/ws';
  url.search = `?session_id=${encodeURIComponent(sessionId)}`;
  return url.toString();
}

function buildApiUrl(webUrl, pathname) {
  const url = new URL(webUrl);
  url.pathname = pathname;
  url.search = '';
  return url.toString();
}

function webAuth(args) {
  if (!args.webUsername) return null;
  return { username: args.webUsername, password: args.webPassword || '' };
}

function basicAuthHeader(auth) {
  if (!auth?.username) return {};
  const token = Buffer.from(`${auth.username}:${auth.password || ''}`).toString('base64');
  return { Authorization: `Basic ${token}` };
}

function redactUrlForLog(urlString) {
  try {
    const url = new URL(urlString);
    if (url.username) url.username = '[REDACTED]';
    if (url.password) url.password = '[REDACTED]';
    for (const key of [...url.searchParams.keys()]) {
      if (/^(?:api[_-]?key|authorization|password|secret|access[_-]?token|token)$/i.test(key)) {
        url.searchParams.set(key, '[REDACTED]');
      }
    }
    return url.toString();
  } catch {
    return String(urlString).replace(/(api[_-]?key=)[^&\s]+/gi, '$1[REDACTED]');
  }
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

function requestJson(urlString, {
  method = 'GET', payload = null, timeoutMs = 30000, auth = null, headers = {},
  tlsRejectUnauthorized = tlsRejectUnauthorizedFor(urlString),
} = {}) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlString);
    const body = payload == null ? null : JSON.stringify(payload);
    const transport = url.protocol === 'https:' ? https : http;
    const req = transport.request({
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: `${url.pathname}${url.search}`,
      method,
      rejectUnauthorized: tlsRejectUnauthorized,
      timeout: timeoutMs,
      headers: {
        ...basicAuthHeader(auth),
        ...headers,
        ...(body ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } : {}),
      },
    }, (res) => {
      let text = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => {
        text += chunk;
      });
      res.on('end', () => {
        const data = text ? tryJson(text) : {};
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(data);
        } else {
          const upstreamError = data?.error;
          const upstreamMessage = typeof upstreamError === 'string'
            ? upstreamError
            : upstreamError?.message;
          const upstreamCode = typeof upstreamError === 'object'
            ? (upstreamError?.code || upstreamError?.type || '')
            : '';
          const detail = upstreamMessage || text.slice(0, 500) || res.statusMessage || 'request failed';
          const error = new Error(
            `HTTP ${res.statusCode}${upstreamCode ? ` ${upstreamCode}` : ''}: ${detail}`,
          );
          error.httpStatus = res.statusCode;
          error.upstreamCode = upstreamCode;
          const retryAfter = Number(res.headers['retry-after']);
          error.retryAfterS = Number.isFinite(retryAfter) && retryAfter >= 0 ? retryAfter : null;
          reject(error);
        }
      });
    });
    req.on('timeout', () => {
      req.destroy(new Error(`Timed out ${method} ${redactUrlForLog(urlString)} after ${timeoutMs}ms`));
    });
    req.on('error', reject);
    if (body) req.write(body);
    req.end();
  });
}

async function preflightVlmModel(args) {
  if (!args.vlmModel) return { required: false, ok: true, status: 'not_configured' };
  if (!args.vlmPreflight) {
    return {
      required: false,
      ok: true,
      status: 'disabled',
      requested_model: args.vlmModel,
    };
  }

  const url = `${args.vlmApiBase.replace(/\/+$/, '')}/models`;
  const apiKey = String(args.vlmApiKey || '').replace(/^Bearer\s+/i, '');
  const checkedAt = new Date().toISOString();
  let payload = null;
  let lastError = null;
  for (let attempt = 1; attempt <= args.healthRetries; attempt += 1) {
    try {
      payload = await requestJson(url, {
        timeoutMs: 60000,
        headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
      });
      break;
    } catch (error) {
      lastError = error;
      if (attempt >= args.healthRetries) break;
      console.warn(
        `VLM preflight request ${attempt}/${args.healthRetries} failed: ${error.message}; `
        + `retrying in ${args.healthIntervalS}s`,
      );
      await sleep(args.healthIntervalS * 1000);
    }
  }
  if (!payload) {
    throw new Error(
      `VLM preflight failed for ${args.vlmModel} via ${args.vlmRoute} after `
      + `${args.healthRetries} request(s): ${lastError?.message || 'unknown error'}`,
    );
  }
  const modelRows = Array.isArray(payload?.models)
    ? payload.models
    : Array.isArray(payload?.data) ? payload.data : [];
  const availableModels = modelRows
    .map((item) => String(item?.id || '').trim()).filter(Boolean);
  if (!availableModels.includes(args.vlmModel)) {
    throw new Error(
      `VLM preflight rejected ${args.vlmModel}: model is absent from upstream /models for `
      + `${args.vlmApiBase}; available=${availableModels.join(', ') || '(none)'}`,
    );
  }
  return {
    required: true,
    ok: true,
    status: 'available',
    checked_at: checkedAt,
    requested_model: args.vlmModel,
    route: args.vlmRoute,
    api_base: args.vlmApiBase,
    available_model_ids: availableModels,
  };
}

function normalizedModelName(value) {
  return String(value || '').trim().toLowerCase();
}

function validateVlmBackendIdentity(args, responses) {
  if (!args.vlmModel) return { required: false, ok: true, status: 'not_configured' };
  const source = args.vlmRoute === 'joyai_adapter'
    ? 'response_payload.streamingharness.main_model'
    : 'response_payload.model';
  if (!args.vlmIdentityCheck) {
    return {
      required: false,
      ok: true,
      status: 'disabled',
      source,
      expected_models: args.vlmBackendAliases,
    };
  }

  const observed = responses.map((response) => (
    args.vlmRoute === 'joyai_adapter'
      ? response?.response_payload?.streamingharness?.main_model
      : response?.response_payload?.model
  )).map((value) => String(value || '').trim()).filter(Boolean);
  const uniqueObserved = [...new Set(observed)];
  const expected = new Set(args.vlmBackendAliases.map(normalizedModelName));
  const mismatches = uniqueObserved.filter((value) => !expected.has(normalizedModelName(value)));
  const missingCount = Math.max(0, responses.length - observed.length);
  const ok = uniqueObserved.length > 0 && mismatches.length === 0;
  return {
    required: true,
    ok,
    status: ok ? 'verified' : (uniqueObserved.length ? 'mismatch' : 'missing'),
    route: args.vlmRoute,
    source,
    expected_models: args.vlmBackendAliases,
    observed_models: uniqueObserved,
    observations: observed.length,
    responses_checked: responses.length,
    responses_without_identity: missingCount,
    mismatched_models: mismatches,
    error: ok
      ? undefined
      : (uniqueObserved.length
        ? `Backend identity mismatch: expected ${args.vlmBackendAliases.join(' | ')}, observed ${uniqueObserved.join(' | ')}`
        : `Backend identity missing at ${source} in ${responses.length} VLM response(s)`),
  };
}

function extractWarmupFrameDataUrl(sourceVideoPath, ffmpegBin, timeoutMs = 60000) {
  return new Promise((resolve, reject) => {
    const inputArgs = sourceVideoPath
      ? ['-ss', '0', '-i', sourceVideoPath]
      : ['-f', 'lavfi', '-i', 'color=c=gray:s=640x360:d=0.1'];
    const child = spawn(ffmpegBin, [
      '-v', 'error',
      ...inputArgs,
      '-frames:v', '1',
      '-vf', 'scale=640:-2',
      '-q:v', '5',
      '-f', 'image2pipe',
      '-vcodec', 'mjpeg',
      'pipe:1',
    ], { stdio: ['ignore', 'pipe', 'pipe'] });
    const chunks = [];
    let byteLength = 0;
    let stderr = '';
    let failure = null;
    let settled = false;
    const finish = (error, value = '') => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error);
      else resolve(value);
    };
    const timer = setTimeout(() => {
      failure = new Error(`Timed out extracting VLM warmup frame after ${timeoutMs}ms`);
      child.kill('SIGKILL');
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      byteLength += chunk.length;
      if (byteLength > 20 * 1024 * 1024) {
        failure = new Error('VLM warmup frame exceeded 20 MiB');
        child.kill('SIGKILL');
        return;
      }
      chunks.push(chunk);
    });
    child.stderr.on('data', (chunk) => {
      if (stderr.length < 8000) stderr += chunk.toString();
    });
    child.once('error', (error) => finish(error));
    child.once('close', (code) => {
      if (failure) {
        finish(failure);
        return;
      }
      if (code !== 0 || byteLength === 0) {
        finish(new Error(
          `Unable to extract VLM warmup frame (ffmpeg exit ${code}): ${stderr.trim().slice(-1000)}`,
        ));
        return;
      }
      finish(null, `data:image/jpeg;base64,${Buffer.concat(chunks).toString('base64')}`);
    });
  });
}

function createNeutralWarmupVideo(outputPath, ffmpegBin, timeoutMs = 60000) {
  return new Promise((resolve, reject) => {
    const temporaryPath = `${outputPath}.tmp-${process.pid}.mp4`;
    const child = spawn(ffmpegBin, [
      '-v', 'error',
      '-y',
      '-f', 'lavfi',
      '-i', 'color=c=gray:s=320x180:r=5:d=1',
      '-an',
      '-c:v', 'libx264',
      '-pix_fmt', 'yuv420p',
      '-movflags', '+faststart',
      temporaryPath,
    ], { stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    let settled = false;
    const finish = (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error);
      else resolve(outputPath);
    };
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(new Error(`Timed out creating neutral VLM warmup video after ${timeoutMs}ms`));
    }, timeoutMs);
    child.stderr.on('data', (chunk) => {
      if (stderr.length < 8000) stderr += chunk.toString();
    });
    child.once('error', (error) => finish(error));
    child.once('close', async (code) => {
      if (settled) return;
      if (code !== 0) {
        await fsp.rm(temporaryPath, { force: true }).catch(() => {});
        finish(new Error(
          `Unable to create neutral VLM warmup video (ffmpeg exit ${code}): ${stderr.trim().slice(-1000)}`,
        ));
        return;
      }
      try {
        await fsp.rename(temporaryPath, outputPath);
        finish(null);
      } catch (error) {
        finish(error);
      }
    });
  });
}

async function requestDirectVlmInferenceWarmup(args, imageDataUrl, timeoutMs) {
  const endpoint = `${args.vlmApiBase.replace(/\/+$/, '')}/chat/completions`;
  const apiKey = String(args.vlmApiKey || '').replace(/^Bearer\s+/i, '');
  const responsePayload = await requestJson(endpoint, {
    method: 'POST',
    timeoutMs,
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    payload: {
      model: args.vlmModel,
      messages: [{
        role: 'user',
        content: [{ type: 'image_url', image_url: { url: imageDataUrl } }],
      }],
      max_tokens: 32,
      temperature: 0,
      frame_time_range: '0.0 seconds',
    },
  });
  if (!responsePayload?.choices?.[0]?.message) {
    throw new Error('VLM warmup returned no assistant message');
  }
  const identity = validateVlmBackendIdentity(args, [{ response_payload: responsePayload }]);
  if (!identity.ok) {
    throw new Error(identity.error || 'VLM warmup backend identity verification failed');
  }
  return {
    strategy: 'direct_api',
    response_model: identity.observed_models?.[0] || responsePayload.model || '',
    backend_identity: identity,
  };
}

async function requestScaffoldProviderWarmup(args, imageDataUrl, timeoutMs) {
  const endpoint = `${args.vlmControlApiBase.replace(/\/+$/, '')}/warmup`;
  const apiKey = String(args.vlmApiKey || '').replace(/^Bearer\s+/i, '');
  const responsePayload = await requestJson(endpoint, {
    method: 'POST',
    timeoutMs,
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    payload: {
      model: args.vlmModel,
      image_url: imageDataUrl,
    },
  });
  if (
    responsePayload?.ok !== true
    || responsePayload?.multimodal_input !== true
    || responsePayload?.session_state_unchanged !== true
  ) {
    throw new Error('JoyAI scaffold provider warmup did not prove isolated multimodal readiness');
  }
  const responseModel = responsePayload.response_model || responsePayload.model || '';
  const identity = validateVlmBackendIdentity(args, [{
    response_payload: { model: responseModel },
  }]);
  if (!identity.ok) {
    throw new Error(identity.error || 'JoyAI scaffold provider warmup identity verification failed');
  }
  return {
    required: true,
    ok: true,
    strategy: 'stateless_multimodal_provider_request',
    endpoint_path: '/v1/warmup',
    response_model: identity.observed_models?.[0] || responseModel,
    response_id: responsePayload.response_id || '',
    latency_s: Number(responsePayload.latency_ms || 0) / 1000,
    multimodal_input: true,
    session_state_unchanged: true,
    backend_identity: identity,
  };
}

async function requestScaffoldQueryEvent(args, payload, timeoutMs = 30000) {
  const endpoint = `${args.vlmControlApiBase.replace(/\/+$/, '')}/query-events`;
  const apiKey = String(args.vlmApiKey || '').replace(/^Bearer\s+/i, '');
  const response = await requestJson(endpoint, {
    method: 'POST',
    timeoutMs,
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    payload,
  });
  if (
    response?.ok !== true
    || response?.query_event_id !== payload.query_event_id
    || !['queued', 'processing', 'delivered'].includes(String(response?.status || ''))
  ) {
    throw new Error('JoyAI scaffold did not acknowledge the Query-time frame event');
  }
  return response;
}

async function requestVlmSessionReset(args, sessionId, timeoutMs = 180000) {
  const endpoint = `${args.vlmControlApiBase.replace(/\/+$/, '')}/streaming/reset`;
  const apiKey = String(args.vlmApiKey || '').replace(/^Bearer\s+/i, '');
  const response = await requestJson(endpoint, {
    method: 'POST',
    timeoutMs,
    headers: {
      ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      'x-streaming-session': sessionId,
    },
    payload: { session_id: sessionId, user: sessionId },
  });
  if (response?.ok !== true || response?.reset_acknowledged !== true) {
    throw new Error('Native adapter did not acknowledge warmup-session reset');
  }
  return response;
}

async function requestAdapterVlmInferenceWarmup(args, imageDataUrl, timeoutMs) {
  const warmupSessionId = `capture-warmup-${crypto.randomUUID()}`;
  const wsUrl = buildWebSocketUrl(args.webUrl, warmupSessionId);
  const auth = webAuth(args);
  let ws = null;
  try {
    ws = await openNodeWebSocket(wsUrl, Math.min(timeoutMs, 30000), auth);
    return await new Promise((resolve, reject) => {
      let settled = false;
      let frameSent = false;
      const finish = (error, value = null) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (error) reject(error);
        else resolve(value);
      };
      const timer = setTimeout(() => {
        finish(new Error(`Timed out waiting for WebUI adapter warmup after ${timeoutMs}ms`));
      }, timeoutMs);

      ws.on('message', (data) => {
        const parsed = tryJson(data.toString());
        if (parsed.type === 'model_updated' && !frameSent) {
          const expectedApiBase = String(args.vlmApiBase || '').replace(/\/+$/, '');
          const observedApiBase = String(parsed.api_base || '').replace(/\/+$/, '');
          if (parsed.model !== args.vlmModel || observedApiBase !== expectedApiBase) return;
          frameSent = true;
          ws.send(JSON.stringify({
            type: 'set_debug',
            show_request_payload: false,
            show_response_payload: true,
            show_memory_state: false,
          }));
          ws.send(JSON.stringify({ type: 'update_prompt', prompt: '' }));
          ws.send(JSON.stringify({
            type: 'uploaded_video_frame',
            upload_id: warmupSessionId,
            frame_sequence: 1,
            media_time: 0,
            image: imageDataUrl,
          }));
          return;
        }
        if (parsed.type !== 'vlm_response' || !frameSent) return;
        const identity = validateVlmBackendIdentity(args, [{
          response_payload: parsed.response_payload,
        }]);
        if (!identity.ok) {
          finish(new Error(identity.error || 'WebUI adapter warmup identity verification failed'));
          return;
        }
        finish(null, {
          strategy: 'webui_adapter_session',
          response_model: identity.observed_models?.[0] || '',
          backend_identity: identity,
        });
      });
      ws.once('close', (code, reason) => {
        finish(new Error(
          `WebUI adapter warmup websocket closed before inference: code=${code}, reason=${reason?.toString?.() || ''}`,
        ));
      });
      ws.once('error', (error) => finish(error));
      ws.send(JSON.stringify({
        type: 'update_model',
        model: args.vlmModel,
        api_base: args.vlmApiBase,
        api_key: args.vlmApiKey,
      }));
    });
  } finally {
    if (ws) {
      try { ws.terminate(); } catch {}
    }
    await cleanupSessionApi(args.webUrl, warmupSessionId, Math.min(timeoutMs, 15000), auth)
      .catch(() => {});
  }
}

async function requestVlmInferenceWarmup(args, imageDataUrl, timeoutMs) {
  if (args.vlmRoute === 'joyai_adapter') {
    return requestAdapterVlmInferenceWarmup(args, imageDataUrl, timeoutMs);
  }
  return requestDirectVlmInferenceWarmup(args, imageDataUrl, timeoutMs);
}

async function uploadVideoApi(webUrl, sessionId, filePath, timeoutMs = 120000, auth = null) {
  const url = new URL(buildApiUrl(webUrl, '/api/video/upload'));
  const boundary = `----vl-interaction-${crypto.randomBytes(12).toString('hex')}`;
  const filename = path.basename(filePath);
  const stat = await fsp.stat(filePath);
  const fieldPart = Buffer.from(
    `--${boundary}\r\n`
    + 'Content-Disposition: form-data; name="session_id"\r\n\r\n'
    + `${sessionId}\r\n`,
  );
  const fileHeader = Buffer.from(
    `--${boundary}\r\n`
    + `Content-Disposition: form-data; name="file"; filename="${filename.replaceAll('"', '\\"')}"\r\n`
    + 'Content-Type: video/mp4\r\n\r\n',
  );
  const trailer = Buffer.from(`\r\n--${boundary}--\r\n`);
  const contentLength = fieldPart.length + fileHeader.length + stat.size + trailer.length;
  const transport = url.protocol === 'https:' ? https : http;

  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      fn(value);
    };
    const req = transport.request({
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: `${url.pathname}${url.search}`,
      method: 'POST',
      rejectUnauthorized: tlsRejectUnauthorizedFor(url.toString()),
      timeout: timeoutMs,
      headers: {
        ...basicAuthHeader(auth),
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        'Content-Length': contentLength,
      },
    }, (res) => {
      let text = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => {
        text += chunk;
      });
      res.on('end', () => {
        const data = text ? tryJson(text) : {};
        if (res.statusCode >= 200 && res.statusCode < 300) {
          finish(resolve, data);
        } else {
          finish(reject, new Error(data?.error || `HTTP ${res.statusCode}: ${text.slice(0, 500)}`));
        }
      });
    });
    req.on('timeout', () => {
      req.destroy(new Error(`Timed out POST ${url.toString()} after ${timeoutMs}ms`));
    });
    req.on('error', (error) => finish(reject, error));
    req.write(fieldPart);
    req.write(fileHeader);
    const stream = fs.createReadStream(filePath);
    stream.on('error', (error) => {
      req.destroy(error);
    });
    stream.on('end', () => {
      req.end(trailer);
    });
    stream.pipe(req, { end: false });
  });
}

function openNodeWebSocket(wsUrl, timeoutMs = 15000, auth = null) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl, {
      rejectUnauthorized: tlsRejectUnauthorizedFor(wsUrl),
      headers: basicAuthHeader(auth),
    });
    const timeout = setTimeout(() => {
      ws.close();
      reject(new Error(`Timed out opening auxiliary websocket: ${wsUrl}`));
    }, timeoutMs);
    ws.once('open', () => {
      clearTimeout(timeout);
      resolve(ws);
    });
    ws.once('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });
  });
}

async function waitForWebSocket(page, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const ready = await page.evaluate(() => {
      try {
        return Boolean(websocket && websocket.readyState === WebSocket.OPEN);
      } catch {
        return false;
      }
    });
    if (ready) return;
    await sleep(200);
  }
  throw new Error('Timed out waiting for WebUI websocket to open');
}

async function isPageWebSocketOpen(page) {
  return page.evaluate(() => {
    try {
      return Boolean(websocket && websocket.readyState === WebSocket.OPEN);
    } catch {
      return false;
    }
  }).catch(() => false);
}

async function sendWebSocketFromPage(page, payload) {
  await page.evaluate((message) => {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
      throw new Error('WebUI websocket is not open');
    }
    websocket.send(JSON.stringify(message));
  }, payload);
}

async function installFirstQueryFrameGate(page) {
  return page.evaluate(() => {
    const existing = window.__CAPTURE_FIRST_QUERY_FRAME_GATE;
    if (existing?.active) {
      return {
        installed: true,
        active: true,
        blocked_frame_count: existing.blockedFrameCount || 0,
      };
    }

    const originalSend = WebSocket.prototype.send;
    const gate = {
      active: true,
      originalSend,
      wrapper: null,
      pending: null,
      lockedPending: null,
      blockedFrameCount: 0,
      firstBlockedMediaTimeS: null,
      lastBlockedMediaTimeS: null,
      installedAtPerformanceMs: performance.now(),
    };
    const wrapper = function captureFirstQueryFrameGate(data) {
      if (gate.active && typeof data === 'string') {
        try {
          const message = JSON.parse(data);
          if (message?.type === 'uploaded_video_frame') {
            gate.blockedFrameCount += 1;
            const mediaTimeS = Number(message.media_time);
            if (gate.firstBlockedMediaTimeS == null && Number.isFinite(mediaTimeS)) {
              gate.firstBlockedMediaTimeS = mediaTimeS;
            }
            if (Number.isFinite(mediaTimeS)) gate.lastBlockedMediaTimeS = mediaTimeS;
            gate.pending = { socket: this, message };
            return undefined;
          }
        } catch {
          // Non-JSON WebSocket traffic is passed through unchanged.
        }
      }
      return originalSend.call(this, data);
    };
    gate.wrapper = wrapper;
    WebSocket.prototype.send = wrapper;
    window.__CAPTURE_FIRST_QUERY_FRAME_GATE = gate;
    return {
      installed: true,
      active: true,
      blocked_frame_count: 0,
      installed_at_performance_ms: gate.installedAtPerformanceMs,
    };
  });
}

async function releaseFirstQueryFrameGate(page) {
  return page.evaluate(() => {
    const gate = window.__CAPTURE_FIRST_QUERY_FRAME_GATE;
    if (!gate?.active) {
      return { installed: Boolean(gate), released: false, reason: 'not_active' };
    }

    const selected = gate.lockedPending || gate.pending;
    const socket = selected?.socket || websocket;
    const message = selected?.message || null;
    if (!message) {
      gate.active = false;
      if (WebSocket.prototype.send === gate.wrapper) {
        WebSocket.prototype.send = gate.originalSend;
      }
      return {
        installed: true,
        released: false,
        disarmed_without_frame: true,
        reason: 'busy_scheduler_had_no_pending_frame',
        blocked_frame_count: gate.blockedFrameCount,
        first_blocked_media_time_s: gate.firstBlockedMediaTimeS,
        last_blocked_media_time_s: gate.lastBlockedMediaTimeS,
        released_at_performance_ms: performance.now(),
      };
    }
    if (
      message?.type !== 'uploaded_video_frame'
      || !String(message.image || '').startsWith('data:image/')
      || !socket
      || socket.readyState !== WebSocket.OPEN
    ) {
      throw new Error('Unable to release a query-aligned uploaded-video frame');
    }

    gate.active = false;
    if (WebSocket.prototype.send === gate.wrapper) {
      WebSocket.prototype.send = gate.originalSend;
    }
    gate.originalSend.call(socket, JSON.stringify(message));
    const result = {
      installed: true,
      released: true,
      blocked_frame_count: gate.blockedFrameCount,
      first_blocked_media_time_s: gate.firstBlockedMediaTimeS,
      last_blocked_media_time_s: gate.lastBlockedMediaTimeS,
      released_media_time_s: Number(message.media_time) || 0,
      released_frame_sequence: Number(message.frame_sequence) || 0,
      released_at_performance_ms: performance.now(),
    };
    gate.pending = null;
    gate.lockedPending = null;
    return result;
  });
}

async function captureDisplayedQueryFrameAtDispatch(page) {
  return page.evaluate(async () => {
    if (
      !videoElement
      || videoElement.readyState < 2
      || videoElement.videoWidth <= 0
      || videoElement.videoHeight <= 0
    ) {
      throw new Error('Displayed video frame is unavailable at Query time');
    }
    const gate = window.__CAPTURE_FIRST_QUERY_FRAME_GATE;
    if (!gate?.active) throw new Error('Query-time uploaded-video frame gate is not active');
    const dispatchMediaTimeS = Number(videoElement.currentTime) || 0;
    const canvas = document.createElement('canvas');
    canvas.width = Number(videoElement.videoWidth);
    canvas.height = Number(videoElement.videoHeight);
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Unable to create Query-time video canvas');
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    const imageUrl = canvas.toDataURL('image/jpeg', 0.9);
    if (!imageUrl.startsWith('data:image/jpeg;base64,')) {
      throw new Error('Unable to encode the Query-time displayed frame as JPEG');
    }
    return {
      image_url: imageUrl,
      media_time_s: dispatchMediaTimeS,
      dispatch_media_time_s: dispatchMediaTimeS,
      width: canvas.width,
      height: canvas.height,
      source_width: Number(videoElement.videoWidth),
      source_height: Number(videoElement.videoHeight),
      capture_surface: 'displayed_video_frame_at_query_dispatch',
      captured_at_performance_ms: performance.now(),
      frame_sequence: Number(window.uploadedVideoFrameSequence) || 0,
    };
  });
}

async function discardFirstQueryFrameGate(page) {
  return page.evaluate(() => {
    const gate = window.__CAPTURE_FIRST_QUERY_FRAME_GATE;
    if (!gate) return false;
    gate.active = false;
    gate.pending = null;
    gate.lockedPending = null;
    if (WebSocket.prototype.send === gate.wrapper) {
      WebSocket.prototype.send = gate.originalSend;
    }
    return true;
  }).catch(() => false);
}

async function resetUploadedWarmupPreserveSession(page) {
  await page.evaluate(() => {
    streamStartToken += 1;
    isAnalysisRunning = false;
    uploadedFirstInferenceSeen = false;
    if (typeof stopUploadedVideoCapture === 'function') stopUploadedVideoCapture();
    videoElement.pause();
    videoElement.srcObject = null;
    videoElement.removeAttribute('src');
    videoElement.loop = false;
    videoElement.load();
    activeStreamSource = null;
    uploadedVideo = null;
    uploadingVideo = false;
    uploadVideoPromise = null;
    if (typeof setStreamLifecycleState === 'function') setStreamLifecycleState('idle');
    if (typeof resetVideoButtons === 'function') resetVideoButtons();
    if (typeof setVideoWaitingForStream === 'function') setVideoWaitingForStream(false);
    if (typeof clearVlmConversation === 'function') clearVlmConversation();
    if (typeof hideTopProgress === 'function') hideTopProgress();
    if (typeof updatePromptAvailability === 'function') updatePromptAvailability();
    if (typeof updateStatus === 'function') updateStatus('Connected', 'connected');
  });
}

async function setVlmControlsInPage(page, config) {
  await page.evaluate(({ model, apiBase, apiKey }) => {
    const apiBaseInput = document.getElementById('apiBaseUrl');
    const apiKeyInput = document.getElementById('apiKey');
    const modelInput = document.getElementById('modelSelect');
    if (!apiBaseInput || !apiKeyInput || !modelInput) {
      throw new Error('WebUI is missing VLM configuration controls');
    }
    apiBaseInput.value = apiBase;
    apiKeyInput.value = apiKey;
    if (![...modelInput.options].some((option) => option.value === model)) {
      modelInput.add(new Option(model, model));
    }
    modelInput.value = model;
    const modelName = document.getElementById('modelName');
    if (modelName) modelName.textContent = model;
    if (typeof checkApiKeyRequirement === 'function') {
      checkApiKeyRequirement(apiBase);
    }
  }, config);
}

async function preparePromptInPage(page, prompt) {
  return page.evaluate((text) => {
    const input = document.getElementById('promptText');
    const clean = typeof cleanBackgroundQuestionText === 'function'
      ? cleanBackgroundQuestionText(String(text || '').trim())
      : String(text || '').trim();

    if (input) input.value = clean;
    try {
      currentPromptText = clean;
    } catch {
      // Older/custom pages may not expose this variable.
    }
    if (clean && typeof appendPromptHistoryEntry === 'function') {
      try {
        appendPromptHistoryEntry(clean);
      } catch {
        // UI history is best-effort; the WebSocket prompt is authoritative.
      }
    }
    if (input) input.value = '';
    if (typeof resizePromptInput === 'function') resizePromptInput();
    return clean;
  }, prompt);
}

async function sendPromptFromPage(page, prompt) {
  const clean = await preparePromptInPage(page, prompt);
  await page.evaluate((message) => {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
      throw new Error('WebUI websocket is not open');
    }
    websocket.send(JSON.stringify(message));
  }, { type: 'update_prompt', prompt: clean });
  return clean;
}

async function selectRtspSource(page, videoUrl) {
  await page.evaluate((url) => {
    document.querySelectorAll('.input-source-tab').forEach((tab) => {
      tab.classList.toggle('active', tab.getAttribute('data-source') === 'rtsp');
    });

    const webcamControls = document.getElementById('webcamControls');
    const rtspControls = document.getElementById('rtspControls');
    const rtspBetaWarning = document.getElementById('rtspBetaWarning');
    if (webcamControls) webcamControls.style.display = 'none';
    if (rtspControls) rtspControls.style.display = 'block';
    if (rtspBetaWarning) rtspBetaWarning.style.display = 'flex';

    const input = document.getElementById('rtspUrl');
    if (!input) throw new Error('Missing #rtspUrl');
    input.disabled = false;
    input.value = String(url || '');
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, videoUrl);
}

function getTaskInputMode(args, task) {
  if (task.local_video_path && args.localVideoMode === 'upload') return 'upload';
  return 'rtsp';
}

async function selectUploadSource(page) {
  await page.evaluate(() => {
    document.querySelectorAll('.input-source-tab').forEach((tab) => {
      tab.classList.toggle('active', tab.getAttribute('data-source') === 'upload');
    });

    const webcamControls = document.getElementById('webcamControls');
    const rtspControls = document.getElementById('rtspControls');
    const videoFileControls = document.getElementById('videoFileControls');
    const rtspBetaWarning = document.getElementById('rtspBetaWarning');
    const quickCameraBtn = document.getElementById('quickCameraBtn');
    if (webcamControls) webcamControls.style.display = 'none';
    if (rtspControls) rtspControls.style.display = 'none';
    if (videoFileControls) videoFileControls.style.display = 'block';
    if (rtspBetaWarning) rtspBetaWarning.style.display = 'none';
    if (quickCameraBtn) quickCameraBtn.style.display = 'none';
    if (typeof syncInputSourceControls === 'function') {
      try {
        syncInputSourceControls('upload');
      } catch {
        // Direct DOM state above is sufficient for the known WebUI.
      }
    }
  });
}

async function prepareUploadedVideoSource(page, localVideoPath) {
  await selectUploadSource(page);
  await page.waitForSelector('#videoFileInput', { state: 'attached' });
  await page.setInputFiles('#videoFileInput', localVideoPath);
  await page.waitForFunction(() => {
    const input = document.getElementById('videoFileInput');
    return Boolean(input?.files?.length);
  }, null, { timeout: 30000 });
}

async function startUploadedVideoFromPage(page, timeoutMs = 60000) {
  await page.evaluate(() => {
    window.__CAPTURE_UPLOADED_START_ERROR = '';
    window.__CAPTURE_UPLOADED_START_TRIGGERED_AT = performance.now();
    Promise.resolve(start()).catch((error) => {
      window.__CAPTURE_UPLOADED_START_ERROR = error?.stack || error?.message || String(error);
      console.error('[capture] uploaded-video start failed:', error);
    });
  });
  await page.waitForFunction(() => {
    if (window.__CAPTURE_UPLOADED_START_ERROR) {
      throw new Error(window.__CAPTURE_UPLOADED_START_ERROR);
    }
    return activeStreamSource === 'upload' && streamLifecycleState !== 'idle';
  }, null, { timeout: timeoutMs });
}

async function stopPageFromPage(page, timeoutMs = 60000) {
  let timedOut = false;
  await Promise.race([
    page.evaluate(async () => {
      try {
        if (typeof window.__CAPTURE_MANUAL_STOP === 'function') {
          await window.__CAPTURE_MANUAL_STOP({ clearConversation: false });
        } else if (typeof stop === 'function') {
          await stop({ clearConversation: false });
        }
      } catch {
        // Ignore cleanup errors in the page; server cleanup follows.
      }
    }),
    sleep(timeoutMs).then(() => {
      timedOut = true;
    }),
  ]);
  if (timedOut) {
    throw new Error(`Timed out waiting for page stop() after ${timeoutMs}ms`);
  }
}

async function uploadSelectedVideoFromPage(page, timeoutMs = 120000) {
  let timedOut = false;
  const result = await Promise.race([
    page.evaluate(async () => {
      if (typeof uploadSelectedVideo !== 'function') {
        throw new Error('WebUI does not expose uploadSelectedVideo()');
      }
      return uploadSelectedVideo({ keepProgressVisible: true });
    }),
    sleep(timeoutMs).then(() => {
      timedOut = true;
      return null;
    }),
  ]);
  if (timedOut) {
    throw new Error(`Timed out waiting for uploadSelectedVideo() after ${timeoutMs}ms`);
  }
  return result;
}

async function setUploadedVideoInfoFromApi(page, uploadResponse, localVideoPath) {
  const stat = await fsp.stat(localVideoPath);
  await page.evaluate(({ data, fallbackName, fallbackSize, fileKey }) => {
    const filename = data.filename || fallbackName;
    const sizeBytes = data.size_bytes || fallbackSize;
    const previewUrl = uploadedVideoPreviewUrl || '';
    uploadedVideo = {
      uploadId: data.upload_id || '',
      filename,
      sizeBytes,
      previewUrl,
      fileKey,
    };
    uploadingVideo = false;
    uploadVideoPromise = null;
    if (typeof setVideoUploadStatus === 'function') {
      setVideoUploadStatus(`${filename} ready (${typeof formatFileSize === 'function' ? formatFileSize(sizeBytes) : `${sizeBytes} bytes`})`, 'ok');
    }
    if (typeof updateStatus === 'function') {
      updateStatus('Video ready', 'connected');
    }
    if (typeof setTopProgress === 'function') {
      setTopProgress('上传完成，正在转换为实时流...', { indeterminate: true });
    }
    if (typeof syncVideoUploadControls === 'function') {
      syncVideoUploadControls();
    }
  }, {
    data: uploadResponse,
    fallbackName: path.basename(localVideoPath),
    fallbackSize: stat.size,
    fileKey: `${path.basename(localVideoPath)}:${stat.size}:${Math.floor(stat.mtimeMs)}`,
  });
}

async function startUploadedVideoApiFromPage(page, sessionId, uploadId, timeoutMs = 60000) {
  return page.evaluate(async ({ sid, uid, timeout }) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch('/api/video/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, upload_id: uid }),
        signal: controller.signal,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || `Failed to start uploaded video (${response.status})`);
      }
      return data;
    } finally {
      clearTimeout(timer);
    }
  }, { sid: sessionId, uid: uploadId, timeout: timeoutMs });
}

async function startUploadedVideoApi(webUrl, sessionId, uploadId, timeoutMs = 60000, auth = null) {
  return requestJson(buildApiUrl(webUrl, '/api/video/start'), {
    method: 'POST',
    payload: { session_id: sessionId, upload_id: uploadId },
    timeoutMs,
    auth,
  });
}

async function stopUploadedVideoApi(webUrl, sessionId, timeoutMs = 15000, auth = null) {
  return requestJson(buildApiUrl(webUrl, '/api/video/stop'), {
    method: 'POST',
    payload: { session_id: sessionId },
    timeoutMs,
    auth,
  });
}

async function cleanupSessionApi(webUrl, sessionId, timeoutMs = 10000, auth = null) {
  return requestJson(buildApiUrl(webUrl, '/api/session/cleanup'), {
    method: 'POST',
    payload: { session_id: sessionId, reset_adapter: true },
    timeoutMs,
    auth,
  });
}

async function getRtspStatusFromPage(page) {
  return page.evaluate(async () => {
    const response = await fetch('/api/rtsp/status');
    return response.json();
  });
}

async function getRtspStatus(webUrl, timeoutMs = 15000, auth = null) {
  return requestJson(buildApiUrl(webUrl, '/api/rtsp/status'), { timeoutMs, auth });
}

async function stopActiveRtspStreams(
  webUrl,
  events,
  startedAtMs,
  task,
  reason,
  timeoutMs = 15000,
  auth = null,
) {
  let status = null;
  try {
    status = await getRtspStatus(webUrl, timeoutMs, auth);
  } catch (error) {
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'capture_active_stream_cleanup_status_failed',
      task_id: task.id,
      reason,
      error: error.message,
    });
    return [];
  }

  const streams = Array.isArray(status?.streams) ? status.streams : [];
  appendJsonl(events, {
    t_ms: monotonicMs() - startedAtMs,
    type: 'capture_active_stream_cleanup_status',
    task_id: task.id,
    reason,
    active_stream_count: streams.length,
    streams,
  });

  const results = [];
  for (const stream of streams) {
    const sid = stream?.session_id;
    if (!sid) continue;
    try {
      const stopResponse = await stopUploadedVideoApi(webUrl, sid, timeoutMs, auth);
      const result = { session_id: sid, status: 'stopped', stop_response: stopResponse };
      results.push(result);
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_active_stream_cleanup_stop_response',
        task_id: task.id,
        reason,
        session_id: sid,
        stop_response: stopResponse,
      });
    } catch (error) {
      const result = { session_id: sid, status: 'error', error: error.message };
      results.push(result);
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_active_stream_cleanup_stop_failed',
        task_id: task.id,
        reason,
        session_id: sid,
        error: error.message,
      });
    }
  }
  return results;
}

function findRtspStatusStream(status, sessionId) {
  const streams = Array.isArray(status?.streams) ? status.streams : [];
  return streams.find((stream) => stream.session_id === sessionId) || null;
}

async function waitForUploadedAnalysisStatusFromApi(webUrl, sessionId, timeoutMs = 60000) {
  const deadline = monotonicMs() + timeoutMs;
  let lastStatus = null;
  while (monotonicMs() < deadline) {
    try {
      const status = await getRtspStatus(webUrl, 15000);
      const stream = findRtspStatusStream(status, sessionId);
      if (stream) {
        lastStatus = stream;
        if (stream.preroll?.analysis_started || stream.connected && !stream.preroll?.enabled) {
          return { stream, status, readyMs: monotonicMs() };
        }
      }
    } catch {
      // Keep polling until the caller's timeout expires.
    }
    await sleep(1000);
  }
  throw new Error(`Timed out waiting for uploaded video analysis status for session ${sessionId}; last=${JSON.stringify(lastStatus)}`);
}

async function waitForUploadedAnalysisStatus(page, sessionId, timeoutMs = 60000) {
  const deadline = monotonicMs() + timeoutMs;
  let lastStatus = null;
  while (monotonicMs() < deadline) {
    try {
      const status = await getRtspStatusFromPage(page);
      const stream = findRtspStatusStream(status, sessionId);
      if (stream) {
        lastStatus = stream;
        if (stream.preroll?.analysis_started || stream.connected && !stream.preroll?.enabled) {
          return { stream, status, readyMs: monotonicMs() };
        }
      }
    } catch {
      // Keep polling until the caller's timeout expires.
    }
    await sleep(1000);
  }
  throw new Error(`Timed out waiting for uploaded video analysis status for session ${sessionId}; last=${JSON.stringify(lastStatus)}`);
}

async function showUploadedLocalPreviewFromPage(page) {
  await page.evaluate(async () => {
    try {
      activeStreamSource = 'upload';
      isAnalysisRunning = true;
      uploadedPrerollActive = false;
      uploadedFirstInferenceSeen = false;
      updatePromptAvailability();
      const bigBtn = document.getElementById('bigStartBtn');
      if (bigBtn) {
        bigBtn.classList.remove('animating');
        bigBtn.classList.add('streaming');
      }
      document.getElementById('smallStopBtn')?.classList.add('show');
      if (uploadedVideo?.previewUrl && videoElement) {
        videoElement.srcObject = null;
        videoElement.src = uploadedVideo.previewUrl;
        videoElement.loop = true;
        videoElement.muted = true;
        try {
          videoElement.currentTime = 0;
        } catch {
          // Best effort.
        }
        await videoElement.play().catch(() => {});
      }
      if (typeof updateStatus === 'function') {
        updateStatus('Analyzing real video...', 'processing');
      }
    } catch (error) {
      console.warn('[capture] show uploaded local preview failed:', error);
    }
  });
}

async function waitForUploadedRealVideoStart(page, timeoutMs = 60000) {
  await page.waitForFunction(() => {
    try {
      if (!isAnalysisRunning || activeStreamSource !== 'upload') return false;
      const prerollSeconds = Number(uploadedPrerollSeconds) || 0;
      if (prerollSeconds <= 0) return true;
      return uploadedPrerollActive === false;
    } catch {
      return false;
    }
  }, null, { timeout: timeoutMs });
}

async function getUploadedVideoInfo(page) {
  return page.evaluate(() => {
    try {
      if (!uploadedVideo) return null;
      return {
        upload_id: uploadedVideo.uploadId || '',
        filename: uploadedVideo.filename || '',
        size_bytes: uploadedVideo.sizeBytes || 0,
        has_preview_url: Boolean(uploadedVideo.previewUrl),
      };
    } catch {
      return null;
    }
  });
}

async function getPageSessionId(page) {
  return page.evaluate(() => {
    try {
      return String(sessionId || '');
    } catch {
      return String(window.__CAPTURE_SESSION_ID || '');
    }
  });
}

async function waitForStreamingReady(page, timeoutMs = 60000) {
  await page.waitForFunction(() => {
    try {
      return Boolean(isAnalysisRunning);
    } catch {
      return false;
    }
  }, null, { timeout: timeoutMs });
}

async function getVideoPlaybackState(page) {
  return page.evaluate(() => ({
    active_source: String(typeof activeStreamSource === 'undefined' ? '' : activeStreamSource || ''),
    lifecycle: String(typeof streamLifecycleState === 'undefined' ? '' : streamLifecycleState || ''),
    analysis_running: Boolean(typeof isAnalysisRunning !== 'undefined' && isAnalysisRunning),
    paused: Boolean(videoElement?.paused),
    ended: Boolean(videoElement?.ended),
    ready_state: Number(videoElement?.readyState) || 0,
    current_time_s: Number(videoElement?.currentTime),
    duration_s: Number(videoElement?.duration),
    video_width: Number(videoElement?.videoWidth) || 0,
    video_height: Number(videoElement?.videoHeight) || 0,
    frame_sequence: Number(typeof uploadedFrameSequence === 'undefined' ? 0 : uploadedFrameSequence) || 0,
  }));
}

async function waitForVideoPlaybackReady(page, inputMode, timeoutMs = 60000) {
  await page.waitForFunction((mode) => {
    try {
      const uploadReady = mode !== 'upload' || (
        activeStreamSource === 'upload'
        && streamLifecycleState === 'running'
        && Number(uploadedFrameSequence) > 0
      );
      return Boolean(
        isAnalysisRunning
        && uploadReady
        && videoElement
        && videoElement.readyState >= 2
        && videoElement.paused === false
        && videoElement.ended === false
        && videoElement.videoWidth > 0
        && videoElement.videoHeight > 0
        && Number.isFinite(videoElement.currentTime)
      );
    } catch {
      return false;
    }
  }, inputMode, { timeout: timeoutMs });

  await page.evaluate(async (timeout) => {
    if (!videoElement || typeof videoElement.requestVideoFrameCallback !== 'function') return;
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Timed out waiting for a displayed video frame')), timeout);
      videoElement.requestVideoFrameCallback(() => {
        clearTimeout(timer);
        resolve();
      });
    });
  }, Math.min(timeoutMs, 10000));
  return getVideoPlaybackState(page);
}

async function waitForUploadedSinglePassPlaybackStart(page, timeoutMs = 60000) {
  await page.waitForFunction(() => {
    try {
      return Boolean(
        activeStreamSource === 'upload'
        && streamLifecycleState !== 'idle'
        && videoElement
        && videoElement.readyState >= 2
        && videoElement.paused === false
        && videoElement.ended === false
        && videoElement.videoWidth > 0
        && videoElement.videoHeight > 0
        && Number.isFinite(videoElement.currentTime)
      );
    } catch {
      return false;
    }
  }, null, { timeout: timeoutMs });

  await page.evaluate(async (timeout) => {
    if (!videoElement || typeof videoElement.requestVideoFrameCallback !== 'function') return;
    await new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('Timed out waiting for the first single-pass displayed frame')),
        timeout,
      );
      videoElement.requestVideoFrameCallback(() => {
        clearTimeout(timer);
        resolve();
      });
    });
  }, Math.min(timeoutMs, 10000));
  return getVideoPlaybackState(page);
}

async function installUploadedPlaybackContinuityGuard(page) {
  return page.evaluate(() => {
    const video = videoElement;
    if (!video) throw new Error('Uploaded playback video element is unavailable');
    if (window.__CAPTURE_UPLOAD_CONTINUITY_GUARD?.installed) {
      window.__CAPTURE_UPLOAD_CONTINUITY_GUARD.active = true;
      return { ...window.__CAPTURE_UPLOAD_CONTINUITY_GUARD, onPause: undefined };
    }
    const guard = {
      installed: true,
      active: true,
      pause_events: 0,
      resume_attempts: 0,
      resume_successes: 0,
      resume_failures: 0,
      last_pause_media_time_s: null,
      last_resume_media_time_s: null,
      onPause: null,
    };
    guard.onPause = () => {
      if (
        !guard.active
        || activeStreamSource !== 'upload'
        || streamLifecycleState !== 'running'
        || !isAnalysisRunning
        || video.ended
      ) return;
      guard.pause_events += 1;
      guard.last_pause_media_time_s = Number(video.currentTime) || 0;
    };
    video.addEventListener('pause', guard.onPause, true);
    window.__CAPTURE_UPLOAD_CONTINUITY_GUARD = guard;
    return { ...guard, onPause: undefined };
  });
}

async function disableUploadedPlaybackContinuityGuard(page) {
  return page.evaluate(() => {
    const guard = window.__CAPTURE_UPLOAD_CONTINUITY_GUARD;
    if (!guard) return null;
    guard.active = false;
    return { ...guard, onPause: undefined };
  });
}

async function installCapturePageHooks(page) {
  await page.evaluate(() => {
    if (window.__CAPTURE_HOOKS_INSTALLED) return;
    window.__CAPTURE_HOOKS_INSTALLED = true;
    window.__CAPTURE_ALLOW_STOP = false;

    const originalConnectWebSocket = window.connectWebSocket;
    window.__CAPTURE_ORIGINAL_CONNECT_WEBSOCKET = originalConnectWebSocket;
    window.__CAPTURE_DISABLE_PAGE_WS = false;
    if (typeof originalConnectWebSocket === 'function') {
      window.connectWebSocket = function captureConnectWebSocketWrapper(...args) {
        if (window.__CAPTURE_DISABLE_PAGE_WS) {
          console.log('[capture] suppressed page websocket reconnect');
          return;
        }
        return originalConnectWebSocket.apply(this, args);
      };
    }
    window.__CAPTURE_DISABLE_PAGE_WS_NOW = () => {
      window.__CAPTURE_DISABLE_PAGE_WS = true;
      try {
        if (websocket) {
          websocket.onclose = null;
          websocket.close();
          websocket = null;
        }
      } catch (error) {
        console.warn('[capture] disable page websocket failed:', error);
      }
    };

    const originalStop = window.stop;
    window.__CAPTURE_ORIGINAL_STOP = originalStop;
    window.stop = async function captureStopWrapper(...stopArgs) {
      if (!window.__CAPTURE_ALLOW_STOP) {
        console.log('[capture] suppressed page stop()');
        return;
      }
      return originalStop.apply(this, stopArgs);
    };

    window.__CAPTURE_MANUAL_STOP = async (options) => {
      window.__CAPTURE_ALLOW_STOP = true;
      try {
        return await originalStop.call(window, options || {});
      } finally {
        window.__CAPTURE_ALLOW_STOP = false;
      }
    };

    window.__CAPTURE_CLOSE_PEER_ONLY = () => {
      try {
        if (peerConnection) {
          peerConnection.oniceconnectionstatechange = null;
          peerConnection.close();
          peerConnection = null;
        }
      } catch (error) {
        console.warn('[capture] close peer failed:', error);
      }
      try {
        isAnalysisRunning = false;
        isStreamStarting = false;
        setVideoWaitingForStream(false);
        updatePromptAvailability();
      } catch (error) {
        console.warn('[capture] reset local streaming flags failed:', error);
      }
    };
  });
}

async function startRtspFromPage(page) {
  await page.evaluate(() => {
    if (typeof start !== 'function') {
      throw new Error('WebUI does not expose start()');
    }
    try {
      const result = start();
      if (result && typeof result.catch === 'function') {
        result.catch((error) => {
          console.error('[capture] async RTSP start() failed:', error);
        });
      }
    } catch (error) {
      console.error('[capture] RTSP start() failed:', error);
      throw error;
    }
  });
}

async function runTask(browser, args, task, index, total, attempt = 1, maxAttempts = 1) {
  const baseTaskName = `${String(index + 1).padStart(3, '0')}_${safeName(task.id)}`;
  const taskDirName = args.keepFailedAttempts && attempt > 1 ? `${baseTaskName}_retry${attempt}` : baseTaskName;
  const taskDir = path.resolve(args.out, taskDirName);
  const summaryPath = path.join(taskDir, 'summary.json');
  if (args.skipExisting && fs.existsSync(summaryPath)) {
    const existing = readJsonIfExists(summaryPath);
    const categorySemanticsReusable = task.category !== '智能体委托'
      || existing?.task?.category === task.category;
    if (
      existing?.status === 'ok'
      && existingSummaryMatchesVlmSelection(existing, args)
      && categorySemanticsReusable
    ) {
      console.log(`[${index + 1}/${total}] ${task.id}: skip existing ok summary`);
      return { status: 'skipped', taskDir, summaryPath };
    }
    console.log(`[${index + 1}/${total}] ${task.id}: existing summary is not reusable for this model; rerun`);
  }

  if (!args.keepFailedAttempts && fs.existsSync(taskDir)) {
    console.log(`[${index + 1}/${total}] ${task.id}: remove incomplete attempt artifacts before rerun`);
    await fsp.rm(taskDir, { recursive: true, force: true });
  }
  await ensureDir(taskDir);
  const eventsPath = path.join(taskDir, 'events.jsonl');
  const events = fs.createWriteStream(eventsPath, { flags: 'w' });
  const startedAtIso = new Date().toISOString();
  const startedAtMs = monotonicMs();
  const durationS = task.duration_s ?? args.defaultDurationS;
  const processIntervalS = task.process_interval_s ?? args.processIntervalS;
  const framesPerBatch = task.frames_per_batch ?? args.framesPerBatch;
  const sessionId = `capture-${safeName(task.id)}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const auth = webAuth(args);
  const vlmResponses = [];
  const vlmResponseKeys = new Set();
  const protocolViolations = [];
  const backgroundTasks = new Map();
  const promptEvents = [];
  const modelUpdateEvents = [];
  let enforceVlmSelection = null;
  let vlmSelectionGuardActive = false;
  let vlmSelectionViolation = null;
  let providerInfrastructureFailure = null;
  let observedSessionId = '';
  let browserVideoPath = '';
  let mp4Path = '';
  let uiMp4Conversion = null;
  let taskMp4Path = '';
  let taskVideoRender = null;
  let videoQuality = null;
  let sourceVideoValidation = null;
  let recordingClockMarkerInstall = null;
  let recordingClockMarkerRemoval = null;
  let browserRecordingClock = null;
  let taskVideoStartWebmPtsS = null;
  let streamReadyOffsetS = null;
  let originalVideoStartOffsetS = null;
  let querySentOffsetS = null;
  let queryVideoTimeS = null;
  let postQueryVlmResponseCount = 0;
  let promptedVlmResponseCount = 0;
  let substantiveVlmResponseCount = 0;
  let completedVlmResponseCount = 0;
  let deferredVlmResponseCount = 0;
  let queryEchoResponseCount = 0;
  let firstPromptedResponseFrameTimeS = null;
  let promptedResponseFrameDriftS = null;
  let vlmBackendIdentity = null;
  const vlmInferenceWarmups = [];
  const playbackStartRequests = [];
  let vlmWarmupFramePromise = null;
  let latestVlmWarmupReadyAtMs = args.vlmModel && args.vlmWarmup ? null : startedAtMs;
  let targetUploadStartedAtMs = null;
  let targetUploadReadyAtMs = null;
  let targetUploadVerification = null;
  let officialPlaybackRequestedAtMs = null;
  let officialPlaybackStartedAtMs = null;
  let officialPlaybackFirstFrameAtMs = null;
  let officialPlaybackAnalyzerReadyAtMs = null;
  let targetPlaybackWasStarted = false;
  let status = 'ok';
  let errorText = '';
  let effectiveVideoUrl = task.video_url;
  const inputMode = getTaskInputMode(args, task);
  const providerWarmupRequired = (
    args.vlmProfileSnapshot?.interaction_scaffold === JOYAI_OFFICIAL_SCAFFOLD
    || args.vlmProfileSnapshot?.input_transport === 'native-video-realtime'
  );
  const nativeSessionResetRequired = (
    args.vlmProfileSnapshot?.input_transport === 'native-video-realtime'
  );
  let localPublisher = null;
  let localRelayInfo = null;
  let localUploadInfo = null;
  let uploadSourcePath = '';
  let uploadedVideoUploadResponse = null;
  let uploadedVideoStartResponse = null;
  let uploadedVideoStartResponseAtMs = null;
  let uploadedStatusReadyInfo = null;
  let uploadedPrerollStartedAtMs = null;
  let uploadedPrerollSeconds = null;
  let uploadedScheduledRealVideoStartAtMs = null;
  let uploadedRealVideoStartAtMs = null;
  const uploadedVideoStatuses = [];
  let uploadedAnalysisStartedAtMs = null;
  let auxWs = null;
  let auxWsUrl = '';
  let streamDisconnected = false;
  let streamDisconnectAtMs = null;
  let streamReconnectCount = 0;
  let streamRecoveryPausedMs = 0;
  let taskTimelineEndMs = null;
  let requiredTailResponseQuery = null;
  let requiredTailResponseQueries = [];
  let tailResponseWait = null;
  let tailResponseWaits = [];
  let allQueryResponseWait = null;
  let modelResponseOutcome = null;
  let postResponseRecording = null;
  let uploadPlaybackHealth = null;
  let uploadControlWsDisconnectedAtMs = null;
  let uploadControlWsCloseCount = 0;
  let uploadControlWsReconnectCount = 0;
  let uploadControlWsTotalDowntimeMs = 0;
  let uploadControlWsMaxDowntimeMs = 0;
  let resolveStreamDisconnect = null;
  let streamDisconnectPromise = null;
  let streamStartFailed = false;
  let streamStartFailureReason = '';
  let resolveStreamStartFailure = null;
  let streamStartFailurePromise = null;
  let pageStopCalled = false;
  const queryRounds = task.queries?.length
    ? task.queries
    : [{ id: 'R1', query: task.query, query_time_s: task.query_time_s }];
  const firstQueryTimeS = Number(queryRounds[0]?.query_time_s);
  const effectiveProcessIntervalS = Math.max(0.1, Number(processIntervalS) || 1);
  let firstQueryFrameGate = {
    required: inputMode === 'upload'
      && Number.isFinite(firstQueryTimeS)
      && firstQueryTimeS <= effectiveProcessIntervalS + 0.05,
    installed: false,
    released: false,
    reason: 'align the first sampled target frame with the first Query',
  };
  let webUiWarmupSessionId = '';
  const maxQueryTimeS = Math.max(...queryRounds.map((item) => item.query_time_s));
  const requiresTailResponse = maxQueryTimeS >= durationS;
  const queryEvents = [];
  const queryDispatchCounts = new Map();
  const derivedAttemptTimeoutS = Math.max(
    300,
    durationS
      + Math.max(0, maxQueryTimeS || 0)
      + (inputMode === 'rtsp' && task.local_video_path ? args.localRtspPrerollS : 0)
      + (inputMode === 'upload' ? args.videoUploadTimeoutS : 0)
      + (queryRounds.length ? args.postVideoResponseTimeoutS : 0)
      + args.streamReadyTimeoutS
      + args.streamReconnectTimeoutS
      + args.attemptTimeoutMarginS,
  );
  const attemptTimeoutS = args.attemptTimeoutS > 0 ? args.attemptTimeoutS : derivedAttemptTimeoutS;
  const attemptDeadlineMs = startedAtMs + attemptTimeoutS * 1000;

  const remainingAttemptMs = (phase) => {
    const remainingMs = attemptDeadlineMs - monotonicMs();
    if (remainingMs <= 0) {
      throw new Error(`Task attempt timed out after ${attemptTimeoutS.toFixed(1)}s during ${phase}`);
    }
    return remainingMs;
  };

  const boundedTimeoutMs = (requestedMs, phase) => Math.max(
    1,
    Math.min(Math.max(1, requestedMs), remainingAttemptMs(phase)),
  );

  const sleepWithAttemptDeadline = async (requestedMs, phase) => {
    const waitMs = Math.min(Math.max(0, requestedMs), remainingAttemptMs(phase));
    if (waitMs > 0) await sleep(waitMs);
    if (requestedMs > waitMs) {
      remainingAttemptMs(phase);
    }
  };

  const requestCurrentWebUiSessionVlmInferenceWarmup = async (imageDataUrl, timeoutMs) => {
    const warmupResponseStartIndex = vlmResponses.length;
    const warmupPromptStartIndex = promptEvents.length;
    const currentSessionId = await getPageSessionId(page) || observedSessionId || sessionId;
    const pageState = await page.evaluate(() => ({
      session_id: typeof sessionId === 'string' ? sessionId : '',
      analysis_running: Boolean(isAnalysisRunning),
      stream_lifecycle: typeof streamLifecycleState === 'string' ? streamLifecycleState : '',
    }));
    if (!currentSessionId || pageState.session_id !== currentSessionId) {
      throw new Error('Unable to bind the VLM warmup to the active WebUI page session');
    }
    if (pageState.analysis_running || pageState.stream_lifecycle !== 'idle') {
      throw new Error('The target WebUI session was already analyzing before the readiness warmup');
    }

    // A retry must not recreate a file that Chromium still references from the
    // previous file-input selection, otherwise it rejects the upload with
    // net::ERR_UPLOAD_FILE_CHANGED.
    const neutralVideoPath = path.join(
      taskDir,
      `.vlm-warmup-neutral-${crypto.randomUUID()}.mp4`,
    );
    let placeholderUploadId = '';
    let placeholderStarted = false;
    let placeholderStartResponse = null;
    let placeholderTrackReadyAtMs = null;
    let promptRequestedAtMs = null;
    let promptAcknowledgement = null;
    let frameRequestedAtMs = null;
    let warmupFramesSent = 0;
    let warmupSessionReset = null;
    let response = null;
    await createNeutralWarmupVideo(
      neutralVideoPath,
      args.ffmpegBin,
      Math.min(timeoutMs, 60000),
    );
    try {
      await sendControlMessage({
        type: 'set_debug',
        show_request_payload: true,
        show_response_payload: true,
        show_memory_state: true,
      });
      promptRequestedAtMs = monotonicMs();
      await sendControlMessage({ type: 'update_prompt', prompt: '' });
      promptAcknowledgement = await waitForPromptUpdate(
        promptRequestedAtMs,
        '',
        Math.min(timeoutMs, 10000),
      );

      // Keep the page's local preview state while using the more reliable
      // Node multipart path for the server-side upload.
      await prepareUploadedVideoSource(page, neutralVideoPath);
      const placeholderUpload = await uploadVideoApi(
        args.webUrl,
        currentSessionId,
        neutralVideoPath,
        Math.min(timeoutMs, 60000),
        auth,
      );
      await setUploadedVideoInfoFromApi(page, placeholderUpload, neutralVideoPath);
      const pagePlaceholderInfo = await getUploadedVideoInfo(page).catch(() => null);
      placeholderUploadId = String(
        placeholderUpload?.upload_id
        || placeholderUpload?.uploadId
        || placeholderUpload?.data?.upload_id
        || pagePlaceholderInfo?.upload_id
        || '',
      );
      if (!placeholderUploadId) {
        throw new Error('Neutral warmup upload returned no upload_id');
      }
      frameRequestedAtMs = monotonicMs();
      await startUploadedVideoFromPage(
        page,
        Math.min(timeoutMs, 60000),
      );
      placeholderStarted = true;
      placeholderStartResponse = uploadedVideoStartResponse;

      const relayDeadlineMs = monotonicMs() + Math.min(timeoutMs, 30000);
      while (monotonicMs() < relayDeadlineMs) {
        const trackReady = uploadedVideoStatuses.find((item) => (
          item.payload?.upload_id === placeholderUploadId
          && item.payload?.phase === 'track_published'
        ));
        if (trackReady) {
          placeholderTrackReadyAtMs = startedAtMs + trackReady.t_ms;
          break;
        }
        await sleep(20);
      }
      if (placeholderTrackReadyAtMs == null) {
        throw new Error('Neutral warmup track was not published before inference');
      }
      const deadlineMs = monotonicMs() + timeoutMs;
      while (monotonicMs() < deadlineMs) {
        response = vlmResponses.slice(warmupResponseStartIndex).find((item) => (
          startedAtMs + item.t_ms >= frameRequestedAtMs
        ));
        if (response) break;
        await sleep(20);
      }
      warmupFramesSent = await page.evaluate(() => Number(uploadedFrameSequence) || 0);
    } finally {
      if (placeholderStarted) {
        await stopUploadedVideoApi(
          args.webUrl,
          currentSessionId,
          Math.min(timeoutMs, 15000),
          auth,
        );
        await resetUploadedWarmupPreserveSession(page);
        await sleep(300);
        if (nativeSessionResetRequired) {
          warmupSessionReset = await requestVlmSessionReset(
            args,
            currentSessionId,
            Math.min(timeoutMs, 180000),
          );
        }
      }
      await fsp.rm(neutralVideoPath, { force: true }).catch(() => {});
    }
    if (!response) {
      throw new Error(`Timed out waiting for same-session WebUI inference after ${timeoutMs}ms`);
    }

    const identity = validateVlmBackendIdentity(args, [response]);
    if (!identity.ok) {
      throw new Error(identity.error || 'Same-session WebUI warmup identity verification failed');
    }
    const sessionAfterWarmup = await getPageSessionId(page) || observedSessionId || sessionId;
    if (sessionAfterWarmup !== currentSessionId) {
      throw new Error(
        `WebUI session changed during warmup: ${currentSessionId} -> ${sessionAfterWarmup}`,
      );
    }

    vlmResponses.splice(warmupResponseStartIndex);
    promptEvents.splice(warmupPromptStartIndex);
    webUiWarmupSessionId = currentSessionId;
    return {
      strategy: 'webui_same_session',
      response_model: identity.observed_models?.[0] || '',
      backend_identity: identity,
      session_id: currentSessionId,
      prompt_acknowledged: true,
      prompt_ack_latency_s: (
        startedAtMs + promptAcknowledgement.t_ms - promptRequestedAtMs
      ) / 1000,
      frame_to_response_s: (startedAtMs + response.t_ms - frameRequestedAtMs) / 1000,
      warmup_frames_sent: warmupFramesSent,
      page_analysis_running_during_warmup: pageState.analysis_running,
      placeholder_video_started: placeholderStarted,
      placeholder_upload_method: 'node_multipart',
      placeholder_transport: placeholderStartResponse?.transport || '',
      placeholder_track_ready: placeholderTrackReadyAtMs != null,
      placeholder_track_ready_offset_s: placeholderTrackReadyAtMs == null
        ? null
        : (placeholderTrackReadyAtMs - startedAtMs) / 1000,
      placeholder_stopped_before_target_upload: true,
      warmup_session_reset_required: nativeSessionResetRequired,
      warmup_session_reset: warmupSessionReset,
      target_video_selected_or_uploaded: false,
    };
  };

  const performVlmInferenceWarmup = async (phase) => {
    if (!args.vlmModel) return null;
    if (!args.vlmWarmup) {
      latestVlmWarmupReadyAtMs = monotonicMs();
      return {
        required: false,
        ok: true,
        status: 'disabled',
        phase,
      };
    }

    latestVlmWarmupReadyAtMs = null;
    const warmupStartedAtMs = monotonicMs();
    // Readiness probing must not upload, stream, or expose the evaluated video.
    // Use an isolated neutral frame so the target remains untouched until the API is ready.
    const sourcePath = '';
    const warmupSource = 'generated_neutral_frame';
    if (!vlmWarmupFramePromise) {
      vlmWarmupFramePromise = extractWarmupFrameDataUrl(
        sourcePath,
        args.ffmpegBin,
        boundedTimeoutMs(60000, `VLM warmup frame extraction during ${phase}`),
      );
    }

    let imageDataUrl;
    try {
      imageDataUrl = await vlmWarmupFramePromise;
    } catch (error) {
      const warmupError = new Error(error.message);
      warmupError.code = 'VLM_WARMUP_FAILED';
      const failed = {
        required: true,
        ok: false,
        status: 'frame_extraction_failed',
        phase,
        started_offset_s: (warmupStartedAtMs - startedAtMs) / 1000,
        completed_offset_s: (monotonicMs() - startedAtMs) / 1000,
        source: warmupSource,
        error: error.message,
      };
      vlmInferenceWarmups.push(failed);
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_vlm_inference_warmup_failed',
        task_id: task.id,
        warmup: failed,
      });
      throw warmupError;
    }

    const frameSha256 = crypto.createHash('sha256').update(imageDataUrl).digest('hex');
    const maxAttempts = args.vlmWarmupRetries + 1;
    const attempts = [];
    for (let warmupAttempt = 1; warmupAttempt <= maxAttempts; warmupAttempt += 1) {
      const requestedAtMs = monotonicMs();
      appendJsonl(events, {
        t_ms: requestedAtMs - startedAtMs,
        type: 'capture_vlm_inference_warmup_started',
        task_id: task.id,
        phase,
        warmup_attempt: warmupAttempt,
        max_warmup_attempts: maxAttempts,
        model: args.vlmModel,
        api_base: args.vlmApiBase,
        source: warmupSource,
        frame_sha256: frameSha256,
      });
      try {
        const warmupTimeoutMs = boundedTimeoutMs(
          args.vlmWarmupTimeoutS * 1000,
          `VLM inference warmup attempt ${warmupAttempt}/${maxAttempts} during ${phase}`,
        );
        const providerWarmup = providerWarmupRequired
          ? await requestScaffoldProviderWarmup(args, imageDataUrl, warmupTimeoutMs)
          : null;
        const response = inputMode === 'upload'
          ? await requestCurrentWebUiSessionVlmInferenceWarmup(imageDataUrl, warmupTimeoutMs)
          : await requestVlmInferenceWarmup(args, imageDataUrl, warmupTimeoutMs);
        const completedAtMs = monotonicMs();
        const attemptAudit = {
          attempt: warmupAttempt,
          ok: true,
          strategy: response.strategy,
          requested_offset_s: (requestedAtMs - startedAtMs) / 1000,
          completed_offset_s: (completedAtMs - startedAtMs) / 1000,
          elapsed_s: (completedAtMs - requestedAtMs) / 1000,
          response_model: response.response_model,
          provider_warmup: providerWarmup,
        };
        attempts.push(attemptAudit);
        latestVlmWarmupReadyAtMs = completedAtMs;
        const succeeded = {
          required: true,
          ok: true,
          status: 'ready',
          phase,
          started_offset_s: (warmupStartedAtMs - startedAtMs) / 1000,
          completed_offset_s: (completedAtMs - startedAtMs) / 1000,
          elapsed_s: (completedAtMs - warmupStartedAtMs) / 1000,
          attempts_used: warmupAttempt,
          max_attempts: maxAttempts,
          model: args.vlmModel,
          api_base: args.vlmApiBase,
          response_model: response.response_model,
          provider_warmup_required: providerWarmupRequired,
          provider_warmup: providerWarmup,
          source: warmupSource,
          strategy: response.strategy,
          session_id: response.session_id || undefined,
          prompt_acknowledged: response.prompt_acknowledged,
          prompt_ack_latency_s: response.prompt_ack_latency_s,
          frame_to_response_s: response.frame_to_response_s,
          warmup_frames_sent: response.warmup_frames_sent,
          placeholder_video_started: response.placeholder_video_started,
          placeholder_transport: response.placeholder_transport,
          placeholder_track_ready: response.placeholder_track_ready,
          placeholder_track_ready_offset_s: response.placeholder_track_ready_offset_s,
          placeholder_stopped_before_target_upload: response.placeholder_stopped_before_target_upload,
          warmup_session_reset_required: response.warmup_session_reset_required,
          warmup_session_reset: response.warmup_session_reset,
          target_video_selected_or_uploaded: response.target_video_selected_or_uploaded,
          frame_sha256: frameSha256,
          backend_identity: response.backend_identity,
          attempts,
        };
        vlmInferenceWarmups.push(succeeded);
        appendJsonl(events, {
          t_ms: completedAtMs - startedAtMs,
          type: 'capture_vlm_inference_warmup_succeeded',
          task_id: task.id,
          warmup: succeeded,
        });
        return succeeded;
      } catch (error) {
        const failedAtMs = monotonicMs();
        const retryDelayS = warmupAttempt < maxAttempts
          ? (error.httpStatus === 429
            ? Math.max(error.retryAfterS || 0, Math.min(60, 15 * (2 ** (warmupAttempt - 1))))
            : Math.max(1, args.healthIntervalS))
          : 0;
        attempts.push({
          attempt: warmupAttempt,
          ok: false,
          requested_offset_s: (requestedAtMs - startedAtMs) / 1000,
          completed_offset_s: (failedAtMs - startedAtMs) / 1000,
          elapsed_s: (failedAtMs - requestedAtMs) / 1000,
          http_status: error.httpStatus || null,
          upstream_code: error.upstreamCode || '',
          retry_delay_s: retryDelayS,
          error: error.message,
        });
        appendJsonl(events, {
          t_ms: failedAtMs - startedAtMs,
          type: 'capture_vlm_inference_warmup_attempt_failed',
          task_id: task.id,
          phase,
          warmup_attempt: warmupAttempt,
          max_warmup_attempts: maxAttempts,
          http_status: error.httpStatus || null,
          upstream_code: error.upstreamCode || '',
          retry_delay_s: retryDelayS,
          error: error.message,
        });
        if (warmupAttempt < maxAttempts) {
          await sleepWithAttemptDeadline(
            retryDelayS * 1000,
            `VLM inference warmup retry gap during ${phase}`,
          );
        }
      }
    }

    const failedAtMs = monotonicMs();
    const failed = {
      required: true,
      ok: false,
      status: 'failed',
      phase,
      started_offset_s: (warmupStartedAtMs - startedAtMs) / 1000,
      completed_offset_s: (failedAtMs - startedAtMs) / 1000,
      elapsed_s: (failedAtMs - warmupStartedAtMs) / 1000,
      attempts_used: attempts.length,
      max_attempts: maxAttempts,
      model: args.vlmModel,
      api_base: args.vlmApiBase,
      source: warmupSource,
      frame_sha256: frameSha256,
      attempts,
      error: attempts[attempts.length - 1]?.error || 'unknown warmup failure',
    };
    vlmInferenceWarmups.push(failed);
    appendJsonl(events, {
      t_ms: failedAtMs - startedAtMs,
      type: 'capture_vlm_inference_warmup_failed',
      task_id: task.id,
      warmup: failed,
    });
    const warmupError = new Error(
      `VLM inference warmup failed before video playback after ${attempts.length} attempt(s): ${failed.error}`,
    );
    warmupError.code = 'VLM_WARMUP_FAILED';
    throw warmupError;
  };

  const markPlaybackStartRequested = (phase) => {
    if (args.vlmModel && args.vlmWarmup && latestVlmWarmupReadyAtMs == null) {
      throw new Error(`Refusing to start video before successful VLM inference warmup during ${phase}`);
    }
    if (inputMode === 'upload' && targetUploadReadyAtMs == null) {
      throw new Error(`Refusing to start uploaded video before the target upload is ready during ${phase}`);
    }
    if (inputMode === 'upload' && targetPlaybackWasStarted) {
      throw new Error(`Refusing to restart an uploaded video in the same attempt during ${phase}`);
    }
    const requestedAtMs = monotonicMs();
    if (inputMode === 'upload') {
      officialPlaybackRequestedAtMs = requestedAtMs;
      targetPlaybackWasStarted = true;
    }
    const event = {
      phase,
      requested_offset_s: (requestedAtMs - startedAtMs) / 1000,
      warmup_completed_offset_s: latestVlmWarmupReadyAtMs == null
        ? null
        : (latestVlmWarmupReadyAtMs - startedAtMs) / 1000,
      warmup_to_playback_request_s: latestVlmWarmupReadyAtMs == null
        ? null
        : (requestedAtMs - latestVlmWarmupReadyAtMs) / 1000,
      upload_ready_offset_s: targetUploadReadyAtMs == null
        ? null
        : (targetUploadReadyAtMs - startedAtMs) / 1000,
      upload_ready_to_playback_request_s: targetUploadReadyAtMs == null
        ? null
        : (requestedAtMs - targetUploadReadyAtMs) / 1000,
    };
    playbackStartRequests.push(event);
    appendJsonl(events, {
      t_ms: requestedAtMs - startedAtMs,
      type: 'capture_playback_start_requested',
      task_id: task.id,
      ...event,
    });
    return requestedAtMs;
  };

  const resetStreamDisconnectSignal = () => {
    streamDisconnected = false;
    streamDisconnectAtMs = null;
    streamDisconnectPromise = new Promise((resolve) => {
      resolveStreamDisconnect = resolve;
    });
  };

  const resetStreamStartFailureSignal = () => {
    streamStartFailed = false;
    streamStartFailureReason = '';
    streamStartFailurePromise = new Promise((resolve) => {
      resolveStreamStartFailure = resolve;
    });
  };

  resetStreamDisconnectSignal();
  resetStreamStartFailureSignal();

  const observeUploadedTimeline = (state, phase) => {
    const currentS = Number(state?.current_time_s);
    const durationS = Number(state?.duration_s);
    if (!Number.isFinite(currentS)) return null;
    if (!uploadPlaybackHealth || !Number.isFinite(durationS) || durationS <= 0) return currentS;

    const previousS = Number(uploadPlaybackHealth.last_timeline_media_time_s);
    const looped = Number.isFinite(previousS) && currentS + 0.25 < previousS;
    if (looped) uploadPlaybackHealth.loop_count += 1;
    uploadPlaybackHealth.last_timeline_media_time_s = currentS;
    const timelineS = uploadPlaybackHealth.loop_count * durationS + currentS;
    uploadPlaybackHealth.last_timeline_time_s = timelineS;
    if (looped) {
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_upload_playback_loop_observed',
        task_id: task.id,
        phase,
        previous_media_time_s: previousS,
        current_media_time_s: currentS,
        duration_s: durationS,
        loop_count: uploadPlaybackHealth.loop_count,
        timeline_time_s: timelineS,
      });
    }
    return timelineS;
  };

  const markStreamDisconnected = (reason) => {
    if (streamDisconnected) return;
    streamDisconnected = true;
    streamDisconnectAtMs = monotonicMs();
    appendJsonl(events, {
      t_ms: streamDisconnectAtMs - startedAtMs,
      type: 'capture_stream_disconnected',
      task_id: task.id,
      reason,
    });
    if (resolveStreamDisconnect) resolveStreamDisconnect(reason);
  };

  const markStreamStartFailure = (reason) => {
    if (streamStartFailed) return;
    streamStartFailed = true;
    streamStartFailureReason = reason;
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'capture_stream_start_failure',
      task_id: task.id,
      reason,
    });
    if (resolveStreamStartFailure) resolveStreamStartFailure(reason);
  };

  const recordVlmResponse = (source, parsed) => {
    const key = JSON.stringify({
      total_inferences: parsed.metrics?.total_inferences ?? null,
      text: parsed.text ?? '',
      user_prompt: parsed.metrics?.user_prompt ?? '',
    });
    if (vlmResponseKeys.has(key)) return;
    vlmResponseKeys.add(key);
    const tMs = monotonicMs() - startedAtMs;
    const responseProtocol = classifyVlmResponseText(
      parsed.text ?? '',
      parsed.response_payload ?? null,
    );
    const response = {
      t_ms: tMs,
      source,
      text: parsed.text ?? '',
      metrics: parsed.metrics ?? null,
      request_payload: parsed.request_payload ?? null,
      response_payload: parsed.response_payload ?? null,
      memory_state: parsed.memory_state ?? null,
      response_protocol: responseProtocol,
    };
    vlmResponses.push(response);
    if (
      responseProtocol.classification === 'provider_error'
      && !providerInfrastructureFailure
    ) {
      providerInfrastructureFailure = {
        t_ms: tMs,
        source,
        text: parsed.text ?? '',
      };
      appendJsonl(events, {
        ...providerInfrastructureFailure,
        type: 'capture_provider_infrastructure_failure',
        task_id: task.id,
      });
    }
    if (!responseProtocol.protocol_valid) {
      const violation = {
        t_ms: tMs,
        response_index: vlmResponses.length - 1,
        classification: responseProtocol.classification,
        violation: responseProtocol.violation,
        substantive: responseProtocol.substantive,
        user_prompt_sha256: parsed.metrics?.user_prompt
          ? crypto.createHash('sha256').update(parsed.metrics.user_prompt).digest('hex')
          : '',
      };
      protocolViolations.push(violation);
      appendJsonl(events, {
        ...violation,
        type: 'capture_vlm_response_protocol_violation',
        task_id: task.id,
      });
    }
  };

  const responseQuerySha256 = (response) => (
    response.response_payload?.streamingharness?.query_frame_queue?.user_query_sha256
    || response.response_payload?.native_realtime?.provider_output_query_sha256
    || ''
  );

  const responseQueryFrameTimeS = (response) => {
    const queuedRange = response.response_payload?.streamingharness
      ?.query_frame_queue?.frame_time_range;
    return parseFrameTimeRangeS(queuedRange ?? response.request_payload?.frame_time_range);
  };

  const recordBackgroundTaskEvent = (source, parsed) => {
    const taskId = String(parsed.task_id || '');
    if (!taskId) return;
    const tMs = monotonicMs() - startedAtMs;
    const existing = backgroundTasks.get(taskId) || {
      task_id: taskId,
      question: '',
      status: 'unknown',
      started_t_ms: null,
      finished_t_ms: null,
      text: '',
      error: '',
      source,
    };
    existing.question = String(parsed.question || existing.question || '').trim();
    existing.source = source;
    if (parsed.type === 'background_task_started') {
      existing.status = 'running';
      existing.started_t_ms = tMs;
      existing.foreground_text = parsed.foreground_text || '';
    } else if (parsed.type === 'background_result_ready') {
      existing.status = 'ready';
      existing.finished_t_ms = tMs;
      existing.text = parsed.text || '';
    } else if (parsed.type === 'background_result_error') {
      existing.status = 'error';
      existing.finished_t_ms = tMs;
      existing.error = parsed.error || 'Background model failed';
    }
    backgroundTasks.set(taskId, existing);
  };

  const recordPromptEvent = (source, direction, payload) => {
    promptEvents.push({ source, direction, t_ms: monotonicMs() - startedAtMs, payload });
  };

  const waitForPromptUpdate = async (requestedAtMs, expectedPrompt, timeoutMs = 10000) => {
    const deadlineMs = monotonicMs() + timeoutMs;
    while (monotonicMs() < deadlineMs) {
      const acknowledgement = promptEvents.find((event) => (
        event.direction === 'received'
        && startedAtMs + event.t_ms >= requestedAtMs
        && event.payload?.type === 'prompt_updated'
        && String(event.payload?.prompt || '') === String(expectedPrompt || '')
      ));
      if (acknowledgement) return acknowledgement;
      await sleep(20);
    }
    throw new Error(`Timed out waiting for the WebUI to acknowledge the prompt after ${timeoutMs}ms`);
  };

  const modelUpdateMatchesExpected = (payload) => {
    if (!args.vlmModel) return true;
    const expectedApiBase = String(args.vlmApiBase || '').replace(/\/+$/, '');
    return payload?.model === args.vlmModel
      && String(payload?.api_base || '').replace(/\/+$/, '') === expectedApiBase;
  };

  const recordModelUpdate = (source, parsed) => {
    const event = {
      source,
      t_ms: monotonicMs() - startedAtMs,
      payload: redactSecretsForLog(parsed),
    };
    modelUpdateEvents.push(event);
    if (vlmSelectionGuardActive && !modelUpdateMatchesExpected(parsed) && !vlmSelectionViolation) {
      vlmSelectionViolation = event;
    }
  };

  const waitForModelUpdate = async (requestedAtMs, timeoutMs = 30000) => {
    const deadlineMs = monotonicMs() + timeoutMs;
    const expectedApiBase = args.vlmApiBase.replace(/\/+$/, '');
    while (monotonicMs() < deadlineMs) {
      const matching = modelUpdateEvents.find((event) => {
        if (startedAtMs + event.t_ms < requestedAtMs) return false;
        const payload = event.payload || {};
        return payload.model === args.vlmModel
          && String(payload.api_base || '').replace(/\/+$/, '') === expectedApiBase;
      });
      if (matching) return matching;
      await sleep(50);
    }
    const latest = modelUpdateEvents.length
      ? JSON.stringify(modelUpdateEvents[modelUpdateEvents.length - 1].payload)
      : 'none';
    throw new Error(
      `Timed out waiting for model_updated for ${args.vlmModel} at ${args.vlmApiBase}; latest=${latest}`,
    );
  };

  const recordUploadedVideoStatus = (source, parsed) => {
    const tMs = monotonicMs() - startedAtMs;
    uploadedVideoStatuses.push({ source, t_ms: tMs, payload: parsed });
    if (parsed.phase === 'preroll_started') {
      uploadedPrerollStartedAtMs = startedAtMs + tMs;
      uploadedPrerollSeconds = Number(parsed.preroll_seconds) || 0;
      uploadedScheduledRealVideoStartAtMs = uploadedPrerollStartedAtMs + uploadedPrerollSeconds * 1000;
    }
    if (parsed.phase === 'analysis_started' && uploadedAnalysisStartedAtMs == null) {
      uploadedAnalysisStartedAtMs = startedAtMs + tMs;
      officialPlaybackAnalyzerReadyAtMs = uploadedAnalysisStartedAtMs;
      if (!targetPlaybackWasStarted) uploadedRealVideoStartAtMs = uploadedAnalysisStartedAtMs;
      if (uploadedStatusReadyInfo) {
        uploadedStatusReadyInfo.analyzerReadyInfo = {
          readyMs: uploadedAnalysisStartedAtMs,
          source: 'analysis_started',
        };
      }
    }
  };

  const waitForUploadedRealVideoStartFromStatus = async (timeoutMs = 60000) => {
    const deadline = monotonicMs() + timeoutMs;
    while (monotonicMs() < deadline) {
      if (uploadedRealVideoStartAtMs != null) {
        return {
          readyMs: uploadedRealVideoStartAtMs,
          preroll_started_at_ms: uploadedPrerollStartedAtMs,
          preroll_seconds: uploadedPrerollSeconds,
          scheduled_real_video_start_at_ms: uploadedScheduledRealVideoStartAtMs,
          source: 'analysis_started',
        };
      }
      await sleep(50);
    }
    const lastStatus = uploadedVideoStatuses.length ? uploadedVideoStatuses[uploadedVideoStatuses.length - 1] : null;
    throw new Error(`Timed out waiting for uploaded video preroll status for session ${observedSessionId || sessionId}; last=${JSON.stringify(lastStatus)}`);
  };

  const connectAuxWebSocket = async () => {
    const sid = await getPageSessionId(page).catch(() => '') || observedSessionId || sessionId;
    auxWsUrl = buildWebSocketUrl(args.webUrl, sid);
    const ws = await openNodeWebSocket(auxWsUrl, 30000, auth);
    auxWs = ws;
    observedSessionId = sid;
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'aux_ws_open',
      url: auxWsUrl,
      session_id: sid,
    });
    ws.on('message', (data) => {
      const parsed = tryJson(data.toString());
      if (parsed.type === 'vlm_response') {
        recordVlmResponse('aux_ws', parsed);
      } else if (parsed.type === 'prompt_updated') {
        recordPromptEvent('aux_ws', 'received', parsed);
      } else if (parsed.type === 'model_updated') {
        recordModelUpdate('aux_ws', parsed);
      } else if (parsed.type === 'uploaded_video_status') {
        recordUploadedVideoStatus('aux_ws', parsed);
      } else if (
        parsed.type === 'background_task_started'
        || parsed.type === 'background_result_ready'
        || parsed.type === 'background_result_error'
      ) {
        recordBackgroundTaskEvent('aux_ws', parsed);
      }
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'aux_ws_frame_received',
        payload: parsed,
      });
    });
    ws.on('close', (code, reason) => {
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'aux_ws_close',
        code,
        reason: reason?.toString?.() || '',
      });
      if (auxWs === ws) auxWs = null;
    });
    ws.on('error', (error) => {
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'aux_ws_error',
        error: error.message,
      });
    });
    return ws;
  };

  const ensureAuxWebSocket = async () => {
    if (auxWs && auxWs.readyState === WebSocket.OPEN) return auxWs;
    await page.evaluate(() => {
      if (typeof window.__CAPTURE_DISABLE_PAGE_WS_NOW === 'function') {
        window.__CAPTURE_DISABLE_PAGE_WS_NOW();
      }
    }).catch(() => {});
    return connectAuxWebSocket();
  };

  const sendControlMessage = async (payload) => {
    if (inputMode === 'upload') {
      if (await isPageWebSocketOpen(page)) {
        try {
          await sendWebSocketFromPage(page, payload);
          if (payload.type === 'update_prompt') {
            recordPromptEvent('page_ws', 'sent', payload);
          }
          return;
        } catch (error) {
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'page_ws_control_send_failed',
            payload,
            error: error.message,
          });
        }
      }
      const ws = await ensureAuxWebSocket();
      ws.send(JSON.stringify(payload));
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'aux_ws_frame_sent',
        payload,
      });
      if (payload.type === 'update_prompt') {
        recordPromptEvent('aux_ws', 'sent', payload);
      }
      return;
    }
    await sendWebSocketFromPage(page, payload);
  };

  const recoverRtspStream = async (phase) => {
    if (inputMode === 'upload') {
      throw new Error(
        `Uploaded video stream disconnected during ${phase} at ${streamDisconnectAtMs == null ? 'unknown' : ((streamDisconnectAtMs - startedAtMs) / 1000).toFixed(3)}s`,
      );
    }
    if (queryEvents.length) {
      throw new Error(
        `RTSP disconnected after Query delivery during ${phase}; restart the task attempt `
        + 'instead of replaying a Query',
      );
    }
    if (streamReconnectCount >= args.streamReconnects) {
      throw new Error(
        `WebRTC stream disconnected during ${phase} at ${((streamDisconnectAtMs - startedAtMs) / 1000).toFixed(3)}s`,
      );
    }
    streamReconnectCount += 1;
    const disconnectedAtS = streamDisconnectAtMs == null ? null : (streamDisconnectAtMs - startedAtMs) / 1000;
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'capture_stream_reconnect_start',
      task_id: task.id,
      phase,
      reconnect_count: streamReconnectCount,
      max_reconnects: args.streamReconnects,
      disconnected_at_s: disconnectedAtS,
      video_url: effectiveVideoUrl,
    });

    const publisherTimelineFrozen = localPublisher?.freezeTimeline?.(
      `webrtc disconnect during ${phase}`,
    ) || false;

    if (publisherTimelineFrozen) {
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_stream_backend_cleanup_start',
        task_id: task.id,
        phase,
        reconnect_count: streamReconnectCount,
      });
      try {
        await stopPageFromPage(
          page,
          boundedTimeoutMs(30000, `WebUI backend cleanup during ${phase}`),
        );
      } catch (error) {
        const cleanupSessionId = observedSessionId
          || await getPageSessionId(page).catch(() => '')
          || sessionId;
        appendJsonl(events, {
          t_ms: monotonicMs() - startedAtMs,
          type: 'capture_stream_backend_page_stop_failed',
          task_id: task.id,
          phase,
          reconnect_count: streamReconnectCount,
          session_id: cleanupSessionId,
          error: error.message,
        });
        try {
          await cleanupSessionApi(
            args.webUrl,
            cleanupSessionId,
            boundedTimeoutMs(15000, `WebUI backend cleanup API fallback during ${phase}`),
            auth,
          );
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_stream_backend_cleanup_api_done',
            task_id: task.id,
            phase,
            reconnect_count: streamReconnectCount,
            session_id: cleanupSessionId,
          });
        } catch (cleanupError) {
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_stream_backend_cleanup_api_failed',
            task_id: task.id,
            phase,
            reconnect_count: streamReconnectCount,
            session_id: cleanupSessionId,
            error: cleanupError.message,
          });
        }
      }
      await waitForWebSocket(
        page,
        boundedTimeoutMs(60000, `WebUI websocket reconnect during ${phase}`),
      );
      observedSessionId = await getPageSessionId(page) || observedSessionId || sessionId;
      await selectRtspSource(page, effectiveVideoUrl);
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_stream_backend_cleanup_done',
        task_id: task.id,
        phase,
        reconnect_count: streamReconnectCount,
        session_id: observedSessionId,
      });
    }

    let publisherReadyInfo = null;
    let publisherPreserved = false;
    if (localPublisher?.keepaliveProc && localPublisher?.waitUntilReady) {
      publisherReadyInfo = await localPublisher.waitUntilReady(15000);
      publisherPreserved = Boolean(publisherReadyInfo);
      if (publisherPreserved) {
        appendJsonl(events, {
          t_ms: monotonicMs() - startedAtMs,
          type: 'capture_stream_reconnect_publisher_preserved',
          task_id: task.id,
          phase,
          reconnect_count: streamReconnectCount,
          publisher_ready_info: publisherReadyInfo,
          relay_url: effectiveVideoUrl,
        });
      }
    }

    if (!publisherPreserved && localPublisher?.restartNow) {
      publisherReadyInfo = await localPublisher.restartNow(`webrtc disconnect during ${phase}`);
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_stream_reconnect_publisher_restarted',
        task_id: task.id,
        phase,
        reconnect_count: streamReconnectCount,
        publisher_restart_info: publisherReadyInfo,
        relay_url: effectiveVideoUrl,
      });
      if (args.localRtspReconnectWarmupMs > 0) {
        await sleepWithAttemptDeadline(args.localRtspReconnectWarmupMs, `RTSP reconnect warmup during ${phase}`);
      }
    }

    const maxReconnectStartAttempts = args.streamStartRetries + 1;
    let lastFailure = '';
    for (let reconnectStartAttempt = 1; reconnectStartAttempt <= maxReconnectStartAttempts; reconnectStartAttempt += 1) {
      if (reconnectStartAttempt > 1 && args.streamStartRetryGapS > 0) {
        await sleepWithAttemptDeadline(
          args.streamStartRetryGapS * 1000,
          `RTSP reconnect retry gap during ${phase}`,
        );
      }
      resetStreamDisconnectSignal();
      resetStreamStartFailureSignal();
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_stream_reconnect_attempt_start',
        task_id: task.id,
        phase,
        reconnect_count: streamReconnectCount,
        reconnect_start_attempt: reconnectStartAttempt,
        max_reconnect_start_attempts: maxReconnectStartAttempts,
      });
      await page.evaluate(async () => {
        if (typeof window.__CAPTURE_CLOSE_PEER_ONLY === 'function') {
          window.__CAPTURE_CLOSE_PEER_ONLY();
        }
      });
      await startRtspFromPage(page);
      const outcome = await Promise.race([
        waitForStreamingReady(
          page,
          boundedTimeoutMs(args.streamReconnectTimeoutS * 1000, `RTSP reconnect wait during ${phase}`),
        )
          .then(() => ({ status: 'ready' }))
          .catch((error) => ({ status: 'timeout', error: error.message })),
        streamDisconnectPromise.then((reason) => ({ status: 'disconnected', error: reason })),
        streamStartFailurePromise.then((reason) => ({ status: 'start_failed', error: reason })),
      ]);
      if (outcome.status === 'ready') {
        if (publisherTimelineFrozen) {
          localPublisher.resumeTimeline(`webrtc reconnect ready during ${phase}`);
        }
        appendJsonl(events, {
          t_ms: monotonicMs() - startedAtMs,
          type: 'capture_stream_reconnect_ready',
          task_id: task.id,
          phase,
          reconnect_count: streamReconnectCount,
          reconnect_start_attempt: reconnectStartAttempt,
        });
        return;
      }
      lastFailure = `${outcome.status}${outcome.error ? `: ${outcome.error}` : ''}`;
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_stream_reconnect_attempt_failed',
        task_id: task.id,
        phase,
        reconnect_count: streamReconnectCount,
        reconnect_start_attempt: reconnectStartAttempt,
        max_reconnect_start_attempts: maxReconnectStartAttempts,
        failure: lastFailure,
      });
    }
    throw new Error(`WebRTC reconnect failed during ${phase} after ${maxReconnectStartAttempts} start attempt(s): ${lastFailure}`);
  };

  const checkUploadedPlaybackHealth = async (phase) => {
    if (inputMode !== 'upload' || !uploadPlaybackHealth) return;
    const checkedAtMs = monotonicMs();
    let state;
    try {
      state = await page.evaluate(() => ({
        active_source: String(activeStreamSource || ''),
        lifecycle: String(streamLifecycleState || ''),
        analysis_running: Boolean(isAnalysisRunning),
        paused: Boolean(videoElement?.paused),
        ended: Boolean(videoElement?.ended),
        loop: Boolean(videoElement?.loop),
        ready_state: Number(videoElement?.readyState) || 0,
        current_time_s: Number(videoElement?.currentTime),
        duration_s: Number(videoElement?.duration),
        frame_sequence: Number(uploadedFrameSequence) || 0,
        websocket_state: Number(websocket?.readyState),
        continuity_guard: window.__CAPTURE_UPLOAD_CONTINUITY_GUARD ? {
          installed: Boolean(window.__CAPTURE_UPLOAD_CONTINUITY_GUARD.installed),
          active: Boolean(window.__CAPTURE_UPLOAD_CONTINUITY_GUARD.active),
          pause_events: Number(window.__CAPTURE_UPLOAD_CONTINUITY_GUARD.pause_events) || 0,
          resume_attempts: Number(window.__CAPTURE_UPLOAD_CONTINUITY_GUARD.resume_attempts) || 0,
          resume_successes: Number(window.__CAPTURE_UPLOAD_CONTINUITY_GUARD.resume_successes) || 0,
          resume_failures: Number(window.__CAPTURE_UPLOAD_CONTINUITY_GUARD.resume_failures) || 0,
          last_pause_media_time_s: window.__CAPTURE_UPLOAD_CONTINUITY_GUARD.last_pause_media_time_s,
          last_resume_media_time_s: window.__CAPTURE_UPLOAD_CONTINUITY_GUARD.last_resume_media_time_s,
        } : null,
      }));
    } catch (error) {
      markStreamDisconnected(`Unable to inspect uploaded playback during ${phase}: ${error.message}`);
      return;
    }

    uploadPlaybackHealth.checks += 1;
    uploadPlaybackHealth.last_state = state;
    uploadPlaybackHealth.continuity_guard = state.continuity_guard;
    observeUploadedTimeline(state, `health:${phase}`);
    const controlWsOpen = state.websocket_state === WebSocket.OPEN;
    if (controlWsOpen && uploadControlWsDisconnectedAtMs != null) {
      const downtimeMs = checkedAtMs - uploadControlWsDisconnectedAtMs;
      uploadControlWsReconnectCount += 1;
      uploadControlWsTotalDowntimeMs += downtimeMs;
      uploadControlWsMaxDowntimeMs = Math.max(uploadControlWsMaxDowntimeMs, downtimeMs);
      uploadControlWsDisconnectedAtMs = null;
      appendJsonl(events, {
        t_ms: checkedAtMs - startedAtMs,
        type: 'capture_upload_control_ws_reconnected',
        task_id: task.id,
        phase,
        reconnect_count: uploadControlWsReconnectCount,
        downtime_ms: downtimeMs,
        recovery_permitted: false,
      });
      markStreamDisconnected('Uploaded-video control WebSocket disconnected after single-pass playback started');
      return;
    }
    const previousMediaTimeS = uploadPlaybackHealth.last_media_time_s;
    const mediaAdvanced = Number.isFinite(state.current_time_s) && (
      !Number.isFinite(previousMediaTimeS)
      || state.current_time_s > previousMediaTimeS + 0.03
      || state.current_time_s + 0.25 < previousMediaTimeS
    );
    if (mediaAdvanced) {
      uploadPlaybackHealth.last_media_progress_at_ms = checkedAtMs;
      uploadPlaybackHealth.last_media_time_s = state.current_time_s;
    }
    if (state.frame_sequence > uploadPlaybackHealth.last_frame_sequence) {
      uploadPlaybackHealth.last_frame_progress_at_ms = checkedAtMs;
      uploadPlaybackHealth.last_frame_sequence = state.frame_sequence;
    }

    const hardFailure = state.active_source !== 'upload'
      || state.lifecycle !== 'running'
      || !state.analysis_running
      || !state.loop
      || state.ended;
    const softFailure = state.paused || state.ready_state < 2;
    if (hardFailure || softFailure) {
      uploadPlaybackHealth.unhealthy_since_ms ??= checkedAtMs;
    } else {
      uploadPlaybackHealth.unhealthy_since_ms = null;
    }

    const mediaStallMs = checkedAtMs - uploadPlaybackHealth.last_media_progress_at_ms;
    const frameStallMs = checkedAtMs - uploadPlaybackHealth.last_frame_progress_at_ms;
    uploadPlaybackHealth.max_media_stall_ms = Math.max(
      uploadPlaybackHealth.max_media_stall_ms,
      mediaStallMs,
    );
    uploadPlaybackHealth.max_frame_stall_ms = Math.max(
      uploadPlaybackHealth.max_frame_stall_ms,
      frameStallMs,
    );
    let reason = '';
    if (hardFailure) {
      reason = `uploaded playback link state invalid: ${JSON.stringify(state)}`;
    } else if (state.paused) {
      reason = `single-pass uploaded playback paused: ${JSON.stringify(state)}`;
    } else if (
      !controlWsOpen
      && uploadControlWsCloseCount > args.uploadControlReconnects
    ) {
      reason = `uploaded-video control WebSocket exceeded ${args.uploadControlReconnects} reconnect(s)`;
    } else if (
      !controlWsOpen
      && uploadControlWsDisconnectedAtMs != null
      && checkedAtMs - uploadControlWsDisconnectedAtMs > args.uploadControlReconnectTimeoutS * 1000
    ) {
      reason = `uploaded-video control WebSocket did not reconnect within ${args.uploadControlReconnectTimeoutS}s`;
    } else if (
      uploadPlaybackHealth.unhealthy_since_ms != null
      && checkedAtMs - uploadPlaybackHealth.unhealthy_since_ms > 2000
    ) {
      reason = `uploaded playback paused/not-ready for more than 2s: ${JSON.stringify(state)}`;
    } else if (
      mediaStallMs > (
        Number(state.continuity_guard?.pause_events) > 0
          ? args.uploadControlReconnectTimeoutS * 1000
          : 2000
      )
    ) {
      reason = `uploaded video media time did not advance for ${(mediaStallMs / 1000).toFixed(3)}s`;
    } else if (uploadControlWsDisconnectedAtMs == null && frameStallMs > 5000) {
      reason = `uploaded video frame relay did not advance for ${(frameStallMs / 1000).toFixed(3)}s`;
    }
    if (reason) {
      appendJsonl(events, {
        t_ms: checkedAtMs - startedAtMs,
        type: 'capture_upload_playback_interrupted',
        task_id: task.id,
        phase,
        reason,
        state,
      });
      markStreamDisconnected(reason);
    }
  };

  const waitOrFailOnDisconnect = async (ms, phase) => {
    let deadline = monotonicMs() + Math.max(0, ms);
    const assertProviderHealthy = () => {
      if (!providerInfrastructureFailure) return;
      throw new Error(
        `Provider infrastructure failed during ${phase}: ${providerInfrastructureFailure.text}`,
      );
    };
    const assertVlmSelection = () => {
      if (!vlmSelectionViolation) return;
      const payload = vlmSelectionViolation.payload || {};
      throw new Error(
        `VLM selection changed during ${phase}: expected ${args.vlmModel} at ${args.vlmApiBase}, `
        + `observed ${payload.model || '(missing)'} at ${payload.api_base || '(missing)'}`,
      );
    };
    const recoverAndExtend = async () => {
      const recoveryStartedMs = monotonicMs();
      await recoverRtspStream(phase);
      const recoveryMs = monotonicMs() - recoveryStartedMs;
      deadline += recoveryMs;
      streamRecoveryPausedMs += recoveryMs;
    };
    while (true) {
      assertProviderHealthy();
      assertVlmSelection();
      remainingAttemptMs(phase);
      await checkUploadedPlaybackHealth(phase);
      if (streamDisconnected) {
        await recoverAndExtend();
      }
      const remainingMs = deadline - monotonicMs();
      if (remainingMs <= 0) return;
      const waitMs = Math.min(
        remainingMs,
        remainingAttemptMs(phase),
        inputMode === 'upload' && uploadPlaybackHealth ? 500 : Number.POSITIVE_INFINITY,
      );
      const outcome = await Promise.race([
        sleep(waitMs).then(() => 'timer'),
        streamDisconnectPromise.then(() => 'disconnected'),
      ]);
      if (outcome === 'timer') {
        assertProviderHealthy();
        assertVlmSelection();
        await checkUploadedPlaybackHealth(phase);
        if (streamDisconnected) {
          await recoverAndExtend();
        }
        if (deadline - monotonicMs() <= 0) return;
        remainingAttemptMs(phase);
        continue;
      }
      await recoverAndExtend();
    }
  };

  const context = await browser.newContext({
    ignoreHTTPSErrors: !DEFAULT_WEB_TLS_REJECT_UNAUTHORIZED,
    ...(args.webUsername ? {
      httpCredentials: {
        username: args.webUsername,
        password: args.webPassword || '',
      },
    } : {}),
    viewport: { width: args.width, height: args.height },
    recordVideo: {
      dir: taskDir,
      size: { width: args.width, height: args.height },
    },
  });

  await context.addInitScript((sid) => {
    window.__CAPTURE_SESSION_ID = sid;
    const cryptoObj = window.crypto;
    if (!cryptoObj || !cryptoObj.randomUUID) return;
    const realRandomUUID = cryptoObj.randomUUID.bind(cryptoObj);
    let used = false;
    try {
      Object.defineProperty(cryptoObj, 'randomUUID', {
        configurable: true,
        value: () => {
          if (!used) {
            used = true;
            return sid;
          }
          return realRandomUUID();
        },
      });
    } catch {
      // Some browsers may expose randomUUID as non-configurable; in that case
      // the script records the generated session id after page load.
    }
  }, sessionId);

  const page = await context.newPage();
  page.setDefaultTimeout(30000);

  const requestImmediateUploadControlReconnect = (phase) => {
    void page.evaluate(() => {
      if (typeof connectWebSocket !== 'function') return false;
      connectWebSocket();
      return true;
    }).then((requested) => {
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_upload_control_ws_reconnect_requested',
        task_id: task.id,
        phase,
        requested,
        close_count: uploadControlWsCloseCount,
      });
    }).catch((error) => {
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_upload_control_ws_reconnect_request_failed',
        task_id: task.id,
        phase,
        error: error.message,
      });
    });
  };

  page.on('console', (message) => {
    const text = message.text();
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'browser_console',
      level: message.type(),
      text,
    });
    if (/ICE connection state:\s*(disconnected|failed)/i.test(text)) {
      markStreamDisconnected(text);
    } else if (/Error starting RTSP|Failed to connect to RTSP stream/i.test(text)) {
      markStreamStartFailure(text);
    }
  });

  page.on('pageerror', (error) => {
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'browser_page_error',
      error: error.message,
    });
  });

  page.on('request', (request) => {
    const url = request.url();
    if (url.includes('/offer') || url.includes('/api/rtsp/') || url.includes('/api/video/')) {
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'browser_http_request',
        method: request.method(),
        url,
      });
    }
  });

  page.on('response', async (response) => {
    const url = response.url();
    if (url.includes('/offer') || url.includes('/api/rtsp/') || url.includes('/api/video/')) {
      let body = '';
      try {
        body = (await response.text()).slice(0, 2000);
      } catch {
        body = '';
      }
      const parsedBody = tryJson(body);
      if (url.includes('/api/video/upload')) {
        uploadedVideoUploadResponse = parsedBody;
      } else if (url.includes('/api/video/start')) {
        uploadedVideoStartResponse = parsedBody;
        uploadedVideoStartResponseAtMs = monotonicMs();
        if (parsedBody?.rtsp_url) effectiveVideoUrl = parsedBody.rtsp_url;
      }
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'browser_http_response',
        url,
        status: response.status(),
        body,
      });
      if (
        response.status() >= 400 &&
        (url.includes('/offer') || url.includes('/api/rtsp/start') || url.includes('/api/video/upload') || url.includes('/api/video/start'))
      ) {
        markStreamStartFailure(`HTTP ${response.status()}: ${body}`);
      }
    }
  });

  page.on('websocket', (ws) => {
    const wsSessionId = extractSessionIdFromUrl(ws.url());
    let isControlWebSocket = false;
    try {
      isControlWebSocket = new URL(ws.url()).pathname === '/ws';
    } catch {}
    if (wsSessionId) observedSessionId = wsSessionId;
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'browser_ws_open',
      url: ws.url(),
      session_id: wsSessionId,
    });
    ws.on('framesent', (frame) => {
      const parsed = tryJson(frame.payload);
      if (parsed.type === 'update_prompt' || parsed.type === 'prompt_updated') {
        recordPromptEvent('page_ws', 'sent', parsed);
      }
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'browser_ws_frame_sent',
        payload: parsed.type === 'uploaded_video_frame'
          ? {
            ...parsed,
            image: parsed.image ? `[omitted ${String(parsed.image).length} chars]` : '',
          }
          : parsed,
      });
    });
    ws.on('framereceived', (frame) => {
      const parsed = tryJson(frame.payload);
      if (parsed.type === 'vlm_response') {
        recordVlmResponse('page_ws', parsed);
      } else if (parsed.type === 'prompt_updated') {
        recordPromptEvent('page_ws', 'received', parsed);
      } else if (parsed.type === 'model_updated') {
        recordModelUpdate('page_ws', parsed);
      } else if (parsed.type === 'uploaded_video_status') {
        recordUploadedVideoStatus('page_ws', parsed);
      } else if (
        parsed.type === 'background_task_started'
        || parsed.type === 'background_result_ready'
        || parsed.type === 'background_result_error'
      ) {
        recordBackgroundTaskEvent('page_ws', parsed);
      }
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'browser_ws_frame_received',
        payload: parsed,
      });
    });
    ws.on('close', () => {
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'browser_ws_close',
        url: ws.url(),
      });
      if (
        inputMode === 'upload'
        && isControlWebSocket
        && uploadPlaybackHealth
        && taskTimelineEndMs == null
      ) {
        if (uploadControlWsDisconnectedAtMs == null) {
          uploadControlWsDisconnectedAtMs = monotonicMs();
          uploadControlWsCloseCount += 1;
          appendJsonl(events, {
            t_ms: uploadControlWsDisconnectedAtMs - startedAtMs,
            type: 'capture_upload_control_ws_disconnected',
            task_id: task.id,
            close_count: uploadControlWsCloseCount,
            max_reconnects: args.uploadControlReconnects,
          });
        }
        if (targetPlaybackWasStarted) {
          markStreamDisconnected('Uploaded-video control WebSocket closed during single-pass playback');
        } else {
          requestImmediateUploadControlReconnect('control WebSocket close before playback');
        }
      }
    });
  });

  const startRtspAndWaitReady = async () => {
    const maxStartAttempts = args.streamStartRetries + 1;
    let lastFailure = '';
    for (let startAttempt = 1; startAttempt <= maxStartAttempts; startAttempt += 1) {
      if (startAttempt === 1) {
        resetStreamDisconnectSignal();
        resetStreamStartFailureSignal();
        await startRtspFromPage(page);
      } else {
        if (args.streamStartRetryGapS > 0) {
          await sleepWithAttemptDeadline(args.streamStartRetryGapS * 1000, 'RTSP start retry gap');
        }
        if (localPublisher?.restartNow) {
          const restartInfo = await localPublisher.restartNow(
            `initial stream start retry ${startAttempt}`,
            { resetTimeline: true },
          );
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_stream_start_publisher_restarted',
            task_id: task.id,
            start_attempt: startAttempt,
            publisher_restart_info: restartInfo,
            relay_url: effectiveVideoUrl,
          });
          if (args.localRtspReconnectWarmupMs > 0) {
            await sleepWithAttemptDeadline(args.localRtspReconnectWarmupMs, 'RTSP start publisher warmup');
          }
        }
        resetStreamDisconnectSignal();
        resetStreamStartFailureSignal();
        await page.evaluate(async () => {
          if (typeof window.__CAPTURE_CLOSE_PEER_ONLY === 'function') {
            window.__CAPTURE_CLOSE_PEER_ONLY();
          }
        });
        await startRtspFromPage(page);
      }

      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_stream_start_clicked',
        task_id: task.id,
        start_attempt: startAttempt,
        max_start_attempts: maxStartAttempts,
        video_url: effectiveVideoUrl,
        source_video_path: task.local_video_path || undefined,
        session_id: observedSessionId,
      });

      const outcome = await Promise.race([
        waitForStreamingReady(
          page,
          boundedTimeoutMs(args.streamReadyTimeoutS * 1000, 'RTSP initial stream ready wait'),
        )
          .then(() => ({ status: 'ready' }))
          .catch((error) => ({ status: 'timeout', error: error.message })),
        streamStartFailurePromise.then((reason) => ({ status: 'start_failed', error: reason })),
        streamDisconnectPromise.then((reason) => ({ status: 'disconnected', error: reason })),
      ]);
      if (outcome.status === 'ready') return;

      lastFailure = `${outcome.status}${outcome.error ? `: ${outcome.error}` : ''}`;
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_stream_start_attempt_failed',
        task_id: task.id,
        start_attempt: startAttempt,
        max_start_attempts: maxStartAttempts,
        failure: lastFailure,
      });
    }
    throw new Error(`RTSP/WebRTC stream did not become ready after ${maxStartAttempts} start attempt(s): ${lastFailure}`);
  };

  const startUploadedAndWaitRealVideo = async () => {
    const maxStartAttempts = args.streamStartRetries + 1;
    let lastFailure = '';
    for (let startAttempt = 1; startAttempt <= maxStartAttempts; startAttempt += 1) {
      vlmSelectionGuardActive = false;
      vlmSelectionViolation = null;
      resetStreamDisconnectSignal();
      resetStreamStartFailureSignal();
      uploadedAnalysisStartedAtMs = null;
      uploadedPrerollStartedAtMs = null;
      uploadedPrerollSeconds = null;
      uploadedScheduledRealVideoStartAtMs = null;
      uploadedRealVideoStartAtMs = null;
      uploadedStatusReadyInfo = null;
      uploadedVideoStartResponse = null;
      uploadedVideoStartResponseAtMs = null;
      if (startAttempt > 1) {
        if (args.streamStartRetryGapS > 0) {
          await sleep(args.streamStartRetryGapS * 1000);
        }
        await stopPageFromPage(page, 15000).catch((error) => {
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_upload_retry_stop_failed',
            task_id: task.id,
            start_attempt: startAttempt,
            error: error.message,
          });
        });
        await waitForWebSocket(page).catch(() => {});
        observedSessionId = await getPageSessionId(page) || observedSessionId || sessionId;
      }

      const uploadedInfoBefore = await getUploadedVideoInfo(page).catch(() => null);
      if (!uploadedInfoBefore?.upload_id && !uploadedInfoBefore?.uploadId) {
        try {
          const uploadResponse = await uploadVideoApi(
            args.webUrl,
            observedSessionId || sessionId,
            task.local_video_path,
            args.streamReadyTimeoutS * 1000,
          );
          uploadedVideoUploadResponse = uploadResponse;
          await setUploadedVideoInfoFromApi(page, uploadResponse, task.local_video_path);
          const uploadInfo = await getUploadedVideoInfo(page).catch(() => null);
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_upload_file_uploaded',
            task_id: task.id,
            start_attempt: startAttempt,
            upload_method: 'node_multipart',
            upload_info: uploadInfo,
            upload_response: uploadedVideoUploadResponse || undefined,
          });
        } catch (error) {
          lastFailure = `upload_failed: ${error.message}`;
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_upload_start_attempt_failed',
            task_id: task.id,
            start_attempt: startAttempt,
            max_start_attempts: maxStartAttempts,
            failure: lastFailure,
          });
          continue;
        }
      }

      const uploadedInfo = await getUploadedVideoInfo(page).catch(() => null);
      const uploadId = uploadedInfo?.upload_id || uploadedInfo?.uploadId || uploadedVideoUploadResponse?.upload_id || '';
      if (!uploadId) {
        lastFailure = 'upload_failed: missing upload_id after uploadSelectedVideo()';
        appendJsonl(events, {
          t_ms: monotonicMs() - startedAtMs,
          type: 'capture_upload_start_attempt_failed',
          task_id: task.id,
          start_attempt: startAttempt,
          max_start_attempts: maxStartAttempts,
          failure: lastFailure,
        });
        continue;
      }

      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_upload_start_attempt',
        task_id: task.id,
        start_attempt: startAttempt,
        max_start_attempts: maxStartAttempts,
        source_video_path: task.local_video_path,
        session_id: observedSessionId,
      });

      let startApiDone = false;
      let startApiError = null;
      const startApiPromise = startUploadedVideoApi(
        args.webUrl,
        observedSessionId || sessionId,
        uploadId,
        Math.min(args.streamReadyTimeoutS * 1000, 15000),
      ).then((response) => {
        startApiDone = true;
        uploadedVideoStartResponse = response;
        uploadedVideoStartResponseAtMs = monotonicMs();
        if (uploadedVideoStartResponse?.rtsp_url) effectiveVideoUrl = uploadedVideoStartResponse.rtsp_url;
        return response;
      }).catch((error) => {
        startApiDone = true;
        startApiError = error;
        return null;
      });

      try {
        const statusInfo = await waitForUploadedRealVideoStartFromStatus(args.streamReadyTimeoutS * 1000);
        uploadedStatusReadyInfo = statusInfo;
        if (!startApiDone) {
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_upload_api_start_pending_after_status_ready',
            task_id: task.id,
            start_attempt: startAttempt,
          });
        } else if (uploadedVideoStartResponse) {
          appendJsonl(events, {
            t_ms: (uploadedVideoStartResponseAtMs ?? monotonicMs()) - startedAtMs,
            type: 'capture_upload_api_start_response',
            task_id: task.id,
            start_attempt: startAttempt,
            start_response: uploadedVideoStartResponse,
          });
        } else if (startApiError) {
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_upload_api_start_error_before_status_ready',
            task_id: task.id,
            start_attempt: startAttempt,
            error: startApiError.message,
          });
        }
        const readyMs = statusInfo.readyMs;
        await showUploadedLocalPreviewFromPage(page);
        appendJsonl(events, {
          t_ms: readyMs - startedAtMs,
          type: 'capture_upload_real_video_ready_status',
          task_id: task.id,
          start_attempt: startAttempt,
          ready_source: statusInfo.source,
          preroll_started_offset_s: uploadedPrerollStartedAtMs == null ? null : (uploadedPrerollStartedAtMs - startedAtMs) / 1000,
          preroll_seconds: uploadedPrerollSeconds,
          scheduled_real_video_start_offset_s: uploadedScheduledRealVideoStartAtMs == null ? null : (uploadedScheduledRealVideoStartAtMs - startedAtMs) / 1000,
          analysis_started_offset_s: uploadedAnalysisStartedAtMs == null ? null : (uploadedAnalysisStartedAtMs - startedAtMs) / 1000,
          real_video_start_offset_s: (readyMs - startedAtMs) / 1000,
          upload_response: uploadedVideoUploadResponse || undefined,
          start_response: uploadedVideoStartResponse || undefined,
          uploaded_status_count: uploadedVideoStatuses.length,
        });
        return readyMs;
      } catch (error) {
        await startApiPromise.catch(() => null);
        if (uploadedVideoStartResponse) {
          appendJsonl(events, {
            t_ms: (uploadedVideoStartResponseAtMs ?? monotonicMs()) - startedAtMs,
            type: 'capture_upload_api_start_response',
            task_id: task.id,
            start_attempt: startAttempt,
            start_response: uploadedVideoStartResponse,
          });
        } else if (startApiError) {
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_upload_api_start_error',
            task_id: task.id,
            start_attempt: startAttempt,
            error: startApiError.message,
          });
        }
        lastFailure = `status_failed: ${error.message}`;
      }

      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_upload_real_video_attempt_failed',
        task_id: task.id,
        start_attempt: startAttempt,
        max_start_attempts: maxStartAttempts,
        failure: lastFailure,
      });
      await stopActiveRtspStreams(
        args.webUrl,
        events,
        startedAtMs,
        task,
        `upload_start_attempt_${startAttempt}_failed`,
        15000,
        auth,
      );
    }
    throw new Error(`Uploaded video did not reach real-video analysis after ${maxStartAttempts} start attempt(s): ${lastFailure}`);
  };

  const startUploadedAndWaitLoopingVideo = async () => {
    const maxStartAttempts = args.streamStartRetries + 1;
    let lastFailure = '';
    for (let startAttempt = 1; startAttempt <= maxStartAttempts; startAttempt += 1) {
      if (vlmSelectionViolation) {
        throw new Error('VLM selection changed after API readiness and before target-video upload');
      }
      resetStreamDisconnectSignal();
      resetStreamStartFailureSignal();
      uploadedAnalysisStartedAtMs = null;
      uploadedRealVideoStartAtMs = null;
      uploadedStatusReadyInfo = null;
      uploadedVideoUploadResponse = null;
      uploadedVideoStartResponse = null;
      uploadedVideoStartResponseAtMs = null;
      targetUploadStartedAtMs = null;
      targetUploadReadyAtMs = null;
      targetUploadVerification = null;
      officialPlaybackRequestedAtMs = null;
      officialPlaybackStartedAtMs = null;
      officialPlaybackFirstFrameAtMs = null;
      officialPlaybackAnalyzerReadyAtMs = null;

      if (startAttempt > 1) {
        if (args.streamStartRetryGapS > 0) {
          await sleepWithAttemptDeadline(
            args.streamStartRetryGapS * 1000,
            'uploaded-video start retry gap',
          );
        }
        await stopPageFromPage(page, 30000).catch(() => {});
        await waitForWebSocket(page, 30000).catch(() => {});
        observedSessionId = await getPageSessionId(page) || observedSessionId || sessionId;
      }

      const activeUploadSessionId = await getPageSessionId(page) || observedSessionId || sessionId;
      observedSessionId = activeUploadSessionId;
      if (
        args.vlmModel
        && args.vlmWarmup
        && webUiWarmupSessionId !== activeUploadSessionId
      ) {
        vlmSelectionGuardActive = false;
        if (enforceVlmSelection) {
          await enforceVlmSelection('same-session readiness refresh before target upload');
        }
        await performVlmInferenceWarmup('same-session readiness refresh before target upload');
        vlmSelectionViolation = null;
        vlmSelectionGuardActive = true;
      }

      const uploadStartedMs = monotonicMs();
      targetUploadStartedAtMs = uploadStartedMs;
      appendJsonl(events, {
        t_ms: uploadStartedMs - startedAtMs,
        type: 'capture_upload_start_attempt',
        task_id: task.id,
        start_attempt: startAttempt,
        max_start_attempts: maxStartAttempts,
        source_video_path: task.local_video_path,
        upload_video_path: uploadSourcePath || task.local_video_path,
        session_id: observedSessionId,
      });

      try {
        const targetUploadPath = uploadSourcePath || task.local_video_path;
        const targetUploadStat = await fsp.stat(targetUploadPath);
        await prepareUploadedVideoSource(page, targetUploadPath);
        const uploadInfo = await uploadSelectedVideoFromPage(
          page,
          boundedTimeoutMs(args.videoUploadTimeoutS * 1000, 'local-video upload'),
        );
        const pageUploadedInfo = await getUploadedVideoInfo(page).catch(() => null);
        const uploadId = uploadInfo?.upload_id
          || uploadInfo?.uploadId
          || uploadInfo?.data?.upload_id
          || pageUploadedInfo?.upload_id
          || '';
        uploadedVideoUploadResponse = {
          ...(pageUploadedInfo || {}),
          ...(uploadInfo || {}),
          upload_id: uploadId,
        };
        const uploadFinishedMs = monotonicMs();
        if (!uploadId) {
          throw new Error('WebUI upload completed without a reusable upload_id');
        }
        const reportedUploadBytes = Number(
          uploadedVideoUploadResponse.size_bytes
          ?? uploadedVideoUploadResponse.sizeBytes,
        );
        if (!Number.isSafeInteger(reportedUploadBytes) || reportedUploadBytes < 1) {
          throw new Error('WebUI upload response did not report the stored MP4 byte count');
        }
        if (reportedUploadBytes !== targetUploadStat.size) {
          throw new Error(
            `WebUI stored MP4 byte count mismatch: local=${targetUploadStat.size}, `
            + `remote=${reportedUploadBytes}`,
          );
        }
        targetUploadVerification = {
          ok: true,
          complete_file_uploaded: true,
          local_path: targetUploadPath,
          local_size_bytes: targetUploadStat.size,
          remote_size_bytes: reportedUploadBytes,
          upload_id: uploadId,
        };
        targetUploadReadyAtMs = uploadFinishedMs;
        appendJsonl(events, {
          t_ms: uploadFinishedMs - startedAtMs,
          type: 'capture_upload_file_uploaded',
          task_id: task.id,
          start_attempt: startAttempt,
          upload_method: 'page_native',
          upload_elapsed_s: (uploadFinishedMs - uploadStartedMs) / 1000,
          complete_file_verification: targetUploadVerification,
          upload_info: uploadedVideoUploadResponse,
        });

        if (
          args.vlmModel
          && args.vlmWarmup
          && (
            latestVlmWarmupReadyAtMs == null
            || latestVlmWarmupReadyAtMs > targetUploadStartedAtMs
          )
        ) {
          throw new Error('Target video was uploaded before the VLM API readiness barrier');
        }
        if (vlmSelectionViolation) {
          throw new Error('VLM selection changed while the target video was being uploaded');
        }

        appendJsonl(events, {
          t_ms: uploadFinishedMs - startedAtMs,
          type: 'capture_recording_start_barrier_satisfied',
          task_id: task.id,
          protocol: CURRENT_RECORDING_PROTOCOL,
          model_ready_offset_s: latestVlmWarmupReadyAtMs == null
            ? null
            : (latestVlmWarmupReadyAtMs - startedAtMs) / 1000,
          webui_warmup_session_id: webUiWarmupSessionId || null,
          target_session_id: observedSessionId || null,
          webui_same_session_model_ready: Boolean(
            webUiWarmupSessionId && webUiWarmupSessionId === observedSessionId
          ),
          target_upload_started_offset_s: (targetUploadStartedAtMs - startedAtMs) / 1000,
          target_upload_ready_offset_s: (targetUploadReadyAtMs - startedAtMs) / 1000,
          target_video_stream_started: false,
          seek_allowed_after_stream_start: false,
        });

        if (firstQueryFrameGate.required) {
          await discardFirstQueryFrameGate(page);
          const installedGate = await installFirstQueryFrameGate(page);
          firstQueryFrameGate = {
            ...firstQueryFrameGate,
            ...installedGate,
            installed_offset_s: (monotonicMs() - startedAtMs) / 1000,
          };
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_first_query_frame_gate_installed',
            task_id: task.id,
            first_query_time_s: firstQueryTimeS,
            process_interval_s: effectiveProcessIntervalS,
            gate: firstQueryFrameGate,
          });
        }

        if (!recordingClockMarkerRemoval) {
          recordingClockMarkerRemoval = await removeRecordingClockMarker(page);
          if (
            recordingClockMarkerRemoval.removed !== true
            || recordingClockMarkerRemoval.absent !== true
          ) {
            throw new Error('Pre-playback WebM clock marker was unavailable at removal time');
          }
          appendJsonl(events, {
            t_ms: recordingClockMarkerRemoval.removed_at_ms - startedAtMs,
            type: 'capture_recording_clock_marker_removed',
            task_id: task.id,
            protocol: 'pre_playback_visual_marker_v1',
            removed_event_offset_s: (
              recordingClockMarkerRemoval.removed_at_ms - startedAtMs
            ) / 1000,
            request_offset_s: (
              recordingClockMarkerRemoval.requested_at_ms - startedAtMs
            ) / 1000,
            acknowledgement_offset_s: (
              recordingClockMarkerRemoval.acknowledged_at_ms - startedAtMs
            ) / 1000,
            page_clock_round_trip_ms: (
              recordingClockMarkerRemoval.page_clock_sample.round_trip_ms
            ),
          });
        }

        markPlaybackStartRequested(`uploaded playback start attempt ${startAttempt}`);
        await startUploadedVideoFromPage(
          page,
          boundedTimeoutMs(args.streamReadyTimeoutS * 1000, 'uploaded-video playback start'),
        );
        const continuityGuard = await installUploadedPlaybackContinuityGuard(page);
        const playbackState = await waitForUploadedSinglePassPlaybackStart(
          page,
          boundedTimeoutMs(args.streamReadyTimeoutS * 1000, 'uploaded-video first single-pass frame'),
        );
        const observedAtMs = monotonicMs();
        const playbackStartMs = observedAtMs - playbackState.current_time_s * 1000;
        officialPlaybackStartedAtMs = playbackStartMs;
        officialPlaybackFirstFrameAtMs = observedAtMs;
        uploadedRealVideoStartAtMs = playbackStartMs;
        uploadPlaybackHealth = {
          started_at_ms: playbackStartMs,
          checks: 0,
          loop_count: 0,
          last_timeline_media_time_s: playbackState.current_time_s,
          last_timeline_time_s: playbackState.current_time_s,
          last_media_time_s: playbackState.current_time_s,
          last_media_progress_at_ms: observedAtMs,
          last_frame_sequence: playbackState.frame_sequence,
          last_frame_progress_at_ms: observedAtMs,
          unhealthy_since_ms: null,
          max_media_stall_ms: 0,
          max_frame_stall_ms: 0,
          last_state: playbackState,
          control_ws_reconnects_allowed: args.uploadControlReconnects,
          control_ws_reconnect_timeout_s: args.uploadControlReconnectTimeoutS,
        };
        uploadedStatusReadyInfo = {
          source: 'webui_session_ready_then_single_pass_uploaded_playback',
          readyMs: playbackStartMs,
          observedAtMs,
          playbackState,
          playbackSync: {
            mode: 'single_pass_no_seek',
            seek_performed: false,
            playback_request_at_ms: officialPlaybackRequestedAtMs,
            playback_start_at_ms: officialPlaybackStartedAtMs,
            first_rendered_frame_at_ms: officialPlaybackFirstFrameAtMs,
          },
          analyzerReadyInfo: officialPlaybackAnalyzerReadyAtMs == null ? undefined : {
            readyMs: officialPlaybackAnalyzerReadyAtMs,
            source: 'analysis_started',
          },
          continuityGuard,
        };
        appendJsonl(events, {
          t_ms: playbackStartMs - startedAtMs,
          type: 'capture_upload_single_pass_playback_started',
          task_id: task.id,
          start_attempt: startAttempt,
          playback_start_offset_s: (playbackStartMs - startedAtMs) / 1000,
          observed_offset_s: (observedAtMs - startedAtMs) / 1000,
          playback_state: playbackState,
          playback_sync: uploadedStatusReadyInfo.playbackSync,
          continuity_guard: continuityGuard,
          start_response: uploadedVideoStartResponse || undefined,
        });
        return playbackStartMs;
      } catch (error) {
        if (error.code === 'VLM_WARMUP_FAILED') throw error;
        await discardFirstQueryFrameGate(page);
        lastFailure = error.message;
        appendJsonl(events, {
          t_ms: monotonicMs() - startedAtMs,
          type: 'capture_upload_real_video_attempt_failed',
          task_id: task.id,
          start_attempt: startAttempt,
          max_start_attempts: maxStartAttempts,
          failure: lastFailure,
        });
        if (targetPlaybackWasStarted) {
          throw new Error(
            `Single-pass uploaded playback failed after stream start; discard this attempt: ${lastFailure}`,
          );
        }
        await stopPageFromPage(page, 30000).catch(() => {});
      }
    }
    throw new Error(
      `Uploaded video did not reach stable looping playback after ${maxStartAttempts} attempt(s): ${lastFailure}`,
    );
  };

  try {
    if (args.sourceDecodeCheck && task.local_video_path) {
      sourceVideoValidation = await validateSourceVideoCached(task.local_video_path, args.ffmpegBin);
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_source_video_validation',
        task_id: task.id,
        validation: sourceVideoValidation,
      });
      if (!sourceVideoValidation.ok) {
        throw new Error(
          `Local source video failed full decode validation: ${sourceVideoValidation.path}: `
          + `${sourceVideoValidation.error || 'unknown decode error'}`,
        );
      }
    }
    console.log(`[${index + 1}/${total}] ${task.id}: open ${args.webUrl}`);
    await page.goto(args.webUrl, {
      waitUntil: 'domcontentloaded',
      timeout: boundedTimeoutMs(60000, 'page goto'),
    });
    await page.waitForSelector('#bigStartBtn', {
      state: 'visible',
      timeout: boundedTimeoutMs(30000, 'wait for start button'),
    });
    await installCapturePageHooks(page);
    if (inputMode === 'upload') {
      const markerInstallRequestedAtMs = monotonicMs();
      recordingClockMarkerInstall = await installRecordingClockMarker(page);
      const markerInstallAcknowledgedAtMs = monotonicMs();
      if (
        recordingClockMarkerInstall.installed !== true
        || recordingClockMarkerInstall.visible !== true
      ) {
        throw new Error('Unable to install the pre-playback WebM clock marker');
      }
      recordingClockMarkerInstall = {
        ...recordingClockMarkerInstall,
        requested_at_ms: markerInstallRequestedAtMs,
        acknowledged_at_ms: markerInstallAcknowledgedAtMs,
      };
      appendJsonl(events, {
        t_ms: markerInstallAcknowledgedAtMs - startedAtMs,
        type: 'capture_recording_clock_marker_installed',
        task_id: task.id,
        protocol: 'pre_playback_visual_marker_v1',
        request_offset_s: (markerInstallRequestedAtMs - startedAtMs) / 1000,
        acknowledgement_offset_s: (markerInstallAcknowledgedAtMs - startedAtMs) / 1000,
        marker: {
          x: RECORDING_CLOCK_MARKER.x,
          y: args.height - RECORDING_CLOCK_MARKER.size,
          size: RECORDING_CLOCK_MARKER.size,
        },
      });
    }
    observedSessionId = await getPageSessionId(page) || observedSessionId || sessionId;
    if (inputMode === 'upload') {
      await waitForWebSocket(page, 60000).catch((error) => {
        appendJsonl(events, {
          t_ms: monotonicMs() - startedAtMs,
          type: 'page_ws_initial_wait_failed',
          task_id: task.id,
          error: error.message,
        });
      });
      observedSessionId = await getPageSessionId(page) || observedSessionId || sessionId;
    } else {
      await waitForWebSocket(page);
      observedSessionId = await getPageSessionId(page) || observedSessionId || sessionId;
    }

    if (args.vlmModel) {
      const vlmConfig = {
        model: args.vlmModel,
        apiBase: args.vlmApiBase,
        apiKey: args.vlmApiKey,
      };
      enforceVlmSelection = async (reason) => {
        const maxApplications = 4;
        for (let application = 1; application <= maxApplications; application += 1) {
          await setVlmControlsInPage(page, vlmConfig);
          const requestedAtMs = monotonicMs();
          await sendControlMessage({
            type: 'update_model',
            model: args.vlmModel,
            api_base: args.vlmApiBase,
            api_key: args.vlmApiKey,
          });
          const acknowledgement = await waitForModelUpdate(
            requestedAtMs,
            boundedTimeoutMs(30000, `VLM model update during ${reason}`),
          );

          let override = null;
          const stableUntilMs = monotonicMs() + 1000;
          while (monotonicMs() < stableUntilMs) {
            const latest = modelUpdateEvents[modelUpdateEvents.length - 1];
            if (latest && !modelUpdateMatchesExpected(latest.payload)) {
              override = latest;
              break;
            }
            await sleep(50);
          }
          if (!override) {
            await setVlmControlsInPage(page, vlmConfig);
            vlmSelectionViolation = null;
            appendJsonl(events, {
              t_ms: monotonicMs() - startedAtMs,
              type: 'capture_vlm_selection_locked',
              task_id: task.id,
              reason,
              application,
              model: args.vlmModel,
              api_base: args.vlmApiBase,
            });
            return acknowledgement;
          }
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_vlm_selection_overridden_before_playback',
            task_id: task.id,
            reason,
            application,
            observed: override.payload,
          });
        }
        throw new Error(
          `WebUI repeatedly overrode VLM selection ${args.vlmModel} during ${reason}`,
        );
      };

      const modelUpdate = await enforceVlmSelection('initial page configuration');
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_vlm_configured',
        task_id: task.id,
        model: args.vlmModel,
        api_base: args.vlmApiBase,
        acknowledgement: modelUpdate,
      });
      console.log(`[${index + 1}/${total}] ${task.id}: VLM switched to ${args.vlmModel}`);
    }

    await sendControlMessage({
      type: 'set_debug',
      show_request_payload: true,
      show_response_payload: true,
      show_memory_state: true,
    });

    if (processIntervalS) {
      await page.fill('#processEvery', String(processIntervalS));
      await page.locator('#processEvery').dispatchEvent('change');
      if (inputMode === 'upload') {
        await sendControlMessage({
          type: 'update_processing',
          process_interval: Number(processIntervalS),
        });
      }
    }
    if (framesPerBatch) {
      await page.fill('#framesPerBatch', String(framesPerBatch));
      await page.locator('#framesPerBatch').dispatchEvent('change');
      if (inputMode === 'upload') {
        await sendControlMessage({
          type: 'update_frames_per_batch',
          frames_per_batch: Number(framesPerBatch),
        });
      }
    }
    if (enforceVlmSelection) {
      await enforceVlmSelection('page initialization settled');
    }

    // Fail closed before the evaluated video is selected or uploaded. A successful
    // model-update acknowledgement is not enough; require one real visual inference.
    if (args.vlmModel) {
      await performVlmInferenceWarmup('API readiness before target-video upload');
      if (enforceVlmSelection && vlmSelectionViolation) {
        throw new Error('VLM selection changed during the API readiness barrier');
      }
      vlmSelectionViolation = null;
      vlmSelectionGuardActive = true;
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_vlm_api_ready_before_target_video',
        task_id: task.id,
        model: args.vlmModel,
        api_base: args.vlmApiBase,
        target_video_selected_or_uploaded: false,
      });
    }

    let streamReadyMs = null;
    if (inputMode === 'upload') {
      const sourcePath = path.resolve(task.local_video_path);
      if (!fs.existsSync(sourcePath)) {
        throw new Error(`Local video not found for ${task.id}: ${sourcePath}`);
      }
      const preparedUpload = await prepareLocalUploadVideo(
        args,
        task,
        sourcePath,
        taskDir,
        events,
        startedAtMs,
      );
      uploadSourcePath = preparedUpload.inputPath;
      await prepareUploadedVideoSource(page, uploadSourcePath);
      localUploadInfo = {
        mode: 'webui_upload',
        source_video_path: sourcePath,
        upload_video_path: uploadSourcePath,
        source_video_codec: preparedUpload.sourceCodec,
        transcoded: preparedUpload.transcoded,
        cache_path: preparedUpload.cachePath || undefined,
        preprocess_log_path: preparedUpload.logPath || undefined,
        web_url: args.webUrl,
      };
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_upload_source_ready',
        task_id: task.id,
        source_video_path: sourcePath,
        upload_video_path: uploadSourcePath,
        source_video_codec: preparedUpload.sourceCodec,
        transcoded: preparedUpload.transcoded,
      });
    } else if (task.local_video_path) {
      const relay = buildRelayEndpoints(args, task, attempt);
      effectiveVideoUrl = relay.playbackUrl;
      localPublisher = await startLocalRtspPublisher(args, task, relay, taskDir, events, startedAtMs);
      const publisherReadyInfo = await localPublisher.waitUntilReady(
        boundedTimeoutMs(90000, 'local relay publisher ready wait'),
      );
      if (!publisherReadyInfo) {
        throw new Error(
          `Local ${localPublisher.publishProtocol.toUpperCase()} publisher did not become ready within 90s: `
          + localPublisher.publishUrl,
        );
      }
      localRelayInfo = {
        source_video_path: localPublisher.sourcePath,
        publisher_input_path: localPublisher.publisherInputPath,
        replay_cache_path: localPublisher.replayCachePath || undefined,
        replay_preprocess_log_path: localPublisher.replayPreprocessLogPath || undefined,
        preroll_s: localPublisher.prerollS,
        postroll_s: localPublisher.postrollS,
        replay_duration_s: localPublisher.replayDurationS || undefined,
        initial_replay_offset_s: localPublisher.initialReplayOffsetS,
        relay_path: localPublisher.relayPath,
        publish_protocol: localPublisher.publishProtocol,
        publish_url: localPublisher.publishUrl,
        playback_url: localPublisher.playbackUrl,
        log_path: localPublisher.logPath,
        keepalive_url: localPublisher.keepaliveUrl,
        keepalive_log_path: localPublisher.keepaliveLogPath,
        pid: localPublisher.proc.pid,
        controlled_frame_relay: localPublisher.controlledFrameRelay,
        publisher_ready: publisherReadyInfo,
        timeline_start_offset_s: (localPublisher.timelineStartMs - startedAtMs) / 1000,
        warmup_ms: args.localRtspWarmupMs,
        reconnect_warmup_ms: args.localRtspReconnectWarmupMs,
        connect_lead_s: args.localRtspConnectLeadS,
      };
      if (args.localRtspWarmupMs > 0) {
        await sleepWithAttemptDeadline(args.localRtspWarmupMs, 'local RTSP initial warmup');
      }
      if (localPublisher.prerollS > 0 && args.localRtspConnectLeadS > 0) {
        const connectAtMs = localPublisher.timelineStartMs
          + localPublisher.prerollS * 1000
          - args.localRtspConnectLeadS * 1000;
        const delayMs = Math.max(0, connectAtMs - monotonicMs());
        appendJsonl(events, {
          t_ms: monotonicMs() - startedAtMs,
          type: 'local_rtsp_delayed_browser_connect',
          task_id: task.id,
          preroll_s: localPublisher.prerollS,
          connect_lead_s: args.localRtspConnectLeadS,
          publisher_timeline_start_offset_s: (localPublisher.timelineStartMs - startedAtMs) / 1000,
          browser_connect_target_offset_s: (connectAtMs - startedAtMs) / 1000,
          wait_before_browser_connect_s: delayMs / 1000,
        });
        if (delayMs > 0) await sleepWithAttemptDeadline(delayMs, 'local RTSP delayed browser connect');
      }
    }

    if (inputMode === 'upload') {
      streamReadyMs = await startUploadedAndWaitLoopingVideo();
      const uploadedInfo = await getUploadedVideoInfo(page).catch(() => null);
      localUploadInfo = {
        ...(localUploadInfo || {}),
        aux_websocket_url: auxWsUrl || undefined,
        uploaded_video: uploadedInfo || undefined,
        upload_response: uploadedVideoUploadResponse || undefined,
        start_response: uploadedVideoStartResponse || undefined,
        ready_status: uploadedStatusReadyInfo || undefined,
        uploaded_video_statuses: uploadedVideoStatuses,
      };
      if (uploadedVideoStartResponse?.rtsp_url) {
        effectiveVideoUrl = uploadedVideoStartResponse.rtsp_url;
      }
    } else {
      await selectRtspSource(page, effectiveVideoUrl);
      if (enforceVlmSelection) {
        vlmSelectionGuardActive = false;
        vlmSelectionViolation = null;
        await enforceVlmSelection('RTSP playback start');
        vlmSelectionViolation = null;
        vlmSelectionGuardActive = true;
      }
      markPlaybackStartRequested('RTSP playback start');
      await startRtspAndWaitReady();
      const rtspPlaybackState = await waitForVideoPlaybackReady(
        page,
        'rtsp',
        boundedTimeoutMs(args.streamReadyTimeoutS * 1000, 'RTSP first rendered frame'),
      );
      streamReadyMs = monotonicMs();
      appendJsonl(events, {
        t_ms: streamReadyMs - startedAtMs,
        type: 'capture_rtsp_first_rendered_frame_ready',
        task_id: task.id,
        playback_state: rtspPlaybackState,
      });
    }
    streamReadyOffsetS = (streamReadyMs - startedAtMs) / 1000;
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'capture_stream_ready',
      task_id: task.id,
      stream_ready_offset_s: streamReadyOffsetS,
      input_mode: inputMode,
    });

    let originalStartMs = null;
    if (localPublisher?.prerollS > 0) {
      originalStartMs = localPublisher.timelineStartMs + localPublisher.prerollS * 1000;
      originalVideoStartOffsetS = (originalStartMs - startedAtMs) / 1000;
      const firstTargetQueryMs = originalStartMs + Math.max(0, queryRounds[0].query_time_s * 1000);
      const waitBeforeFirstQueryMs = Math.max(0, firstTargetQueryMs - monotonicMs());
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'local_rtsp_preroll_timing',
        task_id: task.id,
        publisher_timeline_start_offset_s: (localPublisher.timelineStartMs - startedAtMs) / 1000,
        publisher_start_offset_s: (localPublisher.lastStartMs - startedAtMs) / 1000,
        original_video_start_offset_s: originalVideoStartOffsetS,
        stream_ready_offset_s: streamReadyOffsetS,
        query_time_s: task.query_time_s,
        query_rounds: queryRounds,
        wait_before_query_s: waitBeforeFirstQueryMs / 1000,
        preroll_s: localPublisher.prerollS,
        preroll_missed: streamReadyMs > originalStartMs,
      });
      if (streamReadyMs > originalStartMs) {
        throw new Error(
          `RTSP preroll missed original video start for ${task.id}: ` +
          `stream_ready=${((streamReadyMs - startedAtMs) / 1000).toFixed(3)}s, ` +
          `original_start=${((originalStartMs - startedAtMs) / 1000).toFixed(3)}s; ` +
          `increase --local-rtsp-preroll-s`,
        );
      }
    }
    const currentQueryBaseMs = () => (
      localPublisher?.prerollS > 0
        ? localPublisher.timelineStartMs + localPublisher.prerollS * 1000
        : (originalStartMs ?? streamReadyMs)
    );
    const uploadedTimelineTimeS = (state, phase) => observeUploadedTimeline(state, phase);
    const waitUntilVideoTime = async (videoTimeS, phase, maxWaitMs = 0) => {
      const waitStartedMs = monotonicMs();
      if (inputMode === 'upload') {
        const targetS = Math.max(0, videoTimeS);
        while (true) {
          const state = await getVideoPlaybackState(page);
          const timelineS = uploadedTimelineTimeS(state, `wait:${phase}`);
          if (Number.isFinite(timelineS) && timelineS >= targetS) {
            return monotonicMs() - waitStartedMs;
          }
          const remainingMediaMs = Number.isFinite(timelineS)
            ? Math.max(20, (targetS - timelineS) * 1000)
            : 250;
          let waitMs = Math.min(250, remainingMediaMs);
          if (maxWaitMs > 0) {
            const budgetMs = maxWaitMs - (monotonicMs() - waitStartedMs);
            if (budgetMs <= 0) return monotonicMs() - waitStartedMs;
            waitMs = Math.min(waitMs, budgetMs);
          }
          await waitOrFailOnDisconnect(waitMs, phase);
        }
      }
      while (true) {
        const remainingMs = currentQueryBaseMs() + Math.max(0, videoTimeS * 1000) - monotonicMs();
        if (remainingMs <= 20) return monotonicMs() - waitStartedMs;
        if (maxWaitMs > 0) {
          const budgetMs = maxWaitMs - (monotonicMs() - waitStartedMs);
          if (budgetMs <= 0) return monotonicMs() - waitStartedMs;
          await waitOrFailOnDisconnect(Math.min(remainingMs, budgetMs), phase);
        } else {
          await waitOrFailOnDisconnect(remainingMs, phase);
        }
      }
    };
    const useQueryTimeFrameQueue = inputMode === 'upload' && providerWarmupRequired;
    for (let queryIndex = 0; queryIndex < queryRounds.length; queryIndex += 1) {
      const round = queryRounds[queryIndex];
      if ((queryDispatchCounts.get(round.id) || 0) !== 0) {
        throw new Error(`Refusing to deliver Query ${round.id} more than once`);
      }
      let queryFrameGateInstall = null;
      let queryFramePrearmWaitMs = 0;
      if (useQueryTimeFrameQueue) {
        const prearmLeadS = Math.min(
          Math.max(0, Number(round.query_time_s)),
          effectiveProcessIntervalS + 0.05,
        );
        const prearmAtS = Math.max(0, Number(round.query_time_s) - prearmLeadS);
        queryFramePrearmWaitMs = await waitUntilVideoTime(
          prearmAtS,
          `query frame gate pre-arm ${round.id}`,
        );
        queryFrameGateInstall = await installFirstQueryFrameGate(page);
        appendJsonl(events, {
          t_ms: monotonicMs() - startedAtMs,
          type: 'capture_query_frame_gate_prearmed',
          task_id: task.id,
          query_id: round.id,
          annotated_query_time_s: Number(round.query_time_s),
          prearm_at_s: prearmAtS,
          prearm_lead_s: prearmLeadS,
          gate: queryFrameGateInstall,
        });
      }
      const waitAtQueryMs = await waitUntilVideoTime(round.query_time_s, `query wait ${round.id}`);
      const waitBeforeQueryMs = queryFramePrearmWaitMs + waitAtQueryMs;
      const playbackStateAtQuery = inputMode === 'upload'
        ? await getVideoPlaybackState(page)
        : await waitForVideoPlaybackReady(
          page,
          inputMode,
          boundedTimeoutMs(args.streamReadyTimeoutS * 1000, `video readiness before query ${round.id}`),
        );
      if (
        inputMode === 'upload'
        && (
          playbackStateAtQuery.active_source !== 'upload'
          || playbackStateAtQuery.paused
          || playbackStateAtQuery.ended
          || playbackStateAtQuery.ready_state < 2
        )
      ) {
        throw new Error(
          `Single-pass uploaded playback was not ready at query ${round.id}: `
          + JSON.stringify(playbackStateAtQuery),
        );
      }
      let queryTimeFrame = null;
      if (useQueryTimeFrameQueue) {
        queryTimeFrame = await captureDisplayedQueryFrameAtDispatch(page);
      }
      const queryDispatchVideoTimeS = inputMode === 'upload'
        ? (
          queryTimeFrame
            ? uploadedTimelineTimeS({
              ...playbackStateAtQuery,
              current_time_s: queryTimeFrame.dispatch_media_time_s,
            }, `query-dispatch:${round.id}`)
            : uploadedTimelineTimeS(playbackStateAtQuery, `query-dispatch:${round.id}`)
        )
        : null;
      let sentPrompt = round.query;
      let promptRequestedAtMs = null;
      if (inputMode === 'upload') {
        sentPrompt = await preparePromptInPage(page, round.query);
        promptRequestedAtMs = monotonicMs();
        await sendControlMessage({ type: 'update_prompt', prompt: sentPrompt });
      } else {
        sentPrompt = await sendPromptFromPage(page, round.query);
      }
      const querySentAtMs = monotonicMs();
      const querySentAtIso = new Date().toISOString();
      queryDispatchCounts.set(round.id, 1);
      const queryQueuePromise = useQueryTimeFrameQueue
        ? requestScaffoldQueryEvent(args, {
          session_id: await getPageSessionId(page) || observedSessionId || sessionId,
          query_event_id: `${task.id}:${round.id}`,
          query: sentPrompt,
          image_url: queryTimeFrame.image_url,
          frame_time_range: `${Number(queryDispatchVideoTimeS).toFixed(3)} seconds`,
          ui_query_sent_at: querySentAtIso,
          ui_query_video_time_s: Number(queryDispatchVideoTimeS),
          annotated_query_video_time_s: Number(round.query_time_s),
          captured_media_time_s: Number(queryDispatchVideoTimeS),
          captured_raw_media_time_s: queryTimeFrame.media_time_s,
          query_index: queryIndex + 1,
          query_total: queryRounds.length,
        }, boundedTimeoutMs(30000, `Query-time frame queue for ${round.id}`))
        : null;
      const playbackStateAfterQuery = inputMode === 'upload'
        ? await getVideoPlaybackState(page).catch(() => playbackStateAtQuery)
        : playbackStateAtQuery;
      let promptAcknowledgement = null;
      let frameGateRelease = null;
      let queryFrameQueueAcknowledgement = null;
      if (inputMode === 'upload') {
        [promptAcknowledgement, queryFrameQueueAcknowledgement] = await Promise.all([
          waitForPromptUpdate(
            promptRequestedAtMs,
            sentPrompt,
            boundedTimeoutMs(10000, `WebUI prompt acknowledgement for ${round.id}`),
          ),
          queryQueuePromise || Promise.resolve(null),
        ]);
        if (useQueryTimeFrameQueue || (queryIndex === 0 && firstQueryFrameGate.required)) {
          frameGateRelease = await releaseFirstQueryFrameGate(page);
          if (!frameGateRelease.released && !frameGateRelease.disarmed_without_frame) {
            throw new Error(`Query frame gate was not released: ${frameGateRelease.reason}`);
          }
        }
        if (queryIndex === 0 && firstQueryFrameGate.required) {
          if (!frameGateRelease?.released) {
            throw new Error('The first Query had no gated target frame to release');
          }
          firstQueryFrameGate = {
            ...firstQueryFrameGate,
            ...frameGateRelease,
            prompt_ack_offset_s: promptAcknowledgement.t_ms / 1000,
            prompt_ack_latency_s: (
              startedAtMs + promptAcknowledgement.t_ms - promptRequestedAtMs
            ) / 1000,
            release_offset_s: (monotonicMs() - startedAtMs) / 1000,
          };
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_first_query_frame_gate_released',
            task_id: task.id,
            query_id: round.id,
            query_sent_offset_s: (querySentAtMs - startedAtMs) / 1000,
            gate: firstQueryFrameGate,
          });
        }
        if (useQueryTimeFrameQueue) {
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_query_time_frame_queued',
            task_id: task.id,
            query_id: round.id,
            query_event_id: `${task.id}:${round.id}`,
            query_sent_offset_s: (querySentAtMs - startedAtMs) / 1000,
            captured_media_time_s: queryDispatchVideoTimeS,
            captured_raw_media_time_s: queryTimeFrame.media_time_s,
            dispatch_media_time_s: queryDispatchVideoTimeS,
            annotated_query_time_s: Number(round.query_time_s),
            frame_time_range: `${Number(queryDispatchVideoTimeS).toFixed(3)} seconds`,
            frame_width: queryTimeFrame.width,
            frame_height: queryTimeFrame.height,
            queue_acknowledgement: queryFrameQueueAcknowledgement,
            gate_install: queryFrameGateInstall,
            gate_release: frameGateRelease,
          });
        }
      }
      const roundQuerySentOffsetS = (querySentAtMs - startedAtMs) / 1000;
      const roundQueryVideoTimeS = inputMode === 'upload'
        ? queryDispatchVideoTimeS
        : (querySentAtMs - currentQueryBaseMs()) / 1000;
      const playbackStateWithTimeline = inputMode === 'upload'
        ? {
          ...playbackStateAfterQuery,
          loop_count: uploadPlaybackHealth?.loop_count || 0,
          timeline_time_s: roundQueryVideoTimeS,
        }
        : playbackStateAtQuery;
      const latestTimedResponse = [...vlmResponses].reverse().find((response) => (
        parseFrameTimeRangeS(response.request_payload?.frame_time_range) != null
      ));
      const latestResponseFrameTimeS = latestTimedResponse
        ? parseFrameTimeRangeS(latestTimedResponse.request_payload?.frame_time_range)
        : null;
      const expectedPromptedFrameTimeS = inputMode === 'rtsp'
        ? (
          latestTimedResponse && latestResponseFrameTimeS != null
            ? latestResponseFrameTimeS
              + (querySentAtMs - startedAtMs - latestTimedResponse.t_ms) / 1000
            : (querySentAtMs - streamReadyMs) / 1000
        )
        : round.query_time_s;
      if (queryIndex === 0) {
        querySentOffsetS = roundQuerySentOffsetS;
        queryVideoTimeS = roundQueryVideoTimeS;
      }
      const queryEvent = {
        id: round.id,
        query_index: queryIndex + 1,
        query_total: queryRounds.length,
        query: sentPrompt,
        query_time_s: round.query_time_s,
        query_sent_offset_s: roundQuerySentOffsetS,
        actual_query_video_time_s: roundQueryVideoTimeS,
        expected_prompted_frame_time_s: expectedPromptedFrameTimeS,
        expected_prompted_frame_time_source: latestTimedResponse
          ? 'latest_vlm_response_clock'
          : 'stream_ready_fallback',
        wait_before_query_s: waitBeforeQueryMs / 1000,
        video_ready_before_query: true,
        video_state_before_query: playbackStateWithTimeline,
        prompt_acknowledged: Boolean(promptAcknowledgement),
        prompt_ack_offset_s: promptAcknowledgement ? promptAcknowledgement.t_ms / 1000 : null,
        prompt_ack_latency_s: promptAcknowledgement && promptRequestedAtMs != null
          ? (startedAtMs + promptAcknowledgement.t_ms - promptRequestedAtMs) / 1000
          : null,
        query_frame_queue: useQueryTimeFrameQueue ? {
          policy: 'fifo-query-time-frame',
          query_event_id: `${task.id}:${round.id}`,
          frame_time_range: `${Number(queryDispatchVideoTimeS).toFixed(3)} seconds`,
          annotated_query_time_s: Number(round.query_time_s),
          dispatch_media_time_s: queryDispatchVideoTimeS,
          captured_media_time_s: queryDispatchVideoTimeS,
          captured_raw_media_time_s: queryTimeFrame.media_time_s,
          captured_width: queryTimeFrame.width,
          captured_height: queryTimeFrame.height,
          source_width: queryTimeFrame.source_width,
          source_height: queryTimeFrame.source_height,
          capture_surface: queryTimeFrame.capture_surface,
          acknowledged: queryFrameQueueAcknowledgement?.ok === true,
          queue_position: queryFrameQueueAcknowledgement?.queue_position ?? null,
          gate_released: frameGateRelease?.released === true,
          gate_disarmed_without_frame: frameGateRelease?.disarmed_without_frame === true,
        } : null,
        first_query_frame_gate_release: frameGateRelease,
        post_query_vlm_response_count: 0,
        prompted_vlm_response_count: 0,
        substantive_vlm_response_count: 0,
        completed_vlm_response_count: 0,
        deferred_vlm_response_count: 0,
        query_echo_response_count: 0,
        protocol_violation_count: 0,
        first_prompted_response_frame_time_s: null,
        prompted_response_frame_drift_s: null,
        first_completed_response_offset_s: null,
        first_completed_response_text: '',
      };
      queryEvents.push(queryEvent);
      appendJsonl(events, {
        t_ms: querySentAtMs - startedAtMs,
        type: 'capture_query_sent',
        task_id: task.id,
        ...queryEvent,
      });
    }

    const taskTimeoutMs = args.perTaskTimeoutS > 0 ? args.perTaskTimeoutS * 1000 : 0;
    await waitUntilVideoTime(durationS, 'task recording', taskTimeoutMs);

    const normalizeQuestion = (value) => String(value || '').normalize('NFKC').replace(/\s+/g, '').trim();
    const responseIsSubstantive = (response) => (
      response?.response_protocol?.substantive
      ?? isSubstantiveVlmText(response?.text, response?.response_payload)
    );
    const responseCompletesQuery = (response, queryEvent) => (
      responseIsSubstantive(response)
      && !isDeferredVlmText(response?.text)
      && !isQueryEchoVlmText(response?.text, queryEvent?.query)
    );
    const responsesForQuery = (queryEvent) => {
      const queryIndex = queryEvents.indexOf(queryEvent);
      const nextQueryEvent = queryIndex >= 0 ? queryEvents[queryIndex + 1] || null : null;
      return vlmResponses.filter((response) => (
        responseBelongsToQuery(response, queryEvent, nextQueryEvent)
      ));
    };
    const completedResponseForQuery = (queryEvent) => (
      responsesForQuery(queryEvent).find((response) => (
        responseCompletesQuery(response, queryEvent)
      )) || null
    );
    const backgroundTasksForQuery = (queryEvent) => {
      const querySentMs = queryEvent.query_sent_offset_s * 1000;
      const normalizedQuery = normalizeQuestion(queryEvent.query);
      return [...backgroundTasks.values()].filter((item) => (
        normalizeQuestion(item.question) === normalizedQuery
        && (item.started_t_ms == null || item.started_t_ms >= querySentMs)
      ));
    };
    const queryHasBackgroundTask = (queryEvent) => backgroundTasksForQuery(queryEvent).length > 0;
    const queryHasDeferredResponse = (queryEvent) => {
      return responsesForQuery(queryEvent).some((item) => isDeferredVlmText(item.text));
    };
    const queryRequiresAsyncCompletion = (queryEvent) => (
      task.category === '智能体委托'
      || /后台|异步/.test(queryEvent.query)
      || queryHasBackgroundTask(queryEvent)
      || queryHasDeferredResponse(queryEvent)
    );
    const findQueryOutcome = (queryEvent) => {
      const matchingBackgroundTasks = backgroundTasksForQuery(queryEvent);
      const failedBackgroundTask = matchingBackgroundTasks.find((item) => item.status === 'error');
      if (failedBackgroundTask) {
        return {
          status: 'error',
          source: 'background_result_error',
          t_ms: failedBackgroundTask.finished_t_ms,
          text: '',
          error: failedBackgroundTask.error,
          background_task_id: failedBackgroundTask.task_id,
        };
      }
      const readyBackgroundTask = matchingBackgroundTasks.find((item) => (
        item.status === 'ready'
        && isSubstantiveVlmText(item.text)
        && !isDeferredVlmText(item.text)
      ));
      if (readyBackgroundTask) {
        return {
          status: 'ready',
          source: 'background_result_ready',
          t_ms: readyBackgroundTask.finished_t_ms,
          text: readyBackgroundTask.text,
          error: '',
          background_task_id: readyBackgroundTask.task_id,
        };
      }
      const directResponse = completedResponseForQuery(queryEvent);
      if (directResponse) {
        return {
          status: 'ready',
          source: 'vlm_response',
          t_ms: directResponse.t_ms,
          text: directResponse.text,
          error: '',
          background_task_id: '',
        };
      }
      return {
        status: 'pending',
        source: matchingBackgroundTasks.length ? 'background_task_started' : '',
        t_ms: null,
        text: '',
        error: '',
        background_task_id: matchingBackgroundTasks[0]?.task_id || '',
      };
    };
    requiredTailResponseQueries = queryEvents.filter((item) => (
      item.query_time_s >= durationS || queryRequiresAsyncCompletion(item)
    ));
    requiredTailResponseQuery = requiredTailResponseQueries.length
      ? requiredTailResponseQueries[requiredTailResponseQueries.length - 1]
      : null;
    tailResponseWait = null;
    tailResponseWaits = [];
    const terminalResponseQueryIds = new Set();
    const timedOutResponseQueryIds = new Set();
    let responseWaitWindowStartedMs = null;
    const responseWaitDeadlineMs = () => {
      if (responseWaitWindowStartedMs == null) responseWaitWindowStartedMs = monotonicMs();
      return responseWaitWindowStartedMs + args.postVideoResponseTimeoutS * 1000;
    };
    if (requiredTailResponseQueries.length) {
      const waitStartedMs = monotonicMs();
      const deadlineMs = responseWaitDeadlineMs();
      appendJsonl(events, {
        t_ms: waitStartedMs - startedAtMs,
        type: 'capture_tail_query_responses_wait_start',
        task_id: task.id,
        query_ids: requiredTailResponseQueries.map((item) => item.id),
        query_times_s: requiredTailResponseQueries.map((item) => item.query_time_s),
        timeout_s: args.postVideoResponseTimeoutS,
      });

      let outcomes = [];
      let timedOut = false;
      while (true) {
        outcomes = requiredTailResponseQueries.map((queryEvent) => ({
          queryEvent,
          outcome: findQueryOutcome(queryEvent),
        }));
        const failed = outcomes.find((item) => item.outcome.status === 'error');
        if (failed) {
          throw new Error(
            `Background reply failed for query ${failed.queryEvent.id}: ${failed.outcome.error}`,
          );
        }
        if (outcomes.every((item) => item.outcome.status === 'ready')) break;
        const remainingMs = deadlineMs - monotonicMs();
        if (remainingMs <= 0) {
          const missingQueryIds = outcomes
            .filter((item) => item.outcome.status !== 'ready')
            .map((item) => item.queryEvent.id);
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_tail_query_responses_wait_timeout',
            task_id: task.id,
            query_ids: requiredTailResponseQueries.map((item) => item.id),
            missing_query_ids: missingQueryIds,
            timeout_s: args.postVideoResponseTimeoutS,
          });
          for (const queryId of missingQueryIds) timedOutResponseQueryIds.add(queryId);
          timedOut = true;
          break;
        }
        await waitOrFailOnDisconnect(
          Math.min(250, remainingMs),
          `tail query responses wait ${requiredTailResponseQueries.map((item) => item.id).join(',')}`,
        );
      }
      const waitFinishedMs = monotonicMs();
      tailResponseWaits = outcomes.map(({ queryEvent, outcome }) => {
        const responseReceived = outcome.status === 'ready';
        const responseWait = {
          query_id: queryEvent.id,
          query_time_s: queryEvent.query_time_s,
          query_sent_offset_s: queryEvent.query_sent_offset_s,
          status: responseReceived ? 'received' : 'timeout',
          response_source: responseReceived ? outcome.source : '',
          response_offset_s: responseReceived && outcome.t_ms != null ? outcome.t_ms / 1000 : null,
          response_wait_s: !responseReceived || outcome.t_ms == null
            ? null
            : Math.max(0, outcome.t_ms - queryEvent.query_sent_offset_s * 1000) / 1000,
          response_text: responseReceived ? outcome.text : '',
          background_task_id: responseReceived
            ? (outcome.background_task_id || undefined)
            : undefined,
          timeout_s: responseReceived ? undefined : args.postVideoResponseTimeoutS,
        };
        terminalResponseQueryIds.add(queryEvent.id);
        queryEvent.final_response_received = responseReceived;
        queryEvent.final_response_status = responseWait.status;
        queryEvent.final_response_source = responseWait.response_source;
        queryEvent.final_response_offset_s = responseWait.response_offset_s;
        queryEvent.final_response_wait_s = responseWait.response_wait_s;
        queryEvent.final_response_text = responseWait.response_text;
        queryEvent.background_task_id = responseWait.background_task_id;
        queryEvent.final_response_timeout_s = responseWait.timeout_s;
        return responseWait;
      });
      const lastResponseWait = tailResponseWaits[tailResponseWaits.length - 1];
      tailResponseWait = {
        required: true,
        status: timedOut ? 'timeout' : 'complete',
        query_id: lastResponseWait.query_id,
        query_ids: tailResponseWaits.map((item) => item.query_id),
        timed_out_query_ids: tailResponseWaits
          .filter((item) => item.status === 'timeout')
          .map((item) => item.query_id),
        query_time_s: lastResponseWait.query_time_s,
        timeout_s: args.postVideoResponseTimeoutS,
        wait_s: (waitFinishedMs - waitStartedMs) / 1000,
        response_offset_s: lastResponseWait.response_offset_s,
        response_text: lastResponseWait.response_text,
        query_reactivations: [],
        responses: tailResponseWaits,
      };
      appendJsonl(events, {
        t_ms: waitFinishedMs - startedAtMs,
        type: 'capture_tail_query_responses_wait_done',
        task_id: task.id,
        ...tailResponseWait,
      });
    }
    const updatePostQueryResponseCounts = () => {
      for (let queryIndex = 0; queryIndex < queryEvents.length; queryIndex += 1) {
        const queryEvent = queryEvents[queryIndex];
        const querySentMs = queryEvent.query_sent_offset_s * 1000;
        const postQueryResponses = responsesForQuery(queryEvent);
        const promptedResponses = vlmResponses.filter((response) => (
          response.t_ms >= querySentMs
          && (
            (response.metrics?.user_prompt || '') === queryEvent.query
            || responseQuerySha256(response)
              === crypto.createHash('sha256').update(queryEvent.query).digest('hex')
          )
        ));
        const substantiveResponses = postQueryResponses.filter(responseIsSubstantive);
        const deferredResponses = substantiveResponses.filter((response) => (
          isDeferredVlmText(response.text)
        ));
        const queryEchoResponses = substantiveResponses.filter((response) => (
          isQueryEchoVlmText(response.text, queryEvent.query)
        ));
        const completedResponses = postQueryResponses.filter((response) => (
          responseCompletesQuery(response, queryEvent)
        ));
        const queryProtocolViolations = postQueryResponses.filter((response) => (
          response.response_protocol?.protocol_valid === false
        ));
        queryEvent.post_query_vlm_response_count = postQueryResponses.length;
        queryEvent.prompted_vlm_response_count = promptedResponses.length;
        queryEvent.substantive_vlm_response_count = substantiveResponses.length;
        queryEvent.completed_vlm_response_count = completedResponses.length;
        queryEvent.deferred_vlm_response_count = deferredResponses.length;
        queryEvent.query_echo_response_count = queryEchoResponses.length;
        queryEvent.protocol_violation_count = queryProtocolViolations.length;
        queryEvent.first_prompted_response_frame_time_s = promptedResponses.length
          ? responseQueryFrameTimeS(promptedResponses[0])
          : null;
        queryEvent.prompted_response_frame_drift_s = queryEvent.first_prompted_response_frame_time_s == null
          ? null
          : queryEvent.first_prompted_response_frame_time_s - queryEvent.expected_prompted_frame_time_s;
        queryEvent.first_completed_response_offset_s = completedResponses.length
          ? completedResponses[0].t_ms / 1000
          : null;
        queryEvent.first_completed_response_text = completedResponses.length
          ? completedResponses[0].text
          : '';
      }
      const firstQueryEvent = queryEvents[0] || null;
      postQueryVlmResponseCount = queryEvents.reduce((sum, item) => sum + item.post_query_vlm_response_count, 0);
      promptedVlmResponseCount = queryEvents.reduce((sum, item) => sum + item.prompted_vlm_response_count, 0);
      substantiveVlmResponseCount = queryEvents.reduce(
        (sum, item) => sum + item.substantive_vlm_response_count,
        0,
      );
      completedVlmResponseCount = queryEvents.reduce(
        (sum, item) => sum + item.completed_vlm_response_count,
        0,
      );
      deferredVlmResponseCount = queryEvents.reduce(
        (sum, item) => sum + item.deferred_vlm_response_count,
        0,
      );
      queryEchoResponseCount = queryEvents.reduce(
        (sum, item) => sum + item.query_echo_response_count,
        0,
      );
      firstPromptedResponseFrameTimeS = firstQueryEvent?.first_prompted_response_frame_time_s ?? null;
      promptedResponseFrameDriftS = firstQueryEvent?.prompted_response_frame_drift_s ?? null;
    };
    updatePostQueryResponseCounts();
    const pendingQueries = () => queryEvents.filter((item) => (
      findQueryOutcome(item).status !== 'ready'
      && !terminalResponseQueryIds.has(item.id)
    ));
    let missingCompletedQueries = pendingQueries();
    if (missingCompletedQueries.length) {
      const waitStartedMs = monotonicMs();
      const deadlineMs = responseWaitDeadlineMs();
      appendJsonl(events, {
        t_ms: waitStartedMs - startedAtMs,
        type: 'capture_all_query_responses_wait_start',
        task_id: task.id,
        timeout_s: args.postVideoResponseTimeoutS,
        completion_criterion: MODEL_RESPONSE_OUTCOME_PROTOCOL,
        missing_query_ids: missingCompletedQueries.map((item) => item.id),
        post_query_vlm_response_count: postQueryVlmResponseCount,
        prompted_vlm_response_count: promptedVlmResponseCount,
        substantive_vlm_response_count: substantiveVlmResponseCount,
        completed_vlm_response_count: completedVlmResponseCount,
        deferred_vlm_response_count: deferredVlmResponseCount,
        query_echo_response_count: queryEchoResponseCount,
      });
      while (missingCompletedQueries.length && monotonicMs() < deadlineMs) {
        const failed = missingCompletedQueries.find((item) => (
          findQueryOutcome(item).status === 'error'
        ));
        if (failed) {
          throw new Error(
            `Background reply failed for query ${failed.id}: ${findQueryOutcome(failed).error}`,
          );
        }
        await waitOrFailOnDisconnect(
          Math.min(250, Math.max(1, deadlineMs - monotonicMs())),
          `all query responses wait ${missingCompletedQueries.map((item) => item.id).join(',')}`,
        );
        updatePostQueryResponseCounts();
        missingCompletedQueries = pendingQueries();
      }
      if (missingCompletedQueries.length) {
        appendJsonl(events, {
          t_ms: monotonicMs() - startedAtMs,
          type: 'capture_missing_substantive_vlm_response',
          task_id: task.id,
          completion_criterion: MODEL_RESPONSE_OUTCOME_PROTOCOL,
          missing_query_ids: missingCompletedQueries.map((item) => item.id),
          timeout_s: args.postVideoResponseTimeoutS,
        });
        for (const queryEvent of missingCompletedQueries) {
          terminalResponseQueryIds.add(queryEvent.id);
          timedOutResponseQueryIds.add(queryEvent.id);
          queryEvent.final_response_received = false;
          queryEvent.final_response_status = 'timeout';
          queryEvent.final_response_source = '';
          queryEvent.final_response_offset_s = null;
          queryEvent.final_response_wait_s = null;
          queryEvent.final_response_text = '';
          queryEvent.final_response_timeout_s = args.postVideoResponseTimeoutS;
        }
      }
      allQueryResponseWait = {
        required: true,
        status: missingCompletedQueries.length ? 'timeout' : 'complete',
        completion_criterion: MODEL_RESPONSE_OUTCOME_PROTOCOL,
        timeout_s: args.postVideoResponseTimeoutS,
        wait_s: (monotonicMs() - waitStartedMs) / 1000,
        query_ids: queryEvents.map((item) => item.id),
        timed_out_query_ids: missingCompletedQueries.map((item) => item.id),
      };
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_all_query_responses_wait_done',
        task_id: task.id,
        ...allQueryResponseWait,
      });
    } else {
      allQueryResponseWait = { required: false, wait_s: 0 };
    }

    for (const queryEvent of queryEvents) {
      if (queryEvent.final_response_status) continue;
      const outcome = findQueryOutcome(queryEvent);
      const responseReceived = outcome.status === 'ready';
      queryEvent.final_response_received = responseReceived;
      queryEvent.final_response_status = responseReceived ? 'received' : 'timeout';
      queryEvent.final_response_source = responseReceived ? outcome.source : '';
      queryEvent.final_response_offset_s = responseReceived && outcome.t_ms != null
        ? outcome.t_ms / 1000
        : null;
      queryEvent.final_response_wait_s = responseReceived && outcome.t_ms != null
        ? Math.max(0, outcome.t_ms - queryEvent.query_sent_offset_s * 1000) / 1000
        : null;
      queryEvent.final_response_text = responseReceived ? outcome.text : '';
      if (!responseReceived) {
        timedOutResponseQueryIds.add(queryEvent.id);
        queryEvent.final_response_timeout_s = args.postVideoResponseTimeoutS;
      }
    }
    const timedOutQueryIds = queryEvents
      .map((item) => item.id)
      .filter((queryId) => timedOutResponseQueryIds.has(queryId));
    const repliedQueryIds = queryEvents
      .map((item) => item.id)
      .filter((queryId) => !timedOutResponseQueryIds.has(queryId));
    modelResponseOutcome = {
      policy: MODEL_RESPONSE_TIMEOUT_POLICY,
      response_semantics: MODEL_RESPONSE_OUTCOME_PROTOCOL,
      completion_criterion: 'substantive_non_deferred_non_echo_response',
      timeout_s: args.postVideoResponseTimeoutS,
      timeout_is_recording_failure: false,
      status: timedOutQueryIds.length
        ? (timedOutQueryIds.length === queryEvents.length ? 'timeout' : 'partial_timeout')
        : 'complete',
      expected_query_ids: queryEvents.map((item) => item.id),
      replied_query_ids: repliedQueryIds,
      timed_out_query_ids: timedOutQueryIds,
      protocol_violation_count: protocolViolations.length,
      deferred_response_count: deferredVlmResponseCount,
      query_echo_response_count: queryEchoResponseCount,
      wait_s: responseWaitWindowStartedMs == null
        ? 0
        : (monotonicMs() - responseWaitWindowStartedMs) / 1000,
    };

    const queryResponseOffsetsS = [];
    for (const queryEvent of queryEvents) {
      for (const response of responsesForQuery(queryEvent)) {
        if (!responseCompletesQuery(response, queryEvent)) continue;
        queryResponseOffsetsS.push(response.t_ms / 1000);
      }
    }
    for (const responseWait of tailResponseWaits) {
      if (Number.isFinite(responseWait.response_offset_s)) {
        queryResponseOffsetsS.push(responseWait.response_offset_s);
      }
    }
    const latestResponseOffsetS = queryResponseOffsetsS.length
      ? Math.max(...queryResponseOffsetsS)
      : null;
    if (queryEvents.length && args.postResponseRecordingS > 0) {
      const postResponseStartedMs = monotonicMs();
      const postResponseCaptureS = args.postResponseRecordingS
        + RECORDING_FINALIZATION_GUARD_S;
      appendJsonl(events, {
        t_ms: postResponseStartedMs - startedAtMs,
        type: 'capture_post_response_recording_start',
        task_id: task.id,
        configured_s: args.postResponseRecordingS,
        finalization_guard_s: RECORDING_FINALIZATION_GUARD_S,
        latest_response_offset_s: latestResponseOffsetS,
      });
      await waitOrFailOnDisconnect(
        postResponseCaptureS * 1000,
        'post-response recording',
      );
      const postResponseFinishedMs = monotonicMs();
      postResponseRecording = {
        required: true,
        configured_s: args.postResponseRecordingS,
        finalization_guard_s: RECORDING_FINALIZATION_GUARD_S,
        actual_s: (postResponseFinishedMs - postResponseStartedMs) / 1000,
        latest_response_offset_s: latestResponseOffsetS,
        finished_offset_s: (postResponseFinishedMs - startedAtMs) / 1000,
      };
      appendJsonl(events, {
        t_ms: postResponseFinishedMs - startedAtMs,
        type: 'capture_post_response_recording_done',
        task_id: task.id,
        ...postResponseRecording,
      });
    } else {
      postResponseRecording = {
        required: false,
        configured_s: args.postResponseRecordingS,
        actual_s: 0,
        latest_response_offset_s: latestResponseOffsetS,
      };
    }
    taskTimelineEndMs = monotonicMs();
    if (inputMode === 'upload') {
      const continuityGuard = await disableUploadedPlaybackContinuityGuard(page).catch(() => null);
      if (uploadPlaybackHealth) uploadPlaybackHealth.continuity_guard = continuityGuard;
    }
    const invalidTimingQuery = queryEvents.find((item) => (
      item.prompted_response_frame_drift_s != null
      && Math.abs(item.prompted_response_frame_drift_s) > 3
    ));
    if (invalidTimingQuery) {
      appendJsonl(events, {
        t_ms: monotonicMs() - startedAtMs,
        type: 'capture_prompted_response_timing_invalid',
        task_id: task.id,
        query_id: invalidTimingQuery.id,
        query_time_s: invalidTimingQuery.query_time_s,
        expected_prompted_frame_time_s: invalidTimingQuery.expected_prompted_frame_time_s,
        first_prompted_response_frame_time_s: invalidTimingQuery.first_prompted_response_frame_time_s,
        prompted_response_frame_drift_s: invalidTimingQuery.prompted_response_frame_drift_s,
        max_abs_drift_s: 3,
      });
    }

    if (inputMode === 'upload') {
      const sid = observedSessionId || await getPageSessionId(page) || sessionId;
      await stopUploadedVideoApi(args.webUrl, sid, 15000, auth)
        .then((stopResponse) => {
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_upload_api_stop_response',
            task_id: task.id,
            session_id: sid,
            stop_response: stopResponse,
          });
        })
        .catch(async (error) => {
          appendJsonl(events, {
            t_ms: monotonicMs() - startedAtMs,
            type: 'capture_upload_api_stop_failed',
            task_id: task.id,
            session_id: sid,
            error: error.message,
          });
          await stopPageFromPage(page, 15000).catch((pageStopError) => {
            appendJsonl(events, {
              t_ms: monotonicMs() - startedAtMs,
              type: 'capture_page_stop_failed',
              task_id: task.id,
              error: pageStopError.message,
            });
          });
        });
    } else {
      await stopPageFromPage(page, 60000).catch((error) => {
        appendJsonl(events, {
          t_ms: monotonicMs() - startedAtMs,
          type: 'capture_page_stop_failed',
          task_id: task.id,
          error: error.message,
        });
      });
    }
    pageStopCalled = true;
    await sleep(1000);
  } catch (error) {
    status = 'error';
    errorText = error.stack || error.message || String(error);
    appendJsonl(events, {
      t_ms: monotonicMs() - startedAtMs,
      type: 'capture_error',
      error: errorText,
    });
    console.error(`[${index + 1}/${total}] ${task.id}: ${error.message}`);
  } finally {
    if (!pageStopCalled) {
      if (inputMode === 'upload') {
        const sid = observedSessionId || await getPageSessionId(page).catch(() => '') || sessionId;
        await stopUploadedVideoApi(args.webUrl, sid, 15000, auth).catch(async () => {
          await stopPageFromPage(page, 15000).catch(() => {
            // Page may already be gone or the backend stop endpoint may be stuck.
          });
        });
      } else {
        await stopPageFromPage(page, 60000).catch(() => {
          // Page may already be gone or the backend stop endpoint may be stuck.
        });
      }
    }
    await stopLocalRtspPublisher(localPublisher);
    if (auxWs) {
      try {
        auxWs.close();
      } catch {
        // Best effort.
      }
      auxWs = null;
    }

    try {
      const sid = observedSessionId || await getPageSessionId(page);
      if (sid) {
        if (inputMode === 'upload') {
          await cleanupSessionApi(args.webUrl, sid, 10000, auth).catch(() => {});
        } else {
          await page.evaluate(async (cleanupSessionId) => {
            await fetch('/api/session/cleanup', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ session_id: cleanupSessionId, reset_adapter: true }),
            }).catch(() => {});
          }, sid);
        }
      }
    } catch {
      // Best effort cleanup.
    }

    const video = page.video();
    await page.close().catch(() => {});
    if (video) {
      browserVideoPath = await video.path().catch(() => '');
    }
    await context.close().catch(() => {});
    events.end();
    await new Promise((resolve) => events.on('finish', resolve));

    const finalWebmPath = path.join(taskDir, 'ui.webm');
    if (browserVideoPath && fs.existsSync(browserVideoPath)) {
      await fsp.rename(browserVideoPath, finalWebmPath).catch(async () => {
        await fsp.copyFile(browserVideoPath, finalWebmPath);
      });
      browserVideoPath = finalWebmPath;
    }

    if (inputMode === 'upload' && streamReadyOffsetS != null && browserVideoPath) {
      if (!recordingClockMarkerRemoval) {
        browserRecordingClock = {
          ok: false,
          error: 'The pre-playback WebM clock marker was not removed',
        };
      } else {
        browserRecordingClock = await calibrateWebmClockFromVisualMarker({
          input: browserVideoPath,
          markerRemovedEventOffsetS: (
            recordingClockMarkerRemoval.removed_at_ms - startedAtMs
          ) / 1000,
          videoWidth: args.width,
          videoHeight: args.height,
          markerSize: RECORDING_CLOCK_MARKER.size,
          markerX: RECORDING_CLOCK_MARKER.x,
          ffmpegBin: args.ffmpegBin,
        });
      }
      await fsp.appendFile(eventsPath, `${JSON.stringify(redactSecretsForLog({
        t_ms: monotonicMs() - startedAtMs,
        type: browserRecordingClock.ok
          ? 'capture_recording_clock_calibrated'
          : 'capture_recording_clock_calibration_failed',
        task_id: task.id,
        calibration: browserRecordingClock,
      }))}\n`);
      if (!browserRecordingClock.ok) {
        const calibrationError = `WebM clock calibration failed: ${browserRecordingClock.error}`;
        status = 'error';
        errorText = errorText ? `${errorText}\n${calibrationError}` : calibrationError;
      }
    }

    if (args.mp4 && browserVideoPath && await commandExists(args.ffmpegBin)) {
      if (
        streamReadyOffsetS != null
        && (inputMode !== 'upload' || browserRecordingClock?.ok === true)
      ) {
        taskMp4Path = path.join(taskDir, `${safeName(task.id)}_task.mp4`);
        const trimStartOffsetS = originalVideoStartOffsetS ?? streamReadyOffsetS;
        const trimEndOffsetS = taskTimelineEndMs == null
          ? trimStartOffsetS + Math.max(durationS, maxQueryTimeS)
          : (taskTimelineEndMs - startedAtMs) / 1000;
        const eventClockMinusWebmPtsS = inputMode === 'upload'
          ? browserRecordingClock.event_clock_minus_webm_pts_s
          : 0;
        taskVideoStartWebmPtsS = trimStartOffsetS - eventClockMinusWebmPtsS;
        const outageIntervals = (localPublisher?.outageIntervals || []).map((item) => ({
          start_s: (item.startMs - startedAtMs) / 1000 - eventClockMinusWebmPtsS,
          end_s: (item.endMs - startedAtMs) / 1000 - eventClockMinusWebmPtsS,
          start_source_position_s: item.startSourcePositionS,
          reasons: item.reasons,
        }));
        taskVideoRender = await renderTaskVideo({
          input: browserVideoPath,
          output: taskMp4Path,
          trimStartS: taskVideoStartWebmPtsS,
          trimEndS: trimEndOffsetS - eventClockMinusWebmPtsS,
          outageIntervals,
          ffmpegBin: args.ffmpegBin,
        });
        taskVideoRender = {
          ...taskVideoRender,
          event_clock_minus_webm_pts_s: eventClockMinusWebmPtsS,
          event_trim_start_offset_s: trimStartOffsetS,
          event_trim_end_offset_s: trimEndOffsetS,
        };
        if (!taskVideoRender.ok) taskMp4Path = '';
      }
      if (args.fullUiMp4) {
        mp4Path = path.join(taskDir, 'ui.mp4');
        uiMp4Conversion = await convertWebmToMp4(
          browserVideoPath,
          mp4Path,
          args.ffmpegBin,
        );
        if (!uiMp4Conversion.ok) {
          console.error(uiMp4Conversion.error);
          mp4Path = '';
        }
      }
    }

    const minimumTaskVideoDurationS = Math.max(durationS, maxQueryTimeS);
    const taskVideoStartOffsetS = originalVideoStartOffsetS ?? streamReadyOffsetS;
    const capturedTaskDurationS = taskTimelineEndMs == null || taskVideoStartOffsetS == null
      ? null
      : (taskTimelineEndMs - (startedAtMs + taskVideoStartOffsetS * 1000)) / 1000;
    const expectedTaskVideoDurationS = Number.isFinite(capturedTaskDurationS)
      ? capturedTaskDurationS
      : minimumTaskVideoDurationS;
    const responseWaitWasRequired = Boolean(
      tailResponseWait?.required || allQueryResponseWait?.required,
    );
    const responseCapabilityTimedOut = ['timeout', 'partial_timeout'].includes(
      modelResponseOutcome?.status,
    );
    const taskVideoDurationToleranceS = responseCapabilityTimedOut
      ? Math.max(args.videoDurationToleranceS, 5)
      : args.videoDurationToleranceS;
    const maximumTaskVideoDurationS = calculateMaximumTaskVideoDurationS({
      minimumDurationS: minimumTaskVideoDurationS,
      responseWaitRequired: responseWaitWasRequired,
      responseTimeoutS: args.postVideoResponseTimeoutS,
      postResponseRecordingS: args.postResponseRecordingS,
      maxExtraS: args.videoMaxExtraS,
    });
    const requiredResponseTailEndS = (
      Number.isFinite(postResponseRecording?.latest_response_offset_s)
      && Number.isFinite(taskVideoStartOffsetS)
    )
      ? postResponseRecording.latest_response_offset_s
        - taskVideoStartOffsetS
        + postResponseRecording.configured_s
      : null;
    if (status === 'ok' && providerInfrastructureFailure) {
      status = 'error';
      errorText = `Provider infrastructure failed: ${providerInfrastructureFailure.text}`;
    } else if (status === 'ok' && taskVideoRender?.outages?.length) {
      status = 'error';
      errorText = `Output outage detected during task video; retry the entire task (${taskVideoRender.outages.length} interval(s))`;
    } else if (status === 'ok' && !taskMp4Path) {
      status = 'error';
      errorText = taskVideoRender?.error || 'Task video was not created';
    } else if (status === 'ok' && args.videoQualityCheck) {
      videoQuality = await validateTaskVideo({
        videoPath: taskMp4Path,
        expectedDurationS: expectedTaskVideoDurationS,
        minimumDurationS: minimumTaskVideoDurationS,
        maximumDurationS: maximumTaskVideoDurationS,
        queryEvents,
        expectedQueries: queryRounds,
        ffmpegBin: args.ffmpegBin,
        ffprobeBin: siblingFfprobe(args.ffmpegBin),
        durationToleranceS: taskVideoDurationToleranceS,
        queryTimeToleranceS: args.videoQueryTimeToleranceS,
        maxBlackDurationS: args.videoMaxBlackS,
        requiredRecordedUntilS: requiredResponseTailEndS,
      });
      if (!videoQuality.ok) {
        status = 'error';
        errorText = `Task video quality validation failed: ${videoQuality.errors.join('; ')}`;
      }
    }

    vlmBackendIdentity = validateVlmBackendIdentity(args, vlmResponses);
    if (status === 'ok' && !vlmBackendIdentity.ok) {
      status = 'error';
      errorText = vlmBackendIdentity.error || 'VLM backend identity verification failed';
    }

    const queryDeliveryRounds = queryRounds.map((round) => ({
      id: round.id,
      expected_deliveries: 1,
      actual_deliveries: queryDispatchCounts.get(round.id) || 0,
    }));
    const queryDeliveryAttestation = {
      policy: 'once_per_round_no_replay',
      ok: queryEvents.length === queryRounds.length
        && queryDeliveryRounds.every((round) => round.actual_deliveries === 1),
      expected_rounds: queryRounds.length,
      delivered_rounds: queryEvents.length,
      replay_count: 0,
      cache_into_later_frames: false,
      rounds: queryDeliveryRounds,
    };
    if (status === 'ok' && !queryDeliveryAttestation.ok) {
      status = 'error';
      errorText = 'Query delivery did not satisfy once-per-round/no-replay policy';
    }

    const successfulVlmWarmup = [...vlmInferenceWarmups].reverse().find((item) => item.ok) || null;
    const providerWarmupReady = Boolean(
      !providerWarmupRequired
      || (
        successfulVlmWarmup?.provider_warmup?.ok === true
        && successfulVlmWarmup.provider_warmup.multimodal_input === true
        && successfulVlmWarmup.provider_warmup.session_state_unchanged === true
      )
    );
    const nativeWarmupStateCleared = Boolean(
      !nativeSessionResetRequired
      || (
        successfulVlmWarmup?.warmup_session_reset_required === true
        && successfulVlmWarmup?.warmup_session_reset?.ok === true
        && successfulVlmWarmup?.warmup_session_reset?.reset_acknowledged === true
      )
    );
    const webUiSameSessionModelReady = Boolean(
      inputMode === 'upload'
      && successfulVlmWarmup?.strategy === 'webui_same_session'
      && webUiWarmupSessionId
      && webUiWarmupSessionId === observedSessionId
    );

    const summary = {
      status,
      error: errorText || undefined,
      task,
      input_mode: inputMode,
      effective_video_url: effectiveVideoUrl,
      vlm_profile: args.vlmProfileSnapshot || undefined,
      vlm_model: args.vlmModel || undefined,
      vlm_api_base: args.vlmApiBase ? redactUrlForLog(args.vlmApiBase) : undefined,
      vlm_route: args.vlmModel ? args.vlmRoute : undefined,
      vlm_formal_eval: args.vlmModel ? args.vlmFormalEval : undefined,
      vlm_preflight: args.vlmPreflightResult || undefined,
      vlm_backend_identity: vlmBackendIdentity,
      query_delivery_attestation: queryDeliveryAttestation,
      vlm_model_updates: modelUpdateEvents,
      vlm_inference_warmup: args.vlmModel ? {
        required: args.vlmWarmup,
        ok: args.vlmWarmup
          ? vlmInferenceWarmups.some((item) => item.ok)
          : true,
        status: args.vlmWarmup
          ? (vlmInferenceWarmups.some((item) => item.ok) ? 'ready' : 'not_ready')
          : 'disabled',
        configured_timeout_s: args.vlmWarmupTimeoutS,
        configured_retries: args.vlmWarmupRetries,
        latest_ready_offset_s: latestVlmWarmupReadyAtMs == null
          ? null
          : (latestVlmWarmupReadyAtMs - startedAtMs) / 1000,
        playback_start_requests: playbackStartRequests,
        barrier_satisfied: playbackStartRequests.length > 0 && playbackStartRequests.every((item) => (
          !args.vlmWarmup
          || (
            Number.isFinite(item.warmup_completed_offset_s)
            && item.warmup_completed_offset_s <= item.requested_offset_s
            && (inputMode !== 'upload' || webUiSameSessionModelReady)
            && providerWarmupReady
            && nativeWarmupStateCleared
          )
        )),
        runs: vlmInferenceWarmups,
      } : undefined,
      recording_start_barrier: inputMode === 'upload' ? {
        protocol: CURRENT_RECORDING_PROTOCOL,
        ok: Boolean(
          targetUploadStartedAtMs != null
          && targetUploadReadyAtMs != null
          && targetUploadVerification?.ok === true
          && targetUploadVerification?.complete_file_uploaded === true
          && targetUploadVerification?.local_size_bytes
            === targetUploadVerification?.remote_size_bytes
          && officialPlaybackRequestedAtMs != null
          && officialPlaybackStartedAtMs != null
          && (
            !args.vlmModel
            || !args.vlmWarmup
            || (
              latestVlmWarmupReadyAtMs != null
              && latestVlmWarmupReadyAtMs <= targetUploadStartedAtMs
              && webUiSameSessionModelReady
              && providerWarmupReady
              && nativeWarmupStateCleared
            )
          )
          && targetUploadStartedAtMs <= targetUploadReadyAtMs
          && targetUploadReadyAtMs <= officialPlaybackRequestedAtMs
        ),
        model_api_ready_offset_s: latestVlmWarmupReadyAtMs == null
          ? null
          : (latestVlmWarmupReadyAtMs - startedAtMs) / 1000,
        model_ready_strategy: successfulVlmWarmup?.strategy || null,
        provider_warmup_required: providerWarmupRequired,
        provider_warmup: successfulVlmWarmup?.provider_warmup || null,
        native_warmup_state_cleared: nativeWarmupStateCleared,
        warmup_session_reset: successfulVlmWarmup?.warmup_session_reset || null,
        webui_warmup_session_id: webUiWarmupSessionId || null,
        target_session_id: observedSessionId || null,
        webui_same_session_model_ready: webUiSameSessionModelReady,
        target_upload_started_offset_s: targetUploadStartedAtMs == null
          ? null
          : (targetUploadStartedAtMs - startedAtMs) / 1000,
        target_upload_ready_offset_s: targetUploadReadyAtMs == null
          ? null
          : (targetUploadReadyAtMs - startedAtMs) / 1000,
        playback_requested_offset_s: officialPlaybackRequestedAtMs == null
          ? null
          : (officialPlaybackRequestedAtMs - startedAtMs) / 1000,
        playback_started_offset_s: officialPlaybackStartedAtMs == null
          ? null
          : (officialPlaybackStartedAtMs - startedAtMs) / 1000,
        first_rendered_frame_offset_s: officialPlaybackFirstFrameAtMs == null
          ? null
          : (officialPlaybackFirstFrameAtMs - startedAtMs) / 1000,
        analyzer_ready_offset_s: officialPlaybackAnalyzerReadyAtMs == null
          ? null
          : (officialPlaybackAnalyzerReadyAtMs - startedAtMs) / 1000,
        official_task_recording_start_offset_s: streamReadyOffsetS,
        official_task_recording_start_webm_pts_s: taskVideoStartWebmPtsS,
        target_video_uploaded_after_model_ready: Boolean(
          targetUploadStartedAtMs != null
          && (
            !args.vlmModel
            || !args.vlmWarmup
            || (
              latestVlmWarmupReadyAtMs != null
              && latestVlmWarmupReadyAtMs <= targetUploadStartedAtMs
              && webUiSameSessionModelReady
              && providerWarmupReady
            )
          )
        ),
        complete_file_upload: targetUploadVerification,
        target_video_started_once: targetPlaybackWasStarted,
        seek_performed_after_stream_start: false,
        same_session_playback_restart_allowed: false,
        first_query_frame_gate: firstQueryFrameGate,
      } : undefined,
      local_rtsp_relay: localRelayInfo || undefined,
      local_video_upload: localUploadInfo || undefined,
      upload_playback_health: uploadPlaybackHealth || undefined,
      upload_control_websocket: inputMode === 'upload' ? {
        close_count: uploadControlWsCloseCount,
        reconnect_count: uploadControlWsReconnectCount,
        pending_reconnect: uploadControlWsDisconnectedAtMs != null,
        total_downtime_s: uploadControlWsTotalDowntimeMs / 1000,
        max_downtime_s: uploadControlWsMaxDowntimeMs / 1000,
        reconnects_allowed: args.uploadControlReconnects,
        reconnect_timeout_s: args.uploadControlReconnectTimeoutS,
      } : undefined,
      task_index: index + 1,
      task_total: total,
      attempt,
      max_attempts: maxAttempts,
      session_id: observedSessionId || sessionId,
      started_at: startedAtIso,
      finished_at: new Date().toISOString(),
      duration_s_requested: durationS,
      query_time_s_requested: task.query_time_s,
      query_rounds: queryRounds,
      query_events: queryEvents,
      query_sent_offset_s: querySentOffsetS,
      actual_query_video_time_s: queryVideoTimeS,
      stream_ready_offset_s: streamReadyOffsetS,
      original_video_start_offset_s: originalVideoStartOffsetS ?? streamReadyOffsetS,
      process_interval_s: processIntervalS,
      frames_per_batch: framesPerBatch,
      stream_start_retries: args.streamStartRetries,
      stream_reconnects_requested: args.streamReconnects,
      stream_reconnect_count: streamReconnectCount,
      stream_recovery_paused_s: streamRecoveryPausedMs / 1000,
      captured_task_duration_s: capturedTaskDurationS,
      expected_task_video_duration_s: expectedTaskVideoDurationS,
      minimum_task_video_duration_s: minimumTaskVideoDurationS,
      maximum_task_video_duration_s: maximumTaskVideoDurationS,
      tail_response_wait: tailResponseWait || { required: false },
      tail_response_waits: tailResponseWaits,
      all_query_response_wait: allQueryResponseWait || { required: false },
      model_response_outcome: modelResponseOutcome || {
        policy: MODEL_RESPONSE_TIMEOUT_POLICY,
        response_semantics: MODEL_RESPONSE_OUTCOME_PROTOCOL,
        completion_criterion: 'substantive_non_deferred_non_echo_response',
        timeout_s: args.postVideoResponseTimeoutS,
        timeout_is_recording_failure: false,
        status: queryEvents.length ? 'unknown' : 'not_applicable',
        expected_query_ids: queryEvents.map((item) => item.id),
        replied_query_ids: [],
        timed_out_query_ids: [],
        protocol_violation_count: protocolViolations.length,
        deferred_response_count: deferredVlmResponseCount,
        query_echo_response_count: queryEchoResponseCount,
      },
      post_response_recording: postResponseRecording || { required: false },
      background_tasks: [...backgroundTasks.values()],
      task_video_render: taskVideoRender || undefined,
      ui_mp4_conversion: uiMp4Conversion || undefined,
      video_quality: videoQuality || undefined,
      source_video_validation: sourceVideoValidation || undefined,
      browser_recording_clock: browserRecordingClock || undefined,
      attempt_timeout_s: attemptTimeoutS,
      derived_attempt_timeout_s: derivedAttemptTimeoutS,
      files: {
        events_jsonl: eventsPath,
        ui_webm: browserVideoPath || undefined,
        ui_mp4: mp4Path || undefined,
        task_mp4: taskMp4Path || undefined,
      },
      prompt_events: promptEvents,
      vlm_response_count: vlmResponses.length,
      post_query_vlm_response_count: postQueryVlmResponseCount,
      prompted_vlm_response_count: promptedVlmResponseCount,
      substantive_vlm_response_count: substantiveVlmResponseCount,
      completed_vlm_response_count: completedVlmResponseCount,
      deferred_vlm_response_count: deferredVlmResponseCount,
      query_echo_response_count: queryEchoResponseCount,
      protocol_violation_count: protocolViolations.length,
      protocol_violations: protocolViolations,
      provider_infrastructure_failure: providerInfrastructureFailure,
      first_prompted_response_frame_time_s: firstPromptedResponseFrameTimeS,
      prompted_response_frame_drift_s: promptedResponseFrameDriftS,
      vlm_responses: vlmResponses,
      final_vlm_response: vlmResponses.length ? vlmResponses[vlmResponses.length - 1] : null,
    };
    await fsp.writeFile(
      summaryPath,
      `${JSON.stringify(redactSecretsForLog(summary), null, 2)}\n`,
    );
    console.log(`[${index + 1}/${total}] ${task.id}: ${status}, responses=${vlmResponses.length}, out=${taskDir}`);
    if (status !== 'ok' && errorText) {
      console.error(`[${index + 1}/${total}] ${task.id}: final validation error: ${errorText}`);
    }
  }

  return { status, taskDir, summaryPath };
}

function readJsonIfExists(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function existingSummaryMatchesVlmSelection(summary, args) {
  if (
    summary?.model_response_outcome?.response_semantics
      !== MODEL_RESPONSE_OUTCOME_PROTOCOL
  ) return false;
  if (!args.vlmModel) return true;
  if (args.vlmProfileFingerprint) {
    if (summary?.vlm_profile?.fingerprint !== args.vlmProfileFingerprint) return false;
  } else if (
    summary?.vlm_model !== args.vlmModel
    || normalizeApiBase(summary?.vlm_api_base) !== normalizeApiBase(args.vlmApiBase)
  ) {
    return false;
  }
  if (summary?.vlm_route && summary.vlm_route !== args.vlmRoute) return false;
  if (args.vlmIdentityCheck && summary?.vlm_backend_identity?.ok !== true) return false;
  return true;
}

async function closeBrowserWithTimeout(browser, timeoutMs = 10000) {
  let timedOut = false;
  await Promise.race([
    browser.close().catch(() => {}),
    sleep(timeoutMs).then(() => {
      timedOut = true;
    }),
  ]);
  if (timedOut) {
    console.warn(`browser.close() timed out after ${timeoutMs}ms; killing browser process`);
    try {
      const proc = typeof browser.process === 'function' ? browser.process() : null;
      if (proc && !proc.killed) proc.kill('SIGKILL');
    } catch {
      // Best effort.
    }
  }
}

function isBrowserClosedError(errorText) {
  return /Target page, context or browser has been closed|Target closed|Browser has been closed|browser has been closed/i
    .test(String(errorText || ''));
}

function summaryHasBrowserClosedError(summaryPath) {
  const summary = readJsonIfExists(summaryPath);
  return isBrowserClosedError(summary?.error || '');
}

function firstLine(value) {
  return String(value || '').split(/\r?\n/)[0] || '';
}

function csvCell(value) {
  const text = value == null ? '' : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

function redactArgs(args) {
  return {
    ...args,
    webPassword: args.webPassword ? '[REDACTED]' : '',
    vlmApiKey: args.vlmApiKey ? '[REDACTED]' : '',
  };
}

function attemptRecordFromResult(task, index, attempt, result) {
  const summary = result?.summaryPath ? readJsonIfExists(result.summaryPath) : null;
  return {
    task_index: index + 1,
    task_id: task.id,
    attempt,
    status: summary?.status || result?.status || 'error',
    error: firstLine(summary?.error || result?.error),
    task_dir: result?.taskDir || '',
    summary_path: result?.summaryPath || '',
    input_mode: summary?.input_mode || '',
    effective_video_url: summary?.effective_video_url || '',
    actual_query_video_time_s: summary?.actual_query_video_time_s ?? null,
    stream_ready_offset_s: summary?.stream_ready_offset_s ?? null,
    original_video_start_offset_s: summary?.original_video_start_offset_s ?? null,
    vlm_response_count: summary?.vlm_response_count ?? null,
    post_query_vlm_response_count: summary?.post_query_vlm_response_count ?? null,
    prompted_vlm_response_count: summary?.prompted_vlm_response_count ?? null,
    vlm_profile_id: summary?.vlm_profile?.id || '',
    vlm_model: summary?.vlm_model || '',
    vlm_backend_identity_ok: summary?.vlm_backend_identity?.ok ?? null,
    vlm_observed_models: summary?.vlm_backend_identity?.observed_models?.join(' | ') || '',
    vlm_warmup_ok: summary?.vlm_inference_warmup?.ok ?? null,
    vlm_warmup_barrier_satisfied: summary?.vlm_inference_warmup?.barrier_satisfied ?? null,
    vlm_warmup_latest_ready_offset_s: summary?.vlm_inference_warmup?.latest_ready_offset_s ?? null,
    prompted_response_frame_drift_s: summary?.prompted_response_frame_drift_s ?? null,
    stream_reconnect_count: summary?.stream_reconnect_count ?? null,
    attempt_timeout_s: summary?.attempt_timeout_s ?? null,
    task_mp4: summary?.files?.task_mp4 || summary?.files?.ui_task_mp4 || '',
    started_at: summary?.started_at || '',
    finished_at: summary?.finished_at || '',
    files_kept: Boolean(result?.taskDir),
  };
}

async function deleteFailedAttemptDir(args, result, record) {
  if (args.keepFailedAttempts) return;
  if (record.status === 'ok' || record.status === 'skipped') return;
  if (!result?.taskDir) return;
  try {
    await fsp.rm(result.taskDir, { recursive: true, force: true });
    record.files_kept = false;
    record.task_dir = '';
    record.summary_path = '';
    record.task_mp4 = '';
  } catch (error) {
    record.cleanup_error = error.message;
  }
}

async function writeRunSummary(args, tasks, attemptRecords, runStartedAtIso, runStartedAtMs, status) {
  const taskResults = tasks.map((task, index) => {
    const attempts = attemptRecords.filter((record) => record.task_index === index + 1);
    const successful = attempts.find((record) => record.status === 'ok' || record.status === 'skipped');
    const finalAttempt = successful || attempts[attempts.length - 1] || null;
    return {
      task_index: index + 1,
      task_id: task.id,
      status: finalAttempt?.status || (status === 'running' ? 'pending' : 'not_run'),
      attempts: attempts.length,
      final_attempt: finalAttempt?.attempt ?? null,
      error: finalAttempt?.status === 'error' ? finalAttempt.error : '',
      summary_path: finalAttempt?.summary_path || '',
      task_dir: finalAttempt?.task_dir || '',
      actual_query_video_time_s: finalAttempt?.actual_query_video_time_s ?? null,
      prompted_vlm_response_count: finalAttempt?.prompted_vlm_response_count ?? null,
      vlm_profile_id: finalAttempt?.vlm_profile_id || '',
      vlm_model: finalAttempt?.vlm_model || '',
      vlm_backend_identity_ok: finalAttempt?.vlm_backend_identity_ok ?? null,
      vlm_observed_models: finalAttempt?.vlm_observed_models || '',
      vlm_warmup_ok: finalAttempt?.vlm_warmup_ok ?? null,
      vlm_warmup_barrier_satisfied: finalAttempt?.vlm_warmup_barrier_satisfied ?? null,
      vlm_warmup_latest_ready_offset_s: finalAttempt?.vlm_warmup_latest_ready_offset_s ?? null,
      stream_reconnect_count: finalAttempt?.stream_reconnect_count ?? null,
      task_mp4: finalAttempt?.task_mp4 || '',
    };
  });

  const counts = taskResults.reduce((acc, taskResult) => {
    acc[taskResult.status] = (acc[taskResult.status] || 0) + 1;
    return acc;
  }, {});
  const now = new Date();
  const summary = {
    status,
    output_dir: path.resolve(args.out),
    started_at: runStartedAtIso,
    finished_at: status === 'running' ? null : now.toISOString(),
    elapsed_s: (monotonicMs() - runStartedAtMs) / 1000,
    tasks_total: tasks.length,
    attempts_total: attemptRecords.length,
    counts,
    args: redactArgs(args),
    task_results: taskResults,
    attempts: attemptRecords,
  };
  await fsp.writeFile(path.join(args.out, 'run_summary.json'), `${JSON.stringify(summary, null, 2)}\n`);

  const header = [
    'task_index',
    'task_id',
    'status',
    'attempts',
    'final_attempt',
    'error',
    'summary_path',
    'task_dir',
    'actual_query_video_time_s',
    'prompted_vlm_response_count',
    'vlm_profile_id',
    'vlm_model',
    'vlm_backend_identity_ok',
    'vlm_observed_models',
    'vlm_warmup_ok',
    'vlm_warmup_barrier_satisfied',
    'vlm_warmup_latest_ready_offset_s',
    'stream_reconnect_count',
    'task_mp4',
  ];
  const csvLines = [
    header.join(','),
    ...taskResults.map((row) => header.map((key) => csvCell(row[key])).join(',')),
  ];
  await fsp.writeFile(path.join(args.out, 'run_summary.csv'), `${csvLines.join('\n')}\n`);
}

async function launchCaptureBrowser(args) {
  const browserEnv = Object.fromEntries(Object.entries(process.env).filter(([name]) => (
    !['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']
      .includes(name)
  )));
  return chromium.launch({
    headless: args.headless,
    env: browserEnv,
    args: [
      '--allow-insecure-localhost',
      '--autoplay-policy=no-user-gesture-required',
      '--disable-dev-shm-usage',
      '--no-proxy-server',
    ],
  });
}

async function commandExists(name) {
  if (name.includes(path.sep)) {
    try {
      await fsp.access(name, fs.constants.X_OK);
      return true;
    } catch {
      return false;
    }
  }
  const pathEnv = process.env.PATH || '';
  const dirs = pathEnv.split(path.delimiter);
  for (const dir of dirs) {
    const candidate = path.join(dir, name);
    try {
      await fsp.access(candidate, fs.constants.X_OK);
      return true;
    } catch {
      // continue
    }
  }
  return false;
}

function convertWebmToMp4(input, output, ffmpegBin = 'ffmpeg') {
  return new Promise((resolve) => {
    const ffmpeg = spawn(ffmpegBin, [
      '-y',
      '-i', input,
      '-c:v', 'libx264',
      '-preset', 'veryfast',
      '-threads', '8',
      '-pix_fmt', 'yuv420p',
      '-movflags', '+faststart',
      output,
    ], { stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    let settled = false;
    const finish = async (result) => {
      if (settled) return;
      settled = true;
      if (!result.ok) await fsp.unlink(output).catch(() => {});
      resolve(result);
    };
    ffmpeg.stderr.on('data', (chunk) => {
      stderr = `${stderr}${chunk}`.slice(-4 * 1024 * 1024);
    });
    ffmpeg.once('error', (error) => {
      void finish({
        ok: false,
        code: null,
        signal: null,
        error: `ffmpeg ui.mp4 conversion could not start: ${error.message}`,
      });
    });
    ffmpeg.once('close', (code, signal) => {
      if (code !== 0) {
        const stderrTail = stderr.trim().split('\n').slice(-40).join('\n');
        void finish({
          ok: false,
          code,
          signal: signal || null,
          error: `ffmpeg ui.mp4 conversion failed (code=${code}, signal=${signal || 'none'}): ${stderrTail}`,
        });
      } else {
        void finish({ ok: true, code, signal: signal || null });
      }
    });
  });
}

function trimMp4(input, output, startSeconds, durationSeconds) {
  return new Promise((resolve) => {
    const ffmpeg = spawn('ffmpeg', [
      '-y',
      '-ss', String(Math.max(0, startSeconds)),
      '-i', input,
      '-t', String(Math.max(0.1, durationSeconds)),
      '-c:v', 'libx264',
      '-pix_fmt', 'yuv420p',
      '-movflags', '+faststart',
      output,
    ], { stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    ffmpeg.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    ffmpeg.on('close', (code) => {
      if (code !== 0) {
        console.error(`ffmpeg trim failed for ${input}: ${stderr.split('\n').slice(-6).join('\n')}`);
        resolve(false);
      } else {
        resolve(true);
      }
    });
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const taskPath = path.resolve(args.tasks);
  let tasks = await readTasks(taskPath);
  if (args.limit > 0) tasks = tasks.slice(0, args.limit);
  if (!tasks.length) throw new Error('No tasks found');
  await ensureDir(args.out);
  const runStartedAtIso = new Date().toISOString();
  const runStartedAtMs = monotonicMs();
  const attemptRecords = [];
  const explicitlySkippedTaskIds = new Set(args.skipTaskIds);
  const unknownSkippedTaskIds = args.skipTaskIds.filter(
    (taskId) => !tasks.some((task) => task.id === taskId),
  );
  if (unknownSkippedTaskIds.length) {
    throw new Error(`--skip-task-id did not match task(s): ${unknownSkippedTaskIds.join(', ')}`);
  }
  let stopRequested = '';
  let signalCount = 0;
  const handleSignal = (signal) => {
    signalCount += 1;
    if (signalCount > 1) {
      console.warn(`Received ${signal} again; exiting immediately`);
      process.exit(130);
    }
    stopRequested = signal;
    console.warn(`Received ${signal}; will stop after the current attempt and write run summary`);
  };
  process.once('SIGINT', handleSignal);
  process.once('SIGTERM', handleSignal);

  console.log(`Loaded ${tasks.length} task(s) from ${taskPath}`);
  console.log(`WebUI: ${args.webUrl}`);
  console.log(`Output: ${path.resolve(args.out)}`);
  await writeRunSummary(args, tasks, attemptRecords, runStartedAtIso, runStartedAtMs, 'running');

  if (args.vlmModel) {
    try {
      await waitForWebUiHealthy(args, 'VLM preflight');
      args.vlmPreflightResult = await preflightVlmModel(args);
      console.log(
        `VLM preflight: ${args.vlmModel} is available via ${args.vlmRoute} at ${args.vlmApiBase}`,
      );
      await writeRunSummary(args, tasks, attemptRecords, runStartedAtIso, runStartedAtMs, 'running');
    } catch (error) {
      args.vlmPreflightResult = {
        required: args.vlmPreflight,
        ok: false,
        status: 'failed',
        requested_model: args.vlmModel,
        route: args.vlmRoute,
        api_base: args.vlmApiBase,
        error: error.message,
      };
      await writeRunSummary(args, tasks, attemptRecords, runStartedAtIso, runStartedAtMs, 'preflight_failed');
      throw error;
    }
  }

  let browser = await launchCaptureBrowser(args);

  try {
    for (let i = 0; i < tasks.length; i += 1) {
      if (stopRequested) break;
      if (explicitlySkippedTaskIds.has(tasks[i].id)) {
        console.log(`[${i + 1}/${tasks.length}] ${tasks[i].id}: explicitly quarantined for this partial run`);
        continue;
      }
      if (args.skipExisting) {
        const baseTaskName = `${String(i + 1).padStart(3, '0')}_${safeName(tasks[i].id)}`;
        const taskDir = path.resolve(args.out, baseTaskName);
        const summaryPath = path.join(taskDir, 'summary.json');
        const existing = readJsonIfExists(summaryPath);
        if (existing?.status === 'ok' && existingSummaryMatchesVlmSelection(existing, args)) {
          console.log(`[${i + 1}/${tasks.length}] ${tasks[i].id}: skip existing ok summary`);
          const result = { status: 'skipped', taskDir, summaryPath };
          attemptRecords.push(attemptRecordFromResult(tasks[i], i, 1, result));
          await writeRunSummary(args, tasks, attemptRecords, runStartedAtIso, runStartedAtMs, 'running');
          continue;
        }
      }
      if (i > 0 && args.taskGapS > 0) {
        console.log(`Waiting ${args.taskGapS}s before next task`);
        await sleep(args.taskGapS * 1000);
      }
      const maxAttempts = args.taskRetries + 1;
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        if (stopRequested) break;
        if (attempt > 1 && args.retryGapS > 0) {
          console.log(`[${i + 1}/${tasks.length}] ${tasks[i].id}: waiting ${args.retryGapS}s before retry ${attempt}/${maxAttempts}`);
          await sleep(args.retryGapS * 1000);
        }
        const label = attempt > 1
          ? `[${i + 1}/${tasks.length}] ${tasks[i].id} retry ${attempt}/${maxAttempts}`
          : `[${i + 1}/${tasks.length}] ${tasks[i].id}`;
        await waitForWebUiHealthy(args, label);
        if (args.freshBrowserPerAttempt) {
          await closeBrowserWithTimeout(browser);
          browser = await launchCaptureBrowser(args);
        }
        let result;
        try {
          result = await runTask(browser, args, tasks[i], i, tasks.length, attempt, maxAttempts);
        } catch (error) {
          const errorText = error.stack || error.message || String(error);
          console.error(`[${i + 1}/${tasks.length}] ${tasks[i].id}: ${error.message}`);
          if (isBrowserClosedError(errorText)) {
            console.warn(`[${i + 1}/${tasks.length}] ${tasks[i].id}: browser closed unexpectedly; relaunching Chromium`);
            await closeBrowserWithTimeout(browser);
            browser = await launchCaptureBrowser(args);
          }
          result = { status: 'error', taskDir: '', summaryPath: '', error: errorText };
        }
        const browserClosedInSummary = result.status === 'error'
          && result.summaryPath
          && summaryHasBrowserClosedError(result.summaryPath);
        const attemptRecord = attemptRecordFromResult(tasks[i], i, attempt, result);
        await deleteFailedAttemptDir(args, result, attemptRecord);
        attemptRecords.push(attemptRecord);
        await writeRunSummary(args, tasks, attemptRecords, runStartedAtIso, runStartedAtMs, 'running');
        if (browserClosedInSummary) {
          console.warn(`[${i + 1}/${tasks.length}] ${tasks[i].id}: browser/page closed during attempt; relaunching Chromium`);
          await closeBrowserWithTimeout(browser);
          browser = await launchCaptureBrowser(args);
        }
        if (result.status === 'ok' || result.status === 'skipped') break;
        if (attempt < maxAttempts) {
          console.log(`[${i + 1}/${tasks.length}] ${tasks[i].id}: attempt ${attempt}/${maxAttempts} failed; retrying`);
        }
      }
    }
  } finally {
    await closeBrowserWithTimeout(browser);
    const finalStatus = stopRequested
      ? `stopped:${stopRequested}`
      : explicitlySkippedTaskIds.size > 0 ? 'partial:explicit_skips' : 'complete';
    await writeRunSummary(args, tasks, attemptRecords, runStartedAtIso, runStartedAtMs, finalStatus).catch((error) => {
      console.error(`Failed to write run summary: ${error.message}`);
    });
    process.removeListener('SIGINT', handleSignal);
    process.removeListener('SIGTERM', handleSignal);
  }

  if (!stopRequested) {
    const successfulTaskIndexes = new Set(
      attemptRecords
        .filter((record) => record.status === 'ok' || record.status === 'skipped')
        .map((record) => record.task_index),
    );
    const incomplete = tasks.filter((task, index) => (
      !explicitlySkippedTaskIds.has(task.id)
      && !successfulTaskIndexes.has(index + 1)
    ));
    if (incomplete.length) {
      throw new Error(
        `Capture incomplete: ${incomplete.length}/${tasks.length} task(s) failed after retries: `
        + incomplete.map((task) => task.id).join(', '),
      );
    }
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error.stack || error.message || String(error));
      process.exit(1);
    });
}
