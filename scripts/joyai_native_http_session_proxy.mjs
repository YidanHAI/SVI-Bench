#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeJpegDataUrlToReference } from './jpeg_frame_normalizer.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const host = process.env.PROMPT_PROXY_HOST || '127.0.0.1';
const port = Number(process.env.PROMPT_PROXY_PORT || 18070);
const upstreamKey = String(process.env.UPSTREAM_API_KEY || '');
const accessToken = String(process.env.PROMPT_PROXY_ACCESS_TOKEN || '');
const advertisedModel = String(process.env.PROMPT_PROXY_ADVERTISED_MODEL || '').trim();
const upstreamApiBase = String(process.env.UPSTREAM_API_BASE || '').replace(/\/+$/, '');
const controlApiBase = String(process.env.NATIVE_REALTIME_API_BASE || '').replace(/\/+$/, '');
const realtimeProtocol = String(process.env.NATIVE_REALTIME_PROTOCOL || '').trim();
const realtimeQueryMode = String(process.env.NATIVE_REALTIME_QUERY_MODE || '').trim();
const nativeVideoSchema = String(process.env.NATIVE_VIDEO_SCHEMA || '').trim();
const streamingModePolicy = String(
  process.env.NATIVE_STREAMING_MODE_POLICY || 'proactive',
).trim();
const systemPromptTransport = String(
  process.env.NATIVE_HTTP_SYSTEM_PROMPT_TRANSPORT || 'system-role',
).trim();
const frameClockPolicy = String(
  process.env.NATIVE_HTTP_FRAME_CLOCK || 'inbound-turn',
).trim();
const nonQueryTimeoutMs = Number(process.env.NATIVE_HTTP_NON_QUERY_TIMEOUT_MS || 30000);
const queryTimeoutMs = Number(process.env.NATIVE_HTTP_QUERY_TIMEOUT_MS || 180000);
const resetTimeoutMs = Number(process.env.NATIVE_HTTP_RESET_TIMEOUT_MS || 60000);
const maxBodyBytes = Number(process.env.PROMPT_PROXY_MAX_BODY_BYTES || 32 * 1024 * 1024);
const promptPath = path.resolve(
  process.env.JOYAI_SYSTEM_PROMPT_FILE
    || path.join(ROOT, 'config', 'joyai_system_prompt.txt'),
);
const auditPath = path.resolve(
  process.env.PROMPT_PROXY_AUDIT_PATH
    || path.join(ROOT, 'outputs', 'joyai_native_http_session_proxy_audit.jsonl'),
);
const statePath = path.resolve(
  process.env.PROMPT_PROXY_STATE_PATH || `${auditPath}.session.json`,
);

if (!Number.isInteger(port) || port <= 0 || port > 65535) throw new Error('Invalid PROMPT_PROXY_PORT');
if (!upstreamKey) throw new Error('UPSTREAM_API_KEY is required');
if (accessToken.length < 24) throw new Error('PROMPT_PROXY_ACCESS_TOKEN must contain at least 24 characters');
if (!advertisedModel) throw new Error('PROMPT_PROXY_ADVERTISED_MODEL is required');
if (realtimeProtocol !== 'joyai-http-session-v1') {
  throw new Error(`Unsupported NATIVE_REALTIME_PROTOCOL: ${realtimeProtocol || '(missing)'}`);
}
if (realtimeQueryMode !== 'input-text-once') {
  throw new Error('joyai-http-session-v1 requires NATIVE_REALTIME_QUERY_MODE=input-text-once');
}
if (nativeVideoSchema !== 'joyai-http-session.image_url.jpeg') {
  throw new Error(`Unsupported NATIVE_VIDEO_SCHEMA: ${nativeVideoSchema || '(missing)'}`);
}
if (![
  'proactive',
  'interactive-after-query',
].includes(streamingModePolicy)) {
  throw new Error(`Unsupported NATIVE_STREAMING_MODE_POLICY: ${streamingModePolicy || '(missing)'}`);
}
if (!['system-role', 'inline-user-query'].includes(systemPromptTransport)) {
  throw new Error(
    `Unsupported NATIVE_HTTP_SYSTEM_PROMPT_TRANSPORT: ${systemPromptTransport || '(missing)'}`,
  );
}
if (!['inbound-turn', 'wall-media'].includes(frameClockPolicy)) {
  throw new Error(`Unsupported NATIVE_HTTP_FRAME_CLOCK: ${frameClockPolicy || '(missing)'}`);
}
for (const [name, value] of Object.entries({
  PROMPT_PROXY_PORT: port,
  NATIVE_HTTP_NON_QUERY_TIMEOUT_MS: nonQueryTimeoutMs,
  NATIVE_HTTP_QUERY_TIMEOUT_MS: queryTimeoutMs,
  NATIVE_HTTP_RESET_TIMEOUT_MS: resetTimeoutMs,
  PROMPT_PROXY_MAX_BODY_BYTES: maxBodyBytes,
})) {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be positive`);
}
for (const [name, value] of Object.entries({
  UPSTREAM_API_BASE: upstreamApiBase,
  NATIVE_REALTIME_API_BASE: controlApiBase,
})) {
  let parsed;
  try { parsed = new URL(value); } catch { throw new Error(`Invalid ${name}: ${value || '(missing)'}`); }
  if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error(`${name} must use HTTP or HTTPS`);
}

const promptFileBytes = fs.readFileSync(promptPath);
const systemPromptFileSha256 = crypto.createHash('sha256').update(promptFileBytes).digest('hex');
const systemPrompt = promptFileBytes.toString('utf8').trim();
const systemPromptSha256 = crypto.createHash('sha256').update(systemPrompt).digest('hex');
if (!systemPrompt) throw new Error(`System prompt is empty: ${promptPath}`);
await fsp.mkdir(path.dirname(auditPath), { recursive: true });

const sha256 = (value) => crypto.createHash('sha256').update(value).digest('hex');
let auditTail = Promise.resolve();
let serialTail = Promise.resolve();

function appendAudit(record) {
  const line = `${JSON.stringify({ timestamp: new Date().toISOString(), ...record })}\n`;
  auditTail = auditTail.catch(() => {}).then(() => fsp.appendFile(auditPath, line, 'utf8'));
  return auditTail;
}

function serialize(action) {
  const current = serialTail.then(action, action);
  serialTail = current.catch(() => {});
  return current;
}

function sendJson(res, statusCode, payload) {
  const body = Buffer.from(JSON.stringify(payload));
  res.writeHead(statusCode, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': String(body.length),
  });
  res.end(body);
}

function authorized(req) {
  const actual = Buffer.from(String(req.headers.authorization || ''));
  const expected = Buffer.from(`Bearer ${accessToken}`);
  return actual.length === expected.length && crypto.timingSafeEqual(actual, expected);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > maxBodyBytes) {
        reject(new Error(`Request body exceeds ${maxBodyBytes} bytes`));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function contentText(content) {
  if (typeof content === 'string') return content.trim();
  if (!Array.isArray(content)) return '';
  return content
    .filter((item) => ['text', 'input_text'].includes(item?.type))
    .map((item) => String(item.text || '').trim())
    .filter(Boolean)
    .join('\n');
}

function stripFrameTimeText(value) {
  return String(value || '')
    .split(/\r?\n/)
    .filter((line) => !/^\s*(?:\[?frame[ _-]?time(?:[ _-]?range)?\]?|frames? span)\s*[:=]/i.test(line))
    .join('\n')
    .trim();
}

function requestUserText(messages) {
  return stripFrameTimeText((Array.isArray(messages) ? messages : [])
    .filter((message) => message?.role === 'user')
    .map((message) => contentText(message.content))
    .filter(Boolean)
    .join('\n'));
}

function requestFrames(messages) {
  const frames = [];
  for (const message of Array.isArray(messages) ? messages : []) {
    if (!Array.isArray(message?.content)) continue;
    for (const item of message.content) {
      if (!['image_url', 'input_image'].includes(item?.type)) continue;
      const imageUrl = typeof item.image_url === 'string'
        ? item.image_url
        : String(item.image_url?.url || item.image_url || '');
      if (!/^data:image\/(?:jpeg|jpg);base64,[A-Za-z0-9+/=\r\n]+$/i.test(imageUrl)) {
        throw new Error('Native HTTP session input requires one inline JPEG WebUI frame');
      }
      frames.push(imageUrl.replace(/\s+/g, ''));
    }
  }
  return frames;
}

function requestSessionId(req, payload) {
  return String(
    req.headers['x-streaming-session']
    || req.headers['x-session-id']
    || payload.user
    || '',
  ).trim();
}

function requestFrameTime(req, payload) {
  const value = payload.frame_time_range
    || req.headers['x-frame-time-range']
    || req.headers['x-streaming-time-range']
    || '';
  return String(value).trim();
}

function providerSessionId(localSessionId) {
  const model = advertisedModel.replace(/[^A-Za-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 28);
  return `vlbench-${model}-${sha256(`${localSessionId}:${Date.now()}:${crypto.randomUUID()}`).slice(0, 20)}`;
}

async function fetchJson(url, { payload, headers = {}, timeoutMs }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...headers },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (error) {
    if (controller.signal.aborted) {
      error.deadlineExceeded = true;
      error.timeoutMs = timeoutMs;
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
  const raw = await response.text();
  let json;
  try { json = raw ? JSON.parse(raw) : {}; } catch { json = { raw }; }
  if (!response.ok) {
    const error = new Error(`Provider HTTP ${response.status}: ${raw.slice(0, 1200)}`);
    error.statusCode = response.status;
    throw error;
  }
  return { response, json };
}

function classifyProviderFailure(error) {
  const statusCode = Number(error?.statusCode) || null;
  const message = String(error?.message || error || 'Unknown provider failure');
  const deadlineExceeded = Boolean(
    error?.deadlineExceeded
    || error?.name === 'AbortError'
    || /operation was aborted|timed? out|timeout/i.test(message)
  );
  if (deadlineExceeded) {
    return {
      failure_class: 'model_latency_timeout',
      retryable_infrastructure: false,
      deadline_exceeded: true,
      http_status: statusCode,
    };
  }
  if (/Frames yielded inconsistent|smart_resize/i.test(message)) {
    return {
      failure_class: 'provider_frame_preprocessing_error',
      retryable_infrastructure: true,
      deadline_exceeded: false,
      http_status: statusCode,
    };
  }
  return {
    failure_class: statusCode === 429
      ? 'provider_rate_limit'
      : statusCode != null
        ? 'provider_http_error'
        : 'provider_transport_error',
    retryable_infrastructure: true,
    deadline_exceeded: false,
    http_status: statusCode,
  };
}

function responseText(payload) {
  const value = payload?.choices?.[0]?.message?.content;
  if (typeof value === 'string') return value.trim();
  if (!Array.isArray(value)) return '';
  return value.map((item) => item?.text || item?.content || '').filter(Boolean).join('\n').trim();
}

function normalizeProviderOutput(value) {
  const raw = String(value || '').trim();
  if (!raw || /^<\s*\/?\s*silence\s*>$/i.test(raw)) return '</silence>';

  const responsePrefix = /^<\s*\/?\s*response\s*>\s*/i;
  if (responsePrefix.test(raw)) {
    const body = raw.replace(responsePrefix, '').trim();
    return body ? `</response> ${body}` : '</silence>';
  }

  // Mage's official StreamMind path returns decoded model text, not JoyAI
  // action tags. The boundary adapter owns conversion to the WebUI protocol.
  return `</response> ${raw}`;
}

function openAiSse(payload) {
  const choice = payload.choices[0];
  const first = {
    id: payload.id,
    object: 'chat.completion.chunk',
    created: payload.created,
    model: payload.model,
    choices: [{ index: 0, delta: choice.message, finish_reason: null }],
  };
  const last = {
    id: payload.id,
    object: 'chat.completion.chunk',
    created: payload.created,
    model: payload.model,
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
  };
  return `data: ${JSON.stringify(first)}\n\ndata: ${JSON.stringify(last)}\n\ndata: [DONE]\n\n`;
}

const conversation = {
  localSessionId: '',
  providerSessionId: '',
  mageFrameReferenceDataUrl: '',
  activeQuerySha256: '',
  interactiveQueryActive: false,
  queryRoundCount: 0,
  lastInboundQuery: '',
  deliveredQueryEventIds: new Set(),
  mediaClockAnchorEpochMs: null,
  mediaClockAnchorS: null,
  lastProviderFrameTimeS: null,
};
const queryQueues = new Map();
const queryRecords = new Map();

async function persistProviderState() {
  if (!conversation.providerSessionId) {
    await fsp.rm(statePath, { force: true });
    return;
  }
  const temporary = `${statePath}.tmp.${process.pid}`;
  await fsp.mkdir(path.dirname(statePath), { recursive: true });
  await fsp.writeFile(temporary, `${JSON.stringify({
    provider_session_id: conversation.providerSessionId,
    local_session_id: conversation.localSessionId,
    model: advertisedModel,
    updated_at: new Date().toISOString(),
  }, null, 2)}\n`, { mode: 0o600 });
  await fsp.rename(temporary, statePath);
}

async function resetProviderSession(sessionId, reason) {
  if (!sessionId) return { ok: true, existed: false };
  const resetUrl = `${controlApiBase}/streaming/reset`;
  const { json } = await fetchJson(resetUrl, {
    payload: { session_id: sessionId, user: sessionId },
    headers: { 'x-streaming-session': sessionId },
    timeoutMs: resetTimeoutMs,
  });
  if (json?.ok !== true || String(json.session_id || '') !== sessionId) {
    throw new Error(`Provider reset returned no matching acknowledgement: ${JSON.stringify(json).slice(0, 600)}`);
  }
  await appendAudit({
    event: 'provider_session_closed',
    provider_session_id: sessionId,
    reason,
    reset_acknowledged: true,
  });
  return json;
}

async function resetConversation(reason = 'api_reset') {
  const previousSessionId = conversation.providerSessionId;
  if (previousSessionId) await resetProviderSession(previousSessionId, reason);
  conversation.localSessionId = '';
  conversation.providerSessionId = '';
  conversation.mageFrameReferenceDataUrl = '';
  conversation.activeQuerySha256 = '';
  conversation.interactiveQueryActive = false;
  conversation.queryRoundCount = 0;
  conversation.lastInboundQuery = '';
  conversation.deliveredQueryEventIds.clear();
  conversation.mediaClockAnchorEpochMs = null;
  conversation.mediaClockAnchorS = null;
  conversation.lastProviderFrameTimeS = null;
  queryQueues.clear();
  queryRecords.clear();
  await persistProviderState();
  await appendAudit({
    event: 'conversation_state_reset',
    provider_session_id: previousSessionId,
    input_transport: 'native-video-realtime',
    query_delivery: 'once_per_round',
    reason,
  });
  return { ok: true, provider_session_id: previousSessionId, reset_acknowledged: true };
}

async function recoverPersistedSession() {
  let persisted;
  try { persisted = JSON.parse(await fsp.readFile(statePath, 'utf8')); } catch { return; }
  const sessionId = String(persisted?.provider_session_id || '');
  if (!sessionId) return;
  await resetProviderSession(sessionId, 'proxy_startup_recovery');
  await fsp.rm(statePath, { force: true });
}

function queueFor(sessionId) {
  if (!queryQueues.has(sessionId)) queryQueues.set(sessionId, []);
  return queryQueues.get(sessionId);
}

function frameTimeStart(value) {
  const match = String(value || '').match(/\d+(?:\.\d+)?/);
  if (!match) return null;
  const parsed = Number(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
}

function exactQueryMediaTime(record) {
  for (const candidate of [
    record?.captured_media_time_s,
    record?.ui_query_video_time_s,
    frameTimeStart(record?.frame_time_range),
  ]) {
    const value = Number(candidate);
    if (Number.isFinite(value) && value >= 0) return value;
  }
  return null;
}

function providerFrameTimeRange(inboundRange, queued, nowMs) {
  if (frameClockPolicy === 'inbound-turn') return inboundRange;

  if (queued) {
    const exact = exactQueryMediaTime(queued);
    if (exact != null) {
      conversation.mediaClockAnchorEpochMs = queued.enqueued_epoch_ms;
      conversation.mediaClockAnchorS = exact;
      conversation.lastProviderFrameTimeS = exact;
      return `${exact.toFixed(3)} seconds`;
    }
  }

  const inboundStart = frameTimeStart(inboundRange);
  if (
    conversation.mediaClockAnchorEpochMs == null
    || conversation.mediaClockAnchorS == null
  ) {
    conversation.mediaClockAnchorEpochMs = nowMs;
    conversation.mediaClockAnchorS = inboundStart ?? 0;
  }
  let mediaTime = conversation.mediaClockAnchorS
    + Math.max(0, nowMs - conversation.mediaClockAnchorEpochMs) / 1000;
  if (inboundStart != null) mediaTime = Math.max(mediaTime, inboundStart);
  if (conversation.lastProviderFrameTimeS != null) {
    mediaTime = Math.max(mediaTime, conversation.lastProviderFrameTimeS + 0.001);
  }
  conversation.lastProviderFrameTimeS = mediaTime;
  return `${mediaTime.toFixed(3)} seconds`;
}

function providerQueryText(query) {
  if (!query || systemPromptTransport !== 'inline-user-query') return query;
  return [
    '<system_prompt>',
    systemPrompt,
    '</system_prompt>',
    '<user_query>',
    query,
    '</user_query>',
  ].join('\n');
}

async function handleQueryEvent(req, res) {
  const payload = JSON.parse((await readBody(req)).toString('utf8'));
  const sessionId = String(payload.session_id || '').trim();
  const eventId = String(payload.query_event_id || '').trim();
  const query = String(payload.query || '').trim();
  const imageUrl = String(payload.image_url || '').replace(/\s+/g, '');
  const frameTimeRange = String(payload.frame_time_range || '').trim();
  if (!sessionId || !eventId || !query || !frameTimeRange) {
    throw new Error('session_id, query_event_id, query, and frame_time_range are required');
  }
  if (!/^data:image\/(?:jpeg|jpg);base64,[A-Za-z0-9+/=]+$/i.test(imageUrl)) {
    throw new Error('query event image_url must be an inline JPEG');
  }
  if (queryRecords.has(eventId)) {
    const existing = queryRecords.get(eventId);
    sendJson(res, 200, {
      ok: true,
      idempotent: true,
      query_event_id: eventId,
      status: existing.status,
      queue_position: existing.queue_position,
    });
    return;
  }
  const queue = queueFor(sessionId);
  const acceptedEpochMs = Date.now();
  const uiQuerySentEpochMs = Date.parse(String(payload.ui_query_sent_at || ''));
  const record = {
    session_id: sessionId,
    query_event_id: eventId,
    query,
    user_query_sha256: sha256(query),
    image_url: imageUrl,
    image_sha256: sha256(imageUrl),
    frame_time_range: frameTimeRange,
    ui_query_sent_at: String(payload.ui_query_sent_at || ''),
    ui_query_video_time_s: Number(payload.ui_query_video_time_s),
    annotated_query_video_time_s: Number(payload.annotated_query_video_time_s),
    captured_media_time_s: Number(payload.captured_media_time_s),
    captured_raw_media_time_s: Number(payload.captured_raw_media_time_s),
    enqueued_epoch_ms: acceptedEpochMs,
    acceptance_delay_ms: Number.isFinite(uiQuerySentEpochMs)
      ? Math.max(0, acceptedEpochMs - uiQuerySentEpochMs)
      : null,
    status: 'queued',
    queue_position: queue.length + 1,
  };
  queue.push(record);
  queryRecords.set(eventId, record);
  await appendAudit({
    event: 'query_frame_enqueued',
    query_event_id: eventId,
    session_id: sessionId,
    user_query_present: true,
    user_query_sha256: record.user_query_sha256,
    frame_time_range: frameTimeRange,
    image_sha256: record.image_sha256,
    ui_query_sent_at: record.ui_query_sent_at,
    ui_query_video_time_s: record.ui_query_video_time_s,
    annotated_query_video_time_s: record.annotated_query_video_time_s,
    captured_media_time_s: record.captured_media_time_s,
    captured_raw_media_time_s: record.captured_raw_media_time_s,
    acceptance_delay_ms: record.acceptance_delay_ms,
    queue_position: record.queue_position,
    query_frame_queue_policy: 'fifo-query-time-frame',
  });
  sendJson(res, 200, {
    ok: true,
    idempotent: false,
    query_event_id: eventId,
    status: 'queued',
    queue_position: record.queue_position,
    acceptance_delay_ms: record.acceptance_delay_ms,
  });
}

async function handleProviderWarmup(req, res) {
  const payload = JSON.parse((await readBody(req)).toString('utf8'));
  const imageUrl = String(payload.image_url || '').replace(/\s+/g, '');
  if (!/^data:image\/(?:jpeg|jpg);base64,[A-Za-z0-9+/=]+$/i.test(imageUrl)) {
    throw new Error('image_url must be an inline JPEG');
  }
  const requestId = crypto.randomUUID();
  const started = Date.now();
  const { json } = await fetchJson(`${upstreamApiBase}/chat/completions`, {
    payload: {
      model: advertisedModel,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: [{ type: 'image_url', image_url: { url: imageUrl } }] },
      ],
      max_tokens: 32,
      temperature: 0,
    },
    headers: { authorization: `Bearer ${upstreamKey}` },
    timeoutMs: queryTimeoutMs,
  });
  const responseId = String(json?.id || '');
  const responseModel = String(json?.model || advertisedModel);
  if (!json?.choices?.[0]?.message || !responseId) {
    throw new Error('Provider warmup returned no assistant message or response id');
  }
  await appendAudit({
    event: 'provider_warmup_completed',
    request_id: requestId,
    response_id: responseId,
    response_model: responseModel,
    ok: true,
    multimodal_input: true,
    session_state_unchanged: true,
    latency_ms: Date.now() - started,
  });
  sendJson(res, 200, {
    ok: true,
    model: advertisedModel,
    response_model: responseModel,
    response_id: responseId,
    latency_ms: Date.now() - started,
    multimodal_input: true,
    session_state_unchanged: true,
  });
}

async function handleChatCompletion(req, res) {
  const inbound = JSON.parse((await readBody(req)).toString('utf8'));
  const localSessionId = requestSessionId(req, inbound);
  if (!localSessionId) throw new Error('x-streaming-session is required');
  if (conversation.localSessionId && conversation.localSessionId !== localSessionId) {
    const error = new Error(`Proxy session ${conversation.localSessionId} must be reset before ${localSessionId}`);
    error.statusCode = 409;
    throw error;
  }
  let creatingSession = false;
  if (!conversation.localSessionId) {
    conversation.localSessionId = localSessionId;
    conversation.providerSessionId = providerSessionId(localSessionId);
    creatingSession = true;
    await persistProviderState();
  }

  const messages = Array.isArray(inbound.messages) ? inbound.messages : [];
  const inboundFrames = requestFrames(messages);
  if (inboundFrames.length !== 1) {
    throw new Error(`Expected exactly one causal WebUI JPEG frame, received ${inboundFrames.length}`);
  }
  if (/mage/i.test(advertisedModel) && !conversation.mageFrameReferenceDataUrl) {
    conversation.mageFrameReferenceDataUrl = inboundFrames[0];
  }
  const inboundQuery = requestUserText(messages);
  const queue = queueFor(localSessionId);
  const queued = queue.shift() || null;
  let query = '';
  let frame = inboundFrames[0];
  let frameTimeRange = requestFrameTime(req, inbound);
  const inboundFrameTimeRange = frameTimeRange;
  let queryEventId = '';
  let queryQueueDelayMs = null;
  if (queued) {
    queued.status = 'processing';
    query = queued.query;
    frame = queued.image_url;
    frameTimeRange = queued.frame_time_range;
    queryEventId = queued.query_event_id;
    queryQueueDelayMs = Date.now() - queued.enqueued_epoch_ms;
  } else if (inboundQuery && inboundQuery !== conversation.lastInboundQuery) {
    query = inboundQuery;
  }
  let providerFrameNormalization = null;
  let queryFrameNormalization = null;
  if (/mage/i.test(advertisedModel)) {
    const originalFrame = frame;
    const normalized = await normalizeJpegDataUrlToReference(
      frame,
      conversation.mageFrameReferenceDataUrl,
    );
    frame = normalized.imageUrl;
    providerFrameNormalization = {
      policy: 'mage-session-uniform-frame-size-v2',
      applied: normalized.normalized,
      source_size: normalized.sourceSize,
      target_size: normalized.targetSize,
      preserves_frame_content_and_timing: true,
    };
    if (queued) {
      queryFrameNormalization = {
        ...providerFrameNormalization,
        preserves_query_time_content: true,
      };
    }
    if (normalized.normalized) {
      await appendAudit({
        event: queued
          ? 'query_frame_normalized_for_provider'
          : 'stream_frame_normalized_for_provider',
        query_event_id: queued ? queryEventId : '',
        provider_session_id: conversation.providerSessionId,
        policy: providerFrameNormalization.policy,
        source_size: normalized.sourceSize,
        target_size: normalized.targetSize,
        original_image_sha256: sha256(originalFrame),
        normalized_image_sha256: sha256(frame),
      });
    }
  }
  conversation.lastInboundQuery = inboundQuery;
  const providerFrameRange = providerFrameTimeRange(frameTimeRange, queued, Date.now());
  const querySha256 = query ? sha256(query) : '';
  const providerStreamingMode = (
    streamingModePolicy === 'interactive-after-query'
    && (Boolean(query) || conversation.interactiveQueryActive)
  ) ? 'interactive' : 'proactive';
  const requestId = crypto.randomUUID();
  if (query) {
    await appendAudit({
      event: 'provider_query_submission_started',
      request_id: requestId,
      provider_session_id: conversation.providerSessionId,
      realtime_protocol: realtimeProtocol,
      input_transport: 'native-video-realtime',
      native_video_schema: nativeVideoSchema,
      query_transport: realtimeQueryMode,
      user_query_present: true,
      user_query_sha256: querySha256,
      query_frame_time_range: frameTimeRange,
      query_frame_queue_event_id: queryEventId,
    });
  }

  const content = [{ type: 'image_url', image_url: { url: frame } }];
  const providerQuery = providerQueryText(query);
  if (providerQuery) content.push({ type: 'text', text: providerQuery });
  const providerMessages = [];
  if (systemPromptTransport === 'system-role') {
    providerMessages.push({ role: 'system', content: systemPrompt });
  }
  providerMessages.push({ role: 'user', content });
  const providerPayload = {
    model: advertisedModel,
    messages: providerMessages,
    frame_time_range: providerFrameRange,
    streaming_mode: providerStreamingMode,
  };
  const expectedRuntime = /moss/i.test(advertisedModel) ? 'moss-native' : 'mage-streammind';
  const started = Date.now();
  const requestUsesQueryBudget = Boolean(query) || providerStreamingMode === 'interactive';
  let json;
  try {
    ({ json } = await fetchJson(`${upstreamApiBase}/chat/completions`, {
      payload: providerPayload,
      headers: {
        authorization: `Bearer ${upstreamKey}`,
        'x-streaming-session': conversation.providerSessionId,
        ...(providerFrameRange ? { 'x-frame-time-range': providerFrameRange } : {}),
      },
      timeoutMs: requestUsesQueryBudget ? queryTimeoutMs : nonQueryTimeoutMs,
    }));
  } catch (error) {
    const failure = classifyProviderFailure(error);
    const modelLatencyTimeout = failure.failure_class === 'model_latency_timeout';
    const adapterResponsePolicy = modelLatencyTimeout
      ? 'audited_synthetic_silence_http_200'
      : 'retryable_provider_error';
    await appendAudit({
      event: 'native_realtime_frame_forwarded',
      request_id: requestId,
      model: String(inbound.model || advertisedModel),
      response_model: advertisedModel,
      provider_session_id: conversation.providerSessionId,
      provider_response_id: '',
      realtime_protocol: realtimeProtocol,
      input_transport: 'native-video-realtime',
      native_video_schema: nativeVideoSchema,
      frame_count: 1,
      query_delivery: 'once_per_round',
      query_transport: realtimeQueryMode,
      query_round_index: query ? conversation.queryRoundCount + 1 : null,
      query_frame_queue_policy: queryEventId ? 'fifo-query-time-frame' : '',
      query_frame_queue_event_id: queryEventId,
      query_frame_time_range: query ? frameTimeRange : '',
      inbound_frame_time_range: inboundFrameTimeRange,
      provider_frame_time_range: providerFrameRange,
      query_frame_queue_delay_ms: queryQueueDelayMs,
      query_frame_normalization: queryFrameNormalization,
      provider_frame_normalization: providerFrameNormalization,
      user_query_present: Boolean(query),
      user_query_sha256: querySha256,
      persistent_query_injected: false,
      streaming_mode_policy: streamingModePolicy,
      provider_streaming_mode: providerStreamingMode,
      provider_output: false,
      provider_output_query_sha256: '',
      forwarding_status: modelLatencyTimeout ? 'model_latency_timeout' : 'provider_error',
      adapter_response_policy: adapterResponsePolicy,
      ...failure,
      system_prompt_file_sha256: systemPromptFileSha256,
      system_prompt_sha256: systemPromptSha256,
      system_prompt_transport: systemPromptTransport,
      frame_clock_policy: frameClockPolicy,
    });
    await appendAudit({
      event: 'upstream_response_received',
      request_id: requestId,
      response_id: '',
      response_model: advertisedModel,
      ok: false,
      has_assistant_message: false,
      provider_output: false,
      provider_session_id: conversation.providerSessionId,
      latency_ms: Date.now() - started,
      adapter_response_policy: adapterResponsePolicy,
      ...failure,
      error: String(error?.message || error),
      system_prompt_file_sha256: systemPromptFileSha256,
      system_prompt_sha256: systemPromptSha256,
    });
    if (modelLatencyTimeout) {
      if (query) {
        conversation.activeQuerySha256 = querySha256;
        conversation.interactiveQueryActive = streamingModePolicy === 'interactive-after-query';
        conversation.queryRoundCount += 1;
        if (queryEventId) conversation.deliveredQueryEventIds.add(queryEventId);
        if (queued) queued.status = 'timed_out';
        await appendAudit({
          event: 'provider_query_delivered',
          request_id: requestId,
          provider_session_id: conversation.providerSessionId,
          provider_response_id: '',
          realtime_protocol: realtimeProtocol,
          input_transport: 'native-video-realtime',
          native_video_schema: nativeVideoSchema,
          query_delivery: 'once_per_round',
          query_transport: realtimeQueryMode,
          query_round_index: conversation.queryRoundCount,
          user_query_present: true,
          user_query_sha256: querySha256,
          query_frame_time_range: frameTimeRange,
          query_frame_queue_event_id: queryEventId,
          delivery_status: 'model_latency_timeout',
          system_prompt_file_sha256: systemPromptFileSha256,
          system_prompt_sha256: systemPromptSha256,
        });
      }
      const timeoutQueryQueueMetadata = queryEventId ? {
        policy: 'fifo-query-time-frame',
        query_event_id: queryEventId,
        user_query_sha256: querySha256,
        frame_time_range: frameTimeRange,
        provider_frame_time_range: providerFrameRange,
        inbound_frame_time_range: inboundFrameTimeRange,
        query_queue_delay_ms: queryQueueDelayMs,
        frame_normalization: queryFrameNormalization,
      } : null;
      const timeoutJson = {
        id: `chatcmpl-timeout-${requestId}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: advertisedModel,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: '</silence>' },
          finish_reason: 'stop',
        }],
        usage: null,
        streamingharness: {
          runtime: expectedRuntime,
          session_id: localSessionId,
          provider_session_id: conversation.providerSessionId,
          query_frame_queue: timeoutQueryQueueMetadata,
          provider_output_format: 'mage-decoded-text-adapted-to-joyai-actions',
          synthetic_timeout: true,
          failure_class: failure.failure_class,
          adapter_response_policy: adapterResponsePolicy,
        },
        native_realtime: {
          protocol: realtimeProtocol,
          video_schema: nativeVideoSchema,
          provider_output: false,
          provider_output_query_sha256: '',
          streaming_mode_policy: streamingModePolicy,
          provider_streaming_mode: providerStreamingMode,
          interactive_query_pending: conversation.interactiveQueryActive,
          interactive_segment_consumed: false,
          synthetic_timeout: true,
          failure_class: failure.failure_class,
        },
      };
      if (inbound.stream === true) {
        const body = Buffer.from(openAiSse(timeoutJson));
        res.writeHead(200, {
          'content-type': 'text/event-stream; charset=utf-8',
          'cache-control': 'no-cache',
          'content-length': String(body.length),
        });
        res.end(body);
      } else {
        sendJson(res, 200, timeoutJson);
      }
      return;
    }
    throw error;
  }
  const runtime = String(json?.streamingharness?.runtime || '');
  if (runtime !== expectedRuntime) {
    throw new Error(`Provider did not use ${expectedRuntime}; observed runtime=${runtime || '(missing)'}`);
  }
  const providerResponseId = String(json?.id || '');
  const providerResponseModel = String(json?.model || advertisedModel);
  if (!providerResponseId || !json?.choices?.[0]?.message) {
    throw new Error('Provider returned no chat-completion id or assistant message');
  }
  const providerText = responseText(json);
  const text = normalizeProviderOutput(providerText);
  const substantive = text !== '</silence>';
  const processedSegmentThisTurn = (
    Array.isArray(json?.streamingharness?.gate_probabilities)
    && json.streamingharness.gate_probabilities.length > 0
  );
  if (creatingSession) {
    await appendAudit({
      event: 'provider_session_created',
      provider_session_id: conversation.providerSessionId,
      provider_response_id: providerResponseId,
      realtime_protocol: realtimeProtocol,
      input_transport: 'native-video-realtime',
      native_video_schema: nativeVideoSchema,
      query_delivery: 'once_per_round',
      query_transport: realtimeQueryMode,
      user_query_present: Boolean(query),
      user_query_sha256: querySha256,
      system_prompt_file_sha256: systemPromptFileSha256,
      system_prompt_sha256: systemPromptSha256,
      system_prompt_transport: systemPromptTransport,
      frame_clock_policy: frameClockPolicy,
    });
  }
  if (query) {
    conversation.activeQuerySha256 = querySha256;
    conversation.interactiveQueryActive = (
      streamingModePolicy === 'interactive-after-query'
      && !processedSegmentThisTurn
    );
    conversation.queryRoundCount += 1;
    if (queryEventId) conversation.deliveredQueryEventIds.add(queryEventId);
    if (queued) queued.status = 'delivered';
    await appendAudit({
      event: 'provider_query_delivered',
      request_id: requestId,
      provider_session_id: conversation.providerSessionId,
      provider_response_id: providerResponseId,
      realtime_protocol: realtimeProtocol,
      input_transport: 'native-video-realtime',
      native_video_schema: nativeVideoSchema,
      query_delivery: 'once_per_round',
      query_transport: realtimeQueryMode,
      query_round_index: conversation.queryRoundCount,
      user_query_present: true,
      user_query_sha256: querySha256,
      query_frame_time_range: frameTimeRange,
      query_frame_queue_event_id: queryEventId,
      system_prompt_file_sha256: systemPromptFileSha256,
      system_prompt_sha256: systemPromptSha256,
    });
  } else if (providerStreamingMode === 'interactive' && processedSegmentThisTurn) {
    conversation.interactiveQueryActive = false;
  }
  const queryQueueMetadata = queryEventId ? {
    policy: 'fifo-query-time-frame',
    query_event_id: queryEventId,
    user_query_sha256: querySha256,
      frame_time_range: frameTimeRange,
      provider_frame_time_range: providerFrameRange,
      inbound_frame_time_range: inboundFrameTimeRange,
    query_queue_delay_ms: queryQueueDelayMs,
    frame_normalization: queryFrameNormalization,
  } : null;
  json.model = advertisedModel;
  json.choices[0].message.content = text;
  json.streamingharness = {
    ...(json.streamingharness || {}),
    session_id: localSessionId,
    provider_session_id: conversation.providerSessionId,
    query_frame_queue: queryQueueMetadata,
    provider_output_format: 'mage-decoded-text-adapted-to-joyai-actions',
  };
  json.native_realtime = {
    protocol: realtimeProtocol,
    video_schema: nativeVideoSchema,
    provider_output: substantive,
    provider_output_query_sha256: substantive ? conversation.activeQuerySha256 : '',
    streaming_mode_policy: streamingModePolicy,
    provider_streaming_mode: providerStreamingMode,
    interactive_query_pending: conversation.interactiveQueryActive,
    interactive_segment_consumed: processedSegmentThisTurn,
  };
  await appendAudit({
    event: 'native_realtime_frame_forwarded',
    request_id: requestId,
    model: String(inbound.model || advertisedModel),
    response_model: advertisedModel,
    provider_session_id: conversation.providerSessionId,
    provider_response_id: providerResponseId,
    realtime_protocol: realtimeProtocol,
    input_transport: 'native-video-realtime',
    native_video_schema: nativeVideoSchema,
    frame_count: 1,
    query_delivery: 'once_per_round',
    query_transport: realtimeQueryMode,
    query_round_index: query ? conversation.queryRoundCount : null,
    query_frame_queue_policy: queryEventId ? 'fifo-query-time-frame' : '',
    query_frame_queue_event_id: queryEventId,
    query_frame_time_range: query ? frameTimeRange : '',
    inbound_frame_time_range: inboundFrameTimeRange,
    provider_frame_time_range: providerFrameRange,
    query_frame_queue_delay_ms: queryQueueDelayMs,
    query_frame_normalization: queryFrameNormalization,
    provider_frame_normalization: providerFrameNormalization,
    user_query_present: Boolean(query),
    user_query_sha256: querySha256,
    persistent_query_injected: false,
    streaming_mode_policy: streamingModePolicy,
    provider_streaming_mode: providerStreamingMode,
    interactive_query_pending: conversation.interactiveQueryActive,
    interactive_segment_consumed: processedSegmentThisTurn,
    provider_output: substantive,
    provider_output_query_sha256: substantive ? conversation.activeQuerySha256 : '',
    system_prompt_file_sha256: systemPromptFileSha256,
    system_prompt_sha256: systemPromptSha256,
    system_prompt_transport: systemPromptTransport,
    frame_clock_policy: frameClockPolicy,
  });
  await appendAudit({
    event: 'upstream_response_received',
    request_id: requestId,
    response_id: providerResponseId,
    response_model: providerResponseModel,
    ok: true,
    has_assistant_message: true,
    provider_output: substantive,
    provider_session_id: conversation.providerSessionId,
    latency_ms: Date.now() - started,
    system_prompt_file_sha256: systemPromptFileSha256,
    system_prompt_sha256: systemPromptSha256,
  });

  if (inbound.stream === true) {
    const body = Buffer.from(openAiSse(json));
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache',
      'content-length': String(body.length),
    });
    res.end(body);
  } else {
    sendJson(res, 200, json);
  }
}

await recoverPersistedSession();

const server = http.createServer((req, res) => {
  let requestPath;
  try {
    requestPath = new URL(req.url || '/', 'http://localhost').pathname;
  } catch {
    sendJson(res, 400, { error: { message: 'Malformed request target' } });
    return;
  }
  if (req.method === 'GET' && requestPath === '/health') {
    sendJson(res, 200, {
      ok: true,
      model: advertisedModel,
      upstream_protocol: 'openai-chat',
      realtime_protocol: realtimeProtocol,
      input_transport: 'native-video-realtime',
      native_video_schema: nativeVideoSchema,
      query_delivery: 'once_per_round',
      query_transport: realtimeQueryMode,
      streaming_mode_policy: streamingModePolicy,
      system_prompt_transport: systemPromptTransport,
      frame_clock_policy: frameClockPolicy,
      interactive_query_active: conversation.interactiveQueryActive,
      provider_session_open: Boolean(conversation.providerSessionId),
      provider_session_id: conversation.providerSessionId,
      queued_query_events: [...queryQueues.values()].reduce((sum, queue) => sum + queue.length, 0),
      prompt_source_path: promptPath,
      system_prompt_file_sha256: systemPromptFileSha256,
      system_prompt_sha256: systemPromptSha256,
      audit_path: auditPath,
      state_path: statePath,
    });
    return;
  }
  if (!authorized(req)) {
    sendJson(res, 401, { error: { message: 'Unauthorized native HTTP session proxy request' } });
    return;
  }
  if (req.method === 'GET' && requestPath === '/v1/models') {
    sendJson(res, 200, {
      object: 'list',
      data: [{ id: advertisedModel, object: 'model', owned_by: 'native-http-session-provider' }],
    });
    return;
  }
  if (req.method === 'POST' && ['/reset', '/v1/streaming/reset'].includes(requestPath)) {
    serialize(() => resetConversation(requestPath === '/reset' ? 'api_reset' : 'warmup_session_reset'))
      .then((result) => sendJson(res, 200, result))
      .catch((error) => sendJson(res, error.statusCode || 502, { error: { message: error.message } }));
    return;
  }
  if (req.method === 'POST' && requestPath === '/v1/query-events') {
    // Query ingress is a control-plane event. It must be accepted at the
    // annotated media time even while the serial provider lane is busy.
    handleQueryEvent(req, res).catch((error) => {
      if (!res.headersSent) sendJson(res, error.statusCode || 400, { error: { message: error.message } });
    });
    return;
  }
  if (req.method === 'POST' && requestPath === '/v1/warmup') {
    serialize(() => handleProviderWarmup(req, res)).catch((error) => {
      appendAudit({ event: 'provider_warmup_failed', error: error.message }).catch(() => {});
      if (!res.headersSent) sendJson(res, error.statusCode || 502, { error: { message: error.message } });
    });
    return;
  }
  if (req.method === 'POST' && requestPath === '/v1/chat/completions') {
    serialize(() => handleChatCompletion(req, res)).catch((error) => {
      appendAudit({
        event: 'proxy_error',
        error: error.message,
        provider_session_id: conversation.providerSessionId,
      }).catch(() => {});
      if (!res.headersSent) sendJson(res, error.statusCode || 502, { error: { message: error.message } });
      else res.destroy(error);
    });
    return;
  }
  sendJson(res, 404, { error: { message: 'Not found' } });
});

server.listen(port, host, () => {
  console.log(`[joyai-native-http-session-proxy] listening=http://${host}:${port}`);
  console.log(`[joyai-native-http-session-proxy] protocol=${realtimeProtocol}`);
  console.log(`[joyai-native-http-session-proxy] model=${advertisedModel}`);
  console.log(`[joyai-native-http-session-proxy] system_prompt_transport=${systemPromptTransport}`);
  console.log(`[joyai-native-http-session-proxy] frame_clock_policy=${frameClockPolicy}`);
  console.log(`[joyai-native-http-session-proxy] audit=${auditPath}`);
});

async function shutdown(signal) {
  await appendAudit({ event: 'proxy_shutdown', signal }).catch(() => {});
  await serialize(() => resetConversation('proxy_shutdown')).catch(() => {});
  server.close(() => process.exit(0));
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.once(signal, () => { shutdown(signal).catch(() => process.exit(1)); });
}
