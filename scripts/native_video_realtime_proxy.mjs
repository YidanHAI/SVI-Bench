#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import WebSocket from 'ws';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const host = process.env.PROMPT_PROXY_HOST || '127.0.0.1';
const port = Number(process.env.PROMPT_PROXY_PORT || 18070);
const upstreamKey = String(process.env.UPSTREAM_API_KEY || '');
const accessToken = String(process.env.PROMPT_PROXY_ACCESS_TOKEN || '');
const advertisedModel = String(process.env.PROMPT_PROXY_ADVERTISED_MODEL || '').trim();
const upstreamProtocol = String(
  process.env.PROMPT_PROXY_UPSTREAM_PROTOCOL || 'openai-chat',
).trim();
const realtimeUrl = String(process.env.NATIVE_REALTIME_API_BASE || '').trim();
const realtimeProtocol = String(process.env.NATIVE_REALTIME_PROTOCOL || '').trim();
const realtimeQueryMode = String(process.env.NATIVE_REALTIME_QUERY_MODE || '').trim();
const queryAudioManifestPath = String(
  process.env.NATIVE_QUERY_AUDIO_MANIFEST || '',
).trim();
const queryAudioRealtimePacing = process.env.NATIVE_QUERY_AUDIO_REALTIME_PACING !== '0';
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
const maxSessionS = Number(process.env.NATIVE_REALTIME_MAX_SESSION_S || 300);
const connectTimeoutMs = Number(process.env.NATIVE_REALTIME_CONNECT_TIMEOUT_MS || 30000);
const providerReconnectSettleMs = Number(
  process.env.NATIVE_REALTIME_PROVIDER_RECONNECT_SETTLE_MS || 1000,
);
const outputGraceMs = Number(process.env.NATIVE_REALTIME_OUTPUT_GRACE_MS || 40);
const maxBufferedFrames = Number(process.env.NATIVE_REALTIME_MAX_BUFFERED_FRAMES || 120);
const insecureTls = process.env.NATIVE_REALTIME_INSECURE_TLS === '1';
const maxBodyBytes = Number(process.env.PROMPT_PROXY_MAX_BODY_BYTES || 32 * 1024 * 1024);
const promptPath = path.resolve(
  process.env.JOYAI_SYSTEM_PROMPT_FILE
    || path.join(ROOT, 'config', 'joyai_system_prompt.txt'),
);
const auditPath = path.resolve(
  process.env.PROMPT_PROXY_AUDIT_PATH
    || path.join(ROOT, 'outputs', 'native_video_realtime_proxy_audit.jsonl'),
);

if (!Number.isInteger(port) || port <= 0 || port > 65535) throw new Error('Invalid PROMPT_PROXY_PORT');
if (!upstreamKey) throw new Error('UPSTREAM_API_KEY is required');
if (accessToken.length < 24) throw new Error('PROMPT_PROXY_ACCESS_TOKEN must contain at least 24 characters');
if (!advertisedModel) throw new Error('PROMPT_PROXY_ADVERTISED_MODEL is required');
if (upstreamProtocol !== 'openai-chat') {
  throw new Error('ModelBest realtime proxy requires PROMPT_PROXY_UPSTREAM_PROTOCOL=openai-chat');
}
if (realtimeProtocol !== 'modelbest-video-full-duplex-v1') {
  throw new Error(`Unsupported NATIVE_REALTIME_PROTOCOL: ${realtimeProtocol || '(missing)'}`);
}
if (!['session-instruction', 'input-audio-once'].includes(realtimeQueryMode)) {
  throw new Error(
    `Protocol ${realtimeProtocol} requires NATIVE_REALTIME_QUERY_MODE=`
    + 'session-instruction or input-audio-once',
  );
}
if (realtimeQueryMode === 'input-audio-once' && !queryAudioManifestPath) {
  throw new Error('NATIVE_QUERY_AUDIO_MANIFEST is required for input-audio-once');
}
if (realtimeQueryMode !== 'input-audio-once' && queryAudioManifestPath) {
  throw new Error('NATIVE_QUERY_AUDIO_MANIFEST is only valid for input-audio-once');
}
if (nativeVideoSchema !== 'modelbest-realtime.input.append.video_frames.jpeg') {
  throw new Error(`Unsupported NATIVE_VIDEO_SCHEMA: ${nativeVideoSchema || '(missing)'}`);
}
if (streamingModePolicy !== 'proactive') {
  throw new Error('ModelBest realtime requires NATIVE_STREAMING_MODE_POLICY=proactive');
}
if (systemPromptTransport !== 'system-role') {
  throw new Error('ModelBest realtime requires NATIVE_HTTP_SYSTEM_PROMPT_TRANSPORT=system-role');
}
if (frameClockPolicy !== 'inbound-turn') {
  throw new Error('ModelBest realtime requires NATIVE_HTTP_FRAME_CLOCK=inbound-turn');
}
let parsedRealtimeUrl;
try {
  parsedRealtimeUrl = new URL(realtimeUrl);
} catch {
  throw new Error(`Invalid NATIVE_REALTIME_API_BASE: ${realtimeUrl || '(missing)'}`);
}
if (!['ws:', 'wss:'].includes(parsedRealtimeUrl.protocol)) {
  throw new Error('NATIVE_REALTIME_API_BASE must use ws or wss');
}
for (const [name, value] of Object.entries({
  NATIVE_REALTIME_MAX_SESSION_S: maxSessionS,
  NATIVE_REALTIME_CONNECT_TIMEOUT_MS: connectTimeoutMs,
  NATIVE_REALTIME_OUTPUT_GRACE_MS: outputGraceMs,
  NATIVE_REALTIME_MAX_BUFFERED_FRAMES: maxBufferedFrames,
  PROMPT_PROXY_MAX_BODY_BYTES: maxBodyBytes,
})) {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be positive`);
}
if (
  !Number.isFinite(providerReconnectSettleMs)
  || providerReconnectSettleMs < 0
  || providerReconnectSettleMs > 10000
) {
  throw new Error('NATIVE_REALTIME_PROVIDER_RECONNECT_SETTLE_MS must be between 0 and 10000');
}

const systemPromptFileBytes = fs.readFileSync(promptPath);
const systemPromptFileSha256 = crypto.createHash('sha256').update(systemPromptFileBytes).digest('hex');
const systemPrompt = systemPromptFileBytes.toString('utf8').trim();
const systemPromptSha256 = crypto.createHash('sha256').update(systemPrompt).digest('hex');
if (!systemPrompt) throw new Error(`System prompt is empty: ${promptPath}`);
await fsp.mkdir(path.dirname(auditPath), { recursive: true });

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sha256 = (value) => crypto.createHash('sha256').update(value).digest('hex');
const silencePcmBase64 = Buffer.alloc(16000 * 4).toString('base64');
let auditTail = Promise.resolve();

function loadQueryAudioManifest(manifestPath) {
  const resolvedPath = path.resolve(manifestPath);
  const raw = fs.readFileSync(resolvedPath);
  const parsed = JSON.parse(raw.toString('utf8'));
  if (parsed?.version !== 1 || !parsed.entries || typeof parsed.entries !== 'object') {
    throw new Error(`Invalid Query audio manifest: ${resolvedPath}`);
  }
  const audio = parsed.audio || {};
  if (
    audio.sample_rate_hz !== 16000
    || audio.channels !== 1
    || audio.sample_format !== 'f32le'
    || audio.chunk_duration_ms !== 1000
  ) {
    throw new Error(
      'Query audio manifest must use 16 kHz mono f32le PCM with 1000 ms chunks',
    );
  }
  const entries = new Map();
  for (const [queryHash, entry] of Object.entries(parsed.entries)) {
    if (!/^[a-f0-9]{64}$/.test(queryHash)) {
      throw new Error(`Invalid Query hash in audio manifest: ${queryHash}`);
    }
    const query = String(entry?.query || '').trim();
    if (!query || sha256(query) !== queryHash || entry?.query_sha256 !== queryHash) {
      throw new Error(`Query text/hash mismatch in audio manifest: ${queryHash}`);
    }
    const pcmPath = path.resolve(path.dirname(resolvedPath), String(entry?.pcm_path || ''));
    const stat = fs.statSync(pcmPath);
    const byteLength = Number(entry?.byte_length);
    const sampleCount = Number(entry?.sample_count);
    if (
      !stat.isFile()
      || stat.size <= 0
      || stat.size % 4 !== 0
      || stat.size !== byteLength
      || sampleCount !== stat.size / 4
    ) {
      throw new Error(`Invalid Query PCM metadata: ${pcmPath}`);
    }
    entries.set(queryHash, {
      query,
      pcmPath,
      pcmSha256: String(entry?.pcm_sha256 || ''),
      byteLength,
      sampleCount,
      durationS: sampleCount / 16000,
    });
  }
  return {
    path: resolvedPath,
    sha256: sha256(raw),
    tts: parsed.tts || {},
    entries,
  };
}

const queryAudioManifest = realtimeQueryMode === 'input-audio-once'
  ? loadQueryAudioManifest(queryAudioManifestPath)
  : null;
const queryAudioCache = new Map();

function queryAudioFor(query) {
  const queryHash = sha256(query);
  const entry = queryAudioManifest?.entries.get(queryHash);
  if (!entry) {
    throw new Error(`Query audio manifest has no entry for Query hash ${queryHash}`);
  }
  if (!queryAudioCache.has(queryHash)) {
    const pcm = fs.readFileSync(entry.pcmPath);
    if (pcm.length !== entry.byteLength || sha256(pcm) !== entry.pcmSha256) {
      throw new Error(`Query PCM checksum mismatch: ${entry.pcmPath}`);
    }
    queryAudioCache.set(queryHash, pcm);
  }
  return { ...entry, queryHash, pcm: queryAudioCache.get(queryHash) };
}

function appendAudit(record) {
  const line = `${JSON.stringify({ timestamp: new Date().toISOString(), ...record })}\n`;
  auditTail = auditTail.catch(() => {}).then(() => fsp.appendFile(auditPath, line, 'utf8'));
  return auditTail;
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

function requestUserText(messages) {
  return messages
    .filter((message) => message?.role === 'user')
    .map((message) => contentText(message.content))
    .filter(Boolean)
    .join('\n');
}

function requestVideoFrames(messages) {
  const frames = [];
  for (const message of messages) {
    if (!Array.isArray(message?.content)) continue;
    for (const item of message.content) {
      if (!['image_url', 'input_image'].includes(item?.type)) continue;
      const imageUrl = typeof item.image_url === 'string'
        ? item.image_url
        : String(item.image_url?.url || item.image_url || '');
      const match = imageUrl.match(/^data:image\/(jpeg|jpg);base64,([A-Za-z0-9+/=\r\n]+)$/i);
      if (!match) {
        throw new Error(
          'Native realtime video input requires the WebUI analyzer JPEG data URL; '
          + 'generic image or MP4 batch fallback is forbidden',
        );
      }
      frames.push(match[2].replace(/\s+/g, ''));
    }
  }
  return frames;
}

function openAiResponse({ requestId, content, providerOutput, outputQuerySha256 }) {
  return {
    id: `chatcmpl-${requestId}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: advertisedModel,
    choices: [{
      index: 0,
      message: { role: 'assistant', content },
      finish_reason: 'stop',
    }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    native_realtime: {
      protocol: realtimeProtocol,
      video_schema: nativeVideoSchema,
      provider_output: providerOutput,
      provider_output_query_sha256: outputQuerySha256,
    },
  };
}

function adaptNativeProviderText(value) {
  const rawContent = String(value || '').trim();
  if (!rawContent) {
    return {
      content: '</silence>',
      rawContent: '',
      format: 'native-empty-to-silence',
    };
  }

  // ModelBest's realtime protocol already marks spoken output structurally with
  // kind=text followed by kind=listen. Translate that native boundary to the
  // common WebUI action marker without rewriting an explicit model action.
  if (/^<\/?(?:response|silence)>/i.test(rawContent)) {
    return {
      content: rawContent,
      rawContent,
      format: 'provider-action-token-preserved',
    };
  }
  return {
    content: `</response> ${rawContent}`,
    rawContent,
    format: 'native-text-listen-to-response-marker',
  };
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

class ModelBestVideoSession {
  constructor() {
    this.ws = null;
    this.sessionId = '';
    this.openedAtMs = 0;
    this.generation = 0;
    this.partialText = '';
    this.outputs = [];
    this.failure = null;
    this.activeQuerySha256 = '';
    this.activeQueryEventId = '';
    this.queryAudioStartedAtMs = null;
    this.queryAudioEndedAtMs = null;
    this.firstTextAtMs = null;
  }

  isOpen() {
    return this.ws?.readyState === WebSocket.OPEN && this.sessionId && !this.failure;
  }

  ageS() {
    return this.openedAtMs ? (Date.now() - this.openedAtMs) / 1000 : 0;
  }

  async close(reason = 'reset') {
    const ws = this.ws;
    const sessionId = this.sessionId;
    this.generation += 1;
    this.ws = null;
    this.sessionId = '';
    this.openedAtMs = 0;
    this.partialText = '';
    this.failure = null;
    this.activeQuerySha256 = '';
    this.activeQueryEventId = '';
    this.queryAudioStartedAtMs = null;
    this.queryAudioEndedAtMs = null;
    this.firstTextAtMs = null;
    let closeConfirmed = !ws || ws.readyState === WebSocket.CLOSED;
    let closeWaitMs = 0;
    if (ws && ws.readyState !== WebSocket.CLOSED) {
      const closeStartedAt = Date.now();
      const waitForClose = (timeoutMs) => new Promise((resolve) => {
        if (ws.readyState === WebSocket.CLOSED) {
          resolve(true);
          return;
        }
        let timer = null;
        const finish = (confirmed) => {
          if (timer) clearTimeout(timer);
          ws.off('close', onClose);
          resolve(confirmed);
        };
        const onClose = () => finish(true);
        ws.once('close', onClose);
        timer = setTimeout(
          () => finish(ws.readyState === WebSocket.CLOSED),
          timeoutMs,
        );
      });
      const closePromise = waitForClose(2000);
      if (ws.readyState === WebSocket.OPEN) {
        try { ws.send(JSON.stringify({ type: 'session.close', reason })); } catch {}
      }
      if ([WebSocket.CONNECTING, WebSocket.OPEN].includes(ws.readyState)) {
        try { ws.close(1000, reason.slice(0, 100)); } catch {}
      }
      closeConfirmed = await closePromise;
      if (!closeConfirmed && ws.readyState !== WebSocket.CLOSED) {
        const terminatePromise = waitForClose(500);
        try { ws.terminate(); } catch {}
        closeConfirmed = await terminatePromise;
      }
      closeWaitMs = Date.now() - closeStartedAt;
    }
    if (ws && providerReconnectSettleMs > 0) {
      await sleep(providerReconnectSettleMs);
    }
    if (sessionId) {
      await appendAudit({
        event: 'provider_session_closed',
        provider_session_id: sessionId,
        reason,
        close_confirmed: closeConfirmed,
        close_wait_ms: closeWaitMs,
        reconnect_settle_ms: ws ? providerReconnectSettleMs : 0,
      });
    }
  }

  async connect(instruction, querySha256 = '', requestId = '') {
    await this.close(querySha256 ? 'new_query_round' : 'new_idle_session');
    this.outputs = [];
    const generation = this.generation;
    if (querySha256) {
      await appendAudit({
        event: 'provider_query_submission_started',
        request_id: requestId,
        realtime_protocol: realtimeProtocol,
        input_transport: 'native-video-realtime',
        native_video_schema: nativeVideoSchema,
        query_transport: realtimeQueryMode,
        user_query_present: true,
        user_query_sha256: querySha256,
      });
    }
    const ws = new WebSocket(realtimeUrl, {
      headers: { authorization: `Bearer ${upstreamKey}` },
      handshakeTimeout: connectTimeoutMs,
      rejectUnauthorized: !insecureTls,
      maxPayload: maxBodyBytes,
    });
    this.ws = ws;
    this.failure = null;
    this.activeQuerySha256 = querySha256;

    await new Promise((resolve, reject) => {
      let initialized = false;
      let settled = false;
      const finish = (error = null) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        if (error) reject(error);
        else resolve();
      };
      const timeout = setTimeout(() => finish(new Error(
        `Timed out creating ${realtimeProtocol} session after ${connectTimeoutMs}ms`,
      )), connectTimeoutMs);
      const fail = (error) => {
        const normalized = error instanceof Error ? error : new Error(String(error));
        if (generation === this.generation) this.failure = normalized;
        finish(normalized);
      };

      ws.on('unexpected-response', (_request, response) => {
        response.resume();
        fail(new Error(
          `Realtime provider rejected WebSocket handshake with HTTP ${response.statusCode || 'unknown'}`,
        ));
      });
      ws.on('error', fail);
      ws.on('close', (code, reason) => {
        if (generation !== this.generation) return;
        const error = new Error(
          `Realtime provider socket closed: code=${code}, reason=${reason.toString()}`,
        );
        this.failure = error;
        if (!settled) finish(error);
        appendAudit({
          event: 'provider_socket_closed',
          provider_session_id: this.sessionId,
          code,
          reason: reason.toString(),
          active_query_sha256: this.activeQuerySha256,
        }).catch(() => {});
      });
      ws.on('message', (raw) => {
        if (generation !== this.generation) return;
        let message;
        try { message = JSON.parse(raw.toString()); } catch { return; }
        const type = String(message.type || '');
        if (['session.queue_done', 'queue_done'].includes(type) && !initialized) {
          initialized = true;
          ws.send(JSON.stringify({
            type: 'session.init',
            payload: { system_prompt: instruction },
          }));
          return;
        }
        if (type === 'session.created') {
          this.sessionId = String(message.session_id || '');
          this.openedAtMs = Date.now();
          if (!this.sessionId) {
            finish(new Error('Realtime provider returned session.created without session_id'));
            return;
          }
          appendAudit({
            event: 'provider_session_created',
            request_id: requestId,
            provider_session_id: this.sessionId,
            realtime_protocol: realtimeProtocol,
            input_transport: 'native-video-realtime',
            native_video_schema: nativeVideoSchema,
            query_delivery: 'once_per_round',
            query_transport: realtimeQueryMode,
            user_query_present: Boolean(querySha256),
            user_query_sha256: querySha256,
            system_prompt_file_sha256: systemPromptFileSha256,
            system_prompt_sha256: systemPromptSha256,
          }).catch(() => {});
          finish();
          return;
        }
        if (type === 'response.output.delta' && message.kind === 'text') {
          if (!this.partialText && String(message.text || '')) {
            this.firstTextAtMs = Date.now();
            appendAudit({
              event: 'provider_native_first_text',
              provider_session_id: this.sessionId,
              provider_response_id: String(message.response_id || ''),
              user_query_sha256: this.activeQuerySha256,
              query_frame_queue_event_id: this.activeQueryEventId,
              query_audio_start_to_first_text_ms: this.queryAudioStartedAtMs == null
                ? null
                : this.firstTextAtMs - this.queryAudioStartedAtMs,
              query_audio_end_to_first_text_ms: this.queryAudioEndedAtMs == null
                ? null
                : this.firstTextAtMs - this.queryAudioEndedAtMs,
            }).catch(() => {});
          }
          this.partialText += String(message.text || '');
          return;
        }
        if (type === 'response.output.delta' && message.kind === 'listen') {
          const text = this.partialText.trim();
          if (text) {
            const completedAtMs = Date.now();
            this.outputs.push({
              text,
              responseId: String(message.response_id || ''),
              querySha256: this.activeQuerySha256,
              queryEventId: this.activeQueryEventId,
              firstTextAtMs: this.firstTextAtMs,
              completedAtMs,
            });
            appendAudit({
              event: 'provider_native_response_completed',
              provider_session_id: this.sessionId,
              provider_response_id: String(message.response_id || ''),
              native_response_boundary: 'kind=listen',
              user_query_sha256: this.activeQuerySha256,
              query_frame_queue_event_id: this.activeQueryEventId,
              query_audio_start_to_completion_ms: this.queryAudioStartedAtMs == null
                ? null
                : completedAtMs - this.queryAudioStartedAtMs,
              query_audio_end_to_completion_ms: this.queryAudioEndedAtMs == null
                ? null
                : completedAtMs - this.queryAudioEndedAtMs,
            }).catch(() => {});
          }
          this.partialText = '';
          this.firstTextAtMs = null;
          return;
        }
        if (type === 'response.done') {
          const text = String(message.text || this.partialText).trim();
          if (text) {
            const completedAtMs = Date.now();
            this.outputs.push({
              text,
              responseId: String(message.response_id || ''),
              querySha256: this.activeQuerySha256,
              queryEventId: this.activeQueryEventId,
              firstTextAtMs: this.firstTextAtMs,
              completedAtMs,
            });
          }
          this.partialText = '';
          this.firstTextAtMs = null;
          return;
        }
        if (type === 'error') {
          const error = new Error(`Realtime provider error: ${JSON.stringify(message)}`);
          this.failure = error;
          finish(error);
        }
      });
    });
  }

  async sendInput({ audio, frames = [], forceListen = false }) {
    if (!this.isOpen()) throw this.failure || new Error('Realtime provider session is not open');
    if (this.ageS() >= maxSessionS - 2) {
      throw new Error(
        `Realtime provider session reached its ${maxSessionS}s limit; Query replay is forbidden`,
      );
    }
    await new Promise((resolve, reject) => {
      this.ws.send(JSON.stringify({
        type: 'input.append',
        input: {
          audio: Buffer.isBuffer(audio) ? audio.toString('base64') : String(audio || ''),
          video_frames: frames,
          force_listen: forceListen,
          max_slice_nums: 1,
        },
      }), (error) => (error ? reject(error) : resolve()));
    });
  }

  async sendFrames(frames, { forceListen = false } = {}) {
    for (const frame of frames) {
      await this.sendInput({
        audio: silencePcmBase64,
        frames: [frame],
        forceListen,
      });
    }
  }

  async sendQueryAudio(entry, frame, {
    requestId,
    queryEventId,
    queryFrameTimeRange,
    queryRoundIndex,
  }) {
    if (!this.isOpen()) throw this.failure || new Error('Realtime provider session is not open');
    this.activeQuerySha256 = entry.queryHash;
    this.activeQueryEventId = queryEventId;
    this.queryAudioStartedAtMs = Date.now();
    this.queryAudioEndedAtMs = null;
    this.firstTextAtMs = null;
    await appendAudit({
      event: 'provider_query_submission_started',
      request_id: requestId,
      provider_session_id: this.sessionId,
      realtime_protocol: realtimeProtocol,
      input_transport: 'native-video-realtime',
      native_video_schema: nativeVideoSchema,
      query_transport: realtimeQueryMode,
      query_round_index: queryRoundIndex,
      query_frame_queue_event_id: queryEventId,
      query_frame_time_range: queryFrameTimeRange,
      user_query_present: true,
      user_query_sha256: entry.queryHash,
      query_audio_manifest_sha256: queryAudioManifest.sha256,
      query_audio_pcm_sha256: entry.pcmSha256,
      query_audio_duration_s: entry.durationS,
    });

    const bytesPerSecond = 16000 * 4;
    let chunkCount = 0;
    for (let offset = 0; offset < entry.pcm.length; offset += bytesPerSecond) {
      const targetMs = this.queryAudioStartedAtMs + chunkCount * 1000;
      if (queryAudioRealtimePacing && Date.now() < targetMs) {
        await sleep(targetMs - Date.now());
      }
      const chunk = entry.pcm.subarray(offset, Math.min(entry.pcm.length, offset + bytesPerSecond));
      await this.sendInput({
        audio: chunk,
        frames: chunkCount === 0 ? [frame] : [],
        forceListen: false,
      });
      chunkCount += 1;
    }
    if (queryAudioRealtimePacing) {
      const semanticEndMs = this.queryAudioStartedAtMs + entry.durationS * 1000;
      if (Date.now() < semanticEndMs) await sleep(semanticEndMs - Date.now());
    }
    this.queryAudioEndedAtMs = Date.now();
    await appendAudit({
      event: 'provider_query_audio_completed',
      request_id: requestId,
      provider_session_id: this.sessionId,
      query_frame_queue_event_id: queryEventId,
      user_query_sha256: entry.queryHash,
      query_audio_chunk_count: chunkCount,
      query_audio_duration_s: entry.durationS,
      query_audio_wall_duration_ms: this.queryAudioEndedAtMs - this.queryAudioStartedAtMs,
      query_audio_realtime_pacing: queryAudioRealtimePacing,
    });
    return chunkCount;
  }

  takeOutput() {
    return this.outputs.shift() || null;
  }
}

const provider = new ModelBestVideoSession();
let bufferedFrames = [];
let queryRoundCount = 0;
let suspendedQueryAfterSessionLimit = null;
let sessionContinuityLossCount = 0;
const queryQueues = new Map();
const queryRecords = new Map();
const lastInboundQueries = new Map();

function requestSessionId(req, payload = {}) {
  return String(
    req.headers['x-streaming-session']
    || payload.session_id
    || payload.user
    || 'default',
  ).trim() || 'default';
}

function queueFor(sessionId) {
  if (!queryQueues.has(sessionId)) queryQueues.set(sessionId, []);
  return queryQueues.get(sessionId);
}

function idleInstruction() {
  return `${systemPrompt}\n\nNo user Query is active. Observe the video and remain silent.`;
}

function queryInstruction(query) {
  return [
    systemPrompt,
    '',
    'The following user Query is delivered exactly once for this realtime session.',
    'Continue observing subsequent video frames. Reply only when the Query requires a response.',
    `USER QUERY: ${query}`,
  ].join('\n');
}

async function ensureIdleSession() {
  if (provider.isOpen() && !provider.activeQuerySha256 && provider.ageS() < maxSessionS - 2) return;
  if (provider.activeQuerySha256) return;
  await provider.connect(idleInstruction());
}

async function ensureAudioSession(requestId = '', reason = 'frame') {
  if (provider.isOpen() && provider.ageS() < maxSessionS - 2) {
    return { rotated: false, continuityLost: false };
  }
  const previous = {
    provider_session_id: provider.sessionId,
    session_age_s: provider.ageS(),
    active_query_sha256: provider.activeQuerySha256,
  };
  const continuityLost = Boolean(previous.active_query_sha256);
  if (previous.provider_session_id) {
    if (continuityLost) sessionContinuityLossCount += 1;
    await appendAudit({
      event: 'provider_session_limit_reached',
      request_id: requestId,
      ...previous,
      max_realtime_session_s: maxSessionS,
      classification: 'model_capability',
      query_replayed: false,
      historical_frames_replayed: false,
      continuity_lost: continuityLost,
      renewal_reason: reason,
    });
  }
  await provider.connect(systemPrompt, '', requestId);
  return { rotated: Boolean(previous.provider_session_id), continuityLost };
}

async function activateQuery(query, querySha256, requestId, queryMetadata = {}) {
  suspendedQueryAfterSessionLimit = null;
  await provider.connect(queryInstruction(query), querySha256, requestId);
  queryRoundCount += 1;
  const replayFrames = bufferedFrames.slice(-maxBufferedFrames);
  if (replayFrames.length > 1) {
    await provider.sendFrames(replayFrames.slice(0, -1), { forceListen: true });
  }
  if (replayFrames.length) {
    await provider.sendFrames(replayFrames.slice(-1), { forceListen: false });
  }
  await appendAudit({
    event: 'provider_query_delivered',
    request_id: requestId,
    provider_session_id: provider.sessionId,
    provider_response_id: '',
    realtime_protocol: realtimeProtocol,
    input_transport: 'native-video-realtime',
    native_video_schema: nativeVideoSchema,
    query_delivery: 'once_per_round',
    query_transport: realtimeQueryMode,
    query_round_index: queryRoundCount,
    user_query_present: true,
    user_query_sha256: querySha256,
    ...queryMetadata,
    system_prompt_file_sha256: systemPromptFileSha256,
    system_prompt_sha256: systemPromptSha256,
  });
  return replayFrames.length;
}

async function activateAudioQuery(query, querySha256, frame, requestId, queryMetadata = {}) {
  suspendedQueryAfterSessionLimit = null;
  const session = await ensureAudioSession(requestId, 'query_audio');
  const providerSessionId = provider.sessionId;
  const audioEntry = queryAudioFor(query);
  if (audioEntry.queryHash !== querySha256) {
    throw new Error('Query audio hash does not match the accepted Query');
  }
  queryRoundCount += 1;
  const chunkCount = await provider.sendQueryAudio(audioEntry, frame, {
    requestId,
    queryEventId: queryMetadata.query_frame_queue_event_id || '',
    queryFrameTimeRange: queryMetadata.query_frame_time_range || '',
    queryRoundIndex: queryRoundCount,
  });
  await appendAudit({
    event: 'provider_query_delivered',
    request_id: requestId,
    provider_session_id: provider.sessionId,
    provider_session_id_before_query: providerSessionId,
    provider_session_reused_at_query: !session.rotated,
    provider_response_id: '',
    realtime_protocol: realtimeProtocol,
    input_transport: 'native-video-realtime',
    native_video_schema: nativeVideoSchema,
    query_delivery: 'once_per_round',
    query_transport: realtimeQueryMode,
    query_round_index: queryRoundCount,
    user_query_present: true,
    user_query_sha256: querySha256,
    query_audio_chunk_count: chunkCount,
    query_audio_duration_s: audioEntry.durationS,
    query_audio_manifest_sha256: queryAudioManifest.sha256,
    query_frame_count: 1,
    replayed_frame_count: 0,
    historical_frames_replayed: false,
    ...queryMetadata,
    system_prompt_file_sha256: systemPromptFileSha256,
    system_prompt_sha256: systemPromptSha256,
  });
  return { chunkCount, session };
}

async function resetConversation(reason = 'api_reset') {
  await provider.close(reason);
  bufferedFrames = [];
  queryRoundCount = 0;
  suspendedQueryAfterSessionLimit = null;
  sessionContinuityLossCount = 0;
  provider.outputs = [];
  queryQueues.clear();
  queryRecords.clear();
  lastInboundQueries.clear();
  await appendAudit({
    event: 'conversation_state_reset',
    input_transport: 'native-video-realtime',
    query_delivery: 'once_per_round',
    reason,
  });
  if (realtimeQueryMode === 'input-audio-once') {
    await provider.connect(systemPrompt, '', crypto.randomUUID());
    await appendAudit({
      event: 'provider_target_session_prepared',
      provider_session_id: provider.sessionId,
      realtime_protocol: realtimeProtocol,
      query_transport: realtimeQueryMode,
      system_prompt_sha256: systemPromptSha256,
      ready_before_target_upload: true,
    });
  }
  return { ok: true, reset_acknowledged: true };
}

async function handleQueryEvent(req, res) {
  const payload = JSON.parse((await readBody(req)).toString('utf8'));
  const sessionId = String(payload.session_id || '').trim();
  const eventId = String(payload.query_event_id || '').trim();
  const query = String(payload.query || '').trim();
  const frameTimeRange = String(payload.frame_time_range || '').trim();
  const imageUrl = String(payload.image_url || '');
  if (!sessionId || !eventId || !query || !frameTimeRange) {
    throw new Error('session_id, query_event_id, query, and frame_time_range are required');
  }
  const frames = requestVideoFrames([{
    role: 'user',
    content: [{ type: 'image_url', image_url: { url: imageUrl } }],
  }]);
  if (frames.length !== 1) throw new Error('Query event must contain exactly one JPEG frame');
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
    frame: frames[0],
    image_sha256: sha256(frames[0]),
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
  const frames = requestVideoFrames([{
    role: 'user',
    content: [{
      type: 'image_url',
      image_url: { url: String(payload.image_url || '') },
    }],
  }]);
  if (frames.length !== 1) throw new Error('Warmup requires exactly one JPEG frame');

  const warmupProvider = new ModelBestVideoSession();
  const requestId = crypto.randomUUID();
  const started = Date.now();
  let responseId = '';
  try {
    await warmupProvider.connect(
      realtimeQueryMode === 'input-audio-once' ? systemPrompt : idleInstruction(),
      '',
      requestId,
    );
    const sessionId = warmupProvider.sessionId;
    await warmupProvider.sendFrames(frames, { forceListen: true });
    if (outputGraceMs > 0) await sleep(outputGraceMs);
    const output = warmupProvider.takeOutput();
    responseId = output?.responseId || sessionId;
    if (!responseId) throw new Error('Realtime warmup returned no session or response id');
  } finally {
    await warmupProvider.close('isolated_provider_warmup').catch(() => {});
  }
  const latencyMs = Date.now() - started;
  await appendAudit({
    event: 'provider_warmup_completed',
    request_id: requestId,
    response_id: responseId,
    response_model: advertisedModel,
    ok: true,
    multimodal_input: true,
    session_state_unchanged: true,
    latency_ms: latencyMs,
  });
  sendJson(res, 200, {
    ok: true,
    model: advertisedModel,
    response_model: advertisedModel,
    response_id: responseId,
    latency_ms: latencyMs,
    multimodal_input: true,
    session_state_unchanged: true,
  });
}

async function handleChatCompletion(req, res) {
  const raw = await readBody(req);
  const payload = JSON.parse(raw.toString('utf8'));
  const messages = Array.isArray(payload.messages) ? payload.messages : [];
  const sessionId = requestSessionId(req, payload);
  const inboundQuery = requestUserText(messages);
  const queue = queueFor(sessionId);
  const queued = queue.shift() || null;
  let query = '';
  let queryEventId = '';
  let queryFrameTimeRange = '';
  let queryQueueDelayMs = null;
  let frames = requestVideoFrames(messages);
  if (queued) {
    queued.status = 'processing';
    query = queued.query;
    queryEventId = queued.query_event_id;
    queryFrameTimeRange = queued.frame_time_range;
    queryQueueDelayMs = Date.now() - queued.enqueued_epoch_ms;
    frames = [queued.frame];
  } else if (inboundQuery && inboundQuery !== lastInboundQueries.get(sessionId)) {
    query = inboundQuery;
  }
  lastInboundQueries.set(sessionId, inboundQuery);
  const querySha256 = query ? sha256(query) : '';
  const requestId = crypto.randomUUID();
  if (!frames.length) throw new Error('Native realtime request contains no WebUI video frame');
  if (realtimeQueryMode === 'session-instruction') {
    bufferedFrames.push(...frames);
    if (bufferedFrames.length > maxBufferedFrames) {
      bufferedFrames = bufferedFrames.slice(-maxBufferedFrames);
    }
  }

  let replayedFrameCount = 0;
  let queryAudioChunkCount = 0;
  let sessionLimitCapability = false;
  if (query) {
    try {
      const queryMetadata = {
        query_frame_time_range: queryFrameTimeRange,
        query_frame_queue_event_id: queryEventId,
      };
      if (realtimeQueryMode === 'input-audio-once') {
        const activation = await activateAudioQuery(
          query,
          querySha256,
          frames[0],
          requestId,
          queryMetadata,
        );
        queryAudioChunkCount = activation.chunkCount;
        sessionLimitCapability = activation.session.continuityLost;
      } else {
        replayedFrameCount = await activateQuery(
          query,
          querySha256,
          requestId,
          queryMetadata,
        );
      }
      if (queued) queued.status = 'delivered';
    } catch (error) {
      if (queued) {
        queued.status = 'queued';
        queue.unshift(queued);
      }
      throw error;
    }
  } else if (
    realtimeQueryMode === 'session-instruction'
    && suspendedQueryAfterSessionLimit
  ) {
    sessionLimitCapability = true;
  } else if (
    realtimeQueryMode === 'session-instruction'
    &&
    provider.activeQuerySha256
    && provider.ageS() >= maxSessionS - 2
  ) {
    const expiredQuerySha256 = provider.activeQuerySha256;
    const expiredSessionId = provider.sessionId;
    const expiredAgeS = provider.ageS();
    suspendedQueryAfterSessionLimit = {
      user_query_sha256: expiredQuerySha256,
      provider_session_id: expiredSessionId,
      session_age_s: expiredAgeS,
      max_realtime_session_s: maxSessionS,
      suspended_at: new Date().toISOString(),
    };
    await appendAudit({
      event: 'provider_session_limit_reached',
      request_id: requestId,
      ...suspendedQueryAfterSessionLimit,
      classification: 'model_capability',
      query_replayed: false,
    });
    await provider.close('provider_session_duration_limit');
    sessionLimitCapability = true;
  } else if (realtimeQueryMode === 'session-instruction') {
    await ensureIdleSession();
    await provider.sendFrames(frames, { forceListen: !provider.activeQuerySha256 });
  } else {
    const session = await ensureAudioSession(requestId, 'causal_frame');
    sessionLimitCapability = session.continuityLost;
    await provider.sendFrames(frames, { forceListen: !provider.activeQuerySha256 });
  }
  if (outputGraceMs > 0 && !provider.outputs.length) await sleep(outputGraceMs);
  const output = provider.takeOutput();
  const adaptedOutput = adaptNativeProviderText(output?.text || '');
  const content = adaptedOutput.content;
  const response = openAiResponse({
    requestId,
    content,
    providerOutput: Boolean(output),
    outputQuerySha256: output?.querySha256 || '',
  });
  response.native_realtime.session_limit_capability = sessionLimitCapability;
  response.native_realtime.query_audio_chunk_count = queryAudioChunkCount;
  response.native_realtime.query_event_id = queryEventId;
  response.native_realtime.provider_raw_content = adaptedOutput.rawContent;
  response.native_realtime.provider_output_format = adaptedOutput.format;
  if (queryEventId) {
    response.streamingharness = {
      session_id: sessionId,
      query_frame_queue: {
        policy: 'fifo-query-time-frame',
        query_event_id: queryEventId,
        user_query_sha256: querySha256,
        frame_time_range: queryFrameTimeRange,
        query_queue_delay_ms: queryQueueDelayMs,
      },
    };
  }

  await appendAudit({
    event: 'native_realtime_frame_forwarded',
    request_id: requestId,
    model: String(payload.model || advertisedModel),
    response_model: advertisedModel,
    provider_session_id: provider.sessionId,
    realtime_protocol: realtimeProtocol,
    input_transport: 'native-video-realtime',
    native_video_schema: nativeVideoSchema,
    frame_count: frames.length,
    replayed_frame_count: replayedFrameCount,
    historical_frames_replayed: realtimeQueryMode === 'input-audio-once' ? false : undefined,
    query_audio_chunk_count: queryAudioChunkCount,
    query_delivery: 'once_per_round',
    query_transport: realtimeQueryMode,
    query_round_index: query ? queryRoundCount : null,
    query_frame_queue_policy: queryEventId ? 'fifo-query-time-frame' : '',
    query_frame_queue_event_id: queryEventId,
    query_frame_time_range: query ? queryFrameTimeRange : '',
    query_frame_queue_delay_ms: queryQueueDelayMs,
    user_query_present: Boolean(query),
    user_query_sha256: querySha256,
    persistent_query_injected: false,
    provider_output: Boolean(output),
    provider_response_id: output?.responseId || '',
    provider_output_query_sha256: output?.querySha256 || '',
    session_limit_capability: sessionLimitCapability,
    suspended_query_sha256: suspendedQueryAfterSessionLimit?.user_query_sha256 || '',
    session_continuity_loss_count: sessionContinuityLossCount,
    query_audio_manifest_sha256: queryAudioManifest?.sha256 || '',
    system_prompt_file_sha256: systemPromptFileSha256,
    system_prompt_sha256: systemPromptSha256,
  });
  await appendAudit({
    event: 'upstream_response_received',
    request_id: requestId,
    response_id: output?.responseId || response.id,
    response_model: advertisedModel,
    ok: true,
    has_assistant_message: true,
    provider_output: Boolean(output),
    provider_session_id: provider.sessionId,
    system_prompt_file_sha256: systemPromptFileSha256,
    system_prompt_sha256: systemPromptSha256,
  });

  if (payload.stream === true) {
    const body = Buffer.from(openAiSse(response));
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache',
      'content-length': String(body.length),
    });
    res.end(body);
  } else {
    sendJson(res, 200, response);
  }
}

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
      upstream_protocol: upstreamProtocol,
      realtime_protocol: realtimeProtocol,
      input_transport: 'native-video-realtime',
      native_video_schema: nativeVideoSchema,
      query_delivery: 'once_per_round',
      query_transport: realtimeQueryMode,
      max_realtime_session_s: maxSessionS,
      streaming_mode_policy: streamingModePolicy,
      system_prompt_transport: systemPromptTransport,
      frame_clock_policy: frameClockPolicy,
      session_limit_policy: realtimeQueryMode === 'input-audio-once'
        ? 'renew_base_session_without_query_or_frame_replay'
        : 'suspend_until_next_query_without_query_replay',
      provider_session_open: provider.isOpen(),
      provider_session_id: provider.sessionId,
      provider_target_session_prepared: realtimeQueryMode === 'input-audio-once'
        && provider.isOpen(),
      query_audio_manifest_path: queryAudioManifest?.path || '',
      query_audio_manifest_sha256: queryAudioManifest?.sha256 || '',
      query_audio_entry_count: queryAudioManifest?.entries.size || 0,
      query_audio_realtime_pacing: realtimeQueryMode === 'input-audio-once'
        ? queryAudioRealtimePacing
        : null,
      query_audio_tts: queryAudioManifest?.tts || null,
      queued_query_events: [...queryQueues.values()].reduce(
        (sum, queue) => sum + queue.length,
        0,
      ),
      prompt_source_path: promptPath,
      system_prompt_file_sha256: systemPromptFileSha256,
      system_prompt_sha256: systemPromptSha256,
      audit_path: auditPath,
    });
    return;
  }
  if (!authorized(req)) {
    sendJson(res, 401, { error: { message: 'Unauthorized native realtime proxy request' } });
    return;
  }
  if (req.method === 'GET' && requestPath === '/v1/models') {
    sendJson(res, 200, {
      object: 'list',
      data: [{ id: advertisedModel, object: 'model', owned_by: 'native-realtime-provider' }],
    });
    return;
  }
  if (req.method === 'POST' && ['/reset', '/v1/streaming/reset'].includes(requestPath)) {
    resetConversation(
      requestPath === '/reset' ? 'api_reset' : 'warmup_session_reset',
    ).then((result) => sendJson(res, 200, result)).catch((error) => {
      sendJson(res, 500, { error: { message: error.message } });
    });
    return;
  }
  if (req.method === 'POST' && requestPath === '/v1/query-events') {
    handleQueryEvent(req, res).catch((error) => {
      if (!res.headersSent) sendJson(res, 400, { error: { message: error.message } });
    });
    return;
  }
  if (req.method === 'POST' && requestPath === '/v1/warmup') {
    handleProviderWarmup(req, res).catch(async (error) => {
      await appendAudit({ event: 'provider_warmup_failed', error: error.message }).catch(() => {});
      if (!res.headersSent) sendJson(res, 502, { error: { message: error.message } });
    });
    return;
  }
  if (req.method === 'POST' && requestPath === '/v1/chat/completions') {
    handleChatCompletion(req, res).catch(async (error) => {
      await appendAudit({
        event: 'proxy_error',
        error: error.message,
        provider_session_id: provider.sessionId,
      }).catch(() => {});
      if (!res.headersSent) sendJson(res, 502, { error: { message: error.message } });
      else res.destroy(error);
    });
    return;
  }
  sendJson(res, 404, { error: { message: 'Not found' } });
});

server.listen(port, host, () => {
  console.log(`[native-video-realtime-proxy] listening=http://${host}:${port}`);
  console.log(`[native-video-realtime-proxy] protocol=${realtimeProtocol}`);
  console.log(`[native-video-realtime-proxy] model=${advertisedModel}`);
  console.log(`[native-video-realtime-proxy] audit=${auditPath}`);
});

async function shutdown(signal) {
  await appendAudit({ event: 'proxy_shutdown', signal }).catch(() => {});
  await provider.close('proxy_shutdown').catch(() => {});
  server.close(() => process.exit(0));
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.once(signal, () => { shutdown(signal).catch(() => process.exit(1)); });
}
