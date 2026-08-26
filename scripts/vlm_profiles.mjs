#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const DEFAULT_VLM_CONFIG = path.resolve(__dirname, '..', 'config', 'vlm_models.json');
const ROUTES = new Set(['joyai_adapter', 'direct']);
const INTERACTION_SCAFFOLDS = new Set(['', 'joyai-official-live-adapter']);
const INPUT_TRANSPORTS = new Set([
  'stateful-frame-stream',
  'image-frame-batch',
  'native-video-realtime',
  'native-video-batch',
]);
const REALTIME_PROTOCOLS = new Set([
  'modelbest-video-full-duplex-v1',
  'joyai-http-session-v1',
]);
const REALTIME_QUERY_MODES = new Set([
  'session-instruction',
  'input-audio-once',
  'text-event',
  'input-text-once',
]);

function requiredString(value, field, profileId = '') {
  const normalized = String(value ?? '').trim();
  if (!normalized) {
    throw new Error(`VLM profile ${profileId || '(unknown)'} is missing ${field}`);
  }
  return normalized;
}

function optionalEnvironmentName(value, field, profileId = '') {
  const normalized = String(value || '').trim();
  if (!normalized) return '';
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(normalized)) {
    throw new Error(`VLM profile ${profileId || '(unknown)'} has invalid ${field}`);
  }
  return normalized;
}

export function normalizeApiBase(value) {
  return String(value || '').trim().replace(/\/+$/, '');
}

function normalizeProfile(raw, index) {
  const id = requiredString(raw?.id, 'id', `#${index + 1}`);
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(id)) {
    throw new Error(`VLM profile id must be filesystem-safe: ${id}`);
  }
  const model = requiredString(raw.model, 'model', id);
  const apiBaseEnv = optionalEnvironmentName(raw.api_base_env, 'api_base_env', id);
  const apiBase = normalizeApiBase(requiredString(
    (apiBaseEnv && process.env[apiBaseEnv]) || raw.api_base,
    apiBaseEnv ? `api_base or environment ${apiBaseEnv}` : 'api_base',
    id,
  ));
  let parsedApiBase;
  try {
    parsedApiBase = new URL(apiBase);
  } catch {
    throw new Error(`VLM profile ${id} has invalid api_base: ${apiBase}`);
  }
  if (!['http:', 'https:'].includes(parsedApiBase.protocol)) {
    throw new Error(`VLM profile ${id} api_base must use http or https`);
  }

  const route = requiredString(raw.route, 'route', id);
  if (!ROUTES.has(route)) {
    throw new Error(`VLM profile ${id} route must be joyai_adapter or direct`);
  }
  const interactionScaffold = String(raw.interaction_scaffold || '').trim();
  if (!INTERACTION_SCAFFOLDS.has(interactionScaffold)) {
    throw new Error(
      `VLM profile ${id} has invalid interaction_scaffold: ${interactionScaffold}`,
    );
  }
  const formalEval = raw.formal_eval === true;
  if (
    formalEval
    && route !== 'joyai_adapter'
    && interactionScaffold !== 'joyai-official-live-adapter'
    && String(raw.input_transport || '').trim() !== 'native-video-realtime'
  ) {
    throw new Error(`VLM profile ${id} cannot be formal_eval=true with route=${route}`);
  }

  const backendAliases = [...new Set([
    model,
    ...(Array.isArray(raw.backend_aliases) ? raw.backend_aliases : []),
  ].map((item) => String(item || '').trim()).filter(Boolean))];
  const upstreamProtocol = String(raw.upstream_protocol || 'openai-chat').trim();
  if (upstreamProtocol !== 'openai-chat') {
    throw new Error(`VLM profile ${id} has invalid upstream_protocol: ${upstreamProtocol}`);
  }
  const inputTransport = String(
    raw.input_transport
      || (route === 'joyai_adapter' ? 'stateful-frame-stream' : 'image-frame-batch'),
  ).trim();
  if (!INPUT_TRANSPORTS.has(inputTransport)) {
    throw new Error(
      `VLM profile ${id} has invalid input_transport: ${inputTransport}; `
      + `expected one of ${[...INPUT_TRANSPORTS].join(', ')}`,
    );
  }
  if (interactionScaffold) {
    if (route !== 'direct') {
      throw new Error(
        `VLM profile ${id} interaction_scaffold requires route=direct for its upstream API`,
      );
    }
    if (inputTransport !== 'stateful-frame-stream') {
      throw new Error(
        `VLM profile ${id} interaction_scaffold requires input_transport=stateful-frame-stream`,
      );
    }
  } else if (route === 'direct' && inputTransport === 'stateful-frame-stream') {
    throw new Error(
      `VLM profile ${id} direct stateful-frame-stream requires interaction_scaffold`,
    );
  }
  const nativeVideoSchema = String(raw.native_video_schema || '').trim();
  const nativeVideo = inputTransport.startsWith('native-video-');
  const nativeVideoVerified = nativeVideo && raw.native_video_verified !== false;
  if (nativeVideo && !nativeVideoSchema) {
    throw new Error(`VLM profile ${id} requires native_video_schema for ${inputTransport}`);
  }
  if (!nativeVideo && nativeVideoSchema) {
    throw new Error(
      `VLM profile ${id} cannot declare native_video_schema with input_transport=${inputTransport}`,
    );
  }
  const realtimeProtocol = String(raw.realtime_protocol || '').trim();
  const realtimeApiBaseEnv = optionalEnvironmentName(
    raw.realtime_api_base_env, 'realtime_api_base_env', id,
  );
  const realtimeApiBase = String(
    (realtimeApiBaseEnv && process.env[realtimeApiBaseEnv])
      || raw.realtime_api_base
      || '',
  ).trim();
  const realtimeQueryMode = String(raw.realtime_query_mode || '').trim();
  const nativeRealtime = inputTransport === 'native-video-realtime';
  if (nativeRealtime) {
    if (!REALTIME_PROTOCOLS.has(realtimeProtocol)) {
      throw new Error(
        `VLM profile ${id} has unsupported realtime_protocol: ${realtimeProtocol || '(missing)'}`,
      );
    }
    if (!REALTIME_QUERY_MODES.has(realtimeQueryMode)) {
      throw new Error(
        `VLM profile ${id} has unsupported realtime_query_mode: ${realtimeQueryMode || '(missing)'}`,
      );
    }
    let parsedRealtimeApiBase;
    try {
      parsedRealtimeApiBase = new URL(realtimeApiBase);
    } catch {
      throw new Error(`VLM profile ${id} has invalid realtime_api_base: ${realtimeApiBase || '(missing)'}`);
    }
    const validRealtimeSchemes = realtimeProtocol === 'modelbest-video-full-duplex-v1'
      ? ['ws:', 'wss:']
      : ['http:', 'https:'];
    if (!validRealtimeSchemes.includes(parsedRealtimeApiBase.protocol)) {
      throw new Error(
        `VLM profile ${id} realtime_api_base must use ${validRealtimeSchemes.join(' or ')}`,
      );
    }
  } else if (realtimeProtocol || realtimeApiBase || realtimeQueryMode) {
    throw new Error(
      `VLM profile ${id} cannot declare realtime fields with input_transport=${inputTransport}`,
    );
  }
  const maxRealtimeSessionS = Number(raw.max_realtime_session_s || 0);
  if (nativeRealtime && (!Number.isFinite(maxRealtimeSessionS) || maxRealtimeSessionS <= 0)) {
    throw new Error(`VLM profile ${id} requires positive max_realtime_session_s`);
  }

  return {
    id,
    model,
    api_base: apiBase,
    api_base_env: apiBaseEnv,
    route,
    interaction_scaffold: interactionScaffold,
    formal_eval: formalEval,
    enabled: raw.enabled === true,
    api_key_env: String(raw.api_key_env || '').trim(),
    upstream_protocol: upstreamProtocol,
    input_transport: inputTransport,
    native_video_schema: nativeVideoSchema,
    native_video: nativeVideo,
    native_video_verified: nativeVideoVerified,
    realtime_protocol: realtimeProtocol,
    realtime_api_base: realtimeApiBase,
    realtime_api_base_env: realtimeApiBaseEnv,
    realtime_query_mode: realtimeQueryMode,
    max_realtime_session_s: nativeRealtime ? maxRealtimeSessionS : 0,
    realtime_interaction: ['stateful-frame-stream', 'native-video-realtime'].includes(inputTransport),
    backend_aliases: backendAliases,
    preflight: raw.preflight !== false,
    identity_check: raw.identity_check !== false,
    tags: [...new Set((Array.isArray(raw.tags) ? raw.tags : [])
      .map((item) => String(item || '').trim()).filter(Boolean))],
    description: String(raw.description || '').trim(),
  };
}

export function loadVlmRegistry(configPath = DEFAULT_VLM_CONFIG) {
  const resolvedPath = path.resolve(configPath || DEFAULT_VLM_CONFIG);
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(resolvedPath, 'utf8'));
  } catch (error) {
    throw new Error(`Unable to read VLM config ${resolvedPath}: ${error.message}`);
  }
  if (parsed?.version !== 1) {
    throw new Error(`Unsupported VLM config version in ${resolvedPath}; expected version 1`);
  }
  if (!Array.isArray(parsed.profiles) || !parsed.profiles.length) {
    throw new Error(`VLM config ${resolvedPath} must contain a non-empty profiles array`);
  }
  const profiles = parsed.profiles.map(normalizeProfile);
  const ids = new Set();
  for (const profile of profiles) {
    if (ids.has(profile.id)) throw new Error(`Duplicate VLM profile id: ${profile.id}`);
    ids.add(profile.id);
  }
  return { version: 1, path: resolvedPath, profiles };
}

export function getVlmProfile(registry, id) {
  const profile = registry.profiles.find((item) => item.id === id);
  if (!profile) {
    throw new Error(
      `Unknown VLM profile ${id}; available profiles: ${registry.profiles.map((item) => item.id).join(', ')}`,
    );
  }
  return profile;
}

export function profileFingerprint(profile) {
  const identity = {
    id: profile.id,
    model: profile.model,
    api_base: normalizeApiBase(profile.api_base),
    api_base_env: profile.api_base_env || '',
    route: profile.route,
    interaction_scaffold: profile.interaction_scaffold || '',
    formal_eval: profile.formal_eval,
    api_key_env: profile.api_key_env || '',
    upstream_protocol: profile.upstream_protocol || 'openai-chat',
    input_transport: profile.input_transport,
    native_video_schema: profile.native_video_schema || '',
    native_video_verified: profile.native_video_verified === true,
    realtime_protocol: profile.realtime_protocol || '',
    realtime_api_base: profile.realtime_api_base || '',
    realtime_api_base_env: profile.realtime_api_base_env || '',
    realtime_query_mode: profile.realtime_query_mode || '',
    max_realtime_session_s: profile.max_realtime_session_s || 0,
    backend_aliases: [...profile.backend_aliases].sort(),
    preflight: profile.preflight !== false,
    identity_check: profile.identity_check !== false,
  };
  return crypto.createHash('sha256').update(JSON.stringify(identity)).digest('hex');
}

export function safeProfileSnapshot(profile) {
  return {
    ...profile,
    fingerprint: profileFingerprint(profile),
  };
}

export function resolveProfileApiKey(profile, env = process.env, fallback = '') {
  if (profile.api_key_env && env[profile.api_key_env]) return env[profile.api_key_env];
  return fallback || env.JOYVL_VLM_API_KEY || '';
}

export function selectVlmProfiles(registry, selector = 'all') {
  const tokens = String(selector || 'all').split(',').map((item) => item.trim()).filter(Boolean);
  if (!tokens.length || tokens.includes('all')) {
    return registry.profiles.filter((profile) => profile.enabled);
  }
  const selected = [];
  const seen = new Set();
  for (const token of tokens) {
    const matches = token.startsWith('tag:')
      ? registry.profiles.filter((profile) => profile.tags.includes(token.slice(4)))
      : [getVlmProfile(registry, token)];
    if (!matches.length) throw new Error(`No VLM profiles matched ${token}`);
    for (const profile of matches) {
      if (!seen.has(profile.id)) {
        selected.push(profile);
        seen.add(profile.id);
      }
    }
  }
  return selected;
}

function printProfiles(registry, asJson = false) {
  if (asJson) {
    console.log(JSON.stringify({
      config: registry.path,
      profiles: registry.profiles.map(safeProfileSnapshot),
    }, null, 2));
    return;
  }
  const rows = registry.profiles.map((profile) => ({
    id: profile.id,
    enabled: profile.enabled ? 'yes' : 'no',
    formal: profile.formal_eval ? 'yes' : 'no',
    route: profile.route,
    model: profile.model,
  }));
  const headers = ['id', 'enabled', 'formal', 'route', 'model'];
  const widths = Object.fromEntries(headers.map((header) => [
    header,
    Math.max(header.length, ...rows.map((row) => row[header].length)),
  ]));
  console.log(`Config: ${registry.path}`);
  console.log(headers.map((header) => header.padEnd(widths[header])).join('  '));
  console.log(headers.map((header) => '-'.repeat(widths[header])).join('  '));
  for (const row of rows) {
    console.log(headers.map((header) => row[header].padEnd(widths[header])).join('  '));
  }
}

function cli() {
  let configPath = DEFAULT_VLM_CONFIG;
  let asJson = false;
  const argv = process.argv.slice(2);
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--config') {
      configPath = argv[++index] || '';
    } else if (arg === '--json') {
      asJson = true;
    } else if (arg === '--help' || arg === '-h') {
      console.log('Usage: node scripts/vlm_profiles.mjs [--config PATH] [--json]');
      return;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  printProfiles(loadVlmRegistry(configPath), asJson);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    cli();
  } catch (error) {
    console.error(error.stack || error.message || String(error));
    process.exit(1);
  }
}
