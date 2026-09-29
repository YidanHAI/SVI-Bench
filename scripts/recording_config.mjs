#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

export const DEFAULT_RECORDING_CAMPAIGN_CONFIG = path.resolve(
  ROOT,
  process.env.VL_INTERACTION_CAMPAIGN_CONFIG || 'config/recording_campaign.json',
);

function requiredString(value, field) {
  const result = String(value || '').trim();
  if (!result) throw new Error(`Recording campaign config is missing ${field}`);
  return result;
}

function environmentName(value, field) {
  const result = requiredString(value, field);
  if (!ENV_NAME.test(result)) {
    throw new Error(`Recording campaign ${field} is not a valid environment variable name`);
  }
  return result;
}

export function normalizeRecordingWebUrl(value) {
  let url;
  try {
    url = new URL(requiredString(value, 'webui.url'));
  } catch {
    throw new Error('Recording campaign webui.url is not a valid URL');
  }
  if (url.protocol !== 'https:') {
    throw new Error('Recording campaign webui.url must use HTTPS');
  }
  if (url.username || url.password) {
    throw new Error(
      'Recording campaign webui.url must not embed credentials; use the configured environment variables',
    );
  }
  url.pathname = `${url.pathname.replace(/\/+$/, '')}/`;
  url.search = '';
  url.hash = '';
  return url.toString();
}

export function loadRecordingCampaignConfig(
  configPath = DEFAULT_RECORDING_CAMPAIGN_CONFIG,
) {
  const resolvedPath = path.resolve(ROOT, configPath || DEFAULT_RECORDING_CAMPAIGN_CONFIG);
  let raw;
  try {
    raw = JSON.parse(fs.readFileSync(resolvedPath, 'utf8'));
  } catch (error) {
    throw new Error(`Unable to read recording campaign config ${resolvedPath}: ${error.message}`);
  }
  if (raw?.version !== 2) {
    throw new Error(`Unsupported recording campaign config version in ${resolvedPath}; expected 2`);
  }

  const webUrlEnv = environmentName(raw.webui?.url_env, 'webui.url_env');
  const webUrl = normalizeRecordingWebUrl(
    process.env[webUrlEnv] || raw.webui?.url,
  );
  const inputMode = requiredString(raw.webui?.input_mode, 'webui.input_mode');
  if (inputMode !== 'upload') {
    throw new Error('Recording campaign webui.input_mode must be upload');
  }
  const identityMarkers = (Array.isArray(raw.webui?.identity_markers)
    ? raw.webui.identity_markers
    : [])
    .map((item) => String(item || '').trim())
    .filter(Boolean);
  if (!identityMarkers.length) {
    throw new Error('Recording campaign webui.identity_markers must be non-empty');
  }
  const webui = {
    url: webUrl,
    urlEnv: webUrlEnv,
    inputMode,
    expectedTitle: requiredString(raw.webui?.expected_title, 'webui.expected_title'),
    identityMarkers,
    usernameEnv: environmentName(raw.webui?.username_env, 'webui.username_env'),
    passwordEnv: environmentName(raw.webui?.password_env, 'webui.password_env'),
    tlsRejectUnauthorized: raw.webui?.tls_reject_unauthorized !== false,
  };

  const networkMode = requiredString(raw.network?.mode, 'network.mode');
  if (!['direct', 'wireguard'].includes(networkMode)) {
    throw new Error('Recording campaign network.mode must be direct or wireguard');
  }
  const network = {
    mode: networkMode,
    interface: networkMode === 'wireguard'
      ? requiredString(raw.network?.interface, 'network.interface')
      : '',
    maxHandshakeAgeS: networkMode === 'wireguard'
      ? Number(raw.network?.max_handshake_age_s)
      : 0,
  };
  if (
    network.mode === 'wireguard'
    && (!Number.isInteger(network.maxHandshakeAgeS) || network.maxHandshakeAgeS < 1)
  ) {
    throw new Error('Recording campaign network.max_handshake_age_s must be a positive integer');
  }

  return {
    path: resolvedPath,
    raw,
    webui,
    network,
    outputRoot: path.resolve(ROOT, requiredString(raw.output_root, 'output_root')),
  };
}

export function recordingWebUrl(configPath = DEFAULT_RECORDING_CAMPAIGN_CONFIG) {
  return loadRecordingCampaignConfig(configPath).webui.url;
}
