import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  loadRecordingCampaignConfig,
  normalizeRecordingWebUrl,
} from '../scripts/recording_config.mjs';
import { loadVlmRegistry } from '../scripts/vlm_profiles.mjs';

function campaignConfig(tlsRejectUnauthorized) {
  const webui = {
    url_env: 'SVI_TEST_WEB_URL_UNSET',
    url: 'https://example.invalid:7099/',
    input_mode: 'upload',
    expected_title: 'JoyAI-VL-Interaction',
    identity_markers: ['videoFileInput'],
    username_env: 'JOYVL_WEB_USERNAME',
    password_env: 'JOYVL_WEB_PASSWORD',
  };
  if (tlsRejectUnauthorized !== undefined) {
    webui.tls_reject_unauthorized = tlsRejectUnauthorized;
  }
  return {
    version: 2,
    webui,
    network: { mode: 'direct' },
    output_root: 'outputs/test',
  };
}

test('recording TLS verification defaults to enabled and requires an explicit opt-out', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'svi-config-'));
  const safePath = path.join(directory, 'safe.json');
  const unsafePath = path.join(directory, 'unsafe.json');
  fs.writeFileSync(safePath, JSON.stringify(campaignConfig(undefined)));
  fs.writeFileSync(unsafePath, JSON.stringify(campaignConfig(false)));

  assert.equal(loadRecordingCampaignConfig(safePath).webui.tlsRejectUnauthorized, true);
  assert.equal(loadRecordingCampaignConfig(unsafePath).webui.tlsRejectUnauthorized, false);
});

test('recording and model URLs reject embedded credentials', () => {
  assert.throws(
    () => normalizeRecordingWebUrl('https://user:password@example.invalid:7099/'),
    /must not embed credentials/,
  );

  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'svi-profile-'));
  const configPath = path.join(directory, 'vlm_models.json');
  fs.writeFileSync(configPath, JSON.stringify({
    version: 1,
    profiles: [{
      id: 'unsafe-profile',
      model: 'example-model',
      api_base: 'https://user:password@example.invalid/v1',
      route: 'direct',
      input_transport: 'image-frame-batch',
      enabled: true,
    }],
  }));
  assert.throws(() => loadVlmRegistry(configPath), /must not embed credentials/);
});
