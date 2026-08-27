import crypto from 'node:crypto';

const SILENCE_ONLY_PATTERN = /^<?\/?sil[a-z0-9_-]*>?$/i;
const SILENCE_CONTROL_PATTERN = /<?\/?sil[a-z0-9_-]*>?/i;
const RESPONSE_PREFIX_PATTERN = /^<\/response>\s*/i;
const RESPONSE_CLOSER_PATTERN = /<\/response>/i;
const MALFORMED_ACTION_PREFIX_PATTERN = /^<(?:response|silence)>/i;
const ROLE_TRANSCRIPT_LINE_PATTERN = /(?:^|\n)\s*(?:user|assistant|system)\s*:/i;
const TIMESTAMP_ONLY_PATTERN = /^<\s*-?\d+(?:\.\d+)?\s*(?:seconds?|s)(?:\s*(?:~|-)\s*-?\d+(?:\.\d+)?\s*(?:seconds?|s))?\s*>$/i;
const WEBUI_PROVIDER_ERROR_PATTERN = /^Error:\s+/i;

function normalizedText(value) {
  return String(value || '').normalize('NFKC').replace(/\r\n?/g, '\n').trim();
}

function providerError(responsePayload) {
  if (!responsePayload || typeof responsePayload !== 'object') return null;
  if (responsePayload.error) return responsePayload.error;
  if (String(responsePayload.object || '').toLowerCase() === 'error') return responsePayload;
  return null;
}

function rawUpstreamText(responsePayload) {
  return normalizedText(responsePayload?.streamingharness?.raw_content || '');
}

function invalidProtocol(classification) {
  return {
    classification,
    substantive: false,
    protocol_valid: false,
    violation: classification,
  };
}

export function classifyVlmResponseText(value, responsePayload = null) {
  const text = normalizedText(value);
  if (providerError(responsePayload) || WEBUI_PROVIDER_ERROR_PATTERN.test(text)) {
    return {
      classification: 'provider_error',
      substantive: false,
      protocol_valid: false,
      violation: 'provider_error',
    };
  }
  if (!text) {
    return {
      classification: 'empty_output',
      substantive: false,
      protocol_valid: false,
      violation: 'empty_output',
    };
  }

  // The official adapter may normalize an invalid long completion down to a
  // plausible first line. Inspect its unmodified upstream text before accepting
  // the normalized value so role/transcript continuation cannot end a recording.
  const rawText = rawUpstreamText(responsePayload);
  const protocolBody = (rawText || text).replace(RESPONSE_PREFIX_PATTERN, '').trim();
  if (ROLE_TRANSCRIPT_LINE_PATTERN.test(protocolBody)) {
    return invalidProtocol('role_transcript_continuation');
  }
  if (
    rawText
    && rawText.toLowerCase() !== '</silence>'
    && !RESPONSE_PREFIX_PATTERN.test(rawText)
  ) {
    return invalidProtocol(
      MALFORMED_ACTION_PREFIX_PATTERN.test(rawText)
        ? 'malformed_action_marker'
        : 'missing_response_marker',
    );
  }

  if (text.toLowerCase() === '</silence>') {
    return {
      classification: 'silence',
      substantive: false,
      protocol_valid: true,
      violation: '',
    };
  }

  const hasResponsePrefix = RESPONSE_PREFIX_PATTERN.test(text);
  const responsePayloadText = hasResponsePrefix
    ? text.replace(RESPONSE_PREFIX_PATTERN, '').trim()
    : text;
  const hasUnexpectedResponseCloser = RESPONSE_CLOSER_PATTERN.test(responsePayloadText);
  const content = responsePayloadText.replace(/<\/response>\s*$/i, '').trim();

  if (!content) {
    return {
      classification: hasResponsePrefix ? 'empty_response' : 'empty_output',
      substantive: false,
      protocol_valid: false,
      violation: hasResponsePrefix ? 'empty_response' : 'empty_output',
    };
  }
  if (TIMESTAMP_ONLY_PATTERN.test(content)) {
    return invalidProtocol('timestamp_only_response');
  }
  if (SILENCE_ONLY_PATTERN.test(content.replace(/\s+/g, ''))) {
    const exactSilenceInsideResponse = content.toLowerCase() === '</silence>';
    return {
      classification: exactSilenceInsideResponse
        ? 'response_wrapped_silence'
        : 'malformed_silence',
      substantive: false,
      protocol_valid: false,
      violation: exactSilenceInsideResponse
        ? 'response_wrapped_silence'
        : 'malformed_silence',
    };
  }

  let violation = '';
  if (!hasResponsePrefix) violation = 'missing_response_marker';
  else if (SILENCE_CONTROL_PATTERN.test(content)) violation = 'mixed_silence_control_token';
  else if (hasUnexpectedResponseCloser) violation = 'unexpected_response_closer';

  if (violation) return invalidProtocol(violation);

  return {
    classification: 'substantive_response',
    substantive: true,
    protocol_valid: true,
    violation: '',
  };
}

export function isSubstantiveVlmText(value, responsePayload = null) {
  return classifyVlmResponseText(value, responsePayload).substantive;
}

export function isDeferredVlmText(value) {
  const text = normalizedText(value)
    .replace(/^<\/response>\s*/i, '')
    .replace(/<\/response>\s*$/i, '')
    .trim()
    .toLowerCase();
  if (!text) return false;
  if ([
    '需要调用后台模型',
    '后台处理',
    '后台模型',
    '请稍等',
    '稍后回复',
    '稍后为您',
    '</delegation>',
  ].some((marker) => text.includes(marker))) return true;
  return [
    /(?:我|这边)?(?:会|将|需要)?继续(?:留意|观察|等待|关注)/,
    /(?:警报|告警|提醒|监测|任务|规则)[^。！？]{0,24}(?:已设定|已设置|已开启|已启动|已准备)/,
    /(?:如果|若|一旦)[^。！？]{0,100}(?:我会|我将|会立即|将立即)[^。！？]{0,40}(?:提醒|通知|告警)/,
    /^(?:好的[，,!！\s]*)?(?:已收到|收到|明白)[^。！？]{0,80}(?:留意|观察|监测|提醒|通知)/,
    /^(?:好的[，,!！\s]*)?我(?:已准备|会|将)[^。！？]{0,80}(?:留意|观察|监测|提醒|通知)/,
    /等(?:待)?[^。！？]{0,40}(?:出现|展示|看到)[^。！？]{0,40}(?:后|之后)[^。！？]{0,20}(?:确认|回复|告诉)/,
    /\b(?:i(?:'ll| will)|let me)\s+(?:keep|continue)\s+(?:watching|monitoring|looking)\b/,
    /\b(?:i(?:'ll| will)|ready to)\s+(?:alert|notify|monitor|watch)\b/,
    /\b(?:if|when|once)\b[^.!?]{0,120}\b(?:i(?:'ll| will)|we(?:'ll| will))\b[^.!?]{0,50}\b(?:alert|notify|inform|tell)\b/,
  ].some((pattern) => pattern.test(text));
}

export function parseFrameTimeRangeS(value) {
  if (typeof value !== 'string') return null;
  const match = value.match(/-?\d+(?:\.\d+)?/);
  if (!match) return null;
  const parsed = Number(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
}

export function normalizeQuestion(value) {
  return String(value || '').normalize('NFKC').replace(/\s+/g, '').trim();
}

export function isQueryEchoVlmText(value, query) {
  const responseText = normalizedText(value)
    .replace(/^<\/response>\s*/i, '')
    .replace(/<\/response>\s*$/i, '')
    .trim();
  return Boolean(responseText)
    && normalizeQuestion(responseText) === normalizeQuestion(query);
}

function sha256(value) {
  return crypto.createHash('sha256').update(String(value || '')).digest('hex');
}

function responseQueryEventId(response) {
  return String(
    response?.response_payload?.streamingharness?.query_frame_queue?.query_event_id
    || response?.response_payload?.native_realtime?.query_event_id
    || '',
  );
}

function responseQuerySha256(response) {
  return String(
    response?.response_payload?.streamingharness?.query_frame_queue?.user_query_sha256
    || response?.response_payload?.native_realtime?.provider_output_query_sha256
    || '',
  );
}

export function responseBelongsToQuery(response, queryEvent, nextQueryEvent = null) {
  const expectedEventId = String(queryEvent?.query_frame_queue?.query_event_id || '');
  const explicitEventId = responseQueryEventId(response);
  if (explicitEventId && expectedEventId) return explicitEventId === expectedEventId;

  const explicitHash = responseQuerySha256(response);
  const expectedHash = sha256(queryEvent?.query || '');
  if (explicitHash) return explicitHash === expectedHash;

  const responsePrompt = normalizeQuestion(response?.metrics?.user_prompt);
  const expectedPrompt = normalizeQuestion(queryEvent?.query);
  if (responsePrompt) return responsePrompt === expectedPrompt;

  const queuedRange = response?.response_payload?.streamingharness
    ?.query_frame_queue?.frame_time_range;
  const frameTimeS = parseFrameTimeRangeS(
    queuedRange ?? response?.request_payload?.frame_time_range,
  );
  const queryTimeS = Number(queryEvent?.query_time_s);
  const nextQueryTimeS = Number(nextQueryEvent?.query_time_s);
  if (Number.isFinite(frameTimeS) && Number.isFinite(queryTimeS)) {
    return frameTimeS >= queryTimeS - 0.001
      && (!Number.isFinite(nextQueryTimeS) || frameTimeS < nextQueryTimeS - 0.001);
  }

  const responseMs = Number(response?.t_ms);
  const querySentMs = Number(queryEvent?.query_sent_offset_s) * 1000;
  const nextQuerySentMs = Number(nextQueryEvent?.query_sent_offset_s) * 1000;
  return Number.isFinite(responseMs)
    && Number.isFinite(querySentMs)
    && responseMs >= querySentMs
    && (!Number.isFinite(nextQuerySentMs) || responseMs < nextQuerySentMs);
}
