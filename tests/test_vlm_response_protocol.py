import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_URI = (ROOT / "scripts" / "vlm_response_protocol.mjs").as_uri()


def run_node(expression: str):
    script = f"""
import {{ classifyVlmResponseText, isDeferredVlmText, isQueryEchoVlmText, responseBelongsToQuery }} from {json.dumps(MODULE_URI)};
{expression}
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_response_protocol_classification():
    result = run_node(
        """
const samples = [
  '</silence>',
  '<silence>',
  'silence>',
  '</response>',
  '</response> <silence>',
  '</response> </silism>',
  '</response> 火灾已经发生，请立即撤离。',
  '</response> 火灾已经发生。 </silence>',
  '火灾已经发生，请立即撤离。',
  '</response> <29.0 seconds>',
  '</response> user: <29.0 seconds>',
];
const classified = samples.map((text) => classifyVlmResponseText(text));
classified.push(classifyVlmResponseText(
  '</response> 看起来正常',
  {streamingharness: {raw_content: '</response>\\nuser: <29.0 seconds>\\nassistant: </silence>'}},
));
classified.push(classifyVlmResponseText(
  '</response> <response> 火灾已经发生。',
  {streamingharness: {raw_content: '<response> 火灾已经发生。'}},
));
classified.push(classifyVlmResponseText(
  '</response> 火灾已经发生。',
  {streamingharness: {raw_content: '火灾已经发生。'}},
));
classified.push(classifyVlmResponseText('ignored', {error: {message: 'upstream failed'}}));
classified.push(classifyVlmResponseText(
  'Error: Error code: 500 - provider frame preprocessing failed',
));
console.log(JSON.stringify(classified));
"""
    )
    assert result[0] == {
        "classification": "silence",
        "substantive": False,
        "protocol_valid": True,
        "violation": "",
    }
    assert [item["classification"] for item in result[1:6]] == [
        "malformed_silence",
        "malformed_silence",
        "empty_response",
        "malformed_silence",
        "malformed_silence",
    ]
    assert all(item["substantive"] is False for item in result[1:6])
    assert result[6]["substantive"] is True
    assert result[6]["protocol_valid"] is True
    assert result[7] == {
        "classification": "mixed_silence_control_token",
        "substantive": False,
        "protocol_valid": False,
        "violation": "mixed_silence_control_token",
    }
    assert [item["classification"] for item in result[8:12]] == [
        "missing_response_marker",
        "timestamp_only_response",
        "role_transcript_continuation",
        "role_transcript_continuation",
    ]
    assert all(item["substantive"] is False for item in result[8:12])
    assert result[12]["classification"] == "malformed_action_marker"
    assert result[13]["classification"] == "missing_response_marker"
    assert result[14]["classification"] == "provider_error"
    assert result[14]["substantive"] is False
    assert result[15]["classification"] == "provider_error"
    assert result[15]["substantive"] is False


def test_multi_round_response_association_prefers_query_identity_then_frame_time():
    result = run_node(
        """
const r1 = {
  id: 'R1', query: 'first query', query_time_s: 1, query_sent_offset_s: 5,
  query_frame_queue: {query_event_id: 'T:R1'},
};
const r2 = {
  id: 'R2', query: 'second query', query_time_s: 20, query_sent_offset_s: 24,
  query_frame_queue: {query_event_id: 'T:R2'},
};
const delayedR1 = {
  t_ms: 30000,
  response_payload: {streamingharness: {query_frame_queue: {query_event_id: 'T:R1'}}},
};
const unpromptedR1Frame = {
  t_ms: 30000, metrics: {user_prompt: ''},
  request_payload: {frame_time_range: '12.0 seconds'},
};
const unpromptedR2Frame = {
  t_ms: 31000, metrics: {user_prompt: ''},
  request_payload: {frame_time_range: '22.0 seconds'},
};
console.log(JSON.stringify({
  delayed_r1_matches_r1: responseBelongsToQuery(delayedR1, r1, r2),
  delayed_r1_matches_r2: responseBelongsToQuery(delayedR1, r2, null),
  frame_12_matches_r1: responseBelongsToQuery(unpromptedR1Frame, r1, r2),
  frame_12_matches_r2: responseBelongsToQuery(unpromptedR1Frame, r2, null),
  frame_22_matches_r1: responseBelongsToQuery(unpromptedR2Frame, r1, r2),
  frame_22_matches_r2: responseBelongsToQuery(unpromptedR2Frame, r2, null),
}));
"""
    )
    assert result == {
        "delayed_r1_matches_r1": True,
        "delayed_r1_matches_r2": False,
        "frame_12_matches_r1": True,
        "frame_12_matches_r2": False,
        "frame_22_matches_r1": False,
        "frame_22_matches_r2": True,
    }


def test_monitoring_acknowledgement_and_background_delegation_are_not_final_answers():
    result = run_node(
        """
const samples = [
  '</response> 已收到指令，我已准备在火灾发生时立即提醒您。 </response>',
  '</response> 好的，我会留意宝宝的动向。',
  '</response> 这个问题需要调用后台模型，请稍等。',
  '</response> response> 火灾警报已设定。如果检测到火灾，我会立即提醒你。',
  '</response> I cannot see a fire now. If a fire appears, I will immediately inform you.',
  '</response> 紧急提醒：现场已经发生火灾，请立即撤离。',
];
console.log(JSON.stringify(samples.map(isDeferredVlmText)));
"""
    )
    assert result == [True, True, True, True, True, False]


def test_query_echo_is_not_a_completed_answer():
    result = run_node(
        """
console.log(JSON.stringify([
  isQueryEchoVlmText('</response> 当小狗从笼子里跳出来的时候，提醒我。', '当小狗从笼子里跳出来的时候，提醒我。'),
  isQueryEchoVlmText('</response> 小狗已经从笼子里跳出来了。', '当小狗从笼子里跳出来的时候，提醒我。'),
]));
"""
    )
    assert result == [True, False]


def test_capture_waits_for_substantive_non_deferred_response():
    capture = (ROOT / "scripts" / "capture.mjs").read_text(encoding="utf-8")
    supervisor = (ROOT / "scripts" / "supervise_scaffold_recording.mjs").read_text(
        encoding="utf-8"
    )
    protocol = "strict_protocol_non_deferred_non_echo_response_v4"
    assert f"MODEL_RESPONSE_OUTCOME_PROTOCOL = '{protocol}'" in capture
    assert f"MODEL_RESPONSE_OUTCOME_PROTOCOL = '{protocol}'" in supervisor
    assert "capture_missing_substantive_vlm_response" in capture
    assert "isQueryEchoVlmText" in capture
    assert "item.prompted_vlm_response_count === 0" not in capture
    assert "response_semantics: MODEL_RESPONSE_OUTCOME_PROTOCOL" in capture
    assert "timeout_is_recording_failure: false" in capture
    assert "capture used obsolete model-response completion semantics" in supervisor
