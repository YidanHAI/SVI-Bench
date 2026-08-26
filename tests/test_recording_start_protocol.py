import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (ROOT / "scripts" / "capture.mjs").read_text(encoding="utf-8")
SUPERVISOR = (ROOT / "scripts" / "supervise_scaffold_recording.mjs").read_text(
    encoding="utf-8"
)
CAMPAIGN = (ROOT / "scripts" / "run_recording_campaign.mjs").read_text(
    encoding="utf-8"
)
VIDEO_QUALITY = (ROOT / "scripts" / "video_quality.mjs").read_text(encoding="utf-8")
CONFIG_LOADER = (ROOT / "scripts" / "recording_config.mjs").read_text(
    encoding="utf-8"
)


def test_api_readiness_precedes_target_video_upload():
    warmup = CAPTURE.index(
        "performVlmInferenceWarmup('API readiness before target-video upload')"
    )
    upload_flow_invocation = CAPTURE.index(
        "streamReadyMs = await startUploadedAndWaitLoopingVideo()"
    )
    upload_flow_start = CAPTURE.index(
        "const startUploadedAndWaitLoopingVideo = async () =>"
    )
    upload_flow_end = CAPTURE.index("\n  };\n\n  try {", upload_flow_start)
    upload_flow = CAPTURE[upload_flow_start:upload_flow_end]

    assert warmup < upload_flow_invocation
    assert "await prepareUploadedVideoSource(" in upload_flow
    assert "await uploadSelectedVideoFromPage(" in upload_flow
    assert upload_flow.index(
        "performVlmInferenceWarmup('same-session readiness refresh before target upload')"
    ) < upload_flow.index("const uploadStartedMs = monotonicMs()")
    assert "placeholder_stopped_before_target_upload: true" in CAPTURE
    assert "target_video_selected_or_uploaded: false" in CAPTURE
    assert "provider_warmup?.ok !== true" in SUPERVISOR


def test_single_pass_protocol_has_no_manual_seek_or_same_attempt_restart():
    assert "synchronizeUploadedPlaybackFromZero" not in CAPTURE
    assert re.search(r"video\.currentTime\s*=", CAPTURE) is None
    assert "Single-pass uploaded playback failed after stream start; discard this attempt" in CAPTURE
    assert "Refusing to restart an uploaded video in the same attempt" in CAPTURE


def test_success_requires_auditable_start_barrier():
    protocol = "webui_session_ready_then_single_pass_upload_v2"
    assert f"CURRENT_RECORDING_PROTOCOL = '{protocol}'" in CAPTURE
    assert f"CURRENT_RECORDING_PROTOCOL = '{protocol}'" in SUPERVISOR
    assert "recording_start_barrier?.protocol !== CURRENT_RECORDING_PROTOCOL" in SUPERVISOR
    assert "webui_same_session_model_ready !== true" in SUPERVISOR
    assert "seek_performed_after_stream_start !== false" in SUPERVISOR
    assert "target_video_started_once !== true" in SUPERVISOR


def test_query_frame_is_captured_before_the_first_frame_gate_is_released():
    query_capture = CAPTURE[
        CAPTURE.index("async function captureDisplayedQueryFrameAtDispatch"):
        CAPTURE.index("async function discardFirstQueryFrameGate")
    ]
    gate_release = CAPTURE[
        CAPTURE.index("async function releaseFirstQueryFrameGate"):
        CAPTURE.index("async function captureDisplayedQueryFrameAtDispatch")
    ]
    assert "context.drawImage(videoElement" in query_capture
    assert "canvas.toDataURL('image/jpeg', 0.9)" in query_capture
    assert "capture_surface: 'displayed_video_frame_at_query_dispatch'" in query_capture
    assert "const selected = gate.lockedPending || gate.pending" in gate_release
    assert "context.drawImage" not in gate_release


def test_resume_reuses_only_matching_validated_artifacts():
    assert "--resume-current" in CAMPAIGN
    assert "prepareCurrentResumePlan" in CAMPAIGN
    assert "sameJson(validation?.task_contract, task)" in SUPERVISOR
    assert "sameJson(validation?.source_video_snapshot, sourceVideoSnapshot(task))" in SUPERVISOR
    assert "compatibleEvaluationContract(validation?.evaluation_contract, evaluationContract)" in SUPERVISOR
    assert "failed_task_recovery_available" in CAMPAIGN
    assert "command.argv.push('--retry-failed-tasks')" in CAMPAIGN


def test_campaign_preflight_requires_each_selected_model_key():
    assert "else if (model.profile.api_key_env)" in CAMPAIGN
    assert "names.add(model.profile.api_key_env)" in CAMPAIGN
    assert "if (!launchReady) process.exitCode = 2" in CAMPAIGN


def test_outputs_are_synced_and_decode_is_retried():
    assert "await handle.sync()" in VIDEO_QUALITY
    assert "decodeAttempts = 3" in VIDEO_QUALITY
    assert "attempts_used: attempt" in VIDEO_QUALITY
    assert "if (attempt < attempts) await sleep(retryDelayMs)" in VIDEO_QUALITY


def test_non_response_is_retained_as_a_capability_outcome():
    policy = "wait_then_keep_as_capability_result"
    response_semantics = "strict_protocol_non_deferred_non_echo_response_v4"
    assert f"MODEL_RESPONSE_TIMEOUT_POLICY = '{policy}'" in CAPTURE
    assert f"MODEL_RESPONSE_TIMEOUT_POLICY = '{policy}'" in SUPERVISOR
    assert f"MODEL_RESPONSE_OUTCOME_PROTOCOL = '{response_semantics}'" in CAPTURE
    assert f"MODEL_RESPONSE_OUTCOME_PROTOCOL = '{response_semantics}'" in SUPERVISOR
    assert "timeout_is_recording_failure: false" in CAPTURE
    assert "model_latency_timeout" in SUPERVISOR


def test_webui_address_is_environment_only():
    config = json.loads(
        (ROOT / "config" / "recording_campaign.json").read_text(encoding="utf-8")
    )
    assert config["webui"]["url_env"] == "JOYVL_WEB_URL"
    assert "url" not in config["webui"]
    assert "process.env[webUrlEnv] || raw.webui?.url" in CONFIG_LOADER
    assert "RECORDING_WEB_URL" not in CONFIG_LOADER


def test_task_wall_budget_survives_worker_restart():
    assert "resolveTaskWallWindow" in SUPERVISOR
    assert "task_wall_pass_started_at" in SUPERVISOR
    assert "capture_task_wall_timeout" in SUPERVISOR
    assert "--max-task-wall-s" in CAMPAIGN
