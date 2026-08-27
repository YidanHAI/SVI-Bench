import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (ROOT / "scripts" / "capture.mjs").read_text(encoding="utf-8")
CAMPAIGN = (ROOT / "scripts" / "run_recording_campaign.mjs").read_text(
    encoding="utf-8"
)
SCAFFOLD = (ROOT / "scripts" / "joyai_scaffold_adapter.py").read_text(
    encoding="utf-8"
)
NATIVE_PROXY = (ROOT / "scripts" / "native_video_realtime_proxy.mjs").read_text(
    encoding="utf-8"
)
MAGE_PROXY = (ROOT / "scripts" / "joyai_native_http_session_proxy.mjs").read_text(
    encoding="utf-8"
)
SUPERVISOR = (ROOT / "scripts" / "supervise_scaffold_recording.mjs").read_text(
    encoding="utf-8"
)


def load_campaign():
    return json.loads(
        (ROOT / "config" / "recording_campaign.json").read_text(encoding="utf-8")
    )


def load_profiles():
    registry = json.loads(
        (ROOT / "config" / "vlm_models.json").read_text(encoding="utf-8")
    )
    return {item["id"]: item for item in registry["profiles"]}


def test_campaign_is_the_five_model_mp4_upload_protocol():
    config = load_campaign()
    profiles = load_profiles()

    assert config["webui"]["url_env"] == "JOYVL_WEB_URL"
    assert "url" not in config["webui"]
    assert config["webui"]["input_mode"] == "upload"
    assert config["network"] == {"mode": "direct"}
    assert config["evaluation_protocol"] == (
        "joyai-formal-query-once-frame-stream-v1"
    )
    assert config["frame_scheduler"] == {
        "max_in_flight": 1,
        "busy_policy": "skip",
        "queue_capacity": 0,
        "process_interval_s": 1,
        "frames_per_batch": 1,
    }
    assert config["defaults"]["max_parallel_models"] == 5

    models = config["models"]
    assert [item["id"] for item in models] == [
        "joyai-vl-interaction",
        "doubao-seed-2.1-pro",
        "mage-vl",
        "moss-vl-realtime",
        "minicpmo-4.5-9b-native-video-v2",
    ]
    assert all(item["enabled"] for item in models)
    assert [item["runner"] for item in models] == [
        "adapter",
        "joyai_scaffold",
        "prompt_proxy",
        "prompt_proxy",
        "prompt_proxy",
    ]

    for profile in profiles.values():
        assert "api_base" not in profile
        assert profile["api_base_env"]
        assert profile["api_key_env"]

    native_ids = {
        "mage-vl-native-realtime",
        "moss-vl-realtime-native",
        "minicpm-o-4.5-9b-realtime-v2",
    }
    assert {
        profile_id
        for profile_id, profile in profiles.items()
        if profile["input_transport"] == "native-video-realtime"
    } == native_ids


def test_query_is_delivered_once_and_bound_to_the_displayed_frame():
    assert "EVALUATION_PROTOCOL = \"joyai-formal-query-once-frame-stream-v1\"" in SCAFFOLD
    assert "query_delivery\": \"once_per_round\"" in SCAFFOLD
    assert "query_delivery: 'once_per_round'" in NATIVE_PROXY
    assert "query_delivery: 'once_per_round'" in MAGE_PROXY
    assert "captureDisplayedQueryFrameAtDispatch" in CAPTURE
    assert "Refusing to deliver Query ${round.id} more than once" in CAPTURE
    assert "policy: 'once_per_round_no_replay'" in CAPTURE
    assert "provider_query_submission_started" in SUPERVISOR
    assert "webui-query-time-origin-with-fifo-provider-submission" in SUPERVISOR
    assert "provider_output_query_sha256" in CAPTURE


def test_shared_foreground_scheduler_and_timeouts_are_enforced():
    config = load_campaign()
    assert config["upstream_request_policy"] == {
        "sdk_max_retries": 0,
        "non_query_timeout_s": 30,
        "query_timeout_s": 180,
        "warmup_timeout_s": 180,
    }
    assert "frame_scheduler.busy_policy must be skip" in CAMPAIGN
    assert "--process-interval-s', String(args.processIntervalS)" in SUPERVISOR
    assert "--frames-per-batch', String(args.framesPerBatch)" in SUPERVISOR
    assert "frame scheduler was not the required serial" in SUPERVISOR
    assert "--upstream-sdk-max-retries" in CAMPAIGN
    assert "--non-query-request-timeout-s" in CAMPAIGN
    assert "--query-request-timeout-s" in CAMPAIGN


def test_mage_uses_system_role_and_interactive_after_query():
    config = load_campaign()
    model = next(item for item in config["models"] if item["id"] == "mage-vl")
    assert model["native_system_prompt_transport"] == "system-role"
    assert model["native_streaming_mode_policy"] == "interactive-after-query"
    assert "providerMessages.push({ role: 'system', content: systemPrompt })" in MAGE_PROXY
    image = "const content = [{ type: 'image_url', image_url: { url: frame } }]"
    text = "if (providerQuery) content.push({ type: 'text', text: providerQuery })"
    assert MAGE_PROXY.index(image) < MAGE_PROXY.index(text)
    assert "function normalizeProviderOutput" in MAGE_PROXY


def test_complete_mp4_upload_is_verified_before_single_pass_playback():
    assert "reportedUploadBytes !== targetUploadStat.size" in CAPTURE
    assert "complete_file_uploaded: true" in CAPTURE
    assert "complete_file_upload: targetUploadVerification" in CAPTURE
    warmup = CAPTURE.index(
        "performVlmInferenceWarmup('API readiness before target-video upload')"
    )
    target_source = CAPTURE.index(
        "const sourcePath = path.resolve(task.local_video_path)", warmup
    )
    flow_call = CAPTURE.index("streamReadyMs = await startUploadedAndWaitLoopingVideo()")
    flow_start = CAPTURE.index("const startUploadedAndWaitLoopingVideo = async () =>")
    upload = CAPTURE.index("const uploadInfo = await uploadSelectedVideoFromPage(", flow_start)
    playback = CAPTURE.index("await startUploadedVideoFromPage(", upload)
    assert warmup < target_source < flow_call
    assert flow_start < upload < playback
