import json

import judge_campaign
import pipeline_campaign


def test_recording_snapshot_does_not_treat_empty_state_path_as_cwd(
    tmp_path, monkeypatch
):
    (tmp_path / "current.json").write_text(
        json.dumps({"status": "failed", "state_path": ""}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_campaign, "recording_output_root", lambda: tmp_path)

    snapshot = pipeline_campaign.recording_snapshot()

    assert snapshot["status"] == "failed"
    assert snapshot["state_path"] == ""
    assert snapshot["run_root"] == ""


def test_completed_judge_is_reentered_through_validating_launcher(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        pipeline_campaign, "STATE_PATH", tmp_path / "pipeline_state.json"
    )
    monkeypatch.setenv("JUDGE_OUTPUT_ROOT", str(tmp_path / "judge"))
    monkeypatch.setattr(
        pipeline_campaign,
        "judge_snapshot",
        lambda _root: {"status": "complete", "live": False},
    )
    monkeypatch.setattr(
        pipeline_campaign,
        "run_checked",
        lambda command, *, env=None: calls.append((command, env)),
    )

    snapshot, judge_root = pipeline_campaign.wait_for_judge(
        {"campaign_id": "run-1", "run_root": str(tmp_path / "recording")},
        lambda: False,
        1,
    )

    assert snapshot["status"] == "complete"
    assert judge_root == (tmp_path / "judge").resolve()
    assert len(calls) == 1
    assert calls[0][0][-1] == "start"
    assert calls[0][1]["JUDGE_RECORDING_RUN"] == str(tmp_path / "recording")


def test_pipeline_validate_checks_completed_recording_resume_contract(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setenv("JUDGE_OUTPUT_ROOT", str(tmp_path / "judge"))
    monkeypatch.setattr(
        pipeline_campaign,
        "recording_snapshot",
        lambda: {
            "status": "complete",
            "run_root": str(tmp_path / "recording"),
            "campaign_id": "run-1",
        },
    )
    monkeypatch.setattr(
        pipeline_campaign,
        "run_checked",
        lambda command, *, env=None: calls.append((command, env)),
    )

    assert pipeline_campaign.validate() == 0

    actions = [command[-1] for command, _env in calls]
    assert actions == ["dry-run", "--resume-current", "validate"]


def test_completed_pipeline_start_revalidates_before_reuse(tmp_path, monkeypatch):
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    monkeypatch.setattr(pipeline_campaign, "STATE_PATH", state_path)
    validations = []
    monkeypatch.setattr(
        pipeline_campaign,
        "validate",
        lambda *, emit=True: validations.append(emit) or 0,
    )

    assert pipeline_campaign.start(1) == 0
    assert validations == [False]


def test_completed_judge_campaign_revalidates_execution_identity(
    tmp_path, monkeypatch
):
    state_path = tmp_path / "campaign_state.json"
    index_path = tmp_path / "manifest_index.json"
    index_path.write_text("{}\n", encoding="utf-8")
    index_hash = judge_campaign.sha256(index_path)
    state_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest_index": {"sha256": index_hash},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(judge_campaign, "STATE_PATH", state_path)
    monkeypatch.setattr(judge_campaign, "MANIFEST_INDEX", index_path)
    monkeypatch.setattr(judge_campaign, "require_environment", lambda **_kwargs: "")
    monkeypatch.setattr(judge_campaign, "load_campaign_inputs", lambda: {"models": {}})
    validations = []
    monkeypatch.setattr(judge_campaign, "validate_all", lambda index: validations.append(index))
    monkeypatch.setattr(
        judge_campaign,
        "inspect_model",
        lambda _model_id: {"complete": True, "scores": {}, "result_summary": ""},
    )
    monkeypatch.setattr(judge_campaign, "write_leaderboard", lambda _states: None)

    assert judge_campaign.start(5, 5) == 0
    assert validations == [{"models": {}}]
