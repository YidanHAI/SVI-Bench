import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from requests import Response


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import judge_runtime as pipeline
import run_judge_with_attempt_budget as budget


def test_gateway_key_is_redacted_from_error_response(monkeypatch):
    secret = "pk-unit-test-secret"
    monkeypatch.setenv("OPENAI_API_KEY", f"Bearer {secret}")
    response = Response()
    response.status_code = 401
    response._content = (
        f'{{"error":{{"cause":"api key {secret} invalid"}}}}'.encode()
    )

    budget.redact_response_secrets(response)

    assert secret not in response.text
    assert budget.SECRET_REDACTION in response.text


def test_authentication_failure_is_detected_without_exposing_secret():
    errors = {
        "T1": "RuntimeError('HTTP 401: API KEY无效')",
        "T2": "RuntimeError('HTTP 503: retry later')",
    }

    assert pipeline.authentication_failures(errors, ["T1", "T2"]) == ["T1"]


def write_manifest(path):
    path.write_text(
        json.dumps({
            "contains_human_labels": False,
            "tasks": [{"task_spec": {"id": "T1"}}],
        }),
        encoding="utf-8",
    )


def wrapper_argv(target, ledger, manifest):
    return [
        "run_judge_with_attempt_budget.py",
        "--target",
        str(target),
        "--stage",
        "first_pass",
        "--ledger",
        str(ledger),
        "--manifest",
        str(manifest),
        "--stage-identity-sha256",
        "a" * 64,
        "--max-total-attempts",
        "5",
    ]


def test_http_budget_is_shared_across_process_restarts(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.json"
    target = tmp_path / "target.py"
    write_manifest(manifest)
    target.write_text(
        """
import requests

payload = {
    "messages": [{
        "role": "user",
        "content": 'JUDGE_INPUT_JSON: {"task_spec": {"id": "T1"}}',
    }]
}
for _ in range(3):
    try:
        requests.post("https://judge.invalid/v1", json=payload)
    except BaseException:
        pass
""",
        encoding="utf-8",
    )

    actual_requests = []

    def fake_request(session, method, url, **kwargs):
        actual_requests.append((method, url))
        return SimpleNamespace(status_code=503)

    monkeypatch.setattr(budget.requests.sessions.Session, "request", fake_request)
    for _ in range(3):
        monkeypatch.setattr(sys, "argv", wrapper_argv(target, ledger, manifest))
        budget.main()

    assert len(actual_requests) == 5
    value = json.loads(ledger.read_text(encoding="utf-8"))
    budget.validate_ledger_document(value)
    assert value["tasks"]["T1"]["attempt_count"] == 5
    assert [
        item["attempt"] for item in value["tasks"]["T1"]["attempts"]
    ] == [1, 2, 3, 4, 5]


def test_extract_task_id_accepts_responses_input_payload():
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            'JUDGE_INPUT_JSON: '
                            '{"task_spec": {"id": "T1"}}'
                        ),
                    },
                    {"type": "input_image", "image_url": "data:image/jpeg;base64,x"},
                ],
            }
        ]
    }

    assert budget.extract_task_id(payload) == "T1"


def test_tampered_ledger_fails_before_an_http_request(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    ledger_path = tmp_path / "ledger.json"
    target = tmp_path / "target.py"
    write_manifest(manifest)
    target.write_text("raise AssertionError('target must not run')\n", encoding="utf-8")
    ledger = budget.AttemptLedger(
        ledger_path,
        stage="first_pass",
        max_attempts=5,
        manifest_sha256=budget.sha256(manifest),
        task_ids=["T1"],
        stage_identity_sha256="a" * 64,
    )
    ledger.ensure()
    value = json.loads(ledger_path.read_text(encoding="utf-8"))
    value["max_attempts_per_task"] = 4
    ledger_path.write_text(json.dumps(value), encoding="utf-8")

    actual_requests = []

    def fake_request(*args, **kwargs):
        actual_requests.append(True)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(budget.requests.sessions.Session, "request", fake_request)
    monkeypatch.setattr(sys, "argv", wrapper_argv(target, ledger_path, manifest))
    with pytest.raises(RuntimeError, match="integrity check failed"):
        budget.main()
    assert actual_requests == []


def test_successful_prediction_without_attestation_is_not_reused(tmp_path):
    task_dir = tmp_path / "T1"
    task_dir.mkdir()
    (task_dir / "prediction.json").write_text(
        json.dumps({"ok": True, "task_id": "T1"}), encoding="utf-8"
    )
    identity = {
        "stage": "first_pass",
        "manifest_sha256": "m",
        "max_http_attempts_per_task": 5,
    }

    status = pipeline.phase_status(
        ["T1"], tmp_path, expected_identity=identity,
        ledger_path=tmp_path / "missing-ledger.json",
    )

    assert status["ok"] == []
    assert status["failed"] == ["T1"]
    assert status["errors"]["T1"] == "missing execution_attestation.json"


def test_tampered_execution_attestation_is_a_hard_failure(tmp_path):
    task_dir = tmp_path / "T1"
    task_dir.mkdir()
    (task_dir / "prediction.json").write_text(
        json.dumps({"ok": True, "task_id": "T1"}), encoding="utf-8"
    )
    (task_dir / "execution_attestation.json").write_text(
        json.dumps({
            "schema_version": 1,
            "pass": True,
            "stage": "first_pass",
            "task_id": "T1",
            "stage_identity": {"stage": "different"},
        }),
        encoding="utf-8",
    )
    identity = {
        "stage": "first_pass",
        "manifest_sha256": "m",
        "max_http_attempts_per_task": 5,
    }

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        pipeline.phase_status(
            ["T1"], tmp_path, expected_identity=identity,
            ledger_path=tmp_path / "missing-ledger.json",
        )


def test_aggregate_recomputes_dimension_mean_from_grades(tmp_path):
    task_dir = tmp_path / "T1"
    task_dir.mkdir()
    prediction = {
        "ok": True,
        "task_id": "T1",
        "dimension_mean_score": 0.0,
        "judge_json": {
            "task_id": "T1",
            "dimensions": {"D1": {"grade": "G"}},
        },
    }
    (task_dir / "prediction.json").write_text(
        json.dumps(prediction), encoding="utf-8"
    )
    task_specs = {
        "T1": {
            "id": "T1",
            "dimensions": {"D1": {"applicable": True}},
        }
    }

    with pytest.raises(RuntimeError, match="differs from grades"):
        pipeline.aggregate_results(["T1"], tmp_path, task_specs)


def test_execution_attestation_binds_outputs_and_ledger(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest)
    identity = {
        "stage": "review_1",
        "manifest_sha256": budget.sha256(manifest),
        "script_sha256": "s",
        "prompt_sha256": "p",
        "dependencies": [],
        "request_budget_runner_sha256": "r",
        "requested_model": "GPT-5.5",
        "api_url": "https://judge.invalid/v1",
        "native_video_input": False,
        "input_mode": "first_pass_plus_same_timestamped_evidence",
        "max_http_attempts_per_task": 5,
    }
    ledger_path = tmp_path / "attempts.json"
    ledger = budget.AttemptLedger(
        ledger_path,
        stage="review_1",
        max_attempts=5,
        manifest_sha256=budget.sha256(manifest),
        task_ids=["T1"],
        stage_identity_sha256=pipeline.json_sha256(identity),
    )
    ledger.ensure()
    attempt = ledger.claim("T1", "POST", identity["api_url"], {"x": 1})
    ledger.finish("T1", attempt, status="response", status_code=200)

    task_spec = {"id": "T1"}
    task_dir = tmp_path / "stage" / "T1"
    task_dir.mkdir(parents=True)
    (task_dir / "prediction.json").write_text(json.dumps({
        "ok": True,
        "task_id": "T1",
        "model": identity["requested_model"],
        "api_url": identity["api_url"],
        "prompt_sha256": identity["prompt_sha256"],
        "native_video_input": False,
        "input_mode": identity["input_mode"],
        "judge_json": {"task_id": "T1"},
    }), encoding="utf-8")
    (task_dir / "review_input.json").write_text(
        json.dumps({"task_spec": task_spec}), encoding="utf-8"
    )
    (task_dir / "raw_response.txt").write_text("{}\n", encoding="utf-8")

    pipeline.attest_successful_predictions(
        stage_name="review_1",
        task_ids=["T1"],
        task_specs={"T1": task_spec},
        task_entries={"T1": {"task_spec": task_spec}},
        directory=tmp_path / "stage",
        expected_identity=identity,
        ledger_path=ledger_path,
    )
    verified, error = pipeline.verify_execution_attestation(
        task_id="T1",
        directory=tmp_path / "stage",
        expected_identity=identity,
        ledger_path=ledger_path,
    )
    assert verified, error

    attestation_path = task_dir / "execution_attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["artifacts"] = attestation["artifacts"][1:]
    attestation[pipeline.EXECUTION_ATTESTATION_INTEGRITY_FIELD] = (
        pipeline.execution_attestation_integrity(attestation)
    )
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    verified, error = pipeline.verify_execution_attestation(
        task_id="T1",
        directory=tmp_path / "stage",
        expected_identity=identity,
        ledger_path=ledger_path,
    )
    assert not verified
    assert error == "execution attestation omits required artifacts"
