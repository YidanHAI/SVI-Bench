import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import judge_runtime as pipeline
import run_judge_with_attempt_budget as budget


TASK_ID = "T1"
STAGE = "review_1"
STAGE_IDENTITY_SHA256 = "a" * 64


class _CountingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, _CountingHandler)
        self.request_count = 0
        self.request_lock = threading.Lock()


class _CountingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        with self.server.request_lock:
            self.server.request_count += 1
        body = b'{"error":"retryable"}'
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


@pytest.fixture
def counting_server():
    server = _CountingServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}/judge"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _write_manifest(path):
    path.write_text(
        json.dumps({
            "contains_human_labels": False,
            "tasks": [{"task_spec": {"id": TASK_ID}}],
        }),
        encoding="utf-8",
    )


def _payload_source():
    return (
        "payload = {\n"
        "    'messages': [{\n"
        "        'role': 'user',\n"
        "        'content': "
        "'REVIEW_INPUT_JSON: {\"task_spec\": {\"id\": \"T1\"}}',\n"
        "    }],\n"
        "}\n"
    )


def _wrapper_command(target, ledger, manifest):
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_judge_with_attempt_budget.py"),
        "--target",
        str(target),
        "--stage",
        STAGE,
        "--ledger",
        str(ledger),
        "--manifest",
        str(manifest),
        "--stage-identity-sha256",
        STAGE_IDENTITY_SHA256,
        "--max-total-attempts",
        "5",
    ]


def _run_wrapper(target, ledger, manifest):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        _wrapper_command(target, ledger, manifest),
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )


def _read_ledger(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _task_attempts(path):
    return _read_ledger(path)["tasks"][TASK_ID]["attempts"]


def test_application_retries_and_process_restarts_share_five_real_http_calls(
    tmp_path, counting_server,
):
    server, url = counting_server
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.json"
    target = tmp_path / "retry_target.py"
    probe = tmp_path / "single_target.py"
    _write_manifest(manifest)
    target.write_text(
        "import requests\n"
        + _payload_source()
        + "for evidence_variant in range(2):\n"
        + "    for retry in range(2):\n"
        + "        try:\n"
        + f"            requests.post({url!r}, json=payload, timeout=2)\n"
        + "        except Exception:\n"
        + "            pass\n",
        encoding="utf-8",
    )
    probe.write_text(
        "import requests\n"
        + _payload_source()
        + f"requests.post({url!r}, json=payload, timeout=2)\n",
        encoding="utf-8",
    )

    first = _run_wrapper(target, ledger, manifest)
    second = _run_wrapper(target, ledger, manifest)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert server.request_count == 5
    assert len(_task_attempts(ledger)) == 5

    sixth = _run_wrapper(probe, ledger, manifest)
    assert sixth.returncode != 0
    assert "exhausted its persistent 5-request budget" in sixth.stderr
    assert server.request_count == 5, "the rejected sixth call reached the network"
    assert len(_task_attempts(ledger)) == 5


def test_transport_adapter_retries_cannot_bypass_real_http_budget(
    tmp_path, counting_server,
):
    server, url = counting_server
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.json"
    target = tmp_path / "adapter_retry_target.py"
    _write_manifest(manifest)
    target.write_text(
        "import requests\n"
        "from requests.adapters import HTTPAdapter\n"
        "from urllib3.util.retry import Retry\n"
        + _payload_source()
        + "retry = Retry(\n"
        + "    total=7, status=7, connect=0, read=0, redirect=0,\n"
        + "    status_forcelist=[503], allowed_methods=frozenset(['POST']),\n"
        + "    backoff_factor=0, raise_on_status=False,\n"
        + ")\n"
        + "session = requests.Session()\n"
        + "session.mount('http://', HTTPAdapter(max_retries=retry))\n"
        + f"session.post({url!r}, json=payload, timeout=2)\n",
        encoding="utf-8",
    )

    completed = _run_wrapper(target, ledger, manifest)
    attempts = _task_attempts(ledger)
    assert server.request_count <= 5 and len(attempts) == server.request_count, (
        f"transport retries made {server.request_count} real HTTP calls while "
        f"the persistent ledger recorded {len(attempts)}; stderr={completed.stderr!r}"
    )


def _make_ledger(path, manifest):
    ledger = budget.AttemptLedger(
        path,
        stage=STAGE,
        max_attempts=5,
        manifest_sha256=budget.sha256(manifest),
        task_ids=[TASK_ID],
        stage_identity_sha256=STAGE_IDENTITY_SHA256,
    )
    ledger.ensure()
    attempt = ledger.claim(
        TASK_ID,
        "POST",
        "https://judge.invalid/v1",
        {"messages": [{"role": "user", "content": "original"}]},
    )
    ledger.finish(TASK_ID, attempt, status="response", status_code=200)


def _reseal_ledger(document, *, identity=False):
    if identity:
        document["identity_sha256"] = budget.ledger_identity_sha256(document)
    document[budget.INTEGRITY_FIELD] = budget.ledger_integrity_sha256(document)


def _tamper_header_and_reseal(document):
    document["stage"] = "forged_stage"
    _reseal_ledger(document, identity=True)


def _tamper_event_structure_and_reseal(document):
    document["tasks"][TASK_ID]["attempts"][0]["attempt"] = 2
    _reseal_ledger(document)


def _tamper_event_content_and_reseal(document):
    event = document["tasks"][TASK_ID]["attempts"][0]
    event["url"] = "https://forged.invalid/v1"
    event["request_sha256"] = "f" * 64
    event["status_code"] = 599
    _reseal_ledger(document)


def _tamper_integrity(document):
    document[budget.INTEGRITY_FIELD] = "0" * 64


@pytest.mark.parametrize(
    "mutator",
    [
        _tamper_header_and_reseal,
        _tamper_event_structure_and_reseal,
        _tamper_event_content_and_reseal,
        _tamper_integrity,
    ],
    ids=[
        "header-rehashed",
        "event-structure-rehashed",
        "event-content-rehashed",
        "integrity-field",
    ],
)
def test_any_ledger_tamper_is_rejected_before_network(
    tmp_path, counting_server, mutator,
):
    server, url = counting_server
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.json"
    target = tmp_path / "network_target.py"
    _write_manifest(manifest)
    _make_ledger(ledger, manifest)
    document = _read_ledger(ledger)
    mutator(document)
    ledger.write_text(json.dumps(document), encoding="utf-8")
    target.write_text(
        "import requests\n"
        + _payload_source()
        + f"requests.post({url!r}, json=payload, timeout=2)\n",
        encoding="utf-8",
    )

    completed = _run_wrapper(target, ledger, manifest)
    assert completed.returncode != 0, "tampered ledger was accepted"
    assert server.request_count == 0, "tampered ledger reached the network"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_resume_fixture(tmp_path):
    manifest = tmp_path / "manifest.json"
    script = tmp_path / "judge.py"
    prompt = tmp_path / "prompt.md"
    dependency = tmp_path / "dependency.py"
    request_runner = tmp_path / "request_runner.py"
    _write_manifest(manifest)
    script.write_text("# fixed judge\n", encoding="utf-8")
    prompt.write_text("fixed prompt\n", encoding="utf-8")
    dependency.write_text("# fixed dependency\n", encoding="utf-8")
    request_runner.write_text("# fixed request wrapper\n", encoding="utf-8")
    identity = {
        "stage": STAGE,
        "manifest_sha256": _sha256(manifest),
        "script_sha256": _sha256(script),
        "prompt_sha256": _sha256(prompt),
        "dependencies": [{
            "path": str(dependency.resolve()),
            "sha256": _sha256(dependency),
        }],
        "request_budget_runner_sha256": _sha256(request_runner),
        "module_aliases": {},
        "requested_model": "GPT-5.5",
        "api_url": "https://judge.invalid/v1",
        "native_video_input": False,
        "input_mode": "first_pass_plus_same_timestamped_evidence",
        "max_http_attempts_per_task": 5,
    }
    ledger_path = tmp_path / "attempts.json"
    ledger = budget.AttemptLedger(
        ledger_path,
        stage=STAGE,
        max_attempts=5,
        manifest_sha256=_sha256(manifest),
        task_ids=[TASK_ID],
        stage_identity_sha256=pipeline.json_sha256(identity),
    )
    ledger.ensure()
    attempt = ledger.claim(TASK_ID, "POST", identity["api_url"], {"x": 1})
    ledger.finish(TASK_ID, attempt, status="response", status_code=200)

    task_spec = {"id": TASK_ID, "dimensions": {}}
    stage_dir = tmp_path / "stage"
    task_dir = stage_dir / TASK_ID
    task_dir.mkdir(parents=True)
    prediction = {
        "ok": True,
        "task_id": TASK_ID,
        "model": identity["requested_model"],
        "api_url": identity["api_url"],
        "prompt_sha256": identity["prompt_sha256"],
        "native_video_input": False,
        "input_mode": identity["input_mode"],
        "judge_json": {"task_id": TASK_ID},
    }
    (task_dir / "prediction.json").write_text(
        json.dumps(prediction), encoding="utf-8"
    )
    (task_dir / "review_input.json").write_text(
        json.dumps({"task_spec": task_spec}), encoding="utf-8"
    )
    (task_dir / "raw_response.txt").write_text("{}\n", encoding="utf-8")
    pipeline.attest_successful_predictions(
        stage_name=STAGE,
        task_ids=[TASK_ID],
        task_specs={TASK_ID: task_spec},
        task_entries={TASK_ID: {"task_spec": task_spec}},
        directory=stage_dir,
        expected_identity=identity,
        ledger_path=ledger_path,
    )
    baseline = pipeline.phase_status(
        [TASK_ID],
        stage_dir,
        expected_identity=identity,
        ledger_path=ledger_path,
    )
    assert baseline["ok"] == [TASK_ID]
    return {
        "manifest": manifest,
        "identity": identity,
        "ledger": ledger_path,
        "stage_dir": stage_dir,
        "task_dir": task_dir,
    }


def _assert_not_reused(fixture, identity=None):
    try:
        status = pipeline.phase_status(
            [TASK_ID],
            fixture["stage_dir"],
            expected_identity=identity or fixture["identity"],
            ledger_path=fixture["ledger"],
        )
    except RuntimeError:
        return
    assert status["ok"] == [], f"stale prediction was reused: {status}"


def test_missing_prediction_or_attestation_is_never_reused(tmp_path):
    fixture = _make_resume_fixture(tmp_path)
    prediction = fixture["task_dir"] / "prediction.json"
    prediction.unlink()
    status = pipeline.phase_status(
        [TASK_ID],
        fixture["stage_dir"],
        expected_identity=fixture["identity"],
        ledger_path=fixture["ledger"],
    )
    assert status["ok"] == []
    assert status["missing"] == [TASK_ID]

    prediction.write_text(
        json.dumps({"ok": True, "task_id": TASK_ID}), encoding="utf-8"
    )
    (fixture["task_dir"] / "execution_attestation.json").unlink()
    status = pipeline.phase_status(
        [TASK_ID],
        fixture["stage_dir"],
        expected_identity=fixture["identity"],
        ledger_path=fixture["ledger"],
    )
    assert status["ok"] == []
    assert status["failed"] == [TASK_ID]


@pytest.mark.parametrize(
    "field",
    [
        "manifest_sha256",
        "script_sha256",
        "prompt_sha256",
        "requested_model",
        "api_url",
        "input_mode",
        "dependencies",
    ],
)
def test_changed_execution_provenance_prevents_resume(tmp_path, field):
    fixture = _make_resume_fixture(tmp_path)
    changed = copy.deepcopy(fixture["identity"])
    if field == "dependencies":
        changed[field] = [
            {"path": "/forged/dependency.py", "sha256": "f" * 64}
        ]
    else:
        changed[field] = f"forged-{field}"
    _assert_not_reused(fixture, changed)


def _refresh_attestation_artifact(fixture, artifact_name):
    attestation_path = fixture["task_dir"] / "execution_attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    artifact_path = fixture["task_dir"] / artifact_name
    resolved = str(artifact_path.resolve())
    reference = next(
        item for item in attestation["artifacts"] if item["path"] == resolved
    )
    reference["bytes"] = artifact_path.stat().st_size
    reference["sha256"] = _sha256(artifact_path)
    attestation[pipeline.EXECUTION_ATTESTATION_INTEGRITY_FIELD] = (
        pipeline.execution_attestation_integrity(attestation)
    )
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")


@pytest.mark.parametrize(
    "field",
    ["prompt_sha256", "model", "api_url", "input_mode"],
)
def test_self_consistent_attestation_cannot_hide_missing_prediction_provenance(
    tmp_path, field,
):
    fixture = _make_resume_fixture(tmp_path)
    prediction_path = fixture["task_dir"] / "prediction.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction.pop(field)
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    _refresh_attestation_artifact(fixture, "prediction.json")
    _assert_not_reused(fixture)


def test_self_consistent_attestation_cannot_hide_manifest_task_spec_drift(tmp_path):
    fixture = _make_resume_fixture(tmp_path)
    input_path = fixture["task_dir"] / "review_input.json"
    value = json.loads(input_path.read_text(encoding="utf-8"))
    value["task_spec"]["id"] = "OTHER_TASK"
    input_path.write_text(json.dumps(value), encoding="utf-8")
    _refresh_attestation_artifact(fixture, "review_input.json")
    _assert_not_reused(fixture)


def _tamper_attestation_integrity(fixture):
    path = fixture["task_dir"] / "execution_attestation.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value[pipeline.EXECUTION_ATTESTATION_INTEGRITY_FIELD] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")


def _tamper_attestation_semantics_and_reseal(fixture):
    path = fixture["task_dir"] / "execution_attestation.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["stage"] = "forged_stage"
    value[pipeline.EXECUTION_ATTESTATION_INTEGRITY_FIELD] = (
        pipeline.execution_attestation_integrity(value)
    )
    path.write_text(json.dumps(value), encoding="utf-8")


def _tamper_artifact_bytes(fixture):
    (fixture["task_dir"] / "raw_response.txt").write_text("[]\n", encoding="utf-8")


def _tamper_artifact_reference_and_reseal(fixture):
    path = fixture["task_dir"] / "execution_attestation.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["artifacts"][0]["sha256"] = "0" * 64
    value[pipeline.EXECUTION_ATTESTATION_INTEGRITY_FIELD] = (
        pipeline.execution_attestation_integrity(value)
    )
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize(
    "mutator",
    [
        _tamper_attestation_integrity,
        _tamper_attestation_semantics_and_reseal,
        _tamper_artifact_bytes,
        _tamper_artifact_reference_and_reseal,
    ],
    ids=[
        "attestation-integrity",
        "attestation-semantics-rehashed",
        "artifact-bytes",
        "artifact-reference-rehashed",
    ],
)
def test_execution_attestation_or_artifact_tamper_is_a_hard_failure(
    tmp_path, mutator,
):
    fixture = _make_resume_fixture(tmp_path)
    mutator(fixture)
    with pytest.raises(RuntimeError, match="Refusing to overwrite invalid"):
        pipeline.phase_status(
            [TASK_ID],
            fixture["stage_dir"],
            expected_identity=fixture["identity"],
            ledger_path=fixture["ledger"],
        )
