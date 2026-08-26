import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import judge_runtime as pipeline  # noqa: E402


def test_single_task_batches_are_attested_before_next_request(tmp_path, monkeypatch):
    completed = set()
    events = []

    def status(task_ids, _directory, **_kwargs):
        ok = [task_id for task_id in task_ids if task_id in completed]
        missing = [task_id for task_id in task_ids if task_id not in completed]
        return {"ok": ok, "failed": [], "missing": missing, "errors": {}}

    def run_attempt(command, _log_path):
        task_ids = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--task-id"
        ]
        events.append(("run", task_ids))
        completed.update(task_ids)
        return 0

    def attest_successful_predictions(*, task_ids, **_kwargs):
        events.append(("attest", list(task_ids)))

    monkeypatch.setattr(pipeline, "phase_status", status)
    monkeypatch.setattr(pipeline, "run_attempt", run_attempt)
    monkeypatch.setattr(
        pipeline, "attest_successful_predictions", attest_successful_predictions
    )

    state = {"phases": {}}
    pipeline.supervise_phase(
        name="first_pass",
        script=tmp_path / "judge.py",
        prompt=tmp_path / "prompt.md",
        manifest=tmp_path / "manifest.json",
        task_ids=["A1001", "A1002"],
        out_dir=tmp_path / "out",
        first_pass_dir=None,
        max_attempts=1,
        gap_s=0,
        state=state,
        state_path=tmp_path / "state.json",
        log_path=tmp_path / "pipeline.log",
        task_specs={},
        task_entries={},
        expected_identity={},
        task_batch_size=1,
    )

    assert events == [
        ("run", ["A1001"]),
        ("attest", ["A1001"]),
        ("run", ["A1002"]),
        ("attest", ["A1002"]),
    ]
    assert state["phases"]["first_pass"]["status"] == "complete"
    assert state["phases"]["first_pass"]["ok"] == 2
    assert state["phases"]["first_pass"]["attempts_completed"] == 1


def test_interrupted_batch_resumes_same_phase_attempt(tmp_path, monkeypatch):
    state = {
        "phases": {
            "first_pass": {
                "status": "running",
                "attempts": 1,
                "attempts_completed": 0,
                "started_at": "already-started",
            }
        }
    }
    completed = set()

    def status(task_ids, _directory, **_kwargs):
        ok = [task_id for task_id in task_ids if task_id in completed]
        missing = [task_id for task_id in task_ids if task_id not in completed]
        return {"ok": ok, "failed": [], "missing": missing, "errors": {}}

    def run_attempt(command, _log_path):
        task_id = command[command.index("--task-id") + 1]
        completed.add(task_id)
        return 0

    monkeypatch.setattr(pipeline, "phase_status", status)
    monkeypatch.setattr(pipeline, "run_attempt", run_attempt)

    pipeline.supervise_phase(
        name="first_pass",
        script=tmp_path / "judge.py",
        prompt=tmp_path / "prompt.md",
        manifest=tmp_path / "manifest.json",
        task_ids=["A1001"],
        out_dir=tmp_path / "out",
        first_pass_dir=None,
        max_attempts=1,
        gap_s=0,
        state=state,
        state_path=tmp_path / "state.json",
        log_path=tmp_path / "pipeline.log",
        task_batch_size=1,
    )

    phase = state["phases"]["first_pass"]
    assert phase["status"] == "complete"
    assert phase["current_attempt"] == 1
    assert phase["attempts_completed"] == 1
