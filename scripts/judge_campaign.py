#!/usr/bin/env python3
"""Start, monitor, resume, stop, or validate the five-model Judge campaign."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import prepare_judge_manifests as manifests
from judge_api import DEFAULT_JUDGE_MODEL, chat_completions_url
from judge_workbook_contract import sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = Path(
    os.environ.get("JUDGE_OUTPUT_ROOT", ROOT / "outputs" / "judge_campaign")
).resolve()
MANIFEST_INDEX = CAMPAIGN_ROOT / "manifests" / "manifest_index.json"
STATE_PATH = CAMPAIGN_ROOT / "campaign_state.json"
SUPERVISOR_LOG = CAMPAIGN_ROOT / "supervisor.log"
LOCK_PATH = CAMPAIGN_ROOT / "supervisor.lock"
MODEL_LOG_DIR = CAMPAIGN_ROOT / "logs"
LEADERBOARD_PATH = CAMPAIGN_ROOT / "leaderboard.json"
STAGES = (
    ("first_pass", "01_first_pass"),
    ("review_1", "02_review_1"),
    ("review_2", "03_review_2"),
    ("adjudication", "04_adjudication"),
    ("final_review", "05_final_review"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def require_environment(*, require_key: bool) -> str:
    api_url = chat_completions_url()
    if require_key and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is not set")
    return api_url


def load_campaign_inputs() -> dict[str, Any]:
    index = load_json(MANIFEST_INDEX, "Judge manifest index")
    if (
        index.get("kind") != "five_model_judge_manifest_index"
        or index.get("contains_human_labels") is not False
        or index.get("model_count") != 5
        or index.get("total_tasks") != 375
        or set(index.get("models") or {}) != set(manifests.EXPECTED_MODELS)
    ):
        raise RuntimeError("Judge manifest index has unexpected scope")
    for model_id, item in index["models"].items():
        path = Path(str(item.get("path") or "")).resolve()
        if not path.is_file() or sha256(path) != item.get("sha256"):
            raise RuntimeError(f"Manifest changed or is missing: {model_id}")
    return index


def model_command(model_id: str, manifest_path: Path, validate_only: bool) -> list[str]:
    command = [
        sys.executable,
        str((ROOT / "scripts" / "judge_pipeline.py").resolve()),
        "--manifest", str(manifest_path),
        "--out-root", str((CAMPAIGN_ROOT / "models" / model_id).resolve()),
        "--max-attempts", "5",
        "--max-http-attempts-per-task", "5",
        "--task-batch-size", "1",
        "--gap-s", os.environ.get("JUDGE_REQUEST_GAP_SECONDS", "2"),
        "--judge-model", os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        "--api-url", chat_completions_url(),
    ]
    if validate_only:
        command.append("--validate-only")
    return command


def prediction_count(directory: Path) -> int:
    count = 0
    for path in directory.glob("*/prediction.json"):
        try:
            if load_json(path, "prediction").get("ok") is True:
                count += 1
        except RuntimeError:
            continue
    return count


def inspect_model(model_id: str) -> dict[str, Any]:
    out_root = CAMPAIGN_ROOT / "models" / model_id
    pipeline_state_path = out_root / "pipeline_state.json"
    pipeline: dict[str, Any] = {}
    if pipeline_state_path.is_file():
        try:
            pipeline = load_json(pipeline_state_path, "pipeline state")
        except RuntimeError as exc:
            pipeline = {"status": "unreadable", "error": str(exc)}
    stage_counts = {
        name: prediction_count(out_root / directory) for name, directory in STAGES
    }
    summary_path = out_root / "judge_results_summary.json"
    summary_complete = False
    scores: dict[str, float | None] = {
        "overall_score": None,
        "score_without_d3": None,
        "raw_dimension_mean": None,
    }
    if summary_path.is_file():
        try:
            summary = load_json(summary_path, "Judge results summary")
            summary_complete = (
                summary.get("task_count") == 75
                and summary.get("ok") == 75
                and summary.get("failed") == 0
                and summary.get("judge_stage_count") == 5
                and isinstance(summary.get("overall_score"), (int, float))
                and isinstance(summary.get("score_without_d3"), (int, float))
            )
            for output_name, source_name, scale in (
                ("overall_score", "overall_score", 1.0),
                ("score_without_d3", "score_without_d3", 1.0),
                ("raw_dimension_mean", "dimension_mean_score_mean", 100.0),
            ):
                value = summary.get(source_name)
                scores[output_name] = (
                    scale * float(value) if isinstance(value, (int, float)) else None
                )
        except RuntimeError:
            pass
    return {
        "pipeline_status": pipeline.get("status", "not_started"),
        "current_phase": pipeline.get("current_phase"),
        "stage_counts": stage_counts,
        "stage_predictions_complete": sum(stage_counts.values()),
        "stage_predictions_expected": 375,
        "complete": pipeline.get("status") == "complete" and summary_complete,
        "pipeline_state": str(pipeline_state_path),
        "result_summary": str(summary_path),
        "scores": scores,
        "error": pipeline.get("error"),
    }


def write_leaderboard(model_states: dict[str, dict[str, Any]]) -> None:
    rows = [
        {
            "model_id": model_id,
            "tasks": 75,
            **item["scores"],
            "result_summary": item["result_summary"],
        }
        for model_id, item in model_states.items()
    ]
    rows.sort(
        key=lambda item: (
            item["overall_score"] is not None,
            item["overall_score"] if item["overall_score"] is not None else -1,
        ),
        reverse=True,
    )
    atomic_json(LEADERBOARD_PATH, {
        "schema_version": 1,
        "metric": "item_level_min_d1_d2_then_mean",
        "score_scale": "0-100",
        "judge_model": os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        "judge_stages": 5,
        "models": rows,
    })


def campaign_snapshot(
    *,
    status: str,
    attempts: dict[str, int],
    active: dict[str, subprocess.Popen[Any]] | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    active = active or {}
    model_states = {}
    for model_id in manifests.EXPECTED_MODELS:
        info = inspect_model(model_id)
        info["launcher_attempts"] = int(attempts.get(model_id, 0))
        info["process_pid"] = active[model_id].pid if model_id in active else None
        model_states[model_id] = info
    return {
        "schema_version": 1,
        "kind": "five_model_five_stage_judge_campaign",
        "status": status,
        "updated_at": now(),
        "supervisor_pid": os.getpid() if status in {"running", "stopping"} else None,
        "judge_model": os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        "judge_stages": 5,
        "model_count": 5,
        "tasks_per_model": 75,
        "total_tasks": 375,
        "expected_judge_calls": 1875,
        "completed_models": sum(item["complete"] for item in model_states.values()),
        "completed_final_predictions": sum(
            item["stage_counts"]["final_review"] for item in model_states.values()
        ),
        "completed_stage_predictions": sum(
            item["stage_predictions_complete"] for item in model_states.values()
        ),
        "manifest_index": {
            "path": str(MANIFEST_INDEX),
            "sha256": sha256(MANIFEST_INDEX) if MANIFEST_INDEX.is_file() else None,
        },
        "detail": detail,
        "models": model_states,
    }


def validate_all(index: dict[str, Any]) -> None:
    for model_id in manifests.EXPECTED_MODELS:
        manifest_path = Path(index["models"][model_id]["path"]).resolve()
        completed = subprocess.run(
            model_command(model_id, manifest_path, True),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Validate-only failed for {model_id}:\n{completed.stdout[-8000:]}"
            )


def run_supervisor(max_parallel: int, max_launcher_attempts: int) -> int:
    require_environment(require_key=True)
    index = load_campaign_inputs()
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another Judge campaign supervisor holds the lock") from exc

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    previous = load_json(STATE_PATH, "campaign state") if STATE_PATH.is_file() else {}
    attempts = {
        model_id: int(((previous.get("models") or {}).get(model_id) or {}).get(
            "launcher_attempts", 0
        ))
        for model_id in manifests.EXPECTED_MODELS
    }
    active: dict[str, subprocess.Popen[Any]] = {}
    next_launch_at = {model_id: 0.0 for model_id in manifests.EXPECTED_MODELS}
    MODEL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE_PATH, campaign_snapshot(status="running", attempts=attempts))

    while True:
        for model_id, process in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del active[model_id]
            if return_code != 0:
                next_launch_at[model_id] = time.time() + min(
                    300, 15 * (2 ** max(0, attempts[model_id] - 1))
                )

        inspections = {
            model_id: inspect_model(model_id) for model_id in manifests.EXPECTED_MODELS
        }
        if all(item["complete"] for item in inspections.values()):
            write_leaderboard(inspections)
            atomic_json(STATE_PATH, campaign_snapshot(
                status="complete",
                attempts=attempts,
                detail="All five models completed 75 tasks through five stages",
            ))
            return 0
        if stop_requested:
            atomic_json(STATE_PATH, campaign_snapshot(
                status="stopping", attempts=attempts, active=active
            ))
            for process in active.values():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.time() + 20
            while active and time.time() < deadline:
                for model_id, process in list(active.items()):
                    if process.poll() is not None:
                        del active[model_id]
                time.sleep(0.25)
            for process in active.values():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            atomic_json(STATE_PATH, campaign_snapshot(
                status="stopped",
                attempts=attempts,
                detail="Stopped by operator; completed tasks remain resumable",
            ))
            return 0

        for model_id in manifests.EXPECTED_MODELS:
            if len(active) >= max_parallel:
                break
            if model_id in active or inspections[model_id]["complete"]:
                continue
            if attempts[model_id] >= max_launcher_attempts:
                continue
            if time.time() < next_launch_at[model_id]:
                continue
            manifest_path = Path(index["models"][model_id]["path"]).resolve()
            log_path = MODEL_LOG_DIR / f"{manifests.safe_slug(model_id)}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(f"\n[{now()}] launcher attempt {attempts[model_id] + 1}\n")
            log_handle.flush()
            process = subprocess.Popen(
                model_command(model_id, manifest_path, False),
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            log_handle.close()
            attempts[model_id] += 1
            active[model_id] = process

        exhausted = [
            model_id
            for model_id in manifests.EXPECTED_MODELS
            if not inspections[model_id]["complete"]
            and model_id not in active
            and attempts[model_id] >= max_launcher_attempts
        ]
        status = "failed" if exhausted and not active else "running"
        detail = (
            f"Launcher attempts exhausted for: {', '.join(exhausted)}"
            if exhausted else None
        )
        atomic_json(STATE_PATH, campaign_snapshot(
            status=status, attempts=attempts, active=active, detail=detail
        ))
        if exhausted and not active:
            return 1
        time.sleep(10)


def start(max_parallel: int, max_launcher_attempts: int) -> int:
    require_environment(require_key=True)
    index = load_campaign_inputs()
    current_index_hash = sha256(MANIFEST_INDEX)
    if STATE_PATH.is_file():
        previous = load_json(STATE_PATH, "campaign state")
        pid = previous.get("supervisor_pid")
        previous_index_hash = (previous.get("manifest_index") or {}).get("sha256")
        if previous_index_hash and previous_index_hash != current_index_hash:
            raise RuntimeError(
                "JUDGE_OUTPUT_ROOT belongs to a different recording manifest; "
                "select a new output root"
            )
        if previous.get("status") == "running" and pid_alive(pid):
            print(json.dumps({"status": "already_running", "pid": pid}, indent=2))
            return 0
        # Revalidate every per-model execution identity before accepting or
        # resuming an existing output root. This is local-only and catches a
        # changed model, endpoint, script, prompt, or manifest.
        validate_all(index)
        if previous.get("status") == "complete":
            inspections = {
                model_id: inspect_model(model_id)
                for model_id in manifests.EXPECTED_MODELS
            }
            if not all(item["complete"] for item in inspections.values()):
                raise RuntimeError(
                    "Campaign state says complete but one or more model results are incomplete"
                )
            write_leaderboard(inspections)
            print(json.dumps({"status": "already_complete"}, indent=2))
            return 0
    else:
        validate_all(index)
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "run",
                "--max-parallel", str(max_parallel),
                "--max-launcher-attempts", str(max_launcher_attempts),
            ],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    deadline = time.time() + 10
    while time.time() < deadline:
        if STATE_PATH.is_file():
            state = load_json(STATE_PATH, "campaign state")
            if state.get("supervisor_pid") == process.pid:
                print(json.dumps({
                    "status": "started",
                    "supervisor_pid": process.pid,
                    "campaign_state": str(STATE_PATH),
                    "max_parallel": max_parallel,
                }, ensure_ascii=False, indent=2))
                return 0
        if process.poll() is not None:
            break
        time.sleep(0.25)
    raise RuntimeError(f"Judge supervisor failed to initialize; see {SUPERVISOR_LOG}")


def status() -> int:
    attempts: dict[str, int] = {}
    recorded_status = "not_started"
    supervisor_pid = None
    detail = None
    if STATE_PATH.is_file():
        previous = load_json(STATE_PATH, "campaign state")
        attempts = {
            model_id: int(((previous.get("models") or {}).get(model_id) or {}).get(
                "launcher_attempts", 0
            ))
            for model_id in manifests.EXPECTED_MODELS
        }
        recorded_status = str(previous.get("status") or "unknown")
        supervisor_pid = previous.get("supervisor_pid")
        detail = previous.get("detail")
    snapshot = campaign_snapshot(
        status=recorded_status, attempts=attempts, detail=detail
    )
    snapshot["supervisor_pid"] = supervisor_pid
    snapshot["supervisor_alive"] = pid_alive(supervisor_pid)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return 0


def stop() -> int:
    if not STATE_PATH.is_file():
        print(json.dumps({"status": "not_started"}, indent=2))
        return 0
    state = load_json(STATE_PATH, "campaign state")
    pid = state.get("supervisor_pid")
    if state.get("status") != "running" or not pid_alive(pid):
        print(json.dumps({
            "status": state.get("status"), "supervisor_alive": False
        }, indent=2))
        return 0
    os.kill(int(pid), signal.SIGTERM)
    print(json.dumps({"status": "stop_requested", "supervisor_pid": pid}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "run", "status", "stop", "validate"))
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--max-launcher-attempts", type=int, default=5)
    args = parser.parse_args(argv)
    if not 1 <= args.max_parallel <= 5:
        raise RuntimeError("--max-parallel must be 1..5")
    if not 1 <= args.max_launcher_attempts <= 5:
        raise RuntimeError("--max-launcher-attempts must be 1..5")
    if args.action == "run":
        return run_supervisor(args.max_parallel, args.max_launcher_attempts)
    if args.action == "start":
        return start(args.max_parallel, args.max_launcher_attempts)
    if args.action == "status":
        return status()
    if args.action == "stop":
        return stop()
    require_environment(require_key=False)
    index = load_campaign_inputs()
    validate_all(index)
    print(json.dumps({
        "status": "validated",
        "models": 5,
        "tasks": 375,
        "leaderboard": str(LEADERBOARD_PATH),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
