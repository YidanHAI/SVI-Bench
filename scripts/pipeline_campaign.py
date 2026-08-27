#!/usr/bin/env python3
"""Run recording followed by the five-stage Judge as one resumable workflow."""

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


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = Path(
    os.environ.get("PIPELINE_OUTPUT_ROOT", ROOT / "outputs" / "pipeline")
).resolve()
STATE_PATH = WORKFLOW_ROOT / "pipeline_state.json"
LOG_PATH = WORKFLOW_ROOT / "pipeline.log"
LOCK_PATH = WORKFLOW_ROOT / "pipeline.lock"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def pid_alive(value: Any) -> bool:
    if not isinstance(value, int) or value <= 1:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recording_output_root() -> Path:
    config_path = Path(
        os.environ.get(
            "VL_INTERACTION_CAMPAIGN_CONFIG",
            ROOT / "config" / "recording_campaign.json",
        )
    )
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_json(config_path.resolve())
    if config is None:
        raise RuntimeError(f"Cannot read recording campaign config: {config_path}")
    raw_value = str(config.get("output_root") or "").strip()
    if not raw_value:
        raise RuntimeError("Recording campaign config has no output_root")
    raw = Path(raw_value)
    return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()


def recording_snapshot() -> dict[str, Any]:
    current_path = recording_output_root() / "current.json"
    current = load_json(current_path)
    if current is None:
        return {"status": "not_started", "live": False, "current": str(current_path)}
    raw_state_path = str(current.get("state_path") or "").strip()
    state_path = Path(raw_state_path) if raw_state_path else None
    state = load_json(state_path) if state_path is not None and state_path.is_file() else None
    pid = current.get("pid")
    return {
        "status": str((state or {}).get("status") or current.get("status") or "unknown"),
        "live": pid_alive(pid),
        "pid": pid,
        "campaign_id": str(
            (state or {}).get("campaign_id") or current.get("campaign_id") or ""
        ),
        "state_path": str(state_path) if state_path is not None else "",
        "run_root": (
            str(state_path.parent.resolve())
            if state_path is not None and state_path.is_file()
            else ""
        ),
        "last_error": (state or {}).get("last_error") or current.get("error") or "",
    }


def judge_snapshot(judge_root: Path | None) -> dict[str, Any]:
    if judge_root is None:
        return {"status": "pending", "live": False}
    state_path = judge_root / "campaign_state.json"
    state = load_json(state_path)
    if state is None:
        return {"status": "not_started", "live": False, "state_path": str(state_path)}
    pid = state.get("supervisor_pid")
    return {
        "status": str(state.get("status") or "unknown"),
        "live": pid_alive(pid),
        "pid": pid,
        "completed_models": state.get("completed_models", 0),
        "completed_final_predictions": state.get("completed_final_predictions", 0),
        "completed_stage_predictions": state.get("completed_stage_predictions", 0),
        "state_path": str(state_path),
        "last_error": state.get("detail") or "",
    }


def workflow_state(
    status: str,
    *,
    recording: dict[str, Any] | None = None,
    judge: dict[str, Any] | None = None,
    judge_root: Path | None = None,
    detail: str = "",
) -> dict[str, Any]:
    previous = load_json(STATE_PATH) or {}
    return {
        "schema_version": 1,
        "kind": "record_then_judge_pipeline",
        "status": status,
        "started_at": previous.get("started_at") or now(),
        "updated_at": now(),
        "finished_at": now() if status in {"complete", "failed", "stopped"} else None,
        "supervisor_pid": os.getpid() if status == "running" else None,
        "recording": recording or recording_snapshot(),
        "judge": judge or judge_snapshot(judge_root),
        "judge_output_root": str(judge_root) if judge_root else "",
        "detail": detail,
    }


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def stop_children(judge_root: Path | None) -> None:
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "record_all.sh"), "stop"],
        cwd=ROOT,
        check=False,
    )
    judge_env = os.environ.copy()
    if judge_root is not None:
        judge_env["JUDGE_OUTPUT_ROOT"] = str(judge_root)
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "judge_all.sh"), "stop"],
        cwd=ROOT,
        env=judge_env,
        check=False,
    )


def wait_for_recording(stop_requested, poll_seconds: float) -> dict[str, Any]:
    snapshot = recording_snapshot()
    if snapshot["status"] == "complete":
        run_checked([
            "bash", str(ROOT / "scripts" / "record_all.sh"),
            "dry-run", "--resume-current",
        ])
        return recording_snapshot()

    if not snapshot["live"]:
        command = ["bash", str(ROOT / "scripts" / "record_all.sh"), "start"]
        if snapshot["status"] != "not_started":
            command.append("--resume-current")
        run_checked(command)

    while True:
        if stop_requested():
            raise InterruptedError("stop requested")
        snapshot = recording_snapshot()
        atomic_json(STATE_PATH, workflow_state("running", recording=snapshot))
        if snapshot["status"] == "complete":
            if not snapshot["run_root"]:
                raise RuntimeError("Completed recording campaign has no run directory")
            return snapshot
        if snapshot["status"] in {"failed", "complete_with_errors", "stopped"}:
            raise RuntimeError(
                f"Recording campaign ended with status={snapshot['status']}: "
                f"{snapshot['last_error']}"
            )
        if not snapshot["live"]:
            run_checked([
                "bash", str(ROOT / "scripts" / "record_all.sh"),
                "start", "--resume-current",
            ])
        time.sleep(poll_seconds)


def wait_for_judge(
    recording: dict[str, Any], stop_requested, poll_seconds: float
) -> tuple[dict[str, Any], Path]:
    run_id = recording["campaign_id"]
    if not run_id:
        raise RuntimeError("Completed recording campaign has no campaign_id")
    configured = os.environ.get("JUDGE_OUTPUT_ROOT", "").strip()
    judge_root = (
        Path(configured).resolve()
        if configured
        else (ROOT / "outputs" / "judge_campaign" / "runs" / run_id).resolve()
    )
    judge_env = os.environ.copy()
    judge_env["JUDGE_RECORDING_RUN"] = recording["run_root"]
    judge_env["JUDGE_OUTPUT_ROOT"] = str(judge_root)

    snapshot = judge_snapshot(judge_root)
    if not snapshot["live"]:
        # Always enter through judge_all.sh, including for a completed output.
        # It regenerates the manifest index and verifies that existing results
        # match the current recordings, Judge model, scripts, and prompts.
        run_checked(
            ["bash", str(ROOT / "scripts" / "judge_all.sh"), "start"],
            env=judge_env,
        )

    while True:
        if stop_requested():
            raise InterruptedError("stop requested")
        snapshot = judge_snapshot(judge_root)
        atomic_json(
            STATE_PATH,
            workflow_state(
                "running", recording=recording, judge=snapshot, judge_root=judge_root
            ),
        )
        if snapshot["status"] == "complete":
            return snapshot, judge_root
        if snapshot["status"] in {"failed", "stopped"}:
            raise RuntimeError(
                f"Judge campaign ended with status={snapshot['status']}: "
                f"{snapshot['last_error']}"
            )
        if not snapshot["live"]:
            run_checked(
                ["bash", str(ROOT / "scripts" / "judge_all.sh"), "start"],
                env=judge_env,
            )
        time.sleep(poll_seconds)


def run(poll_seconds: float) -> int:
    WORKFLOW_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another full-pipeline supervisor holds the lock") from exc

    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    judge_root: Path | None = None
    recording: dict[str, Any] | None = None
    try:
        atomic_json(STATE_PATH, workflow_state("running"))
        recording = wait_for_recording(lambda: stopping, poll_seconds)
        judge, judge_root = wait_for_judge(
            recording, lambda: stopping, poll_seconds
        )
        atomic_json(
            STATE_PATH,
            workflow_state(
                "complete",
                recording=recording,
                judge=judge,
                judge_root=judge_root,
                detail="Recording and five-stage Judge completed",
            ),
        )
        return 0
    except InterruptedError:
        stop_children(judge_root)
        atomic_json(
            STATE_PATH,
            workflow_state(
                "stopped",
                recording=recording,
                judge_root=judge_root,
                detail="Stopped by operator; completed work remains resumable",
            ),
        )
        return 0
    except Exception as exc:
        atomic_json(
            STATE_PATH,
            workflow_state(
                "failed",
                recording=recording,
                judge_root=judge_root,
                detail=str(exc),
            ),
        )
        raise


def start(poll_seconds: float) -> int:
    previous = load_json(STATE_PATH) or {}
    if previous.get("status") == "running" and pid_alive(previous.get("supervisor_pid")):
        print(json.dumps({
            "status": "already_running",
            "supervisor_pid": previous["supervisor_pid"],
            "state": str(STATE_PATH),
        }, indent=2))
        return 0
    if previous.get("status") == "complete":
        validate(emit=False)
        print(json.dumps({
            "status": "already_complete",
            "state": str(STATE_PATH),
            "detail": (
                "Use new PIPELINE_OUTPUT_ROOT and JUDGE_OUTPUT_ROOT values "
                "for an independent run"
            ),
        }, indent=2))
        return 0

    WORKFLOW_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "run",
                "--poll-seconds",
                str(poll_seconds),
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    deadline = time.time() + 10
    while time.time() < deadline:
        state = load_json(STATE_PATH) or {}
        if state.get("status") == "running" and state.get("supervisor_pid") == process.pid:
            print(json.dumps({
                "status": "started",
                "supervisor_pid": process.pid,
                "state": str(STATE_PATH),
                "log": str(LOG_PATH),
            }, indent=2))
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.25)
    raise RuntimeError(f"Pipeline supervisor failed to initialize; see {LOG_PATH}")


def status() -> int:
    state = load_json(STATE_PATH)
    if state is None:
        state = workflow_state("not_started")
    else:
        pid = state.get("supervisor_pid")
        state["supervisor_alive"] = pid_alive(pid)
        state["recording"] = recording_snapshot()
        judge_root_raw = str(state.get("judge_output_root") or "")
        judge_root = Path(judge_root_raw) if judge_root_raw else None
        state["judge"] = judge_snapshot(judge_root)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def stop() -> int:
    state = load_json(STATE_PATH) or {}
    pid = state.get("supervisor_pid")
    if pid_alive(pid):
        os.kill(int(pid), signal.SIGTERM)
        print(json.dumps({"status": "stop_requested", "supervisor_pid": pid}, indent=2))
        return 0
    judge_root_raw = str(state.get("judge_output_root") or "")
    stop_children(Path(judge_root_raw) if judge_root_raw else None)
    print(json.dumps({"status": "no_live_supervisor", "children_stop_requested": True}, indent=2))
    return 0


def validate(*, emit: bool = True) -> int:
    run_checked(["bash", str(ROOT / "scripts" / "record_all.sh"), "dry-run"])
    recording = recording_snapshot()
    result: dict[str, Any] = {
        "status": "validated",
        "recording": "ready",
        "judge": "pending_completed_recordings",
    }
    if recording["status"] == "complete" and recording["run_root"]:
        run_checked([
            "bash", str(ROOT / "scripts" / "record_all.sh"),
            "dry-run", "--resume-current",
        ])
        judge_root = Path(
            os.environ.get(
                "JUDGE_OUTPUT_ROOT",
                ROOT / "outputs" / "judge_campaign" / "runs" / recording["campaign_id"],
            )
        ).resolve()
        env = os.environ.copy()
        env["JUDGE_RECORDING_RUN"] = recording["run_root"]
        env["JUDGE_OUTPUT_ROOT"] = str(judge_root)
        run_checked(["bash", str(ROOT / "scripts" / "judge_all.sh"), "validate"], env=env)
        result["judge"] = "ready"
        result["judge_output_root"] = str(judge_root)
    if emit:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "run", "status", "stop", "validate"))
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.environ.get("PIPELINE_POLL_SECONDS", "30")),
    )
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        raise RuntimeError("--poll-seconds must be at least 1")
    if args.action == "run":
        return run(args.poll_seconds)
    if args.action == "start":
        return start(args.poll_seconds)
    if args.action == "status":
        return status()
    if args.action == "stop":
        return stop()
    return validate()


if __name__ == "__main__":
    raise SystemExit(main())
