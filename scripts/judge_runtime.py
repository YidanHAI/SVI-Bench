#!/usr/bin/env python3
"""Shared resume, attestation, and aggregation utilities for Judge stages."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from judge_scoring import apply_dimension_mean_score
from run_judge_with_attempt_budget import (
    SCHEMA_VERSION as ATTEMPT_LEDGER_SCHEMA_VERSION,
    ledger_identity_sha256,
    load_ledger_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIRST_SCRIPT = ROOT / "scripts" / "judge_first_pass.py"
DEFAULT_FIRST_PROMPT = ROOT / "prompts" / "judge_first_pass.md"
DEFAULT_REVIEW_SCRIPT = ROOT / "scripts" / "judge_review.py"
DEFAULT_REVIEW_PROMPT = ROOT / "prompts" / "judge_review_1.md"
EXECUTION_ATTESTATION_INTEGRITY_FIELD = "attestation_integrity_sha256"
RECOVERY_ATTESTATION_NAME = "recovery_attestation.json"
RECOVERY_ATTESTATION_INTEGRITY_FIELD = "recovery_integrity_sha256"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def json_sha256(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execution_attestation_integrity(value):
    unsigned = dict(value)
    unsigned.pop(EXECUTION_ATTESTATION_INTEGRITY_FIELD, None)
    return json_sha256(unsigned)


def recovery_attestation_integrity(value):
    unsigned = dict(value)
    unsigned.pop(RECOVERY_ATTESTATION_INTEGRITY_FIELD, None)
    return json_sha256(unsigned)


def artifact_reference(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_artifact_reference(reference):
    if not isinstance(reference, dict):
        return False, "artifact reference is not an object"
    path = Path(str(reference.get("path") or ""))
    if not path.is_file():
        return False, f"artifact is missing: {path}"
    if path.stat().st_size != reference.get("bytes"):
        return False, f"artifact byte count changed: {path}"
    if sha256(path) != reference.get("sha256"):
        return False, f"artifact hash changed: {path}"
    return True, None


def load_attempt_ledger(path, expected_identity=None):
    if path is None or not Path(path).is_file():
        return None
    try:
        return load_ledger_file(path, expected_identity)
    except RuntimeError as exc:
        raise RuntimeError(f"Invalid attempt ledger {path}: {exc}") from exc


def ledger_task_attempts(ledger_path, task_id, expected_identity=None):
    ledger = load_attempt_ledger(ledger_path, expected_identity)
    task = ((ledger or {}).get("tasks") or {}).get(task_id) or {}
    attempts = task.get("attempts") or []
    return attempts if isinstance(attempts, list) else []


def execution_attestation_path(directory, task_id):
    return directory / task_id / "execution_attestation.json"


def verify_execution_attestation(
    *, task_id, directory, expected_identity, ledger_path,
    expected_task_spec=None,
):
    path = execution_attestation_path(directory, task_id)
    if not path.is_file():
        return False, "missing execution_attestation.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid execution attestation: {exc}"
    if (
        value.get(EXECUTION_ATTESTATION_INTEGRITY_FIELD)
        != execution_attestation_integrity(value)
    ):
        return False, "execution attestation integrity check failed"
    if value.get("schema_version") != 1 or value.get("pass") is not True:
        return False, "execution attestation did not pass"
    if value.get("stage") != expected_identity.get("stage"):
        return False, "execution attestation stage mismatch"
    if value.get("task_id") != task_id:
        return False, "execution attestation task_id mismatch"
    if value.get("stage_identity") != expected_identity:
        return False, "execution attestation stage identity mismatch"
    artifact_references = value.get("artifacts")
    if not isinstance(artifact_references, list):
        return False, "execution attestation artifacts are missing"
    artifact_paths = [item.get("path") for item in artifact_references if isinstance(item, dict)]
    required_paths = {
        str((directory / task_id / "prediction.json").resolve()),
        str((directory / task_id / persisted_input_name(expected_identity["stage"])).resolve()),
        str((directory / task_id / "raw_response.txt").resolve()),
    }
    if len(artifact_paths) != len(set(artifact_paths)):
        return False, "execution attestation has duplicate artifacts"
    if not required_paths.issubset(set(artifact_paths)):
        return False, "execution attestation omits required artifacts"
    for reference in artifact_references:
        ok, error = verify_artifact_reference(reference)
        if not ok:
            return False, error
    prediction_path = directory / task_id / "prediction.json"
    prediction = load_prediction(prediction_path)
    if not prediction or prediction.get("ok") is not True:
        return False, "prediction is missing or unsuccessful"
    if (
        prediction.get("task_id") != task_id
        or (prediction.get("judge_json") or {}).get("task_id") != task_id
    ):
        return False, "prediction task id changed"
    expected_prompt_hash = expected_identity.get(
        "prompt_content_sha256", expected_identity.get("prompt_sha256")
    )
    prediction_contract = {
        "prompt_sha256": expected_prompt_hash,
        "model": expected_identity.get("requested_model"),
        "api_url": expected_identity.get("api_url"),
        "native_video_input": expected_identity.get("native_video_input"),
        "input_mode": expected_identity.get("input_mode"),
    }
    for field, expected in prediction_contract.items():
        if prediction.get(field) != expected:
            return False, f"prediction {field} changed"
    input_path = directory / task_id / persisted_input_name(
        expected_identity["stage"]
    )
    try:
        persisted_input = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"persisted input is invalid: {exc}"
    persisted_task_spec = (
        persisted_input.get("task_spec")
        if isinstance(persisted_input, dict) else None
    )
    if not isinstance(persisted_task_spec, dict):
        return False, "persisted task spec is missing"
    if persisted_task_spec.get("id") != task_id:
        return False, "persisted task id changed"
    if expected_task_spec is not None and persisted_task_spec != expected_task_spec:
        return False, "persisted task spec changed"
    expected_ledger_identity = {
        "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
        "stage": expected_identity.get("stage"),
        "max_attempts_per_task": expected_identity.get(
            "max_http_attempts_per_task"
        ),
        "manifest_sha256": expected_identity.get("manifest_sha256"),
        "stage_identity_sha256": json_sha256(expected_identity),
    }
    ledger_document = load_attempt_ledger(ledger_path, expected_ledger_identity)
    attempts = (
        (((ledger_document or {}).get("tasks") or {}).get(task_id) or {})
        .get("attempts") or []
    )
    ledger_reference = value.get("attempt_ledger") or {}
    if ledger_reference.get("path") != str(Path(ledger_path).resolve()):
        return False, "attempt ledger path changed"
    if ledger_reference.get("identity_sha256") != ledger_identity_sha256(
        ledger_document
    ):
        return False, "attempt ledger identity changed"
    if ledger_reference.get("attempt_count") != len(attempts):
        return False, "attempt ledger count changed"
    if ledger_reference.get("attempts_sha256") != json_sha256(attempts):
        return False, "attempt ledger events changed"
    max_attempts = expected_identity.get("max_http_attempts_per_task")
    if not isinstance(max_attempts, int) or not (1 <= len(attempts) <= max_attempts):
        return False, "attempt ledger count is outside the allowed range"
    final = attempts[-1]
    if final.get("status") != "response" or final.get("status_code") != 200:
        return False, "final HTTP attempt was not a successful response"
    return True, None


def persisted_input_name(stage_name):
    if stage_name == "first_pass":
        return "judge_input.json"
    if stage_name in {"adjudication", "adjudicated"}:
        return "adjudication_input.json"
    return "review_input.json"


def attest_successful_predictions(
    *, stage_name, task_ids, task_specs, task_entries, directory, expected_identity,
    ledger_path,
):
    for task_id in task_ids:
        task_dir = directory / task_id
        prediction_path = task_dir / "prediction.json"
        prediction = load_prediction(prediction_path)
        if not prediction or prediction.get("ok") is not True:
            continue
        errors = []
        if prediction.get("task_id") != task_id:
            errors.append("prediction task_id mismatch")
        judge_json = prediction.get("judge_json") or {}
        if judge_json.get("task_id") != task_id:
            errors.append("judge_json task_id mismatch")
        expected_prompt_hash = expected_identity.get(
            "prompt_content_sha256", expected_identity.get("prompt_sha256")
        )
        if prediction.get("prompt_sha256") != expected_prompt_hash:
            errors.append("prediction prompt hash mismatch")
        if prediction.get("model") != expected_identity.get("requested_model"):
            errors.append("prediction requested model mismatch")
        if prediction.get("api_url") != expected_identity.get("api_url"):
            errors.append("prediction API URL mismatch")
        if prediction.get("native_video_input") is not False:
            errors.append("prediction input modality mismatch")
        if prediction.get("input_mode") != expected_identity.get("input_mode"):
            errors.append("prediction input mode mismatch")
        input_path = task_dir / persisted_input_name(stage_name)
        if not input_path.is_file():
            errors.append(f"missing persisted input: {input_path.name}")
            persisted_input = None
        else:
            try:
                persisted_input = json.loads(input_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"invalid persisted input: {exc}")
                persisted_input = None
        if isinstance(persisted_input, dict):
            if persisted_input.get("task_spec") != task_specs[task_id]:
                errors.append("persisted task_spec differs from manifest")

        attempts = ledger_task_attempts(ledger_path, task_id)
        max_attempts = expected_identity.get("max_http_attempts_per_task")
        if not isinstance(max_attempts, int) or not (1 <= len(attempts) <= max_attempts):
            errors.append("persistent HTTP attempt count is outside the allowed range")
        elif attempts[-1].get("status") != "response" or attempts[-1].get("status_code") != 200:
            errors.append("final HTTP attempt was not a successful response")

        artifacts = []
        for artifact in (prediction_path, input_path, task_dir / "raw_response.txt"):
            try:
                artifacts.append(artifact_reference(artifact))
            except FileNotFoundError:
                errors.append(f"missing output artifact: {artifact.name}")
        recovery_path = task_dir / RECOVERY_ATTESTATION_NAME
        recovery = None
        if recovery_path.is_file():
            try:
                recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"invalid {RECOVERY_ATTESTATION_NAME}: {exc}")
            if isinstance(recovery, dict):
                recorded_integrity = recovery.get(
                    RECOVERY_ATTESTATION_INTEGRITY_FIELD
                )
                if recorded_integrity != recovery_attestation_integrity(recovery):
                    errors.append("recovery attestation integrity check failed")
                if (
                    recovery.get("schema_version") != 1
                    or recovery.get("pass") is not True
                    or recovery.get("stage") != stage_name
                    or recovery.get("task_id") != task_id
                ):
                    errors.append("recovery attestation identity check failed")
                current_prediction = artifact_reference(prediction_path)
                if (
                    recovery.get("recovered_prediction_sha256")
                    != current_prediction["sha256"]
                ):
                    errors.append("recovery attestation prediction hash mismatch")
                try:
                    artifacts.append(artifact_reference(recovery_path))
                except FileNotFoundError:
                    errors.append(f"missing {RECOVERY_ATTESTATION_NAME}")
                for reference in recovery.get("bound_artifacts") or []:
                    ok, error = verify_artifact_reference(reference)
                    if not ok:
                        errors.append(
                            f"recovery attestation artifact invalid: {error}"
                        )
                        continue
                    if reference["path"] not in {
                        item["path"] for item in artifacts
                    }:
                        artifacts.append(reference)
        if stage_name == "first_pass":
            preflight = task_dir / "evidence_preflight.json"
            try:
                artifacts.append(artifact_reference(preflight))
            except FileNotFoundError:
                errors.append("missing evidence_preflight.json")
                preflight_value = None
            else:
                try:
                    preflight_value = json.loads(preflight.read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(f"invalid evidence_preflight.json: {exc}")
                    preflight_value = None
            if isinstance(preflight_value, dict):
                if preflight_value.get("ok") is not True:
                    errors.append("evidence preflight did not pass")
                if preflight_value.get("task_id") != task_id:
                    errors.append("evidence preflight task_id mismatch")
                if preflight_value.get("manifest_sha256") != expected_identity.get("manifest_sha256"):
                    errors.append("evidence preflight manifest hash mismatch")
                if preflight_value.get("judge_script_sha256") != expected_identity.get("script_sha256"):
                    errors.append("evidence preflight script hash mismatch")
                if (
                    isinstance(persisted_input, dict)
                    and preflight_value.get("observation") != persisted_input.get("observation")
                ):
                    errors.append("first-pass observation differs from evidence preflight")
                for item in preflight_value.get("evidence") or []:
                    try:
                        evidence_path = Path(item["path"])
                        reference = artifact_reference(evidence_path)
                    except (KeyError, FileNotFoundError, TypeError):
                        errors.append("evidence preflight references a missing frame")
                        continue
                    if (
                        reference["bytes"] != item.get("bytes")
                        or reference["sha256"] != item.get("sha256")
                    ):
                        errors.append(f"evidence frame differs from preflight: {evidence_path}")
                    artifacts.append(reference)

            task_entry = task_entries[task_id]
            evidence_video = task_entry.get("evidence_video") or {}
            try:
                video_reference = artifact_reference(Path(evidence_video["path"]))
            except (KeyError, FileNotFoundError, TypeError):
                errors.append("manifest evidence video is missing")
            else:
                if (
                    video_reference["bytes"] != evidence_video.get("bytes")
                    or video_reference["sha256"] != evidence_video.get("sha256")
                ):
                    errors.append("manifest evidence video hash or byte count mismatch")
                artifacts.append(video_reference)
            summary_path = (task_entry.get("timing_source") or {}).get("summary_json")
            if summary_path:
                try:
                    summary_reference = artifact_reference(Path(summary_path))
                    artifacts.append(summary_reference)
                    summary_value = json.loads(Path(summary_path).read_text(encoding="utf-8"))
                    source_path = ((summary_value.get("task") or {}).get("local_video_path"))
                    if source_path:
                        artifacts.append(artifact_reference(Path(source_path)))
                except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
                    errors.append(f"timing/source artifact is invalid: {exc}")
            run_summary = (task_entry.get("capture_provenance") or {}).get("run_summary")
            if run_summary:
                try:
                    artifacts.append(artifact_reference(Path(run_summary)))
                except FileNotFoundError:
                    errors.append("capture run summary is missing")
        upstream_paths = []
        first_path = prediction.get("first_pass_prediction")
        if first_path:
            upstream_paths.append((Path(first_path), prediction.get("first_pass_prediction_sha256")))
        for item in prediction.get("candidate_provenance") or []:
            if isinstance(item, dict) and item.get("path"):
                upstream_paths.append((Path(item["path"]), item.get("sha256")))
        for upstream, recorded_hash in upstream_paths:
            if not upstream.is_file() or sha256(upstream) != recorded_hash:
                errors.append(f"upstream prediction mismatch: {upstream}")
                continue
            artifacts.append(artifact_reference(upstream))

        if errors:
            failed = {
                "schema_version": 1,
                "pass": False,
                "stage": stage_name,
                "task_id": task_id,
                "errors": errors,
                "checked_at": now(),
            }
            write_json(execution_attestation_path(directory, task_id), failed)
            continue
        attestation = {
            "schema_version": 1,
            "pass": True,
            "stage": stage_name,
            "task_id": task_id,
            "stage_identity": expected_identity,
            "artifacts": artifacts,
            "attempt_ledger": {
                "path": str(Path(ledger_path).resolve()),
                "identity_sha256": ledger_identity_sha256(
                    load_attempt_ledger(ledger_path)
                ),
                "attempt_count": len(attempts),
                "attempts_sha256": json_sha256(attempts),
            },
            "checked_at": now(),
        }
        if isinstance(recovery, dict):
            attestation["operational_recovery"] = {
                "mode": recovery.get("mode"),
                "attestation": artifact_reference(recovery_path),
            }
        attestation[EXECUTION_ATTESTATION_INTEGRITY_FIELD] = (
            execution_attestation_integrity(attestation)
        )
        write_json(execution_attestation_path(directory, task_id), attestation)


def load_prediction(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def phase_status(
    task_ids, directory, *, expected_identity=None, ledger_path=None,
    task_specs=None,
):
    ok = []
    failed = []
    missing = []
    errors = {}
    for task_id in task_ids:
        path = directory / task_id / "prediction.json"
        if not path.is_file():
            missing.append(task_id)
            continue
        prediction = load_prediction(path)
        if prediction and prediction.get("ok") is True:
            if expected_identity is None:
                ok.append(task_id)
                continue
            verified, error = verify_execution_attestation(
                task_id=task_id,
                directory=directory,
                expected_identity=expected_identity,
                ledger_path=ledger_path,
                expected_task_spec=(task_specs or {}).get(task_id),
            )
            if verified:
                ok.append(task_id)
            else:
                if execution_attestation_path(directory, task_id).is_file():
                    raise RuntimeError(
                        f"Refusing to overwrite invalid execution attestation for "
                        f"{expected_identity.get('stage')}/{task_id}: {error}"
                    )
                failed.append(task_id)
                errors[task_id] = error
        else:
            failed.append(task_id)
            errors[task_id] = (prediction or {}).get("error") or "invalid prediction.json"
    return {
        "ok": ok,
        "failed": failed,
        "missing": missing,
        "errors": errors,
    }


def run_attempt(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] COMMAND {json.dumps(command, ensure_ascii=False)}\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"[{now()}] EXIT {process.returncode}\n")
        return process.returncode


def authentication_failures(errors, task_ids):
    markers = (
        "http 401",
        "unauthenticated",
        "api key无效",
        "invalid api key",
        "api key invalid",
    )
    return [
        task_id
        for task_id in task_ids
        if any(
            marker in str((errors or {}).get(task_id) or "").lower()
            for marker in markers
        )
    ]


def supervise_phase(
    *, name, script, prompt, manifest, task_ids, out_dir, first_pass_dir,
    max_attempts, gap_s, state, state_path, log_path, extra_args=None,
    task_specs=None, task_entries=None, expected_identity=None, request_wrapper=None,
    attempt_ledger=None, max_http_attempts_per_task=5, module_aliases=None,
    task_batch_size=None,
):
    if task_batch_size is not None and task_batch_size < 1:
        raise ValueError("task_batch_size must be positive when set")
    if attempt_ledger is not None and Path(attempt_ledger).is_file():
        load_attempt_ledger(attempt_ledger, {
            "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
            "stage": name,
            "max_attempts_per_task": max_http_attempts_per_task,
            "manifest_sha256": sha256(manifest),
            "task_ids_sha256": json_sha256(sorted(task_ids)),
            "task_count": len(task_ids),
            "stage_identity_sha256": json_sha256(expected_identity),
        })
    phase = state["phases"].setdefault(name, {
        "status": "pending",
        "attempts": 0,
        "attempts_completed": 0,
        "started_at": None,
        "finished_at": None,
    })
    phase["started_at"] = phase.get("started_at") or now()
    if "attempts_completed" in phase:
        attempts_used = int(phase.get("attempts_completed") or 0)
    else:
        recorded_attempts = int(phase.get("attempts") or 0)
        attempts_used = (
            max(0, recorded_attempts - 1)
            if phase.get("status") == "running"
            else recorded_attempts
        )
    for attempt in range(attempts_used + 1, max_attempts + 1):
        status = phase_status(
            task_ids,
            out_dir,
            expected_identity=expected_identity,
            ledger_path=attempt_ledger,
            task_specs=task_specs,
        )
        remaining = status["failed"] + status["missing"]
        if not remaining:
            phase.update({
                "status": "complete",
                "finished_at": now(),
                "ok": len(status["ok"]),
                "failed": 0,
                "missing": 0,
            })
            write_json(state_path, state)
            return

        phase.update({
            "status": "running",
            "attempts": attempt,
            "current_attempt": attempt,
            "remaining_task_ids": remaining,
            "ok": len(status["ok"]),
            "failed": len(status["failed"]),
            "missing": len(status["missing"]),
            "errors": status["errors"],
            "updated_at": now(),
        })
        state["current_phase"] = name
        state["updated_at"] = now()
        write_json(state_path, state)

        effective_batch_size = task_batch_size or len(remaining)
        batches = [
            remaining[offset:offset + effective_batch_size]
            for offset in range(0, len(remaining), effective_batch_size)
        ]
        return_codes = []
        for batch_index, batch_task_ids in enumerate(batches, start=1):
            phase.update({
                "current_batch": batch_index,
                "batch_count": len(batches),
                "current_task_ids": batch_task_ids,
                "updated_at": now(),
            })
            state["updated_at"] = now()
            write_json(state_path, state)

            target_args = [
                "--manifest",
                str(manifest),
                "--system-prompt",
                str(prompt),
                "--out-dir",
                str(out_dir),
                "--gap-s",
                str(gap_s),
            ]
            if first_pass_dir is not None:
                target_args.extend(["--first-pass-dir", str(first_pass_dir)])
            target_args.extend(extra_args or [])
            target_args.append("--force")
            for task_id in batch_task_ids:
                target_args.extend(["--task-id", task_id])
            if request_wrapper is not None:
                if expected_identity is None or attempt_ledger is None:
                    raise RuntimeError("Request budgeting requires stage identity and ledger")
                command = [
                    sys.executable,
                    str(request_wrapper),
                    "--target", str(script),
                    "--stage", name,
                    "--ledger", str(attempt_ledger),
                    "--manifest", str(manifest),
                    "--stage-identity-sha256", json_sha256(expected_identity),
                    "--max-total-attempts", str(max_http_attempts_per_task),
                ]
                for module_name, module_path in sorted((module_aliases or {}).items()):
                    command.extend([
                        "--module-alias", f"{module_name}={Path(module_path).resolve()}"
                    ])
                command.extend(["--", *target_args])
            else:
                command = [sys.executable, str(script), *target_args]
            return_code = run_attempt(command, log_path)
            return_codes.append(return_code)
            if expected_identity is not None:
                attest_successful_predictions(
                    stage_name=name,
                    task_ids=batch_task_ids,
                    task_specs=task_specs,
                    task_entries=task_entries,
                    directory=out_dir,
                    expected_identity=expected_identity,
                    ledger_path=attempt_ledger,
                )

            batch_status = phase_status(
                task_ids,
                out_dir,
                expected_identity=expected_identity,
                ledger_path=attempt_ledger,
                task_specs=task_specs,
            )
            phase.update({
                "last_return_code": return_code,
                "ok": len(batch_status["ok"]),
                "failed": len(batch_status["failed"]),
                "missing": len(batch_status["missing"]),
                "remaining_task_ids": (
                    batch_status["failed"] + batch_status["missing"]
                ),
                "errors": batch_status["errors"],
                "updated_at": now(),
            })
            state["updated_at"] = now()
            write_json(state_path, state)

            auth_failed = authentication_failures(
                batch_status["errors"], batch_task_ids
            )
            if auth_failed:
                phase.update({
                    "status": "failed",
                    "finished_at": now(),
                    "authentication_failed_task_ids": auth_failed,
                })
                write_json(state_path, state)
                raise RuntimeError(
                    f"{name} stopped after an authentication failure: {auth_failed}"
                )

            exhausted = [
                task_id for task_id in batch_task_ids
                if task_id not in batch_status["ok"]
                and len(ledger_task_attempts(attempt_ledger, task_id))
                >= max_http_attempts_per_task
            ] if attempt_ledger is not None else []
            if exhausted:
                phase.update({
                    "status": "failed",
                    "finished_at": now(),
                    "exhausted_task_ids": exhausted,
                })
                write_json(state_path, state)
                raise RuntimeError(
                    f"{name} exhausted the per-task HTTP budget: {exhausted}"
                )
            if batch_index < len(batches) and gap_s > 0:
                time.sleep(gap_s)

        phase["last_return_codes"] = return_codes
        phase["attempts_completed"] = attempt
        phase["updated_at"] = now()
        write_json(state_path, state)
        post_status = phase_status(
            task_ids,
            out_dir,
            expected_identity=expected_identity,
            ledger_path=attempt_ledger,
            task_specs=task_specs,
        )
        post_remaining = post_status["failed"] + post_status["missing"]
        exhausted = [
            task_id for task_id in post_remaining
            if len(ledger_task_attempts(attempt_ledger, task_id))
            >= max_http_attempts_per_task
        ] if attempt_ledger is not None else []
        phase["exhausted_task_ids"] = exhausted
        if exhausted:
            phase.update({
                "status": "failed",
                "finished_at": now(),
                "errors": post_status["errors"],
                "remaining_task_ids": post_remaining,
            })
            write_json(state_path, state)
            raise RuntimeError(
                f"{name} exhausted the per-task HTTP budget: {exhausted}"
            )
        if attempt < max_attempts:
            time.sleep(min(300, 15 * (2 ** (attempt - 1))))

    status = phase_status(
        task_ids,
        out_dir,
        expected_identity=expected_identity,
        ledger_path=attempt_ledger,
        task_specs=task_specs,
    )
    remaining = status["failed"] + status["missing"]
    if remaining:
        phase.update({
            "status": "failed",
            "finished_at": now(),
            "ok": len(status["ok"]),
            "failed": len(status["failed"]),
            "missing": len(status["missing"]),
            "remaining_task_ids": remaining,
            "errors": status["errors"],
        })
        write_json(state_path, state)
        raise RuntimeError(f"{name} incomplete after {max_attempts} attempts: {remaining}")
    phase.update({
        "status": "complete",
        "finished_at": now(),
        "ok": len(status["ok"]),
        "failed": 0,
        "missing": 0,
    })
    write_json(state_path, state)


def aggregate_results(task_ids, review_dir, task_specs=None):
    tasks = []
    for task_id in task_ids:
        prediction = load_prediction(review_dir / task_id / "prediction.json")
        judge_json = (prediction or {}).get("judge_json") or {}
        if task_specs is not None:
            normalized = copy.deepcopy(judge_json)
            recomputed = apply_dimension_mean_score(normalized, task_specs[task_id])
            reported = (prediction or {}).get("dimension_mean_score")
            if recomputed is None or not isinstance(reported, (int, float)):
                raise RuntimeError(f"Cannot recompute score for {task_id}")
            if abs(float(reported) - recomputed) > 1e-12:
                raise RuntimeError(
                    f"Reported dimension mean differs from grades for {task_id}"
                )
            nested_reported = judge_json.get("dimension_mean_score")
            if (
                nested_reported is not None
                and (
                    not isinstance(nested_reported, (int, float))
                    or abs(float(nested_reported) - recomputed) > 1e-12
                )
            ):
                raise RuntimeError(
                    f"Nested dimension mean differs from grades for {task_id}"
                )
            for field in ("benchmark_score", "score_without_d3"):
                expected = normalized.get(field)
                reported = (prediction or {}).get(field)
                nested_reported = judge_json.get(field)
                if (
                    expected is None
                    or not isinstance(reported, (int, float))
                    or abs(float(reported) - expected) > 1e-12
                    or (
                        nested_reported is not None
                        and (
                            not isinstance(nested_reported, (int, float))
                            or abs(float(nested_reported) - expected) > 1e-12
                        )
                    )
                ):
                    raise RuntimeError(
                        f"Reported {field} differs from grades for {task_id}"
                    )
        dimensions = judge_json.get("dimensions") or {}
        grades = {
            name: item.get("grade")
            for name, item in dimensions.items()
            if item.get("grade") in {"G", "S", "B"}
        }
        tasks.append({
            "task_id": task_id,
            "ok": bool((prediction or {}).get("ok")),
            "dimension_mean_score": (prediction or {}).get("dimension_mean_score"),
            "benchmark_score": (prediction or {}).get("benchmark_score"),
            "score_without_d3": (prediction or {}).get("score_without_d3"),
            "grades": grades,
            "prediction": str((review_dir / task_id / "prediction.json").resolve()),
        })
    scores = [item["dimension_mean_score"] for item in tasks if item["dimension_mean_score"] is not None]
    benchmark_scores = [
        item["benchmark_score"] for item in tasks
        if item["benchmark_score"] is not None
    ]
    without_d3_scores = [
        item["score_without_d3"] for item in tasks
        if item["score_without_d3"] is not None
    ]
    return {
        "tasks": tasks,
        "task_count": len(tasks),
        "ok": sum(item["ok"] for item in tasks),
        "failed": sum(not item["ok"] for item in tasks),
        "dimension_mean_score_sum": sum(scores),
        "dimension_mean_score_mean": sum(scores) / len(scores) if scores else None,
        "overall_score": (
            100.0 * sum(benchmark_scores) / len(benchmark_scores)
            if benchmark_scores else None
        ),
        "score_without_d3": (
            100.0 * sum(without_d3_scores) / len(without_d3_scores)
            if without_d3_scores else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--first-script", type=Path, default=DEFAULT_FIRST_SCRIPT)
    parser.add_argument("--first-prompt", type=Path, default=DEFAULT_FIRST_PROMPT)
    parser.add_argument("--review-script", type=Path, default=DEFAULT_REVIEW_SCRIPT)
    parser.add_argument("--review-prompt", type=Path, default=DEFAULT_REVIEW_PROMPT)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--gap-s", type=float, default=3.0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("contains_human_labels") is not False:
        raise RuntimeError("Manifest is not explicitly label-free")
    task_ids = [item["task_spec"]["id"] for item in manifest["tasks"]]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("Manifest contains duplicate task ids")

    args.out_root.mkdir(parents=True, exist_ok=True)
    first_dir = args.out_root / "first_pass"
    review_dir = args.out_root / "final_review"
    state_path = args.out_root / "pipeline_state.json"
    log_path = args.out_root / "pipeline.log"
    state = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "updated_at": now(),
        "current_phase": None,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "capture_model_id": manifest.get("capture_model_id"),
        "task_count": len(task_ids),
        "contains_human_labels": False,
        "judge_pipeline": {
            "first_script": str(args.first_script.resolve()),
            "first_script_sha256": sha256(args.first_script),
            "first_prompt": str(args.first_prompt.resolve()),
            "first_prompt_sha256": sha256(args.first_prompt),
            "review_script": str(args.review_script.resolve()),
            "review_script_sha256": sha256(args.review_script),
            "review_prompt": str(args.review_prompt.resolve()),
            "review_prompt_sha256": sha256(args.review_prompt),
        },
        "phases": {},
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("manifest_sha256") != state["manifest_sha256"]:
            raise RuntimeError("Existing pipeline state belongs to a different manifest")
        state["started_at"] = previous.get("started_at") or state["started_at"]
        state["phases"] = previous.get("phases") or {}

    try:
        supervise_phase(
            name="first_pass",
            script=args.first_script,
            prompt=args.first_prompt,
            manifest=args.manifest,
            task_ids=task_ids,
            out_dir=first_dir,
            first_pass_dir=None,
            max_attempts=args.max_attempts,
            gap_s=args.gap_s,
            state=state,
            state_path=state_path,
            log_path=log_path,
        )
        supervise_phase(
            name="final_review",
            script=args.review_script,
            prompt=args.review_prompt,
            manifest=args.manifest,
            task_ids=task_ids,
            out_dir=review_dir,
            first_pass_dir=first_dir,
            max_attempts=args.max_attempts,
            gap_s=args.gap_s,
            state=state,
            state_path=state_path,
            log_path=log_path,
        )
        aggregate = aggregate_results(task_ids, review_dir)
        write_json(args.out_root / "judge_results_summary.json", {
            "capture_model_id": manifest.get("capture_model_id"),
            "manifest": str(args.manifest.resolve()),
            "judge_pipeline": state["judge_pipeline"],
            **aggregate,
        })
        state.update({
            "status": "complete",
            "current_phase": None,
            "updated_at": now(),
            "finished_at": now(),
            "result_summary": str((args.out_root / "judge_results_summary.json").resolve()),
        })
        write_json(state_path, state)
    except Exception as exc:
        state.update({
            "status": "failed",
            "updated_at": now(),
            "finished_at": now(),
            "error": repr(exc),
        })
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
