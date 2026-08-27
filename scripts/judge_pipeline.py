#!/usr/bin/env python3
"""Run the release five-stage Judge on one label-free 75-task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from judge_api import DEFAULT_JUDGE_MODEL, chat_completions_url
from judge_workbook_contract import (
    DIMENSION_IDS,
    RUBRIC_VERSION,
    canonical_sha256,
    sha256,
)
from judge_runtime import (
    ROOT,
    aggregate_results,
    now,
    supervise_phase,
    write_json,
)


FIRST_SCRIPT = ROOT / "scripts" / "judge_first_pass.py"
REVIEW_SCRIPT = ROOT / "scripts" / "judge_review.py"
ADJUDICATION_SCRIPT = ROOT / "scripts" / "judge_adjudication.py"
REQUEST_BUDGET_RUNNER = ROOT / "scripts" / "run_judge_with_attempt_budget.py"

FIRST_PROMPT = ROOT / "prompts" / "judge_first_pass.md"
REVIEW_1_PROMPT = ROOT / "prompts" / "judge_review_1.md"
REVIEW_2_PROMPT = ROOT / "prompts" / "judge_review_2.md"
ADJUDICATION_PROMPT = ROOT / "prompts" / "judge_adjudication.md"
FINAL_REVIEW_PROMPT = ROOT / "prompts" / "judge_final_review.md"

STAGE_INPUT_MODES = {
    "first_pass": "timestamped_images_temporal_sheets_plus_capture_timeline",
    "review_1": "first_pass_plus_same_timestamped_evidence",
    "review_2": "first_pass_plus_same_timestamped_evidence",
    "adjudication": "candidate_judgments_plus_same_timestamped_evidence",
    "final_review": "first_pass_plus_same_timestamped_evidence",
}


def load_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def resolve_bound_path(raw: object, manifest_path: Path) -> Path:
    path = Path(str(raw or ""))
    if path.is_absolute():
        return path.resolve()
    candidates = [
        (manifest_path.parent / path).resolve(),
        (ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_manifest(path: Path) -> dict:
    manifest = load_json(path, "Judge manifest")
    if manifest.get("schema_version") != 2:
        raise RuntimeError("Judge manifest must use schema_version=2")
    if manifest.get("rubric_version") != RUBRIC_VERSION:
        raise RuntimeError(f"Judge manifest must use rubric_version={RUBRIC_VERSION!r}")
    if manifest.get("contains_human_labels") is not False:
        raise RuntimeError("Judge manifest must be explicitly label-free")

    contract = manifest.get("rubric_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Judge manifest has no rubric_contract object")
    contract_hash = canonical_sha256(contract)
    if manifest.get("rubric_contract_sha256") != contract_hash:
        raise RuntimeError("Judge manifest rubric contract hash is invalid")
    if contract.get("version") != RUBRIC_VERSION:
        raise RuntimeError("Judge manifest embeds a different rubric version")
    dimensions = contract.get("dimensions") or {}
    if tuple(dimensions) != DIMENSION_IDS:
        raise RuntimeError("Rubric contract must define D1 through D5 in order")
    d3_definition = str((dimensions.get("D3") or {}).get("definition") or "")
    if "只对有效响应计时" not in d3_definition or "无有效响应记 0" not in d3_definition:
        raise RuntimeError("D3 contract lacks the no-valid-response score-0 rule")
    if (dimensions.get("D4") or {}).get("name") != "响应内容正确性":
        raise RuntimeError("D4 contract is not 响应内容正确性")
    rules = contract.get("judge_rules") or {}
    if rules.get("D3_no_valid_response_score") != 0.0:
        raise RuntimeError("D3 no-response rule is invalid")
    if rules.get("D4_scope") != "response_content_correctness":
        raise RuntimeError("D4 scope is invalid")

    validation = manifest.get("rubric_validation") or {}
    blocking = [
        item
        for item in validation.get("issues") or []
        if isinstance(item, dict) and item.get("severity") == "error"
    ]
    if validation.get("formal_ready") is not True or blocking:
        raise RuntimeError("Judge manifest has unresolved rubric/workbook errors")

    workbook = resolve_bound_path(manifest.get("source_workbook"), path)
    expected_workbook_hash = manifest.get("source_workbook_sha256")
    if not workbook.is_file() or not expected_workbook_hash:
        raise RuntimeError("Judge manifest source workbook is missing")
    if sha256(workbook) != expected_workbook_hash:
        raise RuntimeError("Source workbook changed after manifest generation")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 75:
        raise RuntimeError("The release Judge requires exactly 75 tasks")
    if manifest.get("task_count") != len(tasks):
        raise RuntimeError("Judge manifest task_count does not match tasks")
    task_ids = []
    for item in tasks:
        spec = item.get("task_spec") if isinstance(item, dict) else None
        if not isinstance(spec, dict):
            raise RuntimeError("Judge manifest contains a task without task_spec")
        task_id = str(spec.get("id") or "").strip()
        if not task_id:
            raise RuntimeError("Judge manifest contains a task without id")
        task_ids.append(task_id)
        if spec.get("rubric_version") != RUBRIC_VERSION:
            raise RuntimeError(f"Task {task_id} is not bound to workbook-v2")
        if spec.get("rubric_contract_sha256") != contract_hash:
            raise RuntimeError(f"Task {task_id} has a different rubric hash")
        if spec.get("rubric_contract") != contract:
            raise RuntimeError(f"Task {task_id} embeds a different rubric contract")
        if tuple((spec.get("dimensions") or {}).keys()) != DIMENSION_IDS:
            raise RuntimeError(f"Task {task_id} does not define D1 through D5 in order")
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("Judge manifest contains duplicate task ids")
    return manifest


def run_audit(command: list[str], output_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if completed.returncode:
        output_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        raise RuntimeError(f"Audit failed: {output_path}")

    declared_output = None
    if "--out" in command:
        index = command.index("--out")
        if index + 1 < len(command):
            declared_output = Path(command[index + 1]).resolve()
    if declared_output != output_path.resolve():
        output_path.write_text(completed.stdout, encoding="utf-8")
    elif not output_path.is_file():
        raise RuntimeError(f"Audit did not create its declared output: {output_path}")
    if completed.stderr:
        output_path.with_suffix(output_path.suffix + ".stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )


def hashed_reference(path: Path) -> dict:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256(resolved)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--max-http-attempts-per-task", type=int, default=5)
    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=1,
        help="Tasks per subprocess; one gives the safest durable resume behavior",
    )
    parser.add_argument("--gap-s", type=float, default=2.0)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--api-url", default="")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    args.manifest = args.manifest.resolve()
    args.out_root = args.out_root.resolve()
    args.api_url = args.api_url or chat_completions_url()
    manifest = validate_manifest(args.manifest)
    task_ids = [item["task_spec"]["id"] for item in manifest["tasks"]]
    task_specs = {item["task_spec"]["id"]: item["task_spec"] for item in manifest["tasks"]}
    task_entries = {item["task_spec"]["id"]: item for item in manifest["tasks"]}

    if not 1 <= args.max_attempts <= 5:
        raise RuntimeError("--max-attempts must be between 1 and 5")
    if not 1 <= args.max_http_attempts_per_task <= 5:
        raise RuntimeError("--max-http-attempts-per-task must be between 1 and 5")
    if args.task_batch_size != 1:
        raise RuntimeError("The release pipeline requires --task-batch-size=1")

    directories = {
        "first_pass": args.out_root / "01_first_pass",
        "review_1": args.out_root / "02_review_1",
        "review_2": args.out_root / "03_review_2",
        "adjudication": args.out_root / "04_adjudication",
        "final_review": args.out_root / "05_final_review",
    }
    scripts = {
        "first_pass": FIRST_SCRIPT,
        "review_1": REVIEW_SCRIPT,
        "review_2": REVIEW_SCRIPT,
        "adjudication": ADJUDICATION_SCRIPT,
        "final_review": REVIEW_SCRIPT,
    }
    prompts = {
        "first_pass": FIRST_PROMPT,
        "review_1": REVIEW_1_PROMPT,
        "review_2": REVIEW_2_PROMPT,
        "adjudication": ADJUDICATION_PROMPT,
        "final_review": FINAL_REVIEW_PROMPT,
    }
    shared_dependencies = [
        ROOT / "scripts" / "judge_transport.py",
        ROOT / "scripts" / "judge_api.py",
        ROOT / "scripts" / "judge_scoring.py",
    ]
    dependencies = {
        "first_pass": shared_dependencies,
        "review_1": [FIRST_SCRIPT, *shared_dependencies],
        "review_2": [FIRST_SCRIPT, *shared_dependencies],
        "adjudication": [FIRST_SCRIPT, REVIEW_SCRIPT, *shared_dependencies],
        "final_review": [FIRST_SCRIPT, *shared_dependencies],
    }
    stage_names = tuple(directories)
    module_aliases = {name: {} for name in stage_names}
    request_budget_runner = REQUEST_BUDGET_RUNNER.resolve()

    args.out_root.mkdir(parents=True, exist_ok=True)
    state_path = args.out_root / "pipeline_state.json"
    log_path = args.out_root / "pipeline.log"
    state = {
        "schema_version": 1,
        "pipeline": "five_stage_judge",
        "status": "running",
        "started_at": now(),
        "updated_at": now(),
        "current_phase": None,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "capture_model_id": manifest.get("capture_model_id"),
        "task_count": len(task_ids),
        "contains_human_labels": False,
        "judge_stage_count": 5,
        "judge_model": args.judge_model,
        "api_url": args.api_url,
        "execution_strategy": {
            "task_batch_size": 1,
            "attest_after_each_batch": True,
            "judge_stage_count": 5,
        },
        "evidence_module": str(FIRST_SCRIPT.resolve()),
        "evidence_module_sha256": sha256(FIRST_SCRIPT),
        "stages": {},
        "phases": {},
    }
    for name in stage_names:
        prompt_content_hash = hashlib.sha256(
            prompts[name].read_text(encoding="utf-8").strip().encode("utf-8")
        ).hexdigest()
        identity = {
            "stage": name,
            "manifest_sha256": state["manifest_sha256"],
            "script_sha256": sha256(scripts[name]),
            "prompt_sha256": sha256(prompts[name]),
            "prompt_content_sha256": prompt_content_hash,
            "dependencies": [hashed_reference(path) for path in dependencies[name]],
            "request_budget_runner_sha256": sha256(request_budget_runner),
            "module_aliases": {},
            "requested_model": args.judge_model,
            "api_url": args.api_url,
            "native_video_input": False,
            "input_mode": STAGE_INPUT_MODES[name],
            "max_http_attempts_per_task": args.max_http_attempts_per_task,
        }
        state["stages"][name] = {
            "directory": str(directories[name]),
            "script": str(scripts[name].resolve()),
            "script_sha256": sha256(scripts[name]),
            "prompt": str(prompts[name].resolve()),
            "prompt_sha256": sha256(prompts[name]),
            "prompt_content_sha256": prompt_content_hash,
            "dependencies": identity["dependencies"],
            "request_budget_runner": hashed_reference(request_budget_runner),
            "module_aliases": {},
            "execution_identity": identity,
        }

    if state_path.is_file():
        previous = load_json(state_path, "pipeline state")
        immutable_fields = (
            "schema_version",
            "pipeline",
            "manifest_sha256",
            "capture_model_id",
            "task_count",
            "contains_human_labels",
            "judge_stage_count",
            "judge_model",
            "api_url",
            "execution_strategy",
            "evidence_module_sha256",
        )
        for field in immutable_fields:
            if previous.get(field) != state.get(field):
                raise RuntimeError(f"Existing pipeline state has different {field}")
        previous_stages = previous.get("stages") or {}
        for name, current in state["stages"].items():
            old = previous_stages.get(name) or {}
            for field in (
                "script_sha256",
                "prompt_sha256",
                "prompt_content_sha256",
                "dependencies",
                "execution_identity",
            ):
                if old.get(field) != current.get(field):
                    raise RuntimeError(f"Existing pipeline state has different {name} {field}")
        state["started_at"] = previous.get("started_at") or state["started_at"]
        state["phases"] = previous.get("phases") or {}

    if args.validate_only:
        print(json.dumps({
            "status": "validated",
            "manifest": state["manifest"],
            "manifest_sha256": state["manifest_sha256"],
            "capture_model_id": state["capture_model_id"],
            "task_count": state["task_count"],
            "judge_model": state["judge_model"],
            "judge_stage_count": 5,
            "stages": {
                name: item["execution_identity"]
                for name, item in state["stages"].items()
            },
        }, ensure_ascii=False, indent=2))
        return 0

    common = {
        "manifest": args.manifest,
        "task_ids": task_ids,
        "task_specs": task_specs,
        "task_entries": task_entries,
        "max_attempts": args.max_attempts,
        "gap_s": args.gap_s,
        "state": state,
        "state_path": state_path,
        "log_path": log_path,
        "request_wrapper": request_budget_runner,
        "max_http_attempts_per_task": args.max_http_attempts_per_task,
        "task_batch_size": 1,
    }
    ledgers = {
        name: args.out_root / "attempt_ledgers" / f"{name}.json"
        for name in stage_names
    }
    transport_args = [
        "--model", args.judge_model,
        "--api-url", args.api_url,
        "--retries", str(args.max_http_attempts_per_task),
    ]

    try:
        audit_dir = args.out_root / "audits"
        state["current_phase"] = "evidence_preflight"
        state["updated_at"] = now()
        write_json(state_path, state)
        run_audit([
            sys.executable,
            str(ROOT / "scripts" / "preflight_full_judge_evidence.py"),
            "--manifest", str(args.manifest),
            "--out-dir", str(directories["first_pass"]),
            "--judge-script", str(FIRST_SCRIPT),
        ], audit_dir / "evidence_preflight.log")

        supervise_phase(
            name="first_pass",
            script=FIRST_SCRIPT,
            prompt=FIRST_PROMPT,
            out_dir=directories["first_pass"],
            first_pass_dir=None,
            expected_identity=state["stages"]["first_pass"]["execution_identity"],
            attempt_ledger=ledgers["first_pass"],
            extra_args=transport_args,
            module_aliases=module_aliases["first_pass"],
            **common,
        )
        for name, prompt in (("review_1", REVIEW_1_PROMPT), ("review_2", REVIEW_2_PROMPT)):
            supervise_phase(
                name=name,
                script=REVIEW_SCRIPT,
                prompt=prompt,
                out_dir=directories[name],
                first_pass_dir=directories["first_pass"],
                expected_identity=state["stages"][name]["execution_identity"],
                attempt_ledger=ledgers[name],
                extra_args=transport_args,
                module_aliases=module_aliases[name],
                **common,
            )

        candidates = [
            directories["review_1"],
            directories["first_pass"],
            directories["review_2"],
        ]
        adjudication_args = list(transport_args)
        for directory in candidates:
            adjudication_args.extend(["--candidate-dir", str(directory)])
        supervise_phase(
            name="adjudication",
            script=ADJUDICATION_SCRIPT,
            prompt=ADJUDICATION_PROMPT,
            out_dir=directories["adjudication"],
            first_pass_dir=None,
            expected_identity=state["stages"]["adjudication"]["execution_identity"],
            attempt_ledger=ledgers["adjudication"],
            extra_args=adjudication_args,
            module_aliases=module_aliases["adjudication"],
            **common,
        )
        supervise_phase(
            name="final_review",
            script=REVIEW_SCRIPT,
            prompt=FINAL_REVIEW_PROMPT,
            out_dir=directories["final_review"],
            first_pass_dir=directories["adjudication"],
            expected_identity=state["stages"]["final_review"]["execution_identity"],
            attempt_ledger=ledgers["final_review"],
            extra_args=transport_args,
            module_aliases=module_aliases["final_review"],
            **common,
        )

        full_stage_audits = {}
        for name in stage_names:
            output = audit_dir / f"{name}_full.json"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "audit_full_judge_stage.py"),
                "--manifest", str(args.manifest),
                "--pipeline-state", str(state_path),
                "--stage", name,
                "--stage-dir", str(directories[name]),
                "--script", str(scripts[name]),
                "--system-prompt", str(prompts[name]),
                "--attempt-ledger", str(ledgers[name]),
                "--expected-tasks", str(len(task_ids)),
                "--out", str(output),
            ]
            for dependency in dependencies[name]:
                command.extend(["--dependency", str(dependency)])
            command.extend(["--dependency", str(request_budget_runner)])
            run_audit(command, output)
            full_stage_audits[name] = output

        run_audit([
            sys.executable,
            str(ROOT / "scripts" / "audit_judge_first_pass_blindness.py"),
            "--manifest", str(args.manifest),
            "--judge-script", str(FIRST_SCRIPT),
            "--system-prompt", str(FIRST_PROMPT),
            "--first-pass-input-dir", str(directories["first_pass"]),
            "--expected-tasks", str(len(task_ids)),
        ], audit_dir / "first_pass_blindness.json")
        for name, prompt in (("review_1", REVIEW_1_PROMPT), ("review_2", REVIEW_2_PROMPT)):
            run_audit([
                sys.executable,
                str(ROOT / "scripts" / "audit_judge_review_blindness.py"),
                "--manifest", str(args.manifest),
                "--first-pass-dir", str(directories["first_pass"]),
                "--review-script", str(REVIEW_SCRIPT),
                "--system-prompt", str(prompt),
                "--review-input-dir", str(directories[name]),
                "--expected-tasks", str(len(task_ids)),
            ], audit_dir / f"{name}_blindness.json")

        adjudication_audit = [
            sys.executable,
            str(ROOT / "scripts" / "audit_judge_adjudication_blindness.py"),
            "--manifest", str(args.manifest),
            "--adjudicator-script", str(ADJUDICATION_SCRIPT),
            "--system-prompt", str(ADJUDICATION_PROMPT),
            "--adjudication-input-dir", str(directories["adjudication"]),
            "--expected-tasks", str(len(task_ids)),
            "--out", str(audit_dir / "adjudication_blindness.json"),
        ]
        for directory in candidates:
            adjudication_audit.extend(["--candidate-dir", str(directory)])
        run_audit(adjudication_audit, audit_dir / "adjudication_blindness.log")
        run_audit([
            sys.executable,
            str(ROOT / "scripts" / "audit_judge_review_blindness.py"),
            "--manifest", str(args.manifest),
            "--first-pass-dir", str(directories["adjudication"]),
            "--review-script", str(REVIEW_SCRIPT),
            "--system-prompt", str(FINAL_REVIEW_PROMPT),
            "--review-input-dir", str(directories["final_review"]),
            "--expected-tasks", str(len(task_ids)),
        ], audit_dir / "final_review_blindness.json")

        evidence_attestation = audit_dir / "full_evidence_attestation.json"
        run_audit([
            sys.executable,
            str(ROOT / "scripts" / "attest_full_judge_evidence.py"),
            "--manifest", str(args.manifest),
            "--out-root", str(args.out_root),
            "--evidence-module", str(FIRST_SCRIPT),
            "--out", str(evidence_attestation),
        ], audit_dir / "full_evidence_attestation.log")
        evidence = load_json(evidence_attestation, "full evidence attestation")
        if (
            evidence.get("pass") is not True
            or (evidence.get("summary") or {}).get("tasks_passed") != len(task_ids)
        ):
            raise RuntimeError("Full evidence attestation is incomplete")

        aggregate = aggregate_results(task_ids, directories["final_review"], task_specs)
        result_path = args.out_root / "judge_results_summary.json"
        write_json(result_path, {
            "capture_model_id": manifest.get("capture_model_id"),
            "judge_model": args.judge_model,
            "judge_stage_count": 5,
            "final_stage": "final_review",
            "manifest": str(args.manifest),
            "manifest_sha256": state["manifest_sha256"],
            "evidence_module": state["evidence_module"],
            "evidence_module_sha256": state["evidence_module_sha256"],
            "stage_provenance": state["stages"],
            "audits": {
                **{
                    f"{name}_full": hashed_reference(path)
                    for name, path in full_stage_audits.items()
                },
                "first_pass_blindness": hashed_reference(
                    audit_dir / "first_pass_blindness.json"
                ),
                "review_1_blindness": hashed_reference(
                    audit_dir / "review_1_blindness.json"
                ),
                "review_2_blindness": hashed_reference(
                    audit_dir / "review_2_blindness.json"
                ),
                "adjudication_blindness": hashed_reference(
                    audit_dir / "adjudication_blindness.json"
                ),
                "final_review_blindness": hashed_reference(
                    audit_dir / "final_review_blindness.json"
                ),
                "full_evidence_attestation": hashed_reference(evidence_attestation),
            },
            **aggregate,
        })
        state.update({
            "status": "complete",
            "current_phase": None,
            "updated_at": now(),
            "finished_at": now(),
            "result_summary": str(result_path),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
