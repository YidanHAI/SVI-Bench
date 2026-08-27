#!/usr/bin/env python3
"""Audit one full-data Judge stage for blindness and execution provenance."""

import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

from judge_runtime import (
    json_sha256,
    load_attempt_ledger,
    load_prediction,
    sha256,
    verify_execution_attestation,
)
from judge_scoring import apply_dimension_mean_score
from run_judge_with_attempt_budget import (
    SCHEMA_VERSION as ATTEMPT_LEDGER_SCHEMA_VERSION,
    ledger_seal_path,
)


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SOURCE_MARKERS = (
    "人工打分表",
    "human_alignment_report",
    "gold_labels.json",
    "expert_scores.xlsx",
    "VL·D1",
    "VL·D2",
    "VL·D3",
    "VL·D4",
    "VL·D5",
)
FORBIDDEN_KEYS = {
    "gold",
    "gold_label",
    "ground_truth",
    "human_label",
    "expert_label",
    "expert_score",
}
WORKBOOK_READERS = {"load_workbook", "read_excel", "ExcelFile"}
EXPECTED_INPUT_KEYS = {
    "first_pass": {"task_spec", "observation"},
    "review_1": {
        "task_spec", "observation", "first_pass_judgment", "first_pass_provenance",
    },
    "review_2": {
        "task_spec", "observation", "first_pass_judgment", "first_pass_provenance",
    },
    "adjudication": {"task_spec", "observation", "candidate_judgments"},
    "final_review": {
        "task_spec", "observation", "first_pass_judgment", "first_pass_provenance",
    },
}
INPUT_FILENAMES = {
    "first_pass": "judge_input.json",
    "review_1": "review_input.json",
    "review_2": "review_input.json",
    "adjudication": "adjudication_input.json",
    "final_review": "review_input.json",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def walk_json(value, path="$", findings=None):
    findings = findings if findings is not None else []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).strip().lower() in FORBIDDEN_KEYS:
                findings.append({"reason": "forbidden_human_label_key", "path": child_path})
            walk_json(child, child_path, findings)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_json(child, f"{path}[{index}]", findings)
    elif isinstance(value, str):
        for marker in FORBIDDEN_SOURCE_MARKERS:
            if marker in value:
                findings.append({
                    "reason": "forbidden_human_source_reference",
                    "path": path,
                    "marker": marker,
                })
    return findings


def local_imports(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError):
        return []
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        for name in names:
            relative = Path(*name.split("."))
            for candidate in (
                path.parent / f"{relative}.py",
                ROOT / "scripts" / f"{relative.name}.py",
            ):
                if candidate.is_file():
                    found.append(candidate.resolve())
                    break
    return found


def dependency_closure(paths):
    pending = [Path(item).resolve() for item in paths]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        if path.suffix == ".py":
            pending.extend(item for item in local_imports(path) if item not in seen)
    return sorted(seen, key=str)


def audit_sources(paths, prompt, task_ids):
    findings = []
    source_paths = dependency_closure(paths)
    for path in [*source_paths, prompt.resolve()]:
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_SOURCE_MARKERS:
            if marker in text:
                findings.append({
                    "reason": "human_source_marker",
                    "source": str(path),
                    "marker": marker,
                })
        literal_ids = sorted(
            task_id for task_id in task_ids
            if re.search(
                rf"(?<![A-Z0-9]){re.escape(task_id)}(?![A-Z0-9])", text
            )
        )
        if literal_ids:
            findings.append({
                "reason": "literal_task_ids",
                "source": str(path),
                "task_ids": literal_ids,
            })
        if path.suffix != ".py":
            continue
        tree = ast.parse(text, filename=str(path))
        readers = sorted({
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name) and node.func.id in WORKBOOK_READERS
                or isinstance(node.func, ast.Attribute) and node.func.attr in WORKBOOK_READERS
            )
        })
        if readers:
            findings.append({
                "reason": "human_workbook_reader_in_execution_path",
                "source": str(path),
                "calls": readers,
            })
    references = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in source_paths
    ]
    return findings, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pipeline-state", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(EXPECTED_INPUT_KEYS), required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--dependency", type=Path, action="append", default=[])
    parser.add_argument("--expected-tasks", type=int, default=75)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    state = load_json(args.pipeline_state)
    findings = []
    if manifest.get("contains_human_labels") is not False:
        findings.append({"reason": "manifest_not_explicitly_label_free"})
    tasks = {item["task_spec"]["id"]: item for item in manifest.get("tasks") or []}
    if len(tasks) != args.expected_tasks:
        findings.append({
            "reason": "unexpected_manifest_task_count",
            "expected": args.expected_tasks,
            "actual": len(tasks),
        })
    stage_state = ((state.get("stages") or {}).get(args.stage) or {})
    identity = stage_state.get("execution_identity")
    if not isinstance(identity, dict):
        findings.append({"reason": "missing_stage_execution_identity"})
        identity = {}
    if identity.get("script_sha256") != sha256(args.script):
        findings.append({"reason": "stage_script_hash_mismatch"})
    if identity.get("prompt_sha256") != sha256(args.system_prompt):
        findings.append({"reason": "stage_prompt_hash_mismatch"})
    prompt_content_hash = hashlib.sha256(
        args.system_prompt.read_text(encoding="utf-8").strip().encode("utf-8")
    ).hexdigest()
    if identity.get("prompt_content_sha256") != prompt_content_hash:
        findings.append({"reason": "stage_prompt_content_hash_mismatch"})
    if identity.get("manifest_sha256") != sha256(args.manifest):
        findings.append({"reason": "stage_manifest_hash_mismatch"})

    expected_dependencies = list(stage_state.get("dependencies") or [])
    request_runner = stage_state.get("request_budget_runner") or {}
    if request_runner:
        expected_dependencies.append(request_runner)
    expected_dependency_map = {
        str(Path(item.get("path") or "").resolve()): item.get("sha256")
        for item in expected_dependencies
    }
    actual_dependency_map = {
        str(path.resolve()): sha256(path) for path in args.dependency
    }
    if actual_dependency_map != expected_dependency_map:
        findings.append({
            "reason": "stage_direct_dependency_mismatch",
            "expected": expected_dependency_map,
            "actual": actual_dependency_map,
        })
    if identity.get("dependencies") != stage_state.get("dependencies"):
        findings.append({"reason": "stage_identity_dependency_mismatch"})
    if identity.get("request_budget_runner_sha256") != request_runner.get("sha256"):
        findings.append({"reason": "request_budget_runner_hash_mismatch"})
    module_aliases = stage_state.get("module_aliases") or {}
    if identity.get("module_aliases") != module_aliases:
        findings.append({"reason": "stage_module_alias_mismatch"})
    alias_paths = []
    for module_name, item in module_aliases.items():
        path = Path(str(item.get("path") or ""))
        if not path.is_file() or sha256(path) != item.get("sha256"):
            findings.append({
                "reason": "stage_module_alias_artifact_mismatch",
                "module": module_name,
            })
        else:
            alias_paths.append(path)

    source_findings, source_references = audit_sources(
        [args.script, *args.dependency, *alias_paths],
        args.system_prompt,
        set(tasks),
    )
    findings.extend(source_findings)
    try:
        ledger = load_attempt_ledger(args.attempt_ledger, {
            "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
            "stage": args.stage,
            "manifest_sha256": sha256(args.manifest),
            "task_ids_sha256": json_sha256(sorted(tasks)),
            "task_count": len(tasks),
            "stage_identity_sha256": json_sha256(identity),
            "max_attempts_per_task": identity.get(
                "max_http_attempts_per_task"
            ),
        })
        if ledger is None:
            raise RuntimeError("attempt ledger is missing")
    except RuntimeError as exc:
        findings.append({"reason": "invalid_attempt_ledger", "detail": str(exc)})
        ledger = None

    prediction_scope = {
        path.parent.name for path in args.stage_dir.glob("*/prediction.json")
    }
    input_scope = {
        path.parent.name
        for path in args.stage_dir.glob(f"*/{INPUT_FILENAMES[args.stage]}")
    }
    expected_scope = set(tasks)
    if prediction_scope != expected_scope:
        findings.append({
            "reason": "prediction_scope_mismatch",
            "missing": sorted(expected_scope - prediction_scope),
            "extra": sorted(prediction_scope - expected_scope),
        })
    if input_scope != expected_scope:
        findings.append({
            "reason": "persisted_input_scope_mismatch",
            "missing": sorted(expected_scope - input_scope),
            "extra": sorted(input_scope - expected_scope),
        })

    checked_inputs = 0
    checked_attestations = 0
    checked_attempt_ledgers = 0
    for task_id, task in sorted(tasks.items()):
        prediction_path = args.stage_dir / task_id / "prediction.json"
        prediction = load_prediction(prediction_path)
        if not prediction or prediction.get("ok") is not True:
            findings.append({"reason": "missing_or_failed_prediction", "task_id": task_id})
            continue
        if prediction.get("task_id") != task_id or (prediction.get("judge_json") or {}).get("task_id") != task_id:
            findings.append({"reason": "prediction_task_id_mismatch", "task_id": task_id})
        if prediction.get("model") != identity.get("requested_model"):
            findings.append({"reason": "prediction_model_mismatch", "task_id": task_id})
        if prediction.get("api_url") != identity.get("api_url"):
            findings.append({"reason": "prediction_api_url_mismatch", "task_id": task_id})
        if prediction.get("native_video_input") != identity.get("native_video_input"):
            findings.append({"reason": "prediction_modality_mismatch", "task_id": task_id})
        if prediction.get("input_mode") != identity.get("input_mode"):
            findings.append({"reason": "prediction_input_mode_mismatch", "task_id": task_id})
        if prediction.get("prompt_sha256") != prompt_content_hash:
            findings.append({"reason": "prediction_prompt_mismatch", "task_id": task_id})
        normalized = copy.deepcopy(prediction.get("judge_json") or {})
        recomputed_score = apply_dimension_mean_score(
            normalized, task.get("task_spec") or {}
        )
        reported_score = prediction.get("dimension_mean_score")
        if (
            recomputed_score is None
            or not isinstance(reported_score, (int, float))
            or abs(float(reported_score) - recomputed_score) > 1e-12
        ):
            findings.append({"reason": "dimension_mean_score_mismatch", "task_id": task_id})
        for field in ("benchmark_score", "score_without_d3"):
            expected = normalized.get(field)
            reported = prediction.get(field)
            if (
                expected is None
                or not isinstance(reported, (int, float))
                or abs(float(reported) - expected) > 1e-12
            ):
                findings.append({"reason": f"{field}_mismatch", "task_id": task_id})
        try:
            verified, error = verify_execution_attestation(
                task_id=task_id,
                directory=args.stage_dir,
                expected_identity=identity,
                ledger_path=args.attempt_ledger,
            )
        except RuntimeError as exc:
            verified, error = False, str(exc)
        if not verified:
            findings.append({
                "reason": "execution_attestation_failed",
                "task_id": task_id,
                "detail": error,
            })
        else:
            checked_attestations += 1
        input_path = args.stage_dir / task_id / INPUT_FILENAMES[args.stage]
        if not input_path.is_file():
            findings.append({"reason": "missing_persisted_input", "task_id": task_id})
        else:
            value = load_json(input_path)
            checked_inputs += 1
            if set(value) != EXPECTED_INPUT_KEYS[args.stage]:
                findings.append({
                    "reason": "unexpected_persisted_input_shape",
                    "task_id": task_id,
                    "keys": sorted(value),
                })
            if value.get("task_spec") != task.get("task_spec"):
                findings.append({"reason": "task_spec_differs_from_manifest", "task_id": task_id})
            findings.extend({"task_id": task_id, **item} for item in walk_json(value))
        attempts = (
            ((((ledger or {}).get("tasks") or {}).get(task_id) or {})
             .get("attempts") or [])
        )
        limit = identity.get("max_http_attempts_per_task")
        valid_attempts = (
            isinstance(limit, int)
            and 1 <= len(attempts) <= limit
            and [item.get("attempt") for item in attempts] == list(range(1, len(attempts) + 1))
            and all(item.get("request_sha256") for item in attempts)
            and attempts[-1].get("status") == "response"
            and attempts[-1].get("status_code") == 200
        )
        if not valid_attempts:
            findings.append({"reason": "invalid_attempt_history", "task_id": task_id})
        else:
            checked_attempt_ledgers += 1

    report = {
        "schema_version": 1,
        "audit": "full_judge_stage_blindness_and_provenance",
        "stage": args.stage,
        "status": "pass" if not findings else "fail",
        "scope": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "pipeline_state": str(args.pipeline_state.resolve()),
            "pipeline_state_sha256": sha256(args.pipeline_state),
            "stage_dir": str(args.stage_dir.resolve()),
            "script": str(args.script.resolve()),
            "script_sha256": sha256(args.script),
            "system_prompt": str(args.system_prompt.resolve()),
            "system_prompt_sha256": sha256(args.system_prompt),
            "attempt_ledger": str(args.attempt_ledger.resolve()),
            "attempt_ledger_sha256": (
                sha256(args.attempt_ledger) if args.attempt_ledger.is_file() else None
            ),
            "attempt_ledger_seal": str(
                ledger_seal_path(args.attempt_ledger).resolve()
            ),
            "attempt_ledger_seal_sha256": (
                sha256(ledger_seal_path(args.attempt_ledger))
                if ledger_seal_path(args.attempt_ledger).is_file() else None
            ),
            "source_dependency_closure": source_references,
        },
        "summary": {
            "expected_tasks": args.expected_tasks,
            "manifest_tasks": len(tasks),
            "persisted_inputs_checked": checked_inputs,
            "execution_attestations_checked": checked_attestations,
            "attempt_histories_checked": checked_attempt_ledgers,
            "prediction_scope_exact": prediction_scope == expected_scope,
            "persisted_input_scope_exact": input_scope == expected_scope,
            "contains_human_labels": False if not findings else None,
            "contains_literal_task_special_cases": False if not findings else None,
        },
        "findings": findings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not findings else 1)


if __name__ == "__main__":
    main()
