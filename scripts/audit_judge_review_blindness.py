#!/usr/bin/env python3
"""Audit review source, transitive local imports, and persisted requests."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_judge_first_pass_blindness import (
    SUPPORTED_COHORTS,
    audit_exact_task_set,
    audit_input_builder,
    audit_prompt,
    audit_source_file,
    audit_user_prompt_boundary,
    canonical_sha256,
    discover_local_dependencies,
    file_set_sha256,
    function_by_name,
    load_json,
    load_manifest_scope,
    scope_base,
    sha256,
    source_files,
    walk_json,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = Path(__file__).resolve()
DEFAULT_MANIFEST = ROOT / "data" / "judge_manifest.json"
DEFAULT_FIRST_PASS = ROOT / "outputs" / "judge" / "01_first_pass"
DEFAULT_REVIEW_SCRIPT = ROOT / "scripts" / "judge_review.py"
DEFAULT_PROMPT = ROOT / "prompts" / "judge_review_1.md"


def assigned_dict_keys(function, variable):
    """Compatibility helper retained for existing audit unit tests."""
    matches = []
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == variable for target in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            matches.append(None)
            continue
        keys = []
        for key in node.value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                keys = None
                break
            keys.append(key.value)
        matches.append(keys)
    return matches


def subscript_string_key(node):
    value = node.slice
    if isinstance(value, ast.Index):  # pragma: no cover - Python 3.8
        value = value.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    if isinstance(value, ast.Str):  # pragma: no cover - Python 3.8
        return value.s
    return None


def audit_review_task_shape(tree, findings):
    """Verify the only request object is blind evidence plus first-pass output."""
    review_task = function_by_name(tree, "review_task")
    if review_task is None:
        findings.append({"reason": "review_task_function_missing"})
        return
    assignments = assigned_dict_keys(review_task, "review_input")
    expected = {"task_spec", "observation", "first_pass_judgment", "first_pass_provenance"}
    if len(assignments) != 1 or set(assignments[0] or []) != expected:
        findings.append({
            "reason": "unexpected_review_input_shape",
            "assignments": assignments,
            "expected_keys": sorted(expected),
        })
    first_reads = [
        node
        for node in ast.walk(review_task)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "first_result"
    ]
    constants = {subscript_string_key(node) for node in first_reads}
    constants.discard(None)
    if "judge_json" not in constants:
        findings.append({"reason": "first_pass_judgment_not_sourced_from_judge_json"})


def audit_review_builder(paths):
    findings = audit_input_builder(
        paths,
        "review_task",
        "review_input",
        {"task_spec", "observation", "first_pass_judgment", "first_pass_provenance"},
    )
    implementations = []
    for path in paths:
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            continue
        if function_by_name(tree, "review_task") is not None:
            implementations.append(tree)
    if len(implementations) == 1:
        shape_findings = []
        audit_review_task_shape(implementations[0], shape_findings)
        # The generic builder already reports the same assignment-shape error.
        findings.extend(
            item for item in shape_findings
            if item.get("reason") != "unexpected_review_input_shape"
        )
    findings.extend(audit_user_prompt_boundary(
        paths, "review_task", "review_input"
    ))
    return findings


def audit_first_pass_predictions(first_pass_dir, manifest_tasks):
    findings = []
    files, set_findings = audit_exact_task_set(
        first_pass_dir, "prediction.json", manifest_tasks, "review_first_pass"
    )
    findings.extend(set_findings)
    versions = set()
    checked = 0
    for task_id in sorted(manifest_tasks):
        path = files.get(task_id)
        if path is None:
            continue
        try:
            result = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            findings.append({
                "reason": "invalid_first_pass_prediction_json",
                "task_id": task_id,
                "error": repr(exc),
            })
            continue
        checked += 1
        if not isinstance(result, dict) or not result.get("ok") or not isinstance(result.get("judge_json"), dict):
            findings.append({"reason": "invalid_first_pass_prediction", "task_id": task_id})
            continue
        if result.get("task_id") != task_id or result["judge_json"].get("task_id") != task_id:
            findings.append({"reason": "first_pass_task_id_mismatch", "task_id": task_id})
        prompt_hash = result.get("prompt_sha256")
        if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
            findings.append({"reason": "first_pass_prompt_hash_missing", "task_id": task_id})
        versions.add((
            prompt_hash,
            result.get("judge_script_sha256"),
            result.get("evidence_module_sha256"),
        ))
        for finding in walk_json(result.get("judge_json")):
            findings.append({"task_id": task_id, **finding})
    if len(versions) != 1:
        findings.append({
            "reason": "first_pass_not_single_bound_version",
            "version_count": len(versions),
        })
    return findings, files, checked


def audit_persisted_inputs(review_input_dir, first_pass_dir, manifest_tasks):
    findings = []
    files, set_findings = audit_exact_task_set(
        review_input_dir, "review_input.json", manifest_tasks, "review"
    )
    findings.extend(set_findings)
    checked = 0
    for task_id, task in sorted(manifest_tasks.items()):
        path = files.get(task_id)
        if path is None:
            continue
        try:
            value = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            findings.append({
                "reason": "invalid_review_input_json",
                "task_id": task_id,
                "error": repr(exc),
            })
            continue
        checked += 1
        expected_keys = {
            "task_spec", "observation", "first_pass_judgment", "first_pass_provenance"
        }
        if not isinstance(value, dict) or set(value) != expected_keys:
            findings.append({
                "reason": "persisted_review_input_shape",
                "task_id": task_id,
                "keys": sorted(value) if isinstance(value, dict) else None,
            })
            continue
        if value.get("task_spec") != task["task_spec"]:
            findings.append({"reason": "task_spec_differs_from_blind_manifest", "task_id": task_id})
        first_path = Path(first_pass_dir) / task_id / "prediction.json"
        if not first_path.is_file():
            findings.append({"reason": "missing_first_pass_prediction", "task_id": task_id})
            continue
        try:
            first = load_json(first_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            findings.append({
                "reason": "invalid_first_pass_prediction_json",
                "task_id": task_id,
                "error": repr(exc),
            })
            continue
        if value.get("first_pass_judgment") != first.get("judge_json"):
            findings.append({"reason": "first_pass_judgment_was_modified", "task_id": task_id})
        provenance = value.get("first_pass_provenance") or {}
        if provenance.get("prediction_file_sha256") != sha256(first_path):
            findings.append({"reason": "first_pass_provenance_hash_mismatch", "task_id": task_id})
        if provenance.get("prompt_sha256") != first.get("prompt_sha256"):
            findings.append({"reason": "first_pass_prompt_provenance_mismatch", "task_id": task_id})
        if provenance.get("model") != first.get("model"):
            findings.append({"reason": "first_pass_model_provenance_mismatch", "task_id": task_id})
        for finding in walk_json(value):
            findings.append({"task_id": task_id, **finding})
    return findings, files, checked


def audit_review(
    manifest_path,
    first_pass_dir,
    review_script,
    system_prompt,
    review_input_dir,
    expected_tasks=None,
):
    _, tasks, expected, cohort, findings = load_manifest_scope(
        manifest_path, expected_tasks
    )
    dependencies, dependency_findings = discover_local_dependencies(review_script)
    findings.extend(dependency_findings)
    paths = source_files(review_script, dependencies)
    for path in paths:
        findings.extend(audit_source_file(path, set(tasks)))
    findings.extend(audit_prompt(system_prompt, set(tasks)))
    findings.extend(audit_review_builder(paths))

    prediction_findings, prediction_files, predictions_checked = (
        audit_first_pass_predictions(first_pass_dir, tasks)
    )
    findings.extend(prediction_findings)
    input_findings, input_files, checked = audit_persisted_inputs(
        review_input_dir, first_pass_dir, tasks
    )
    findings.extend(input_findings)
    if checked != expected:
        findings.append({
            "reason": "checked_task_count_mismatch",
            "expected_tasks": expected,
            "checked_tasks": checked,
        })
    if predictions_checked != expected:
        findings.append({
            "reason": "first_pass_checked_task_count_mismatch",
            "expected_tasks": expected,
            "checked_tasks": predictions_checked,
        })

    scope = scope_base(
        manifest_path=manifest_path,
        script_path=review_script,
        script_key="review_script",
        prompt_path=system_prompt,
        input_dir=review_input_dir,
        expected_tasks=expected,
        checked_tasks=checked,
        cohort=cohort,
        dependencies=dependencies,
        input_files=input_files,
        auditor_path=AUDITOR_PATH,
    )
    scope.update({
        "review_input_dir": scope["input_dir"],
        "first_pass_dir": str(Path(first_pass_dir).resolve()),
        "first_pass_predictions_checked": predictions_checked,
        "first_pass_predictions_sha256": file_set_sha256(prediction_files),
        "task_ids_sha256": canonical_sha256(sorted(tasks)),
        "task_specs_sha256": canonical_sha256({
            task_id: task["task_spec"] for task_id, task in sorted(tasks.items())
        }),
    })
    return {
        "audit": "judge_review_blindness",
        "status": "pass" if not findings else "fail",
        "scope": scope,
        "summary": {
            "expected_tasks": expected,
            "checked_tasks": checked,
            "task_count": len(tasks),
            "persisted_review_inputs_checked": checked,
            "human_gold_in_review_request": False if not findings else None,
            "task_id_specific_review_logic": False if not findings else None,
            "recursive_local_dependencies_checked": len(dependencies),
        },
        "findings": findings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--first-pass-dir", type=Path, default=DEFAULT_FIRST_PASS)
    parser.add_argument("--review-script", type=Path, default=DEFAULT_REVIEW_SCRIPT)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--review-input-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, choices=sorted(SUPPORTED_COHORTS))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = audit_review(
        args.manifest,
        args.first_pass_dir,
        args.review_script,
        args.system_prompt,
        args.review_input_dir,
        args.expected_tasks,
    )
    write_report(report, args.out)
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
