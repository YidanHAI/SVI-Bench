#!/usr/bin/env python3
"""Audit adjudication source, transitive imports, and persisted requests."""

from __future__ import annotations

import argparse
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
    load_json,
    load_manifest_scope,
    scope_base,
    sha256,
    source_files,
    walk_json,
    write_report,
)


AUDITOR_PATH = Path(__file__).resolve()


def audit_candidate_directories(candidate_dirs, tasks):
    findings = []
    all_files = []
    checked_counts = []
    resolved = [Path(path).resolve() for path in candidate_dirs]
    if len(set(resolved)) != len(resolved):
        findings.append({"reason": "duplicate_candidate_directories"})
    for index, directory in enumerate(resolved):
        files, set_findings = audit_exact_task_set(
            directory,
            "prediction.json",
            tasks,
            f"adjudication_candidate_{index + 1}",
        )
        findings.extend(set_findings)
        checked = 0
        for task_id in sorted(tasks):
            path = files.get(task_id)
            if path is None:
                continue
            try:
                value = load_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                findings.append({
                    "reason": "invalid_candidate_prediction_json",
                    "task_id": task_id,
                    "index": index,
                    "error": repr(exc),
                })
                continue
            checked += 1
            if not isinstance(value, dict) or not value.get("ok") or not isinstance(value.get("judge_json"), dict):
                findings.append({
                    "reason": "invalid_candidate_prediction",
                    "task_id": task_id,
                    "index": index,
                })
            for finding in walk_json(value.get("judge_json") if isinstance(value, dict) else value):
                findings.append({"task_id": task_id, "index": index, **finding})
        all_files.append(files)
        checked_counts.append(checked)
    return findings, all_files, checked_counts


def audit_persisted_inputs(
    adjudication_input_dir,
    candidate_dirs,
    candidate_files,
    manifest_tasks,
):
    findings = []
    files, set_findings = audit_exact_task_set(
        adjudication_input_dir,
        "adjudication_input.json",
        manifest_tasks,
        "adjudication",
    )
    findings.extend(set_findings)
    checked = 0
    for task_id, task in sorted(manifest_tasks.items()):
        input_path = files.get(task_id)
        prediction_path = Path(adjudication_input_dir) / task_id / "prediction.json"
        if input_path is None:
            continue
        try:
            value = load_json(input_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            findings.append({
                "reason": "invalid_adjudication_input_json",
                "task_id": task_id,
                "error": repr(exc),
            })
            continue
        checked += 1
        expected_keys = {"task_spec", "observation", "candidate_judgments"}
        if not isinstance(value, dict) or set(value) != expected_keys:
            findings.append({
                "reason": "unexpected_adjudication_input_shape",
                "task_id": task_id,
                "keys": sorted(value) if isinstance(value, dict) else None,
            })
            continue
        if value.get("task_spec") != task["task_spec"]:
            findings.append({"reason": "task_spec_differs_from_blind_manifest", "task_id": task_id})
        candidates = value.get("candidate_judgments")
        if not isinstance(candidates, list) or len(candidates) != len(candidate_dirs):
            findings.append({
                "reason": "candidate_count_mismatch",
                "task_id": task_id,
                "expected": len(candidate_dirs),
                "actual": len(candidates) if isinstance(candidates, list) else None,
            })
            candidates = candidates if isinstance(candidates, list) else []
        for index, directory in enumerate(candidate_dirs):
            if index >= len(candidates):
                break
            candidate = candidates[index]
            if not isinstance(candidate, dict) or set(candidate) != {"candidate", "judge_json", "provenance"}:
                findings.append({
                    "reason": "unexpected_candidate_input_shape",
                    "task_id": task_id,
                    "index": index,
                })
                continue
            expected_path = candidate_files[index].get(task_id)
            if expected_path is None:
                continue
            try:
                expected = load_json(expected_path)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                findings.append({
                    "reason": "invalid_candidate_prediction_json",
                    "task_id": task_id,
                    "index": index,
                    "error": repr(exc),
                })
                continue
            if candidate.get("candidate") != f"candidate_{index + 1}":
                findings.append({"reason": "candidate_identity_mismatch", "task_id": task_id, "index": index})
            if candidate.get("judge_json") != expected.get("judge_json"):
                findings.append({"reason": "candidate_judgment_modified", "task_id": task_id, "index": index})
            provenance = candidate.get("provenance") or {}
            if provenance.get("sha256") != sha256(expected_path):
                findings.append({"reason": "candidate_provenance_hash_mismatch", "task_id": task_id, "index": index})
            if provenance.get("path") != str(expected_path.resolve()):
                findings.append({"reason": "candidate_provenance_path_mismatch", "task_id": task_id, "index": index})
            expected_stage = expected.get("review_stage") or "first_pass"
            for key, expected_value in (
                ("model", expected.get("model")),
                ("stage", expected_stage),
                ("prompt_sha256", expected.get("prompt_sha256")),
            ):
                if provenance.get(key) != expected_value:
                    findings.append({
                        "reason": "candidate_provenance_metadata_mismatch",
                        "task_id": task_id,
                        "index": index,
                        "field": key,
                    })
        for finding in walk_json(value):
            findings.append({"task_id": task_id, **finding})
        try:
            prediction = load_json(prediction_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            findings.append({
                "reason": "missing_or_invalid_adjudicated_prediction",
                "task_id": task_id,
                "error": repr(exc),
            })
        else:
            if not isinstance(prediction, dict) or not prediction.get("ok"):
                findings.append({"reason": "missing_or_invalid_adjudicated_prediction", "task_id": task_id})
    return findings, files, checked


def audit_adjudication(
    manifest_path,
    adjudicator_script,
    system_prompt,
    adjudication_input_dir,
    candidate_dirs,
    expected_tasks=None,
):
    _, tasks, expected, cohort, findings = load_manifest_scope(
        manifest_path, expected_tasks
    )
    dependencies, dependency_findings = discover_local_dependencies(adjudicator_script)
    findings.extend(dependency_findings)
    paths = source_files(adjudicator_script, dependencies)
    for path in paths:
        findings.extend(audit_source_file(path, set(tasks)))
    findings.extend(audit_prompt(system_prompt, set(tasks)))
    findings.extend(audit_input_builder(
        paths,
        "adjudicate_task",
        "adjudication_input",
        {"task_spec", "observation", "candidate_judgments"},
    ))
    findings.extend(audit_user_prompt_boundary(
        paths, "adjudicate_task", "adjudication_input"
    ))
    if len(candidate_dirs) < 2:
        findings.append({"reason": "insufficient_candidate_directories", "count": len(candidate_dirs)})

    candidate_findings, candidate_files, candidate_checked = (
        audit_candidate_directories(candidate_dirs, tasks)
    )
    findings.extend(candidate_findings)
    input_findings, input_files, checked = audit_persisted_inputs(
        adjudication_input_dir,
        [Path(item).resolve() for item in candidate_dirs],
        candidate_files,
        tasks,
    )
    findings.extend(input_findings)
    if checked != expected:
        findings.append({
            "reason": "checked_task_count_mismatch",
            "expected_tasks": expected,
            "checked_tasks": checked,
        })
    for index, count in enumerate(candidate_checked):
        if count != expected:
            findings.append({
                "reason": "candidate_checked_task_count_mismatch",
                "index": index,
                "expected_tasks": expected,
                "checked_tasks": count,
            })

    scope = scope_base(
        manifest_path=manifest_path,
        script_path=adjudicator_script,
        script_key="adjudicator_script",
        prompt_path=system_prompt,
        input_dir=adjudication_input_dir,
        expected_tasks=expected,
        checked_tasks=checked,
        cohort=cohort,
        dependencies=dependencies,
        input_files=input_files,
        auditor_path=AUDITOR_PATH,
    )
    scope.update({
        "adjudication_input_dir": scope["input_dir"],
        "candidate_dirs": [str(Path(path).resolve()) for path in candidate_dirs],
        "candidate_predictions_checked": candidate_checked,
        "candidate_predictions_sha256": [
            file_set_sha256(files) for files in candidate_files
        ],
        "task_ids_sha256": canonical_sha256(sorted(tasks)),
        "task_specs_sha256": canonical_sha256({
            task_id: task["task_spec"] for task_id, task in sorted(tasks.items())
        }),
    })
    return {
        "audit": "judge_adjudication_blindness",
        "status": "pass" if not findings else "fail",
        "scope": scope,
        "summary": {
            "expected_tasks": expected,
            "checked_tasks": checked,
            "task_count": len(tasks),
            "persisted_adjudication_inputs_checked": checked,
            "human_gold_in_adjudication_request": False if not findings else None,
            "task_id_specific_grading": False if not findings else None,
            "recursive_local_dependencies_checked": len(dependencies),
        },
        "findings": findings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adjudicator-script", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, required=True)
    parser.add_argument("--adjudication-input-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True)
    parser.add_argument("--expected-tasks", type=int, choices=sorted(SUPPORTED_COHORTS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = audit_adjudication(
        args.manifest,
        args.adjudicator_script,
        args.system_prompt,
        args.adjudication_input_dir,
        args.candidate_dir,
        args.expected_tasks,
    )
    write_report(report, args.out)
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
