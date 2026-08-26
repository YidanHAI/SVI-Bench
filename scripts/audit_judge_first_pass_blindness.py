#!/usr/bin/env python3
"""Statically audit first-pass Judge code and its persisted blind requests.

The audit is deliberately offline.  It never imports the Judge implementation,
opens a video, or contacts a model endpoint.  The helpers in this module are
also shared by the review and adjudication blindness audits.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = Path(__file__).resolve()
SUPPORTED_COHORTS = {75: "formal_75"}
TASK_ID_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Z][0-9]{4}(?![A-Za-z0-9_])")

# These names identify post-hoc human judgments, not rubric fields such as
# expected_key_information that are intentionally part of the blind manifest.
FORBIDDEN_KEYS = {
    "gold",
    "gold_grade",
    "gold_label",
    "gold_score",
    "ground_truth",
    "human_grade",
    "human_label",
    "human_score",
    "expert_grade",
    "expert_label",
    "expert_score",
    "target_label",
}
FORBIDDEN_SOURCE_MARKERS = (
    "人工打分表",
    "human_alignment_report",
    "gold_labels.json",
    "expert_scores.xlsx",
    "人工打分表",
    "专家打分表",
    "vl·d1",
    "vl·d2",
    "vl·d3",
    "vl·d4",
    "vl·d5",
)
FORBIDDEN_WORKBOOK_CALLS = {"load_workbook", "read_excel", "ExcelFile"}
FORBIDDEN_DYNAMIC_CALLS = {"eval", "exec", "compile"}
FORBIDDEN_IMPORT_ROOTS = {"openpyxl", "pandas"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalized_key(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


FORBIDDEN_NORMALIZED_KEYS = {normalized_key(item) for item in FORBIDDEN_KEYS}


def is_forbidden_key(value) -> bool:
    key = normalized_key(value)
    return key in FORBIDDEN_NORMALIZED_KEYS or key.rstrip("s") in FORBIDDEN_NORMALIZED_KEYS


def walk_json(value, path="$", findings=None):
    """Find human-label fields and known post-hoc artifact references."""
    findings = findings if findings is not None else []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if is_forbidden_key(key):
                findings.append({
                    "reason": "forbidden_human_label_key",
                    "path": child_path,
                })
            walk_json(child, child_path, findings)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_json(child, f"{path}[{index}]", findings)
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in FORBIDDEN_SOURCE_MARKERS:
            if marker in lowered:
                findings.append({
                    "reason": "forbidden_human_source_reference",
                    "path": path,
                    "marker": marker,
                })
    return findings


def _constant_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):  # pragma: no cover - Python 3.8 compatibility
        return node.s
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_string(node.left)
        right = _constant_string(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts = []
        for item in node.values:
            text = _constant_string(item)
            if text is None:
                return None
            parts.append(text)
        return "".join(parts)
    return None


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _module_candidates(module: str, importer: Path, level: int, search_roots):
    parts = [item for item in module.split(".") if item]
    if level:
        base = importer.parent
        for _ in range(max(0, level - 1)):
            base = base.parent
        bases = [base]
    else:
        bases = [importer.parent, *search_roots]
    seen = set()
    for base in bases:
        base = Path(base).resolve()
        stem = base.joinpath(*parts) if parts else base
        for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def _resolve_local_module(module, importer, level, search_roots):
    if not module and not level:
        return None
    for candidate in _module_candidates(module or "", importer, level, search_roots):
        if candidate.is_file():
            return candidate
    return None


def discover_local_dependencies(entry_script: Path):
    """Return every statically reachable local Python import and scan errors."""
    entry_script = Path(entry_script).resolve()
    search_roots = tuple(dict.fromkeys((entry_script.parent, ROOT / "scripts", ROOT)))
    queue = deque([(entry_script, 0)])
    visited = set()
    records = {}
    findings = []

    while queue:
        path, depth = queue.popleft()
        path = path.resolve()
        if path in visited:
            continue
        visited.add(path)
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            findings.append({
                "reason": "python_source_not_auditable",
                "source": str(path),
                "error": repr(exc),
            })
            continue

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((item.name, 0, item.name) for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append((module, node.level, "." * node.level + module))
                # ``from package import module`` can refer to a submodule.
                for item in node.names:
                    imports.append((
                        ".".join(part for part in (module, item.name) if part),
                        node.level,
                        "." * node.level + ".".join(
                            part for part in (module, item.name) if part
                        ),
                    ))

        for module, level, display_name in imports:
            dependency = _resolve_local_module(module, path, level, search_roots)
            if dependency is None or dependency == path:
                continue
            record = records.setdefault(dependency, {
                "module": display_name,
                "path": str(dependency),
                "sha256": sha256(dependency),
                "depth": depth + 1,
                "imported_by": [],
            })
            record["depth"] = min(record["depth"], depth + 1)
            importer_text = str(path)
            if importer_text not in record["imported_by"]:
                record["imported_by"].append(importer_text)
            queue.append((dependency, depth + 1))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func)
            if call_name not in {"__import__", "importlib.import_module", "import_module"}:
                continue
            module = _constant_string(node.args[0]) if node.args else None
            if not module:
                findings.append({
                    "reason": "dynamic_import_not_statically_auditable",
                    "source": str(path),
                    "line": getattr(node, "lineno", None),
                })
                continue
            dependency = _resolve_local_module(module, path, 0, search_roots)
            if dependency is None:
                continue
            record = records.setdefault(dependency, {
                "module": module,
                "path": str(dependency),
                "sha256": sha256(dependency),
                "depth": depth + 1,
                "imported_by": [],
            })
            if str(path) not in record["imported_by"]:
                record["imported_by"].append(str(path))
            queue.append((dependency, depth + 1))

    dependencies = []
    for path, record in sorted(records.items(), key=lambda item: str(item[0])):
        record["imported_by"].sort()
        record["role"] = dependency_role(path)
        dependencies.append(record)
    return dependencies, findings


def dependency_role(path: Path) -> str:
    name = Path(path).name.lower()
    if "scoring" in name:
        return "scoring"
    if "video" in name:
        return "video"
    if "judge" in name or "evidence" in name:
        return "evidence"
    if "review" in name:
        return "review"
    return "local_dependency"


def source_files(entry_script: Path, dependencies):
    return [Path(entry_script).resolve(), *(Path(item["path"]) for item in dependencies)]


def audit_source_file(path: Path, task_ids, source_name=None):
    findings = []
    path = Path(path).resolve()
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [{
            "reason": "python_source_not_auditable",
            "source": str(path),
            "error": repr(exc),
        }]

    lowered = text.lower()
    for marker in FORBIDDEN_SOURCE_MARKERS:
        if marker in lowered:
            findings.append({
                "reason": "human_source_marker",
                "source": source_name or str(path),
                "marker": marker,
            })

    literal_ids = set(TASK_ID_RE.findall(text))
    literal_ids.update(task_id for task_id in task_ids if task_id in text)
    for node in ast.walk(tree):
        folded = _constant_string(node)
        if folded:
            literal_ids.update(TASK_ID_RE.findall(folded))
    if literal_ids:
        findings.append({
            "reason": "literal_task_ids",
            "source": source_name or str(path),
            "task_ids": sorted(literal_ids),
        })

    forbidden_identifiers = set()
    forbidden_calls = set()
    forbidden_imports = set()
    sys_path_mutations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if is_forbidden_key(name):
                forbidden_identifiers.add(name)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                    forbidden_imports.add(name)
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        leaf = call_name.rsplit(".", 1)[-1]
        # Workbook helpers are commonly called as module attributes (for example,
        # pandas.read_excel), so matching their leaf name is intentional. Dynamic
        # execution primitives are Python builtins; matching every attribute named
        # ``compile`` incorrectly rejects safe calls such as ``re.compile``.
        forbidden_dynamic_call = (
            call_name in FORBIDDEN_DYNAMIC_CALLS
            or (
                call_name.startswith("builtins.")
                and leaf in FORBIDDEN_DYNAMIC_CALLS
            )
        )
        if leaf in FORBIDDEN_WORKBOOK_CALLS or forbidden_dynamic_call:
            forbidden_calls.add(call_name)
        if call_name in {"sys.path.append", "sys.path.insert", "sys.path.extend"}:
            sys_path_mutations.append(getattr(node, "lineno", None))

    if forbidden_identifiers:
        findings.append({
            "reason": "human_label_identifier_in_source",
            "source": source_name or str(path),
            "identifiers": sorted(forbidden_identifiers),
        })
    if forbidden_calls:
        findings.append({
            "reason": "forbidden_source_call",
            "source": source_name or str(path),
            "calls": sorted(forbidden_calls),
        })
    if forbidden_imports:
        findings.append({
            "reason": "workbook_library_in_judge_path",
            "source": source_name or str(path),
            "imports": sorted(forbidden_imports),
        })
    if sys_path_mutations:
        findings.append({
            "reason": "runtime_import_path_mutation",
            "source": source_name or str(path),
            "lines": sorted(sys_path_mutations),
        })
    return findings


def audit_prompt(prompt: Path, task_ids):
    findings = []
    text = Path(prompt).read_text(encoding="utf-8")
    lowered = text.lower()
    for marker in FORBIDDEN_SOURCE_MARKERS:
        if marker in lowered:
            findings.append({
                "reason": "human_source_marker",
                "source": "system_prompt",
                "marker": marker,
            })
    literal_ids = sorted(set(TASK_ID_RE.findall(text)) | {
        task_id for task_id in task_ids if task_id in text
    })
    if literal_ids:
        findings.append({
            "reason": "literal_task_ids",
            "source": "system_prompt",
            "task_ids": literal_ids,
        })
    return findings


def function_by_name(tree, name):
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ),
        None,
    )


def assigned_dict_keys(function, variable):
    matches = []
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == variable for target in targets):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            matches.append(None)
            continue
        keys = []
        for key in value.keys:
            key_text = _constant_string(key)
            if key_text is None:
                keys = None
                break
            keys.append(key_text)
        matches.append(keys)
    return matches


def find_function_implementations(paths, function_name):
    implementations = []
    for path in paths:
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            continue
        function = function_by_name(tree, function_name)
        if function is not None:
            implementations.append((Path(path).resolve(), tree, function))
    return implementations


def audit_input_builder(paths, function_name, variable, expected_keys):
    findings = []
    implementations = find_function_implementations(paths, function_name)
    if len(implementations) != 1:
        findings.append({
            "reason": f"{function_name}_implementation_count",
            "count": len(implementations),
            "sources": [str(item[0]) for item in implementations],
        })
        return findings
    path, _, function = implementations[0]
    assignments = assigned_dict_keys(function, variable)
    if len(assignments) != 1 or set(assignments[0] or []) != set(expected_keys):
        findings.append({
            "reason": f"unexpected_{variable}_shape",
            "source": str(path),
            "assignments": assignments,
            "expected_keys": sorted(expected_keys),
        })
    return findings


def audit_user_prompt_boundary(paths, function_name, input_variable):
    """Prove the API text is only constants plus the persisted input JSON."""
    findings = []
    implementations = find_function_implementations(paths, function_name)
    if len(implementations) != 1:
        return findings  # The implementation-count finding is emitted elsewhere.
    path, _, function = implementations[0]
    assignments = []
    mutations = []
    for node in ast.walk(function):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "user_prompt" for target in targets):
                assignments.append(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "user_prompt":
            mutations.append(getattr(node, "lineno", None))
    if len(assignments) != 1 or mutations:
        findings.append({
            "reason": "user_prompt_assignment_not_single_and_immutable",
            "source": str(path),
            "assignments": len(assignments),
            "mutation_lines": mutations,
        })
        return findings

    expression = assignments[0]
    loaded_names = {
        node.id
        for node in ast.walk(expression)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    unexpected_names = sorted(loaded_names - {input_variable, "json"})
    if unexpected_names:
        findings.append({
            "reason": "unapproved_user_prompt_inputs",
            "source": str(path),
            "inputs": unexpected_names,
        })
    dumps_calls = [
        node
        for node in ast.walk(expression)
        if isinstance(node, ast.Call) and _call_name(node.func) == "json.dumps"
    ]
    valid_dumps = [
        node
        for node in dumps_calls
        if node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == input_variable
    ]
    if len(dumps_calls) != 1 or len(valid_dumps) != 1:
        findings.append({
            "reason": "persisted_input_not_exclusive_json_prompt_payload",
            "source": str(path),
            "input_variable": input_variable,
            "json_dumps_calls": len(dumps_calls),
            "valid_json_dumps_calls": len(valid_dumps),
        })

    request_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node.func).rsplit(".", 1)[-1] == "call_gpt"
    ]
    if len(request_calls) != 1:
        findings.append({
            "reason": "judge_request_call_count",
            "source": str(path),
            "count": len(request_calls),
        })
    else:
        request_names = {
            node.id
            for argument in request_calls[0].args
            for node in ast.walk(argument)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for required in ("user_prompt", "frames"):
            if required not in request_names:
                findings.append({
                    "reason": "judge_request_missing_blind_input",
                    "source": str(path),
                    "input": required,
                })
    return findings


def load_manifest_scope(manifest_path: Path, expected_tasks=None):
    findings = []
    manifest_path = Path(manifest_path).resolve()
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    if manifest.get("contains_human_labels") is not False:
        findings.append({"reason": "manifest_not_explicitly_label_free"})
    # Source workbook metadata is hash-bound provenance and is not included in
    # a model request. Scan every other manifest field and every task.
    request_relevant_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {
            "source_workbook",
            "source_workbook_sha256",
            "source_sheet",
            "video_directory",
            "capture_run_summary",
            "capture_run_summary_sha256",
        }
    }
    findings.extend(walk_json(request_relevant_manifest, path="$manifest"))

    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, list):
        raw_tasks = []
        findings.append({"reason": "manifest_tasks_not_a_list"})
    tasks = {}
    duplicates = []
    malformed = []
    for index, item in enumerate(raw_tasks):
        task_spec = item.get("task_spec") if isinstance(item, dict) else None
        task_id = task_spec.get("id") if isinstance(task_spec, dict) else None
        if not isinstance(task_id, str) or not task_id.strip():
            malformed.append(index)
            continue
        if task_id in tasks:
            duplicates.append(task_id)
            continue
        if not isinstance(item.get("evidence_video"), dict):
            malformed.append(index)
        tasks[task_id] = item
    if malformed:
        findings.append({"reason": "malformed_manifest_tasks", "indexes": malformed})
    if duplicates:
        findings.append({"reason": "duplicate_manifest_task_ids", "task_ids": sorted(set(duplicates))})

    declared = manifest.get("task_count")
    if not isinstance(declared, int) or declared != len(raw_tasks):
        findings.append({
            "reason": "manifest_task_count_mismatch",
            "declared": declared,
            "actual": len(raw_tasks),
        })
    declared_expected = manifest.get("expected_task_count")
    if declared_expected is not None and declared_expected != len(raw_tasks):
        findings.append({
            "reason": "manifest_expected_task_count_mismatch",
            "declared": declared_expected,
            "actual": len(raw_tasks),
        })

    resolved_expected = expected_tasks
    if resolved_expected is None:
        resolved_expected = declared_expected if isinstance(declared_expected, int) else declared
    if not isinstance(resolved_expected, int):
        resolved_expected = len(raw_tasks)
    if resolved_expected not in SUPPORTED_COHORTS:
        findings.append({
            "reason": "unsupported_cohort_size",
            "expected_tasks": resolved_expected,
            "allowed": sorted(SUPPORTED_COHORTS),
        })
    if len(raw_tasks) != resolved_expected:
        findings.append({
            "reason": "manifest_cohort_size_mismatch",
            "expected_tasks": resolved_expected,
            "manifest_tasks": len(raw_tasks),
        })
    cohort = SUPPORTED_COHORTS.get(resolved_expected, f"unsupported_{resolved_expected}")
    return manifest, tasks, resolved_expected, cohort, findings


def _stage_files(input_dir: Path, filename: str):
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        return {}
    return {
        path.parent.name: path.resolve()
        for path in input_dir.glob(f"*/{filename}")
        if path.is_file()
    }


def audit_exact_task_set(input_dir, filename, expected_task_ids, stage):
    files = _stage_files(input_dir, filename)
    actual = set(files)
    expected = set(expected_task_ids)
    findings = []
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        findings.append({
            "reason": "missing_stage_inputs",
            "stage": stage,
            "task_ids": missing,
        })
    if unexpected:
        findings.append({
            "reason": "unexpected_stage_inputs",
            "stage": stage,
            "task_ids": unexpected,
        })
    return files, findings


def file_set_sha256(files):
    return canonical_sha256({task_id: sha256(path) for task_id, path in sorted(files.items())})


def scope_base(
    *, manifest_path, script_path, script_key, prompt_path, input_dir,
    expected_tasks, checked_tasks, cohort, dependencies, input_files,
    auditor_path,
):
    manifest_path = Path(manifest_path).resolve()
    script_path = Path(script_path).resolve()
    prompt_path = Path(prompt_path).resolve()
    auditor_path = Path(auditor_path).resolve()
    value = {
        "cohort": cohort,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        script_key: str(script_path),
        f"{script_key}_sha256": sha256(script_path),
        "system_prompt": str(prompt_path),
        "system_prompt_sha256": sha256(prompt_path),
        "input_dir": str(Path(input_dir).resolve()),
        "expected_tasks": expected_tasks,
        "checked_tasks": checked_tasks,
        "request_inputs_sha256": file_set_sha256(input_files),
        "dependencies": dependencies,
        "auditor": str(auditor_path),
        "auditor_sha256": sha256(auditor_path),
    }
    if auditor_path != AUDITOR_PATH:
        value["audit_library"] = str(AUDITOR_PATH)
        value["audit_library_sha256"] = sha256(AUDITOR_PATH)
    return value


def audit_first_pass(
    manifest_path: Path,
    judge_script: Path,
    system_prompt: Path,
    input_dir: Path,
    expected_tasks=None,
):
    manifest, tasks, expected, cohort, findings = load_manifest_scope(
        manifest_path, expected_tasks
    )
    del manifest
    dependencies, dependency_findings = discover_local_dependencies(judge_script)
    findings.extend(dependency_findings)
    paths = source_files(judge_script, dependencies)
    for path in paths:
        findings.extend(audit_source_file(path, set(tasks)))
    findings.extend(audit_prompt(system_prompt, set(tasks)))
    findings.extend(audit_input_builder(
        paths, "judge_task", "judge_input", {"task_spec", "observation"}
    ))
    findings.extend(audit_user_prompt_boundary(
        paths, "judge_task", "judge_input"
    ))

    files, set_findings = audit_exact_task_set(
        input_dir, "judge_input.json", tasks, "first_pass"
    )
    findings.extend(set_findings)
    checked = 0
    for task_id, task in sorted(tasks.items()):
        path = files.get(task_id)
        if path is None:
            continue
        try:
            value = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            findings.append({
                "reason": "invalid_first_pass_input_json",
                "task_id": task_id,
                "error": repr(exc),
            })
            continue
        checked += 1
        if not isinstance(value, dict) or set(value) != {"task_spec", "observation"}:
            findings.append({
                "reason": "unexpected_first_pass_input_shape",
                "task_id": task_id,
                "keys": sorted(value) if isinstance(value, dict) else None,
            })
            continue
        if value.get("task_spec") != task["task_spec"]:
            findings.append({
                "reason": "task_spec_differs_from_blind_manifest",
                "task_id": task_id,
            })
        for finding in walk_json(value):
            findings.append({"task_id": task_id, **finding})
    if checked != expected:
        findings.append({
            "reason": "checked_task_count_mismatch",
            "expected_tasks": expected,
            "checked_tasks": checked,
        })

    scope = scope_base(
        manifest_path=manifest_path,
        script_path=judge_script,
        script_key="judge_script",
        prompt_path=system_prompt,
        input_dir=input_dir,
        expected_tasks=expected,
        checked_tasks=checked,
        cohort=cohort,
        dependencies=dependencies,
        input_files=files,
        auditor_path=AUDITOR_PATH,
    )
    scope["first_pass_input_dir"] = scope["input_dir"]
    scope["task_ids_sha256"] = canonical_sha256(sorted(tasks))
    scope["task_specs_sha256"] = canonical_sha256({
        task_id: task["task_spec"] for task_id, task in sorted(tasks.items())
    })
    return {
        "audit": "judge_first_pass_blindness",
        "status": "pass" if not findings else "fail",
        "scope": scope,
        "summary": {
            "expected_tasks": expected,
            "checked_tasks": checked,
            "human_gold_in_first_pass_request": False if not findings else None,
            "task_id_specific_first_pass_logic": False if not findings else None,
            "recursive_local_dependencies_checked": len(dependencies),
        },
        "findings": findings,
    }


def write_report(report, out=None):
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--judge-script", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, required=True)
    parser.add_argument("--first-pass-input-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, choices=sorted(SUPPORTED_COHORTS))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = audit_first_pass(
        args.manifest,
        args.judge_script,
        args.system_prompt,
        args.first_pass_input_dir,
        args.expected_tasks,
    )
    write_report(report, args.out)
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
