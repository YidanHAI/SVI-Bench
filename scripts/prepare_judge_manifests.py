#!/usr/bin/env python3
"""Build five label-free Judge manifests from a completed recording campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from judge_workbook_contract import (
    DEFAULT_RUBRIC_SHEET,
    DEFAULT_TASK_SHEET,
    bind_contract_to_spec,
    build_workbook_v2_contract,
    format_blocking_issues,
    load_task_specs,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = (
    "joyai-vl-interaction",
    "doubao-seed-2.1-pro",
    "mage-vl",
    "moss-vl-realtime",
    "minicpmo-4.5-9b-native-video-v2",
)


def env_path(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback))


def load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Cannot read recording index {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected an object at {path}:{line_number}")
        records.append(value)
    return records


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == encoded:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.is_file() and path.read_text(encoding="utf-8") == encoded:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


def normalized_prompt(text: str) -> str:
    without_round_labels = re.sub(
        r"(?m)^\s*R\d+\s*[:：]\s*", "", str(text or ""), flags=re.IGNORECASE
    )
    return re.sub(r"\s+", "", without_round_labels)


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def resolve_index_video(index_path: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("Recording row has no canonical_video")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    root = index_path.parent.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"canonical_video escapes recording index root: {raw}") from exc
    return resolved


def find_model_summary(run_root: Path, model_id: str) -> Path:
    model_root = run_root / "models" / model_id
    candidates = [model_root / "run_summary.json", *model_root.glob("*/run_summary.json")]
    found = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(found) != 1:
        raise RuntimeError(f"Expected one run_summary.json for {model_id}, found {found}")
    return found[0]


def records_from_run(run_root: Path) -> list[dict[str, Any]]:
    spec_path = run_root / "campaign_spec.json"
    spec = load_json(spec_path, "recording campaign spec")
    if tuple(spec.get("model_order") or ()) != EXPECTED_MODELS:
        raise RuntimeError("Recording campaign model_order is not the release five-model set")
    if int(spec.get("tasks_total") or 0) != 75:
        raise RuntimeError("Recording campaign must contain exactly 75 tasks")

    records: list[dict[str, Any]] = []
    for model_id in EXPECTED_MODELS:
        summary_path = find_model_summary(run_root, model_id)
        run_summary = load_json(summary_path, f"recording summary for {model_id}")
        if run_summary.get("status") != "complete":
            raise RuntimeError(f"Recording campaign is incomplete for {model_id}")
        rows = run_summary.get("task_results") or []
        if not isinstance(rows, list) or len(rows) != 75:
            raise RuntimeError(f"{model_id}: expected 75 recording results")
        for row in rows:
            if row.get("status") not in {"ok", "skipped"}:
                raise RuntimeError(f"{model_id}/{row.get('task_id')}: recording is not valid")
            task_id = str(row.get("task_id") or "").strip()
            video = Path(str(row.get("task_mp4") or "")).resolve()
            summary = Path(str(row.get("summary_path") or ""))
            if not summary.is_file():
                summary = Path(str(row.get("task_dir") or "")) / "summary.json"
            summary = summary.resolve()
            if not task_id or not video.is_file() or not summary.is_file():
                raise RuntimeError(f"{model_id}/{task_id or '?'}: recording artifacts are missing")
            capture = load_json(summary, f"capture summary for {model_id}/{task_id}")
            if capture.get("status") != "ok":
                raise RuntimeError(f"{model_id}/{task_id}: capture summary is not ok")
            if str((capture.get("task") or {}).get("id") or "") != task_id:
                raise RuntimeError(f"{model_id}/{task_id}: capture task id mismatch")
            queries = [
                {
                    "id": event.get("id"),
                    "query": event.get("query"),
                    "query_time_s": event.get("query_time_s"),
                }
                for event in capture.get("query_events") or []
            ]
            if not queries:
                queries = list((capture.get("task") or {}).get("queries") or [])
            records.append({
                "model_id": model_id,
                "model_name": model_id,
                "task_index": int(row.get("task_index") or 0),
                "task_id": task_id,
                "queries": queries,
                "canonical_video": str(video),
                "summary_json": str(summary),
                "size_bytes": video.stat().st_size,
                "sha256": sha256(video),
            })
    return records


def template_from_workbook(workbook: Path, task_sheet: str, rubric_sheet: str) -> dict:
    specs = load_task_specs(workbook, task_sheet)
    if len(specs) != 75:
        raise RuntimeError(f"Workbook has {len(specs)} selected tasks; expected 75")
    bundle = build_workbook_v2_contract(workbook, specs, rubric_sheet=rubric_sheet)
    validation = bundle["rubric_validation"]
    if validation.get("formal_ready") is not True:
        raise RuntimeError(
            "Workbook rubric is not formal-ready:\n"
            + format_blocking_issues(validation.get("issues") or [])
        )
    return {
        "schema_version": 2,
        "contains_human_labels": False,
        "source_workbook": str(workbook.resolve()),
        "source_workbook_sha256": sha256(workbook),
        "source_sheet": task_sheet,
        **bundle,
        "tasks": [
            {"task_spec": bind_contract_to_spec(spec, bundle)}
            for spec in specs.values()
        ],
    }


def build_manifest(
    model_id: str,
    records: list[dict[str, Any]],
    template: dict[str, Any],
    recording_index: Path,
) -> dict[str, Any]:
    template_by_id = {
        str(item["task_spec"]["id"]): item for item in template["tasks"]
    }
    ordered = sorted(records, key=lambda item: int(item.get("task_index") or 0))
    task_ids = [str(item.get("task_id") or "") for item in ordered]
    if len(ordered) != 75 or len(set(task_ids)) != 75:
        raise RuntimeError(f"{model_id}: expected 75 unique task IDs")
    if set(task_ids) != set(template_by_id):
        missing = sorted(set(template_by_id) - set(task_ids))
        extra = sorted(set(task_ids) - set(template_by_id))
        raise RuntimeError(f"{model_id}: task mismatch missing={missing} extra={extra}")
    if [int(item.get("task_index") or 0) for item in ordered] != list(range(1, 76)):
        raise RuntimeError(f"{model_id}: task_index must be exactly 1..75")

    tasks = []
    for record in ordered:
        task_id = str(record["task_id"])
        task_spec = copy.deepcopy(template_by_id[task_id]["task_spec"])
        queries = record.get("queries")
        if not isinstance(queries, list) or not queries:
            raise RuntimeError(f"{model_id}/{task_id}: queries are missing")
        recorded_prompt = "\n".join(str(item.get("query") or "") for item in queries)
        if normalized_prompt(recorded_prompt) != normalized_prompt(task_spec["user_prompt"]):
            raise RuntimeError(f"{model_id}/{task_id}: recorded query differs from workbook")

        video = resolve_index_video(recording_index, record.get("canonical_video"))
        if not video.is_file():
            raise RuntimeError(f"{model_id}/{task_id}: missing video {video}")
        if video.stat().st_size != int(record.get("size_bytes") or -1):
            raise RuntimeError(f"{model_id}/{task_id}: video byte count mismatch")
        if sha256(video) != record.get("sha256"):
            raise RuntimeError(f"{model_id}/{task_id}: video SHA256 mismatch")

        raw_summary = record.get("summary_json")
        summary = Path(str(raw_summary)).resolve() if raw_summary else video.parent / "summary.json"
        capture = load_json(summary, f"capture summary for {model_id}/{task_id}")
        if capture.get("status") != "ok":
            raise RuntimeError(f"{model_id}/{task_id}: capture summary is not ok")
        tasks.append({
            "task_spec": task_spec,
            "evidence_video": {
                "path": str(video),
                "sha256": record["sha256"],
                "bytes": int(record["size_bytes"]),
            },
            "timing_source": {
                "summary_json": str(summary),
                "summary_json_sha256": sha256(summary),
                "recording_match": "same_capture_task",
            },
            "capture_provenance": {
                "model_id": model_id,
                "task_index": int(record["task_index"]),
                "recording_index": str(recording_index),
                "recording_index_sha256": sha256(recording_index),
            },
        })

    manifest = copy.deepcopy(template)
    manifest.update({
        "capture_model_id": model_id,
        "recording_index": str(recording_index),
        "recording_index_sha256": sha256(recording_index),
        "expected_task_count": 75,
        "task_count": 75,
        "missing_task_ids": [],
        "tasks": tasks,
    })
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--recording-run", type=Path)
    source.add_argument("--recording-index", type=Path)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=env_path("BENCHMARK_WORKBOOK", "data/SVI_bench_tasks_and_anchors.xlsx"),
    )
    parser.add_argument("--task-sheet", default=DEFAULT_TASK_SHEET)
    parser.add_argument("--rubric-sheet", default=DEFAULT_RUBRIC_SHEET)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=env_path("JUDGE_MANIFEST_DIR", "outputs/judge_campaign/manifests"),
    )
    args = parser.parse_args(argv)

    recording_run = args.recording_run or (
        Path(os.environ["JUDGE_RECORDING_RUN"])
        if os.getenv("JUDGE_RECORDING_RUN") else None
    )
    recording_index = args.recording_index or (
        Path(os.environ["JUDGE_RECORDING_INDEX"])
        if os.getenv("JUDGE_RECORDING_INDEX") else None
    )
    if (recording_run is None) == (recording_index is None):
        raise RuntimeError(
            "Set exactly one of JUDGE_RECORDING_RUN/--recording-run or "
            "JUDGE_RECORDING_INDEX/--recording-index"
        )

    out_dir = args.out_dir.resolve()
    workbook = args.workbook.resolve()
    if not workbook.is_file():
        raise FileNotFoundError(f"Benchmark workbook not found: {workbook}")
    if recording_run is not None:
        records = records_from_run(recording_run.resolve())
        recording_index = out_dir / "recording_index.jsonl"
        atomic_jsonl(recording_index, records)
    else:
        recording_index = recording_index.resolve()
        records = load_jsonl(recording_index)

    if len(records) != 375:
        raise RuntimeError(f"Expected 375 recording rows, got {len(records)}")
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for record in records:
        pair = (str(record.get("model_id") or ""), str(record.get("task_id") or ""))
        if pair in seen:
            raise RuntimeError(f"Duplicate model/task recording: {pair}")
        seen.add(pair)
        by_model[pair[0]].append(record)
    if set(by_model) != set(EXPECTED_MODELS):
        raise RuntimeError(f"Recording model set differs: {sorted(by_model)}")

    template = template_from_workbook(workbook, args.task_sheet, args.rubric_sheet)
    index_models = {}
    for model_id in EXPECTED_MODELS:
        manifest = build_manifest(model_id, by_model[model_id], template, recording_index)
        path = out_dir / f"judge_manifest_{safe_slug(model_id)}.json"
        atomic_json(path, manifest)
        index_models[model_id] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "task_count": 75,
        }

    manifest_index = {
        "schema_version": 1,
        "kind": "five_model_judge_manifest_index",
        "contains_human_labels": False,
        "recording_index": {
            "path": str(recording_index.resolve()),
            "sha256": sha256(recording_index),
            "recording_count": len(records),
        },
        "rubric_version": template["rubric_version"],
        "rubric_contract_sha256": template["rubric_contract_sha256"],
        "source_workbook_sha256": template["source_workbook_sha256"],
        "model_count": 5,
        "task_count_per_model": 75,
        "total_tasks": 375,
        "models": index_models,
    }
    index_path = out_dir / "manifest_index.json"
    atomic_json(index_path, manifest_index)
    print(json.dumps({
        "status": "complete",
        "manifest_index": str(index_path),
        "models": 5,
        "tasks": 375,
        "contains_human_labels": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
