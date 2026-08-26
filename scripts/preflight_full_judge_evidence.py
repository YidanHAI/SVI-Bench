#!/usr/bin/env python3
"""Build and hash every frame input before starting a full Judge run."""

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE_SCRIPT = ROOT / "scripts" / "judge_first_pass.py"


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
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_build_evidence(script_path):
    spec = importlib.util.spec_from_file_location(
        "pinned_full_judge_first_pass", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Judge script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_evidence


def reusable_preflight(path, *, task_id, manifest_hash, script_hash):
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        value.get("ok") is not True
        or value.get("task_id") != task_id
        or value.get("manifest_sha256") != manifest_hash
        or value.get("judge_script_sha256") != script_hash
        or not isinstance(value.get("observation"), dict)
        or not isinstance(value.get("evidence"), list)
        or not value["evidence"]
    ):
        return None
    for item in value["evidence"]:
        try:
            evidence_path = Path(item["path"])
            if (
                not evidence_path.is_file()
                or evidence_path.stat().st_size != item.get("bytes")
                or sha256(evidence_path) != item.get("sha256")
            ):
                return None
        except (KeyError, OSError, TypeError):
            return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--judge-script", type=Path, default=DEFAULT_JUDGE_SCRIPT,
        help="Pinned first-pass implementation used to build evidence",
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--max-temporal-sheets", type=int, default=18)
    parser.add_argument("--frame-workers", type=int, default=4)
    args = parser.parse_args()
    judge_script = args.judge_script.resolve()
    build_evidence = load_build_evidence(judge_script)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("contains_human_labels") is not False:
        raise RuntimeError("Refusing a manifest that is not explicitly label-free")
    selected = set(args.task_id)
    tasks = [
        item for item in manifest["tasks"]
        if not selected or item["task_spec"]["id"] in selected
    ]
    missing = selected - {item["task_spec"]["id"] for item in tasks}
    if missing:
        raise KeyError(f"Unknown task ids: {sorted(missing)}")

    script_hash = sha256(judge_script)
    manifest_hash = sha256(args.manifest)
    results = []
    for index, task in enumerate(tasks, start=1):
        task_id = task["task_spec"]["id"]
        task_dir = args.out_dir / task_id
        preflight_path = task_dir / "evidence_preflight.json"
        existing = reusable_preflight(
            preflight_path,
            task_id=task_id,
            manifest_hash=manifest_hash,
            script_hash=script_hash,
        )
        reused = existing is not None
        if reused:
            result = existing
            results.append(result)
            print(json.dumps({
                "index": index,
                "total": len(tasks),
                "task_id": task_id,
                "ok": True,
                "evidence_count": result.get("evidence_count"),
                "reused": True,
                "error": None,
            }, ensure_ascii=False), flush=True)
            continue
        try:
            observation, frames = build_evidence(
                task,
                task_dir,
                args.max_images,
                args.frame_workers,
                args.max_temporal_sheets,
            )
            evidence = [
                {
                    "index": frame_index,
                    "timestamp_s": timestamp,
                    "view": view,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for frame_index, (timestamp, path, view) in enumerate(frames)
            ]
            result = {
                "ok": True,
                "task_id": task_id,
                "manifest_sha256": manifest_hash,
                "judge_script_sha256": script_hash,
                "evidence_count": len(evidence),
                "observation": observation,
                "evidence": evidence,
                "finished_at": now(),
            }
        except Exception as exc:
            result = {
                "ok": False,
                "task_id": task_id,
                "manifest_sha256": manifest_hash,
                "judge_script_sha256": script_hash,
                "error": repr(exc),
                "finished_at": now(),
            }
        write_json(preflight_path, result)
        results.append(result)
        print(json.dumps({
            "index": index,
            "total": len(tasks),
            "task_id": task_id,
            "ok": result["ok"],
            "evidence_count": result.get("evidence_count"),
            "reused": False,
            "error": result.get("error"),
        }, ensure_ascii=False), flush=True)

    summary = {
        "schema_version": 1,
        "status": "complete" if all(item["ok"] for item in results) else "failed",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "contains_human_labels": False,
        "judge_script": str(judge_script),
        "judge_script_sha256": script_hash,
        "tasks_total": len(results),
        "ok": sum(item["ok"] for item in results),
        "failed": sum(not item["ok"] for item in results),
        "failed_task_ids": [item["task_id"] for item in results if not item["ok"]],
        "finished_at": now(),
    }
    write_json(args.out_dir / "evidence_preflight_summary.json", summary)
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
