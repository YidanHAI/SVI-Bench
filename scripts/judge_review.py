#!/usr/bin/env python3
"""Blind evidence review for Judge predictions."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from judge_api import chat_completions_url

from judge_first_pass import (
    DEFAULT_API_URL,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    apply_dimension_mean_score,
    build_evidence,
    call_gpt,
    load_api_key,
    parse_json_object,
    validate_prediction,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT / "prompts" / "judge_review_1.md"
DEFAULT_OUT = ROOT / "outputs" / "judge" / "review"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_non_applicable_dimensions(prediction, task_spec):
    """Fill deterministic schema fields that carry no score for this task."""
    if not isinstance(prediction, dict):
        return prediction
    dimensions = prediction.setdefault("dimensions", {})
    if not isinstance(dimensions, dict):
        return prediction
    for name, spec in task_spec["dimensions"].items():
        if spec["applicable"]:
            continue
        dimensions[name] = {
            "applicable": False,
            "grade": "not_applicable",
            "score": None,
            "threshold_used": spec.get("threshold") or "-",
            "evidence": [],
            "reason": "该维度不适用于当前任务，不参与评分。",
            "confidence": 1.0,
        }
    return prediction


def review_task(args, task, system_prompt, api_key):
    task_spec = task["task_spec"]
    task_id = task_spec["id"]
    task_dir = args.out_dir / task_id
    prediction_path = task_dir / "prediction.json"
    if prediction_path.exists() and not args.force:
        existing = json.loads(prediction_path.read_text(encoding="utf-8"))
        if existing.get("ok"):
            return existing

    first_path = args.first_pass_dir / task_id / "prediction.json"
    if not first_path.is_file():
        raise FileNotFoundError(f"Missing first-pass prediction: {first_path}")
    first_result = json.loads(first_path.read_text(encoding="utf-8"))
    if not first_result.get("ok") or not isinstance(first_result.get("judge_json"), dict):
        raise RuntimeError(f"First-pass prediction is not valid: {first_path}")

    # Reuse the exact first-pass evidence directory when possible. This keeps
    # the review evidence identical and avoids duplicating large frame sets.
    observation, frames = build_evidence(
        task,
        args.first_pass_dir / task_id,
        args.max_images,
        args.frame_workers,
        args.max_temporal_sheets,
    )
    review_input = {
        "task_spec": task_spec,
        "observation": observation,
        "first_pass_judgment": first_result["judge_json"],
        "first_pass_provenance": {
            "model": first_result.get("model"),
            "prompt_sha256": first_result.get("prompt_sha256"),
            "prediction_file_sha256": file_sha256(first_path),
        },
    }
    user_prompt = (
        "请复核这条首轮判定。人工评分未包含在输入中。"
        "首轮意见不可信，必须按 system prompt 独立核验证据。\n\n"
        "REVIEW_INPUT_JSON:\n"
        + json.dumps(review_input, ensure_ascii=False, indent=2)
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    write_json(task_dir / "review_input.json", review_input)
    response = call_gpt(
        args.api_url,
        api_key,
        args.model,
        system_prompt,
        user_prompt,
        frames,
        args.timeout,
        args.max_tokens,
        args.retries,
    )
    (task_dir / "raw_response.txt").write_text(
        response["content"] + "\n", encoding="utf-8"
    )
    prediction = normalize_non_applicable_dimensions(
        parse_json_object(response["content"]), task_spec
    )
    validation_errors = validate_prediction(prediction, task_spec)
    dimension_mean_score = apply_dimension_mean_score(prediction, task_spec)
    result = {
        "ok": not validation_errors,
        "task_id": task_id,
        "model": args.model,
        "response_model": response.get("response_model"),
        "api_url": args.api_url,
        "prompt_file": str(args.system_prompt.resolve()),
        "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "review_stage": "blind_evidence_review",
        "first_pass_prediction": str(first_path.resolve()),
        "first_pass_prediction_sha256": file_sha256(first_path),
        "native_video_input": False,
        "input_mode": "first_pass_plus_same_timestamped_evidence",
        "evidence_image_count": len(frames),
        "evidence_variant": response.get("evidence_variant"),
        "evidence_image_count_sent": response.get("evidence_image_count_sent"),
        "judge_json": prediction,
        "dimension_mean_score": dimension_mean_score,
        "benchmark_score": prediction.get("benchmark_score"),
        "score_without_d3": prediction.get("score_without_d3"),
        "validation_errors": validation_errors,
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage"),
        "request_attempt": response.get("attempt"),
    }
    write_json(prediction_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--first-pass-dir",
        type=Path,
        default=(
            Path(os.environ["JUDGE_FIRST_PASS_DIR"])
            if os.getenv("JUDGE_FIRST_PASS_DIR")
            else None
        ),
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--max-temporal-sheets", type=int, default=18)
    parser.add_argument("--frame-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--gap-s", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.api_url = args.api_url or chat_completions_url()

    if args.first_pass_dir is None or not args.first_pass_dir.is_dir():
        raise FileNotFoundError(
            "Provide --first-pass-dir or JUDGE_FIRST_PASS_DIR with valid predictions"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("contains_human_labels") is not False:
        raise RuntimeError("Refusing a manifest that is not explicitly label-free")
    selected = set(args.task_id)
    tasks = [
        item
        for item in manifest["tasks"]
        if not selected or item["task_spec"]["id"] in selected
    ]
    missing = selected - {item["task_spec"]["id"] for item in tasks}
    if missing:
        raise KeyError(f"Unknown task ids: {sorted(missing)}")

    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    api_key = load_api_key()
    results = []
    for index, task in enumerate(tasks, start=1):
        task_id = task["task_spec"]["id"]
        try:
            result = review_task(args, task, system_prompt, api_key)
        except Exception as exc:
            result = {"ok": False, "task_id": task_id, "error": repr(exc)}
            write_json(args.out_dir / task_id / "prediction.json", result)
        results.append(result)
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(tasks),
                    "task_id": task_id,
                    "ok": result.get("ok"),
                    "error": result.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if index < len(tasks):
            time.sleep(args.gap_s)

    write_json(
        args.out_dir / "batch_summary.json",
        {
            "manifest": str(args.manifest.resolve()),
            "first_pass_dir": str(args.first_pass_dir.resolve()),
            "model": args.model,
            "review_stage": "blind_evidence_review",
            "total": len(results),
            "ok": sum(bool(item.get("ok")) for item in results),
            "failed": sum(not bool(item.get("ok")) for item in results),
            "results": [
                {key: item.get(key) for key in ("task_id", "ok", "error")}
                for item in results
            ],
        },
    )


if __name__ == "__main__":
    main()
