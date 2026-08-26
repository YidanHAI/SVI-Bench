#!/usr/bin/env python3
"""Final first-pass Judge with a fixed seed and timeline-safe evidence.

The evidence rule is task-agnostic.  Complete evidence is always attempted
first.  If the gateway rejects that many images, the first fallback keeps the
temporal sheets and generic tail UI zooms, then fills the remaining slots with
evenly sampled individual frames.  It never reads a task id, human label,
expert answer, historical prediction, or alignment metric.
"""

from __future__ import annotations

import copy
import io
import re
import subprocess
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

import judge_transport as _implementation


REQUEST_SEED = 20260821
TAIL_OFFSETS_FROM_END_S = (1.5, 0.25)
TIMELINE_FALLBACK_IMAGE_BUDGET = 24
_ORIGINAL_REQUEST_PAYLOAD = _implementation.request_payload
_ORIGINAL_D3_CLOCK_POLICY = _implementation.d3_clock_policy
_ORIGINAL_BUILD_EVIDENCE = _implementation.build_evidence
_ORIGINAL_EVIDENCE_VARIANTS = _implementation.evidence_variants


def request_payload(model, system_prompt, user_prompt, frames, max_tokens):
    payload = _ORIGINAL_REQUEST_PAYLOAD(
        model, system_prompt, user_prompt, frames, max_tokens
    )
    payload["seed"] = REQUEST_SEED
    return payload


def d3_clock_policy(task_spec, timeline):
    policy = _ORIGINAL_D3_CLOCK_POLICY(task_spec, timeline)
    if policy.get("mode") != "visual_event_end_to_end":
        return policy

    trigger = str(task_spec.get("trigger") or "")
    d3 = ((task_spec.get("dimensions") or {}).get("D3") or {})
    threshold = str(d3.get("threshold") or "")
    explicit_trigger = any(
        marker in trigger
        for marker in (
            "用户 query 输入完成后",
            "提问后模型作答",
            "发送 query",
            "query 从提交",
        )
    )
    response_threshold = any(
        marker in threshold for marker in ("首 token", "R1", "R2", "前台响应", "委托确认")
    )
    if not (explicit_trigger and response_threshold):
        return policy

    projected = copy.deepcopy(task_spec)
    projected_d3 = projected.setdefault("dimensions", {}).setdefault("D3", {})
    projected_d3["threshold"] = threshold + "\n用户 query 输入完成后"
    return _ORIGINAL_D3_CLOCK_POLICY(projected, timeline)


_implementation.request_payload = request_payload
_implementation.d3_clock_policy = d3_clock_policy


def needs_tail_zoom(task_spec):
    dimensions = task_spec.get("dimensions") or {}
    d5_applicable = bool((dimensions.get("D5") or {}).get("applicable"))
    prompt = str(task_spec.get("user_prompt") or "")
    round_markers = sum(
        bool(re.match(r"\s*R\d+\s*[:：]", line, flags=re.IGNORECASE))
        for line in prompt.splitlines()
    )
    return d5_applicable and round_markers >= 2


def decode_frame(video: Path, timestamp: float) -> Image.Image:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if ok:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    decoded = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{timestamp:.3f}", "-i", str(video),
            "-frames:v", "1", "-an", "-f", "image2pipe",
            "-vcodec", "png", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if decoded.returncode != 0 or not decoded.stdout:
        raise RuntimeError(f"Could not decode tail UI at t={timestamp:.3f}s")
    with Image.open(io.BytesIO(decoded.stdout)) as image:
        return image.convert("RGB")


def write_tail_zoom(video: Path, timestamp: float, out_path: Path) -> None:
    if out_path.is_file():
        return
    opened = decode_frame(video, timestamp)
    cropped = _implementation.crop_for_view(opened, "right_ui")
    zoomed = cropped.resize(
        (cropped.width * 2, cropped.height * 2), Image.Resampling.LANCZOS
    )
    draw = ImageDraw.Draw(zoomed)
    label = f"FINAL UI t={timestamp:.3f}s"
    font = _implementation.font_for(max(zoomed.width, 1200))
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, box[2] + 16, box[3] + 16), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    zoomed.save(out_path, format="JPEG", quality=92, optimize=True)


def evenly_spaced(indices, count):
    indices = list(indices)
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return indices
    if count == 1:
        return [indices[len(indices) // 2]]
    return [
        indices[round(position * (len(indices) - 1) / (count - 1))]
        for position in range(count)
    ]


def reorder_for_timeline_fallback(observation, evidence):
    """Put a self-contained <=24-image fallback at the front of evidence."""
    evidence = list(evidence)
    sampling = observation["screenshot_sampling"]
    items = list(sampling["items"])
    if len(items) != len(evidence):
        raise RuntimeError("Evidence and screenshot metadata are not aligned")

    temporal = [
        index for index, (_, _, view) in enumerate(evidence)
        if view.startswith("temporal_sheet_")
    ]
    tail = [
        index for index, (_, _, view) in enumerate(evidence)
        if view == "right_ui_tail_zoom"
    ]
    critical = sorted(set(temporal + tail))
    ordinary = [index for index in range(len(evidence)) if index not in critical]
    budget = min(TIMELINE_FALLBACK_IMAGE_BUDGET, len(evidence))

    if len(critical) <= budget:
        selected = critical + evenly_spaced(ordinary, budget - len(critical))
    else:
        kept_tail = tail[:budget]
        remaining = budget - len(kept_tail)
        kept_temporal = evenly_spaced(
            [index for index in temporal if index not in kept_tail], remaining
        )
        selected = kept_tail + kept_temporal
    selected = sorted(set(selected))
    if len(selected) != budget:
        fill = [index for index in range(len(evidence)) if index not in selected]
        selected.extend(evenly_spaced(fill, budget - len(selected)))
        selected = sorted(set(selected))
    order = selected + [index for index in range(len(evidence)) if index not in selected]

    reordered_evidence = [evidence[index] for index in order]
    reordered_items = []
    for new_index, old_index in enumerate(order):
        item = dict(items[old_index])
        item["evidence_id"] = f"EVIDENCE_IMAGE_{new_index:03d}"
        reordered_items.append(item)
    sampling["items"] = reordered_items
    sampling["count"] = len(reordered_evidence)
    observation["timeline_fallback_evidence"] = {
        "selection_rule": (
            "all temporal sheets and generic tail UI zooms, then evenly sampled "
            "individual frames within the fixed image budget"
        ),
        "image_budget": TIMELINE_FALLBACK_IMAGE_BUDGET,
        "selected_count": budget,
        "temporal_sheet_count": len(temporal),
        "tail_zoom_count": len(tail),
        "uses_task_id": False,
        "uses_human_labels_or_answers": False,
        "uses_historical_predictions_or_metrics": False,
    }
    return reordered_evidence


def build_evidence(task, task_dir, max_images, workers, max_temporal_sheets):
    observation, evidence = _ORIGINAL_BUILD_EVIDENCE(
        task, task_dir, max_images, workers, max_temporal_sheets
    )
    task_spec = task["task_spec"]
    if needs_tail_zoom(task_spec):
        duration = float(observation["recording_duration_s"])
        video = Path(task["evidence_video"]["path"])
        evidence_dir = task_dir / "evidence"
        additions = []
        for offset in TAIL_OFFSETS_FROM_END_S:
            timestamp = round(max(0.0, duration - offset), 3)
            out_path = evidence_dir / f"tail_zoom_right_ui_{timestamp:010.3f}s.jpg"
            write_tail_zoom(video, timestamp, out_path)
            additions.append((timestamp, out_path, "right_ui_tail_zoom"))

        evidence = list(evidence) + additions
        sampling = observation["screenshot_sampling"]
        start_index = len(sampling["items"])
        for offset, (timestamp, _, view) in enumerate(additions):
            sampling["items"].append(
                {
                    "evidence_id": f"EVIDENCE_IMAGE_{start_index + offset:03d}",
                    "time_s": timestamp,
                    "view": view,
                    "clock": "recording wall time",
                }
            )
        sampling["count"] = len(evidence)
        observation["tail_zoom_evidence"] = {
            "selection_rule": (
                "D5-applicable public task spec with at least two explicit R-numbered rounds"
            ),
            "offsets_from_recording_end_s": list(TAIL_OFFSETS_FROM_END_S),
            "instruction": (
                "Compare the independent final UI crops with the last temporal sheet. "
                "Do not infer a response merely because task_spec lists an expected answer."
            ),
            "uses_task_id": False,
            "uses_human_labels_or_answers": False,
        }

    evidence = reorder_for_timeline_fallback(observation, evidence)
    return observation, evidence


def evidence_variants(frames):
    yield "complete_evidence", frames
    priority = frames[: min(TIMELINE_FALLBACK_IMAGE_BUDGET, len(frames))]
    if priority != frames:
        yield "timeline_priority_fallback", priority
    for variant_name, variant_frames in _ORIGINAL_EVIDENCE_VARIANTS(frames):
        if variant_name == "complete_evidence" or variant_frames == priority:
            continue
        yield variant_name, variant_frames


_implementation.build_evidence = build_evidence
_implementation.evidence_variants = evidence_variants

DEFAULT_API_URL = _implementation.DEFAULT_API_URL
DEFAULT_MANIFEST = _implementation.DEFAULT_MANIFEST
DEFAULT_MODEL = _implementation.DEFAULT_MODEL
apply_dimension_mean_score = _implementation.apply_dimension_mean_score
call_gpt = _implementation.call_gpt
load_api_key = _implementation.load_api_key
parse_json_object = _implementation.parse_json_object
validate_prediction = _implementation.validate_prediction
write_json = _implementation.write_json
judge_task = _implementation.judge_task


def main():
    _implementation.main()


if __name__ == "__main__":
    main()
