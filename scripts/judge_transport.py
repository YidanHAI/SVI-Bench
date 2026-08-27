#!/usr/bin/env python3
"""Grade recordings with timestamped visual evidence through a chat API."""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

from judge_api import DEFAULT_JUDGE_MODEL, chat_completions_url, load_api_key
from judge_scoring import apply_dimension_mean_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "judge_manifest.json"
DEFAULT_PROMPT = ROOT / "prompts" / "judge_first_pass.md"
DEFAULT_OUT = ROOT / "outputs" / "judge" / "first_pass"
DEFAULT_API_URL = ""
DEFAULT_MODEL = DEFAULT_JUDGE_MODEL


def run(command):
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def probe_duration(video):
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video),
    ])
    return float(result.stdout.strip())


def probe_last_video_frame_time(video, duration=None):
    duration = probe_duration(video) if duration is None else float(duration)
    start = max(0.0, duration - 10.0)
    result = run([
        "ffprobe", "-v", "error", "-read_intervals", f"{start:.3f}%",
        "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(video),
    ])
    timestamps = []
    for line in result.stdout.splitlines():
        try:
            timestamps.append(float(line.strip().split(",", 1)[0]))
        except (TypeError, ValueError):
            continue
    if not timestamps:
        raise RuntimeError(f"Could not locate a decodable video frame in {video}")
    return max(timestamps)


def is_silence(text):
    normalized = re.sub(r"\s+", "", str(text or "")).lower()
    return normalized in {"", "</silence>", "<silence>", "silence", "null", "none"}


def clean_response(text):
    return re.sub(r"^\s*</?response>\s*", "", str(text or "")).strip()


def parse_frame_time(value):
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group(0)) if match else None


def numeric_summary(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "max": round(max(values), 3),
    }


def video_time(t_ms, origin_s):
    if t_ms is None or origin_s is None:
        return None
    return round(float(t_ms) / 1000.0 - float(origin_s), 3)


def dedupe_events(events):
    result = []
    seen = set()
    for event in events:
        key = (event.get("type"), event.get("text"), round(event.get("time_s") or -1, 2))
        if key in seen:
            continue
        seen.add(key)
        result.append(event)
    return result


def load_timeline(summary_path):
    if not summary_path:
        return None
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    origin_s = summary.get("original_video_start_offset_s")
    if origin_s is None:
        origin_s = summary.get("stream_ready_offset_s")

    queries = []
    for event in summary.get("prompt_events") or []:
        payload = event.get("payload") or {}
        if event.get("direction") != "sent" or payload.get("type") != "update_prompt":
            continue
        queries.append({
            "type": "query",
            "time_s": video_time(event.get("t_ms"), origin_s),
            "text": payload.get("prompt"),
        })

    outputs = []
    silence_count = 0
    for item in summary.get("vlm_responses") or []:
        text = item.get("text")
        if is_silence(text):
            silence_count += 1
            continue
        metrics = item.get("metrics") or {}
        request_payload = item.get("request_payload") or {}
        outputs.append({
            "type": "model_output",
            "time_s": video_time(item.get("t_ms"), origin_s),
            "recording_wall_time_s": video_time(item.get("t_ms"), origin_s),
            "text": clean_response(text),
            "latency_ms": metrics.get("last_latency_ms"),
            "model_reported_latency_ms": metrics.get("last_latency_ms"),
            "active_user_prompt": metrics.get("user_prompt"),
            "frame_time_range": request_payload.get("frame_time_range"),
            "analyzer_frame_clock_s": parse_frame_time(request_payload.get("frame_time_range")),
        })

    queries = dedupe_events(queries)
    outputs = dedupe_events(outputs)
    wall_intervals = [
        current["time_s"] - previous["time_s"]
        for previous, current in zip(outputs, outputs[1:])
        if previous.get("time_s") is not None and current.get("time_s") is not None
    ]
    frame_times = [item.get("analyzer_frame_clock_s") for item in outputs]
    frame_intervals = [
        current - previous
        for previous, current in zip(frame_times, frame_times[1:])
        if previous is not None and current is not None
    ]
    latencies = [item.get("latency_ms") for item in outputs]
    first_output_after_queries = []
    for index, query in enumerate(queries):
        start = query.get("time_s")
        next_start = queries[index + 1].get("time_s") if index + 1 < len(queries) else None
        match = next((
            item for item in outputs
            if item.get("time_s") is not None
            and start is not None
            and item["time_s"] >= start
            and (next_start is None or item["time_s"] < next_start)
        ), None)
        first_output_after_queries.append({
            "query_time_s": start,
            "query": query.get("text"),
            "first_output_time_s": match.get("time_s") if match else None,
            "wall_delay_s": round(match["time_s"] - start, 3) if match and start is not None else None,
            "model_reported_latency_ms": match.get("latency_ms") if match else None,
            "text": match.get("text") if match else None,
        })

    query_rounds = []
    for event in summary.get("query_events") or []:
        query_time = event.get("actual_query_video_time_s")
        response_text = clean_response(event.get("final_response_text"))
        matching_output = next((
            item for item in outputs
            if response_text and item.get("text") == response_text
        ), None)
        query_rounds.append({
            "round_id": event.get("id"),
            "query": event.get("query"),
            "query_recording_wall_time_s": query_time,
            "final_response_received": event.get("final_response_received"),
            "end_to_end_response_wait_s": event.get("final_response_wait_s"),
            "model_reported_latency_ms": (
                matching_output.get("model_reported_latency_ms") if matching_output else None
            ),
            "response_text": response_text or None,
        })

    ready_status = ((summary.get("local_video_upload") or {}).get("ready_status") or {})
    ready_playback = ready_status.get("playbackState") or {}
    playback_health = summary.get("upload_playback_health") or {}
    playback_alignment = {
        "input_mode": summary.get("input_mode"),
        "source_media_time_at_recording_start_s": ready_playback.get("current_time_s"),
        "paused_at_start": ready_playback.get("paused"),
        "loop_count_during_recording": playback_health.get("loop_count"),
        "max_media_stall_ms": playback_health.get("max_media_stall_ms"),
        "source_media_and_recording_wall_clock_aligned": bool(
            summary.get("input_mode") == "upload"
            and ready_playback.get("paused") is False
            and float(ready_playback.get("current_time_s") or 0.0) < 0.25
            and not playback_health.get("loop_count")
        ),
    }

    return {
        "capture_status": summary.get("status"),
        "timing_precision": "capture_log",
        "time_origin": "recording t=0",
        "queries": queries,
        "query_rounds": query_rounds,
        "model_outputs": outputs,
        "playback_alignment": playback_alignment,
        "derived_timing": {
            "model_output_count": len(outputs),
            "wall_start_interval_s": numeric_summary(wall_intervals),
            "wall_start_intervals_s": [round(value, 3) for value in wall_intervals],
            "analyzer_frame_interval_s": numeric_summary(frame_intervals),
            "analyzer_frame_intervals_s": [round(value, 3) for value in frame_intervals],
            "model_reported_latency_ms": numeric_summary(latencies),
            "first_output_after_each_query": first_output_after_queries,
            "warning": (
                "recording wall time, analyzer-internal frame clock, and model-reported latency are "
                "distinct clocks. analyzer_frame_clock_s is not source-media playback time and must "
                "never be subtracted from source-video or recording timestamps."
            ),
        },
        "silence_observation_count": silence_count,
        "raw_response_observation_count": len(summary.get("vlm_responses") or []),
    }


def evenly_sample(items, count):
    if len(items) <= count:
        return list(items)
    if count <= 1:
        return [items[0]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indices]


def choose_frame_times(duration, timeline, max_images, last_video_frame_s=None):
    duration = max(duration, 0.1)
    # Individual ffmpeg snapshots preserve the release sampling schedule.
    # The exact last-frame bound is required only by OpenCV temporal seeking.
    tail_margin = min(0.5, max(0.05, duration * 0.1))
    end = duration - tail_margin
    end = max(0.05, end)
    if timeline is None:
        if duration > 300:
            uniform = [0.5 + index * (end - 0.5) / 15 for index in range(16)]
            dense_start = max(0.5, duration - 28.0)
            dense = [dense_start + index for index in range(math.floor(end - dense_start) + 1)]
            times = uniform + dense
        else:
            stride = max(1.0, duration / max(1, max_images - 2))
            times = [0.5]
            cursor = 1.0
            while cursor < end:
                times.append(cursor)
                cursor += stride
            times.append(end)
        return sorted({round(min(end, max(0.0, value)), 3) for value in times})[:max_images]

    uniform_count = min(18, max(8, max_images // 3))
    uniform = [0.5 + index * (end - 0.5) / max(1, uniform_count - 1) for index in range(uniform_count)]
    query_times = [item["time_s"] for item in timeline["queries"] if item.get("time_s") is not None]
    output_times = [item["time_s"] for item in timeline["model_outputs"] if item.get("time_s") is not None]
    output_times = evenly_sample(output_times, min(18, max_images // 2))
    event_times = []
    for value in query_times + output_times:
        event_times.extend([value - 0.6, value, value + 0.8])

    prioritized = []
    for value in event_times + uniform:
        value = round(min(end, max(0.0, value)), 3)
        if all(abs(value - existing) >= 0.18 for existing in prioritized):
            prioritized.append(value)
        if len(prioritized) >= max_images:
            break
    return sorted(prioritized)


def font_for(image_width):
    size = max(24, round(image_width / 55))
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def extract_frame(video, timestamp, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(".raw.jpg")
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1",
        "-vf", "scale='min(1600,iw)':-2", "-q:v", "3", str(temporary),
    ])
    with Image.open(temporary) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = font_for(image.width)
        label = f"t={timestamp:.3f}s"
        box = draw.textbbox((0, 0), label, font=font)
        width = box[2] - box[0]
        height = box[3] - box[1]
        pad = max(8, round(font.size * 0.3)) if hasattr(font, "size") else 8
        draw.rectangle((8, 8, 8 + width + 2 * pad, 8 + height + 2 * pad), fill=(0, 0, 0))
        draw.text((8 + pad, 8 + pad), label, fill=(255, 255, 0), font=font)
        image.save(out_path, format="JPEG", quality=82, optimize=True)
    temporary.unlink(missing_ok=True)


def make_video_crop(full_path, crop_path, timestamp):
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(full_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        # The recording layout keeps the source video in the left pane.
        left = round(width * 0.045)
        top = round(height * 0.075)
        right = round(width * 0.56)
        bottom = round(height * 0.92)
        crop = image.crop((left, top, right, bottom))
        if crop.width < 1600:
            new_height = round(crop.height * 1600 / crop.width)
            crop = crop.resize((1600, new_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(crop)
        font = font_for(crop.width)
        label = f"t={timestamp:.3f}s video_crop"
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        pad = max(8, round(font.size * 0.3)) if hasattr(font, "size") else 8
        draw.rectangle((8, 8, 8 + text_width + 2 * pad, 8 + text_height + 2 * pad), fill=(0, 0, 0))
        draw.text((8 + pad, 8 + pad), label, fill=(0, 255, 255), font=font)
        crop.save(crop_path, format="JPEG", quality=84, optimize=True)


def choose_crop_times(full_times, timeline, crop_budget):
    if crop_budget <= 0 or not full_times:
        return []
    if timeline is None:
        return evenly_sample(full_times, crop_budget)

    candidates = []
    for output in timeline.get("model_outputs") or []:
        timestamp = output.get("time_s")
        if timestamp is None:
            continue
        candidates.extend([timestamp - 0.6, timestamp, timestamp + 0.8])
    if not candidates:
        candidates = full_times

    selected = []
    for candidate in evenly_sample(candidates, crop_budget):
        nearest = min(full_times, key=lambda value: abs(value - candidate))
        if nearest not in selected:
            selected.append(nearest)
    return sorted(selected)


def query_segments(duration, timeline, before=1.0, after=4.0):
    if timeline:
        centers = [
            item.get("time_s")
            for item in timeline.get("queries") or []
            if item.get("time_s") is not None
        ]
        if centers:
            return [
                (max(0.0, center - before), min(duration, center + after))
                for center in centers
            ]
    if duration <= 120:
        return [(0.0, duration)]
    return [(0.0, min(35.0, duration)), (max(0.0, duration - 45.0), duration)]


def temporal_plans(task_spec, recording_duration, timeline, source_duration=None):
    """Choose evidence density from public task metadata, never from task IDs."""
    category = str(task_spec.get("category") or "")
    dimensions = task_spec.get("dimensions") or {}
    d1 = bool((dimensions.get("D1") or {}).get("applicable"))
    d5 = bool((dimensions.get("D5") or {}).get("applicable"))
    plans = []

    media_view = "source_video" if source_duration is not None else "left_video"
    media_end = min(source_duration, recording_duration) if source_duration is not None else recording_duration
    if media_end > 0:
        if category == "实时计数":
            fps, max_frames = 4.0, 240
        elif category == "实时翻译":
            fps, max_frames = 2.0, 240
        elif d1 and media_end <= 180:
            fps, max_frames = 2.0, 240
        elif d1:
            fps, max_frames = 1.0, 240
        else:
            fps, max_frames = 0.5, 180
        media_segments = (
            [(0.0, media_end)]
            if d1 else query_segments(media_end, timeline, before=35.0, after=8.0)
        )
        if category == "App引导":
            media_sheet_cap = 8
        elif category in {"实时计数", "实时翻译"}:
            media_sheet_cap = 14
        else:
            media_sheet_cap = 10
        plans.append({
            "view": media_view,
            "fps": fps,
            "segments": media_segments,
            "max_frames": max_frames,
            "tiles": 12,
            "max_sheets": media_sheet_cap,
        })

    needs_dense_ui = category in {"App引导", "智能体委托"} or d5 or source_duration is None
    if needs_dense_ui:
        full_ui = category == "App引导" and recording_duration <= 120
        segments = (
            [(0.0, recording_duration)]
            if full_ui else query_segments(recording_duration, timeline)
        )
        plans.append({
            "view": "right_ui",
            "fps": 4.0 if d5 else 2.0,
            "segments": segments,
            "max_frames": 120,
            "tiles": 6,
            "max_sheets": 10,
        })
    return plans


def sample_times_from_plan(plan, duration, last_video_frame_s=None):
    values = []
    step = 1.0 / plan["fps"]
    if last_video_frame_s is None:
        tail_margin = min(0.5, max(0.05, duration * 0.1))
        decodable_end = duration - tail_margin
    else:
        decodable_end = min(duration, float(last_video_frame_s)) - 0.01
    decodable_end = max(0.0, decodable_end)
    for start, end in plan["segments"]:
        cursor = max(0.0, start)
        end = min(decodable_end, end)
        while cursor <= end:
            values.append(round(cursor, 3))
            cursor += step
    values = sorted(set(values))
    return evenly_sample(values, plan["max_frames"])


def crop_for_view(image, view):
    width, height = image.size
    if view == "right_ui":
        box = (round(width * 0.56), round(height * 0.04), round(width * 0.995), round(height * 0.94))
    elif view.startswith("source_video"):
        return image
    else:
        box = (round(width * 0.04), round(height * 0.07), round(width * 0.56), round(height * 0.93))
    return image.crop(box)


def build_temporal_sheet(tiles, out_path, view):
    columns = 2 if view == "right_ui" else 3
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (tiles[0].width * columns, tiles[0].height * rows), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    sheet.save(out_path, format="JPEG", quality=82, optimize=True)


def load_temporal_tiles(video, timestamps, view):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video with OpenCV: {video}")
    target_width = 760 if view == "right_ui" else 600
    target_height = 620 if view == "right_ui" else 500
    tiles = []
    try:
        for index, timestamp in enumerate(timestamps):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if ok:
                opened = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                # OpenCV random seeks can fail near the tail of valid VFR MP4s.
                # ffmpeg's decoder is slower but handles those files reliably.
                decoded = subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{timestamp:.3f}", "-i", str(video),
                    "-frames:v", "1", "-an", "-f", "image2pipe",
                    "-vcodec", "png", "pipe:1",
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if decoded.returncode != 0 or not decoded.stdout:
                    raise RuntimeError(f"Could not read {video} at t={timestamp:.3f}s")
                with Image.open(io.BytesIO(decoded.stdout)) as image:
                    opened = image.convert("RGB")
            tile = crop_for_view(opened, view)
            tile.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (target_width, target_height), "white")
            canvas.paste(tile, ((target_width - tile.width) // 2, (target_height - tile.height) // 2))
            draw = ImageDraw.Draw(canvas)
            font = font_for(target_width)
            label = f"t={timestamp:.3f}s"
            box = draw.textbbox((0, 0), label, font=font)
            draw.rectangle((0, 0, box[2] + 12, box[3] + 12), fill=(0, 0, 0))
            draw.text((6, 6), label, fill=(255, 255, 0), font=font)
            tiles.append(canvas)
    finally:
        capture.release()
    return tiles


def build_temporal_evidence(
    task_spec, video, duration, timeline, evidence_dir, max_sheets, source_video=None,
    recording_last_frame_s=None,
):
    if max_sheets <= 0:
        return [], None
    source_duration = probe_duration(source_video) if source_video else None
    source_last_frame_s = (
        probe_last_video_frame_time(source_video, source_duration)
        if source_video else None
    )
    plans = temporal_plans(task_spec, duration, timeline, source_duration)
    if not plans:
        return [], None
    items = []
    metadata = []
    remaining = max_sheets
    for plan in plans:
        if remaining <= 0:
            break
        is_source = plan["view"].startswith("source_video")
        plan_duration = source_duration if is_source else duration
        plan_video = source_video if is_source else video
        plan_last_frame_s = source_last_frame_s if is_source else recording_last_frame_s
        timestamps = sample_times_from_plan(plan, plan_duration, plan_last_frame_s)
        sheet_budget = min(remaining, plan["max_sheets"])
        timestamps = evenly_sample(timestamps, sheet_budget * plan["tiles"])
        chunks = [
            timestamps[index:index + plan["tiles"]]
            for index in range(0, len(timestamps), plan["tiles"])
        ][:sheet_budget]
        if not chunks:
            continue
        flat_times = [value for chunk in chunks for value in chunk]
        all_tiles = load_temporal_tiles(plan_video, flat_times, plan["view"])
        tile_offset = 0
        for index, chunk in enumerate(chunks):
            out_path = evidence_dir / f"temporal_v6_{plan['view']}_{index:03d}.jpg"
            if not out_path.exists():
                build_temporal_sheet(
                    all_tiles[tile_offset:tile_offset + len(chunk)], out_path, plan["view"]
                )
            tile_offset += len(chunk)
            items.append((chunk[0], out_path, f"temporal_sheet_{plan['view']}"))
        remaining -= len(chunks)
        metadata.append({
            "view": plan["view"],
            "sampling_fps": plan["fps"],
            "sheet_count": len(chunks),
            "sample_count": len(flat_times),
            "segments_s": plan["segments"],
            "clock": (
                "source-media playback time aligned to recording wall time t=0"
                if is_source else "recording wall time"
            ),
        })
    return items, {
        "plans": metadata,
        "instruction": (
            "Read each sheet left-to-right, top-to-bottom. For exact upload captures, source_video "
            "starts at media t=0 together with recording t=0. left_video/right_ui labels use recording "
            "wall time. No sheet timestamp may be compared with analyzer_frame_clock_s."
        ),
    }


def source_video_from_summary(summary_path):
    if not summary_path:
        return None
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    raw_path = (summary.get("task") or {}).get("local_video_path")
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_file() else None


def aligned_source_video(summary_path, timeline):
    source_video = source_video_from_summary(summary_path)
    if not source_video or not timeline:
        return None
    alignment = timeline.get("playback_alignment") or {}
    if not alignment.get("source_media_and_recording_wall_clock_aligned"):
        return None
    return source_video


def d3_clock_policy(task_spec, timeline):
    d3 = (task_spec.get("dimensions") or {}).get("D3") or {}
    threshold = str(d3.get("threshold") or "")
    category = str(task_spec.get("category") or "")
    explicit_markers = ("query", "输入问题", "提问", "提交", "首次点评启动")
    explicit_query = category in {"App引导", "实时翻译"} or any(
        marker in threshold for marker in explicit_markers
    )
    if explicit_query:
        measurements = []
        if timeline:
            for item in (timeline.get("derived_timing") or {}).get("first_output_after_each_query") or []:
                measurements.append({
                    "query": item.get("query"),
                    "model_reported_latency_ms": item.get("model_reported_latency_ms"),
                    "end_to_end_wall_delay_s_for_cross_check_only": item.get("wall_delay_s"),
                })
        return {
            "mode": "explicit_query_or_startup_model_latency",
            "authoritative_clock": "model_reported_latency_ms when present; otherwise UI Latency",
            "measurements": measurements,
            "instruction": (
                "D3 must use the per-inference model/UI latency as the first-token measure. "
                "Wall delay is cross-check evidence only and includes polling/scheduling overhead."
            ),
        }
    return {
        "mode": "visual_event_end_to_end",
        "authoritative_clock": "recording/source-media aligned event time to recording output time",
        "measurements": [],
        "instruction": (
            "D3 must locate the visual event on an aligned media/recording clock and compare it "
            "with the corresponding output recording time. Analyzer frame clock is forbidden."
        ),
    }


def build_evidence(task, task_dir, max_images, workers, max_temporal_sheets):
    video = Path(task["evidence_video"]["path"])
    duration = probe_duration(video)
    recording_last_frame_s = probe_last_video_frame_time(video, duration)
    timing_source = task.get("timing_source") or {}
    summary_path = timing_source.get("summary_json")
    timeline = load_timeline(summary_path)
    source_video = aligned_source_video(summary_path, timeline)
    crop_budget = min(12, max(0, max_images // 3))
    full_budget = max(1, max_images - crop_budget)
    times = choose_frame_times(
        duration, timeline, full_budget, recording_last_frame_s
    )
    evidence_dir = task_dir / "evidence"
    frames = [evidence_dir / f"frame_{index:03d}_{timestamp:010.3f}s.jpg" for index, timestamp in enumerate(times)]
    jobs = [(timestamp, path) for timestamp, path in zip(times, frames) if not path.exists()]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(extract_frame, video, timestamp, path) for timestamp, path in jobs]
        for future in futures:
            future.result()

    crop_times = choose_crop_times(times, timeline, crop_budget)
    crops = []
    for index, timestamp in enumerate(crop_times):
        full_index = min(range(len(times)), key=lambda item: abs(times[item] - timestamp))
        crop_path = evidence_dir / f"crop_{index:03d}_{timestamp:010.3f}s.jpg"
        if not crop_path.exists():
            make_video_crop(frames[full_index], crop_path, timestamp)
        crops.append((timestamp, crop_path))

    evidence = [(timestamp, path, "full_ui") for timestamp, path in zip(times, frames)]
    evidence.extend((timestamp, path, "video_crop") for timestamp, path in crops)
    evidence.sort(key=lambda item: (item[0], 0 if item[2] == "full_ui" else 1))
    temporal_evidence, temporal_metadata = build_temporal_evidence(
        task["task_spec"], video, duration, timeline, evidence_dir,
        max_temporal_sheets, source_video=source_video,
        recording_last_frame_s=recording_last_frame_s,
    )
    evidence.extend(temporal_evidence)

    observation = {
        "recording_duration_s": round(duration, 3),
        "recording_file": video.name,
        "source_video_available": source_video is not None,
        "source_video_file": source_video.name if source_video else None,
        "timing_precision": "capture_log" if timeline else "sampled_video",
        "clock_definitions": {
            "recording_wall_time_s": "seconds since the task recording begins",
            "source_video_time_s": (
                "source media playback seconds aligned to recording t=0 for exact upload captures"
                if source_video else None
            ),
            "analyzer_frame_clock_s": (
                "internal analyzer observation clock; preserves order but is not media playback time"
                if timeline else None
            ),
            "model_reported_latency_ms": "latency reported by the tested model UI/API for one inference",
        },
        "d3_clock_policy": d3_clock_policy(task["task_spec"], timeline),
        "timeline": timeline,
        "screenshot_sampling": {
            "count": len(evidence),
            "items": [
                {
                    "evidence_id": f"EVIDENCE_IMAGE_{index:03d}",
                    "time_s": timestamp,
                    "view": view,
                    "clock": (
                        "source video time"
                        if view == "source_frame" or view.startswith("temporal_sheet_source_video")
                        else "recording wall time"
                    ),
                }
                for index, (timestamp, _, view) in enumerate(evidence)
            ],
            "warning": "Screenshots are sampled from the complete recording; sampling gaps are not proof of no event.",
        },
        "temporal_evidence": temporal_metadata,
    }
    return observation, evidence


def parse_json_object(text):
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def validate_prediction(prediction, task_spec):
    if not isinstance(prediction, dict):
        return ["response is not a JSON object"]
    errors = []
    if prediction.get("task_id") != task_spec["id"]:
        errors.append("task_id mismatch")
    if prediction.get("judge_status") != "ok":
        errors.append(f"judge_status={prediction.get('judge_status')!r}, expected 'ok'")
    dimensions = prediction.get("dimensions") or {}
    for name, spec in task_spec["dimensions"].items():
        item = dimensions.get(name) or {}
        expected = {"G", "S", "B"} if spec["applicable"] else {"not_applicable"}
        if item.get("grade") not in expected:
            errors.append(f"{name}.grade={item.get('grade')!r}, expected {sorted(expected)}")
    return errors


def image_data_url(path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def motion_edge_image(path):
    target = path.with_name(path.stem + ".motion_edges.jpg")
    if target.exists():
        return target
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read evidence image for edge fallback: {path}")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    line_art = 255 - edges
    if not cv2.imwrite(str(target), line_art, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write edge fallback image: {target}")
    return target


def request_payload(model, system_prompt, user_prompt, frames, max_tokens):
    content = [{"type": "text", "text": user_prompt}]
    for index, (timestamp, frame, view) in enumerate(frames):
        content.append({
            "type": "text",
            "text": f"EVIDENCE_IMAGE_{index:03d} t={timestamp:.3f}s view={view}",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": image_data_url(frame), "detail": "high"},
        })
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "stream": False,
        "max_tokens": max_tokens,
    }
    return payload


def evidence_variants(frames):
    yield "complete_evidence", frames
    individual = [item for item in frames if not item[2].startswith("temporal_sheet_")]
    if individual and individual != frames:
        yield "individual_frames_only", individual
        sparse = evenly_sample(individual, min(12, len(individual)))
        if sparse != individual:
            yield "sparse_individual_frames", sparse
    edge_frames = [
        (timestamp, motion_edge_image(path), f"{view}_edge_motion")
        for timestamp, path, view in frames
    ]
    yield "motion_edge_evidence", edge_frames


def call_gpt(api_url, api_key, model, system_prompt, user_prompt, frames, timeout, max_tokens, retries):
    last_error = None
    request_attempt = 0
    for variant_name, variant_frames in evidence_variants(frames):
        payload = request_payload(model, system_prompt, user_prompt, variant_frames, max_tokens)
        for attempt in range(1, retries + 1):
            request_attempt += 1
            try:
                response = requests.post(
                    api_url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    result = {
                        "status_code": response.status_code,
                        "response_model": data.get("model"),
                        "finish_reason": choice.get("finish_reason"),
                        "content": (message.get("content") or "").strip(),
                        "usage": data.get("usage"),
                        "attempt": request_attempt,
                        "evidence_variant": variant_name,
                        "evidence_image_count_sent": len(variant_frames),
                    }
                    if result["content"] and parse_json_object(result["content"]) is not None:
                        return result
                    last_error = (
                        f"HTTP 200 without a complete JSON object; "
                        f"finish_reason={result['finish_reason']!r}"
                    )
                    if attempt < retries:
                        payload["max_tokens"] = min(20000, max(payload["max_tokens"] * 2, 12000))
                        continue
                    break
                last_error = f"HTTP {response.status_code}: {response.text[:2000]}"
                fallback_rejection = (
                    response.status_code in {400, 413}
                    and any(marker in response.text.lower() for marker in (
                        "content_policy_violation", "content safety", "too many images",
                    ))
                )
                if fallback_rejection:
                    break
                if response.status_code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(last_error)
            except requests.RequestException as exc:
                last_error = repr(exc)
            if attempt < retries:
                time.sleep(min(60, 5 * (2 ** (attempt - 1))))
    raise RuntimeError(last_error or "GPT-5.5 request failed")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def judge_task(args, task, system_prompt, api_key):
    task_spec = task["task_spec"]
    task_id = task_spec["id"]
    task_dir = args.out_dir / task_id
    prediction_path = task_dir / "prediction.json"
    if prediction_path.exists() and not args.force:
        existing = json.loads(prediction_path.read_text(encoding="utf-8"))
        if existing.get("ok"):
            return existing

    observation, frames = build_evidence(
        task, task_dir, args.max_images, args.frame_workers, args.max_temporal_sheets
    )
    judge_input = {"task_spec": task_spec, "observation": observation}
    user_prompt = (
        "请独立评测这条录屏。人工评分未包含在输入中。"
        "严格使用 system prompt 的 JSON 结构。\n\nJUDGE_INPUT_JSON:\n"
        + json.dumps(judge_input, ensure_ascii=False, indent=2)
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "judge_input.json").write_text(
        json.dumps(judge_input, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    response = call_gpt(
        args.api_url, api_key, args.model, system_prompt, user_prompt, frames,
        args.timeout, args.max_tokens, args.retries,
    )
    (task_dir / "raw_response.txt").write_text(response["content"] + "\n", encoding="utf-8")
    prediction = parse_json_object(response["content"])
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
        "native_video_input": False,
        "input_mode": "timestamped_images_temporal_sheets_plus_capture_timeline",
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

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("contains_human_labels") is not False:
        raise RuntimeError("Refusing a manifest that is not explicitly label-free")
    selected = set(args.task_id)
    tasks = [item for item in manifest["tasks"] if not selected or item["task_spec"]["id"] in selected]
    missing = selected - {item["task_spec"]["id"] for item in tasks}
    if missing:
        raise KeyError(f"Unknown task ids: {sorted(missing)}")

    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    args.api_url = args.api_url or chat_completions_url()
    api_key = load_api_key()
    results = []
    for index, task in enumerate(tasks, start=1):
        task_id = task["task_spec"]["id"]
        try:
            result = judge_task(args, task, system_prompt, api_key)
        except Exception as exc:
            result = {"ok": False, "task_id": task_id, "error": repr(exc)}
            write_json(args.out_dir / task_id / "prediction.json", result)
        results.append(result)
        print(json.dumps({
            "index": index,
            "total": len(tasks),
            "task_id": task_id,
            "ok": result.get("ok"),
            "error": result.get("error"),
        }, ensure_ascii=False), flush=True)
        if index < len(tasks):
            time.sleep(args.gap_s)

    write_json(args.out_dir / "batch_summary.json", {
        "manifest": str(args.manifest.resolve()),
        "model": args.model,
        "native_video_input": False,
        "input_mode": "timestamped_images_temporal_sheets_plus_capture_timeline",
        "total": len(results),
        "ok": sum(bool(item.get("ok")) for item in results),
        "failed": sum(not bool(item.get("ok")) for item in results),
        "results": [{key: item.get(key) for key in ("task_id", "ok", "error")} for item in results],
    })


if __name__ == "__main__":
    main()
