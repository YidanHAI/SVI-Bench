#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import unicodedata
from datetime import date, datetime
from pathlib import Path

from judge_workbook_contract import load_workbook_compatible


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = ROOT / "data" / "SVI_bench_tasks_and_anchors.xlsx"
DEFAULT_VIDEO_DIR = ROOT / "data" / "source_videos"
DEFAULT_SHEET = "题目池"
REQUIRED_HEADERS = (
    "id",
    "分类",
    "场景标签",
    "用户 query",
    "query发送时间",
    "视频时长",
)
VIDEO_PATH_FIELDS = (
    "video",
    "video_path",
    "local_video_path",
    "path",
    "filename",
)


def clean_cell(value):
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
        return value if value else None
    return value


def normalize_name(value):
    text = str(value or "")
    text = text.replace("📄", "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\s_\-—–]+", "", text)
    return text.lower()


def strip_video_label(value):
    text = str(value or "").strip()
    text = text.replace("📄", "").strip()
    return text or None


def parse_seconds(value, default=None):
    value = clean_cell(value)
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    if "视频开始" in text or "开始就发送" in text:
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)\s*(s|秒|sec|second)?", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default


def normalize_round_label(value):
    text = str(value or "").strip().upper()
    match = re.search(r"R?\s*(\d+)", text)
    if not match:
        return text
    return f"R{int(match.group(1))}"


def split_labeled_text(value, allow_numbered=False):
    text = clean_cell(value)
    if text is None:
        return []
    text = str(text).strip()
    patterns = [r"(?im)(?:^|\n|\s)(R\d+)\s*[：:]"]
    if allow_numbered:
        patterns.append(r"(?m)(?:^|\n|\s)(\d+)\s*[\.．、]")
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if len(matches) < 2:
            continue
        parts = []
        for index, match in enumerate(matches):
            label = normalize_round_label(match.group(1))
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if content:
                parts.append((label, content))
        if parts:
            return parts
    return []


def parse_query_rounds(query_value, time_value, default_query_time_s=0.0):
    query_text = clean_cell(query_value) or ""
    time_text = clean_cell(time_value)
    time_parts = split_labeled_text(time_text)
    query_parts = split_labeled_text(query_text, allow_numbered=True)

    if not time_parts and not query_parts:
        return [{
            "id": "R1",
            "query": str(query_text).strip(),
            "query_time_s": parse_seconds(time_text, default_query_time_s),
        }]

    time_by_label = {
        label: parse_seconds(content, default_query_time_s)
        for label, content in time_parts
    }
    ordered_times = [parse_seconds(content, default_query_time_s) for _, content in time_parts]

    if query_parts:
        rounds = []
        for index, (label, content) in enumerate(query_parts):
            query_time_s = time_by_label.get(label)
            if query_time_s is None and index < len(ordered_times):
                query_time_s = ordered_times[index]
            if query_time_s is None:
                query_time_s = parse_seconds(time_text, default_query_time_s)
            rounds.append({
                "id": label or f"R{index + 1}",
                "query": content,
                "query_time_s": query_time_s,
            })
        return rounds

    rounds = []
    for index, (label, content) in enumerate(time_parts):
        rounds.append({
            "id": label or f"R{index + 1}",
            "query": str(query_text).strip(),
            "query_time_s": parse_seconds(content, default_query_time_s),
        })
    return rounds


def ffprobe_duration(path):
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nk=1:nw=1",
                str(path),
            ],
            text=True,
            timeout=20,
        ).strip()
        return float(output)
    except Exception:
        return None


def iter_video_files(video_dir):
    for path in sorted(Path(video_dir).rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in {".mp4", ".mov", ".m4v"}:
            continue
        yield path


def build_video_index(video_dir):
    index = {}
    duplicates = {}
    for path in iter_video_files(video_dir):
        key = normalize_name(path.stem)
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(path)
        else:
            index[key] = path
    return index, duplicates


def load_selected_rows(xlsx_path, sheet_name):
    wb = load_workbook_compatible(xlsx_path, read_only=True, data_only=True)
    try:
        if sheet_name not in wb.sheetnames:
            raise ValueError(f"Sheet {sheet_name!r} not found in {xlsx_path}")
        ws = wb[sheet_name]
        headers = [clean_cell(ws.cell(1, col).value) for col in range(1, ws.max_column + 1)]
        missing_headers = [name for name in REQUIRED_HEADERS if name not in headers]
        if missing_headers:
            raise ValueError(
                f"Sheet {sheet_name!r} is missing required columns: {missing_headers}"
            )
        has_selection_column = "入选状态" in headers
        rows = []
        seen_ids = set()
        for row_idx in range(2, ws.max_row + 1):
            row = {
                headers[col - 1]: clean_cell(ws.cell(row_idx, col).value)
                for col in range(1, ws.max_column + 1)
                if headers[col - 1]
            }
            task_id = str(row.get("id") or "").strip()
            if not task_id:
                continue
            if has_selection_column and str(row.get("入选状态") or "").strip() != "入选":
                continue
            if task_id in seen_ids:
                raise ValueError(f"Duplicate task id {task_id!r} at row {row_idx}")
            seen_ids.add(task_id)
            row["_source_row"] = row_idx
            rows.append(row)
        return rows
    finally:
        wb.close()


def _read_video_map_rows(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("items", "videos", "tasks"):
            if isinstance(value.get(key), list):
                return value[key]
        return [{"id": task_id, "video": video} for task_id, video in value.items()]
    raise ValueError(f"Unsupported video map structure: {path}")


def load_video_map(path, video_dir):
    if path is None:
        return {}
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video map does not exist: {path}")
    video_dir = Path(video_dir).resolve()
    result = {}
    for index, item in enumerate(_read_video_map_rows(path), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Video map row {index} is not an object")
        task_id = str(item.get("id") or item.get("task_id") or "").strip()
        raw_video = next(
            (item.get(field) for field in VIDEO_PATH_FIELDS if item.get(field)),
            None,
        )
        if not task_id or raw_video is None:
            raise ValueError(f"Video map row {index} needs id and video path")
        if task_id in result:
            raise ValueError(f"Duplicate task id {task_id!r} in video map")
        candidate = Path(str(raw_video))
        if not candidate.is_absolute():
            relative_to_map = (path.parent / candidate).resolve()
            relative_to_video_dir = (video_dir / candidate).resolve()
            candidate = (
                relative_to_map if relative_to_map.is_file() else relative_to_video_dir
            )
        result[task_id] = candidate
    return result


def find_video(row, video_index, video_map=None):
    task_id = str(row.get("id") or "").strip()
    mapped = (video_map or {}).get(task_id)
    tried = []
    if mapped is not None:
        tried.append(str(mapped))
        if mapped.is_file():
            return mapped, tried
    candidates = [
        strip_video_label(row.get("query视频")),
        row.get("场景标签"),
        row.get("id"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        key = normalize_name(candidate)
        tried.append(candidate)
        if key in video_index:
            return video_index[key], tried
    return None, tried


def build_task(row, video_path, duration, query_rounds):
    first_round = query_rounds[0]
    task = {
        "id": str(row.get("id")).strip(),
        "category": row.get("分类"),
        "scene": row.get("场景标签"),
        "local_video_path": str(video_path),
        "query": first_round["query"],
        "query_time_s": first_round["query_time_s"],
        "queries": query_rounds,
        "duration_s": duration,
        "excel_query_send_time": row.get("query发送时间"),
        "excel_video_duration": row.get("视频时长"),
        "excel_query_video": row.get("query视频"),
        "excel_source_row": row.get("_source_row"),
        "excel_query_round_count": len(query_rounds),
    }
    return {key: value for key, value in task.items() if value is not None}


def main():
    parser = argparse.ArgumentParser(description="Build capture tasks from the SVI-Bench workbook and local MP4 files.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument(
        "--video-map",
        type=Path,
        help=(
            "Optional JSON/JSONL mapping from task id to video path. "
            "Without it, videos are matched by legacy query-video label, scene, then id."
        ),
    )
    parser.add_argument("--out", type=Path, default=ROOT / "tasks.local.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "tasks.local.report.json")
    parser.add_argument("--default-query-time-s", type=float, default=0.0)
    parser.add_argument("--default-duration-s", type=float, default=45.0)
    parser.add_argument("--duration-cap-s", type=float, default=0.0, help="Optional duration cap; 0 keeps the full duration.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-id", action="append", dest="task_ids", help="Include only these Excel ids.")
    parser.add_argument("--task-list", type=Path, help="Text/markdown file containing task ids, one id per line.")
    parser.add_argument("--category", action="append", dest="categories", help="Include only these categories.")
    parser.add_argument("--allow-missing", action="store_true", help="Write matched tasks even if some selected rows are missing videos.")
    args = parser.parse_args()

    video_index, duplicates = build_video_index(args.video_dir)
    video_map = load_video_map(args.video_map, args.video_dir)
    rows = load_selected_rows(args.xlsx, args.sheet)
    task_ids_from_file = []
    if args.task_list:
        for line in args.task_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task_ids_from_file.append(re.split(r"[\s,，]+", line)[0])
    if args.task_ids:
        wanted = set(args.task_ids)
        rows = [row for row in rows if str(row.get("id")) in wanted]
    if task_ids_from_file:
        wanted = set(task_ids_from_file)
        rows = [row for row in rows if str(row.get("id")) in wanted]
    if args.categories:
        wanted_categories = set(args.categories)
        rows = [row for row in rows if row.get("分类") in wanted_categories]

    tasks = []
    missing = []
    for row in rows:
        video_path, tried = find_video(row, video_index, video_map)
        if not video_path:
            missing.append({
                "id": row.get("id"),
                "category": row.get("分类"),
                "scene": row.get("场景标签"),
                "query_video": row.get("query视频"),
                "tried": tried,
                "source_row": row.get("_source_row"),
            })
            continue

        query_rounds = parse_query_rounds(
            row.get("用户 query"),
            row.get("query发送时间"),
            args.default_query_time_s,
        )
        query_rounds = [
            {
                "id": str(item.get("id") or f"R{index + 1}"),
                "query": clean_cell(item.get("query")),
                "query_time_s": round(float(item.get("query_time_s")), 3),
            }
            for index, item in enumerate(query_rounds)
            if clean_cell(item.get("query")) is not None and item.get("query_time_s") is not None
        ]
        if not query_rounds:
            missing.append({
                "id": row.get("id"),
                "reason": "missing user query",
                "source_row": row.get("_source_row"),
            })
            continue

        duration = parse_seconds(row.get("视频时长"), None)
        probed_duration = ffprobe_duration(video_path)
        if duration is None:
            duration = probed_duration or args.default_duration_s
        if args.duration_cap_s and args.duration_cap_s > 0:
            duration = min(duration, args.duration_cap_s)

        task = build_task(row, video_path, round(float(duration), 3), query_rounds)
        task["local_video_duration_s"] = round(float(probed_duration), 3) if probed_duration else None
        tasks.append(task)
        if args.limit and len(tasks) >= args.limit:
            break

    if missing and not args.allow_missing and not args.limit and not args.task_ids and not args.categories:
        print(f"WARNING: {len(missing)} selected rows did not match local videos; writing matched tasks anyway.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    report = {
        "xlsx": str(args.xlsx),
        "sheet": args.sheet,
        "video_dir": str(args.video_dir),
        "video_map": str(args.video_map) if args.video_map else None,
        "selected_rows": len(rows),
        "tasks_written": len(tasks),
        "unique_video_count": len({task["local_video_path"] for task in tasks}),
        "multi_round_tasks": sum(1 for task in tasks if len(task.get("queries") or []) > 1),
        "missing_count": len(missing),
        "missing": missing,
        "duplicate_video_keys": {
            key: [str(path) for path in paths]
            for key, paths in duplicates.items()
        },
        "out": str(args.out),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "tasks_written": len(tasks),
        "missing_count": len(missing),
        "out": str(args.out),
        "report": str(args.report),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
