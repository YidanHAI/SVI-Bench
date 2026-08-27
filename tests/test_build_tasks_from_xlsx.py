import json
import sys
from pathlib import Path

from openpyxl import Workbook


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_tasks_from_xlsx as builder  # noqa: E402


def make_public_workbook(path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "题目池"
    sheet.append([
        "id",
        "分类",
        "场景标签",
        "用户 query",
        "query发送时间",
        "视频时长",
    ])
    sheet.append([
        "A1001",
        "监控预警",
        "合成场景一",
        "看到目标时提醒我",
        "1s",
        "10s",
    ])
    sheet.append([
        "B3001",
        "时间感知",
        "合成场景二",
        "R1：开始计时\nR2：画面是什么",
        "R1：1s\nR2：5s",
        "20s",
    ])
    workbook.save(path)


def test_public_workbook_rows_do_not_require_selection_column(tmp_path):
    workbook = tmp_path / "SVIBench.xlsx"
    make_public_workbook(workbook)

    rows = builder.load_selected_rows(workbook, "题目池")

    assert [row["id"] for row in rows] == ["A1001", "B3001"]
    assert [row["_source_row"] for row in rows] == [2, 3]


def test_builder_matches_id_named_media_and_optional_video_map(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    video_dir = data_dir / "videos"
    video_dir.mkdir(parents=True)
    workbook = data_dir / "SVIBench.xlsx"
    make_public_workbook(workbook)
    (video_dir / "A1001.mp4").write_bytes(b"first")
    (video_dir / "descriptive-name.mp4").write_bytes(b"second")
    video_map = data_dir / "media_index.jsonl"
    video_map.write_text(
        json.dumps({"id": "B3001", "video": "videos/descriptive-name.mp4"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "tasks.jsonl"
    report = tmp_path / "report.json"
    monkeypatch.setattr(builder, "ffprobe_duration", lambda _path: None)
    monkeypatch.setattr(sys, "argv", [
        "build_tasks_from_xlsx.py",
        "--xlsx", str(workbook),
        "--sheet", "题目池",
        "--video-dir", str(video_dir),
        "--video-map", str(video_map),
        "--out", str(output),
        "--report", str(report),
    ])

    builder.main()

    tasks = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [task["id"] for task in tasks] == ["A1001", "B3001"]
    assert Path(tasks[0]["local_video_path"]).name == "A1001.mp4"
    assert Path(tasks[1]["local_video_path"]).name == "descriptive-name.mp4"
    assert [round_["query_time_s"] for round_ in tasks[1]["queries"]] == [1.0, 5.0]
    assert json.loads(report.read_text(encoding="utf-8"))["missing_count"] == 0
