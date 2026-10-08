import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "interactflow" / "scripts" / "interactflow.py"
SPEC = importlib.util.spec_from_file_location("interactflow_skill_entry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
interactflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = interactflow
SPEC.loader.exec_module(interactflow)


def make_context(tmp_path):
    video_dir = tmp_path / "data" / "source_videos"
    video_dir.mkdir(parents=True)
    workbook = tmp_path / "data" / "SVI_bench_tasks_and_anchors.xlsx"
    workbook.touch()
    config = {
        "version": 2,
        "webui": {
            "input_mode": "upload",
            "url_env": "JOYVL_WEB_URL",
            "username_env": "JOYVL_WEB_USERNAME",
            "password_env": "JOYVL_WEB_PASSWORD",
            "identity_markers": ["videoFileInput"],
        },
        "network": {"mode": "direct"},
        "frame_scheduler": {
            "max_in_flight": 1,
            "busy_policy": "skip",
            "queue_capacity": 0,
        },
        "vlm_registry": "config/vlm_models.json",
        "task_source": {
            "manifest": "data/recording_tasks_75.jsonl",
            "xlsx": "data/SVI_bench_tasks_and_anchors.xlsx",
            "sheet": "题目池",
            "video_dir": "data/source_videos",
        },
        "models": [
            {"id": model_id, "enabled": True, "vlm_profile": f"profile-{index}"}
            for index, model_id in enumerate(interactflow.EXPECTED_MODELS)
        ],
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "recording_campaign.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    registry = {
        "version": 1,
        "profiles": [
            {
                "id": f"profile-{index}",
                "api_base_env": name,
                "api_key_env": key,
                **(
                    {"realtime_api_base_env": realtime}
                    if realtime
                    else {}
                ),
            }
            for index, (name, key, realtime) in enumerate(
                (
                    ("JOYAI_API_BASE", "JOYAI_API_KEY", ""),
                    ("DOUBAO_API_BASE", "ARK_API_KEY", ""),
                    ("MAGE_API_BASE", "MAGE_API_KEY", "MAGE_REALTIME_API_BASE"),
                    ("MOSS_API_BASE", "MOSS_API_KEY", "MOSS_REALTIME_API_BASE"),
                    (
                        "MODELBEST_API_BASE",
                        "MODELBEST_API_KEY",
                        "MODELBEST_REALTIME_API_BASE",
                    ),
                )
            )
        ],
    }
    (config_dir / "vlm_models.json").write_text(json.dumps(registry), encoding="utf-8")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in interactflow.EXPECTED_PROMPTS:
        (prompts / name).touch()

    env = {name: "secret-value" for name in interactflow.FALLBACK_ENVIRONMENT}
    for name in (
        "JOYVL_WEB_URL",
        "JOYAI_API_BASE",
        "DOUBAO_API_BASE",
        "MAGE_API_BASE",
        "MAGE_REALTIME_API_BASE",
        "MOSS_API_BASE",
        "MOSS_REALTIME_API_BASE",
        "MODELBEST_API_BASE",
        "OPENAI_BASE_URL",
    ):
        env[name] = "https://example.invalid/v1"
    env["MODELBEST_REALTIME_API_BASE"] = "wss://example.invalid/realtime"
    env["JUDGE_MODEL"] = "GPT-5.5"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(f"{name}={value}" for name, value in env.items()) + "\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    return interactflow.Context(
        root=tmp_path,
        env_file=env_file,
        env=env,
        config_path=config_path,
        config=config,
        workbook=workbook,
        sheet="题目池",
        video_dir=video_dir,
        task_manifest=tmp_path / "data" / "recording_tasks_75.jsonl",
        task_report=tmp_path / "data" / "recording_tasks_75.report.json",
        audio_dir=tmp_path / "data" / "minicpmo_query_audio",
        video_map=None,
    )


def test_read_only_check_can_report_ready_without_exposing_values(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    monkeypatch.setattr(interactflow, "build_context", lambda *_args: (context, []))
    monkeypatch.setattr(
        interactflow,
        "executable_version",
        lambda command: (True, "v20.0.0" if command == "node" else "available"),
    )
    monkeypatch.setattr(interactflow, "chromium_available", lambda *_args: True)
    monkeypatch.setattr(
        interactflow,
        "workbook_contract",
        lambda _context: ({"tasks": 75, "formal_ready": True, "warnings": 0}, []),
    )
    monkeypatch.setattr(
        interactflow,
        "run_mapping_check",
        lambda _context: (
            {
                "selected_rows": 75,
                "tasks_written": 75,
                "missing_count": 0,
                "unique_videos": 71,
            },
            [],
        ),
    )

    report, returned_context = interactflow.inspect_setup(tmp_path)

    assert report["status"] == "ready"
    assert returned_context is context
    assert "secret-value" not in json.dumps(report)


def test_failed_preflight_starts_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        interactflow,
        "inspect_setup",
        lambda *_args: (
            {
                "status": "needs_configuration",
                "missing": {"environment_variables": ["OPENAI_API_KEY"]},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        interactflow, "prepare_inputs", lambda _context: calls.append("prepare")
    )

    with pytest.raises(interactflow.InteractFlowError):
        interactflow.action_start(tmp_path, None)

    assert calls == []


def test_explicit_invalid_root_never_falls_back_to_current_checkout(tmp_path):
    with pytest.raises(interactflow.InteractFlowError):
        interactflow.resolve_root(str(tmp_path))


def test_prepare_builds_and_reuses_valid_75_task_inputs(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    videos = []
    for index in range(71):
        path = context.video_dir / f"T{index:03d}.mp4"
        path.write_bytes(b"synthetic-video")
        videos.append(path)

    calls = []

    def fake_run(command, _context, *, capture=False):
        calls.append(Path(command[1]).name)
        if Path(command[1]).name == "build_tasks_from_xlsx.py":
            output = Path(command[command.index("--out") + 1])
            report_path = Path(command[command.index("--report") + 1])
            tasks = [
                {
                    "id": f"T{index:03d}",
                    "local_video_path": str(videos[index % len(videos)]),
                    "query": f"query {index}",
                    "queries": [{"id": "R1", "query": f"query {index}", "query_time_s": 0}],
                }
                for index in range(75)
            ]
            output.write_text(
                "".join(json.dumps(task) + "\n" for task in tasks),
                encoding="utf-8",
            )
            report_path.write_text(
                json.dumps(
                    {
                        "selected_rows": 75,
                        "tasks_written": 75,
                        "missing_count": 0,
                        "duplicate_video_keys": {},
                    }
                ),
                encoding="utf-8",
            )
        else:
            audio_dir = Path(command[command.index("--out-dir") + 1])
            audio_files = audio_dir / "audio"
            audio_files.mkdir(parents=True, exist_ok=True)
            tasks = interactflow.load_tasks(context.task_manifest)
            entries = {}
            for task in tasks:
                query = task["query"]
                query_hash = hashlib.sha256(query.encode()).hexdigest()
                pcm = audio_files / f"{query_hash}.f32le"
                pcm.write_bytes(b"\x00\x00\x00\x00")
                entries[query_hash] = {
                    "query_sha256": query_hash,
                    "pcm_path": f"audio/{pcm.name}",
                    "pcm_sha256": interactflow.sha256(pcm),
                }
            (audio_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "source_tasks_sha256": interactflow.sha256(context.task_manifest),
                        "entries": entries,
                    }
                ),
                encoding="utf-8",
            )
        return None

    monkeypatch.setattr(interactflow, "run_checked", fake_run)

    first = interactflow.prepare_inputs(context)
    second = interactflow.prepare_inputs(context)

    assert first["minicpmo_audio"] == "generated"
    assert second["minicpmo_audio"] == "reused"
    assert calls.count("prepare_minicpmo_query_audio.py") == 1
    assert len(interactflow.load_tasks(context.task_manifest)) == 75
