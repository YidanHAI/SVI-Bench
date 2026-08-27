#!/usr/bin/env python3
"""Safe one-command front door for the released InteractFlow pipeline.

This wrapper performs local, secret-safe setup checks and prepares derived
inputs before delegating execution to the repository's supported entrypoints.
It intentionally contains no recording or Judge implementation of its own.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


EXPECTED_MODELS = (
    "joyai-vl-interaction",
    "doubao-seed-2.1-pro",
    "mage-vl",
    "moss-vl-realtime",
    "minicpmo-4.5-9b-native-video-v2",
)
EXPECTED_PROMPTS = (
    "judge_first_pass.md",
    "judge_review_1.md",
    "judge_review_2.md",
    "judge_adjudication.md",
    "judge_final_review.md",
)
FALLBACK_ENVIRONMENT = (
    "JOYVL_WEB_URL",
    "JOYVL_WEB_USERNAME",
    "JOYVL_WEB_PASSWORD",
    "JOYAI_API_BASE",
    "JOYAI_API_KEY",
    "DOUBAO_API_BASE",
    "ARK_API_KEY",
    "MAGE_API_BASE",
    "MAGE_REALTIME_API_BASE",
    "MAGE_API_KEY",
    "MOSS_API_BASE",
    "MOSS_REALTIME_API_BASE",
    "MOSS_API_KEY",
    "MODELBEST_API_BASE",
    "MODELBEST_REALTIME_API_BASE",
    "MODELBEST_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)
PYTHON_DEPENDENCIES = {
    "aiohttp": "aiohttp",
    "edge_tts": "edge-tts",
    "openai": "openai",
    "openpyxl": "openpyxl",
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "requests": "requests",
}
ROOT_MARKERS = (
    "scripts/run_all.sh",
    "scripts/record_all.sh",
    "scripts/judge_all.sh",
    "config/recording_campaign.json",
)
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class InteractFlowError(RuntimeError):
    """An actionable, secret-safe setup or orchestration error."""


@dataclass
class Context:
    root: Path
    env_file: Path
    env: dict[str, str]
    config_path: Path
    config: dict[str, Any]
    workbook: Path
    sheet: str
    video_dir: Path
    task_manifest: Path
    task_report: Path
    audio_dir: Path
    video_map: Path | None


def repository_markers_exist(path: Path) -> bool:
    return path.is_dir() and all((path / marker).is_file() for marker in ROOT_MARKERS)


def resolve_root(explicit: str | None) -> Path:
    if explicit:
        resolved = Path(explicit).expanduser().resolve()
        if repository_markers_exist(resolved):
            return resolved
        raise InteractFlowError(
            f"The requested repository does not contain the InteractFlow entrypoints: {resolved}"
        )
    configured = os.environ.get("INTERACTFLOW_ROOT", "").strip()
    if configured:
        resolved = Path(configured).expanduser().resolve()
        if repository_markers_exist(resolved):
            return resolved
        raise InteractFlowError(
            "INTERACTFLOW_ROOT does not contain the InteractFlow entrypoints: "
            f"{resolved}"
        )

    candidates: list[Path] = []
    current = Path.cwd().resolve()
    candidates.extend((current, *current.parents))
    script_path = Path(__file__).resolve()
    candidates.extend(script_path.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if repository_markers_exist(resolved):
            return resolved
    raise InteractFlowError(
        "Cannot locate the InteractFlow repository. Run inside the clone or set "
        "INTERACTFLOW_ROOT to its path."
    )


def resolve_from_root(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def parse_dotenv(path: Path) -> tuple[dict[str, str], list[str]]:
    """Parse the simple KEY=value format shipped in .env.example without eval."""
    values: dict[str, str] = {}
    issues: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return values, [f"Cannot read environment file {path}: {exc}"]

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            issues.append(f"Invalid .env assignment at line {line_number}")
            continue
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not ENVIRONMENT_NAME.fullmatch(name):
            issues.append(f"Invalid .env variable name at line {line_number}")
            continue
        value = raw_value.strip()
        if value.startswith("'"):
            if len(value) < 2 or not value.endswith("'"):
                issues.append(f"Unclosed single quote in .env at line {line_number}")
                continue
            value = value[1:-1]
        elif value.startswith('"'):
            if len(value) < 2 or not value.endswith('"'):
                issues.append(f"Unclosed double quote in .env at line {line_number}")
                continue
            try:
                decoded = ast.literal_eval(value)
                if not isinstance(decoded, str):
                    raise ValueError("not a string")
                value = decoded
            except (SyntaxError, ValueError):
                issues.append(f"Invalid escape in .env at line {line_number}")
                continue
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[name] = value
    return values, issues


def load_effective_environment(root: Path) -> tuple[Path, dict[str, str], list[str]]:
    configured = os.environ.get("VL_INTERACTION_ENV_FILE", "").strip()
    env_file = resolve_from_root(root, configured) if configured else root / ".env"
    effective = dict(os.environ)
    issues: list[str] = []
    if env_file.exists():
        if not env_file.is_file():
            issues.append(f"Environment path is not a file: {env_file}")
        else:
            mode = stat.S_IMODE(env_file.stat().st_mode)
            if mode & 0o077:
                issues.append(
                    f"Environment file permissions must be 600: {env_file}"
                )
            parsed, parse_issues = parse_dotenv(env_file)
            effective.update(parsed)
            issues.extend(parse_issues)
    effective["VL_INTERACTION_ENV_FILE"] = str(env_file)
    return env_file, effective, issues


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InteractFlowError(f"Cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InteractFlowError(f"{description} must be a JSON object: {path}")
    return value


def select_video_map(root: Path, explicit: str | None) -> Path | None:
    if explicit:
        return resolve_from_root(root, explicit)
    default = root / "data" / "media_index.jsonl"
    return default if default.is_file() else None


def build_context(root: Path, video_map: str | None) -> tuple[Context | None, list[str]]:
    env_file, effective, issues = load_effective_environment(root)
    raw_config = effective.get(
        "VL_INTERACTION_CAMPAIGN_CONFIG",
        "config/recording_campaign.json",
    ).strip()
    config_path = resolve_from_root(root, raw_config)
    if not config_path.is_file():
        issues.append(f"Missing campaign config: {config_path}")
        return None, issues
    try:
        config = load_json_object(config_path, "campaign config")
    except InteractFlowError as exc:
        issues.append(str(exc))
        return None, issues

    source = config.get("task_source") if isinstance(config.get("task_source"), dict) else {}
    workbook = resolve_from_root(
        root, source.get("xlsx") or "data/SVIBench-开源表.xlsx"
    )
    if not effective.get("BENCHMARK_WORKBOOK", "").strip():
        effective["BENCHMARK_WORKBOOK"] = str(workbook)
    video_dir = resolve_from_root(
        root, source.get("video_dir") or "data/interaction-75题"
    )
    task_manifest = resolve_from_root(
        root, source.get("manifest") or "data/recording_tasks_75.jsonl"
    )
    task_report = task_manifest.with_suffix(".report.json")
    audio_manifest_value = "data/minicpmo_query_audio/manifest.json"
    for model in config.get("models", []):
        if (
            isinstance(model, dict)
            and model.get("id") == "minicpmo-4.5-9b-native-video-v2"
            and model.get("enabled") is True
        ):
            audio_manifest_value = str(
                model.get("native_query_audio_manifest") or audio_manifest_value
            )
            break
    audio_dir = resolve_from_root(root, audio_manifest_value).parent
    return Context(
        root=root,
        env_file=env_file,
        env=effective,
        config_path=config_path,
        config=config,
        workbook=workbook,
        sheet=str(source.get("sheet") or "题目池"),
        video_dir=video_dir,
        task_manifest=task_manifest,
        task_report=task_report,
        audio_dir=audio_dir,
        video_map=select_video_map(root, video_map),
    ), issues


def required_environment(context: Context) -> set[str]:
    required = {"OPENAI_BASE_URL", "OPENAI_API_KEY"}
    config = context.config
    webui = config.get("webui") if isinstance(config.get("webui"), dict) else {}
    for field in ("url_env", "username_env", "password_env"):
        name = str(webui.get(field) or "").strip()
        if name:
            required.add(name)

    registry_value = str(config.get("vlm_registry") or "config/vlm_models.json")
    registry_path = resolve_from_root(context.root, registry_value)
    if not registry_path.is_file():
        return required | set(FALLBACK_ENVIRONMENT[3:-2])
    registry = load_json_object(registry_path, "VLM registry")
    profiles = {
        str(item.get("id") or ""): item
        for item in registry.get("profiles", [])
        if isinstance(item, dict)
    }
    for model in config.get("models", []):
        if not isinstance(model, dict) or model.get("enabled") is not True:
            continue
        profile = profiles.get(str(model.get("vlm_profile") or ""), {})
        for env_field, value_field in (
            ("api_base_env", "api_base"),
            ("realtime_api_base_env", "realtime_api_base"),
            ("api_key_env", "api_key"),
        ):
            name = str(profile.get(env_field) or "").strip()
            fallback = str(profile.get(value_field) or "").strip()
            if name and (env_field == "api_key_env" or not fallback):
                required.add(name)
    return required


def valid_url(value: str, schemes: set[str]) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in schemes and bool(parsed.netloc)


def executable_version(command: str) -> tuple[bool, str]:
    executable = shutil.which(command)
    if not executable:
        return False, ""
    flag = "-version" if command in {"ffmpeg", "ffprobe"} else "--version"
    try:
        completed = subprocess.run(
            [executable, flag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            check=False,
        )
        first_line = (completed.stdout or "").splitlines()
        return completed.returncode == 0, first_line[0] if first_line else "available"
    except (OSError, subprocess.SubprocessError):
        return False, ""


def chromium_available(root: Path, env: dict[str, str]) -> bool:
    node = shutil.which("node")
    if not node or not (root / "node_modules" / "playwright" / "package.json").is_file():
        return False
    script = (
        "import fs from 'node:fs'; import { chromium } from 'playwright'; "
        "process.exit(fs.existsSync(chromium.executablePath()) ? 0 : 2);"
    )
    try:
        completed = subprocess.run(
            [node, "--input-type=module", "-e", script],
            cwd=root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def workbook_contract(context: Context) -> tuple[dict[str, Any] | None, list[str]]:
    scripts_path = str(context.root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    try:
        from judge_workbook_contract import (  # type: ignore
            build_workbook_v2_contract,
            load_task_specs,
        )

        specs = load_task_specs(context.workbook, context.sheet)
        bundle = build_workbook_v2_contract(context.workbook, specs)
    except Exception as exc:
        return None, [f"Workbook contract validation failed: {exc}"]
    errors = [
        issue
        for issue in bundle.get("rubric_validation", {}).get("issues", [])
        if issue.get("severity") == "error"
    ]
    messages = [
        f"Workbook error {item.get('code', 'unknown')} at "
        f"{item.get('cell') or item.get('task_id') or item.get('sheet') or 'workbook'}"
        for item in errors
    ]
    return {
        "tasks": len(specs),
        "formal_ready": not errors,
        "warnings": sum(
            1
            for item in bundle.get("rubric_validation", {}).get("issues", [])
            if item.get("severity") == "warning"
        ),
    }, messages


def mapping_command(
    context: Context, output: Path, report: Path
) -> list[str]:
    command = [
        sys.executable,
        str(context.root / "scripts" / "build_tasks_from_xlsx.py"),
        "--xlsx",
        str(context.workbook),
        "--sheet",
        context.sheet,
        "--video-dir",
        str(context.video_dir),
        "--out",
        str(output),
        "--report",
        str(report),
    ]
    if context.video_map is not None:
        command.extend(("--video-map", str(context.video_map)))
    return command


def run_mapping_check(context: Context) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        with tempfile.TemporaryDirectory(prefix="interactflow-check-") as temporary:
            temporary_path = Path(temporary)
            tasks_path = temporary_path / "tasks.jsonl"
            report_path = temporary_path / "tasks.report.json"
            completed = subprocess.run(
                mapping_command(context, tasks_path, report_path),
                cwd=context.root,
                env=context.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode or not report_path.is_file():
                return None, [
                    "The workbook-to-video mapping check failed; run the manifest "
                    "builder directly for row-level diagnostics."
                ]
            report = load_json_object(report_path, "temporary task report")
            tasks = [
                json.loads(line)
                for line in tasks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    except Exception as exc:
        return None, [f"The workbook-to-video mapping check failed: {exc}"]

    errors: list[str] = []
    selected = report.get("selected_rows")
    written = report.get("tasks_written")
    missing = report.get("missing_count")
    task_ids = [str(item.get("id") or "") for item in tasks]
    video_paths = [str(item.get("local_video_path") or "") for item in tasks]
    if (selected, written, missing) != (75, 75, 0):
        errors.append(
            f"Formal mapping must be 75 selected / 75 written / 0 missing; "
            f"found {selected} / {written} / {missing}"
        )
    if len(set(task_ids)) != len(task_ids):
        errors.append("Generated task ids are not unique")
    if len(set(video_paths)) != len(video_paths):
        errors.append("The 75 formal tasks do not map to 75 unique videos")
    if report.get("duplicate_video_keys") and context.video_map is None:
        errors.append("Video directory contains ambiguous normalized filenames")
    return {
        "selected_rows": selected,
        "tasks_written": written,
        "missing_count": missing,
        "unique_videos": len(set(video_paths)),
    }, errors


def configuration_issues(context: Context) -> list[str]:
    config = context.config
    issues: list[str] = []
    if config.get("version") != 2:
        issues.append("Campaign config version must be 2")
    webui = config.get("webui") if isinstance(config.get("webui"), dict) else {}
    if webui.get("input_mode") != "upload":
        issues.append("Campaign WebUI input_mode must be upload")
    for field in ("url_env", "username_env", "password_env"):
        if not ENVIRONMENT_NAME.fullmatch(str(webui.get(field) or "")):
            issues.append(f"webui.{field} must name an environment variable")
    if not webui.get("identity_markers"):
        issues.append("webui.identity_markers must be non-empty")
    network = config.get("network") if isinstance(config.get("network"), dict) else {}
    if network.get("mode") != "direct":
        issues.append("Formal campaign network mode must be direct")
    scheduler = (
        config.get("frame_scheduler")
        if isinstance(config.get("frame_scheduler"), dict)
        else {}
    )
    expected_scheduler = {
        "max_in_flight": 1,
        "busy_policy": "skip",
        "queue_capacity": 0,
    }
    for key, expected in expected_scheduler.items():
        if scheduler.get(key) != expected:
            issues.append(f"frame_scheduler.{key} must be {expected!r}")
    enabled_models = [
        str(item.get("id") or "")
        for item in config.get("models", [])
        if isinstance(item, dict) and item.get("enabled") is True
    ]
    if tuple(enabled_models) != EXPECTED_MODELS:
        issues.append("Formal campaign must enable the expected five models in order")
    registry_path = resolve_from_root(
        context.root, str(config.get("vlm_registry") or "config/vlm_models.json")
    )
    if registry_path.is_file():
        try:
            registry = load_json_object(registry_path, "VLM registry")
            if registry.get("version") != 1:
                issues.append("VLM registry version must be 1")
            profiles = {
                str(item.get("id") or ""): item
                for item in registry.get("profiles", [])
                if isinstance(item, dict)
            }
            for model in config.get("models", []):
                if not isinstance(model, dict) or model.get("enabled") is not True:
                    continue
                profile_id = str(model.get("vlm_profile") or "")
                profile = profiles.get(profile_id)
                if profile is None:
                    issues.append(
                        f"Enabled model {model.get('id')} references missing profile {profile_id}"
                    )
                    continue
                if str(profile.get("api_key") or "").strip():
                    issues.append(
                        f"VLM profile {profile_id} must not contain a literal API key"
                    )
        except InteractFlowError as exc:
            issues.append(str(exc))
    if context.sheet != "题目池":
        issues.append("Formal task sheet must be 题目池")
    judge_workbook = context.env.get("BENCHMARK_WORKBOOK", "").strip()
    if judge_workbook and resolve_from_root(context.root, judge_workbook) != context.workbook:
        issues.append(
            "BENCHMARK_WORKBOOK and campaign task_source.xlsx must resolve to the same file"
        )
    for prompt in EXPECTED_PROMPTS:
        if not (context.root / "prompts" / prompt).is_file():
            issues.append(f"Missing formal Judge prompt: prompts/{prompt}")
    model = context.env.get("JUDGE_MODEL", "").strip() or "GPT-5.5"
    if model != "GPT-5.5":
        issues.append("JUDGE_MODEL must be GPT-5.5 for a formal run")
    return issues


def inspect_setup(root: Path, video_map: str | None = None) -> tuple[dict[str, Any], Context | None]:
    context, issues = build_context(root, video_map)
    missing_files: list[str] = []
    missing_environment: list[str] = []
    missing_executables: list[str] = []
    missing_python_packages: list[str] = []
    checks: dict[str, Any] = {}

    executable_names = ("bash", "node", "npm", "ffmpeg", "ffprobe", "cloudflared")
    executable_checks: dict[str, Any] = {}
    for command in executable_names:
        available, version = executable_version(command)
        executable_checks[command] = {"available": available, "version": version}
        if not available:
            missing_executables.append(command)
    checks["executables"] = executable_checks

    package_checks = {
        package: importlib.util.find_spec(module) is not None
        for module, package in PYTHON_DEPENDENCIES.items()
    }
    missing_python_packages.extend(
        package for package, available in package_checks.items() if not available
    )
    checks["python_packages"] = package_checks

    if sys.version_info < (3, 10):
        issues.append("Python 3.10 or newer is required")
    node_version = executable_checks.get("node", {}).get("version", "")
    match = re.search(r"v?(\d+)", node_version)
    if match and int(match.group(1)) < 18:
        issues.append("Node.js 18 or newer is required")

    if context is None:
        missing_environment = sorted(FALLBACK_ENVIRONMENT)
    else:
        issues.extend(configuration_issues(context))
        try:
            required = required_environment(context)
        except InteractFlowError as exc:
            issues.append(str(exc))
            required = set(FALLBACK_ENVIRONMENT)
        missing_environment = sorted(
            name for name in required if not context.env.get(name, "").strip()
        )
        url_schemes = {
            "JOYVL_WEB_URL": {"https"},
            "OPENAI_BASE_URL": {"http", "https"},
            "JOYAI_API_BASE": {"http", "https"},
            "DOUBAO_API_BASE": {"http", "https"},
            "MAGE_API_BASE": {"http", "https"},
            "MAGE_REALTIME_API_BASE": {"http", "https"},
            "MOSS_API_BASE": {"http", "https"},
            "MOSS_REALTIME_API_BASE": {"http", "https"},
            "MODELBEST_API_BASE": {"http", "https"},
            "MODELBEST_REALTIME_API_BASE": {"ws", "wss"},
        }
        for name, schemes in url_schemes.items():
            value = context.env.get(name, "").strip()
            if value and not valid_url(value, schemes):
                issues.append(
                    f"{name} must be an absolute {'/'.join(sorted(schemes))} URL"
                )

        for path in (context.workbook, context.video_dir):
            if not path.exists():
                missing_files.append(str(path))
        if context.video_map is not None and not context.video_map.is_file():
            missing_files.append(str(context.video_map))
        registry_path = resolve_from_root(
            root, str(context.config.get("vlm_registry") or "config/vlm_models.json")
        )
        if not registry_path.is_file():
            missing_files.append(str(registry_path))
        if not chromium_available(root, context.env):
            missing_files.append("Playwright Chromium (run: npx playwright install chromium)")

        if (
            context.workbook.is_file()
            and importlib.util.find_spec("openpyxl") is not None
        ):
            contract, contract_issues = workbook_contract(context)
            checks["workbook"] = contract
            issues.extend(contract_issues)
        if (
            context.workbook.is_file()
            and context.video_dir.is_dir()
            and not missing_python_packages
            and shutil.which("ffprobe")
        ):
            mapping, mapping_issues = run_mapping_check(context)
            checks["task_mapping"] = mapping
            issues.extend(mapping_issues)

    ready = not any(
        (issues, missing_files, missing_environment, missing_executables, missing_python_packages)
    )
    next_steps: list[str] = []
    if missing_executables:
        next_steps.append(
            "Install required executables and add them to PATH: "
            + ", ".join(sorted(set(missing_executables)))
        )
    if missing_python_packages:
        next_steps.append("Install Python packages: pip install -r requirements.txt")
    if "node" not in missing_executables and (
        context is not None
        and not (root / "node_modules" / "playwright" / "package.json").is_file()
    ):
        next_steps.append("Install Node packages: npm ci")
    if any("Playwright Chromium" in item for item in missing_files):
        next_steps.append("Install Chromium: npx playwright install chromium")
    if missing_environment:
        next_steps.append(
            "Copy .env.example to .env, fill the listed variables, and run chmod 600 .env"
        )
    data_paths = {
        str(context.workbook),
        str(context.video_dir),
        str(context.video_map) if context and context.video_map is not None else "",
    } if context is not None else set()
    if any(item in data_paths for item in missing_files):
        next_steps.append("Place the workbook and 75 source MP4 files as documented in DATA.md")
    if ready:
        next_steps.append(
            "Run `python3 skills/interactflow/scripts/interactflow.py start` to prepare and launch"
        )

    report = {
        "status": "ready" if ready else "needs_configuration",
        "repository": str(root),
        "environment_file": {
            "path": str(context.env_file if context else root / ".env"),
            "exists": bool(context and context.env_file.is_file()),
        },
        "checks": checks,
        "missing": {
            "environment_variables": missing_environment,
            "files": sorted(set(missing_files)),
            "executables": sorted(set(missing_executables)),
            "python_packages": sorted(set(missing_python_packages)),
        },
        "invalid": sorted(set(issues)),
        "next_steps": next_steps,
    }
    return report, context


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tasks(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_prepared_tasks(
    tasks_path: Path,
    report_path: Path,
    *,
    allow_duplicate_video_keys: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = load_json_object(report_path, "task report")
    tasks = load_tasks(tasks_path)
    if (
        report.get("selected_rows") != 75
        or report.get("tasks_written") != 75
        or report.get("missing_count") != 0
        or len(tasks) != 75
    ):
        raise InteractFlowError(
            "Task preparation did not produce 75 selected / 75 written / 0 missing"
        )
    ids = [str(task.get("id") or "") for task in tasks]
    videos = [str(task.get("local_video_path") or "") for task in tasks]
    if len(set(ids)) != 75 or not all(ids):
        raise InteractFlowError("Prepared task ids must be 75 unique non-empty values")
    if len(set(videos)) != 75 or not all(Path(value).is_file() for value in videos):
        raise InteractFlowError("Prepared tasks must reference 75 unique existing videos")
    if report.get("duplicate_video_keys") and not allow_duplicate_video_keys:
        raise InteractFlowError("Video directory contains ambiguous normalized filenames")
    return tasks, report


def audio_manifest_valid(audio_dir: Path, tasks_path: Path, tasks: list[dict[str, Any]]) -> bool:
    manifest_path = audio_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_json_object(manifest_path, "MiniCPM audio manifest")
        if manifest.get("source_tasks_sha256") != sha256(tasks_path):
            return False
        expected_hashes = {
            hashlib.sha256(str(round_item.get("query") or "").strip().encode("utf-8")).hexdigest()
            for task in tasks
            for round_item in (
                task.get("queries")
                or [{"query": task.get("query", "")}]
            )
            if str(round_item.get("query") or "").strip()
        }
        entries = manifest.get("entries")
        if not isinstance(entries, dict) or set(entries) != expected_hashes:
            return False
        for query_hash, item in entries.items():
            if not isinstance(item, dict) or item.get("query_sha256") != query_hash:
                return False
            pcm_path = audio_dir / str(item.get("pcm_path") or "")
            if not pcm_path.is_file() or pcm_path.stat().st_size <= 0:
                return False
            if sha256(pcm_path) != item.get("pcm_sha256"):
                return False
        return True
    except (InteractFlowError, OSError, ValueError, TypeError):
        return False


def run_checked(command: list[str], context: Context, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=context.root,
        env=context.env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = ""
        if capture:
            detail = ((completed.stderr or completed.stdout or "").strip())[-1200:]
        suffix = f": {detail}" if detail else ""
        raise InteractFlowError(
            f"Supported entrypoint failed with exit code {completed.returncode}{suffix}"
        )
    return completed


def prepare_inputs(context: Context) -> dict[str, Any]:
    context.task_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_tasks = context.task_manifest.with_name(
        f".{context.task_manifest.name}.tmp.{os.getpid()}"
    )
    temporary_report = context.task_report.with_name(
        f".{context.task_report.name}.tmp.{os.getpid()}"
    )
    try:
        run_checked(
            mapping_command(context, temporary_tasks, temporary_report),
            context,
            capture=True,
        )
        tasks, report = validate_prepared_tasks(
            temporary_tasks,
            temporary_report,
            allow_duplicate_video_keys=context.video_map is not None,
        )
        report["out"] = str(context.task_manifest)
        report["report"] = str(context.task_report)
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_tasks, context.task_manifest)
        os.replace(temporary_report, context.task_report)
    finally:
        temporary_tasks.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)

    tasks, _ = validate_prepared_tasks(
        context.task_manifest,
        context.task_report,
        allow_duplicate_video_keys=context.video_map is not None,
    )
    audio_reused = audio_manifest_valid(context.audio_dir, context.task_manifest, tasks)
    if not audio_reused:
        run_checked(
            [
                sys.executable,
                str(context.root / "scripts" / "prepare_minicpmo_query_audio.py"),
                "--tasks",
                str(context.task_manifest),
                "--out-dir",
                str(context.audio_dir),
            ],
            context,
        )
        if not audio_manifest_valid(context.audio_dir, context.task_manifest, tasks):
            raise InteractFlowError("MiniCPM query-audio manifest failed integrity validation")

    return {
        "status": "prepared",
        "task_manifest": str(context.task_manifest),
        "tasks": 75,
        "unique_videos": 75,
        "minicpmo_audio": "reused" if audio_reused else "generated",
    }


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def action_check(root: Path, video_map: str | None) -> int:
    report, _ = inspect_setup(root, video_map)
    emit(report)
    return 0 if report["status"] == "ready" else 2


def require_ready(root: Path, video_map: str | None) -> Context:
    report, context = inspect_setup(root, video_map)
    if report["status"] != "ready" or context is None:
        emit(report)
        raise InteractFlowError(
            "Preflight did not pass; no recording or Judge process was started"
        )
    return context


def action_prepare(root: Path, video_map: str | None) -> int:
    context = require_ready(root, video_map)
    emit(prepare_inputs(context))
    return 0


def action_start(root: Path, video_map: str | None) -> int:
    context = require_ready(root, video_map)
    emit(prepare_inputs(context))
    run_checked(["bash", "scripts/run_all.sh", "validate"], context)
    run_checked(["bash", "scripts/run_all.sh", "start"], context)
    run_checked(["bash", "scripts/run_all.sh", "status"], context)
    return 0


def action_delegate(root: Path, action: str, video_map: str | None) -> int:
    context, issues = build_context(root, video_map)
    if context is None:
        emit({"status": "invalid_configuration", "invalid": issues})
        return 2
    if issues:
        emit({"status": "invalid_configuration", "invalid": issues})
        return 2
    return run_checked(["bash", "scripts/run_all.sh", action], context).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check, prepare, start, inspect, or stop InteractFlow safely."
    )
    parser.add_argument(
        "action", choices=("check", "prepare", "start", "status", "stop")
    )
    parser.add_argument("--root", help="Repository clone; otherwise auto-detected")
    parser.add_argument(
        "--video-map",
        help="Optional JSON/JSONL task-id to video mapping (defaults to data/media_index.jsonl)",
    )
    args = parser.parse_args(argv)
    try:
        root = resolve_root(args.root)
        if args.action == "check":
            return action_check(root, args.video_map)
        if args.action == "prepare":
            return action_prepare(root, args.video_map)
        if args.action == "start":
            return action_start(root, args.video_map)
        return action_delegate(root, args.action, args.video_map)
    except InteractFlowError as exc:
        print(f"interactflow: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
