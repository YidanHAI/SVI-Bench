#!/usr/bin/env python3
"""Independently attest all evidence inputs of a 75-task Judge run.

The attestation is deliberately label-free.  It binds the capture artifacts,
rebuilds evidence through a pinned implementation, and verifies the task and
observation copied into every saved Judge-stage input.  It never calls a
model.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TASK_COUNT = 75
MAX_IMAGES = 24
MAX_TEMPORAL_SHEETS = 18
FRAME_WORKERS = 1
STAGES = (
    ("first_pass", "01_first_pass", "judge_input.json"),
    ("review_1", "02_review_1", "review_input.json"),
    ("review_2", "03_review_2", "review_input.json"),
    ("adjudication", "04_adjudication", "adjudication_input.json"),
    ("final_review", "05_final_review", "review_input.json"),
)
PREFLIGHT_NAME = "evidence_preflight.json"
PIPELINE_STATE_NAME = "pipeline_state.json"
PREDICTION_NAME = "prediction.json"
EXECUTION_ATTESTATION_NAME = "execution_attestation.json"
MOTION_EDGE_SUFFIX = ".motion_edges.jpg"
REQUEST_MAX_TOKEN_CANDIDATES = (12000, 20000)
STAGE_USER_PROMPT_PREFIXES = {
    "first_pass": (
        "请独立评测这条录屏。人工评分未包含在输入中。"
        "严格使用 system prompt 的 JSON 结构。\n\nJUDGE_INPUT_JSON:\n"
    ),
    "review_1": (
        "请复核这条首轮判定。人工评分未包含在输入中。"
        "首轮意见不可信，必须按 system prompt 独立核验证据。\n\n"
        "REVIEW_INPUT_JSON:\n"
    ),
    "review_2": (
        "请复核这条首轮判定。人工评分未包含在输入中。"
        "首轮意见不可信，必须按 system prompt 独立核验证据。\n\n"
        "REVIEW_INPUT_JSON:\n"
    ),
    "adjudication": (
        "请裁决这些候选判定。人工评分未包含在输入中。"
        "不得多数投票，必须回看共同的原始证据。\n\n"
        "ADJUDICATION_INPUT_JSON:\n"
    ),
    "final_review": (
        "请复核这条首轮判定。人工评分未包含在输入中。"
        "首轮意见不可信，必须按 system prompt 独立核验证据。\n\n"
        "REVIEW_INPUT_JSON:\n"
    ),
}
LEDGER_IDENTITY_FIELDS = (
    "schema_version",
    "stage",
    "max_attempts_per_task",
    "manifest_sha256",
    "task_ids_sha256",
    "task_count",
    "stage_identity_sha256",
)
TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_LABEL_KEYS = {
    "expertlabel",
    "expertscore",
    "humanlabel",
    "humanscore",
    "goldlabel",
    "goldscore",
    "referencelabel",
    "referencescore",
    "groundtruthlabel",
    "groundtruthscore",
    "人工标签",
    "人工打分",
    "人工评分",
    "专家标签",
    "专家打分",
    "专家评分",
    "金标准",
}


class AttestationError(RuntimeError):
    """A closed-fail validation error with a non-sensitive error code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AttestationError("json_duplicate_key")
        value[key] = item
    return value


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except AttestationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttestationError("json_unreadable") from exc
    if not isinstance(value, dict):
        raise AttestationError("json_not_object")
    return value


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AttestationError("non_canonical_json") from exc


def canonical_value(value: Any) -> Any:
    return json.loads(canonical_bytes(value).decode("utf-8"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _fingerprint(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _hash_regular_file(path: Path) -> tuple[dict[str, Any], tuple[int, ...]]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise AttestationError("source_not_regular_file")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except AttestationError:
        raise
    except (OSError, RuntimeError) as exc:
        raise AttestationError("source_file_unavailable") from exc
    if _fingerprint(before) != _fingerprint(after):
        raise AttestationError("source_changed_while_hashing")
    return (
        {
            "path": str(resolved),
            "bytes": after.st_size,
            "sha256": digest.hexdigest(),
        },
        _fingerprint(after),
    )


class FileBinder:
    """Hash files once and detect mutations through the whole attestation."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[dict[str, Any], tuple[int, ...]]] = {}

    def bind(self, path: Path | str) -> dict[str, Any]:
        try:
            resolved = Path(path).expanduser().resolve(strict=True)
            current = _fingerprint(resolved.stat())
        except (OSError, RuntimeError) as exc:
            raise AttestationError("source_file_unavailable") from exc
        key = str(resolved)
        cached = self._cache.get(key)
        if cached is not None:
            binding, original = cached
            if current != original:
                raise AttestationError("source_changed_during_attestation")
            return dict(binding)
        binding, fingerprint = _hash_regular_file(resolved)
        self._cache[key] = (binding, fingerprint)
        return dict(binding)

    def revalidate(self) -> None:
        for path, (_, expected) in self._cache.items():
            try:
                current = _fingerprint(Path(path).stat())
            except OSError as exc:
                raise AttestationError("source_disappeared_during_attestation") from exc
            if current != expected:
                raise AttestationError("source_changed_during_attestation")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def normalize_label_key(key: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", key.casefold())


def contains_forbidden_label_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = normalize_label_key(str(key))
            english_label_field = (
                any(
                    subject in normalized
                    for subject in ("expert", "human", "gold", "groundtruth", "reference")
                )
                and any(
                    measure in normalized
                    for measure in ("label", "score", "rating", "grade", "annotation")
                )
            )
            chinese_label_field = (
                any(subject in normalized for subject in ("人工", "专家", "金标准"))
                and any(measure in normalized for measure in ("标签", "打分", "评分", "得分"))
            )
            if normalized != "containshumanlabels" and (
                normalized in FORBIDDEN_LABEL_KEYS
                or english_label_field
                or chinese_label_field
            ):
                return True
            if contains_forbidden_label_key(item):
                return True
    elif isinstance(value, list):
        return any(contains_forbidden_label_key(item) for item in value)
    return False


def resolve_reference(
    raw_path: Any, anchors: Iterable[Path], error_code: str
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip() or "://" in raw_path:
        raise AttestationError(error_code)
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    choices = [anchor / candidate for anchor in anchors]
    for choice in choices:
        if choice.is_file():
            return choice
    return choices[0]


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_evidence_module(path: Path):
    module_name = "attested_full_evidence_" + hashlib.sha256(
        str(path).encode("utf-8")
    ).hexdigest()[:16]
    previous_path = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise AttestationError("evidence_module_import_failed")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except AttestationError:
        raise
    except Exception as exc:
        raise AttestationError("evidence_module_import_failed") from exc
    finally:
        sys.path[:] = previous_path
    if not callable(getattr(module, "build_evidence", None)):
        raise AttestationError("evidence_module_missing_build_evidence")
    return module


def add_error(errors: list[str], code: str) -> None:
    if code not in errors:
        errors.append(code)


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _motion_edge_path(path: Path) -> Path:
    return path.with_name(path.stem + MOTION_EDGE_SUFFIX)


def _artifact_matches(reference: Any, binding: dict[str, Any]) -> bool:
    return (
        isinstance(reference, dict)
        and reference.get("path") == binding.get("path")
        and reference.get("bytes") == binding.get("bytes")
        and reference.get("sha256") == binding.get("sha256")
    )


def _prompt_content_sha256(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise AttestationError("stage_prompt_unreadable") from exc
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def bind_pipeline_state(
    *,
    out_root: Path,
    manifest_sha256: str,
    evidence_sha256: str,
    binder: FileBinder,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    path = out_root / PIPELINE_STATE_NAME
    result: dict[str, Any] = {"path": str(path.resolve()), "pass": False}
    try:
        result["file"] = binder.bind(path)
        state = load_json_object(path)
    except AttestationError as exc:
        add_error(errors, f"pipeline_state_{exc.code}")
        return result, None

    header_pass = (
        state.get("contains_human_labels") is False
        and state.get("task_count") == EXPECTED_TASK_COUNT
        and state.get("manifest_sha256") == manifest_sha256
        and state.get("evidence_module_sha256") == evidence_sha256
        and state.get("pipeline") == "five_stage_judge"
        and isinstance(state.get("judge_model"), str)
        and bool(state.get("judge_model"))
        and state.get("judge_stage_count") == 5
    )
    stage_results: dict[str, Any] = {}
    stages = state.get("stages")
    for stage, directory_name, _ in STAGES:
        stage_result: dict[str, Any] = {"pass": False}
        entry = stages.get(stage) if isinstance(stages, dict) else None
        if not isinstance(entry, dict):
            stage_results[stage] = stage_result
            continue
        identity = entry.get("execution_identity")
        try:
            script = binder.bind(
                resolve_reference(
                    entry.get("script"), (ROOT, out_root), "stage_script_path_invalid"
                )
            )
            prompt_path = resolve_reference(
                entry.get("prompt"), (ROOT, out_root), "stage_prompt_path_invalid"
            )
            prompt = binder.bind(prompt_path)
            identity_pass = (
                isinstance(identity, dict)
                and identity.get("stage") == stage
                and identity.get("manifest_sha256") == manifest_sha256
                and identity.get("script_sha256") == script["sha256"]
                and identity.get("prompt_sha256") == prompt["sha256"]
                and identity.get("prompt_content_sha256")
                == _prompt_content_sha256(Path(prompt["path"]))
                and identity.get("max_http_attempts_per_task") == 5
            )
            directory_pass = Path(str(entry.get("directory") or "")).resolve() == (
                out_root / directory_name
            ).resolve()
            hashes_pass = (
                entry.get("script_sha256") == script["sha256"]
                and entry.get("prompt_sha256") == prompt["sha256"]
                and entry.get("prompt_content_sha256")
                == identity.get("prompt_content_sha256")
            )
            stage_result.update(
                {
                    "directory": entry.get("directory"),
                    "script": script,
                    "prompt": prompt,
                    "execution_identity_sha256": (
                        canonical_sha256(identity) if isinstance(identity, dict) else None
                    ),
                    "pass": directory_pass and hashes_pass and identity_pass,
                }
            )
        except AttestationError as exc:
            stage_result["error"] = exc.code
        stage_results[stage] = stage_result

    stages_pass = len(stage_results) == len(STAGES) and all(
        item.get("pass") is True for item in stage_results.values()
    )
    result.update(
        {
            "header_pass": header_pass,
            "stages": stage_results,
            "pass": header_pass and stages_pass,
        }
    )
    if not result["pass"]:
        add_error(errors, "pipeline_state_contract_mismatch")
    return result, state


def _load_bound_ledger(
    *,
    out_root: Path,
    stage: str,
    task_id: str,
    stage_identity: dict[str, Any],
    manifest_sha256: str,
    binder: FileBinder,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    path = out_root / "attempt_ledgers" / f"{stage}.json"
    seal_path = path.with_name(path.name + ".seal")
    binding = binder.bind(path)
    seal_binding = binder.bind(seal_path)
    ledger = load_json_object(path)
    seal = load_json_object(seal_path)

    unsigned_ledger = dict(ledger)
    unsigned_ledger.pop("integrity_sha256", None)
    identity = {key: ledger.get(key) for key in LEDGER_IDENTITY_FIELDS}
    identity_sha256 = canonical_sha256(identity)
    expected_stage_identity_sha256 = canonical_sha256(stage_identity)
    ledger_header_pass = (
        ledger.get("schema_version") == 2
        and ledger.get("stage") == stage
        and ledger.get("max_attempts_per_task") == 5
        and ledger.get("manifest_sha256") == manifest_sha256
        and ledger.get("task_count") == EXPECTED_TASK_COUNT
        and ledger.get("stage_identity_sha256") == expected_stage_identity_sha256
        and ledger.get("identity_sha256") == identity_sha256
        and ledger.get("integrity_sha256") == canonical_sha256(unsigned_ledger)
    )
    unsigned_seal = dict(seal)
    recorded_seal_sha256 = unsigned_seal.pop("seal_sha256", None)
    expected_seal = {
        "schema_version": 1,
        "identity_sha256": identity_sha256,
        "ledger_integrity_sha256": ledger.get("integrity_sha256"),
        "revision": ledger.get("revision"),
    }
    expected_seal["seal_sha256"] = canonical_sha256(expected_seal)
    seal_pass = (
        recorded_seal_sha256 == canonical_sha256(unsigned_seal)
        and seal == expected_seal
    )
    task_entry = (ledger.get("tasks") or {}).get(task_id)
    attempts = task_entry.get("attempts") if isinstance(task_entry, dict) else None
    attempts_pass = (
        isinstance(attempts, list)
        and 1 <= len(attempts) <= 5
        and task_entry.get("attempt_count") == len(attempts)
        and all(
            isinstance(item, dict)
            and item.get("attempt") == index
            and item.get("status") in {"claimed", "response", "exception"}
            and valid_sha256(item.get("request_sha256"))
            for index, item in enumerate(attempts, start=1)
        )
    )
    if not (ledger_header_pass and seal_pass and attempts_pass):
        raise AttestationError("motion_edge_attempt_ledger_invalid")
    return ledger, attempts, {
        "file": binding,
        "seal": seal_binding,
        "identity_sha256": identity_sha256,
        "attempts_sha256": canonical_sha256(attempts),
    }


def _request_payload_sha256(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    frames: list[dict[str, Any]],
    max_tokens: int,
    request_seed: int | None,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for index, frame in enumerate(frames):
        path = Path(frame["motion_edge"]["path"])
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise AttestationError("motion_edge_file_unreadable") from exc
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"EVIDENCE_IMAGE_{index:03d} "
                        f"t={float(frame['timestamp_s']):.3f}s "
                        f"view={frame['view']}_edge_motion"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + encoded,
                        "detail": "high",
                    },
                },
            ]
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "stream": False,
        "max_tokens": max_tokens,
    }
    if request_seed is not None:
        payload["seed"] = request_seed
    return canonical_sha256(payload)


def _validate_motion_edge_stage(
    *,
    stage: str,
    directory_name: str,
    input_filename: str,
    task_id: str,
    frames: list[dict[str, Any]],
    out_root: Path,
    manifest_sha256: str,
    pipeline_state: dict[str, Any],
    binder: FileBinder,
    request_seed: int | None,
) -> dict[str, Any]:
    task_dir = out_root / directory_name / task_id
    prediction_path = task_dir / PREDICTION_NAME
    prediction_binding = binder.bind(prediction_path)
    prediction = load_json_object(prediction_path)
    if prediction.get("evidence_variant") != "motion_edge_evidence":
        return {"declares_motion_edge": False, "pass": True}

    stage_entry = (pipeline_state.get("stages") or {}).get(stage)
    identity = stage_entry.get("execution_identity") if isinstance(stage_entry, dict) else None
    if not isinstance(identity, dict):
        raise AttestationError("motion_edge_stage_identity_missing")

    attestation_path = task_dir / EXECUTION_ATTESTATION_NAME
    attestation_binding = binder.bind(attestation_path)
    execution = load_json_object(attestation_path)
    unsigned_execution = dict(execution)
    recorded_integrity = unsigned_execution.pop("attestation_integrity_sha256", None)
    artifacts = execution.get("artifacts")
    artifact_pass = isinstance(artifacts, list) and any(
        _artifact_matches(item, prediction_binding) for item in artifacts
    )
    execution_pass = (
        recorded_integrity == canonical_sha256(unsigned_execution)
        and execution.get("schema_version") == 1
        and execution.get("pass") is True
        and execution.get("stage") == stage
        and execution.get("task_id") == task_id
        and execution.get("stage_identity") == identity
        and artifact_pass
    )
    if not execution_pass:
        raise AttestationError("motion_edge_execution_attestation_invalid")

    ledger, attempts, ledger_binding = _load_bound_ledger(
        out_root=out_root,
        stage=stage,
        task_id=task_id,
        stage_identity=identity,
        manifest_sha256=manifest_sha256,
        binder=binder,
    )
    ledger_reference = execution.get("attempt_ledger")
    final_attempt = attempts[-1]
    ledger_reference_pass = (
        isinstance(ledger_reference, dict)
        and ledger_reference.get("path") == ledger_binding["file"]["path"]
        and ledger_reference.get("identity_sha256")
        == ledger_binding["identity_sha256"]
        and ledger_reference.get("attempt_count") == len(attempts)
        and ledger_reference.get("attempts_sha256")
        == ledger_binding["attempts_sha256"]
        and final_attempt.get("status") == "response"
        and final_attempt.get("status_code") == 200
    )
    prediction_pass = (
        prediction.get("ok") is True
        and prediction.get("task_id") == task_id
        and (prediction.get("judge_json") or {}).get("task_id") == task_id
        and prediction.get("model") == identity.get("requested_model")
        and prediction.get("api_url") == identity.get("api_url")
        and prediction.get("native_video_input")
        == identity.get("native_video_input")
        and prediction.get("input_mode") == identity.get("input_mode")
        and prediction.get("evidence_image_count") == len(frames)
        and prediction.get("evidence_image_count_sent") == len(frames)
        and prediction.get("request_attempt") == len(attempts)
        and final_attempt.get("url") == prediction.get("api_url")
    )
    if not (ledger_reference_pass and prediction_pass):
        raise AttestationError("motion_edge_request_provenance_invalid")

    prompt_path = Path((stage_entry.get("prompt") or "")).resolve()
    prompt_binding = binder.bind(prompt_path)
    if (
        prediction.get("prompt_file") != prompt_binding["path"]
        or prediction.get("prompt_sha256") != _prompt_content_sha256(prompt_path)
    ):
        raise AttestationError("motion_edge_prompt_provenance_invalid")
    input_path = task_dir / input_filename
    persisted_input = load_json_object(input_path)
    user_prompt = STAGE_USER_PROMPT_PREFIXES[stage] + json.dumps(
        persisted_input, ensure_ascii=False, indent=2
    )
    try:
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise AttestationError("motion_edge_prompt_unreadable") from exc
    request_hashes = {
        max_tokens: _request_payload_sha256(
            model=prediction["model"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            frames=frames,
            max_tokens=max_tokens,
            request_seed=request_seed,
        )
        for max_tokens in REQUEST_MAX_TOKEN_CANDIDATES
    }
    matching_tokens = [
        max_tokens
        for max_tokens, request_hash in request_hashes.items()
        if request_hash == final_attempt.get("request_sha256")
    ]
    if len(matching_tokens) != 1:
        raise AttestationError("motion_edge_request_hash_mismatch")
    return {
        "declares_motion_edge": True,
        "pass": True,
        "stage": stage,
        "prediction": prediction_binding,
        "execution_attestation": attestation_binding,
        "attempt_ledger": ledger_binding,
        "request_attempt": len(attempts),
        "request_sha256": final_attempt["request_sha256"],
        "matched_max_tokens": matching_tokens[0],
        "evidence_image_count_sent": prediction["evidence_image_count_sent"],
    }


def attest_evidence_directory(
    *,
    task_id: str,
    task_dir: Path,
    saved_frames: list[dict[str, Any]],
    out_root: Path,
    manifest_sha256: str,
    pipeline_state: dict[str, Any] | None,
    binder: FileBinder,
    errors: list[str],
    request_seed: int | None,
) -> dict[str, Any]:
    evidence_dir = (task_dir / "evidence").resolve()
    original_paths = {
        frame.get("path") for frame in saved_frames if isinstance(frame.get("path"), str)
    }
    try:
        discovered = {
            str(path.resolve()) for path in evidence_dir.rglob("*") if path.is_file()
        }
    except OSError:
        add_error(errors, "preflight_evidence_directory_unreadable")
        return {"pass": False, "mode": "unreadable"}

    if discovered == original_paths:
        return {
            "pass": True,
            "mode": "preflight_only",
            "preflight_file_count": len(original_paths),
            "motion_edge_file_count": 0,
        }

    expected_edges = {
        str(_motion_edge_path(Path(path)).resolve()) for path in original_paths
    }
    extras = discovered - original_paths
    complete_one_to_one = (
        len(original_paths) == len(saved_frames)
        and len(expected_edges) == len(original_paths)
        and extras == expected_edges
        and original_paths.issubset(discovered)
    )
    if not complete_one_to_one or pipeline_state is None:
        add_error(errors, "preflight_evidence_directory_inventory_mismatch")
        return {
            "pass": False,
            "mode": "invalid_extra_files",
            "preflight_file_count": len(original_paths),
            "extra_file_count": len(extras),
            "expected_motion_edge_file_count": len(expected_edges),
        }

    edge_frames: list[dict[str, Any]] = []
    for frame in saved_frames:
        edge_path = _motion_edge_path(Path(frame["path"]))
        edge_binding = binder.bind(edge_path)
        edge_frames.append(
            {
                "index": frame["index"],
                "timestamp_s": frame["timestamp_s"],
                "view": frame["view"],
                "source_sha256": frame["sha256"],
                "motion_edge": edge_binding,
            }
        )

    provenance: list[dict[str, Any]] = []
    declared_count = 0
    for stage, directory_name, input_filename in STAGES:
        try:
            item = _validate_motion_edge_stage(
                stage=stage,
                directory_name=directory_name,
                input_filename=input_filename,
                task_id=task_id,
                frames=edge_frames,
                out_root=out_root,
                manifest_sha256=manifest_sha256,
                pipeline_state=pipeline_state,
                binder=binder,
                request_seed=request_seed,
            )
            if item.get("declares_motion_edge"):
                declared_count += 1
                provenance.append(item)
        except AttestationError as exc:
            add_error(errors, f"stage_{stage}_{exc.code}")
            provenance.append(
                {"stage": stage, "declares_motion_edge": True, "pass": False}
            )
    provenance_pass = declared_count > 0 and all(
        item.get("pass") is True for item in provenance
    )
    if not provenance_pass:
        add_error(errors, "motion_edge_request_provenance_missing_or_invalid")
    return {
        "pass": complete_one_to_one and provenance_pass,
        "mode": "motion_edge_evidence",
        "preflight_file_count": len(original_paths),
        "motion_edge_file_count": len(edge_frames),
        "complete_one_to_one": complete_one_to_one,
        "motion_edge_files": edge_frames,
        "successful_request_provenance": provenance,
    }


def stage_inventory(
    out_root: Path, expected_ids: list[str]
) -> tuple[dict[str, Any], bool]:
    result: dict[str, Any] = {}
    all_passed = True
    for stage, directory_name, filename in STAGES:
        directory = out_root / directory_name
        expected = {directory / task_id / filename for task_id in expected_ids}
        discovered = set(directory.rglob(filename)) if directory.is_dir() else set()
        missing = sorted(
            task_id
            for task_id in expected_ids
            if directory / task_id / filename not in discovered
        )
        extra_count = len(discovered - expected)
        passed = (
            len(discovered) == EXPECTED_TASK_COUNT
            and not missing
            and extra_count == 0
        )
        result[stage] = {
            "directory": str(directory.resolve()),
            "input_filename": filename,
            "input_count": len(discovered),
            "missing_task_ids": missing,
            "extra_input_count": extra_count,
            "pass": passed,
        }
        all_passed = all_passed and passed
    return result, all_passed


def preflight_inventory(out_root: Path, expected_ids: list[str]) -> dict[str, Any]:
    directory = out_root / STAGES[0][1]
    expected = {directory / task_id / PREFLIGHT_NAME for task_id in expected_ids}
    discovered = set(directory.rglob(PREFLIGHT_NAME)) if directory.is_dir() else set()
    missing = sorted(
        task_id
        for task_id in expected_ids
        if directory / task_id / PREFLIGHT_NAME not in discovered
    )
    extra_count = len(discovered - expected)
    passed = (
        len(discovered) == EXPECTED_TASK_COUNT
        and not missing
        and extra_count == 0
    )
    return {
        "directory": str(directory.resolve()),
        "preflight_count": len(discovered),
        "missing_task_ids": missing,
        "extra_preflight_count": extra_count,
        "pass": passed,
    }


def bind_manifest_video(
    task: dict[str, Any], manifest_path: Path, binder: FileBinder, errors: list[str]
) -> tuple[dict[str, Any], Path | None]:
    evidence_video = task.get("evidence_video")
    output: dict[str, Any] = {"pass": False}
    if not isinstance(evidence_video, dict):
        add_error(errors, "evidence_video_invalid")
        return output, None
    expected_bytes = evidence_video.get("bytes")
    expected_hash = evidence_video.get("sha256")
    output["manifest"] = {
        "path": evidence_video.get("path"),
        "bytes": expected_bytes,
        "sha256": expected_hash,
    }
    try:
        path = resolve_reference(
            evidence_video.get("path"),
            (ROOT, manifest_path.parent),
            "evidence_video_path_invalid",
        )
        current = binder.bind(path)
        matches = (
            isinstance(expected_bytes, int)
            and expected_bytes >= 0
            and valid_sha256(expected_hash)
            and current["bytes"] == expected_bytes
            and current["sha256"] == expected_hash
        )
        output.update({"current": current, "manifest_matches_current": matches, "pass": matches})
        if not matches:
            add_error(errors, "evidence_video_manifest_mismatch")
        return output, Path(current["path"])
    except AttestationError as exc:
        add_error(errors, exc.code)
        return output, None


def _add_source_candidate(
    candidates: list[tuple[str, Any, tuple[Path, ...], str | None]],
    role: str,
    raw_path: Any,
    anchors: tuple[Path, ...],
    expected_sha256: str | None = None,
) -> None:
    if raw_path is not None:
        candidates.append((role, raw_path, anchors, expected_sha256))


def bind_source_files(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    task: dict[str, Any],
    task_id: str,
    observation: dict[str, Any] | None,
    evidence_video_path: Path | None,
    binder: FileBinder,
    errors: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    candidates: list[tuple[str, Any, tuple[Path, ...], str | None]] = []
    root_anchors = (ROOT, manifest_path.parent)
    _add_source_candidate(
        candidates,
        "manifest.capture_run_summary",
        manifest.get("capture_run_summary"),
        root_anchors,
        manifest.get("capture_run_summary_sha256"),
    )
    _add_source_candidate(
        candidates,
        "manifest.recording_index",
        manifest.get("recording_index"),
        root_anchors,
        manifest.get("recording_index_sha256"),
    )
    timing_source = task.get("timing_source")
    summary_path: Path | None = None
    if not isinstance(timing_source, dict):
        add_error(errors, "timing_source_invalid")
    else:
        for key, value in sorted(timing_source.items()):
            if key.endswith(("_json", "_jsonl", "_path", "_file")):
                _add_source_candidate(
                    candidates, f"timing_source.{key}", value, root_anchors
                )
        raw_summary = timing_source.get("summary_json")
        if raw_summary is not None:
            try:
                summary_path = resolve_reference(
                    raw_summary, root_anchors, "timing_summary_path_invalid"
                ).resolve(strict=True)
            except (AttestationError, OSError):
                add_error(errors, "timing_summary_unavailable")

    capture_provenance = task.get("capture_provenance")
    if isinstance(capture_provenance, dict):
        for key, value in sorted(capture_provenance.items()):
            if key == "run_summary" or key.endswith(("_json", "_jsonl", "_path", "_file")):
                _add_source_candidate(
                    candidates, f"capture_provenance.{key}", value, root_anchors
                )
    else:
        add_error(errors, "capture_provenance_invalid")

    timing_summary: dict[str, Any] | None = None
    if summary_path is not None:
        try:
            timing_summary = load_json_object(summary_path)
            summary_task = timing_summary.get("task")
            if isinstance(summary_task, dict):
                if summary_task.get("id") not in (None, task_id):
                    add_error(errors, "timing_summary_task_mismatch")
                _add_source_candidate(
                    candidates,
                    "timing_summary.task.local_video_path",
                    summary_task.get("local_video_path"),
                    (ROOT, summary_path.parent, manifest_path.parent),
                )
            upload = timing_summary.get("local_video_upload")
            if isinstance(upload, dict):
                _add_source_candidate(
                    candidates,
                    "timing_summary.local_video_upload.source_video_path",
                    upload.get("source_video_path"),
                    (ROOT, summary_path.parent, manifest_path.parent),
                )
            files = timing_summary.get("files")
            if isinstance(files, dict):
                for key, value in sorted(files.items()):
                    _add_source_candidate(
                        candidates,
                        f"timing_summary.files.{key}",
                        value,
                        (ROOT, summary_path.parent, manifest_path.parent),
                    )
        except AttestationError as exc:
            add_error(errors, exc.code)

    by_path: dict[str, dict[str, Any]] = {}
    for role, raw_path, anchors, expected_hash in candidates:
        try:
            path = resolve_reference(raw_path, anchors, "source_path_invalid")
            binding = binder.bind(path)
            entry = by_path.setdefault(
                binding["path"], {**binding, "roles": [], "expected_sha256": []}
            )
            if role not in entry["roles"]:
                entry["roles"].append(role)
            if expected_hash is not None:
                expectation = {
                    "role": role,
                    "sha256": expected_hash,
                    "matches": valid_sha256(expected_hash)
                    and expected_hash == binding["sha256"],
                }
                entry["expected_sha256"].append(expectation)
                if not expectation["matches"]:
                    add_error(errors, "source_expected_hash_mismatch")
        except AttestationError as exc:
            add_error(errors, exc.code)

    entries = sorted(by_path.values(), key=lambda item: item["path"])
    for entry in entries:
        entry["roles"].sort()
        entry["expected_sha256"].sort(key=lambda item: item["role"])

    if observation is not None and evidence_video_path is not None:
        if observation.get("recording_file") != evidence_video_path.name:
            add_error(errors, "observation_recording_file_mismatch")
        if observation.get("source_video_available") is True:
            source_name = observation.get("source_video_file")
            source_entries = [
                entry
                for entry in entries
                if any("source_video" in role or "local_video_path" in role for role in entry["roles"])
            ]
            if not isinstance(source_name, str) or not any(
                Path(entry["path"]).name == source_name for entry in source_entries
            ):
                add_error(errors, "observation_source_video_unbound")

    if timing_summary is not None and evidence_video_path is not None:
        files = timing_summary.get("files")
        task_mp4 = files.get("task_mp4") if isinstance(files, dict) else None
        if task_mp4 is not None:
            try:
                resolved_task_mp4 = resolve_reference(
                    task_mp4,
                    (ROOT, summary_path.parent if summary_path else ROOT),
                    "capture_task_video_path_invalid",
                ).resolve(strict=True)
                resolved_evidence_video = evidence_video_path.resolve(strict=True)
                same_file = os.path.samefile(
                    resolved_task_mp4, resolved_evidence_video
                )
                if not same_file:
                    task_binding = binder.bind(resolved_task_mp4)
                    evidence_binding = binder.bind(resolved_evidence_video)
                    same_file = (
                        task_binding["bytes"] == evidence_binding["bytes"]
                        and task_binding["sha256"] == evidence_binding["sha256"]
                    )
                if not same_file:
                    add_error(errors, "capture_task_video_mismatch")
            except (AttestationError, OSError):
                add_error(errors, "capture_task_video_unavailable")
    return entries, timing_summary


def load_preflight(
    *,
    task_id: str,
    task_dir: Path,
    manifest_sha256: str,
    evidence_sha256: str,
    binder: FileBinder,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    path = task_dir / PREFLIGHT_NAME
    output: dict[str, Any] = {"path": str(path.resolve()), "pass": False}
    try:
        output["file"] = binder.bind(path)
        saved = load_json_object(path)
    except AttestationError as exc:
        add_error(errors, exc.code)
        return output, None, []

    observation = saved.get("observation")
    if not isinstance(observation, dict):
        add_error(errors, "preflight_observation_invalid")
        observation = None
    evidence = saved.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        add_error(errors, "preflight_evidence_invalid")
        evidence = []
    header_pass = (
        saved.get("ok") is True
        and saved.get("task_id") == task_id
        and saved.get("manifest_sha256") == manifest_sha256
        and saved.get("judge_script_sha256") == evidence_sha256
        and saved.get("evidence_count") == len(evidence)
    )
    if not header_pass:
        add_error(errors, "preflight_header_mismatch")

    frames: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    evidence_dir = (task_dir / "evidence").resolve()
    for expected_index, item in enumerate(evidence):
        frame: dict[str, Any] = {"index": expected_index, "pass": False}
        if not isinstance(item, dict):
            add_error(errors, "preflight_frame_invalid")
            frames.append(frame)
            continue
        frame.update({
            "timestamp_s": item.get("timestamp_s"),
            "view": item.get("view"),
            "saved_path": item.get("path"),
            "saved_bytes": item.get("bytes"),
            "saved_sha256": item.get("sha256"),
        })
        try:
            declared = Path(item.get("path"))
            current = binder.bind(declared)
            current_path = Path(current["path"])
            relative_path = current_path.relative_to(task_dir.resolve())
            current_path.relative_to(evidence_dir)
            metadata_pass = (
                item.get("index") == expected_index
                and isinstance(item.get("timestamp_s"), (int, float))
                and not isinstance(item.get("timestamp_s"), bool)
                and math.isfinite(float(item["timestamp_s"]))
                and isinstance(item.get("view"), str)
                and bool(item["view"])
                and isinstance(item.get("bytes"), int)
                and item["bytes"] == current["bytes"]
                and valid_sha256(item.get("sha256"))
                and item["sha256"] == current["sha256"]
                and current["path"] not in seen_paths
            )
            seen_paths.add(current["path"])
            frame.update({
                "path": current["path"],
                "relative_path": relative_path.as_posix(),
                "bytes": current["bytes"],
                "sha256": current["sha256"],
                "pass": metadata_pass,
            })
            if not metadata_pass:
                add_error(errors, "preflight_frame_metadata_mismatch")
        except (AttestationError, OSError, TypeError, ValueError) as exc:
            add_error(
                errors,
                exc.code if isinstance(exc, AttestationError) else "preflight_frame_path_invalid",
            )
        frames.append(frame)

    metadata_pass = header_pass and observation is not None and all(
        frame.get("pass") is True for frame in frames
    ) and len(frames) > 0
    output.update({
        "header_pass": header_pass,
        "metadata_pass": metadata_pass,
        "observation_sha256": (
            canonical_sha256(observation) if observation is not None else None
        ),
        "evidence_count": len(evidence),
        "frames": frames,
    })
    output["pass"] = metadata_pass
    return output, observation, frames


def rebuild_evidence(
    *,
    module: Any,
    task: dict[str, Any],
    task_id: str,
    temporary_root: Path,
    errors: list[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    task_dir = temporary_root / task_id
    task_dir.mkdir(parents=True, exist_ok=False)
    try:
        task_copy = canonical_value(task)
        with working_directory(ROOT):
            observation, raw_frames = module.build_evidence(
                task_copy,
                task_dir,
                MAX_IMAGES,
                FRAME_WORKERS,
                MAX_TEMPORAL_SHEETS,
            )
        if canonical_bytes(task_copy) != canonical_bytes(task):
            raise AttestationError("evidence_module_mutated_task")
        if not isinstance(observation, dict) or not isinstance(raw_frames, list):
            raise AttestationError("evidence_rebuild_invalid")
        canonical_observation = canonical_value(observation)
        frames: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw_frame in enumerate(raw_frames):
            if not isinstance(raw_frame, (list, tuple)) or len(raw_frame) != 3:
                raise AttestationError("rebuilt_frame_invalid")
            timestamp, raw_path, view = raw_frame
            if (
                not isinstance(timestamp, (int, float))
                or isinstance(timestamp, bool)
                or not math.isfinite(float(timestamp))
                or not isinstance(view, str)
                or not view
            ):
                raise AttestationError("rebuilt_frame_metadata_invalid")
            binding, _ = _hash_regular_file(Path(raw_path))
            resolved = Path(binding["path"])
            try:
                relative = resolved.relative_to(task_dir.resolve())
            except ValueError as exc:
                raise AttestationError("rebuilt_frame_outside_temporary_root") from exc
            if binding["path"] in seen:
                raise AttestationError("rebuilt_frame_duplicate")
            seen.add(binding["path"])
            frames.append({
                "index": index,
                "timestamp_s": timestamp,
                "view": view,
                "relative_path": relative.as_posix(),
                "bytes": binding["bytes"],
                "sha256": binding["sha256"],
            })
        if not frames:
            raise AttestationError("rebuilt_evidence_empty")
        return canonical_observation, frames
    except AttestationError as exc:
        add_error(errors, exc.code)
        return None, []
    except Exception:
        add_error(errors, "evidence_rebuild_failed")
        return None, []
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


def compare_rebuilt_frames(
    saved_frames: list[dict[str, Any]], rebuilt_frames: list[dict[str, Any]]
) -> bool:
    if len(saved_frames) != len(rebuilt_frames) or not saved_frames:
        return False
    keys = ("index", "timestamp_s", "view", "relative_path", "bytes", "sha256")
    return all(
        all(canonical_bytes(saved.get(key)) == canonical_bytes(rebuilt.get(key)) for key in keys)
        for saved, rebuilt in zip(saved_frames, rebuilt_frames)
    )


def bind_stage_inputs(
    *,
    task_id: str,
    task_spec: dict[str, Any],
    observation: dict[str, Any] | None,
    out_root: Path,
    binder: FileBinder,
    errors: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    expected_task = canonical_bytes(task_spec)
    expected_observation = canonical_bytes(observation) if observation is not None else None
    for stage, directory_name, filename in STAGES:
        path = out_root / directory_name / task_id / filename
        stage_result: dict[str, Any] = {"path": str(path.resolve()), "pass": False}
        try:
            binding = binder.bind(path)
            saved = load_json_object(path)
            saved_task = saved.get("task_spec")
            saved_observation = saved.get("observation")
            task_matches = canonical_bytes(saved_task) == expected_task
            observation_matches = (
                expected_observation is not None
                and canonical_bytes(saved_observation) == expected_observation
            )
            stage_result.update({
                "file": binding,
                "task_spec_sha256": canonical_sha256(saved_task),
                "observation_sha256": canonical_sha256(saved_observation),
                "task_spec_matches_manifest": task_matches,
                "observation_matches_rebuilt": observation_matches,
                "pass": task_matches and observation_matches,
            })
            if not task_matches:
                add_error(errors, f"stage_{stage}_task_spec_mismatch")
            if not observation_matches:
                add_error(errors, f"stage_{stage}_observation_mismatch")
        except AttestationError as exc:
            add_error(errors, f"stage_{stage}_{exc.code}")
        result[stage] = stage_result
    return result


def validate_manifest(
    manifest: dict[str, Any], global_errors: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    if manifest.get("contains_human_labels") is not False:
        add_error(global_errors, "manifest_not_explicitly_label_free")
    if contains_forbidden_label_key(manifest):
        add_error(global_errors, "manifest_contains_forbidden_label_fields")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        add_error(global_errors, "manifest_tasks_invalid")
        return [], []
    if len(tasks) != EXPECTED_TASK_COUNT:
        add_error(global_errors, "manifest_task_count_not_75")
    if manifest.get("task_count") not in (None, EXPECTED_TASK_COUNT):
        add_error(global_errors, "manifest_declared_task_count_mismatch")
    if manifest.get("expected_task_count") not in (None, EXPECTED_TASK_COUNT):
        add_error(global_errors, "manifest_expected_task_count_mismatch")
    if manifest.get("missing_task_ids") not in (None, []):
        add_error(global_errors, "manifest_declares_missing_tasks")
    has_capture_summary = (
        isinstance(manifest.get("capture_run_summary"), str)
        and valid_sha256(manifest.get("capture_run_summary_sha256"))
    )
    has_recording_index = (
        isinstance(manifest.get("recording_index"), str)
        and valid_sha256(manifest.get("recording_index_sha256"))
    )
    if not (has_capture_summary or has_recording_index):
        add_error(global_errors, "manifest_capture_source_binding_missing")
    task_ids: list[str] = []
    valid_tasks: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("task_spec"), dict):
            add_error(global_errors, "manifest_task_invalid")
            continue
        task_id = task["task_spec"].get("id")
        if not isinstance(task_id, str) or TASK_ID_RE.fullmatch(task_id) is None:
            add_error(global_errors, "manifest_task_id_invalid")
            continue
        task_ids.append(task_id)
        valid_tasks.append(task)
    if len(task_ids) != len(set(task_ids)):
        add_error(global_errors, "manifest_duplicate_task_ids")
    if len(valid_tasks) != EXPECTED_TASK_COUNT:
        add_error(global_errors, "manifest_valid_task_count_not_75")
    return valid_tasks, task_ids


def attest(
    manifest_path: Path,
    out_root: Path,
    evidence_module_path: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    evidence_module_path = evidence_module_path.expanduser().resolve()
    binder = FileBinder()
    global_errors: list[str] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": now(),
        "pass": False,
        "contains_human_labels": False,
        "contract": {
            "expected_tasks": EXPECTED_TASK_COUNT,
            "expected_stages": [stage for stage, _, _ in STAGES],
            "required_stage_inputs": EXPECTED_TASK_COUNT * len(STAGES),
            "max_images": MAX_IMAGES,
            "max_temporal_sheets": MAX_TEMPORAL_SHEETS,
            "calls_model": False,
            "comparison": "canonical JSON plus current file size/SHA-256",
        },
        "inputs": {
            "manifest": {"path": str(manifest_path)},
            "out_root": str(out_root),
            "evidence_module": {"path": str(evidence_module_path)},
        },
        "errors": global_errors,
        "tasks": [],
    }
    try:
        manifest_binding = binder.bind(manifest_path)
        evidence_binding = binder.bind(evidence_module_path)
        report["inputs"]["manifest"] = manifest_binding
        report["inputs"]["evidence_module"] = evidence_binding
        manifest = load_json_object(manifest_path)
        tasks, task_ids = validate_manifest(manifest, global_errors)
    except AttestationError as exc:
        add_error(global_errors, exc.code)
        report["summary"] = {
            "tasks_expected": EXPECTED_TASK_COUNT,
            "tasks_attested": 0,
            "tasks_passed": 0,
            "stage_inputs_expected": EXPECTED_TASK_COUNT * len(STAGES),
            "stage_inputs_attested": 0,
        }
        return report

    if any(
        code in global_errors
        for code in (
            "manifest_not_explicitly_label_free",
            "manifest_contains_forbidden_label_fields",
        )
    ):
        report["summary"] = {
            "tasks_expected": EXPECTED_TASK_COUNT,
            "tasks_attested": 0,
            "tasks_passed": 0,
            "stage_inputs_expected": EXPECTED_TASK_COUNT * len(STAGES),
            "stage_inputs_attested": 0,
        }
        return report

    inventories, inventories_pass = stage_inventory(out_root, task_ids)
    preflights = preflight_inventory(out_root, task_ids)
    report["stage_inventory"] = inventories
    report["preflight_inventory"] = preflights
    if not inventories_pass:
        add_error(global_errors, "stage_inventory_mismatch")
    if not preflights["pass"]:
        add_error(global_errors, "preflight_inventory_mismatch")

    try:
        module = load_evidence_module(evidence_module_path)
    except AttestationError as exc:
        add_error(global_errors, exc.code)
        module = None
    request_seed = getattr(module, "REQUEST_SEED", None) if module is not None else None
    if request_seed is not None and not isinstance(request_seed, int):
        add_error(global_errors, "evidence_module_request_seed_invalid")
        request_seed = None

    manifest_hash = report["inputs"]["manifest"]["sha256"]
    evidence_hash = report["inputs"]["evidence_module"]["sha256"]
    has_motion_edge_files = any(
        (out_root / STAGES[0][1]).glob(f"*/evidence/*{MOTION_EDGE_SUFFIX}")
    )
    pipeline_state: dict[str, Any] | None = None
    if has_motion_edge_files:
        pipeline_state_report, pipeline_state = bind_pipeline_state(
            out_root=out_root,
            manifest_sha256=manifest_hash,
            evidence_sha256=evidence_hash,
            binder=binder,
            errors=global_errors,
        )
        report["pipeline_state"] = pipeline_state_report
    tasks_by_id = {
        task["task_spec"]["id"]: task
        for task in tasks
        if isinstance(task.get("task_spec", {}).get("id"), str)
    }
    with tempfile.TemporaryDirectory(prefix="full-evidence-attestation-") as temporary:
        temporary_root = Path(temporary)
        for task_id in task_ids:
            task = tasks_by_id.get(task_id)
            if task is None:
                continue
            errors: list[str] = []
            task_spec = canonical_value(task["task_spec"])
            task_dir = out_root / STAGES[0][1] / task_id
            video, video_path = bind_manifest_video(
                task, manifest_path, binder, errors
            )
            preflight, saved_observation, saved_frames = load_preflight(
                task_id=task_id,
                task_dir=task_dir,
                manifest_sha256=manifest_hash,
                evidence_sha256=evidence_hash,
                binder=binder,
                errors=errors,
            )
            evidence_directory = attest_evidence_directory(
                task_id=task_id,
                task_dir=task_dir,
                saved_frames=saved_frames,
                out_root=out_root,
                manifest_sha256=manifest_hash,
                pipeline_state=pipeline_state,
                binder=binder,
                errors=errors,
                request_seed=request_seed,
            )
            preflight["evidence_directory"] = evidence_directory
            preflight["pass"] = (
                preflight.get("metadata_pass") is True
                and evidence_directory.get("pass") is True
            )
            rebuilt_observation: dict[str, Any] | None = None
            rebuilt_frames: list[dict[str, Any]] = []
            if module is not None:
                rebuilt_observation, rebuilt_frames = rebuild_evidence(
                    module=module,
                    task=task,
                    task_id=task_id,
                    temporary_root=temporary_root,
                    errors=errors,
                )
            else:
                add_error(errors, "evidence_module_unavailable")

            observation_matches_preflight = (
                rebuilt_observation is not None
                and saved_observation is not None
                and canonical_bytes(rebuilt_observation)
                == canonical_bytes(saved_observation)
            )
            if not observation_matches_preflight:
                add_error(errors, "rebuilt_observation_preflight_mismatch")
            frames_match_rebuild = compare_rebuilt_frames(saved_frames, rebuilt_frames)
            if not frames_match_rebuild:
                add_error(errors, "rebuilt_frames_preflight_mismatch")

            source_files, _ = bind_source_files(
                manifest=manifest,
                manifest_path=manifest_path,
                task=task,
                task_id=task_id,
                observation=rebuilt_observation,
                evidence_video_path=video_path,
                binder=binder,
                errors=errors,
            )
            stages = bind_stage_inputs(
                task_id=task_id,
                task_spec=task_spec,
                observation=rebuilt_observation,
                out_root=out_root,
                binder=binder,
                errors=errors,
            )
            task_pass = (
                not errors
                and video.get("pass") is True
                and preflight.get("pass") is True
                and observation_matches_preflight
                and frames_match_rebuild
                and len(stages) == len(STAGES)
                and all(item.get("pass") is True for item in stages.values())
            )
            report["tasks"].append({
                "task_id": task_id,
                "pass": task_pass,
                "errors": errors,
                "manifest_task_spec": {
                    "value": task_spec,
                    "sha256": canonical_sha256(task_spec),
                },
                "evidence_video": video,
                "evidence_preflight": preflight,
                "rebuilt_evidence": {
                    "observation": rebuilt_observation,
                    "observation_sha256": (
                        canonical_sha256(rebuilt_observation)
                        if rebuilt_observation is not None else None
                    ),
                    "observation_matches_preflight": observation_matches_preflight,
                    "frame_count": len(rebuilt_frames),
                    "frames": rebuilt_frames,
                    "frames_match_preflight": frames_match_rebuild,
                },
                "stage_inputs": stages,
                "timing_and_capture_sources": source_files,
            })

    try:
        generator = binder.bind(Path(__file__))
        binder.revalidate()
        report["generator"] = generator
    except AttestationError as exc:
        add_error(global_errors, exc.code)

    stage_inputs_attested = sum(
        len(task["stage_inputs"]) for task in report["tasks"]
    )
    report["summary"] = {
        "tasks_expected": EXPECTED_TASK_COUNT,
        "tasks_attested": len(report["tasks"]),
        "tasks_passed": sum(task["pass"] for task in report["tasks"]),
        "preflight_frames_attested": sum(
            len(task["evidence_preflight"].get("frames", []))
            for task in report["tasks"]
        ),
        "motion_edge_frames_attested": sum(
            task["evidence_preflight"]
            .get("evidence_directory", {})
            .get("motion_edge_file_count", 0)
            for task in report["tasks"]
        ),
        "rebuilt_frames_attested": sum(
            task["rebuilt_evidence"]["frame_count"] for task in report["tasks"]
        ),
        "stage_inputs_expected": EXPECTED_TASK_COUNT * len(STAGES),
        "stage_inputs_attested": stage_inputs_attested,
        "timing_and_capture_source_references": sum(
            len(task["timing_and_capture_sources"]) for task in report["tasks"]
        ),
    }
    report["checks"] = {
        "manifest_explicitly_label_free": manifest.get("contains_human_labels") is False,
        "exactly_75_manifest_tasks": len(task_ids) == EXPECTED_TASK_COUNT,
        "exactly_75_preflights": preflights["pass"],
        "exactly_75_inputs_per_stage": inventories_pass,
        "all_tasks_match": (
            len(report["tasks"]) == EXPECTED_TASK_COUNT
            and all(task["pass"] for task in report["tasks"])
        ),
        "all_375_stage_inputs_match": (
            stage_inputs_attested == EXPECTED_TASK_COUNT * len(STAGES)
            and all(
                stage.get("pass") is True
                for task in report["tasks"]
                for stage in task["stage_inputs"].values()
            )
        ),
    }
    report["pass"] = (
        not global_errors
        and all(report["checks"].values())
        and report["summary"]["tasks_passed"] == EXPECTED_TASK_COUNT
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attest a complete label-free 75-task, five-stage Judge run"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--evidence-module", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        help="Defaults to <out-root>/full_evidence_attestation.json",
    )
    args = parser.parse_args(argv)
    output = args.out or (args.out_root / "full_evidence_attestation.json")
    try:
        report = attest(args.manifest, args.out_root, args.evidence_module)
    except Exception:
        report = {
            "schema_version": 1,
            "generated_at": now(),
            "pass": False,
            "contains_human_labels": False,
            "errors": ["unhandled_attestation_failure"],
            "inputs": {
                "manifest": {"path": str(args.manifest.expanduser().resolve())},
                "out_root": str(args.out_root.expanduser().resolve()),
                "evidence_module": {
                    "path": str(args.evidence_module.expanduser().resolve())
                },
            },
        }
    atomic_write_json(output, report)
    print(json.dumps({
        "out": str(output.expanduser().resolve()),
        "pass": report.get("pass") is True,
        "tasks_attested": (report.get("summary") or {}).get("tasks_attested", 0),
        "tasks_passed": (report.get("summary") or {}).get("tasks_passed", 0),
    }, ensure_ascii=False))
    return 0 if report.get("pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
