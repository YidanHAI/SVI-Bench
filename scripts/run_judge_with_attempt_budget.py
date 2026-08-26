#!/usr/bin/env python3
"""Run a Judge stage with a persistent per-task HTTP attempt budget."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SCHEMA_VERSION = 2
INTEGRITY_FIELD = "integrity_sha256"
IDENTITY_FIELDS = (
    "schema_version",
    "stage",
    "max_attempts_per_task",
    "manifest_sha256",
    "task_ids_sha256",
    "task_count",
    "stage_identity_sha256",
)
SEAL_SCHEMA_VERSION = 1
INPUT_MARKERS = (
    "JUDGE_INPUT_JSON:",
    "REVIEW_INPUT_JSON:",
    "ADJUDICATION_INPUT_JSON:",
)
SECRET_REDACTION = "[REDACTED_OPENAI_API_KEY]"


class AttemptBudgetExceeded(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ledger_identity(value):
    if not isinstance(value, dict):
        raise RuntimeError("Attempt ledger must be a JSON object")
    return {key: value.get(key) for key in IDENTITY_FIELDS}


def ledger_identity_sha256(value):
    return json_sha256(ledger_identity(value))


def ledger_integrity_sha256(value):
    if not isinstance(value, dict):
        raise RuntimeError("Attempt ledger must be a JSON object")
    unsigned = dict(value)
    unsigned.pop(INTEGRITY_FIELD, None)
    return json_sha256(unsigned)


def validate_ledger_document(value, expected_identity=None):
    """Validate the complete ledger before any request or reuse decision."""
    if not isinstance(value, dict):
        raise RuntimeError("Attempt ledger must be a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported attempt ledger schema: {value.get('schema_version')!r}"
        )
    recorded_integrity = value.get(INTEGRITY_FIELD)
    computed_integrity = ledger_integrity_sha256(value)
    if recorded_integrity != computed_integrity:
        raise RuntimeError("Attempt ledger integrity check failed")
    if value.get("identity_sha256") != ledger_identity_sha256(value):
        raise RuntimeError("Attempt ledger identity hash check failed")
    if expected_identity is not None:
        actual = ledger_identity(value)
        mismatches = {
            key: {"expected": wanted, "actual": actual.get(key)}
            for key, wanted in expected_identity.items()
            if actual.get(key) != wanted
        }
        if mismatches:
            raise RuntimeError(
                "Attempt ledger identity mismatch: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )

    max_attempts = value.get("max_attempts_per_task")
    tasks = value.get("tasks")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise RuntimeError("Attempt ledger has an invalid request budget")
    if not isinstance(tasks, dict):
        raise RuntimeError("Attempt ledger tasks must be an object")
    allowed_statuses = {"claimed", "response", "exception"}
    for task_id, task in tasks.items():
        if not isinstance(task_id, str) or not isinstance(task, dict):
            raise RuntimeError("Attempt ledger contains an invalid task entry")
        attempts = task.get("attempts")
        if not isinstance(attempts, list):
            raise RuntimeError(f"Attempt ledger has invalid attempts for {task_id}")
        if task.get("attempt_count") != len(attempts):
            raise RuntimeError(f"Attempt ledger count mismatch for {task_id}")
        if len(attempts) > max_attempts:
            raise RuntimeError(f"Attempt ledger exceeds its budget for {task_id}")
        for index, event in enumerate(attempts, start=1):
            if not isinstance(event, dict) or event.get("attempt") != index:
                raise RuntimeError(f"Attempt ledger sequence is corrupt for {task_id}")
            if event.get("status") not in allowed_statuses:
                raise RuntimeError(f"Attempt ledger status is invalid for {task_id}")
            if not event.get("request_sha256") or not event.get("claimed_at"):
                raise RuntimeError(f"Attempt ledger event is incomplete for {task_id}")
            if event.get("status") != "claimed" and not event.get("finished_at"):
                raise RuntimeError(f"Attempt ledger event has no finish time for {task_id}")
    return value


def ledger_seal_path(path):
    path = Path(path)
    return path.with_name(path.name + ".seal")


def build_ledger_seal(value):
    seal = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "identity_sha256": ledger_identity_sha256(value),
        "ledger_integrity_sha256": value.get(INTEGRITY_FIELD),
        "revision": value.get("revision"),
    }
    seal["seal_sha256"] = json_sha256(seal)
    return seal


def validate_ledger_seal(value, seal):
    if not isinstance(seal, dict) or seal.get("schema_version") != SEAL_SCHEMA_VERSION:
        raise RuntimeError("Attempt ledger seal is missing or invalid")
    recorded = seal.get("seal_sha256")
    unsigned = dict(seal)
    unsigned.pop("seal_sha256", None)
    if recorded != json_sha256(unsigned):
        raise RuntimeError("Attempt ledger seal integrity check failed")
    expected = build_ledger_seal(value)
    if seal != expected:
        raise RuntimeError("Attempt ledger does not match its external seal")
    return seal


def load_ledger_file(path, expected_identity=None):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        seal = json.loads(ledger_seal_path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read attempt ledger and seal {path}: {exc}") from exc
    validate_ledger_document(value, expected_identity)
    validate_ledger_seal(value, seal)
    return value


def disable_transport_retries(session):
    """Force one adapter send per budgeted Session.request call."""
    for prefix, adapter in session.adapters.items():
        if not isinstance(adapter, HTTPAdapter):
            raise RuntimeError(
                f"Unsupported requests adapter for {prefix!r}; cannot audit retries"
            )
        adapter.max_retries = Retry(
            total=0,
            connect=0,
            read=0,
            redirect=0,
            status=0,
            other=0,
            raise_on_status=False,
        )


def redact_gateway_secrets(value):
    """Remove the configured gateway credential from an HTTP response body."""
    secret = os.getenv("OPENAI_API_KEY", "").strip()
    variants = {secret}
    if secret.startswith("Bearer "):
        variants.add(secret.removeprefix("Bearer ").strip())
    variants.discard("")
    if isinstance(value, bytes):
        redacted = value
        replacement = SECRET_REDACTION.encode("utf-8")
        for item in variants:
            redacted = redacted.replace(item.encode("utf-8"), replacement)
        return redacted
    redacted = str(value)
    for item in variants:
        redacted = redacted.replace(item, SECRET_REDACTION)
    return redacted


def redact_response_secrets(response):
    """Sanitize a requests response before Judge code can persist an error."""
    content = getattr(response, "_content", None)
    if isinstance(content, bytes):
        response._content = redact_gateway_secrets(content)
    return response


def extract_task_id(payload):
    if not isinstance(payload, dict):
        raise RuntimeError("Judge request has no JSON payload")
    user_text = []
    messages = payload.get("messages")
    if messages is None:
        messages = payload.get("input")
    if isinstance(messages, str):
        user_text.append(messages)
        messages = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            user_text.append(content)
        elif isinstance(content, list):
            user_text.extend(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
                and item.get("type") in {"text", "input_text"}
            )
    text = "\n".join(user_text)
    decoder = json.JSONDecoder()
    for marker in INPUT_MARKERS:
        offset = text.find(marker)
        if offset < 0:
            continue
        raw = text[offset + len(marker):].lstrip()
        try:
            value, _ = decoder.raw_decode(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Cannot decode Judge input after {marker}: {exc}") from exc
        task_id = ((value or {}).get("task_spec") or {}).get("id")
        if isinstance(task_id, str) and task_id:
            return task_id
        raise RuntimeError(f"Judge input after {marker} has no task_spec.id")
    raise RuntimeError("Judge request does not contain a recognized input marker")


class AttemptLedger:
    def __init__(
        self, path, *, stage, max_attempts, manifest_sha256, task_ids,
        stage_identity_sha256,
    ):
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.seal_path = ledger_seal_path(self.path)
        self.stage = stage
        self.max_attempts = max_attempts
        self.manifest_sha256 = manifest_sha256
        self.task_ids = set(task_ids)
        self.task_ids_sha256 = json_sha256(sorted(self.task_ids))
        self.stage_identity_sha256 = stage_identity_sha256

    def _new(self):
        value = {
            "schema_version": SCHEMA_VERSION,
            "stage": self.stage,
            "max_attempts_per_task": self.max_attempts,
            "manifest_sha256": self.manifest_sha256,
            "task_ids_sha256": self.task_ids_sha256,
            "task_count": len(self.task_ids),
            "stage_identity_sha256": self.stage_identity_sha256,
            "identity_sha256": None,
            "created_at": now(),
            "updated_at": now(),
            "revision": 0,
            "tasks": {},
        }
        value["identity_sha256"] = ledger_identity_sha256(value)
        value[INTEGRITY_FIELD] = ledger_integrity_sha256(value)
        return value

    def _validate(self, value):
        expected = {
            "schema_version": SCHEMA_VERSION,
            "stage": self.stage,
            "max_attempts_per_task": self.max_attempts,
            "manifest_sha256": self.manifest_sha256,
            "task_ids_sha256": self.task_ids_sha256,
            "task_count": len(self.task_ids),
            "stage_identity_sha256": self.stage_identity_sha256,
        }
        validate_ledger_document(value, expected)

    def _locked_update(self, callback):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.is_file():
                value = load_ledger_file(self.path)
                self._validate(value)
            else:
                if self.seal_path.exists():
                    raise RuntimeError("Attempt ledger seal exists without its ledger")
                value = self._new()
            result = callback(value)
            value["updated_at"] = now()
            value["revision"] = int(value.get("revision") or 0) + 1
            value["identity_sha256"] = ledger_identity_sha256(value)
            value[INTEGRITY_FIELD] = ledger_integrity_sha256(value)
            temporary = self.path.with_name(
                f".{self.path.name}.{os.getpid()}.tmp"
            )
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            seal = build_ledger_seal(value)
            seal_temporary = self.seal_path.with_name(
                f".{self.seal_path.name}.{os.getpid()}.tmp"
            )
            with seal_temporary.open("w", encoding="utf-8") as handle:
                json.dump(seal, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.replace(seal_temporary, self.seal_path)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return result

    def ensure(self):
        self._locked_update(lambda value: None)

    def claim(self, task_id, method, url, payload):
        if task_id not in self.task_ids:
            raise RuntimeError(f"Attempt ledger rejected unknown task_id {task_id!r}")

        def update(value):
            task = value["tasks"].setdefault(task_id, {"attempts": []})
            attempts = task["attempts"]
            if len(attempts) >= self.max_attempts:
                raise AttemptBudgetExceeded(
                    f"{self.stage}/{task_id} exhausted its persistent "
                    f"{self.max_attempts}-request budget"
                )
            attempt = len(attempts) + 1
            attempts.append({
                "attempt": attempt,
                "claimed_at": now(),
                "status": "claimed",
                "method": method.upper(),
                "url": str(url),
                "request_sha256": json_sha256(payload),
            })
            task["attempt_count"] = len(attempts)
            return attempt

        return self._locked_update(update)

    def finish(self, task_id, attempt, *, status, status_code=None, error=None):
        def update(value):
            task = value["tasks"].get(task_id) or {}
            attempts = task.get("attempts") or []
            if attempt < 1 or attempt > len(attempts):
                raise RuntimeError(
                    f"Attempt ledger cannot finish {self.stage}/{task_id} #{attempt}"
                )
            event = attempts[attempt - 1]
            if event.get("attempt") != attempt:
                raise RuntimeError("Attempt ledger sequence is corrupt")
            if event.get("status") != "claimed":
                raise RuntimeError("Attempt ledger event was already finished")
            event.update({"status": status, "finished_at": now()})
            if status_code is not None:
                event["status_code"] = int(status_code)
            if error is not None:
                event["error_type"] = type(error).__name__
                event["error"] = repr(error)[:2000]

        self._locked_update(update)


def load_manifest(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contains_human_labels") is not False:
        raise RuntimeError("Refusing a manifest that is not explicitly label-free")
    task_ids = [item["task_spec"]["id"] for item in value.get("tasks") or []]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise RuntimeError("Manifest task ids are empty or duplicated")
    return task_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage-identity-sha256", required=True)
    parser.add_argument("--max-total-attempts", type=int, default=5)
    parser.add_argument("--module-alias", action="append", default=[])
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.max_total_attempts < 1:
        raise ValueError("--max-total-attempts must be positive")
    target = args.target.resolve()
    manifest = args.manifest.resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    task_ids = load_manifest(manifest)
    ledger = AttemptLedger(
        args.ledger,
        stage=args.stage,
        max_attempts=args.max_total_attempts,
        manifest_sha256=sha256(manifest),
        task_ids=task_ids,
        stage_identity_sha256=args.stage_identity_sha256,
    )
    ledger.ensure()

    original_request = requests.sessions.Session.request

    def budgeted_request(session, method, url, **kwargs):
        payload = kwargs.get("json")
        task_id = extract_task_id(payload)
        disable_transport_retries(session)
        attempt = ledger.claim(task_id, method, url, payload)
        try:
            response = original_request(session, method, url, **kwargs)
        except BaseException as exc:
            ledger.finish(task_id, attempt, status="exception", error=exc)
            raise
        redact_response_secrets(response)
        ledger.finish(
            task_id,
            attempt,
            status="response",
            status_code=getattr(response, "status_code", None),
        )
        return response

    requests.sessions.Session.request = budgeted_request
    try:
        for raw_alias in args.module_alias:
            if "=" not in raw_alias:
                raise ValueError(
                    f"--module-alias expects NAME=PATH, got {raw_alias!r}"
                )
            module_name, raw_path = raw_alias.split("=", 1)
            module_name = module_name.strip()
            module_path = Path(raw_path).expanduser().resolve()
            if not module_name or not module_path.is_file():
                raise ValueError(f"Invalid --module-alias {raw_alias!r}")
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Cannot load module alias {module_name}={module_path}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        target_args = list(args.target_args)
        if target_args and target_args[0] == "--":
            target_args.pop(0)
        sys.argv = [str(target), *target_args]
        runpy.run_path(str(target), run_name="__main__")
    finally:
        requests.sessions.Session.request = original_request


if __name__ == "__main__":
    main()
