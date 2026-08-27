#!/usr/bin/env python3
"""Run the pinned JoyAI live adapter against an external OpenAI-compatible VLM.

The upstream adapter owns all streaming semantics: session state, Query retention,
frame history, chunk rollover, Q&A history, forced pre-Query silence, and optional
mid/long-term summarization. This wrapper only adds deployment concerns required by
the recording campaign: bearer authentication, provider-safe generation options,
health metadata, reset-all, and append-only audit records.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from aiohttp import web
from openai import APIConnectionError, APITimeoutError, OpenAI


ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "third_party" / "joyai_vl_interaction_webinfer"
sys.path.insert(0, str(VENDOR_DIR))

import live_adapter as joyai  # noqa: E402
from memory_summarizer import SummarizerModel  # noqa: E402


LOGGER = logging.getLogger("joyai_scaffold_adapter")
EVALUATION_PROTOCOL = "joyai-formal-query-once-frame-stream-v1"
INPUT_TRANSPORT = "stateful-frame-stream"
UPSTREAM_META = json.loads((VENDOR_DIR / "UPSTREAM.json").read_text(encoding="utf-8"))
REQUEST_CONTEXT: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "joyai_scaffold_request_context",
    default=None,
)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_nonnegative_int(name: str, default: int) -> int:
    value = int(str(os.environ.get(name) or default))
    if value < 0:
        raise RuntimeError(f"{name} must be non-negative")
    return value


def env_positive_float(name: str, default: float) -> float:
    value = float(str(os.environ.get(name) or default))
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def required_env(name: str) -> str:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def response_text_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("value")
        else:
            text = getattr(item, "text", None) or getattr(item, "content", None)
        if text:
            parts.append(str(text).strip())
    return "\n".join(part for part in parts if part)


def extract_assistant_text(message: Any) -> tuple[str, str, dict[str, int]]:
    lengths: dict[str, int] = {}
    for field in ("content", "reasoning_content", "reasoning", "refusal"):
        text = response_text_value(getattr(message, field, None))
        lengths[field] = len(text)
        if text:
            return text, field, lengths
    return "", "empty", lengths


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(item.get("text") or "").strip()
        for item in content
        if isinstance(item, dict)
        and item.get("type") in {"text", "input_text"}
        and str(item.get("text") or "").strip()
    )


def request_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        content_text(message.get("content"))
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and content_text(message.get("content"))
    )


def count_images(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    count = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        count += sum(
            1
            for item in content
            if isinstance(item, dict) and item.get("type") in {"image_url", "input_image"}
        )
    return count


def classify_upstream_failure(
    exc: Exception,
    *,
    hard_deadline_exceeded: bool = False,
) -> dict[str, Any]:
    status = int(getattr(exc, "status_code", 0) or 0)
    if hard_deadline_exceeded or isinstance(exc, (APITimeoutError, asyncio.TimeoutError)):
        return {
            "failure_class": "model_latency_timeout",
            "deadline_exceeded": True,
            "retryable_infrastructure": False,
            "http_status": status,
        }
    if status == 429:
        failure_class = "provider_rate_limit"
        retryable = True
    elif status >= 500:
        failure_class = "provider_service_error"
        retryable = True
    elif isinstance(exc, APIConnectionError):
        failure_class = "provider_transport_error"
        retryable = True
    elif status in {401, 403}:
        failure_class = "provider_auth_error"
        retryable = False
    else:
        failure_class = "provider_request_error"
        retryable = True
    return {
        "failure_class": failure_class,
        "deadline_exceeded": False,
        "retryable_infrastructure": retryable,
        "http_status": status,
    }


class JsonlAudit:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def append(self, record: dict[str, Any]) -> None:
        safe_record = {
            "timestamp": utc_timestamp(),
            **record,
        }
        line = json.dumps(safe_record, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append_sync, line)

    def _append_sync(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)


class ProviderSafeSummarizer(SummarizerModel):
    """Use the official summarizer prompts/state without vLLM-only request fields."""

    def __init__(self, *args: Any, api_key: str, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._client = OpenAI(
            api_key=api_key,
            base_url=kwargs["api_base"],
            max_retries=0,
        )
        longterm_api_base = kwargs.get("longterm_api_base")
        if longterm_api_base and longterm_api_base != kwargs["api_base"]:
            self._longterm_client = OpenAI(
                api_key=api_key,
                base_url=longterm_api_base,
                max_retries=0,
            )
        else:
            self._longterm_client = self._client

    def _chat(
        self,
        messages: list,
        max_tokens: int,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        client: OpenAI = None,
        model_name: str = None,
    ) -> str:
        del top_k, repetition_penalty
        response = (client or self._client).chat.completions.create(
            model=model_name or self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
        )
        return response.choices[0].message.content.strip() if response.choices else ""


class UnifiedJoyAIAdapter(joyai.StreamingInferAdapter):
    def __init__(
        self,
        config: joyai.AdapterConfig,
        *,
        audit: JsonlAudit,
        system_prompt_sha256: str,
        system_prompt_file_sha256: str,
        provider_extra_body: dict[str, Any],
        sdk_max_retries: int,
        non_query_timeout_seconds: float,
        query_timeout_seconds: float,
        warmup_timeout_seconds: float,
    ):
        super().__init__(config)
        self.audit = audit
        self.system_prompt_sha256 = system_prompt_sha256
        self.system_prompt_file_sha256 = system_prompt_file_sha256
        self.provider_extra_body = provider_extra_body
        self.sdk_max_retries = sdk_max_retries
        self.non_query_timeout_seconds = non_query_timeout_seconds
        self.query_timeout_seconds = query_timeout_seconds
        self.warmup_timeout_seconds = warmup_timeout_seconds
        self.session_frame_indices: dict[str, int] = {}
        self.session_query_queues: dict[str, list[dict[str, Any]]] = {}
        self.session_query_records: dict[str, dict[str, dict[str, Any]]] = {}
        self.provider_warmup_lock = asyncio.Lock()
        self.provider_warmup_status: dict[str, Any] = {
            "status": "not_run",
            "attempts": 0,
        }

    def _main_generation_kwargs(self, inbound_payload: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "max_tokens": self.config.main_max_tokens,
            "temperature": self.config.main_temperature,
            "top_p": self.config.main_top_p,
        }
        if self.config.honor_inbound_generation_params:
            result = {
                "max_tokens": inbound_payload.get("max_tokens", self.config.main_max_tokens),
                "temperature": inbound_payload.get("temperature", self.config.main_temperature),
                "top_p": inbound_payload.get("top_p", self.config.main_top_p),
            }
        if self.provider_extra_body:
            result["extra_body"] = dict(self.provider_extra_body)
        return result

    async def _call_main_model(
        self,
        inbound_payload: dict[str, Any],
        api_messages: list[dict[str, Any]],
        *,
        client=None,
        model_name: Optional[str] = None,
        session_state=None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        http_messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, Optional[dict[str, Any]]]:
        client = client or self.main_client
        model_name = model_name or self.config.main_model
        messages = http_messages or self._build_main_http_messages(api_messages, inbound_payload)
        generation_kwargs = generation_kwargs or self._main_generation_kwargs(inbound_payload)
        context = REQUEST_CONTEXT.get() or {}
        request_id = str(context.get("request_id") or uuid.uuid4())
        query_text = str(context.get("query_text") or "")
        request_kind = "query" if query_text else "background_frame"
        timeout_seconds = (
            self.query_timeout_seconds
            if query_text
            else self.non_query_timeout_seconds
        )
        await self.audit.append({
            "event": "joyai_scaffold_upstream_request",
            "request_id": request_id,
            "model": model_name,
            "session_id": context.get("session_id", "default"),
            "upstream_context_image_count": count_images(messages),
            "ingress_user_query_present": bool(query_text),
            "ingress_user_query_sha256": sha256_text(query_text) if query_text else "",
            "user_query_present": bool(query_text),
            "user_query_sha256": sha256_text(query_text) if query_text else "",
            "query_context_policy": "official-live-adapter-history-replay",
            "generation_parameter_policy": (
                "honor-webui-ingress"
                if self.config.honor_inbound_generation_params
                else "adapter-defaults"
            ),
            "max_tokens": generation_kwargs.get("max_tokens"),
            "temperature": generation_kwargs.get("temperature"),
            "top_p": generation_kwargs.get("top_p"),
            "thinking_mode": (
                (generation_kwargs.get("extra_body") or {}).get("thinking") or {}
            ).get("type", "provider-default"),
            "query_frame_queue_event_id": context.get("query_event_id", ""),
            "query_frame_queue_delay_ms": context.get("query_queue_delay_ms"),
            "query_frame_time_range": context.get("query_frame_time_range", ""),
            "request_kind": request_kind,
            "request_timeout_s": timeout_seconds,
            "sdk_max_retries": self.sdk_max_retries,
            "provider_attempt_budget": 1 + self.sdk_max_retries,
            "scaffold_upstream_commit": UPSTREAM_META["commit"],
        })
        started = time.perf_counter()
        try:
            request_client = client.with_options(
                max_retries=self.sdk_max_retries,
                timeout=timeout_seconds,
            )
            response = await asyncio.wait_for(
                request_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **generation_kwargs,
                ),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            failure = classify_upstream_failure(
                exc,
                hard_deadline_exceeded=isinstance(exc, asyncio.TimeoutError),
            )
            await self.audit.append({
                "event": "upstream_response_received",
                "request_id": request_id,
                "request_model": model_name,
                "http_status": failure["http_status"],
                "ok": False,
                "has_assistant_message": False,
                "error_type": type(exc).__name__,
                "failure_class": failure["failure_class"],
                "deadline_exceeded": failure["deadline_exceeded"],
                "retryable_infrastructure": failure["retryable_infrastructure"],
                "request_kind": request_kind,
                "request_timeout_s": timeout_seconds,
                "sdk_max_retries": self.sdk_max_retries,
                "provider_attempt_budget": 1 + self.sdk_max_retries,
                "latency_ms": elapsed_ms,
                "adapter_response_policy": (
                    "audited_synthetic_silence_http_200"
                    if failure["failure_class"] == "model_latency_timeout"
                    else "non_retryable_http_424"
                ),
                "system_prompt_file_sha256": self.system_prompt_file_sha256,
                "system_prompt_sha256": self.system_prompt_sha256,
            })
            if failure["failure_class"] == "model_latency_timeout":
                # A non-2xx response would make the WebUI's OpenAI client retry the
                # same stale frame. The audit event remains the source of truth.
                return "</silence>", None
            raise web.HTTPFailedDependency(
                text=json.dumps({
                    "error": {
                        "message": "Upstream provider request failed",
                        "type": failure["failure_class"],
                    }
                }),
                content_type="application/json",
            ) from exc
        message = response.choices[0].message if response.choices else None
        raw_text, response_text_source, response_field_lengths = extract_assistant_text(message)
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        await self.audit.append({
            "event": "upstream_response_received",
            "request_id": request_id,
            "request_model": model_name,
            "response_id": str(getattr(response, "id", "") or ""),
            "response_model": str(getattr(response, "model", "") or model_name),
            "http_status": 200,
            "ok": True,
            "has_assistant_message": bool(response.choices),
            "response_text_source": response_text_source,
            "response_field_lengths": response_field_lengths,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "failure_class": "none",
            "deadline_exceeded": False,
            "retryable_infrastructure": False,
            "request_kind": request_kind,
            "request_timeout_s": timeout_seconds,
            "sdk_max_retries": self.sdk_max_retries,
            "provider_attempt_budget": 1 + self.sdk_max_retries,
            "system_prompt_file_sha256": self.system_prompt_file_sha256,
            "system_prompt_sha256": self.system_prompt_sha256,
        })
        return raw_text or "", usage

    def _query_records(self, session_id: str) -> dict[str, dict[str, Any]]:
        return self.session_query_records.setdefault(session_id, {})

    def _query_queue(self, session_id: str) -> list[dict[str, Any]]:
        return self.session_query_queues.setdefault(session_id, [])

    def _observe_webui_query(self, session_id: str, query_text: str) -> Optional[str]:
        if not query_text:
            return None
        query_hash = sha256_text(query_text)
        for record in self._query_records(session_id).values():
            if record["user_query_sha256"] == query_hash and not record["webui_ingress_seen"]:
                record["webui_ingress_seen"] = True
                return str(record["query_event_id"])
        return None

    @staticmethod
    def _payload_for_queued_query(
        inbound_payload: dict[str, Any],
        record: dict[str, Any],
    ) -> dict[str, Any]:
        payload = dict(inbound_payload)
        payload["messages"] = [{
            "role": "user",
            "content": [
                {"type": "text", "text": record["query"]},
                {"type": "image_url", "image_url": {"url": record["image_url"]}},
            ],
        }]
        payload["frame_time_ranges"] = [record["frame_time_range"]]
        payload.pop("frame_time_range", None)
        return payload

    async def handle_query_event(self, request: web.Request) -> web.Response:
        payload = await joyai._read_json(request)
        session_id = joyai._safe_session_id(str(payload.get("session_id") or ""))
        query_event_id = str(payload.get("query_event_id") or "").strip()
        query = str(payload.get("query") or "").strip()
        image_url = str(payload.get("image_url") or "")
        frame_time_range = str(payload.get("frame_time_range") or "").strip()
        if not session_id or not query_event_id or not query or not frame_time_range:
            raise web.HTTPBadRequest(
                text="session_id, query_event_id, query, and frame_time_range are required"
            )
        if not image_url.startswith("data:image/jpeg;base64,"):
            raise web.HTTPBadRequest(text="image_url must be an inline JPEG data URL")
        if len(image_url) > 28 * 1024 * 1024:
            raise web.HTTPRequestEntityTooLarge(
                max_size=28 * 1024 * 1024,
                actual_size=len(image_url),
            )

        records = self._query_records(session_id)
        existing = records.get(query_event_id)
        query_hash = sha256_text(query)
        if existing is not None:
            if (
                existing["user_query_sha256"] != query_hash
                or existing["frame_time_range"] != frame_time_range
            ):
                raise web.HTTPConflict(text="query_event_id was reused with different content")
            return web.json_response({
                "ok": True,
                "idempotent": True,
                "query_event_id": query_event_id,
                "status": existing["status"],
                "queue_position": next(
                    (
                        index + 1
                        for index, item in enumerate(self._query_queue(session_id))
                        if item["query_event_id"] == query_event_id
                    ),
                    0,
                ),
            })

        now_epoch_ms = time.time() * 1000
        try:
            ui_query_sent_epoch_ms = datetime.fromisoformat(
                str(payload.get("ui_query_sent_at") or "").replace("Z", "+00:00")
            ).timestamp() * 1000
        except (TypeError, ValueError):
            ui_query_sent_epoch_ms = None
        record = {
            "query_event_id": query_event_id,
            "session_id": session_id,
            "query": query,
            "user_query_sha256": query_hash,
            "image_url": image_url,
            "image_sha256": sha256_text(image_url),
            "frame_time_range": frame_time_range,
            "ui_query_sent_at": str(payload.get("ui_query_sent_at") or ""),
            "ui_query_video_time_s": payload.get("ui_query_video_time_s"),
            "annotated_query_video_time_s": payload.get(
                "annotated_query_video_time_s"
            ),
            "captured_media_time_s": payload.get("captured_media_time_s"),
            "captured_raw_media_time_s": payload.get("captured_raw_media_time_s"),
            "enqueued_at": utc_timestamp(),
            "enqueued_epoch_ms": now_epoch_ms,
            "acceptance_delay_ms": (
                max(0.0, now_epoch_ms - ui_query_sent_epoch_ms)
                if ui_query_sent_epoch_ms is not None
                else None
            ),
            "status": "queued",
            "webui_ingress_seen": False,
        }
        records[query_event_id] = record
        queue = self._query_queue(session_id)
        queue.append(record)
        await self.audit.append({
            "event": "query_frame_queued",
            "query_event_id": query_event_id,
            "session_id": session_id,
            "user_query_present": True,
            "user_query_sha256": query_hash,
            "frame_time_range": frame_time_range,
            "image_sha256": record["image_sha256"],
            "ui_query_sent_at": record["ui_query_sent_at"],
            "ui_query_video_time_s": record["ui_query_video_time_s"],
            "annotated_query_video_time_s": record[
                "annotated_query_video_time_s"
            ],
            "captured_media_time_s": record["captured_media_time_s"],
            "captured_raw_media_time_s": record["captured_raw_media_time_s"],
            "acceptance_delay_ms": record["acceptance_delay_ms"],
            "queue_position": len(queue),
            "query_frame_queue_policy": "fifo-query-time-frame",
        })
        return web.json_response({
            "ok": True,
            "idempotent": False,
            "query_event_id": query_event_id,
            "status": "queued",
            "queue_position": len(queue),
            "acceptance_delay_ms": record["acceptance_delay_ms"],
        })

    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        inbound_payload = await joyai._read_json(request)
        session_id = joyai._safe_session_id(
            joyai._request_session_id(request, inbound_payload)
        )
        requested_model = inbound_payload.get("model")
        client, model_name = self._resolve_backend(requested_model)
        inbound_messages = inbound_payload.get("messages") or []
        inbound_query_text = joyai._strip_time_range_from_text(
            request_user_text(inbound_messages)
        )
        inbound_image_count = count_images(inbound_messages)
        state = self.get_session(session_id)
        async with state.lock:
            observed_query_event_id = self._observe_webui_query(
                session_id, inbound_query_text
            )
            queue = self._query_queue(session_id)
            queued_query = queue.pop(0) if queue else None
            if queued_query is not None:
                queued_query["status"] = "processing"
                queued_query["dequeued_at"] = utc_timestamp()
                queued_query["query_queue_delay_ms"] = round(
                    time.time() * 1000 - queued_query["enqueued_epoch_ms"], 3
                )
                payload = self._payload_for_queued_query(inbound_payload, queued_query)
                query_text = str(queued_query["query"])
                image_count = 1
            else:
                payload = inbound_payload
                query_text = inbound_query_text
                image_count = inbound_image_count

            frame_index = self.session_frame_indices.get(session_id, 0) + image_count
            self.session_frame_indices[session_id] = frame_index
            request_id = str(uuid.uuid4())
            await self.audit.append({
                "event": "joyai_scaffold_frame_received",
                "request_id": request_id,
                "model": model_name,
                "session_id": session_id,
                "frame_index": frame_index,
                "image_count": image_count,
                "inbound_image_count": inbound_image_count,
                "video_input_count": 0,
                "input_transport": INPUT_TRANSPORT,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "query_delivery": "once_per_round",
                "query_round_control": (
                    "fifo-query-time-frame" if queued_query else "webui-ingress-event"
                ),
                "query_frame_queue_policy": (
                    "fifo-query-time-frame" if queued_query else ""
                ),
                "query_frame_queue_event_id": (
                    queued_query["query_event_id"] if queued_query else ""
                ),
                "query_frame_time_range": (
                    queued_query["frame_time_range"] if queued_query else ""
                ),
                "query_frame_queue_delay_ms": (
                    queued_query["query_queue_delay_ms"] if queued_query else None
                ),
                "ui_query_sent_at": (
                    queued_query["ui_query_sent_at"] if queued_query else ""
                ),
                "ui_query_video_time_s": (
                    queued_query["ui_query_video_time_s"] if queued_query else None
                ),
                "webui_ingress_query_present": bool(inbound_query_text),
                "webui_ingress_query_event_id": observed_query_event_id or "",
                "user_query_present": bool(query_text),
                "user_query_sha256": sha256_text(query_text) if query_text else "",
                "persistent_query_injected": False,
                "system_prompt_file_sha256": self.system_prompt_file_sha256,
                "system_prompt_sha256": self.system_prompt_sha256,
                "scaffold_upstream_commit": UPSTREAM_META["commit"],
            })
            token = REQUEST_CONTEXT.set({
                "request_id": request_id,
                "session_id": session_id,
                "query_text": query_text,
                "query_event_id": (
                    queued_query["query_event_id"] if queued_query else ""
                ),
                "query_queue_delay_ms": (
                    queued_query["query_queue_delay_ms"] if queued_query else None
                ),
                "query_frame_time_range": (
                    queued_query["frame_time_range"] if queued_query else ""
                ),
            })
            try:
                try:
                    result = await self._handle_chat_payload(
                        state,
                        payload,
                        request,
                        client=client,
                        model_name=model_name,
                    )
                except web.HTTPException:
                    raise
                except Exception as exc:
                    LOGGER.exception("chat completion failed")
                    return joyai._openai_error_response(str(exc), status=502)
            finally:
                REQUEST_CONTEXT.reset(token)
            if queued_query is not None:
                queued_query["status"] = "delivered"
                queued_query["request_id"] = request_id
                queued_query["delivered_at"] = utc_timestamp()
                harness = result.setdefault("streamingharness", {})
                harness["query_frame_queue"] = {
                    "policy": "fifo-query-time-frame",
                    "query_event_id": queued_query["query_event_id"],
                    "user_query_sha256": queued_query["user_query_sha256"],
                    "frame_time_range": queued_query["frame_time_range"],
                    "ui_query_sent_at": queued_query["ui_query_sent_at"],
                    "ui_query_video_time_s": queued_query["ui_query_video_time_s"],
                    "query_queue_delay_ms": queued_query["query_queue_delay_ms"],
                    "webui_ingress_seen": queued_query["webui_ingress_seen"],
                }
                await self.audit.append({
                    "event": "query_frame_dequeued",
                    "query_event_id": queued_query["query_event_id"],
                    "request_id": request_id,
                    "session_id": session_id,
                    "user_query_present": True,
                    "user_query_sha256": queued_query["user_query_sha256"],
                    "frame_time_range": queued_query["frame_time_range"],
                    "query_queue_delay_ms": queued_query["query_queue_delay_ms"],
                    "webui_ingress_seen": queued_query["webui_ingress_seen"],
                    "query_frame_queue_policy": "fifo-query-time-frame",
                })
        return web.json_response(result)

    async def handle_health(self, request: web.Request) -> web.Response:
        del request
        return web.json_response({
            "ok": True,
            "adapter": "joyai-official-live-adapter",
            "model": self.config.main_model,
            "backends": list(self.main_clients.keys()),
            "sessions": len(self.sessions),
            "summarizer_enabled": self.summarizer is not None,
            "upstream_protocol": "openai-chat",
            "input_transport": INPUT_TRANSPORT,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "query_delivery": "once_per_round",
            "query_frame_queue": {
                "enabled": True,
                "policy": "fifo-query-time-frame",
                "queued_events": sum(len(queue) for queue in self.session_query_queues.values()),
            },
            "query_context_policy": "official-live-adapter-history-replay",
            "generation_parameter_policy": (
                "honor-webui-ingress"
                if self.config.honor_inbound_generation_params
                else "adapter-defaults"
            ),
            "main_generation_defaults": {
                "max_tokens": self.config.main_max_tokens,
                "temperature": self.config.main_temperature,
                "top_p": self.config.main_top_p,
            },
            "upstream_request_policy": {
                "sdk_max_retries": self.sdk_max_retries,
                "non_query_timeout_s": self.non_query_timeout_seconds,
                "query_timeout_s": self.query_timeout_seconds,
                "warmup_timeout_s": self.warmup_timeout_seconds,
                "timeout_classification": "model_capability",
                "timeout_transport": "audited_synthetic_silence_http_200",
                "rate_limit_and_transport_classification": "retryable_infrastructure",
                "infrastructure_error_transport": "non_retryable_http_424",
            },
            "system_prompt_file_sha256": self.system_prompt_file_sha256,
            "system_prompt_sha256": self.system_prompt_sha256,
            "scaffold_upstream_repository": UPSTREAM_META["repository"],
            "scaffold_upstream_commit": UPSTREAM_META["commit"],
            "provider_warmup": dict(self.provider_warmup_status),
        })

    async def handle_provider_warmup(self, request: web.Request) -> web.Response:
        """Run one real multimodal upstream request without touching session state."""
        payload = await joyai._read_json(request)
        requested_model = payload.get("model") or self.config.main_model
        client, model_name = self._resolve_backend(requested_model)
        image_url = str(payload.get("image_url") or "")
        if not image_url.startswith("data:image/jpeg;base64,"):
            raise web.HTTPBadRequest(text="image_url must be an inline JPEG data URL")
        if len(image_url) > 28 * 1024 * 1024:
            raise web.HTTPRequestEntityTooLarge(max_size=28 * 1024 * 1024, actual_size=len(image_url))

        request_id = str(uuid.uuid4())
        sessions_before = set(self.sessions)
        await self.audit.append({
            "event": "provider_warmup_started",
            "request_id": request_id,
            "model": model_name,
            "multimodal_input": True,
            "session_state_isolated": True,
            "request_timeout_s": self.warmup_timeout_seconds,
            "sdk_max_retries": self.sdk_max_retries,
            "provider_attempt_budget": 1 + self.sdk_max_retries,
            "scaffold_upstream_commit": UPSTREAM_META["commit"],
        })
        started = time.perf_counter()
        try:
            async with self.provider_warmup_lock:
                generation_kwargs = self._main_generation_kwargs({})
                generation_kwargs["max_tokens"] = min(
                    int(generation_kwargs.get("max_tokens") or 32),
                    32,
                )
                request_client = client.with_options(
                    max_retries=self.sdk_max_retries,
                    timeout=self.warmup_timeout_seconds,
                )
                response = await asyncio.wait_for(
                    request_client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "user",
                            "content": [{
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            }],
                        }],
                        **generation_kwargs,
                    ),
                    timeout=self.warmup_timeout_seconds,
                )
            if not response.choices or response.choices[0].message is None:
                raise RuntimeError("Provider warmup returned no assistant message")
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            failure = classify_upstream_failure(
                exc,
                hard_deadline_exceeded=isinstance(exc, asyncio.TimeoutError),
            )
            upstream_status = failure["http_status"]
            response_status = upstream_status if 400 <= upstream_status <= 599 else 502
            self.provider_warmup_status = {
                "status": "failed",
                "attempts": int(self.provider_warmup_status.get("attempts") or 0) + 1,
                "last_attempt_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "latency_ms": latency_ms,
                "http_status": upstream_status,
                "error_type": type(exc).__name__,
                "failure_class": failure["failure_class"],
                "deadline_exceeded": failure["deadline_exceeded"],
                "retryable_infrastructure": failure["retryable_infrastructure"],
                "request_timeout_s": self.warmup_timeout_seconds,
                "sdk_max_retries": self.sdk_max_retries,
            }
            await self.audit.append({
                "event": "provider_warmup_completed",
                "request_id": request_id,
                "model": model_name,
                "ok": False,
                "http_status": upstream_status,
                "latency_ms": latency_ms,
                "error_type": type(exc).__name__,
                "failure_class": failure["failure_class"],
                "deadline_exceeded": failure["deadline_exceeded"],
                "retryable_infrastructure": failure["retryable_infrastructure"],
                "request_timeout_s": self.warmup_timeout_seconds,
                "sdk_max_retries": self.sdk_max_retries,
                "provider_attempt_budget": 1 + self.sdk_max_retries,
                "session_state_unchanged": sessions_before == set(self.sessions),
            })
            return web.json_response({
                "ok": False,
                "error": {
                    "message": f"Provider warmup failed: {type(exc).__name__}",
                    "type": type(exc).__name__,
                },
            }, status=response_status)

        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        session_state_unchanged = sessions_before == set(self.sessions)
        response_model = str(getattr(response, "model", "") or model_name)
        response_id = str(getattr(response, "id", "") or "")
        self.provider_warmup_status = {
            "status": "ready",
            "attempts": int(self.provider_warmup_status.get("attempts") or 0) + 1,
            "last_attempt_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "latency_ms": latency_ms,
            "response_model": response_model,
            "multimodal_input": True,
            "session_state_unchanged": session_state_unchanged,
        }
        await self.audit.append({
            "event": "provider_warmup_completed",
            "request_id": request_id,
            "model": model_name,
            "response_id": response_id,
            "response_model": response_model,
            "ok": True,
            "http_status": 200,
            "latency_ms": latency_ms,
            "failure_class": "none",
            "deadline_exceeded": False,
            "retryable_infrastructure": False,
            "request_timeout_s": self.warmup_timeout_seconds,
            "sdk_max_retries": self.sdk_max_retries,
            "provider_attempt_budget": 1 + self.sdk_max_retries,
            "multimodal_input": True,
            "session_state_unchanged": session_state_unchanged,
        })
        return web.json_response({
            "ok": True,
            "model": model_name,
            "response_model": response_model,
            "response_id": response_id,
            "latency_ms": latency_ms,
            "multimodal_input": True,
            "session_state_unchanged": session_state_unchanged,
        })

    async def handle_reset_all(self, request: web.Request) -> web.Response:
        del request
        states = list(self.sessions.values())
        self.sessions.clear()
        self.session_frame_indices.clear()
        self.session_query_queues.clear()
        self.session_query_records.clear()
        for state in states:
            for job in state.async_pending_summary_jobs:
                job["task"].cancel()
            await self._flush_session_outputs(state)
        await self.audit.append({
            "event": "conversation_state_reset",
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "sessions_removed": len(states),
        })
        return web.json_response({"ok": True, "sessions_removed": len(states)})

    async def handle_reset(self, request: web.Request) -> web.Response:
        payload = await joyai._read_json(request)
        session_id = joyai._safe_session_id(joyai._request_session_id(request, payload))
        removed_state = self.sessions.pop(session_id, None)
        self.session_frame_indices.pop(session_id, None)
        self.session_query_queues.pop(session_id, None)
        self.session_query_records.pop(session_id, None)
        if removed_state is not None:
            for job in removed_state.async_pending_summary_jobs:
                job["task"].cancel()
            await self._flush_session_outputs(removed_state)
        return web.json_response({
            "ok": True,
            "session_id": session_id,
            "removed": removed_state is not None,
        })


def build_provider_extra_body() -> dict[str, Any]:
    raw = str(os.environ.get("JOYAI_SCAFFOLD_PROVIDER_EXTRA_BODY_JSON") or "").strip()
    result = json.loads(raw) if raw else {}
    if not isinstance(result, dict):
        raise RuntimeError("JOYAI_SCAFFOLD_PROVIDER_EXTRA_BODY_JSON must be a JSON object")
    if env_bool("PROMPT_PROXY_DISABLE_THINKING"):
        result.setdefault("thinking", {"type": "disabled"})
    return result


def create_application() -> web.Application:
    host = str(os.environ.get("PROMPT_PROXY_HOST") or "127.0.0.1")
    port = int(os.environ.get("PROMPT_PROXY_PORT") or "18070")
    upstream_api_base = required_env("UPSTREAM_API_BASE").rstrip("/")
    upstream_api_key = required_env("UPSTREAM_API_KEY")
    upstream_model = required_env("PROMPT_PROXY_ADVERTISED_MODEL")
    access_token = required_env("PROMPT_PROXY_ACCESS_TOKEN")
    if len(access_token) < 24:
        raise RuntimeError("PROMPT_PROXY_ACCESS_TOKEN must contain at least 24 characters")

    prompt_path = Path(
        os.environ.get("JOYAI_SYSTEM_PROMPT_FILE")
        or ROOT / "config" / "joyai_system_prompt.txt"
    ).resolve()
    prompt_bytes = prompt_path.read_bytes()
    system_prompt = prompt_bytes.decode("utf-8").strip()
    if not system_prompt:
        raise RuntimeError(f"System prompt is empty: {prompt_path}")
    system_prompt_file_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    system_prompt_sha256 = sha256_text(system_prompt)
    audit_path = Path(
        os.environ.get("PROMPT_PROXY_AUDIT_PATH")
        or ROOT / "outputs" / "joyai_scaffold_adapter_audit.jsonl"
    ).resolve()
    audit = JsonlAudit(audit_path)

    enable_summarizer = env_bool("JOYAI_SCAFFOLD_ENABLE_SUMMARIZER", False)
    summary_api_base = str(os.environ.get("JOYAI_SCAFFOLD_SUMMARIZER_API_BASE") or "").strip()
    summary_model = str(os.environ.get("JOYAI_SCAFFOLD_SUMMARIZER_MODEL") or "").strip()
    summary_api_key = str(os.environ.get("JOYAI_SCAFFOLD_SUMMARIZER_API_KEY") or "").strip()
    if enable_summarizer and not (summary_api_base and summary_model and summary_api_key):
        raise RuntimeError(
            "Summarization requires JOYAI_SCAFFOLD_SUMMARIZER_API_BASE, "
            "JOYAI_SCAFFOLD_SUMMARIZER_MODEL, and JOYAI_SCAFFOLD_SUMMARIZER_API_KEY"
        )

    sdk_max_retries = env_nonnegative_int("JOYAI_SCAFFOLD_SDK_MAX_RETRIES", 0)
    non_query_timeout_seconds = env_positive_float(
        "JOYAI_SCAFFOLD_NON_QUERY_TIMEOUT_SECONDS", 30.0
    )
    query_timeout_seconds = env_positive_float(
        "JOYAI_SCAFFOLD_QUERY_TIMEOUT_SECONDS", 180.0
    )
    warmup_timeout_seconds = env_positive_float(
        "JOYAI_SCAFFOLD_WARMUP_TIMEOUT_SECONDS", 180.0
    )
    if non_query_timeout_seconds > query_timeout_seconds:
        raise RuntimeError(
            "JOYAI_SCAFFOLD_NON_QUERY_TIMEOUT_SECONDS must not exceed "
            "JOYAI_SCAFFOLD_QUERY_TIMEOUT_SECONDS"
        )

    config = joyai.AdapterConfig(
        host=host,
        port=port,
        adapter_model=upstream_model,
        main_api_base=upstream_api_base,
        main_model=upstream_model,
        api_key=upstream_api_key,
        frame_seconds=1.0,
        max_pixels=int(os.environ.get("JOYAI_SCAFFOLD_MAX_PIXELS") or "262144"),
        main_max_tokens=int(os.environ.get("JOYAI_SCAFFOLD_MAX_TOKENS") or "512"),
        main_temperature=float(os.environ.get("JOYAI_SCAFFOLD_TEMPERATURE") or "0.7"),
        main_top_p=float(os.environ.get("JOYAI_SCAFFOLD_TOP_P") or "0.9"),
        honor_inbound_generation_params=env_bool(
            "JOYAI_SCAFFOLD_HONOR_INBOUND_GENERATION_PARAMS",
            True,
        ),
        chunk=int(os.environ.get("JOYAI_SCAFFOLD_CHUNK") or "100"),
        compress_every_n_chunks=int(
            os.environ.get("JOYAI_SCAFFOLD_COMPRESS_EVERY_N_CHUNKS") or "5"
        ),
        async_summary_lead_frames=int(
            os.environ.get("JOYAI_SCAFFOLD_ASYNC_SUMMARY_LEAD_FRAMES") or "20"
        ),
        use_prompt_as_query=True,
        force_silence_before_query=True,
        keep_qa_history=True,
        normalize_output=True,
        enable_summarizer=False,
        request_timeout_seconds=max(
            non_query_timeout_seconds,
            query_timeout_seconds,
            warmup_timeout_seconds,
        ),
        session_timeout_seconds=float(
            os.environ.get("JOYAI_SCAFFOLD_SESSION_TIMEOUT_SECONDS") or "3600"
        ),
        per_session_dirs=False,
        save_model_inputs=False,
        save_debug_inputs=False,
        summarizer_debug=False,
        frame_save_dir=str(Path("/tmp") / f"vl-interaction-joyai-scaffold-{port}"),
        language="en",
        system_prompt=system_prompt,
    )
    adapter = UnifiedJoyAIAdapter(
        config,
        audit=audit,
        system_prompt_sha256=system_prompt_sha256,
        system_prompt_file_sha256=system_prompt_file_sha256,
        provider_extra_body=build_provider_extra_body(),
        sdk_max_retries=sdk_max_retries,
        non_query_timeout_seconds=non_query_timeout_seconds,
        query_timeout_seconds=query_timeout_seconds,
        warmup_timeout_seconds=warmup_timeout_seconds,
    )
    if enable_summarizer:
        adapter.summarizer = ProviderSafeSummarizer(
            model_name=summary_model,
            api_base=summary_api_base,
            longterm_model_name=summary_model,
            longterm_api_base=summary_api_base,
            mid_term_max_tokens=config.mid_term_max_tokens,
            mid_term_target_tokens=config.mid_term_target_tokens,
            long_term_max_tokens=config.long_term_max_tokens,
            long_term_target_tokens=config.long_term_target_tokens,
            key_frames_per_chunk=config.summarizer_key_frames,
            max_pixels=config.summarizer_max_pixels,
            prompt_phase_seconds=config.summarizer_phase_seconds,
            mid_term_temperature=config.mid_term_temperature,
            mid_term_top_p=config.mid_term_top_p,
            mid_term_top_k=config.mid_term_top_k,
            mid_term_repetition_penalty=config.mid_term_repetition_penalty,
            mid_term_presence_penalty=config.mid_term_presence_penalty,
            long_term_temperature=config.long_term_temperature,
            long_term_top_p=config.long_term_top_p,
            long_term_top_k=config.long_term_top_k,
            long_term_repetition_penalty=config.long_term_repetition_penalty,
            long_term_presence_penalty=config.long_term_presence_penalty,
            debug=config.summarizer_debug,
            api_key=summary_api_key,
        )

    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        if request.path == "/health":
            return await handler(request)
        expected = f"Bearer {access_token}"
        actual = str(request.headers.get("authorization") or "")
        if not hmac.compare_digest(actual, expected):
            return web.json_response(
                {"error": {"message": "Unauthorized JoyAI scaffold request"}},
                status=401,
            )
        return await handler(request)

    app = web.Application(client_max_size=128 * 1024 * 1024, middlewares=[auth_middleware])
    app["adapter"] = adapter

    async def on_startup(_app: web.Application) -> None:
        adapter.start_background_tasks()

    async def on_cleanup(_app: web.Application) -> None:
        states = list(adapter.sessions.values())
        adapter.sessions.clear()
        for state in states:
            for job in state.async_pending_summary_jobs:
                job["task"].cancel()
            await adapter._flush_session_outputs(state)
        await adapter.main_client.close()
        for client, _model in adapter.main_clients.values():
            if client is not adapter.main_client:
                await client.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/health", adapter.handle_health)
    app.router.add_get("/v1/models", adapter.handle_models)
    app.router.add_post("/v1/warmup", adapter.handle_provider_warmup)
    app.router.add_post("/v1/query-events", adapter.handle_query_event)
    app.router.add_post("/v1/chat/completions", adapter.handle_chat_completions)
    app.router.add_post("/v1/streaming/reset", adapter.handle_reset)
    app.router.add_post("/reset", adapter.handle_reset_all)
    app["listen_host"] = host
    app["listen_port"] = port
    return app


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_application()
    LOGGER.info(
        "Starting pinned JoyAI scaffold commit=%s model=%s summarizer=%s",
        UPSTREAM_META["commit"],
        app["adapter"].config.main_model,
        app["adapter"].summarizer is not None,
    )
    web.run_app(
        app,
        host=app["listen_host"],
        port=app["listen_port"],
        print=None,
    )


if __name__ == "__main__":
    main()
