#!/usr/bin/env python3
"""Serve MOSS-Realtime or Mage through OpenAI and JoyAI-compatible APIs."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from service.backends import MageVLBackend, MossVLBackend
from service.streaming import (
    MageStreamingEngine,
    MageStreamState,
    RealtimeOutputBuffer,
    resolve_frame_windows,
    streaming_response,
)


LOGGER = logging.getLogger(__name__)


class SessionCapacityError(RuntimeError):
    """Raised when the single native MOSS session is occupied."""


def decode_image_url(value: Any, allowed_roots: tuple[Path, ...]) -> Image.Image:
    if isinstance(value, dict):
        value = value.get("url", "")
    value = str(value or "")
    if value.startswith("data:image/"):
        try:
            raw = base64.b64decode(value.split(",", 1)[1], validate=True)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except (IndexError, ValueError, OSError) as exc:
            raise ValueError("Invalid image data URL") from exc
    if value.startswith("file://"):
        value = value[7:]
    path = Path(value).expanduser().resolve()
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise ValueError(f"Local image is outside ALLOWED_LOCAL_MEDIA_ROOTS: {path}")
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def resolve_video_url(value: Any, allowed_roots: tuple[Path, ...]) -> str:
    if isinstance(value, dict):
        value = value.get("url", value.get("path", ""))
    value = str(value or "")
    if value.startswith("file://"):
        value = value[7:]
    path = Path(value).expanduser().resolve()
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise ValueError(f"Local video is outside ALLOWED_LOCAL_MEDIA_ROOTS: {path}")
    if not path.is_file():
        raise ValueError(f"Video does not exist: {path}")
    return str(path)


def parse_messages(
    messages: Any,
    allowed_roots: tuple[Path, ...],
    *,
    allow_empty: bool = False,
) -> tuple[str, list[Image.Image], Optional[str]]:
    if allow_empty and not messages:
        return "", [], None
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    transcript: list[str] = []
    images: list[Image.Image] = []
    video_path = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in ("text", "input_text"):
                    texts.append(str(item.get("text", "")))
                elif item_type in ("image_url", "input_image", "image"):
                    source = item.get("image_url", item.get("image", ""))
                    images.append(decode_image_url(source, allowed_roots))
                elif item_type in ("video_url", "input_video", "video"):
                    source = item.get("video_url", item.get("video", ""))
                    video_path = resolve_video_url(source, allowed_roots)
        text = "\n".join(part for part in texts if part).strip()
        if text:
            transcript.append(f"{role}: {text}")
    return "\n".join(transcript).strip(), images, video_path


def latest_role_text(messages: Any, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or str(message.get("role", "user")) != role:
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, dict)
                and item.get("type") in ("text", "input_text")
                and str(item.get("text", "")).strip()
            ]
            return "\n".join(parts).strip()
    return ""


@dataclass
class SessionState:
    frames: list[Image.Image] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    system_prompt: str = ""
    last_prompt: str = ""
    moss_output: RealtimeOutputBuffer = field(default_factory=RealtimeOutputBuffer)
    mage: MageStreamState = field(default_factory=MageStreamState)
    updated_at: float = field(default_factory=time.time)


class RuntimeService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_name = args.served_model_name
        self.allowed_roots = tuple(
            Path(item).expanduser().resolve()
            for item in args.allowed_local_media_roots.split(os.pathsep)
            if item
        )
        backend_args = SimpleNamespace(
            model_path=args.model_path,
            device=args.device,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        )
        self.backend: Any = (
            MageVLBackend(backend_args)
            if args.backend == "mage"
            else MossVLBackend(backend_args)
        )
        self.sessions: dict[str, SessionState] = {}
        self.inference_lock = asyncio.Lock()
        self.native_session = None
        self.native_session_id: Optional[str] = None
        self.mage_streaming_engine = (
            MageStreamingEngine(self.backend, args)
            if args.backend == "mage"
            and args.mage_streaming
            and hasattr(self.backend.model, "streammind_gate_forward_segments")
            else None
        )

    @property
    def native_realtime(self) -> bool:
        return (
            self.args.backend == "moss"
            and self.args.native_realtime
            and hasattr(self.backend.model, "create_realtime_session")
        )

    @property
    def mage_streaming(self) -> bool:
        return self.mage_streaming_engine is not None

    def _completion(
        self,
        text: str,
        *,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        tokens = len(self.backend.processor.tokenizer.encode(text, add_special_tokens=False))
        payload: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": tokens,
                "total_tokens": tokens,
            },
        }
        if session_id:
            payload["streamingharness"] = {
                "session_id": session_id,
                "native_realtime": self.native_realtime,
                "frame_count": len(self.sessions.get(session_id, SessionState()).frames),
                **(metadata or {}),
            }
        return payload

    def _run_plain(
        self,
        prompt: str,
        images: list[Image.Image],
        video_path: Optional[str],
        body: dict[str, Any],
    ) -> str:
        max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 128)))
        temperature = float(body.get("temperature", 0.0))
        top_p = float(body.get("top_p", 0.9))
        if self.args.backend == "mage":
            return self.backend.generate_media(
                prompt,
                video_path=video_path,
                images=images,
                max_new_tokens=max_tokens,
                num_frames=self.args.num_frames,
                max_pixels=self.args.max_pixels,
                codec=bool(video_path and self.args.mage_video_backend == "codec"),
                temperature=temperature,
                top_p=top_p,
            )
        content: list[dict[str, Any]] = []
        if video_path:
            content.append({"type": "video", "video": video_path})
        content.extend({"type": "image", "image": image} for image in images)
        content.append({"type": "text", "text": prompt})
        return self.backend.generate_query({
            "messages": [{"role": "user", "content": content}],
            "media_kwargs": {
                "video_fps": self.args.fps,
                "min_frames": 1,
                "max_frames": self.args.num_frames,
            },
            "generate_kwargs": {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else 1.0,
                "top_p": top_p,
                "repetition_penalty": 1.0,
                "vision_chunked_length": 64,
            },
        })

    def _native_step(
        self,
        session_id: str,
        state: SessionState,
        prompt: str,
        images: list[Image.Image],
        timestamps: list[float],
        body: dict[str, Any],
        prompt_changed: bool,
    ) -> tuple[str, dict[str, Any]]:
        new_session = self.native_session is None
        if new_session:
            if self.native_session_id not in (None, session_id):
                raise SessionCapacityError(
                    f"MOSS-Realtime worker is occupied by session {self.native_session_id!r}"
                )
            self.native_session = self.backend.model.create_realtime_session(
                self.backend.processor,
                initial_prompt=prompt,
                system_prompt=latest_role_text(body.get("messages"), "system") or None,
                frame_queue_size=self.args.realtime_frame_queue_size,
                max_tokens_per_turn=self.args.max_tokens_per_second,
                max_new_tokens=int(
                    body.get("realtime_max_new_tokens", self.args.realtime_max_new_tokens)
                ),
                do_sample=False,
                repetition_penalty=1.0,
            )
            self.native_session_id = session_id
        elif self.native_session_id != session_id:
            raise SessionCapacityError(
                f"MOSS-Realtime worker is occupied by session {self.native_session_id!r}"
            )

        if new_session:
            self.native_session.start()
            for image, timestamp in zip(images, timestamps):
                self.native_session.push_frame(image, timestamp=timestamp)
        elif prompt_changed and images:
            for image, timestamp in zip(images[:-1], timestamps[:-1]):
                self.native_session.push_frame(image, timestamp=timestamp)
            self.native_session.push_prompt_frame(
                prompt, images[-1], timestamp=timestamps[-1]
            )
        elif prompt_changed:
            self.native_session.push_prompt(prompt)
        else:
            for image, timestamp in zip(images, timestamps):
                self.native_session.push_frame(image, timestamp=timestamp)

        deadline = time.monotonic() + self.args.response_wait_seconds
        while time.monotonic() < deadline:
            chunk = self.native_session.poll_output(timeout=0.05)
            if chunk is not None:
                state.moss_output.feed(chunk)
        return state.moss_output.finish_poll_window(), {
            "runtime": "moss-native",
            "pending_frames": int(getattr(self.native_session, "pending_frames", 0)),
            "max_tokens_per_second": self.args.max_tokens_per_second,
        }

    def _close_native(self) -> None:
        if self.native_session is not None:
            self.native_session.close(timeout=30.0)
        self.native_session = None
        self.native_session_id = None

    def _expire_sessions(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, state in self.sessions.items()
            if now - state.updated_at > self.args.session_timeout_seconds
        ]
        for session_id in expired:
            self.sessions.pop(session_id, None)
            if self.native_session_id == session_id:
                self._close_native()

    async def chat(self, body: dict[str, Any], session_id: Optional[str]) -> dict[str, Any]:
        messages = body.get("messages")
        prompt, images, video_path = parse_messages(
            messages, self.allowed_roots, allow_empty=bool(session_id)
        )
        async with self.inference_lock:
            if not session_id:
                text = await asyncio.to_thread(
                    self._run_plain, prompt, images, video_path, body
                )
                return self._completion(text)

            self._expire_sessions()
            state = self.sessions.setdefault(session_id, SessionState())
            request_system_prompt = latest_role_text(messages, "system")
            if request_system_prompt:
                state.system_prompt = request_system_prompt
            session_prompt = latest_role_text(messages, "user")
            prompt_changed = bool(session_prompt and session_prompt != state.last_prompt)
            windows = resolve_frame_windows(
                body,
                len(images),
                default_index=len(state.timestamps),
                frame_seconds=1.0 / self.args.fps,
            )
            for image, window in zip(images, windows):
                state.frames.append(image)
                state.timestamps.append(window.start)
            if len(state.frames) > self.args.session_max_frames:
                state.frames = state.frames[-self.args.session_max_frames:]
                state.timestamps = state.timestamps[-self.args.session_max_frames:]
            state.updated_at = time.time()

            if self.native_realtime and not video_path:
                text, metadata = await asyncio.to_thread(
                    self._native_step,
                    session_id,
                    state,
                    session_prompt,
                    images,
                    [window.start for window in windows],
                    body,
                    prompt_changed,
                )
            elif self.mage_streaming:
                streaming_mode = str(body.get("streaming_mode", "proactive"))
                text, metadata = await asyncio.to_thread(
                    self.mage_streaming_engine.step,
                    state.mage,
                    system_prompt=state.system_prompt,
                    prompt=session_prompt or state.last_prompt,
                    images=images,
                    windows=windows,
                    video_path=video_path,
                    max_new_tokens=int(body.get("max_tokens", self.args.mage_max_new_tokens)),
                    end_of_stream=bool(body.get("end_of_stream", False)),
                    streaming_mode=streaming_mode,
                )
            else:
                timestamped_prompt = session_prompt or state.last_prompt
                if state.timestamps:
                    timestamped_prompt = (
                        f"Frames span {state.timestamps[0]:.1f}s to "
                        f"{state.timestamps[-1]:.1f}s.\n{timestamped_prompt}"
                    )
                raw = await asyncio.to_thread(
                    self._run_plain,
                    timestamped_prompt,
                    state.frames,
                    video_path,
                    body,
                )
                text = streaming_response(raw)
                metadata = {"runtime": "accumulated-frame-fallback"}
            if session_prompt:
                state.last_prompt = session_prompt
            return self._completion(text, session_id=session_id, metadata=metadata)

    async def reset(self, session_id: str) -> dict[str, Any]:
        async with self.inference_lock:
            existed = self.sessions.pop(session_id, None) is not None
            if self.native_session_id == session_id:
                await asyncio.to_thread(self._close_native)
        return {"ok": True, "session_id": session_id, "existed": existed}


def sse_response(payload: dict[str, Any]) -> StreamingResponse:
    choice = payload["choices"][0]
    chunk = {
        "id": payload["id"],
        "object": "chat.completion.chunk",
        "created": payload["created"],
        "model": payload["model"],
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": choice["message"]["content"]},
            "finish_reason": None,
        }],
    }
    end = dict(chunk)
    end["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]

    async def events():
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(end, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def create_app(runtime: RuntimeService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        try:
            yield
        finally:
            async with runtime.inference_lock:
                if runtime.native_session is not None:
                    await asyncio.to_thread(runtime._close_native)

    app = FastAPI(title="MOSS-Realtime and Mage API", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": runtime.model_name,
            "backend": runtime.args.backend,
            "sessions": len(runtime.sessions),
            "native_realtime": runtime.native_realtime,
            "mage_streaming": runtime.mage_streaming,
            "mage_gate_policy": (
                "joyai-proactive-or-interactive-force"
                if runtime.mage_streaming else None
            ),
            "mage_gate_threshold": (
                runtime.args.mage_gate_threshold if runtime.mage_streaming else None
            ),
            "device_map_strategy": runtime.args.device_map,
            "model_devices": runtime.backend.model_devices,
            "model_parallel": len(runtime.backend.model_devices) > 1,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "streaming_protocol": "joyai-http-session",
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{
                "id": runtime.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        x_streaming_session: Optional[str] = Header(default=None),
    ):
        try:
            body = await request.json()
            header_timestamp = (
                request.headers.get("x-frame-time-range")
                or request.headers.get("x-streaming-time-range")
            )
            if header_timestamp and "frame_time_range" not in body:
                body["frame_time_range"] = header_timestamp
            requested_model = body.get("model")
            if requested_model and requested_model != runtime.model_name:
                raise ValueError(
                    f"Unknown model {requested_model!r}; use {runtime.model_name!r}"
                )
            session_id = (
                x_streaming_session
                or request.headers.get("x-session-id")
                or body.get("user")
            )
            streaming_control = body.get("streaming_control")
            if streaming_control is not None:
                if not isinstance(streaming_control, dict):
                    raise ValueError("streaming_control must be an object")
                action = str(streaming_control.get("action") or "").strip().lower()
                if action != "reset":
                    raise ValueError(
                        f"Unsupported streaming_control action {action!r}; use 'reset'"
                    )
                control_session_id = (
                    session_id
                    or streaming_control.get("session_id")
                    or body.get("session_id")
                )
                if not control_session_id:
                    raise ValueError("A session id is required for streaming_control reset")
                reset_result = await runtime.reset(str(control_session_id))
                acknowledgement = {
                    "action": "reset",
                    "acknowledged": True,
                    **reset_result,
                }
                payload = runtime._completion(
                    "</silence>",
                    session_id=str(control_session_id),
                    metadata={
                        "runtime": "streaming-control",
                        "streaming_control": acknowledgement,
                    },
                )
                payload["streaming_control"] = acknowledgement
                return sse_response(payload) if body.get("stream") else JSONResponse(payload)
            payload = await runtime.chat(body, str(session_id) if session_id else None)
            return sse_response(payload) if body.get("stream") else JSONResponse(payload)
        except SessionCapacityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Chat completion failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/streaming/reset")
    async def reset(
        request: Request,
        x_streaming_session: Optional[str] = Header(default=None),
    ):
        body = await request.json()
        session_id = (
            x_streaming_session
            or request.headers.get("x-session-id")
            or body.get("user")
            or body.get("session_id")
        )
        if not session_id:
            raise HTTPException(status_code=400, detail="A session id is required")
        return await runtime.reset(str(session_id))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("moss", "mage"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        choices=("balanced", "auto", "balanced_low_0", "sequential", "single"),
        default="balanced",
        help="Accelerate model placement; balanced uses every visible GPU.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=150000)
    parser.add_argument("--mage-video-backend", choices=("codec", "frames"), default="codec")
    parser.add_argument("--session-max-frames", type=int, default=32)
    parser.add_argument("--response-wait-seconds", type=float, default=1.0)
    parser.add_argument("--max-tokens-per-second", type=int, default=12)
    parser.add_argument("--realtime-max-new-tokens", type=int, default=4096)
    parser.add_argument("--realtime-frame-queue-size", type=int, default=256)
    parser.add_argument("--native-realtime", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mage-streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mage-segment-seconds", type=float, default=8.0)
    parser.add_argument("--mage-segment-frames", type=int, default=16)
    parser.add_argument("--mage-cur-fps", type=float, default=2.0)
    parser.add_argument("--mage-gate-threshold", type=float, default=0.5)
    parser.add_argument("--mage-max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--mage-max-segments",
        type=int,
        default=0,
        help="Gate history cap; zero preserves the full official causal history.",
    )
    parser.add_argument("--session-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--allowed-local-media-roots",
        default=os.environ.get("ALLOWED_LOCAL_MEDIA_ROOTS", "/tmp"),
    )
    args = parser.parse_args()
    positive = {
        "fps": args.fps,
        "response_wait_seconds": args.response_wait_seconds,
        "max_tokens_per_second": args.max_tokens_per_second,
        "realtime_max_new_tokens": args.realtime_max_new_tokens,
        "realtime_frame_queue_size": args.realtime_frame_queue_size,
        "mage_segment_seconds": args.mage_segment_seconds,
        "mage_segment_frames": args.mage_segment_frames,
        "mage_cur_fps": args.mage_cur_fps,
        "mage_max_new_tokens": args.mage_max_new_tokens,
        "session_timeout_seconds": args.session_timeout_seconds,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error(f"These values must be positive: {', '.join(invalid)}")
    if not 0 <= args.mage_gate_threshold <= 1:
        parser.error("--mage-gate-threshold must be between 0 and 1")
    return args


def main() -> None:
    import uvicorn

    args = parse_args()
    runtime = RuntimeService(args)
    uvicorn.run(create_app(runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
