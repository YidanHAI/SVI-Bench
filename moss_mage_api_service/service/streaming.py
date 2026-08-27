"""Stateful JoyAI-compatible streaming helpers."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image, ImageOps


SILENCE = "</silence>"
RESPONSE = "</response>"
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class FrameWindow:
    start: float
    end: float
    label: str


def parse_frame_window(value: Any, frame_seconds: float = 1.0) -> Optional[FrameWindow]:
    """Parse JoyAI's ``N seconds`` and ``N-M seconds`` timestamp formats."""
    text = str(value or "").strip()
    numbers = [float(item) for item in _NUMBER_RE.findall(text)]
    if not numbers:
        return None
    start = numbers[0]
    end = numbers[1] if len(numbers) > 1 else start + frame_seconds
    if end < start:
        raise ValueError(f"Frame time range must be non-decreasing: {text!r}")
    return FrameWindow(start=start, end=end, label=text)


def resolve_frame_windows(
    payload: dict[str, Any],
    image_count: int,
    *,
    default_index: int,
    frame_seconds: float,
) -> list[FrameWindow]:
    """Resolve JoyAI timestamp fields and the legacy streaming timestamp."""
    if image_count <= 0:
        return []
    raw_ranges = payload.get("frame_time_ranges")
    candidates = list(raw_ranges) if isinstance(raw_ranges, list) else []
    if not candidates:
        single = (
            payload.get("frame_time_range")
            or payload.get("x_frame_time_range")
            or payload.get("streaming_timestamp")
        )
        if single is not None:
            candidates = [single]

    windows: list[FrameWindow] = []
    for index in range(image_count):
        candidate = candidates[index] if index < len(candidates) else None
        if candidate is None and len(candidates) == 1:
            base = parse_frame_window(candidates[0], frame_seconds)
            if base is not None:
                start = base.start + index * frame_seconds
                windows.append(
                    FrameWindow(start, start + frame_seconds, f"{start:g} seconds")
                )
                continue
        parsed = parse_frame_window(candidate, frame_seconds)
        if parsed is None:
            start = (default_index + index) * frame_seconds
            parsed = FrameWindow(start, start + frame_seconds, f"{start:g} seconds")
        windows.append(parsed)
    return windows


def streaming_response(text: str) -> str:
    text = " ".join(str(text or "").split()).strip()
    if not text or text == SILENCE:
        return SILENCE
    if text.startswith(RESPONSE):
        return text
    return f"{RESPONSE} {text}"


@dataclass
class RealtimeOutputBuffer:
    """Turn MOSS control-token chunks into complete JoyAI response events."""

    pending: str = ""
    completed: list[str] = field(default_factory=list)

    @staticmethod
    def _clean_response(value: str) -> str:
        for token in ("<|round_start|>", "<|round_end|>", "<|response|>"):
            value = value.replace(token, "")
        value = value.replace("<|silence|>", "")
        return " ".join(value.split()).strip()

    def feed(self, chunk: str) -> None:
        self.pending += str(chunk or "")
        while self.pending:
            round_start = self.pending.find("<|round_start|>")
            silence = self.pending.find("<|silence|>")
            if round_start >= 0 and (silence < 0 or round_start <= silence):
                prefix = self.pending[:round_start]
                cleaned_prefix = self._clean_response(prefix)
                if cleaned_prefix:
                    self.completed.append(streaming_response(cleaned_prefix))
                round_end = self.pending.find("<|round_end|>", round_start)
                if round_end < 0:
                    silence_in_round = self.pending.find("<|silence|>", round_start)
                    if silence_in_round >= 0:
                        raw_round = self.pending[round_start:silence_in_round]
                        cleaned = self._clean_response(raw_round)
                        self.completed.append(
                            streaming_response(cleaned) if cleaned else SILENCE
                        )
                        self.pending = self.pending[
                            silence_in_round + len("<|silence|>"):
                        ]
                        continue
                    return
                end = round_end + len("<|round_end|>")
                raw_round = self.pending[round_start:end]
                self.pending = self.pending[end:]
                self.completed.append(streaming_response(self._clean_response(raw_round)))
                continue
            if silence >= 0:
                prefix = self.pending[:silence]
                cleaned_prefix = self._clean_response(prefix)
                if cleaned_prefix:
                    self.completed.append(streaming_response(cleaned_prefix))
                self.completed.append(SILENCE)
                self.pending = self.pending[silence + len("<|silence|>"):]
                continue
            return

    def finish_poll_window(self) -> str:
        # Keep an unfinished structured round for the next HTTP heartbeat.
        if "<|round_start|>" not in self.pending:
            cleaned = self._clean_response(self.pending)
            if cleaned:
                self.completed.append(streaming_response(cleaned))
            self.pending = ""
        events = self.completed
        self.completed = []
        responses = [
            event[len(RESPONSE):].strip()
            for event in events
            if event.startswith(RESPONSE)
        ]
        return (
            streaming_response(" ".join(item for item in responses if item))
            if responses
            else SILENCE
        )


@dataclass
class TimedFrame:
    image: Image.Image
    window: FrameWindow


@dataclass
class MageStreamState:
    pending_frames: list[TimedFrame] = field(default_factory=list)
    gate_segments: list[dict[str, Any]] = field(default_factory=list)
    next_segment_start: Optional[float] = None
    segments_processed: int = 0
    last_gate_probability: Optional[float] = None


class MageStreamingEngine:
    """HTTP session wrapper for Mage's official StreamMind segment protocol."""

    DEFAULT_PROMPT = (
        "Please describe the video content in detail based on the provided information."
    )

    def __init__(self, backend: Any, args: Any):
        self.backend = backend
        self.args = args
        self._last_frame_normalization: dict[str, Any] = {}

    @staticmethod
    def _uniform_sample(frames: list[Image.Image], count: int) -> list[Image.Image]:
        if not frames or count <= 0 or len(frames) == count:
            return frames
        if count == 1:
            return [frames[0]]
        indices = [
            round(index * (len(frames) - 1) / (count - 1))
            for index in range(count)
        ]
        return [frames[index] for index in indices]

    @staticmethod
    def _normalize_frame_sizes(
        frames: list[Image.Image],
    ) -> tuple[list[Image.Image], dict[str, Any]]:
        if not frames:
            return [], {
                "target_size": None,
                "source_sizes": [],
                "resized_frame_count": 0,
            }

        rgb_frames = [frame.convert("RGB") if frame.mode != "RGB" else frame for frame in frames]
        sizes = [frame.size for frame in rgb_frames]
        counts = Counter(sizes)
        last_positions = {size: index for index, size in enumerate(sizes)}
        target_size = max(counts, key=lambda size: (counts[size], last_positions[size]))
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        normalized: list[Image.Image] = []
        resized_count = 0
        for frame in rgb_frames:
            if frame.size == target_size:
                normalized.append(frame)
                continue
            contained = ImageOps.contain(frame, target_size, method=resampling)
            canvas = Image.new("RGB", target_size, "black")
            offset = (
                (target_size[0] - contained.width) // 2,
                (target_size[1] - contained.height) // 2,
            )
            canvas.paste(contained, offset)
            normalized.append(canvas)
            resized_count += 1
        return normalized, {
            "target_size": list(target_size),
            "source_sizes": [
                {"size": list(size), "count": count}
                for size, count in sorted(counts.items())
            ],
            "resized_frame_count": resized_count,
        }

    def _to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                result[key] = value
            elif key == "pixel_values":
                result[key] = value.to(
                    device=self.backend.model.device, dtype=self.backend.model.dtype
                )
            else:
                result[key] = value.to(self.backend.model.device)
        return result

    def _build_inputs(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        frames: Optional[list[Image.Image]] = None,
        video_path: Optional[str] = None,
    ) -> dict[str, Any]:
        prompt = prompt or self.DEFAULT_PROMPT
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        })
        text = self.backend.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if video_path:
            self._last_frame_normalization = {
                "target_size": None,
                "source_sizes": [],
                "resized_frame_count": 0,
                "input_kind": "video_path",
            }
            inputs = self.backend.processor(
                text=[text],
                videos=[video_path],
                video_backend="codec",
                codec_config={"patch": 16, "max_pixels": self.args.max_pixels},
                max_pixels=self.args.max_pixels,
                return_tensors="pt",
                padding=False,
            )
        else:
            normalized_frames, normalization = self._normalize_frame_sizes(
                list(frames or [])
            )
            self._last_frame_normalization = {
                **normalization,
                "input_kind": "causal_frames",
            }
            sampled = self._uniform_sample(
                normalized_frames, self.args.mage_segment_frames
            )
            if not sampled:
                raise ValueError("Mage streaming segment contains no frames")
            inputs = self.backend.processor(
                text=[text],
                videos=[sampled],
                video_backend="frames",
                num_frames=self.args.mage_segment_frames,
                target_fps=self.args.mage_cur_fps,
                return_tensors="pt",
                padding=False,
            )
        return self._to_device(dict(inputs))

    def _run_segment(
        self,
        state: MageStreamState,
        inputs: dict[str, Any],
        *,
        max_new_tokens: int,
        force_response: bool,
    ) -> tuple[Optional[str], float]:
        import torch

        visual = {
            key: inputs[key]
            for key in ("pixel_values", "image_grid_thw", "patch_positions")
            if key in inputs
        }
        if "pixel_values" not in visual or "image_grid_thw" not in visual:
            raise RuntimeError("Mage processor did not return gate-compatible visual tensors")
        state.gate_segments.append(visual)
        if self.args.mage_max_segments > 0:
            state.gate_segments = state.gate_segments[-self.args.mage_max_segments:]
        with torch.inference_mode():
            logits = self.backend.model.streammind_gate_forward_segments(
                state.gate_segments
            )[0]
        lengths = [
            int(segment["image_grid_thw"][:, 0].sum().item())
            for segment in state.gate_segments
        ]
        boundary = int(torch.tensor(lengths).cumsum(0)[-1].item() - 1)
        probability = float(torch.softmax(logits[boundary].float(), dim=-1)[1].item())
        state.last_gate_probability = probability
        state.segments_processed += 1
        if not force_response and probability < self.args.mage_gate_threshold:
            return None, probability
        with torch.inference_mode():
            output = self.backend.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        return self.backend._decode(output, inputs["input_ids"].shape[1]), probability

    def step(
        self,
        state: MageStreamState,
        *,
        system_prompt: str,
        prompt: str,
        images: list[Image.Image],
        windows: list[FrameWindow],
        video_path: Optional[str],
        max_new_tokens: int,
        end_of_stream: bool,
        streaming_mode: str,
    ) -> tuple[str, dict[str, Any]]:
        if streaming_mode not in ("proactive", "interactive"):
            raise ValueError(f"Unsupported Mage streaming mode: {streaming_mode!r}")
        force_response = streaming_mode == "interactive"
        responses: list[str] = []
        probabilities: list[float] = []
        frame_normalizations: list[dict[str, Any]] = []
        if video_path:
            response, probability = self._run_segment(
                state,
                self._build_inputs(
                    prompt,
                    system_prompt=system_prompt,
                    video_path=video_path,
                ),
                max_new_tokens=max_new_tokens,
                force_response=force_response,
            )
            probabilities.append(probability)
            frame_normalizations.append(dict(self._last_frame_normalization))
            if response:
                responses.append(response)

        for image, window in zip(images, windows):
            state.pending_frames.append(TimedFrame(image=image, window=window))
        if state.pending_frames and state.next_segment_start is None:
            first = state.pending_frames[0].window.start
            state.next_segment_start = (
                math.floor(first / self.args.mage_segment_seconds)
                * self.args.mage_segment_seconds
            )

        while state.pending_frames and state.next_segment_start is not None:
            segment_start = state.next_segment_start
            segment_end = segment_start + self.args.mage_segment_seconds
            latest_end = max(item.window.end for item in state.pending_frames)
            if latest_end < segment_end and not end_of_stream:
                break
            selected = [
                item
                for item in state.pending_frames
                if item.window.start < segment_end and item.window.end > segment_start
            ]
            if not selected:
                state.next_segment_start = segment_end
                if end_of_stream:
                    break
                continue
            response, probability = self._run_segment(
                state,
                self._build_inputs(
                    prompt,
                    system_prompt=system_prompt,
                    frames=[item.image for item in selected],
                ),
                max_new_tokens=max_new_tokens,
                force_response=force_response,
            )
            probabilities.append(probability)
            frame_normalizations.append(dict(self._last_frame_normalization))
            if response:
                responses.append(response)
            state.pending_frames = [
                item
                for item in state.pending_frames
                if item.window.start >= segment_end
            ]
            state.next_segment_start = segment_end
            if end_of_stream:
                break

        content = streaming_response(" ".join(responses)) if responses else SILENCE
        return content, {
            "runtime": "mage-streammind",
            "segments_processed": state.segments_processed,
            "pending_frames": len(state.pending_frames),
            "gate_probability": state.last_gate_probability,
            "gate_probabilities": probabilities,
            "gate_policy": (
                "joyai-interactive-force" if force_response else "official-threshold"
            ),
            "gate_threshold": self.args.mage_gate_threshold,
            "streaming_mode": streaming_mode,
            "interactive_force_response": force_response,
            "system_prompt_applied": bool(system_prompt),
            "frame_normalizations": frame_normalizations,
            "normalized_frame_count": sum(
                int(item.get("resized_frame_count", 0))
                for item in frame_normalizations
            ),
        }
