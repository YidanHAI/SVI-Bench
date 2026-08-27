from __future__ import annotations

import asyncio
import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from fastapi.testclient import TestClient
from PIL import Image

from service.backends import (
    model_cuda_devices,
    resolve_device_map,
    validate_balanced_placement,
)
from service.server import (
    RuntimeService,
    SessionState,
    create_app,
    decode_image_url,
    parse_messages,
    resolve_video_url,
)
from service.streaming import (
    FrameWindow,
    MageStreamState,
    MageStreamingEngine,
    RealtimeOutputBuffer,
    resolve_frame_windows,
)


def image_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class DevicePlacementTest(unittest.TestCase):
    def test_balanced_and_single_device_maps(self):
        self.assertEqual(resolve_device_map("balanced", "cuda:0"), "balanced")
        self.assertEqual(resolve_device_map("single", "cuda:3"), {"": "cuda:3"})

    def test_cuda_devices_are_reported_and_balanced_is_validated(self):
        model = SimpleNamespace(hf_device_map={"visual": 0, "layer.0": "cuda:1"})
        self.assertEqual(model_cuda_devices(model), ["cuda:0", "cuda:1"])
        with patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(
                validate_balanced_placement(model, "balanced"),
                ["cuda:0", "cuda:1"],
            )

    def test_balanced_rejects_partial_visible_gpu_use(self):
        model = SimpleNamespace(hf_device_map={"": 0})
        with patch("torch.cuda.device_count", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "did not use every visible GPU"):
                validate_balanced_placement(model, "balanced")


class MediaParsingTest(unittest.TestCase):
    def test_data_url_and_text(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url()}},
                {"type": "text", "text": "What is shown?"},
            ],
        }]
        prompt, images, video = parse_messages(messages, (Path("/tmp"),))
        self.assertEqual(prompt, "user: What is shown?")
        self.assertEqual(len(images), 1)
        self.assertIsNone(video)

    def test_local_media_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image_path = root / "frame.png"
            Image.new("RGB", (2, 2)).save(image_path)
            self.assertEqual(decode_image_url(str(image_path), (root,)).size, (2, 2))
            with self.assertRaisesRegex(ValueError, "outside"):
                decode_image_url(str(image_path), (Path("/other"),))

    def test_video_must_exist(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            resolve_video_url("/tmp/not-a-video.mp4", (Path("/tmp"),))


class StreamingProtocolTest(unittest.TestCase):
    def test_time_fields(self):
        windows = resolve_frame_windows(
            {"frame_time_ranges": ["0-1 seconds", "1 seconds"]},
            2,
            default_index=0,
            frame_seconds=1.0,
        )
        self.assertEqual([(w.start, w.end) for w in windows], [(0.0, 1.0), (1.0, 2.0)])

    def test_moss_round_across_requests(self):
        output = RealtimeOutputBuffer()
        output.feed("<|round_start|><|response|>hello")
        self.assertEqual(output.finish_poll_window(), "</silence>")
        output.feed(" world<|round_end|>")
        self.assertEqual(output.finish_poll_window(), "</response> hello world")

    def test_moss_native_prompt_deduplication(self):
        session = MagicMock()
        session.poll_output.return_value = None
        runtime = RuntimeService.__new__(RuntimeService)
        runtime.backend = SimpleNamespace(model=MagicMock(), processor=object())
        runtime.args = SimpleNamespace(
            max_tokens_per_second=12,
            response_wait_seconds=0.0,
            realtime_frame_queue_size=256,
            realtime_max_new_tokens=4096,
        )
        runtime.native_session = session
        runtime.native_session_id = "session-1"
        runtime._native_step(
            "session-1",
            SessionState(last_prompt="Watch"),
            "Watch",
            [Image.new("RGB", (2, 2), "red")],
            [1.0],
            {"messages": []},
            False,
        )
        session.push_frame.assert_called_once()
        session.push_prompt.assert_not_called()


class APIContractTest(unittest.TestCase):
    def test_joyai_headers(self):
        class FakeRuntime:
            model_name = "fake-model"
            sessions = {}
            native_session = None
            inference_lock = asyncio.Lock()
            args = SimpleNamespace(backend="fake")
            native_realtime = False
            mage_streaming = False

            def __init__(self):
                self.request = None
                self.reset_session_id = None

            async def chat(self, body, session_id):
                self.request = (body, session_id)
                return {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "</silence>"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }

            async def reset(self, session_id):
                self.reset_session_id = session_id
                return {"ok": True, "session_id": session_id}

            def _completion(self, text, *, session_id=None, metadata=None):
                return {
                    "id": "chatcmpl-control-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "streamingharness": {
                        "session_id": session_id,
                        **(metadata or {}),
                    },
                }

        runtime = FakeRuntime()
        with TestClient(create_app(runtime)) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"x-session-id": "s1", "x-frame-time-range": "2-3 seconds"},
                json={"model": "fake-model", "messages": []},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime.request[1], "s1")
        self.assertEqual(runtime.request[0]["frame_time_range"], "2-3 seconds")

    def test_in_band_reset_over_chat_completions(self):
        class FakeRuntime:
            model_name = "fake-model"
            sessions = {}
            native_session = None
            inference_lock = asyncio.Lock()
            args = SimpleNamespace(backend="fake")
            native_realtime = False
            mage_streaming = False

            def __init__(self):
                self.reset_session_id = None

            async def chat(self, body, session_id):
                raise AssertionError("reset control must not run inference")

            async def reset(self, session_id):
                self.reset_session_id = session_id
                return {"ok": True, "session_id": session_id, "existed": True}

            def _completion(self, text, *, session_id=None, metadata=None):
                return {
                    "id": "chatcmpl-control-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "streamingharness": {
                        "session_id": session_id,
                        **(metadata or {}),
                    },
                }

        runtime = FakeRuntime()
        with TestClient(create_app(runtime)) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"x-streaming-session": "s-reset"},
                json={
                    "model": "fake-model",
                    "messages": [],
                    "streaming_control": {"action": "reset"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime.reset_session_id, "s-reset")
        payload = response.json()
        self.assertEqual(payload["choices"][0]["message"]["content"], "</silence>")
        self.assertEqual(payload["streaming_control"]["action"], "reset")
        self.assertTrue(payload["streaming_control"]["acknowledged"])
        self.assertTrue(
            payload["streamingharness"]["streaming_control"]["acknowledged"]
        )

    def test_in_band_reset_rejects_unknown_action(self):
        runtime = MagicMock()
        runtime.model_name = "fake-model"
        runtime.sessions = {}
        runtime.native_session = None
        runtime.inference_lock = asyncio.Lock()
        runtime.args = SimpleNamespace(backend="fake")
        runtime.native_realtime = False
        runtime.mage_streaming = False
        with TestClient(create_app(runtime)) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"x-streaming-session": "s-reset"},
                json={
                    "model": "fake-model",
                    "messages": [],
                    "streaming_control": {"action": "close"},
                },
            )
        self.assertEqual(response.status_code, 400)


class MageStreamingTest(unittest.TestCase):
    @staticmethod
    def _engine_args():
        return SimpleNamespace(
            max_pixels=150000,
            mage_segment_frames=1,
            mage_cur_fps=1.0,
            mage_max_segments=0,
            mage_gate_threshold=0.5,
            mage_segment_seconds=1.0,
        )

    def test_gate_and_generation(self):
        processor = MagicMock()
        processor.apply_chat_template.return_value = "prompt"
        processor.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros(4, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "patch_positions": torch.zeros(4, 3, dtype=torch.long),
        }
        model = MagicMock(device=torch.device("cpu"), dtype=torch.float32)
        model.streammind_gate_forward_segments.return_value = torch.tensor([[[0.0, 1.0]]])
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        backend = SimpleNamespace(
            processor=processor,
            model=model,
            _decode=MagicMock(return_value="A person entered."),
        )
        args = SimpleNamespace(
            max_pixels=150000,
            mage_segment_frames=1,
            mage_cur_fps=1.0,
            mage_max_segments=0,
            mage_gate_threshold=0.5,
            mage_segment_seconds=1.0,
        )
        content, metadata = MageStreamingEngine(backend, args).step(
            MageStreamState(),
            system_prompt="Remain silent until the requested event occurs.",
            prompt="Describe changes.",
            images=[Image.new("RGB", (2, 2), "red")],
            windows=[FrameWindow(0.0, 1.0, "0-1 seconds")],
            video_path=None,
            max_new_tokens=8,
            end_of_stream=False,
            streaming_mode="proactive",
        )
        self.assertEqual(content, "</response> A person entered.")
        self.assertEqual(metadata["segments_processed"], 1)
        model.streammind_gate_forward_segments.assert_called_once()
        model.generate.assert_called_once()
        messages = processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0], {
            "role": "system",
            "content": "Remain silent until the requested event occurs.",
        })
        self.assertEqual(messages[1]["role"], "user")
        self.assertTrue(metadata["system_prompt_applied"])

    def test_interactive_mode_forces_generation_below_gate(self):
        processor = MagicMock()
        processor.apply_chat_template.return_value = "prompt"
        processor.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros(4, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "patch_positions": torch.zeros(4, 3, dtype=torch.long),
        }
        model = MagicMock(device=torch.device("cpu"), dtype=torch.float32)
        model.streammind_gate_forward_segments.return_value = torch.tensor([[[1.0, 0.0]]])
        backend = SimpleNamespace(
            processor=processor,
            model=model,
            _decode=MagicMock(return_value="Fire detected."),
        )

        content, metadata = MageStreamingEngine(backend, self._engine_args()).step(
            MageStreamState(),
            system_prompt="Remain silent until the requested event occurs.",
            prompt="Alert on fire.",
            images=[Image.new("RGB", (2, 2), "red")],
            windows=[FrameWindow(0.0, 1.0, "0-1 seconds")],
            video_path=None,
            max_new_tokens=8,
            end_of_stream=False,
            streaming_mode="interactive",
        )

        self.assertEqual(content, "</response> Fire detected.")
        model.generate.assert_called_once()
        self.assertEqual(metadata["gate_policy"], "joyai-interactive-force")
        self.assertEqual(metadata["streaming_mode"], "interactive")
        self.assertTrue(metadata["interactive_force_response"])

    def test_mixed_frame_sizes_are_normalized_before_video_processing(self):
        processor = MagicMock()
        processor.apply_chat_template.return_value = "prompt"
        processor.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros(4, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "patch_positions": torch.zeros(4, 3, dtype=torch.long),
        }
        model = MagicMock(device=torch.device("cpu"), dtype=torch.float32)
        model.streammind_gate_forward_segments.return_value = torch.tensor([[[0.0, 1.0]]])
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        backend = SimpleNamespace(
            processor=processor,
            model=model,
            _decode=MagicMock(return_value="normalized"),
        )
        args = self._engine_args()
        args.mage_segment_frames = 4
        args.mage_segment_seconds = 2.0

        content, metadata = MageStreamingEngine(backend, args).step(
            MageStreamState(),
            system_prompt="Watch causally.",
            prompt="Alert on fire.",
            images=[
                Image.new("RGB", (8, 4), "red"),
                Image.new("RGB", (4, 2), "blue"),
            ],
            windows=[
                FrameWindow(0.0, 1.0, "0 seconds"),
                FrameWindow(1.0, 2.0, "1 second"),
            ],
            video_path=None,
            max_new_tokens=8,
            end_of_stream=False,
            streaming_mode="proactive",
        )

        self.assertEqual(content, "</response> normalized")
        submitted_frames = processor.call_args.kwargs["videos"][0]
        self.assertEqual(len(submitted_frames), 4)
        self.assertEqual({frame.size for frame in submitted_frames}, {(4, 2)})
        self.assertEqual(metadata["normalized_frame_count"], 1)
        self.assertEqual(
            metadata["frame_normalizations"][0]["target_size"], [4, 2]
        )

    def test_sparse_frames_keep_media_time_gaps(self):
        processor = MagicMock()
        processor.apply_chat_template.return_value = "prompt"
        processor.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros(4, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "patch_positions": torch.zeros(4, 3, dtype=torch.long),
        }
        model = MagicMock(device=torch.device("cpu"), dtype=torch.float32)
        model.streammind_gate_forward_segments.side_effect = lambda segments: torch.tensor([
            [[1.0, 0.0] for _ in segments]
        ])
        backend = SimpleNamespace(processor=processor, model=model, _decode=MagicMock())
        args = self._engine_args()
        args.mage_segment_seconds = 8.0
        engine = MageStreamingEngine(backend, args)
        state = MageStreamState()

        _, metadata = engine.step(
            state,
            system_prompt="Watch causally.",
            prompt="Alert on fire.",
            images=[
                Image.new("RGB", (2, 2), "red"),
                Image.new("RGB", (2, 2), "blue"),
            ],
            windows=[
                FrameWindow(0.0, 1.0, "0 seconds"),
                FrameWindow(9.0, 10.0, "9 seconds"),
            ],
            video_path=None,
            max_new_tokens=8,
            end_of_stream=False,
            streaming_mode="proactive",
        )
        self.assertEqual(metadata["segments_processed"], 1)
        self.assertEqual(metadata["pending_frames"], 1)

        _, metadata = engine.step(
            state,
            system_prompt="Watch causally.",
            prompt="Alert on fire.",
            images=[Image.new("RGB", (2, 2), "green")],
            windows=[FrameWindow(16.0, 17.0, "16 seconds")],
            video_path=None,
            max_new_tokens=8,
            end_of_stream=False,
            streaming_mode="proactive",
        )
        self.assertEqual(metadata["segments_processed"], 2)
        self.assertEqual(metadata["pending_frames"], 1)


if __name__ == "__main__":
    unittest.main()
