import hashlib
import json
import os
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

JPEG_32X24 = (
    "data:image/jpeg;base64,"
    "/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzU4LjU0LjEwMAD/2wBDAAgEBAQEBAUFBQUFBQYGBgYGBgYGBgYGBgYHBwcICAgHBwcGBgcHCAgICAkJCQgICAgJCQoKCgwMCwsODg4RERT/xABLAAEBAAAAAAAAAAAAAAAAAAAACAEBAAAAAAAAAAAAAAAAAAAAABABAAAAAAAAAAAAAAAAAAAAABEBAAAAAAAAAAAAAAAAAAAAAP/AABEIABgAIAMBIgACEQADEQD/2gAMAwEAAhEDEQA/AJ/AAAAAAAB//9k="
)
JPEG_16X16 = (
    "data:image/jpeg;base64,"
    "/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzU4LjU0LjEwMAD/2wBDAAgEBAQEBAUFBQUFBQYGBgYGBgYGBgYGBgYHBwcICAgHBwcGBgcHCAgICAkJCQgICAgJCQoKCgwMCwsODg4RERT/xABLAAEBAAAAAAAAAAAAAAAAAAAABwEBAAAAAAAAAAAAAAAAAAAAABABAAAAAAAAAAAAAAAAAAAAABEBAAAAAAAAAAAAAAAAAAAAAP/AABEIABAAEAMBIgACEQADEQD/2gAMAwEAAhEDEQA/AL+AD//Z"
)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post_json(url, payload, token, extra_headers=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    headers.update(extra_headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_health(url, process):
    for _ in range(100):
        if process.poll() is not None:
            raise AssertionError(process.stderr.read())
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                payload = json.loads(response.read())
            if payload.get("ok"):
                return payload
        except Exception:
            time.sleep(0.05)
    raise AssertionError(f"service did not become healthy: {url}")


def terminate(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class MageSessionUpstream(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, MageSessionHandler)
        self.requests = []


class MageSessionHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def send_payload(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size) or b"{}")
        if self.path == "/v1/streaming/reset":
            session_id = str(payload.get("session_id", ""))
            self.send_payload(200, {"ok": True, "session_id": session_id})
            return
        if self.path != "/v1/chat/completions":
            self.send_payload(404, {"error": {"message": "not found"}})
            return

        self.server.requests.append(payload)
        index = len(self.server.requests) - 1
        if index == 3:
            self.send_payload(
                500,
                {
                    "error": {
                        "message": (
                            "Frames yielded inconsistent grid_thw during smart_resize"
                        )
                    }
                },
            )
            return
        if index == 4:
            time.sleep(0.25)

        self.send_payload(
            200,
            {
                "id": f"mage-response-{index}",
                "model": "Mage-VL",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "</silence>"},
                        "finish_reason": "stop",
                    }
                ],
                "streamingharness": {
                    "runtime": "mage-streammind",
                    "gate_probabilities": [] if index == 0 else [0.01],
                },
            },
        )


class MonotonicClockUpstream(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, MonotonicClockHandler)
        self.requests = []
        self.delay_next_s = 0.0
        self.request_started = threading.Event()


class MonotonicClockHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def send_payload(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size) or b"{}")
        if self.path == "/v1/streaming/reset":
            session_id = str(payload.get("session_id", ""))
            self.send_payload(200, {"ok": True, "session_id": session_id})
            return
        if self.path != "/v1/chat/completions":
            self.send_payload(404, {"error": {"message": "not found"}})
            return

        self.server.requests.append(payload)
        index = len(self.server.requests) - 1
        self.server.request_started.set()
        delay_s = self.server.delay_next_s
        self.server.delay_next_s = 0.0
        if delay_s:
            time.sleep(delay_s)
        self.send_payload(
            200,
            {
                "id": f"moss-response-{index}",
                "model": "MOSS-VL-Realtime",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "</silence>"},
                        "finish_reason": "stop",
                    }
                ],
                "streamingharness": {
                    "runtime": "moss-native",
                    "gate_probabilities": [],
                },
            },
        )


def test_modelbest_native_realtime_forwards_video_and_query_once(tmp_path):
    upstream_port = free_port()
    proxy_port = free_port()
    upstream_events = tmp_path / "upstream.jsonl"
    proxy_audit = tmp_path / "proxy.jsonl"
    token = "native-realtime-test-token-123456789"

    upstream = subprocess.Popen(
        ["node", str(ROOT / "tests/fixtures/mock_modelbest_realtime_server.mjs")],
        cwd=ROOT,
        env={
            **os.environ,
            "MOCK_REALTIME_PORT": str(upstream_port),
            "MOCK_REALTIME_EVENTS": str(upstream_events),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert upstream.stdout.readline().strip() == "ready"

    proxy = subprocess.Popen(
        ["node", str(ROOT / "scripts/native_video_realtime_proxy.mjs")],
        cwd=ROOT,
        env={
            **os.environ,
            "UPSTREAM_API_KEY": "mock-upstream-key",
            "PROMPT_PROXY_ACCESS_TOKEN": token,
            "PROMPT_PROXY_ADVERTISED_MODEL": "MiniCPM-O-4.5-9B",
            "PROMPT_PROXY_PORT": str(proxy_port),
            "PROMPT_PROXY_AUDIT_PATH": str(proxy_audit),
            "NATIVE_REALTIME_API_BASE": f"ws://127.0.0.1:{upstream_port}/v1/realtime?mode=video",
            "NATIVE_REALTIME_PROTOCOL": "modelbest-video-full-duplex-v1",
            "NATIVE_REALTIME_QUERY_MODE": "session-instruction",
            "NATIVE_VIDEO_SCHEMA": "modelbest-realtime.input.append.video_frames.jpeg",
            "NATIVE_REALTIME_MAX_SESSION_S": "300",
            "NATIVE_REALTIME_PROVIDER_RECONNECT_SETTLE_MS": "10",
            "NATIVE_REALTIME_OUTPUT_GRACE_MS": "100",
            "JOYAI_SYSTEM_PROMPT_FILE": str(ROOT / "config/joyai_system_prompt.txt"),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        health = wait_health(f"http://127.0.0.1:{proxy_port}/health", proxy)
        assert health["input_transport"] == "native-video-realtime"
        assert health["upstream_protocol"] == "openai-chat"
        assert health["realtime_protocol"] == "modelbest-video-full-duplex-v1"
        assert health["streaming_mode_policy"] == "proactive"
        assert health["system_prompt_transport"] == "system-role"
        assert health["frame_clock_policy"] == "inbound-turn"
        assert health["session_limit_policy"] == (
            "suspend_until_next_query_without_query_replay"
        )
        endpoint = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
        query_endpoint = f"http://127.0.0.1:{proxy_port}/v1/query-events"
        reset_endpoint = f"http://127.0.0.1:{proxy_port}/v1/streaming/reset"
        warmup_endpoint = f"http://127.0.0.1:{proxy_port}/v1/warmup"
        session_headers = {"X-Streaming-Session": "modelbest-session-1"}
        frame = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,/9j/2Q=="},
        }

        warmup = post_json(
            warmup_endpoint,
            {
                "model": "MiniCPM-O-4.5-9B",
                "image_url": "data:image/jpeg;base64,/9j/2Q==",
            },
            token,
        )
        assert warmup["ok"] is True
        assert warmup["multimodal_input"] is True
        assert warmup["session_state_unchanged"] is True
        assert warmup["response_model"] == "MiniCPM-O-4.5-9B"
        assert warmup["response_id"]

        post_json(
            endpoint,
            {"model": "MiniCPM-O-4.5-9B", "messages": [{"role": "user", "content": [frame]}]},
            token,
            session_headers,
        )
        query = "alert once"
        queued = post_json(
            query_endpoint,
            {
                "session_id": "modelbest-session-1",
                "query_event_id": "A1001:R1",
                "query": query,
                "image_url": "data:image/jpeg;base64,/9j/2Q==",
                "frame_time_range": "1.000 seconds",
                "ui_query_sent_at": "2026-08-21T00:00:00Z",
                "ui_query_video_time_s": 1.0,
                "annotated_query_video_time_s": 1.0,
                "captured_media_time_s": 1.0,
                "captured_raw_media_time_s": 1.0,
            },
            token,
        )
        assert queued["status"] == "queued"
        assert queued["idempotent"] is False
        prompted = post_json(
            endpoint,
            {
                "model": "MiniCPM-O-4.5-9B",
                "messages": [{"role": "user", "content": [frame]}],
            },
            token,
            session_headers,
        )
        assert prompted["streamingharness"]["query_frame_queue"]["query_event_id"] == (
            "A1001:R1"
        )
        duplicate = post_json(
            query_endpoint,
            {
                "session_id": "modelbest-session-1",
                "query_event_id": "A1001:R1",
                "query": query,
                "image_url": "data:image/jpeg;base64,/9j/2Q==",
                "frame_time_range": "1.000 seconds",
            },
            token,
        )
        assert duplicate["idempotent"] is True
        assert duplicate["status"] == "delivered"
        post_json(
            endpoint,
            {"model": "MiniCPM-O-4.5-9B", "messages": [{"role": "user", "content": [frame]}]},
            token,
            session_headers,
        )

        assert prompted["choices"][0]["message"]["content"] == (
            "</response> mock reply"
        )
        assert prompted["native_realtime"]["provider_raw_content"] == "mock reply"
        assert prompted["native_realtime"]["provider_output_format"] == (
            "native-text-listen-to-response-marker"
        )
        expected_hash = hashlib.sha256(query.encode()).hexdigest()
        assert (
            prompted["native_realtime"]["provider_output_query_sha256"]
            == expected_hash
        )
        time.sleep(0.1)
        provider_events = [
            json.loads(line) for line in upstream_events.read_text().splitlines() if line
        ]
        query_inits = [
            event
            for event in provider_events
            if event.get("event") == "session.init" and query in event.get("instruction", "")
        ]
        assert len(query_inits) == 1
        inputs = [event for event in provider_events if event.get("event") == "input.append"]
        assert len(inputs) >= 4
        assert all(event["frame_count"] == 1 for event in inputs)
        assert all(event["audio_present"] for event in inputs)
        assert all(not event["has_text_field"] for event in inputs)
        assert any(event["query_session"] and not event["force_listen"] for event in inputs)

        audits = [json.loads(line) for line in proxy_audit.read_text().splitlines() if line]
        forwarded = [
            event for event in audits if event.get("event") == "native_realtime_frame_forwarded"
        ]
        assert [event["user_query_present"] for event in forwarded] == [False, True, False]
        assert [
            event["user_query_sha256"]
            for event in forwarded
            if event["user_query_present"]
        ] == [expected_hash]
        assert all(event["input_transport"] == "native-video-realtime" for event in forwarded)
        accepted = [
            event for event in audits if event.get("event") == "query_frame_enqueued"
        ]
        assert len(accepted) == 1
        assert accepted[0]["query_event_id"] == "A1001:R1"
        delivered = [
            event for event in audits if event.get("event") == "provider_query_delivered"
        ]
        assert len(delivered) == 1
        assert delivered[0]["query_frame_queue_event_id"] == "A1001:R1"
        assert delivered[0]["query_frame_time_range"] == "1.000 seconds"
        warmups = [
            event for event in audits if event.get("event") == "provider_warmup_completed"
        ]
        assert len(warmups) == 1
        reset = post_json(reset_endpoint, {}, token, session_headers)
        assert reset == {"ok": True, "reset_acknowledged": True}
        audits_after_reset = [
            json.loads(line) for line in proxy_audit.read_text().splitlines() if line
        ]
        closed = [
            event
            for event in audits_after_reset
            if event.get("event") == "provider_session_closed"
        ]
        assert closed
        assert all(event["close_confirmed"] is True for event in closed)
        assert all(event["reconnect_settle_ms"] == 10 for event in closed)
        assert not any("/9j/2Q==" in json.dumps(event) for event in audits)
    finally:
        terminate(proxy)
        terminate(upstream)


def test_modelbest_native_realtime_v2_injects_query_audio_without_reconnect(tmp_path):
    upstream_port = free_port()
    proxy_port = free_port()
    upstream_events = tmp_path / "upstream-v2.jsonl"
    proxy_audit = tmp_path / "proxy-v2.jsonl"
    token = "native-realtime-v2-test-token-123456789"
    query = "alert once"
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    audio_dir = tmp_path / "query-audio" / "audio"
    audio_dir.mkdir(parents=True)
    pcm = struct.pack("<160f", *([0.25] * 160))
    pcm_path = audio_dir / f"{query_hash}.f32le"
    pcm_path.write_bytes(pcm)
    manifest_path = audio_dir.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "tts": {"engine": "test", "voice": "test"},
                "audio": {
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "sample_format": "f32le",
                    "chunk_duration_ms": 1000,
                },
                "entries": {
                    query_hash: {
                        "query": query,
                        "query_sha256": query_hash,
                        "pcm_path": f"audio/{pcm_path.name}",
                        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                        "byte_length": len(pcm),
                        "sample_count": len(pcm) // 4,
                        "duration_s": 0.01,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    upstream = subprocess.Popen(
        ["node", str(ROOT / "tests/fixtures/mock_modelbest_realtime_server.mjs")],
        cwd=ROOT,
        env={
            **os.environ,
            "MOCK_REALTIME_PORT": str(upstream_port),
            "MOCK_REALTIME_EVENTS": str(upstream_events),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert upstream.stdout.readline().strip() == "ready"

    proxy = subprocess.Popen(
        ["node", str(ROOT / "scripts/native_video_realtime_proxy.mjs")],
        cwd=ROOT,
        env={
            **os.environ,
            "UPSTREAM_API_KEY": "mock-upstream-key",
            "PROMPT_PROXY_ACCESS_TOKEN": token,
            "PROMPT_PROXY_ADVERTISED_MODEL": "MiniCPM-O-4.5-9B",
            "PROMPT_PROXY_PORT": str(proxy_port),
            "PROMPT_PROXY_AUDIT_PATH": str(proxy_audit),
            "NATIVE_REALTIME_API_BASE": (
                f"ws://127.0.0.1:{upstream_port}/v1/realtime?mode=video"
            ),
            "NATIVE_REALTIME_PROTOCOL": "modelbest-video-full-duplex-v1",
            "NATIVE_REALTIME_QUERY_MODE": "input-audio-once",
            "NATIVE_QUERY_AUDIO_MANIFEST": str(manifest_path),
            "NATIVE_VIDEO_SCHEMA": (
                "modelbest-realtime.input.append.video_frames.jpeg"
            ),
            "NATIVE_REALTIME_MAX_SESSION_S": "300",
            "NATIVE_REALTIME_PROVIDER_RECONNECT_SETTLE_MS": "10",
            "NATIVE_REALTIME_OUTPUT_GRACE_MS": "100",
            "JOYAI_SYSTEM_PROMPT_FILE": str(ROOT / "config/joyai_system_prompt.txt"),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        health_url = f"http://127.0.0.1:{proxy_port}/health"
        health = wait_health(health_url, proxy)
        assert health["query_transport"] == "input-audio-once"
        assert health["query_audio_entry_count"] == 1
        assert health["query_audio_realtime_pacing"] is True
        endpoint = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
        query_endpoint = f"http://127.0.0.1:{proxy_port}/v1/query-events"
        reset_endpoint = f"http://127.0.0.1:{proxy_port}/v1/streaming/reset"
        session_headers = {"X-Streaming-Session": "modelbest-v2-session-1"}
        frame = {
            "type": "image_url",
            "image_url": {"url": JPEG_16X16},
        }

        assert post_json(reset_endpoint, {}, token, session_headers) == {
            "ok": True,
            "reset_acknowledged": True,
        }
        health = wait_health(health_url, proxy)
        assert health["provider_target_session_prepared"] is True
        queued = post_json(
            query_endpoint,
            {
                "session_id": "modelbest-v2-session-1",
                "query_event_id": "A1001:R1",
                "query": query,
                "image_url": JPEG_16X16,
                "frame_time_range": "1.000 seconds",
            },
            token,
        )
        assert queued["status"] == "queued"
        prompted = post_json(
            endpoint,
            {
                "model": "MiniCPM-O-4.5-9B",
                "messages": [{"role": "user", "content": [frame]}],
            },
            token,
            session_headers,
        )
        assert prompted["choices"][0]["message"]["content"] == (
            "</response> mock reply"
        )
        assert prompted["native_realtime"]["provider_raw_content"] == "mock reply"
        assert prompted["native_realtime"]["provider_output_format"] == (
            "native-text-listen-to-response-marker"
        )
        assert prompted["native_realtime"]["query_audio_chunk_count"] == 1

        time.sleep(0.1)
        provider_events = [
            json.loads(line) for line in upstream_events.read_text().splitlines() if line
        ]
        session_inits = [event for event in provider_events if event["event"] == "session.init"]
        assert len(session_inits) == 1
        assert "USER QUERY:" not in session_inits[0]["instruction"]
        inputs = [event for event in provider_events if event["event"] == "input.append"]
        assert len(inputs) == 1
        assert inputs[0]["frame_count"] == 1
        assert inputs[0]["audio_nonzero"] is True

        audits = [
            json.loads(line) for line in proxy_audit.read_text().splitlines() if line
        ]
        delivered = [event for event in audits if event["event"] == "provider_query_delivered"]
        assert len(delivered) == 1
        assert delivered[0]["provider_session_reused_at_query"] is True
        assert delivered[0]["historical_frames_replayed"] is False
        assert delivered[0]["replayed_frame_count"] == 0
        assert delivered[0]["query_frame_count"] == 1
    finally:
        terminate(proxy)
        terminate(upstream)


def test_wall_media_clock_and_query_ingress_remain_monotonic_under_backpressure(
    tmp_path,
):
    upstream = MonotonicClockUpstream(("127.0.0.1", 0))
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    proxy_port = free_port()
    proxy_audit = tmp_path / "moss-clock-proxy.jsonl"
    token = "moss-clock-test-token-123456789"
    api_base = f"http://127.0.0.1:{upstream.server_port}/v1"
    proxy = subprocess.Popen(
        ["node", str(ROOT / "scripts/joyai_native_http_session_proxy.mjs")],
        cwd=ROOT,
        env={
            **os.environ,
            "UPSTREAM_API_KEY": "mock-moss-key",
            "UPSTREAM_API_BASE": api_base,
            "PROMPT_PROXY_ACCESS_TOKEN": token,
            "PROMPT_PROXY_ADVERTISED_MODEL": "MOSS-VL-Realtime",
            "PROMPT_PROXY_PORT": str(proxy_port),
            "PROMPT_PROXY_AUDIT_PATH": str(proxy_audit),
            "PROMPT_PROXY_STATE_PATH": str(tmp_path / "moss-clock-state.json"),
            "NATIVE_REALTIME_API_BASE": api_base,
            "NATIVE_REALTIME_PROTOCOL": "joyai-http-session-v1",
            "NATIVE_REALTIME_QUERY_MODE": "input-text-once",
            "NATIVE_VIDEO_SCHEMA": "joyai-http-session.image_url.jpeg",
            "NATIVE_STREAMING_MODE_POLICY": "proactive",
            "NATIVE_HTTP_SYSTEM_PROMPT_TRANSPORT": "system-role",
            "NATIVE_HTTP_FRAME_CLOCK": "wall-media",
            "NATIVE_HTTP_NON_QUERY_TIMEOUT_MS": "2000",
            "NATIVE_HTTP_QUERY_TIMEOUT_MS": "2000",
            "JOYAI_SYSTEM_PROMPT_FILE": str(ROOT / "config/joyai_system_prompt.txt"),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        wait_health(f"http://127.0.0.1:{proxy_port}/health", proxy)
        endpoint = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
        query_endpoint = f"http://127.0.0.1:{proxy_port}/v1/query-events"
        session_headers = {"X-Streaming-Session": "moss-clock-session"}
        frame = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,/9j/2Q=="},
        }

        def send_frame(timestamp):
            return post_json(
                endpoint,
                {
                    "model": "MOSS-VL-Realtime",
                    "messages": [{"role": "user", "content": [frame]}],
                },
                token,
                {**session_headers, "X-Frame-Time-Range": timestamp},
            )

        send_frame("398.000 seconds")
        upstream.request_started.clear()
        upstream.delay_next_s = 0.5
        worker = threading.Thread(target=send_frame, args=("399.000 seconds",))
        worker.start()
        assert upstream.request_started.wait(timeout=2)

        sent_at = time.time()
        queued = post_json(
            query_endpoint,
            {
                "session_id": "moss-clock-session",
                "query_event_id": "C1010:R1",
                "query": "answer once",
                "image_url": "data:image/jpeg;base64,/9j/2Q==",
                "frame_time_range": "1503.000 seconds",
                "ui_query_sent_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(sent_at)
                ),
                "ui_query_video_time_s": 1503.0,
                "captured_media_time_s": 1503.0,
            },
            token,
        )
        acceptance_elapsed_s = time.time() - sent_at
        assert queued["status"] == "queued"
        assert acceptance_elapsed_s < 0.25
        worker.join(timeout=2)
        assert not worker.is_alive()

        send_frame("400.000 seconds")
        send_frame("401.000 seconds")
        provider_times = [
            float(item["frame_time_range"].split()[0]) for item in upstream.requests
        ]
        assert provider_times[:2] == [398.0, 399.0]
        assert provider_times[2] == 1503.0
        assert provider_times[3] >= 1503.0
        assert provider_times == sorted(provider_times)

        audits = [
            json.loads(line) for line in proxy_audit.read_text().splitlines() if line
        ]
        accepted = [item for item in audits if item.get("event") == "query_frame_enqueued"]
        assert len(accepted) == 1
        assert accepted[0]["query_event_id"] == "C1010:R1"
        delivered = [
            item for item in audits if item.get("event") == "provider_query_delivered"
        ]
        assert len(delivered) == 1
        assert delivered[0]["query_frame_time_range"] == "1503.000 seconds"
    finally:
        terminate(proxy)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)


def test_mage_http_session_consumes_interactive_segment_and_audits_provider_500(
    tmp_path,
):
    upstream = MageSessionUpstream(("127.0.0.1", 0))
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    proxy_port = free_port()
    proxy_audit = tmp_path / "mage-proxy.jsonl"
    token = "mage-session-test-token-123456789"
    api_base = f"http://127.0.0.1:{upstream.server_port}/v1"
    proxy = subprocess.Popen(
        ["node", str(ROOT / "scripts/joyai_native_http_session_proxy.mjs")],
        cwd=ROOT,
        env={
            **os.environ,
            "UPSTREAM_API_KEY": "mock-mage-key",
            "UPSTREAM_API_BASE": api_base,
            "PROMPT_PROXY_ACCESS_TOKEN": token,
            "PROMPT_PROXY_ADVERTISED_MODEL": "Mage-VL",
            "PROMPT_PROXY_PORT": str(proxy_port),
            "PROMPT_PROXY_AUDIT_PATH": str(proxy_audit),
            "PROMPT_PROXY_STATE_PATH": str(tmp_path / "mage-proxy-state.json"),
            "NATIVE_REALTIME_API_BASE": api_base,
            "NATIVE_REALTIME_PROTOCOL": "joyai-http-session-v1",
            "NATIVE_REALTIME_QUERY_MODE": "input-text-once",
            "NATIVE_VIDEO_SCHEMA": "joyai-http-session.image_url.jpeg",
            "NATIVE_STREAMING_MODE_POLICY": "interactive-after-query",
            "NATIVE_HTTP_SYSTEM_PROMPT_TRANSPORT": "system-role",
            "NATIVE_HTTP_FRAME_CLOCK": "inbound-turn",
            "NATIVE_HTTP_NON_QUERY_TIMEOUT_MS": "100",
            "NATIVE_HTTP_QUERY_TIMEOUT_MS": "100",
            "JOYAI_SYSTEM_PROMPT_FILE": str(ROOT / "config/joyai_system_prompt.txt"),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        wait_health(f"http://127.0.0.1:{proxy_port}/health", proxy)
        endpoint = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
        headers = {
            "X-Streaming-Session": "mage-session-1",
            "X-Frame-Time-Range": "1.000 seconds",
        }
        frame = {
            "type": "image_url",
            "image_url": {"url": JPEG_32X24},
        }
        smaller_frame = {
            "type": "image_url",
            "image_url": {"url": JPEG_16X16},
        }

        post_json(
            endpoint,
            {
                "model": "Mage-VL",
                "messages": [
                    {
                        "role": "user",
                        "content": [frame, {"type": "text", "text": "alert once"}],
                    }
                ],
            },
            token,
            headers,
        )
        for timestamp in ("2.000 seconds", "3.000 seconds"):
            post_json(
                endpoint,
                {
                    "model": "Mage-VL",
                    "messages": [{"role": "user", "content": [smaller_frame]}],
                },
                token,
                {**headers, "X-Frame-Time-Range": timestamp},
            )

        with pytest.raises(urllib.error.HTTPError) as failure:
            post_json(
                endpoint,
                {
                    "model": "Mage-VL",
                    "messages": [{"role": "user", "content": [frame]}],
                },
                token,
                {**headers, "X-Frame-Time-Range": "4.000 seconds"},
            )
        assert failure.value.code == 500

        post_json(
            f"http://127.0.0.1:{proxy_port}/reset",
            {},
            token,
        )
        timeout_response = post_json(
            endpoint,
            {
                "model": "Mage-VL",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            frame,
                            {"type": "text", "text": "timeout query once"},
                        ],
                    }
                ],
            },
            token,
            {
                "X-Streaming-Session": "mage-session-2",
                "X-Frame-Time-Range": "1.000 seconds",
            },
        )
        assert timeout_response["choices"][0]["message"]["content"] == "</silence>"
        assert timeout_response["streamingharness"]["synthetic_timeout"] is True
        assert timeout_response["native_realtime"]["failure_class"] == (
            "model_latency_timeout"
        )

        assert [item["streaming_mode"] for item in upstream.requests] == [
            "interactive",
            "interactive",
            "proactive",
            "proactive",
            "interactive",
        ]
        text_counts = [
            sum(
                content.get("type") == "text"
                for message in item["messages"]
                for content in (
                    message.get("content", [])
                    if isinstance(message.get("content"), list)
                    else []
                )
            )
            for item in upstream.requests
        ]
        assert text_counts == [1, 0, 0, 0, 1]

        audits = [
            json.loads(line) for line in proxy_audit.read_text().splitlines() if line
        ]
        forwarded = [
            item
            for item in audits
            if item.get("event") == "native_realtime_frame_forwarded"
        ]
        assert [item["provider_streaming_mode"] for item in forwarded[:3]] == [
            "interactive",
            "interactive",
            "proactive",
        ]
        assert forwarded[0]["interactive_query_pending"] is True
        assert forwarded[1]["interactive_segment_consumed"] is True
        normalized = [
            item
            for item in audits
            if item.get("event") == "stream_frame_normalized_for_provider"
        ]
        assert normalized
        assert all(
            item["target_size"] == {"width": 32, "height": 24}
            for item in normalized
        )
        assert forwarded[1]["interactive_query_pending"] is False

        failed_forward = next(
            item
            for item in forwarded
            if item.get("failure_class") == "provider_frame_preprocessing_error"
        )
        assert failed_forward["forwarding_status"] == "provider_error"
        assert failed_forward["failure_class"] == "provider_frame_preprocessing_error"
        assert failed_forward["retryable_infrastructure"] is True
        failed_response = next(
            item
            for item in audits
            if item.get("event") == "upstream_response_received"
            and item.get("ok") is False
        )
        assert failed_response["request_id"] == failed_forward["request_id"]
        assert failed_response["failure_class"] == (
            "provider_frame_preprocessing_error"
        )
        timeout_forward = next(
            item
            for item in forwarded
            if item.get("failure_class") == "model_latency_timeout"
        )
        assert timeout_forward["forwarding_status"] == "model_latency_timeout"
        assert timeout_forward["retryable_infrastructure"] is False
        assert timeout_forward["adapter_response_policy"] == (
            "audited_synthetic_silence_http_200"
        )
        timeout_audit = next(
            item
            for item in audits
            if item.get("event") == "upstream_response_received"
            and item.get("failure_class") == "model_latency_timeout"
        )
        assert timeout_audit["ok"] is False
        assert timeout_audit["deadline_exceeded"] is True
        timeout_delivery = next(
            item
            for item in audits
            if item.get("event") == "provider_query_delivered"
            and item.get("delivery_status") == "model_latency_timeout"
        )
        assert timeout_delivery["user_query_present"] is True
    finally:
        terminate(proxy)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
