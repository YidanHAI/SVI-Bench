import base64
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def jpeg_data_url(color):
    image = Image.new("RGB", (4, 4), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def request_json(url, *, method="GET", payload=None, bearer=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"}
    if bearer:
        headers["authorization"] = f"Bearer {bearer}"
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read())


class MockProvider:
    def __init__(self):
        self.requests = []
        self.forced_status = 0
        self.delay_seconds = 0.0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                return

            def do_GET(self):
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                self._json(200, {"object": "list", "data": [{"id": "mock-vlm"}]})

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                size = int(self.headers.get("content-length") or 0)
                payload = json.loads(self.rfile.read(size))
                owner.requests.append(payload)
                forced_status = owner.forced_status
                delay_seconds = owner.delay_seconds
                if delay_seconds:
                    time.sleep(delay_seconds)
                if forced_status:
                    self._json(forced_status, {
                        "error": {
                            "message": "forced provider failure",
                            "type": "mock_provider_error",
                        }
                    })
                    return
                text = "</response> acknowledged" if len(owner.requests) == 2 else "</silence>"
                self._json(200, {
                    "id": f"mock-{len(owner.requests)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "mock-vlm",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                })

            def _json(self, status, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except BrokenPipeError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self):
        return self.server.server_address[1]

    def start(self):
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def frame_payload(*, query="", color=(0, 0, 0)):
    content = []
    if query:
        content.append({"type": "text", "text": query})
    content.append({"type": "image_url", "image_url": {"url": jpeg_data_url(color)}})
    return {
        "model": "mock-vlm",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.85,
    }


def count_images(messages):
    return sum(
        1
        for message in messages
        for item in (message.get("content") if isinstance(message.get("content"), list) else [])
        if item.get("type") == "image_url"
    )


def test_official_scaffold_query_once_and_history_replay(tmp_path):
    provider = MockProvider()
    provider.start()
    port = free_port()
    token = "test-access-token-at-least-24-chars"
    audit_path = tmp_path / "audit.jsonl"
    env = {
        **os.environ,
        "UPSTREAM_API_BASE": f"http://127.0.0.1:{provider.port}/v1",
        "UPSTREAM_API_KEY": "test-upstream-key",
        "PROMPT_PROXY_ADVERTISED_MODEL": "mock-vlm",
        "PROMPT_PROXY_ACCESS_TOKEN": token,
        "PROMPT_PROXY_HOST": "127.0.0.1",
        "PROMPT_PROXY_PORT": str(port),
        "PROMPT_PROXY_AUDIT_PATH": str(audit_path),
        "JOYAI_SYSTEM_PROMPT_FILE": str(ROOT / "config" / "joyai_system_prompt.txt"),
        "JOYAI_SCAFFOLD_ENABLE_SUMMARIZER": "0",
        "JOYAI_SCAFFOLD_SDK_MAX_RETRIES": "0",
        "JOYAI_SCAFFOLD_NON_QUERY_TIMEOUT_SECONDS": "0.2",
        "JOYAI_SCAFFOLD_QUERY_TIMEOUT_SECONDS": "0.3",
        "JOYAI_SCAFFOLD_WARMUP_TIMEOUT_SECONDS": "2",
        "LOG_LEVEL": "WARNING",
    }
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "joyai_scaffold_adapter.py")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=1)
                raise AssertionError(f"adapter exited early\nstdout={stdout}\nstderr={stderr}")
            try:
                status, health = request_json(f"{base}/health")
                if status == 200:
                    break
            except Exception:
                time.sleep(0.05)
        else:
            raise AssertionError("adapter did not become healthy")

        assert health["adapter"] == "joyai-official-live-adapter"
        assert health["summarizer_enabled"] is False
        assert health["query_context_policy"] == "official-live-adapter-history-replay"
        assert health["generation_parameter_policy"] == "honor-webui-ingress"
        assert health["main_generation_defaults"] == {
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        assert health["upstream_request_policy"] == {
            "sdk_max_retries": 0,
            "non_query_timeout_s": 0.2,
            "query_timeout_s": 0.3,
            "warmup_timeout_s": 2.0,
            "timeout_classification": "model_capability",
            "timeout_transport": "audited_synthetic_silence_http_200",
            "rate_limit_and_transport_classification": "retryable_infrastructure",
            "infrastructure_error_transport": "non_retryable_http_424",
        }
        assert health["provider_warmup"]["status"] == "not_run"

        status, warmup = request_json(
            f"{base}/v1/warmup",
            method="POST",
            payload={"model": "mock-vlm", "image_url": jpeg_data_url((5, 5, 5))},
            bearer=token,
        )
        assert status == 200
        assert warmup["ok"] is True
        assert warmup["multimodal_input"] is True
        assert warmup["session_state_unchanged"] is True
        assert len(provider.requests) == 1
        assert count_images(provider.requests[0]["messages"]) == 1

        status, health = request_json(f"{base}/health")
        assert status == 200
        assert health["sessions"] == 0
        assert health["provider_warmup"]["status"] == "ready"

        headers = {"authorization": f"Bearer {token}", "x-streaming-session": "test-session"}

        def chat(payload, session_id="test-session"):
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=data,
                method="POST",
                headers={
                    **headers,
                    "x-streaming-session": session_id,
                    "content-type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read())

        pre_query = chat(frame_payload(color=(10, 10, 10)))
        assert pre_query["choices"][0]["message"]["content"] == "</silence>"
        assert len(provider.requests) == 1

        query = "Tell me when the alarm condition occurs."
        query_response = chat(frame_payload(query=query, color=(20, 20, 20)))
        assert query_response["choices"][0]["message"]["content"] == "</response> acknowledged"
        assert len(provider.requests) == 2
        assert count_images(provider.requests[1]["messages"]) == 2
        assert provider.requests[1]["max_tokens"] == 512
        assert provider.requests[1]["temperature"] == 0.7
        assert provider.requests[1]["top_p"] == 0.85

        later_response = chat(frame_payload(color=(30, 30, 30)))
        assert later_response["choices"][0]["message"]["content"] == "</silence>"
        assert len(provider.requests) == 3
        assert count_images(provider.requests[2]["messages"]) == 3
        serialized = json.dumps(provider.requests[2]["messages"], ensure_ascii=False)
        assert serialized.count(query) == 1

        queue_session = "queue-session"
        queued_queries = [
            ("R1", "First queued question", (40, 0, 0), "5.000 seconds"),
            ("R2", "Second queued question", (0, 40, 0), "6.000 seconds"),
        ]
        for query_id, text, color, frame_time_range in queued_queries:
            status, queued = request_json(
                f"{base}/v1/query-events",
                method="POST",
                payload={
                    "session_id": queue_session,
                    "query_event_id": f"T1:{query_id}",
                    "query": text,
                    "image_url": jpeg_data_url(color),
                    "frame_time_range": frame_time_range,
                    "ui_query_sent_at": "2026-08-19T08:00:00.000Z",
                    "ui_query_video_time_s": float(frame_time_range.split()[0]),
                },
                bearer=token,
            )
            assert status == 200
            assert queued["ok"] is True
            assert queued["status"] == "queued"

        first_queued = chat(
            frame_payload(query=queued_queries[1][1], color=(0, 0, 40)),
            session_id=queue_session,
        )
        first_meta = first_queued["streamingharness"]["query_frame_queue"]
        assert first_meta["query_event_id"] == "T1:R1"
        assert first_meta["frame_time_range"] == "5.000 seconds"
        assert first_meta["query_queue_delay_ms"] >= 0
        first_serialized = json.dumps(provider.requests[-1]["messages"], ensure_ascii=False)
        assert "First queued question" in first_serialized
        assert "Second queued question" not in first_serialized

        second_queued = chat(frame_payload(color=(0, 0, 50)), session_id=queue_session)
        second_meta = second_queued["streamingharness"]["query_frame_queue"]
        assert second_meta["query_event_id"] == "T1:R2"
        assert second_meta["frame_time_range"] == "6.000 seconds"
        second_serialized = json.dumps(provider.requests[-1]["messages"], ensure_ascii=False)
        assert second_serialized.count("First queued question") == 1
        assert second_serialized.count("Second queued question") == 1

        provider.forced_status = 429
        request_count = len(provider.requests)
        with pytest.raises(urllib.error.HTTPError) as rate_limit_error:
            chat(frame_payload(query="Rate-limit probe"), session_id="rate-limit-session")
        assert rate_limit_error.value.code == 424
        assert len(provider.requests) == request_count + 1
        provider.forced_status = 0

        provider.delay_seconds = 0.8
        request_count = len(provider.requests)
        timeout_response = chat(
            frame_payload(query="Deadline probe"),
            session_id="timeout-session",
        )
        assert timeout_response["choices"][0]["message"]["content"] == "</silence>"
        assert len(provider.requests) == request_count + 1
        provider.delay_seconds = 0.0

        status, reset = request_json(
            f"{base}/reset",
            method="POST",
            payload={},
            bearer=token,
        )
        assert status == 200
        assert reset["sessions_removed"] == 4

        events = [json.loads(line) for line in audit_path.read_text().splitlines()]
        ingress = [event for event in events if event["event"] == "joyai_scaffold_frame_received"]
        assert len(ingress) == 7
        assert [event["user_query_present"] for event in ingress[:3]] == [False, True, False]
        queued_ingress = ingress[3:5]
        assert [event["query_frame_queue_event_id"] for event in queued_ingress] == [
            "T1:R1",
            "T1:R2",
        ]
        assert all(
            event["query_frame_queue_policy"] == "fifo-query-time-frame"
            for event in queued_ingress
        )
        upstream_events = [
            event for event in events
            if event["event"] == "joyai_scaffold_upstream_request"
        ]
        assert len(upstream_events) == 6
        assert upstream_events[0]["generation_parameter_policy"] == "honor-webui-ingress"
        assert upstream_events[0]["max_tokens"] == 512
        assert upstream_events[0]["temperature"] == 0.7
        assert upstream_events[0]["top_p"] == 0.85
        assert upstream_events[0]["thinking_mode"] == "provider-default"
        assert upstream_events[0]["sdk_max_retries"] == 0
        assert upstream_events[0]["provider_attempt_budget"] == 1
        assert upstream_events[0]["request_kind"] == "query"
        assert upstream_events[0]["request_timeout_s"] == 0.3
        failed_upstream_events = [
            event for event in events
            if event["event"] == "upstream_response_received" and event["ok"] is False
        ]
        assert [event["failure_class"] for event in failed_upstream_events] == [
            "provider_rate_limit",
            "model_latency_timeout",
        ]
        assert failed_upstream_events[0]["retryable_infrastructure"] is True
        assert failed_upstream_events[0]["provider_attempt_budget"] == 1
        assert failed_upstream_events[1]["deadline_exceeded"] is True
        assert failed_upstream_events[1]["retryable_infrastructure"] is False
        assert failed_upstream_events[1]["adapter_response_policy"] == (
            "audited_synthetic_silence_http_200"
        )
        assert failed_upstream_events[1]["latency_ms"] < 1000
        queued_events = [event for event in events if event["event"] == "query_frame_queued"]
        dequeued_events = [event for event in events if event["event"] == "query_frame_dequeued"]
        assert [event["query_event_id"] for event in queued_events] == ["T1:R1", "T1:R2"]
        assert [event["query_event_id"] for event in dequeued_events] == ["T1:R1", "T1:R2"]
        warmup_events = [event for event in events if event["event"] == "provider_warmup_completed"]
        assert len(warmup_events) == 1
        assert warmup_events[0]["ok"] is True
        assert warmup_events[0]["session_state_unchanged"] is True
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        provider.close()
