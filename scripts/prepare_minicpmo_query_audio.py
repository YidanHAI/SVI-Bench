#!/usr/bin/env python3
"""Build deterministic MiniCPM native-video Query audio from a task manifest."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import edge_tts


SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
SAMPLE_FORMAT = "f32le"
CHUNK_DURATION_MS = 1_000


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural")
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--pitch", default="+0Hz")
    parser.add_argument("--volume", default="+0%")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--task-ids", default="")
    return parser.parse_args()


def load_queries(tasks_path: Path, selected_ids: set[str]) -> dict[str, str]:
    queries: dict[str, str] = {}
    with tasks_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            task_id = str(task.get("id", "")).strip()
            if selected_ids and task_id not in selected_ids:
                continue
            rounds = task.get("queries") or [
                {"query": task.get("query", ""), "id": "R1"}
            ]
            for round_item in rounds:
                query = str(round_item.get("query", "")).strip()
                if not query:
                    raise ValueError(
                        f"Empty Query at {tasks_path}:{line_number} task={task_id}"
                    )
                queries.setdefault(sha256_text(query), query)
    if not queries:
        raise ValueError("No Query text was selected")
    return queries


async def synthesize_mp3(
    query: str,
    target: Path,
    *,
    voice: str,
    rate: str,
    pitch: str,
    volume: str,
    retries: int,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            communicator = edge_tts.Communicate(
                query,
                voice=voice,
                rate=rate,
                pitch=pitch,
                volume=volume,
            )
            await communicator.save(str(target))
            if target.stat().st_size <= 0:
                raise RuntimeError("TTS returned an empty MP3")
            return
        except Exception as error:  # edge service failures are transient
            last_error = error
            if attempt < retries:
                await asyncio.sleep(min(20, 2 ** (attempt - 1)))
    raise RuntimeError(f"TTS failed after {retries} attempts: {last_error}")


async def build_entry(
    query_hash: str,
    query: str,
    audio_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    pcm_path = audio_dir / f"{query_hash}.f32le"
    if not pcm_path.exists() or pcm_path.stat().st_size == 0:
        with tempfile.TemporaryDirectory(prefix="minicpmo-tts-") as temporary:
            mp3_path = Path(temporary) / "query.mp3"
            pcm_temporary = Path(temporary) / "query.f32le"
            await synthesize_mp3(
                query,
                mp3_path,
                voice=args.voice,
                rate=args.rate,
                pitch=args.pitch,
                volume=args.volume,
                retries=args.retries,
            )
            subprocess.run(
                [
                    args.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(mp3_path),
                    "-ar",
                    str(SAMPLE_RATE_HZ),
                    "-ac",
                    str(CHANNELS),
                    "-f",
                    SAMPLE_FORMAT,
                    str(pcm_temporary),
                ],
                check=True,
            )
            payload = pcm_temporary.read_bytes()
            if not payload or len(payload) % 4:
                raise RuntimeError(f"Invalid float32 PCM generated for Query {query_hash}")
            staging = pcm_path.with_suffix(f".tmp.{os.getpid()}")
            staging.write_bytes(payload)
            os.replace(staging, pcm_path)

    payload = pcm_path.read_bytes()
    if not payload or len(payload) % 4:
        raise RuntimeError(f"Invalid cached float32 PCM: {pcm_path}")
    sample_count = len(payload) // 4
    return {
        "query": query,
        "query_sha256": query_hash,
        "pcm_path": str(Path("audio") / pcm_path.name),
        "pcm_sha256": sha256_bytes(payload),
        "byte_length": len(payload),
        "sample_count": sample_count,
        "duration_s": sample_count / SAMPLE_RATE_HZ,
    }


async def main() -> None:
    args = parse_args()
    if args.retries < 1 or args.retries > 10:
        raise ValueError("--retries must be between 1 and 10")
    tasks_path = args.tasks.resolve()
    out_dir = args.out_dir.resolve()
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = {
        value.strip() for value in args.task_ids.split(",") if value.strip()
    }
    queries = load_queries(tasks_path, selected_ids)

    entries: dict[str, dict[str, object]] = {}
    for index, (query_hash, query) in enumerate(queries.items(), start=1):
        entries[query_hash] = await build_entry(
            query_hash, query, audio_dir, args
        )
        print(f"[{index}/{len(queries)}] {query_hash[:12]}", flush=True)

    manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tasks": str(tasks_path),
        "source_tasks_sha256": sha256_bytes(tasks_path.read_bytes()),
        "tts": {
            "engine": "edge-tts",
            "engine_version": importlib.metadata.version("edge-tts"),
            "voice": args.voice,
            "rate": args.rate,
            "pitch": args.pitch,
            "volume": args.volume,
        },
        "audio": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": CHANNELS,
            "sample_format": SAMPLE_FORMAT,
            "chunk_duration_ms": CHUNK_DURATION_MS,
        },
        "entry_count": len(entries),
        "entries": entries,
    }
    manifest_path = out_dir / "manifest.json"
    staging = manifest_path.with_suffix(f".tmp.{os.getpid()}")
    staging.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, manifest_path)
    print(f"manifest={manifest_path} entries={len(entries)}")


if __name__ == "__main__":
    asyncio.run(main())
