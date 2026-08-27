#!/usr/bin/env python3
"""Shared configuration helpers for the OpenAI-compatible Judge API."""

from __future__ import annotations

import os
from urllib.parse import urlparse


DEFAULT_JUDGE_MODEL = "GPT-5.5"


def load_api_key() -> str:
    value = os.getenv("OPENAI_API_KEY", "").strip().removeprefix("Bearer ").strip()
    if not value:
        raise RuntimeError("OPENAI_API_KEY is required")
    return value


def chat_completions_url() -> str:
    base = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise RuntimeError("OPENAI_BASE_URL is required")
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("OPENAI_BASE_URL must be an absolute HTTP(S) URL")
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        return base
    return base + "/chat/completions"
