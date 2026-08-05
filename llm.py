"""OpenAI Responses API over raw requests — no SDK, Pi-friendly.

Lifted from the QOTD project. Kept deliberately thin: one generate() that takes
instructions (system) separately from input (turn content), and a defensive text
extractor that walks the output array instead of blind-indexing, since reasoning
items can interleave with the message item.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

DEFAULT_MODEL = "gpt-5.4-mini"
OPENAI_URL = "https://api.openai.com/v1/responses"


class LLMError(RuntimeError):
    pass


def generate(
    input_messages: list[dict[str, Any]],
    *,
    instructions: str | None = None,
    model: str | None = None,
    timeout: float = 120.0,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not set")

    payload: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "input": input_messages,
    }
    if instructions:
        payload["instructions"] = instructions
    if max_output_tokens:
        payload["max_output_tokens"] = max_output_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise LLMError(f"API {resp.status_code}: {resp.text[:400]}")
    return _extract_text(resp.json())


def _extract_text(data: dict[str, Any]) -> str:
    convenience = data.get("output_text")
    if isinstance(convenience, str) and convenience.strip():
        return convenience.strip()

    parts: list[str] = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for chunk in item.get("content", []):
            if chunk.get("type") in ("output_text", "text") and "text" in chunk:
                parts.append(chunk["text"])

    text = "".join(parts).strip()
    if not text:
        raise LLMError(f"no text found in response: {json.dumps(data)[:300]}")
    return text
