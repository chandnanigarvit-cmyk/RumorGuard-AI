"""Thin async wrapper around the LLM API. Swap providers by editing only this file."""
import json
import logging
import re

from anthropic import APIError, AsyncAnthropic

from config import settings
from models import AppError

logger = logging.getLogger("rumorguard.llm")
_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if not settings.anthropic_api_key:
        raise AppError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.", 503)
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


def extract_json(text: str) -> dict:
    """Parse a JSON object out of an LLM reply (tolerates ```json fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    return obj


async def call_llm_json(system: str, user: str, max_tokens: int | None = None) -> dict:
    client = _get_client()
    last_err = None
    for attempt in range(2):  # one retry if the model returns malformed JSON
        try:
            resp = await client.messages.create(
                model=settings.llm_model,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except APIError as e:
            logger.exception("LLM API error")
            raise AppError(f"LLM API error: {e}", 502)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            return extract_json(text)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            logger.warning("Invalid JSON from LLM (attempt %d): %s", attempt + 1, e)
    raise AppError(f"LLM returned invalid JSON: {last_err}", 502)
