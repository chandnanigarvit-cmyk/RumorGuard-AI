"""Async wrapper around the Google Gemini API."""
import json
import logging
import re

from google import genai
from google.genai import types

from config import settings
from models import AppError

logger = logging.getLogger("rumorguard.llm")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client

    if not settings.gemini_api_key:
        raise AppError(
            "GEMINI_API_KEY is not set. Add it to the environment variables.",
            503,
        )

    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)

    return _client


def extract_json(text: str) -> dict:
    """Parse a JSON object from a Gemini response."""
    text = re.sub(r"```(?:json)?", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end <= start:
        raise ValueError("No JSON object found")

    obj = json.loads(text[start:end + 1])

    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")

    return obj


async def call_llm_json(
    system: str,
    user: str,
    max_tokens: int | None = None,
) -> dict:

    client = _get_client()
    last_err = None

    for attempt in range(2):

        try:
            response = await client.aio.models.generate_content(
                model=settings.llm_model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                    max_output_tokens=max_tokens or settings.llm_max_tokens,
                    response_mime_type="application/json",
                ),
            )

            text = response.text or ""

        except Exception as e:
            logger.exception("Gemini API error")
            raise AppError(f"LLM API error: {e}", 502)

        try:
            return extract_json(text)

        except (ValueError, json.JSONDecodeError) as e:
            last_err = e

            logger.warning(
                "Invalid JSON from Gemini (attempt %d): %s",
                attempt + 1,
                e,
            )

    raise AppError(
        f"LLM returned invalid JSON: {last_err}",
        502,
    )
