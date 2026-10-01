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

_PREVIEW_CHARS = 500


class _EmptyLLMResponse(ValueError):
    """Gemini answered (HTTP 200) but returned no usable text."""


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


def _response_text(response) -> str:
    """Return response.text, or "" if it is missing, None, or raises."""
    try:
        text = response.text
    except Exception:
        # The SDK can raise when the response has no text parts.
        return ""
    return text if isinstance(text, str) else ""


def _first_candidate(response):
    """Return the first candidate, or None if there is none."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    try:
        return candidates[0]
    except Exception:
        return None


def _candidate_text(response) -> str:
    """Join text parts from the first candidate (skipping thought parts)."""
    candidate = _first_candidate(response)
    if candidate is None:
        return ""

    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []

    chunks: list[str] = []
    for part in parts:
        if getattr(part, "thought", False):
            continue
        part_text = getattr(part, "text", None)
        if isinstance(part_text, str) and part_text:
            chunks.append(part_text)

    return "".join(chunks)


def _enum_name(value) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "name", value))


def _describe_response(response, sdk_text: str, text: str) -> str:
    """Build a safe, short diagnostic summary. Never includes secrets."""
    try:
        candidates = getattr(response, "candidates", None) or []
        candidate = _first_candidate(response)
        finish_reason = _enum_name(getattr(candidate, "finish_reason", None))
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = _enum_name(getattr(feedback, "block_reason", None))
        preview = repr(text)[:_PREVIEW_CHARS]

        return (
            f"response.text_empty={not sdk_text.strip()} "
            f"candidates={len(candidates)} "
            f"finish_reason={finish_reason} "
            f"block_reason={block_reason} "
            f"text_len={len(text)} "
            f"preview={preview}"
        )
    except Exception as diag_err:  # diagnostics must never break the request
        return f"diagnostics unavailable ({type(diag_err).__name__})"


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

        except Exception as e:
            logger.exception("Gemini API error")
            raise AppError(f"LLM API error: {e}", 502)

        # 1) Prefer response.text.
        sdk_text = _response_text(response)
        text = sdk_text

        # 2) Fall back to the first candidate's content parts.
        if not text.strip():
            text = _candidate_text(response)
            if text.strip():
                logger.info(
                    "Gemini response.text was empty (attempt %d); "
                    "recovered text from candidate parts",
                    attempt + 1,
                )

        try:
            if not text.strip():
                candidate = _first_candidate(response)
                finish_reason = _enum_name(
                    getattr(candidate, "finish_reason", None)
                )
                feedback = getattr(response, "prompt_feedback", None)
                block_reason = _enum_name(
                    getattr(feedback, "block_reason", None)
                )
                raise _EmptyLLMResponse(
                    "Gemini returned HTTP 200 but no text "
                    f"(candidates={len(getattr(response, 'candidates', None) or [])}, "
                    f"finish_reason={finish_reason}, "
                    f"block_reason={block_reason})"
                )

            return extract_json(text)

        except (ValueError, json.JSONDecodeError) as e:
            last_err = e

            logger.warning(
                "Invalid JSON from Gemini (attempt %d): %s | %s",
                attempt + 1,
                e,
                _describe_response(response, sdk_text, text),
            )

    if isinstance(last_err, _EmptyLLMResponse):
        raise AppError(f"LLM returned no usable text: {last_err}", 502)

    raise AppError(
        f"LLM returned invalid JSON: {last_err}",
        502,
    )
