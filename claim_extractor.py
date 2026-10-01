"""Step: extract a structured claim from raw text."""
from config import settings
from llm import call_llm_json
from models import ExtractedClaim
from prompts import EXTRACTION_SYSTEM, build_extraction_user_prompt

_EMPTY = {"null", "none", "n/a", ""}


def _clean(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return None if s.lower() in _EMPTY else s


async def extract_claim(text: str) -> ExtractedClaim:
    text = text.strip()[: settings.max_input_chars]
    data = await call_llm_json(EXTRACTION_SYSTEM, build_extraction_user_prompt(text), max_tokens=600)
    return ExtractedClaim(
        claim=(_clean(data.get("claim")) or "").rstrip("."),
        location=_clean(data.get("location")),
        entity=_clean(data.get("entity")),
        event=_clean(data.get("event")),
        scope=_clean(data.get("scope")),
        duration=_clean(data.get("duration")),
        date_time=_clean(data.get("date_time")),
    )
