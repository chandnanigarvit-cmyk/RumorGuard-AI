"""Verification, claim-drift detection, ML scoring, correction generation and pipeline."""
from __future__ import annotations

import asyncio
import re

import rag
from claim_extractor import extract_claim
from llm import call_llm_json
from ml_drift import predict_drift
from models import (
    AppError,
    ClaimDrift,
    EvidenceItem,
    ExtractedClaim,
    MLDriftScores,
    Verdict,
    VerifyResponse,
)
from prompts import (
    CORRECTION_SYSTEM,
    VERIFY_SYSTEM,
    build_correction_user_prompt,
    build_verify_user_prompt,
)

NO_EVIDENCE_CORRECTION = (
    "No official source in the knowledge base addresses this claim, so it can be "
    "neither confirmed nor denied. Please check the organisation's official "
    "channels before sharing it."
)


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    s = s.lower()

    for a, b in (
        ("’", "'"),
        ("‘", "'"),
        ("“", '"'),
        ("”", '"'),
        ("–", "-"),
        ("—", "-"),
    ):
        s = s.replace(a, b)

    return re.sub(r"\s+", " ", s).strip()


def _str_or_none(v) -> str | None:
    if v is None:
        return None

    s = str(v).strip()

    return None if s.lower() in {"", "null", "none", "n/a"} else s


def _str_list(v) -> list[str]:
    if not isinstance(v, list):
        return []

    return [
        str(x).strip()
        for x in v
        if str(x).strip()
    ]


def _resolve_evidence(items, chunks) -> list[EvidenceItem]:
    """
    Keep only evidence that really exists in retrieved chunks.

    The LLM cannot invent evidence because every returned evidence item
    must correspond to an actual retrieved RAG chunk.
    """

    by_id = {
        c.chunk_id: c
        for c in chunks
    }

    resolved: list[EvidenceItem] = []
    seen = set()

    for item in items if isinstance(items, list) else []:

        if not isinstance(item, dict):
            continue

        chunk = by_id.get(
            str(item.get("chunk_id", "")).strip()
        )

        if not chunk:
            continue

        quote = str(
            item.get("quote", "")
        ).strip()

        if quote and _norm(quote) in _norm(chunk.text):
            text = quote
        else:
            text = chunk.text

        key = (
            chunk.chunk_id,
            _norm(text),
        )

        if key in seen:
            continue

        seen.add(key)

        resolved.append(
            EvidenceItem(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                text=text,
                similarity=chunk.similarity,
            )
        )

    return resolved


def _build_drift(raw) -> ClaimDrift:
    """
    Convert Gemini's structured claim-drift analysis
    into the API ClaimDrift model.
    """

    raw = raw if isinstance(raw, dict) else {}

    drift = ClaimDrift(
        scope_change=_str_or_none(
            raw.get("scope_change")
        ),
        location_change=_str_or_none(
            raw.get("location_change")
        ),
        duration_change=_str_or_none(
            raw.get("duration_change")
        ),
        date_time_change=_str_or_none(
            raw.get("date_time_change")
        ),
        context_removed=_str_list(
            raw.get("context_removed")
        ),
    )

    drift.drift_detected = any(
        [
            drift.scope_change,
            drift.location_change,
            drift.duration_change,
            drift.date_time_change,
            drift.context_removed,
        ]
    )

    return drift


def _parse_verdict(v) -> Verdict:
    try:
        return Verdict(
            str(v).strip().upper()
        )
    except ValueError:
        return Verdict.INSUFFICIENT_EVIDENCE


def _clamp(v) -> int:
    try:
        return max(
            0,
            min(
                100,
                int(round(float(v))),
            ),
        )
    except (TypeError, ValueError):
        return 0


def _run_ml_drift(
    official_text: str,
    rumor_text: str,
) -> MLDriftScores:
    """
    Run the lightweight supervised ML model.

    The ML model compares the retrieved official evidence
    against the extracted rumor claim.
    """

    if not official_text or not rumor_text:
        return MLDriftScores()

    try:
        scores = predict_drift(
            official_claim=official_text,
            rumor_claim=rumor_text,
        )

        return MLDriftScores(
            scope=float(scores.get("scope", 0.0)),
            duration=float(scores.get("duration", 0.0)),
            location=float(scores.get("location", 0.0)),
            context=float(scores.get("context", 0.0)),
            overall=float(scores.get("overall", 0.0)),
        )

    except Exception:
        # ML is an additional signal. A failure here should not
        # break the core RAG + Gemini verification pipeline.
        return MLDriftScores()


def _insufficient(
    raw_text: str,
    input_type: str,
    extracted: ExtractedClaim,
    explanation: str,
) -> VerifyResponse:

    return VerifyResponse(
        verdict=Verdict.INSUFFICIENT_EVIDENCE,
        confidence=0,
        claim=extracted.claim,
        input_type=input_type,
        source_text=raw_text,
        extracted_claim=extracted,
        evidence=[],
        contradictions=[],
        missing_context=[],
        claim_drift=ClaimDrift(),
        explanation=explanation,
        correction=NO_EVIDENCE_CORRECTION,
        alternatives=[],
        sources=[],
    )


# ----------------------------------------------------------------------------- #
# LLM stages
# ----------------------------------------------------------------------------- #

async def verify_claim(
    extracted: ExtractedClaim,
    chunks,
) -> dict:

    return await call_llm_json(
        VERIFY_SYSTEM,
        build_verify_user_prompt(
            extracted.model_dump(),
            chunks,
        ),
    )


async def generate_correction(
    claim: str,
    verdict: Verdict,
    evidence: list[EvidenceItem],
    drift: ClaimDrift,
    explanation: str,
) -> dict:

    return await call_llm_json(
        CORRECTION_SYSTEM,
        build_correction_user_prompt(
            claim,
            verdict.value,
            [e.text for e in evidence],
            drift.model_dump(),
            explanation,
        ),
        max_tokens=600,
    )


# ----------------------------------------------------------------------------- #
# Full pipeline
# ----------------------------------------------------------------------------- #

async def run_pipeline(
    raw_text: str,
    input_type: str,
) -> VerifyResponse:

    # -------------------------------------------------------------------------
    # 1. Extract factual claim
    # -------------------------------------------------------------------------

    extracted = await extract_claim(raw_text)

    if not extracted.claim:
        raise AppError(
            "No verifiable factual claim was found in the input.",
            422,
        )

    # -------------------------------------------------------------------------
    # 2. Retrieve official evidence using RAG
    # -------------------------------------------------------------------------

    chunks = await asyncio.to_thread(
        rag.retrieve,
        extracted.claim,
    )

    if not chunks:
        return _insufficient(
            raw_text,
            input_type,
            extracted,
            "No relevant official document was retrieved for this claim, "
            "so it cannot be verified.",
        )

    # -------------------------------------------------------------------------
    # 3. Gemini verification + claim drift reasoning
    # -------------------------------------------------------------------------

    result = await verify_claim(
        extracted,
        chunks,
    )

    verdict = _parse_verdict(
        result.get("verdict")
    )

    evidence = _resolve_evidence(
        result.get("evidence"),
        chunks,
    )

    explanation = str(
        result.get("explanation") or ""
    ).strip()

    # A verdict without real retrieved evidence is not accepted.
    if (
        verdict != Verdict.INSUFFICIENT_EVIDENCE
        and not evidence
    ):
        verdict = Verdict.INSUFFICIENT_EVIDENCE

        explanation = (
            "The model could not ground its verdict in any retrieved "
            "official text. "
            + explanation
        )

    if verdict == Verdict.INSUFFICIENT_EVIDENCE:
        return _insufficient(
            raw_text,
            input_type,
            extracted,
            explanation
            or "The retrieved official documents do not address this claim.",
        )

    # -------------------------------------------------------------------------
    # 4. Gemini claim-drift analysis
    # -------------------------------------------------------------------------

    drift = _build_drift(
        result.get("claim_drift")
    )

    # -------------------------------------------------------------------------
    # 5. ML claim-drift analysis
    # -------------------------------------------------------------------------
    #
    # IMPORTANT:
    # The ML model receives:
    #
    #   official evidence + rumor claim
    #
    # It independently calculates probabilities for:
    #
    #   scope
    #   duration
    #   location
    #   context
    #
    # Gemini remains responsible for semantic reasoning.
    # ML provides an additional quantitative signal.
    # -------------------------------------------------------------------------

    official_text = "\n\n".join(
        evidence_item.text
        for evidence_item in evidence
    )

    ml_scores = _run_ml_drift(
        official_text=official_text,
        rumor_text=extracted.claim,
    )

    drift.ml_scores = ml_scores

    # -------------------------------------------------------------------------
    # 6. Generate moderator-ready correction
    # -------------------------------------------------------------------------

    corr = await generate_correction(
        extracted.claim,
        verdict,
        evidence,
        drift,
        explanation,
    )

    # -------------------------------------------------------------------------
    # 7. Return complete result
    # -------------------------------------------------------------------------

    return VerifyResponse(
        verdict=verdict,
        confidence=_clamp(
            result.get("confidence")
        ),
        claim=extracted.claim,
        input_type=input_type,
        source_text=raw_text,
        extracted_claim=extracted,
        evidence=evidence,
        contradictions=_str_list(
            result.get("contradictions")
        ),
        missing_context=_str_list(
            result.get("missing_context")
        ),
        claim_drift=drift,
        explanation=explanation,
        correction=str(
            corr.get("correction") or ""
        ).strip(),
        alternatives=_str_list(
            corr.get("alternatives")
        ),
        sources=sorted(
            {
                e.source
                for e in evidence
            }
        ),
    )
