"""Lightweight Gemini-powered RAG for RumorGuard AI."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

from config import settings
from models import AppError

logger = logging.getLogger("rumorguard.rag")

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

INDEX_FILE = settings.data_dir / "rag_index.json"


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    text: str
    similarity: float


def _get_client():
    from google import genai

    if not settings.gemini_api_key:
        raise AppError("GEMINI_API_KEY is not configured.", 503)

    return genai.Client(api_key=settings.gemini_api_key)


def _embed_documents(texts: list[str]) -> list[list[float]]:
    """Generate document embeddings using Gemini."""
    if not texts:
        return []

    from google.genai import types

    client = _get_client()

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768,
        ),
    )

    return [list(e.values) for e in result.embeddings]


def _embed_query(text: str) -> list[float]:
    """Generate a query embedding using Gemini."""
    from google.genai import types

    client = _get_client()

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[text],
        config=types.EmbedContentConfig(
            task_type="FACT_VERIFICATION",
            output_dimensionality=768,
        ),
    )

    return list(result.embeddings[0].values)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)


def _load_index() -> list[dict]:
    if not INDEX_FILE.exists():
        return []

    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Could not load RAG index")
        return []


def _save_index(items: list[dict]):
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    INDEX_FILE.write_text(
        json.dumps(items, ensure_ascii=False),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------------- #
# Reading + chunking
# ----------------------------------------------------------------------------- #

def read_document(path: Path) -> str:
    ext = path.suffix.lower()

    if ext == ".txt":
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    if ext == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))

            return "\n\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        except Exception as e:
            raise AppError(
                f"Could not read PDF '{path.name}': {e}",
                400,
            )

    raise AppError(
        f"Unsupported file type '{ext}'. Use .txt or .pdf",
        415,
    )


def _split_long(paragraph: str, max_chars: int) -> list[str]:

    out = []
    current = ""

    for sentence in re.split(
        r"(?<=[.!?])\s+",
        paragraph,
    ):

        while len(sentence) > max_chars:

            if current:
                out.append(current)
                current = ""

            out.append(sentence[:max_chars])
            sentence = sentence[max_chars:]

        if current and len(current) + len(sentence) + 1 > max_chars:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        out.append(current)

    return out


def chunk_text(
    text: str,
    max_chars: int | None = None,
) -> list[str]:

    max_chars = max_chars or settings.chunk_size

    chunks = []

    for section in re.split(
        r"(?m)^\s*-{3,}\s*$",
        text.replace("\r\n", "\n"),
    ):

        units = []

        for para in re.split(
            r"\n\s*\n",
            section,
        ):

            para = re.sub(
                r"\s*\n\s*",
                " ",
                re.sub(r"[ \t]+", " ", para),
            ).strip()

            if not para:
                continue

            if len(para) <= max_chars:
                units.append(para)
            else:
                units.extend(
                    _split_long(
                        para,
                        max_chars,
                    )
                )

        current = ""

        for unit in units:

            if (
                current
                and len(current) + len(unit) + 2 > max_chars
            ):
                chunks.append(current)
                current = unit

            else:

                current = (
                    f"{current}\n\n{unit}"
                    if current
                    else unit
                )

        if current:
            chunks.append(current)

    return chunks


# ----------------------------------------------------------------------------- #
# Ingestion
# ----------------------------------------------------------------------------- #

def ingest_file(path: Path) -> int:

    chunks = chunk_text(
        read_document(path)
    )

    if not chunks:
        raise AppError(
            f"No text could be extracted from '{path.name}'.",
            422,
        )

    embeddings = _embed_documents(chunks)

    index = _load_index()

    # Remove old chunks from this source
    index = [
        item
        for item in index
        if item["source"] != path.name
    ]

    for i, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        index.append(
            {
                "chunk_id": f"{path.name}::{i}",
                "source": path.name,
                "text": chunk,
                "embedding": embedding,
            }
        )

    _save_index(index)

    logger.info(
        "Indexed %s (%d chunks)",
        path.name,
        len(chunks),
    )

    return len(chunks)


def _kb_files() -> list[Path]:

    settings.knowledge_base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sorted(
        p
        for p in settings.knowledge_base_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def ingest_directory() -> dict:

    files = _kb_files()

    result = {
        "indexed": {},
        "errors": {},
        "removed": [],
    }

    for path in files:

        try:

            result["indexed"][path.name] = ingest_file(path)

        except Exception as e:

            logger.exception(
                "Failed to index %s",
                path.name,
            )

            result["errors"][path.name] = str(e)

    index = _load_index()

    on_disk = {
        p.name
        for p in files
    }

    removed_sources = {
        item["source"]
        for item in index
    } - on_disk

    if removed_sources:

        index = [
            item
            for item in index
            if item["source"] not in removed_sources
        ]

        _save_index(index)

        result["removed"] = list(
            removed_sources
        )

    return result


def list_documents() -> list[dict]:

    index = _load_index()

    counts = {}

    for item in index:

        source = item["source"]

        counts[source] = (
            counts.get(source, 0) + 1
        )

    docs = []

    for path in _kb_files():

        count = counts.get(
            path.name,
            0,
        )

        docs.append(
            {
                "name": path.name,
                "chunks": count,
                "indexed": count > 0,
                "size_bytes": path.stat().st_size,
            }
        )

    return docs


# ----------------------------------------------------------------------------- #
# Retrieval
# ----------------------------------------------------------------------------- #

def retrieve(
    query: str,
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> list[RetrievedChunk]:

    top_k = top_k or settings.top_k

    min_similarity = (
        settings.min_similarity
        if min_similarity is None
        else min_similarity
    )

    index = _load_index()

    if not index:
        return []

    query_embedding = _embed_query(query)

    scored = []

    for item in index:

        similarity = _cosine(
            query_embedding,
            item["embedding"],
        )

        if similarity >= min_similarity:

            scored.append(
                RetrievedChunk(
                    chunk_id=item["chunk_id"],
                    source=item["source"],
                    text=item["text"],
                    similarity=round(
                        similarity,
                        4,
                    ),
                )
            )

    scored.sort(
        key=lambda x: x.similarity,
        reverse=True,
    )

    return scored[:top_k]
