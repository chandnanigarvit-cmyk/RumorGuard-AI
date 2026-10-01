"""RAG: ingest trusted official documents into ChromaDB and retrieve evidence for a claim."""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from config import settings
from models import AppError

logger = logging.getLogger("rumorguard.rag")

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}
_lock = threading.RLock()
_model = None
_collection = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    text: str
    similarity: float


# ----------------------------------------------------------------------------- #
# Lazy singletons
# ----------------------------------------------------------------------------- #
def _get_model():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s ...", settings.embedding_model)
            _model = SentenceTransformer(settings.embedding_model)
    return _model


def get_collection():
    global _collection
    with _lock:
        if _collection is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(settings.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
            )
            _collection = client.get_or_create_collection(
                name=settings.collection_name, metadata={"hnsw:space": "cosine"}
            )
    return _collection


def _embed(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


# ----------------------------------------------------------------------------- #
# Reading + chunking
# ----------------------------------------------------------------------------- #
def read_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            raise AppError(f"Could not read PDF '{path.name}': {e}", 400)
    raise AppError(f"Unsupported file type '{ext}'. Use .txt or .pdf", 415)


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries (hard-split as a last resort)."""
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sent) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.append(sent[:max_chars])
            sent = sent[max_chars:]
        if cur and len(cur) + len(sent) + 1 > max_chars:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, max_chars: int | None = None) -> list[str]:
    """Paragraph-aware chunking. A line containing only '---' forces a chunk boundary
    (use it to separate distinct notices inside one file)."""
    max_chars = max_chars or settings.chunk_size
    chunks: list[str] = []
    for section in re.split(r"(?m)^\s*-{3,}\s*$", text.replace("\r\n", "\n")):
        units: list[str] = []
        for para in re.split(r"\n\s*\n", section):
            para = re.sub(r"\s*\n\s*", " ", re.sub(r"[ \t]+", " ", para)).strip()
            if not para:
                continue
            units.extend([para] if len(para) <= max_chars else _split_long(para, max_chars))
        cur = ""
        for unit in units:
            if cur and len(cur) + len(unit) + 2 > max_chars:
                chunks.append(cur)
                cur = unit
            else:
                cur = f"{cur}\n\n{unit}" if cur else unit
        if cur:
            chunks.append(cur)
    return chunks


# ----------------------------------------------------------------------------- #
# Ingestion
# ----------------------------------------------------------------------------- #
def ingest_file(path: Path) -> int:
    """(Re)index one document. Returns number of chunks stored."""
    chunks = chunk_text(read_document(path))
    if not chunks:
        raise AppError(f"No text could be extracted from '{path.name}'.", 422)
    embeddings = _embed(chunks)
    col = get_collection()
    with _lock:
        col.delete(where={"source": path.name})
        col.add(
            ids=[f"{path.name}::{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": path.name, "chunk_index": i} for i in range(len(chunks))],
        )
    logger.info("Indexed %s (%d chunks)", path.name, len(chunks))
    return len(chunks)


def _kb_files() -> list[Path]:
    settings.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in settings.knowledge_base_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def ingest_directory() -> dict:
    """Index every TXT/PDF in knowledge_base/ and drop index entries for deleted files."""
    files = _kb_files()
    result = {"indexed": {}, "errors": {}, "removed": []}
    for path in files:
        try:
            result["indexed"][path.name] = ingest_file(path)
        except Exception as e:  # keep going if one file is bad
            logger.exception("Failed to index %s", path.name)
            result["errors"][path.name] = str(getattr(e, "message", e))
    on_disk = {p.name for p in files}
    col = get_collection()
    for source in {m["source"] for m in (col.get(include=["metadatas"])["metadatas"] or [])} - on_disk:
        col.delete(where={"source": source})
        result["removed"].append(source)
    return result


def list_documents() -> list[dict]:
    counts: dict[str, int] = {}
    for meta in get_collection().get(include=["metadatas"])["metadatas"] or []:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    docs = []
    for path in _kb_files():
        docs.append(
            {
                "name": path.name,
                "chunks": counts.get(path.name, 0),
                "indexed": counts.get(path.name, 0) > 0,
                "size_bytes": path.stat().st_size,
            }
        )
    return docs


# ----------------------------------------------------------------------------- #
# Retrieval
# ----------------------------------------------------------------------------- #
def retrieve(query: str, top_k: int | None = None, min_similarity: float | None = None) -> list[RetrievedChunk]:
    """Return the most relevant official chunks (cosine similarity >= threshold)."""
    top_k = top_k or settings.top_k
    min_similarity = settings.min_similarity if min_similarity is None else min_similarity
    col = get_collection()
    total = col.count()
    if total == 0:
        return []
    res = col.query(
        query_embeddings=_embed([query]),
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = round(1.0 - float(dist), 4)
        if sim >= min_similarity:
            out.append(RetrievedChunk(chunk_id=cid, source=meta["source"], text=doc, similarity=sim))
    return out
