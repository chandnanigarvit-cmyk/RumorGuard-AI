"""RumorGuard AI - FastAPI backend.  Run:  uvicorn main:app --reload"""
import asyncio
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import ocr
import rag
import speech
import verifier
from config import settings
from models import (
    AppError, DocumentInfo, DocumentsResponse, IngestResult, VerifyResponse, VerifyTextRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rumorguard")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".webm", ".flac", ".aac", ".mp4", ".3gp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        result = await asyncio.to_thread(rag.ingest_directory)
        logger.info("Knowledge base ready: %s", result)
    except Exception:
        logger.exception("Knowledge base indexing failed at startup (POST /documents/reindex to retry)")
    yield


app = FastAPI(
    title="RumorGuard AI",
    description="Rumor verification, claim-drift detection and evidence-backed corrections using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,  # must be False when origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def _real_file(f: UploadFile | None) -> UploadFile | None:
    """Browsers send an empty file part for untouched <input type=file>; treat it as absent."""
    return f if f is not None and f.filename else None


def detect_input_type(file: UploadFile | None, text: str | None) -> str:
    if file is not None:
        ctype = (file.content_type or "").lower()
        ext = Path(file.filename or "").suffix.lower()
        if ctype.startswith("image/") or ext in IMAGE_EXT:
            return "image"
        if ctype.startswith("audio/") or ctype.startswith("video/") or ext in AUDIO_EXT:
            return "audio"
        raise AppError(f"Unsupported file type '{ctype or ext}'. Send an image or audio file.", 415)
    if text and text.strip():
        return "text"
    raise AppError("Provide 'text', or an image/audio file.", 400)


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise AppError("Uploaded file is empty.", 400)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise AppError(f"File too large (max {settings.max_upload_mb} MB).", 413)
    return data


def require_admin(x_admin_token: str | None = Header(default=None)):
    if settings.admin_token and x_admin_token != settings.admin_token:
        raise AppError("Invalid or missing X-Admin-Token header.", 401)


# ----------------------------------------------------------------------------- #
# routes
# ----------------------------------------------------------------------------- #
@app.get("/health", tags=["system"])
async def health():
    try:
        chunks = await asyncio.to_thread(lambda: rag.get_collection().count())
    except Exception:
        chunks = None
    return {
        "status": "ok",
        "llm_configured": bool(settings.anthropic_api_key),
        "llm_model": settings.llm_model,
        "indexed_chunks": chunks,
    }


@app.post("/verify", response_model=VerifyResponse, tags=["verify"])
async def verify(
    text: str | None = Form(default=None, description="Rumor text"),
    file: UploadFile | None = File(default=None, description="Screenshot (image) or voice message (audio)"),
    image: UploadFile | None = File(default=None, description="Alias for 'file' (image)"),
    audio: UploadFile | None = File(default=None, description="Alias for 'file' (audio)"),
):
    """Multipart endpoint. Send `text`, or an image / audio file. The input type is detected automatically.
    If both text and a file are sent, the OCR/transcript and the text are combined."""
    upload = _real_file(file) or _real_file(image) or _real_file(audio)
    input_type = detect_input_type(upload, text)

    if input_type == "text":
        raw_text = text.strip()
    else:
        data = await _read_upload(upload)
        if input_type == "image":
            raw_text = await asyncio.to_thread(ocr.extract_text_from_image, data)
        else:
            raw_text = await asyncio.to_thread(speech.transcribe_audio, data, upload.filename or "audio.wav")
        if text and text.strip():
            raw_text = f"{raw_text}\n{text.strip()}"

    return await verifier.run_pipeline(raw_text, input_type)


@app.post("/verify/text", response_model=VerifyResponse, tags=["verify"])
async def verify_text(body: VerifyTextRequest):
    """JSON convenience endpoint: {"text": "..."}"""
    return await verifier.run_pipeline(body.text.strip(), "text")


@app.get("/documents", response_model=DocumentsResponse, tags=["documents"])
async def list_documents():
    docs = await asyncio.to_thread(rag.list_documents)
    return DocumentsResponse(count=len(docs), documents=[DocumentInfo(**d) for d in docs])


@app.post("/documents", response_model=IngestResult, tags=["documents"], dependencies=[Depends(require_admin)])
async def add_document(file: UploadFile = File(..., description="Official document (.txt or .pdf)")):
    """Upload a trusted official document; it is saved to knowledge_base/ and indexed immediately.
    Uploading a file with an existing name replaces it."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "").name)
    if Path(name).suffix.lower() not in rag.SUPPORTED_EXTENSIONS:
        raise AppError("Only .txt and .pdf documents are supported.", 415)
    data = await _read_upload(file)
    settings.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    path = settings.knowledge_base_dir / name
    path.write_bytes(data)
    try:
        chunks = await asyncio.to_thread(rag.ingest_file, path)
    except Exception:
        path.unlink(missing_ok=True)  # don't leave an unindexable file behind
        raise
    return IngestResult(name=name, chunks=chunks, message="Document indexed.")


@app.post("/documents/reindex", tags=["documents"], dependencies=[Depends(require_admin)])
async def reindex_documents():
    """Re-scan knowledge_base/ and rebuild the index."""
    return await asyncio.to_thread(rag.ingest_directory)
