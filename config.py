"""Central configuration. All secrets come from environment variables / .env."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # LLM
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.5-flash")
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 1500)

    # Paths
    knowledge_base_dir: Path = BASE_DIR / "knowledge_base"
    data_dir: Path = BASE_DIR / "data"
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    collection_name: str = "official_documents"

    # RAG
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = _int("CHUNK_SIZE", 700)
    top_k: int = _int("TOP_K", 4)
    min_similarity: float = _float("MIN_SIMILARITY", 0.30)

    # Speech / OCR
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    ocr_lang: str = os.getenv("OCR_LANG", "eng")
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")

    # API
    cors_origins: tuple = tuple(
        o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
    )
    admin_token: str = os.getenv("ADMIN_TOKEN", "")
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 25)
    max_input_chars: int = 4000


settings = Settings()
