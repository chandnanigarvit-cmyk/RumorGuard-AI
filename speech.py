"""Audio -> text using Whisper (faster-whisper implementation; bundles its own audio decoder)."""
import logging
import os
import tempfile
import threading
from pathlib import Path

from config import settings
from models import AppError

logger = logging.getLogger("rumorguard.speech")
_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel

            logger.info("Loading Whisper model '%s' (first run downloads it)...", settings.whisper_model_size)
            _model = WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8")
    return _model


def transcribe_audio(data: bytes, filename: str = "audio.wav") -> str:
    suffix = Path(filename).suffix or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        segments, _info = _get_model().transcribe(tmp_path, beam_size=1, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except AppError:
        raise
    except Exception as e:
        raise AppError(f"Audio transcription failed: {e}", 422)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    if not text:
        raise AppError("No speech was detected in the audio.", 422)
    return text
