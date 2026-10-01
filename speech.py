"""Audio transcription using Gemini."""

import logging
import mimetypes

from config import settings
from models import AppError

logger = logging.getLogger("rumorguard.speech")


def transcribe_audio(data: bytes, filename: str = "audio.wav") -> str:
    """
    Transcribe uploaded audio using Gemini 3.5 Transcribe.

    The audio is uploaded to Gemini's Files API and then
    passed to the transcription model.
    """

    if not settings.gemini_api_key:
        raise AppError(
            "GEMINI_API_KEY is not configured.",
            503,
        )

    if not data:
        raise AppError(
            "Audio file is empty.",
            400,
        )

    try:
        from google import genai

        client = genai.Client(
            api_key=settings.gemini_api_key
        )

        mime_type = mimetypes.guess_type(filename)[0]

        if not mime_type:
            mime_type = "audio/wav"

        # Gemini expects audio MIME types.
        if not mime_type.startswith("audio/"):
            mime_type = "audio/wav"

        logger.info(
            "Uploading audio for transcription: %s (%s)",
            filename,
            mime_type,
        )

        # Write the uploaded bytes to a temporary file because
        # the Gemini Files API accepts a file path.
        import os
        import tempfile

        suffix = ""
        if "." in filename:
            suffix = "." + filename.rsplit(".", 1)[-1]

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp:
                temp.write(data)
                temp_path = temp.name

            audio_file = client.files.upload(
                file=temp_path,
                config={
                    "mime_type": mime_type,
                },
            )

            response = client.models.generate_content(
                model="gemini-3.5-transcribe",
                contents=[audio_file],
            )

            text = (response.text or "").strip()

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

        if not text:
            raise AppError(
                "No speech was detected in the audio.",
                422,
            )

        logger.info(
            "Audio transcription completed successfully."
        )

        return text

    except AppError:
        raise

    except Exception as exc:
        logger.exception(
            "Gemini audio transcription failed."
        )
        raise AppError(
            f"Audio transcription failed: {exc}",
            422,
        )
