"""Image -> text using Tesseract OCR."""
import io
import re

from config import settings
from models import AppError


def extract_text_from_image(data: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except ImportError as e:  # pragma: no cover
        raise AppError(f"OCR dependencies missing: {e}", 500)

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise AppError("Could not read the uploaded image. Use PNG/JPG/WEBP.", 400)

    if img.width < 1000:  # small screenshots OCR much better when upscaled
        img = img.resize((img.width * 2, img.height * 2))

    try:
        text = pytesseract.image_to_string(img, lang=settings.ocr_lang)
    except pytesseract.TesseractNotFoundError:
        raise AppError(
            "Tesseract is not installed or not on PATH. Install it (see README) or set TESSERACT_CMD.", 503
        )
    except Exception as e:
        raise AppError(f"OCR failed: {e}", 500)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise AppError("No readable text was found in the image.", 422)
    return text
