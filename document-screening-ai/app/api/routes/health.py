"""Health check route."""

from fastapi import APIRouter
from app.config import settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns service health status and configured engines."""
    available = ["mock"]
    try:
        import pytesseract
        available.append("tesseract")
    except ImportError:
        pass
    try:
        import paddleocr
        available.append("paddleocr")
    except ImportError:
        pass

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        service=settings.APP_NAME,
        available_ocr_engines=available
    )
