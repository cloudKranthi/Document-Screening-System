"""Ultra-lightweight health check route for orchestrator and load balancer probes."""

from fastapi import APIRouter
from app.config import settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["Health"])

# Precomputed static health response for sub-millisecond, non-blocking health checks
STATIC_HEALTH_RESPONSE = HealthResponse(
    status="ok",
    version=settings.APP_VERSION,
    service=settings.APP_NAME,
    available_ocr_engines=["mock", "tesseract"]
)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Returns static service health status immediately without blocking operations."""
    return STATIC_HEALTH_RESPONSE
