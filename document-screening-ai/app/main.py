"""FastAPI Application entrypoint for AI-Based Fake Identity & Document Screening System."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health, ocr, tampering
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan context."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Configured OCR engine: {settings.OCR_ENGINE}")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready OCR extraction and document verification microservice for the "
        "AI-Based Fake Identity & Document Screening System (SIH Project). "
        "Supports Passports (ICAO 9303 TD3 MRZ check-digit validation), Visas, and National IDs."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error occurred during document processing.",
            "detail": str(exc) if settings.DEBUG else "An error occurred. Please check server logs."
        }
    )

# Include Routers
app.include_router(health.router, prefix="")
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(ocr.router, prefix=settings.API_V1_PREFIX)
app.include_router(tampering.router, prefix=settings.API_V1_PREFIX)
app.include_router(tampering.router, prefix="")



@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "AI-Based Fake Identity & Document Screening - OCR Microservice is running.",
        "docs": "/docs",
        "health": "/health",
        "api_v1": f"{settings.API_V1_PREFIX}/ocr/extract"
    }
