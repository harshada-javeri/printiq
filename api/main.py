"""
FastAPI application for PrintIQ inference service.

Usage:
    uvicorn api.main:app --reload
    # or
    python -m api.main
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from ..src.schema import HealthResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown logic."""
    logger.info("🚀 PrintIQ API starting up...")
    try:
        # Verify models are available on startup
        from .deps import get_model_container
        container = get_model_container()
        logger.info("✓ Models loaded and ready")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    
    yield
    
    logger.info("🛑 PrintIQ API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="PrintIQ",
    description="AI-Driven Print Failure & Quality Intelligence Platform",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware (allow all origins for demo, restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status of the service
    """
    return HealthResponse(status="ok")


@app.get("/", tags=["info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PrintIQ",
        "description": "AI-Driven Print Failure & Quality Intelligence Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict_failure": "POST /api/v1/predict/failure",
            "predict_quality": "POST /api/v1/predict/quality",
            "explain_failure": "POST /api/v1/explain/failure",
            "explain_quality": "POST /api/v1/explain/quality",
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if __debug__ else "An unexpected error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
