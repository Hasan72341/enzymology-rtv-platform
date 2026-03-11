"""Health check endpoints."""
from fastapi import APIRouter
from datetime import datetime

from app.schemas.responses import HealthResponse
from app.services import model_service
from app import __version__

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=__version__
    )


@router.get("/ready", response_model=HealthResponse, tags=["Health"])
async def readiness_check():
    """Readiness check - verify models are loaded."""
    models_info = model_service.get_all_models_info()
    loaded_count = sum(1 for info in models_info.values() if info.get('loaded', False))
    
    return HealthResponse(
        status="ready" if loaded_count > 0 else "not ready",
        timestamp=datetime.utcnow(),
        models_loaded=loaded_count,
        version=__version__
    )
