"""API v1 router."""
from fastapi import APIRouter

from .endpoints import health, models, predictions, optimization

router = APIRouter()

# Include all endpoint routers
router.include_router(health.router)
router.include_router(models.router)
router.include_router(predictions.router)
router.include_router(optimization.router)
