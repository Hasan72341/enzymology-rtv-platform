"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils import setup_logging, get_logger
from app.services import model_service
from app.api.v1.router import router as api_v1_router
from app import __version__


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger = get_logger(__name__)
    logger.info("Starting application...")
    
    # Load models
    logger.info("Loading models...")
    model_service.load_models()
    logger.info("Models loaded successfully")
    
    # Load datasets for optimization
    logger.info("Loading datasets...")
    from app.services import data_service
    data_service.load_datasets()
    logger.info("Datasets loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="API for enzyme activity prediction and bioprocess optimization using ML models",
    lifespan=lifespan
)

# Add CORS middleware - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Cannot use credentials with wildcard origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the browser
)

# Include API v1 router
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs_url": "/docs",
        "api_v1": settings.api_v1_prefix
    }
