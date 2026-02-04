"""FastAPI application for VLGM local inference service."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from vlgm_inference import __version__
from vlgm_inference.api.routes import qalign_router, turbodiffusion_router
from vlgm_inference.api.routes.qalign import set_model as set_qalign_model
from vlgm_inference.api.routes.turbodiffusion import set_model as set_turbodiffusion_model
from vlgm_inference.config import get_settings
from vlgm_inference.models.qalign import QAlignModel
from vlgm_inference.models.turbodiffusion import TurboDiffusionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    settings = get_settings()

    # Create output directory
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models (lazy loading - loaded on first request)
    logger.info("Initializing models...")

    if settings.qalign_enabled:
        qalign_model = QAlignModel(
            model_id=settings.qalign_model_id,
            device=settings.device,
            dtype=settings.torch_dtype,
        )
        set_qalign_model(qalign_model)
        logger.info("Q-Align model initialized (will load on first request)")

    if settings.turbodiffusion_enabled:
        turbodiffusion_model = TurboDiffusionModel(
            model_name=settings.turbodiffusion_model,
            checkpoint_dir=settings.turbodiffusion_checkpoint_dir,
            device=settings.device,
            dtype=settings.torch_dtype,
            output_dir=settings.output_dir,
        )
        set_turbodiffusion_model(turbodiffusion_model)
        logger.info("TurboDiffusion model initialized (will load on first request)")

    logger.info("VLGM Inference Service ready")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down VLGM Inference Service...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="VLGM Local Inference Service",
        description="""
## Visual-Language Generative Models Local Inference Service

This service provides local inference for:

### Q-Align (`/qalign/evaluate`)
Image quality and aesthetic assessment. Returns a score from 1-5.

### TurboDiffusion (`/turbodiffusion/generate`)
Fast text-to-video generation with 100-200x acceleration.

Models are loaded automatically on first request.
        """,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files for output directory
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(output_dir)), name="outputs")

    # Include routers
    if settings.qalign_enabled:
        app.include_router(qalign_router)
    if settings.turbodiffusion_enabled:
        app.include_router(turbodiffusion_router)

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API info."""
        return {
            "service": "VLGM Local Inference Service",
            "version": __version__,
            "endpoints": {
                "qalign": "/qalign/evaluate",
                "turbodiffusion": "/turbodiffusion/generate",
            },
            "docs": "/docs",
        }

    return app


# Create app instance
app = create_app()


def main():
    """Entry point for running the server."""
    settings = get_settings()
    uvicorn.run(
        "vlgm_inference.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
