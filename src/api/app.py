"""FastAPI application for VLGM local inference service."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — each model's transitive deps may not be installed in every
# container.  Missing deps produce a warning at startup, not a crash.
# ---------------------------------------------------------------------------
_qalign_available = False
_turbodiffusion_available = False
_fastvideo_available = False

try:
    from .routes.qalign import router as qalign_router, set_model as set_qalign_model
    from ..models.qalign import QAlignModel
    _qalign_available = True
except ImportError:
    pass

try:
    from .routes.turbodiffusion import router as turbodiffusion_router, set_model as set_turbodiffusion_model
    from ..models.turbodiffusion import TurboDiffusionModel
    _turbodiffusion_available = True
except ImportError:
    pass

try:
    from .routes.fastvideo import router as fastvideo_router, set_model as set_fastvideo_model
    from ..models.fastvideo_local import FastVideoModel
    _fastvideo_available = True
except ImportError:
    pass


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
        if _qalign_available:
            qalign_model = QAlignModel(
                model_id=settings.qalign_model_id,
                device=settings.device,
                dtype=settings.torch_dtype,
            )
            set_qalign_model(qalign_model)
            logger.info("Q-Align model initialized (will load on first request)")
        else:
            logger.warning("Q-Align enabled but dependencies are not installed — skipping")

    if settings.turbodiffusion_enabled:
        if _turbodiffusion_available:
            turbodiffusion_model = TurboDiffusionModel(
                model_name=settings.turbodiffusion_model,
                dit_path=settings.turbodiffusion_dit_path,
                vae_path=settings.turbodiffusion_vae_path,
                text_encoder_path=settings.turbodiffusion_text_encoder_path,
                attention_type=settings.turbodiffusion_attention_type,
                sla_topk=settings.turbodiffusion_sla_topk,
                quant_linear=settings.turbodiffusion_quant_linear,
                default_norm=settings.turbodiffusion_default_norm,
                output_dir=settings.output_dir,
                high_noise_model_path=settings.turbodiffusion_high_noise_model_path,
                low_noise_model_path=settings.turbodiffusion_low_noise_model_path,
                boundary=settings.turbodiffusion_boundary,
            )
            set_turbodiffusion_model(turbodiffusion_model)
            logger.info("TurboDiffusion model initialized (will load on first request)")
        else:
            logger.warning("TurboDiffusion enabled but dependencies are not installed — skipping")

    if settings.fastvideo_enabled:
        if _fastvideo_available:
            fastvideo_model = FastVideoModel(
                model_id=settings.fastvideo_model_id,
                device=settings.device,
                dtype=settings.torch_dtype,
                output_dir=settings.output_dir,
            )
            set_fastvideo_model(fastvideo_model)
            logger.info("FastVideo model initialized (will load on first request)")
        else:
            logger.warning("FastVideo enabled but dependencies are not installed — skipping")

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

This service provides local inference for visual-language generative models.
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

    # Include routers — only when deps are available
    if settings.qalign_enabled and _qalign_available:
        app.include_router(qalign_router)
    if settings.turbodiffusion_enabled and _turbodiffusion_available:
        app.include_router(turbodiffusion_router)
    if settings.fastvideo_enabled and _fastvideo_available:
        app.include_router(fastvideo_router)

    @app.get("/health", tags=["Health"])
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API info."""
        endpoints = {}
        if settings.qalign_enabled and _qalign_available:
            endpoints["qalign"] = "/qalign/evaluate"
        if settings.turbodiffusion_enabled and _turbodiffusion_available:
            endpoints["turbodiffusion"] = "/turbodiffusion/generate"
        if settings.fastvideo_enabled and _fastvideo_available:
            endpoints["fastvideo"] = "/fastvideo/generate"
        return {
            "service": "VLGM Local Inference Service",
            "version": __version__,
            "endpoints": endpoints,
            "docs": "/docs",
        }

    return app


# Create app instance
app = create_app()


def main():
    """Entry point for running the server."""
    settings = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
