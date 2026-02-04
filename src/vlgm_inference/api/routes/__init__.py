"""API route modules."""

from vlgm_inference.api.routes.qalign import router as qalign_router
from vlgm_inference.api.routes.turbodiffusion import router as turbodiffusion_router

__all__ = ["qalign_router", "turbodiffusion_router"]
