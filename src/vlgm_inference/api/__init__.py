"""FastAPI application for VLGM inference service."""

from vlgm_inference.api.app import app, create_app

__all__ = ["app", "create_app"]
