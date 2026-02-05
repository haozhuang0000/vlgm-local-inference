"""TurboDiffusion API routes for video generation."""

import base64
import logging
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from ...schema import (
    ErrorResponse,
    TurboDiffusionInput,
    TurboDiffusionRequest,
    TurboDiffusionResponse,
)
from ...models.turbodiffusion import TurboDiffusionModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/turbodiffusion", tags=["TurboDiffusion"])

# Global model instance (initialized in app.py)
_model: TurboDiffusionModel | None = None


def get_model() -> TurboDiffusionModel:
    """Dependency to get the TurboDiffusion model instance."""
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TurboDiffusion model not initialized",
        )
    return _model


def set_model(model: TurboDiffusionModel) -> None:
    """Set the global model instance."""
    global _model
    _model = model


async def _load_image_from_request(request: TurboDiffusionRequest) -> bytes | None:
    """Decode image from base64 or fetch from URL. Returns None if neither provided."""
    if request.image_base64:
        try:
            return base64.b64decode(request.image_base64)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 image data: {e}",
            )

    if request.image_url:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(request.image_url, timeout=60.0)
                response.raise_for_status()
                return response.content
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to fetch image from URL: {e}",
            )

    return None


@router.post(
    "/generate",
    response_model=TurboDiffusionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Generate video from text or image",
    description="""
Generate a video using TurboDiffusion (rCM + SLA, 100-200x faster than standard diffusion).

**Text-to-Video**: provide only `prompt`.

**Image-to-Video**: provide `prompt` plus `image_base64` or `image_url`.
*(Requires the Wan2.2-A14B model to be configured on the server.)*

**Input**:
- `prompt`: Text description (required, prefer long English prompts)
- `num_steps`: Sampling steps 1–4 (default 4)
- `num_frames`: Frames to generate (default 81)
- `resolution`: `480p` or `720p` (default `480p`)
- `aspect_ratio`: e.g. `16:9` (default `16:9`)
- `seed`: Random seed (optional)
- `image_base64` / `image_url`: Input image for I2V (optional)

**Output**:
- `video_url`: Download path for the generated video
- `video_path`: Server-side file path
- `metadata`: Generation parameters used
""",
)
async def generate_video(
    request: TurboDiffusionRequest,
    model: Annotated[TurboDiffusionModel, Depends(get_model)],
) -> TurboDiffusionResponse:
    """Generate a video from text prompt (and optionally an image)."""
    # Auto-load model on first request
    if not model.is_loaded:
        try:
            logger.info("Auto-loading TurboDiffusion model...")
            model.load()
        except Exception as e:
            logger.error(f"Failed to load TurboDiffusion model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load model: {e}",
            )

    # Resolve image for I2V
    image_data = await _load_image_from_request(request)
    task = "image2video" if image_data else "text2video"

    input_data = TurboDiffusionInput(
        prompt=request.prompt,
        num_steps=request.num_steps,
        num_frames=request.num_frames,
        seed=request.seed,
        resolution=request.resolution,
        aspect_ratio=request.aspect_ratio,
        task=task,
        image=image_data,
    )

    try:
        result = model.predict(input_data)

        video_filename = Path(result.video_path).name
        video_url = f"/outputs/{video_filename}"
        video_base64 = base64.b64encode(Path(result.video_path).read_bytes()).decode("ascii")

        return TurboDiffusionResponse(
            video_url=video_url,
            video_path=result.video_path,
            video_base64=video_base64,
            metadata=result.metadata,
        )
    except Exception as e:
        logger.error(f"TurboDiffusion generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {e}",
        )
