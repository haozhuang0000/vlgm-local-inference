"""FastVideo API routes for accelerated video generation."""

import base64
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ...schema import (
    ErrorResponse,
    FastVideoInput,
    FastVideoRequest,
    FastVideoResponse,
)
from ...models.fastvideo import FastVideoModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fastvideo", tags=["FastVideo"])

# Global model instance (initialized in app.py)
_model: FastVideoModel | None = None


def get_model() -> FastVideoModel:
    """Dependency to get the FastVideo model instance."""
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FastVideo model not initialized",
        )
    return _model


def set_model(model: FastVideoModel) -> None:
    """Set the global model instance."""
    global _model
    _model = model


@router.post(
    "/generate",
    response_model=FastVideoResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Generate video from text",
    description="""
Generate a video from a text prompt using FastVideo's distilled diffusion models.

**Input**:
- `prompt`: Text description of the video to generate
- `num_frames`: Number of frames (default: 49)
- `height`: Video height in pixels (default: 480)
- `width`: Video width in pixels (default: 832)
- `num_inference_steps`: Diffusion steps (default: 8, recommended for distilled models)
- `guidance_scale`: CFG scale (default: 1.0)
- `seed`: Random seed for reproducibility (optional)

**Output**:
- `video_url`: URL to download the generated video
- `video_path`: Server path to the video file
- `video_base64`: Base64-encoded MP4 data
- `metadata`: Generation parameters
""",
)
async def generate_video(
    request: FastVideoRequest,
    model: Annotated[FastVideoModel, Depends(get_model)],
) -> FastVideoResponse:
    """Generate video from text prompt."""
    # Auto-load model if not loaded
    if not model.is_loaded:
        try:
            logger.info("Auto-loading FastVideo model...")
            model.load()
        except Exception as e:
            logger.error(f"Failed to load FastVideo model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load model: {e}",
            )

    # Create input and run prediction
    input_data = FastVideoInput(
        prompt=request.prompt,
        num_frames=request.num_frames,
        height=request.height,
        width=request.width,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
    )

    try:
        result = model.predict(input_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video generation failed: {e}",
        )

    # Read and encode video
    try:
        with open(result.video_path, "rb") as f:
            video_bytes = f.read()
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to read generated video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read generated video: {e}",
        )

    # Build video URL (relative to outputs mount)
    video_filename = result.video_path.split("/")[-1].split("\\")[-1]
    video_url = f"/outputs/{video_filename}"

    return FastVideoResponse(
        video_url=video_url,
        video_path=result.video_path,
        video_base64=video_base64,
        metadata=result.metadata,
    )
