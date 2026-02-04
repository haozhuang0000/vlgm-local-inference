"""TurboDiffusion API routes for video generation."""

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from vlgm_inference.schema import (
    ErrorResponse,
    TurboDiffusionRequest,
    TurboDiffusionResponse,
)
from vlgm_inference.models.turbodiffusion import TurboDiffusionModel
from vlgm_inference.schema import TurboDiffusionInput
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


@router.post(
    "/generate",
    response_model=TurboDiffusionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Generate video from text",
    description="""
Generate a video from a text prompt.

**Input**:
- `prompt`: Text description of the video to generate (required)
- `negative_prompt`: Elements to avoid (optional)
- `num_steps`: Inference steps, 4 recommended (default: 4)
- `guidance_scale`: Prompt adherence strength (default: 7.5)
- `width`: Output width in pixels (default: 832)
- `height`: Output height in pixels (default: 480)
- `num_frames`: Number of frames (default: 81)
- `fps`: Frames per second (default: 16)
- `seed`: Random seed for reproducibility (optional)

**Output**:
- `video_url`: URL to download the generated video
- `video_path`: Server path to the video file
- `metadata`: Generation parameters used
""",
)
async def generate_video(
    request: TurboDiffusionRequest,
    model: Annotated[TurboDiffusionModel, Depends(get_model)],
) -> TurboDiffusionResponse:
    """Generate a video from text prompt."""
    # Auto-load model if not loaded
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

    input_data = TurboDiffusionInput(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_steps=request.num_steps,
        guidance_scale=request.guidance_scale,
        width=request.width,
        height=request.height,
        num_frames=request.num_frames,
        fps=request.fps,
        seed=request.seed,
    )

    try:
        result = model.predict(input_data)

        # Generate URL for video download
        video_filename = Path(result.video_path).name
        video_url = f"/outputs/{video_filename}"

        return TurboDiffusionResponse(
            video_url=video_url,
            video_path=result.video_path,
            metadata=result.metadata,
        )
    except Exception as e:
        logger.error(f"TurboDiffusion generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {e}",
        )
