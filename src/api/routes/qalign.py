"""Q-Align API routes for video quality assessment."""

import base64
import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from ...schema import (
    ErrorResponse,
    QAlignInput,
    QAlignRequest,
    QAlignResponse,
)
from ...models.qalign import QAlignModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qalign", tags=["Q-Align"])

# Global model instance (initialized in app.py)
_model: QAlignModel | None = None


def get_model() -> QAlignModel:
    """Dependency to get the Q-Align model instance."""
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Q-Align model not initialized",
        )
    return _model


def set_model(model: QAlignModel) -> None:
    """Set the global model instance."""
    global _model
    _model = model


async def _load_video_from_request(request: QAlignRequest) -> bytes:
    """Load video from request (base64 or URL)."""
    if request.video_base64:
        try:
            video_data = base64.b64decode(request.video_base64)
            return video_data
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 video data: {e}",
            )

    elif request.video_url:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(request.video_url, timeout=120.0)
                response.raise_for_status()
                return response.content
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to fetch video from URL: {e}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid video data from URL: {e}",
            )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either video_base64 or video_url must be provided",
        )


@router.post(
    "/evaluate",
    response_model=QAlignResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Evaluate video quality",
    description="""
Evaluate the quality of a video.

**Input**:
- `video_base64`: Base64-encoded video data, OR
- `video_url`: URL of video to evaluate
- `task`: "quality" (default)

**Output**:
- `score`: Quality score on 1-5 scale

Score interpretation:
- 5: Excellent
- 4: Good
- 3: Fair
- 2: Poor
- 1: Bad
""",
)
async def evaluate_video(
    request: QAlignRequest,
    model: Annotated[QAlignModel, Depends(get_model)],
) -> QAlignResponse:
    """Evaluate video quality."""
    # Auto-load model if not loaded
    if not model.is_loaded:
        try:
            logger.info("Auto-loading Q-Align model...")
            model.load()
        except Exception as e:
            logger.error(f"Failed to load Q-Align model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load model: {e}",
            )

    # Load video from request
    video_data = await _load_video_from_request(request)

    # Create input and run prediction
    input_data = QAlignInput(video=video_data, task=request.task)

    result = model.predict(input_data)
    return QAlignResponse(
        score=result.score,
        task=result.task,
    )
