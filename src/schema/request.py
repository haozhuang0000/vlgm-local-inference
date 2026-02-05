"""Pydantic schemas for API request/response models."""

from typing import Literal
from pydantic import BaseModel, Field


# ============================================================================
# Common Schemas
# ============================================================================


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Additional details")


# ============================================================================
# Q-Align Schemas
# ============================================================================


class QAlignRequest(BaseModel):
    """
    Request for Q-Align video quality evaluation.

    Provide either `video_base64` or `video_url` (not both).
    """

    video_base64: str | None = Field(
        default=None,
        description="Base64-encoded video data",
    )
    video_url: str | None = Field(
        default=None,
        description="URL of video to evaluate",
    )
    task: Literal["quality"] = Field(
        default="quality",
        description="Evaluation task: 'quality' for VQA",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "video_url": "https://example.com/video.mp4",
                    "task": "quality",
                },
            ]
        }
    }


class QAlignResponse(BaseModel):
    """
    Response from Q-Align video quality evaluation.

    Scores are on a 1-5 scale:
    - 5: Excellent
    - 4: Good
    - 3: Fair
    - 2: Poor
    - 1: Bad
    """

    score: float = Field(
        ge=1.0,
        le=5.0,
        description="Quality score (1-5 scale)",
    )
    task: str = Field(description="Evaluation task performed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "score": 4.23,
                    "task": "quality",
                }
            ]
        }
    }


# ============================================================================
# TurboDiffusion Schemas
# ============================================================================


class TurboDiffusionRequest(BaseModel):
    """
    Request for TurboDiffusion video generation.

    For best results:
    - Use long, descriptive English prompts
    - Set num_steps to 4 for best quality
    - Provide image_base64 or image_url to trigger image-to-video mode
    """

    prompt: str = Field(
        min_length=1,
        max_length=500,
        description="Text description of the video to generate",
    )
    num_steps: int = Field(
        default=4,
        ge=1,
        le=4,
        description="Inference steps (1-4, 4 recommended)",
    )
    num_frames: int = Field(
        default=81,
        ge=1,
        le=200,
        description="Number of frames to generate",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Random seed for reproducibility",
    )
    resolution: str = Field(
        default="480p",
        description="Output resolution: 480p or 720p",
    )
    aspect_ratio: str = Field(
        default="16:9",
        description="Aspect ratio (width:height)",
    )
    # image-to-video (provide one of these to switch to I2V mode)
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded input image (triggers image-to-video)",
    )
    image_url: str | None = Field(
        default=None,
        description="URL of input image (triggers image-to-video)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage",
                    "num_steps": 4,
                    "num_frames": 81,
                    "resolution": "480p",
                    "aspect_ratio": "16:9",
                }
            ]
        }
    }


class TurboDiffusionResponse(BaseModel):
    """Response from TurboDiffusion video generation."""

    video_url: str = Field(description="URL to download the generated video")
    video_path: str = Field(description="Server path to the generated video")
    video_base64: str = Field(description="Base64-encoded MP4 video data")
    metadata: dict = Field(description="Generation metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "video_url": "/outputs/video_abc12345.mp4",
                    "video_path": "./outputs/video_abc12345.mp4",
                    "video_base64": "<base64-encoded MP4>",
                    "metadata": {
                        "prompt": "A cat walking through a garden",
                        "num_steps": 4,
                        "width": 832,
                        "height": 480,
                        "seed": 42,
                    },
                }
            ]
        }
    }
