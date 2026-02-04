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
    confidence: dict[str, float] = Field(
        description="Confidence scores for each quality level",
    )
    task: str = Field(description="Evaluation task performed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "score": 4.23,
                    "confidence": {
                        "excellent": 0.35,
                        "good": 0.45,
                        "fair": 0.15,
                        "poor": 0.04,
                        "bad": 0.01,
                    },
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
    - Use clear, descriptive prompts
    - Keep prompts under 200 characters
    - Use negative prompts to avoid unwanted elements
    """

    prompt: str = Field(
        min_length=1,
        max_length=500,
        description="Text description of the video to generate",
    )
    negative_prompt: str = Field(
        default="",
        max_length=500,
        description="Elements to avoid in the generation",
    )
    num_steps: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Number of inference steps (4 recommended for speed)",
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="How closely to follow the prompt",
    )
    width: int = Field(
        default=832,
        ge=256,
        le=1920,
        description="Output video width in pixels",
    )
    height: int = Field(
        default=480,
        ge=256,
        le=1080,
        description="Output video height in pixels",
    )
    num_frames: int = Field(
        default=81,
        ge=1,
        le=200,
        description="Number of frames to generate",
    )
    fps: int = Field(
        default=16,
        ge=1,
        le=60,
        description="Frames per second for output video",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Random seed for reproducibility",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "A cat walking through a sunlit garden with flowers",
                    "negative_prompt": "blurry, low quality",
                    "num_steps": 4,
                    "guidance_scale": 7.5,
                    "width": 832,
                    "height": 480,
                    "num_frames": 81,
                    "fps": 16,
                }
            ]
        }
    }


class TurboDiffusionResponse(BaseModel):
    """Response from TurboDiffusion video generation."""

    video_url: str = Field(description="URL to download the generated video")
    video_path: str = Field(description="Server path to the generated video")
    metadata: dict = Field(description="Generation metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "video_url": "/outputs/video_abc12345.mp4",
                    "video_path": "./outputs/video_abc12345.mp4",
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
