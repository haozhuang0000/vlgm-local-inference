"""Service-level input/output schemas for FastVideo model."""

from pydantic import BaseModel, Field
from typing import Any, Literal


class FastVideoInput(BaseModel):
    """
    Internal input for FastVideo model.

    Attributes:
        prompt: Text description of the video to generate
        num_frames: Number of frames to generate
        height: Output video height
        width: Output video width
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
    """

    prompt: str = Field(min_length=1, description="Text description of the video")
    num_frames: int = Field(default=49, ge=1, le=200, description="Number of frames to generate")
    height: int = Field(default=480, ge=256, le=1080, description="Output video height")
    width: int = Field(default=832, ge=256, le=1920, description="Output video width")
    num_inference_steps: int = Field(default=8, ge=1, le=50, description="Number of diffusion steps")
    guidance_scale: float = Field(default=1.0, ge=0.0, le=20.0, description="Guidance scale")
    seed: int | None = Field(default=None, ge=0, description="Random seed for reproducibility")

    model_config = {"arbitrary_types_allowed": True}


class FastVideoOutput(BaseModel):
    """
    Internal output from FastVideo model.

    Attributes:
        video_path: Path to the generated video file
        metadata: Generation metadata including parameters used
    """

    video_path: str = Field(description="Path to generated video")
    metadata: dict = Field(default_factory=dict, description="Generation metadata")
