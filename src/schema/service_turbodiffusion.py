from pydantic import BaseModel, Field
from typing import Any, Literal


class TurboDiffusionInput(BaseModel):
    """Internal input for TurboDiffusion model."""

    prompt: str = Field(min_length=1, description="Text description of the video")
    num_steps: int = Field(default=4, ge=1, le=4, description="Inference steps (1-4)")
    num_frames: int = Field(default=81, ge=1, le=200, description="Number of frames")
    seed: int | None = Field(default=None, ge=0, description="Random seed")
    resolution: str | None = Field(default=None, description="Resolution: 480p or 720p")
    aspect_ratio: str | None = Field(default=None, description="Aspect ratio e.g. 16:9")
    task: Literal["text2video", "image2video"] = Field(default="text2video", description="Task type")
    image: Any = Field(default=None, description="Input image bytes for image2video")

    model_config = {"arbitrary_types_allowed": True}


class TurboDiffusionOutput(BaseModel):
    """Internal output from TurboDiffusion model."""

    video_path: str = Field(description="Path to generated video")
    metadata: dict = Field(default_factory=dict, description="Generation metadata")
