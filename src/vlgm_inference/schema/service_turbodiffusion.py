from pydantic import BaseModel, Field
from typing import Any, Literal
from pathlib import Path

class TurboDiffusionInput(BaseModel):
    """
    Input data for TurboDiffusion model.

    Attributes:
        prompt: Text description of the video to generate
        negative_prompt: Things to avoid in the generation
        num_steps: Number of inference steps (fewer = faster, more = higher quality)
        guidance_scale: How closely to follow the prompt (higher = more faithful)
        width: Output video width in pixels
        height: Output video height in pixels
        num_frames: Number of frames to generate
        fps: Frames per second for output video
        seed: Random seed for reproducibility (None for random)
        task: Generation task type
        input_image: Path to input image for image2video task
    """

    prompt: str = Field(min_length=1, description="Text description of the video")
    negative_prompt: str = Field(default="", description="Elements to avoid")
    num_steps: int = Field(default=4, ge=1, le=50, description="Inference steps")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Guidance scale")
    width: int = Field(default=832, ge=256, le=1920, description="Output width")
    height: int = Field(default=480, ge=256, le=1080, description="Output height")
    num_frames: int = Field(default=81, ge=1, le=200, description="Number of frames")
    fps: int = Field(default=16, ge=1, le=60, description="Frames per second")
    seed: int | None = Field(default=None, ge=0, description="Random seed")
    task: Literal["text2video", "image2video"] = Field(default="text2video", description="Task type")
    input_image: str | Path | None = Field(default=None, description="Input image for i2v")

    model_config = {"arbitrary_types_allowed": True}


class TurboDiffusionOutput(BaseModel):
    """
    Output from TurboDiffusion model.

    Attributes:
        video_path: Path to the generated video file
        frames: Optional list of frame paths
        metadata: Generation metadata
    """

    video_path: str = Field(description="Path to generated video")
    frames: list[str] = Field(default_factory=list, description="List of frame paths")
    metadata: dict = Field(default_factory=dict, description="Generation metadata")