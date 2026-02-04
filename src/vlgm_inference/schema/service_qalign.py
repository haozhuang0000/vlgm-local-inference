from pydantic import BaseModel, Field
from typing import Any, Literal


class QAlignInput(BaseModel):
    """
    Input data for Q-Align model.

    Attributes:
        video: Path to video file or raw video bytes
        task: Type of assessment to perform
            - "quality": Video quality assessment (VQA)
    """

    video: Any = Field(description="Path to video file or raw video bytes")
    task: Literal["quality"] = Field(
        default="quality",
        description="Assessment task type",
    )

    model_config = {"arbitrary_types_allowed": True}


class QAlignOutput(BaseModel):
    """
    Output from Q-Align model.

    Attributes:
        score: Predicted quality score (1-5 scale)
        confidence: Confidence scores for each level
        task: The task that was performed
    """

    score: float = Field(ge=1.0, le=5.0, description="Quality score (1-5)")
    confidence: dict[str, float] = Field(description="Confidence for each quality level")
    task: str = Field(description="Task performed")
