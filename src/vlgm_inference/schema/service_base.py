from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class ModelStatus(str, Enum):
    """Model loading status."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    description: str = Field(description="Model description")
    status: ModelStatus = Field(default=ModelStatus.UNLOADED, description="Current status")
    device: str = Field(default="cpu", description="Device the model is loaded on")
    dtype: str = Field(default="float32", description="Model data type")
    memory_usage_mb: float = Field(default=0.0, description="GPU memory usage in MB")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"arbitrary_types_allowed": True}