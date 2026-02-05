"""Abstract base class for all inference models."""

from abc import ABC, abstractmethod
from typing import Any, Generic

from ..schema import InputT, OutputT, ModelInfo, ModelStatus

class VlgmBase(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all inference models.

    This class defines the interface that all model implementations must follow.
    It provides a consistent API for loading, unloading, and running inference.

    Type Parameters:
        InputT: The type of input data the model accepts
        OutputT: The type of output data the model produces

    Example:
        class MyModel(BaseModel_[MyInput, MyOutput]):
            def load(self) -> None:
                # Load model weights
                pass

            def predict(self, input_data: MyInput) -> MyOutput:
                # Run inference
                pass
    """

    def __init__(self, name: str, version: str = "1.0.0", description: str = "") -> None:
        """
        Initialize the base model.

        Args:
            name: Human-readable name for the model
            version: Model version string
            description: Brief description of what the model does
        """
        self._info = ModelInfo(
            name=name,
            version=version,
            description=description,
            status=ModelStatus.UNLOADED,
        )
        self._model: Any = None

    @property
    def info(self) -> ModelInfo:
        """Get model information."""
        return self._info

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._info.status == ModelStatus.READY

    @abstractmethod
    def load(self) -> None:
        """
        Load the model into memory.

        This method should:
        1. Set status to LOADING
        2. Download/load model weights
        3. Move model to appropriate device
        4. Set status to READY on success, ERROR on failure

        Raises:
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model from memory.

        This method should:
        1. Release model weights from memory
        2. Clear any caches
        3. Set status to UNLOADED
        """
        pass

    @abstractmethod
    def predict(self, input_data: InputT) -> OutputT:
        """
        Run inference on the input data.

        Args:
            input_data: Input data matching the model's expected format

        Returns:
            Model output in the expected format

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input data is invalid
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: InputT) -> bool:
        """
        Validate input data before inference.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        pass

    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded before inference."""
        if not self.is_loaded:
            raise RuntimeError(
                f"Model '{self._info.name}' is not loaded. "
                f"Current status: {self._info.status.value}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._info.name}', status={self._info.status.value})"
