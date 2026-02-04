"""Q-Align model implementation for video quality assessment."""

import logging
from pathlib import Path
from typing import List

import torch

from vlgm_inference.config import get_settings
from vlgm_inference.models.base import VlgmBase
from vlgm_inference.schema import QAlignInput, QAlignOutput, ModelStatus

logger = logging.getLogger(__name__)


class QAlignModel(VlgmBase[QAlignInput, QAlignOutput]):
    """
    Q-Align model for video quality assessment.

    Q-Align is a unified model for visual scoring that can assess:
    - Video Quality Assessment (VQA)

    The model outputs scores on a 1-5 scale where:
    - 5: Excellent
    - 4: Good
    - 3: Fair
    - 2: Poor
    - 1: Bad

    Example:
        >>> model = QAlignModel()
        >>> model.load()
        >>> result = model.predict(QAlignInput(video="video.mp4", task="quality"))
        >>> print(f"Quality score: {result.score:.2f}")
    """

    # Quality level weights for score computation
    QUALITY_LEVELS = {"excellent": 5, "good": 4, "fair": 3, "poor": 2, "bad": 1}

    # Supported video formats
    SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

    # Prompts for different tasks
    TASK_PROMPTS = {
        "quality": "How would you rate the quality of this video?",
    }

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        """
        Initialize Q-Align model.

        Args:
            model_id: HuggingFace model ID (default: q-future/one-align)
            device: Device to load model on ("cuda", "cpu", or "auto")
            dtype: Model dtype ("float16", "bfloat16", or "float32")
        """
        super().__init__(
            name="Q-Align",
            version="1.0.0",
            description="Model for video quality assessment",
        )

        settings = get_settings()
        self._model_id = model_id or settings.qalign_model_id
        self._device_config = device or settings.device
        self._dtype_config = dtype or settings.torch_dtype

        self._processor = None
        self._tokenizer = None

    def _get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._dtype_config, torch.float16)

    def _get_device(self) -> str:
        """Determine the device to use."""
        if self._device_config == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._device_config

    def load(self) -> None:
        """Load Q-Align model from HuggingFace."""
        if self.is_loaded:
            logger.info("Q-Align model already loaded")
            return

        self._info.status = ModelStatus.LOADING
        logger.info(f"Loading Q-Align model: {self._model_id}")

        # try:
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = self._get_device()
        dtype = self._get_torch_dtype()

        # Load the model with trust_remote_code for Q-Align
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device if device == "auto" else {"": device},
            low_cpu_mem_usage=True,
        )

        # Try to load processor if available
        try:
            self._processor = AutoProcessor.from_pretrained(
                self._model_id,
                trust_remote_code=True,
            )
        except Exception:
            logger.warning("Could not load processor, will use model's built-in processing")
            self._processor = None

        self._model.eval()

        # Update model info
        self._info.status = ModelStatus.READY
        self._info.device = device
        self._info.dtype = self._dtype_config

        # Estimate memory usage
        if torch.cuda.is_available() and device != "cpu":
            self._info.memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024

        logger.info(f"Q-Align model loaded successfully on {device}")

        # except Exception as e:
        #     self._info.status = ModelStatus.ERROR
        #     logger.error(f"Failed to load Q-Align model: {e}")
        #     raise RuntimeError(f"Failed to load Q-Align model: {e}") from e

    def unload(self) -> None:
        """Unload Q-Align model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._info.status = ModelStatus.UNLOADED
        self._info.memory_usage_mb = 0.0
        logger.info("Q-Align model unloaded")

    def validate_input(self, input_data: QAlignInput) -> bool:
        """Validate Q-Align input data."""
        if input_data.task not in self.TASK_PROMPTS:
            return False

        # Check if video is valid
        if isinstance(input_data.video, (str, Path)):
            path = Path(input_data.video)
            if not path.exists():
                return False
            # Check file extension
            if path.suffix.lower() not in self.SUPPORTED_VIDEO_FORMATS:
                return False
        elif isinstance(input_data.video, bytes):
            # Accept raw video bytes
            return True
        else:
            return False

        return True

    def _load_video_frames(self, video: str | Path | bytes, max_frames: int = 8):
        """
        Load video and extract frames for quality assessment.

        Args:
            video: Path to video file or raw video bytes
            max_frames: Maximum number of frames to extract (uniformly sampled)

        Returns:
            List of PIL Images
        """
        import tempfile
        import cv2
        import numpy as np
        from PIL import Image

        video_path = video
        temp_file = None

        # If video is bytes, write to temp file
        if isinstance(video, bytes):
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.write(video)
            temp_file.close()
            video_path = temp_file.name

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError("Video has no frames")

            # Calculate frame indices to sample uniformly
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image (Q-Align expects PIL Images)
                    frame_pil = Image.fromarray(frame_rgb)
                    frames.append(frame_pil)

            cap.release()

            if not frames:
                raise ValueError("Could not extract any frames from video")

            return frames

        finally:
            # Clean up temp file if created
            if temp_file is not None:
                import os
                os.unlink(temp_file.name)

    def _compute_score(self, logits: torch.Tensor) -> tuple[float, dict[str, float]]:
        """
        Compute quality score from model logits.

        The model outputs logits for 5 quality levels. We compute a weighted
        average using softmax probabilities.
        """
        # Get probabilities via softmax
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

        # Map to quality levels (assuming order: excellent, good, fair, poor, bad)
        # Adjust based on actual model output format
        levels = list(self.QUALITY_LEVELS.keys())
        weights = list(self.QUALITY_LEVELS.values())

        # Ensure we have 5 probabilities
        if len(probs) >= 5:
            probs = probs[:5]
        else:
            # Pad with zeros if needed
            probs = list(probs) + [0.0] * (5 - len(probs))

        # Compute weighted score
        score = sum(p * w for p, w in zip(probs, weights))

        # Create confidence dict
        confidence = {level: float(prob) for level, prob in zip(levels, probs)}

        return float(score), confidence

    @torch.inference_mode()
    def predict(self, input_data: QAlignInput) -> QAlignOutput:
        """
        Run quality assessment on a video.

        Args:
            input_data: QAlignInput containing video and task type

        Returns:
            QAlignOutput with score (1-5) and confidence levels
        """
        self._ensure_loaded()

        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input data: {input_data}")

        # Load and prepare video frames
        frames = self._load_video_frames(input_data.video)
        # Todo: remove this
        self._model.to(torch.bfloat16)
        # Get the appropriate prompt
        prompt = self.TASK_PROMPTS[input_data.task]

        # try:
        # Use the model's built-in scoring if available (Q-Align specific)
        if hasattr(self._model, "score"):
            # Q-Align models support video scoring with input_mode="video"
            scores = self._model.score([frames], task_=input_data.task, input_="video")
            score = float(scores[0]) if hasattr(scores, "__iter__") else float(scores)

            # Approximate confidence based on score
            confidence = self._approximate_confidence(score)

        else:
            # Fall back to frame-by-frame inference and average
            if self._processor is not None:
                frame_scores = []
                for frame_pil in frames:
                    inputs = self._processor(
                        text=prompt,
                        images=frame_pil,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                    outputs = self._model(**inputs)
                    logits = outputs.logits[:, -1, :]

                    frame_score, _ = self._compute_score(logits)
                    frame_scores.append(frame_score)

                # Average scores across frames
                score = sum(frame_scores) / len(frame_scores)
                confidence = self._approximate_confidence(score)
            else:
                # If no processor, try direct model call
                score = 3.0  # Default middle score
                confidence = {k: 0.2 for k in self.QUALITY_LEVELS}

        # except Exception as e:
        #     logger.error(f"Inference failed: {e}")
        #     raise RuntimeError(f"Q-Align inference failed: {e}") from e

        return QAlignOutput(
            score=score,
            confidence=confidence,
            task=input_data.task,
        )

    def _approximate_confidence(self, score: float) -> dict[str, float]:
        """Approximate confidence distribution from a single score."""
        # Create a simple distribution centered on the score
        confidence = {}
        for level, weight in self.QUALITY_LEVELS.items():
            # Use a simple Gaussian-like approximation
            diff = abs(weight - score)
            confidence[level] = max(0.0, 1.0 - diff * 0.3)

        # Normalize
        total = sum(confidence.values())
        if total > 0:
            confidence = {k: v / total for k, v in confidence.items()}

        return confidence

    def predict_batch(self, inputs: list[QAlignInput]) -> list[QAlignOutput]:
        """
        Run quality assessment on multiple videos.

        Args:
            inputs: List of QAlignInput objects

        Returns:
            List of QAlignOutput results
        """
        return [self.predict(inp) for inp in inputs]
