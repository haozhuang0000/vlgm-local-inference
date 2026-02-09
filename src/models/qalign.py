"""Q-Align model implementation for video quality assessment."""

import logging
import os
import sys
import tempfile
from pathlib import Path

import torch

from ..config import get_settings
from .base import VlgmBase
from ..schema import QAlignInput, QAlignOutput, ModelStatus

# Make the Q-Align submodule importable
_QALIGN_DIR = str(Path(__file__).parent / "Q-Align")
if _QALIGN_DIR not in sys.path:
    sys.path.insert(0, _QALIGN_DIR)

from q_align import QAlignVideoScorer

logger = logging.getLogger(__name__)


class QAlignModel(VlgmBase[QAlignInput, QAlignOutput]):
    """Q-Align video quality scorer. Returns scores on a 1-5 scale."""

    SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

    def __init__(self, model_id: str | None = None, device: str | None = None, dtype: str | None = None) -> None:
        super().__init__(name="Q-Align", version="1.0.0", description="Video quality assessment")
        settings = get_settings()
        self._model_id = model_id or settings.qalign_model_id
        self._device_config = device or settings.device
        self._scorer = None

    def _get_device(self) -> str:
        logger.debug(f"Determining device from config: {self._device_config}")
        if self._device_config == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.debug(f"Auto device selection: CUDA available={torch.cuda.is_available()}, selected={device}")
            return device
        if self._device_config == "cuda":
            logger.debug("Explicit CUDA config, using cuda:0")
            return "cuda:0"
        logger.debug(f"Using configured device: {self._device_config}")
        return self._device_config

    def load(self) -> None:
        if self.is_loaded:
            logger.debug("Model already loaded, skipping load")
            return
        logger.info("Starting model load process")
        self._info.status = ModelStatus.LOADING
        device = self._get_device()
        logger.info(f"Loading QAlignVideoScorer on {device}")
        logger.debug(f"Model ID: {self._model_id}")

        logger.debug("Initializing QAlignVideoScorer...")
        self._scorer = QAlignVideoScorer(pretrained=self._model_id, device=device)
        self._model = self._scorer.model
        self._info.status = ModelStatus.READY
        self._info.device = device
        logger.info(f"Q-Align model loaded successfully on device: {device}")

    def unload(self) -> None:
        logger.info("Unloading Q-Align model")
        self._scorer = None
        self._model = None
        if torch.cuda.is_available():
            logger.debug("Clearing CUDA cache")
            torch.cuda.empty_cache()
        self._info.status = ModelStatus.UNLOADED
        logger.info("Q-Align model unloaded successfully")

    def validate_input(self, input_data: QAlignInput) -> bool:
        logger.debug(f"Validating input - task: {input_data.task}")
        if input_data.task != "quality":
            logger.warning(f"Invalid task type: {input_data.task}, expected 'quality'")
            return False
        if isinstance(input_data.video, (str, Path)):
            path = Path(input_data.video)
            exists = path.exists()
            valid_format = path.suffix.lower() in self.SUPPORTED_VIDEO_FORMATS
            logger.debug(f"Video path validation - exists: {exists}, valid_format: {valid_format}, suffix: {path.suffix}")
            return exists and valid_format
        is_bytes = isinstance(input_data.video, bytes)
        logger.debug(f"Video input is bytes: {is_bytes}, size: {len(input_data.video) if is_bytes else 'N/A'} bytes")
        return is_bytes

    @torch.inference_mode()
    def predict(self, input_data: QAlignInput) -> QAlignOutput:
        import gc
        import time

        logger.info("Starting prediction")
        start_time = time.time()

        logger.debug("Ensuring model is loaded")
        self._ensure_loaded()
        if not self.validate_input(input_data):
            logger.error(f"Invalid input data: {input_data}")
            raise ValueError(f"Invalid input: {input_data}")

        from q_align import load_video

        # If input is raw bytes, write to a temp file for decord
        if isinstance(input_data.video, bytes):
            logger.debug(f"Input is bytes ({len(input_data.video)} bytes), writing to temp file")
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(input_data.video)
            tmp.close()
            video_path = tmp.name
            logger.debug(f"Temp file created: {video_path}")
        else:
            video_path = str(input_data.video)
            logger.debug(f"Using video path directly: {video_path}")

        try:
            logger.debug(f"Loading video frames from: {video_path}")
            load_start = time.time()
            frames = load_video(video_path)
            logger.debug(f"Video frames loaded in {time.time() - load_start:.3f}s, frames shape: {frames.shape if hasattr(frames, 'shape') else 'N/A'}")

            # Scorer returns raw [0, 1]; map to API's [1, 5] scale
            # Weight tensor: [1, 0.75, 0.5, 0.25, 0] for [excellent, good, fair, poor, bad]
            # Mapping: 0.0→1 (bad), 0.25→2 (poor), 0.5→3 (fair), 0.75→4 (good), 1.0→5 (excellent)
            logger.debug("Running scorer on frames")
            score_start = time.time()
            raw_score = self._scorer([frames]).item()
            score = raw_score * 4 + 1  # Convert [0, 1] to [1, 5]
            logger.debug(f"Scoring completed in {time.time() - score_start:.3f}s, raw_score: {raw_score}, mapped_score: {score}")

            # Clear frames reference to allow VideoReader cleanup
            del frames
            logger.debug("Frames reference cleared")
        finally:
            # Force GC to ensure VideoReader is destroyed before file deletion (critical on Windows)
            logger.debug("Running garbage collection")
            gc.collect()
            if isinstance(input_data.video, bytes):
                # Retry with delay to handle Windows file locking from Decord's VideoReader
                for attempt in range(5):
                    try:
                        logger.debug(f"Deleting temp file (attempt {attempt + 1}): {video_path}")
                        os.unlink(video_path)
                        break
                    except PermissionError:
                        if attempt < 4:
                            logger.warning(f"Temp file still locked, retrying in 0.2s...")
                            time.sleep(0.2)
                            gc.collect()
                        else:
                            logger.error(f"Failed to delete temp file after 5 attempts: {video_path}")
                    except FileNotFoundError:
                        # File already deleted, ignore
                        break

        total_time = time.time() - start_time
        logger.info(f"Prediction completed in {total_time:.3f}s, score: {score}")

        return QAlignOutput(score=score, task=input_data.task)
