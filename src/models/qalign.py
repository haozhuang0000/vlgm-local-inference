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
        if self._device_config == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if self._device_config == "cuda":
            return "cuda:0"
        return self._device_config

    def load(self) -> None:
        if self.is_loaded:
            return
        self._info.status = ModelStatus.LOADING
        device = self._get_device()
        logger.info(f"Loading QAlignVideoScorer on {device}")

        self._scorer = QAlignVideoScorer(pretrained=self._model_id, device=device)
        self._model = self._scorer.model
        self._info.status = ModelStatus.READY
        self._info.device = device
        logger.info("Q-Align model loaded")

    def unload(self) -> None:
        self._scorer = None
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._info.status = ModelStatus.UNLOADED

    def validate_input(self, input_data: QAlignInput) -> bool:
        if input_data.task != "quality":
            return False
        if isinstance(input_data.video, (str, Path)):
            path = Path(input_data.video)
            return path.exists() and path.suffix.lower() in self.SUPPORTED_VIDEO_FORMATS
        return isinstance(input_data.video, bytes)

    @torch.inference_mode()
    def predict(self, input_data: QAlignInput) -> QAlignOutput:
        import gc

        self._ensure_loaded()
        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input: {input_data}")

        from q_align import load_video

        # If input is raw bytes, write to a temp file for decord
        if isinstance(input_data.video, bytes):
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(input_data.video)
            tmp.close()
            video_path = tmp.name
        else:
            video_path = str(input_data.video)

        try:
            frames = load_video(video_path)
            # Scorer returns raw [0, 1]; map to API's [1, 5] scale
            raw_score = self._scorer([frames]).item()
            # Clear frames reference to allow VideoReader cleanup
            del frames
        finally:
            # Force GC to ensure VideoReader is destroyed before file deletion (critical on Windows)
            gc.collect()
            if isinstance(input_data.video, bytes):
                os.unlink(video_path)

        return QAlignOutput(score=raw_score * 4 + 1, task=input_data.task)
