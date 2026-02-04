"""TurboDiffusion model implementation for fast video generation."""

import logging
import uuid
from pathlib import Path
import torch

from vlgm_inference.config import get_settings
from vlgm_inference.models.base import VlgmBase
from vlgm_inference.schema import TurboDiffusionInput, TurboDiffusionOutput, ModelStatus
logger = logging.getLogger(__name__)

class TurboDiffusionModel(VlgmBase[TurboDiffusionInput, TurboDiffusionOutput]):
    """
    TurboDiffusion model for fast video generation.

    TurboDiffusion achieves 100-200x acceleration over standard diffusion models
    through SageAttention, SLA, and timestep distillation techniques.

    Available Models:
    - TurboWan2.1-T2V-1.3B-480P: Lightweight text-to-video (smallest)
    - TurboWan2.1-T2V-14B-480P: Higher quality text-to-video
    - TurboWan2.1-T2V-14B-720P: Best quality text-to-video
    - TurboWan2.2-I2V-A14B-720P: Image-to-video

    Example:
        >>> model = TurboDiffusionModel()
        >>> model.load()
        >>> result = model.predict(TurboDiffusionInput(
        ...     prompt="A cat walking through a garden",
        ...     num_steps=4
        ... ))
        >>> print(f"Video saved to: {result.video_path}")
    """

    # Model configurations
    MODEL_CONFIGS = {
        "Wan2.1-1.3B": {
            "checkpoint": "TurboWan2.1-T2V-1.3B-480P",
            "resolution": (832, 480),
            "task": "text2video",
        },
        "Wan2.1-14B-480P": {
            "checkpoint": "TurboWan2.1-T2V-14B-480P",
            "resolution": (832, 480),
            "task": "text2video",
        },
        "Wan2.1-14B-720P": {
            "checkpoint": "TurboWan2.1-T2V-14B-720P",
            "resolution": (1280, 720),
            "task": "text2video",
        },
        "Wan2.2-I2V": {
            "checkpoint": "TurboWan2.2-I2V-A14B-720P",
            "resolution": (1280, 720),
            "task": "image2video",
        },
    }

    def __init__(
        self,
        model_name: str | None = None,
        checkpoint_dir: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        """
        Initialize TurboDiffusion model.

        Args:
            model_name: Model variant to use (default: Wan2.1-1.3B)
            checkpoint_dir: Directory containing model checkpoints
            device: Device to load model on
            dtype: Model dtype
            output_dir: Directory to save generated videos
        """
        super().__init__(
            name="TurboDiffusion",
            version="1.0.0",
            description="Fast video generation with 100-200x acceleration",
        )

        settings = get_settings()
        self._model_name = model_name or settings.turbodiffusion_model
        self._checkpoint_dir = Path(checkpoint_dir or settings.turbodiffusion_checkpoint_dir)
        self._device_config = device or settings.device
        self._dtype_config = dtype or settings.torch_dtype
        self._output_dir = Path(output_dir or settings.output_dir)

        # Validate model name
        if self._model_name not in self.MODEL_CONFIGS:
            available = ", ".join(self.MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown model: {self._model_name}. Available: {available}")

        self._config = self.MODEL_CONFIGS[self._model_name]
        self._pipeline = None
        self._vae = None
        self._text_encoder = None

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
        """
        Load TurboDiffusion model components.

        Note: TurboDiffusion requires specific checkpoint files:
        - VAE: Wan2.1_VAE.pth
        - Text encoder: models_t5_umt5-xxl-enc-bf16.pth
        - DiT checkpoint: Model-specific .pth file
        """
        if self.is_loaded:
            logger.info("TurboDiffusion model already loaded")
            return

        self._info.status = ModelStatus.LOADING
        logger.info(f"Loading TurboDiffusion model: {self._model_name}")

        device = self._get_device()

        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Try to import turbodiffusion
            try:
                from turbodiffusion import TurboWanPipeline
                self._load_turbodiffusion_native(device)
            except ImportError:
                logger.warning(
                    "turbodiffusion package not installed. "
                    "Using mock implementation for demonstration."
                )
                self._load_mock_pipeline(device)

            self._info.status = ModelStatus.READY
            self._info.device = device
            self._info.dtype = self._dtype_config

            if torch.cuda.is_available() and device != "cpu":
                self._info.memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024

            logger.info(f"TurboDiffusion model loaded successfully on {device}")

        except Exception as e:
            self._info.status = ModelStatus.ERROR
            logger.error(f"Failed to load TurboDiffusion model: {e}")
            raise RuntimeError(f"Failed to load TurboDiffusion model: {e}") from e

    def _load_turbodiffusion_native(self, device: str) -> None:
        """Load the actual TurboDiffusion pipeline."""
        from turbodiffusion import TurboWanPipeline

        checkpoint = self._config["checkpoint"]
        dit_path = self._checkpoint_dir / f"{checkpoint}.pth"
        vae_path = self._checkpoint_dir / "Wan2.1_VAE.pth"
        t5_path = self._checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"

        # Check if checkpoints exist
        for path, name in [(dit_path, "DiT"), (vae_path, "VAE"), (t5_path, "T5")]:
            if not path.exists():
                logger.warning(f"{name} checkpoint not found at {path}")

        self._pipeline = TurboWanPipeline.from_pretrained(
            dit_path=str(dit_path),
            vae_path=str(vae_path),
            t5_path=str(t5_path),
            device=device,
            dtype=self._get_torch_dtype(),
        )

    def _load_mock_pipeline(self, device: str) -> None:
        """Load a mock pipeline for demonstration/testing."""
        logger.info("Using mock TurboDiffusion pipeline")
        self._pipeline = MockTurboDiffusionPipeline(device)

    def unload(self) -> None:
        """Unload TurboDiffusion model from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        if self._vae is not None:
            del self._vae
            self._vae = None

        if self._text_encoder is not None:
            del self._text_encoder
            self._text_encoder = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._info.status = ModelStatus.UNLOADED
        self._info.memory_usage_mb = 0.0
        logger.info("TurboDiffusion model unloaded")

    def validate_input(self, input_data: TurboDiffusionInput) -> bool:
        """Validate TurboDiffusion input data."""
        # Check prompt
        if not input_data.prompt or not input_data.prompt.strip():
            return False

        # Check dimensions
        if input_data.width <= 0 or input_data.height <= 0:
            return False

        if input_data.num_frames <= 0:
            return False

        if input_data.num_steps <= 0:
            return False

        # Check task-specific requirements
        if input_data.task == "image2video":
            if input_data.input_image is None:
                return False
            if isinstance(input_data.input_image, (str, Path)):
                if not Path(input_data.input_image).exists():
                    return False

        return True

    @torch.inference_mode()
    def predict(self, input_data: TurboDiffusionInput) -> TurboDiffusionOutput:
        """
        Generate a video from the input prompt.

        Args:
            input_data: TurboDiffusionInput with prompt and generation parameters

        Returns:
            TurboDiffusionOutput with path to generated video
        """
        self._ensure_loaded()

        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input data: {input_data}")

        # Set seed for reproducibility
        seed = input_data.seed if input_data.seed is not None else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self._info.device).manual_seed(seed)

        # Generate unique output filename
        output_id = str(uuid.uuid4())[:8]
        video_path = self._output_dir / f"video_{output_id}.mp4"

        try:
            if hasattr(self._pipeline, "generate"):
                # Native TurboDiffusion pipeline
                frames = self._pipeline.generate(
                    prompt=input_data.prompt,
                    negative_prompt=input_data.negative_prompt,
                    num_inference_steps=input_data.num_steps,
                    guidance_scale=input_data.guidance_scale,
                    width=input_data.width,
                    height=input_data.height,
                    num_frames=input_data.num_frames,
                    generator=generator,
                )

                # Save video
                self._save_video(frames, video_path, input_data.fps)

            elif hasattr(self._pipeline, "__call__"):
                # Mock or compatible pipeline
                result = self._pipeline(
                    prompt=input_data.prompt,
                    num_steps=input_data.num_steps,
                    width=input_data.width,
                    height=input_data.height,
                    num_frames=input_data.num_frames,
                    output_path=str(video_path),
                )
                if isinstance(result, dict) and "video_path" in result:
                    video_path = Path(result["video_path"])

        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise RuntimeError(f"TurboDiffusion generation failed: {e}") from e

        metadata = {
            "prompt": input_data.prompt,
            "negative_prompt": input_data.negative_prompt,
            "num_steps": input_data.num_steps,
            "guidance_scale": input_data.guidance_scale,
            "width": input_data.width,
            "height": input_data.height,
            "num_frames": input_data.num_frames,
            "fps": input_data.fps,
            "seed": seed,
            "model": self._model_name,
        }

        return TurboDiffusionOutput(
            video_path=str(video_path),
            metadata=metadata,
        )

    def _save_video(
        self,
        frames: list | torch.Tensor,
        output_path: Path,
        fps: int,
    ) -> None:
        """Save frames as video file."""
        try:
            import torchvision.io as io

            if isinstance(frames, list):
                frames = torch.stack(frames)

            # Ensure correct format: (T, H, W, C) for write_video
            if frames.dim() == 4 and frames.shape[1] == 3:
                frames = frames.permute(0, 2, 3, 1)

            # Ensure uint8 format
            if frames.dtype != torch.uint8:
                frames = (frames * 255).clamp(0, 255).to(torch.uint8)

            io.write_video(str(output_path), frames.cpu(), fps)

        except Exception as e:
            logger.warning(f"Failed to save video with torchvision: {e}")
            # Fallback: save as individual frames
            self._save_frames(frames, output_path.parent, output_path.stem)

    def _save_frames(
        self,
        frames: torch.Tensor,
        output_dir: Path,
        prefix: str,
    ) -> list[str]:
        """Save individual frames as images."""
        from PIL import Image
        import numpy as np

        frame_paths = []
        frames_dir = output_dir / f"{prefix}_frames"
        frames_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(frames):
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()

            if frame.shape[0] == 3:  # CHW -> HWC
                frame = np.transpose(frame, (1, 2, 0))

            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)

            img = Image.fromarray(frame.astype(np.uint8))
            path = frames_dir / f"frame_{i:04d}.png"
            img.save(path)
            frame_paths.append(str(path))

        return frame_paths


class MockTurboDiffusionPipeline:
    """Mock pipeline for testing when turbodiffusion is not installed."""

    def __init__(self, device: str) -> None:
        self.device = device

    def __call__(
        self,
        prompt: str,
        num_steps: int,
        width: int,
        height: int,
        num_frames: int,
        output_path: str,
        **kwargs,
    ) -> dict:
        """Generate a placeholder video."""
        import numpy as np
        from PIL import Image

        logger.info(f"Mock generation: '{prompt[:50]}...' -> {output_path}")

        # Create simple gradient frames as placeholder
        frames = []
        for i in range(min(num_frames, 16)):  # Limit frames for mock
            # Create a gradient that changes over time
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            t = i / max(num_frames - 1, 1)

            # Simple animated gradient
            for y in range(height):
                for x in range(width):
                    frame[y, x, 0] = int(255 * (x / width) * (1 - t) + 255 * t * 0.5)
                    frame[y, x, 1] = int(255 * (y / height) * t)
                    frame[y, x, 2] = int(255 * (1 - x / width) * (1 - t))

            frames.append(frame)

        # Save as simple video or frames
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try to save as video
        try:
            import torchvision.io as io

            frames_tensor = torch.from_numpy(np.stack(frames))
            io.write_video(str(output_path), frames_tensor, fps=8)
        except Exception:
            # Fallback: save first frame as image
            img = Image.fromarray(frames[0])
            output_path = output_path.with_suffix(".png")
            img.save(output_path)

        return {"video_path": str(output_path)}
