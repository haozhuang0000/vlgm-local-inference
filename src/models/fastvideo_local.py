"""FastVideo model implementation for accelerated video generation."""

import logging
import os
import uuid
from pathlib import Path
import argparse

import torch

from ..config import get_settings
from .base import VlgmBase
from ..schema import FastVideoInput, FastVideoOutput, ModelStatus

logger = logging.getLogger(__name__)


class FastVideoModel(VlgmBase[FastVideoInput, FastVideoOutput]):
    """
    FastVideo accelerated video generation model.

    Uses distilled diffusion models for fast text-to-video generation.
    Supports FastWan2.1-T2V-1.3B and other distilled models.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        super().__init__(
            name="FastVideo",
            version="1.0.0",
            description="Accelerated video generation with distilled diffusion models",
        )
        settings = get_settings()
        self._model_id = model_id or settings.fastvideo_model_id
        self._device_config = device or settings.device
        self._dtype_config = dtype or settings.torch_dtype
        self._output_dir = Path(output_dir or settings.output_dir)
        self._generator = None

    def _get_device(self) -> str:
        if self._device_config == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._device_config

    def _get_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._dtype_config, torch.bfloat16)

    def load(self) -> None:
        if self.is_loaded:
            return

        self._info.status = ModelStatus.LOADING
        device = self._get_device()
        dtype = self._get_dtype()
        logger.info(f"Loading FastVideo model '{self._model_id}' on {device} with {dtype}")

        # try:
            # Import fastvideo
        from fastvideo import VideoGenerator

        # Set attention backend for optimal performance
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"

        # Load the generator from pretrained model
        self._generator = VideoGenerator.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            device=device,
        )
        self._model = self._generator

        self._info.status = ModelStatus.READY
        self._info.device = device
        self._info.dtype = str(dtype)
        logger.info(f"FastVideo model loaded successfully on {device}")

        # except Exception as e:
        #     self._info.status = ModelStatus.ERROR
        #     logger.error(f"Failed to load FastVideo model: {e}")
        #     raise RuntimeError(f"Failed to load FastVideo model: {e}")

    def unload(self) -> None:
        self._generator = None
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._info.status = ModelStatus.UNLOADED
        self._info.memory_usage_mb = 0.0
        logger.info("FastVideo model unloaded")

    def validate_input(self, input_data: FastVideoInput) -> bool:
        if not input_data.prompt or len(input_data.prompt.strip()) == 0:
            return False
        if input_data.num_frames < 1 or input_data.num_frames > 200:
            return False
        if input_data.num_inference_steps < 1 or input_data.num_inference_steps > 50:
            return False
        if input_data.height < 256 or input_data.height > 1080:
            return False
        if input_data.width < 256 or input_data.width > 1920:
            return False
        return True

    @torch.inference_mode()
    def predict(self, input_data: FastVideoInput) -> FastVideoOutput:
        self._ensure_loaded()

        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input: {input_data}")

        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        video_id = uuid.uuid4().hex[:12]
        output_path = self._output_dir / f"fastvideo_{video_id}.mp4"

        logger.info(f"Generating video with prompt: '{input_data.prompt[:50]}...'")

        # Set seed if provided
        seed = input_data.seed
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # Generate video using FastVideo
        self._generator.generate_video(
            prompt=input_data.prompt,
            num_frames=input_data.num_frames,
            height=input_data.height,
            width=input_data.width,
            num_inference_steps=input_data.num_inference_steps,
            guidance_scale=input_data.guidance_scale,
            seed=seed,
            save_video=True,
            output_path=str(output_path),
        )

        metadata = {
            "prompt": input_data.prompt,
            "num_frames": input_data.num_frames,
            "height": input_data.height,
            "width": input_data.width,
            "num_inference_steps": input_data.num_inference_steps,
            "guidance_scale": input_data.guidance_scale,
            "seed": seed,
            "model_id": self._model_id,
        }

        logger.info(f"Video generated: {output_path}")

        return FastVideoOutput(
            video_path=str(output_path),
            metadata=metadata,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local FastVideo test runner")
    parser.add_argument("--prompt", default="generate a cat", help="Text prompt for generation")
    parser.add_argument("--num-frames", type=int, default=49, help="Number of frames")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num-inference-steps", type=int, default=8, help="Diffusion steps")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model-id", default=None, help="Override model id")
    parser.add_argument("--device", default=None, help="cuda | cpu | auto")
    parser.add_argument("--dtype", default=None, help="float16 | bfloat16 | float32")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    return parser


def _run_local() -> int:
    args = _build_arg_parser().parse_args()
    model = FastVideoModel(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        output_dir=args.output_dir,
    )
    model.load()
    input_data = FastVideoInput(
        prompt=args.prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    result = model.predict(input_data)
    print(f"Video saved: {result.video_path}")
    print(f"Metadata: {result.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_local())
