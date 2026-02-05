"""TurboDiffusion model implementation for fast video generation."""

import argparse
import logging
import math
import os
import sys
import tempfile
import uuid
from pathlib import Path

import torch
from einops import rearrange

from ..config import get_settings
from .base import VlgmBase
from ..schema import TurboDiffusionInput, TurboDiffusionOutput, ModelStatus

# ---------------------------------------------------------------------------
# Make the TurboDiffusion package importable.
# In Docker the repo is cloned to TURBODIFFUSION_REPO_DIR (/app/TurboDiffusion).
# ---------------------------------------------------------------------------
_TD_REPO_DIR = os.environ.get(
    "TURBODIFFUSION_REPO_DIR",
    str(Path(__file__).parent / "TurboDiffusion"),   # local-dev fallback
)
_TURBODIFFUSION_PKG = str(Path(_TD_REPO_DIR) / "turbodiffusion")
if _TURBODIFFUSION_PKG not in sys.path:
    sys.path.insert(0, _TURBODIFFUSION_PKG)

# inference/ dir is needed so that "from modify_model import ..." resolves
_INFERENCE_DIR = str(Path(_TD_REPO_DIR) / "turbodiffusion" / "inference")
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)

from modify_model import tensor_kwargs, create_model
from rcm.utils.umt5 import get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from imaginaire.utils.io import save_image_or_video

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build_t_steps(sigma_max: float, num_steps: int, device: torch.device) -> torch.Tensor:
    """Build rCM timestep schedule (TrigFlow -> RectifiedFlow)."""
    mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
    t = torch.tensor([math.atan(sigma_max), *mid_t, 0], dtype=torch.float64, device=device)
    return torch.sin(t) / (torch.cos(t) + torch.sin(t))


# ---------------------------------------------------------------------------
# model wrapper
# ---------------------------------------------------------------------------
class TurboDiffusionModel(VlgmBase[TurboDiffusionInput, TurboDiffusionOutput]):
    """TurboDiffusion video generator (rCM distillation + SLA, 100-200x faster)."""

    def __init__(
        self,
        model_name: str | None = None,
        dit_path: str | None = None,
        vae_path: str | None = None,
        text_encoder_path: str | None = None,
        attention_type: str | None = None,
        sla_topk: float | None = None,
        quant_linear: bool | None = None,
        default_norm: bool | None = None,
        output_dir: str | None = None,
        high_noise_model_path: str | None = None,
        low_noise_model_path: str | None = None,
        boundary: float | None = None,
    ) -> None:
        super().__init__(
            name="TurboDiffusion",
            version="1.0.0",
            description="Fast video generation with 100-200x acceleration",
        )

        settings = get_settings()
        self._model_name = model_name or settings.turbodiffusion_model
        self._dit_path = dit_path or settings.turbodiffusion_dit_path
        self._vae_path = vae_path or settings.turbodiffusion_vae_path
        self._text_encoder_path = text_encoder_path or settings.turbodiffusion_text_encoder_path
        self._attention_type = attention_type or settings.turbodiffusion_attention_type
        self._sla_topk = sla_topk if sla_topk is not None else settings.turbodiffusion_sla_topk
        self._quant_linear = quant_linear if quant_linear is not None else settings.turbodiffusion_quant_linear
        self._default_norm = default_norm if default_norm is not None else settings.turbodiffusion_default_norm
        self._output_dir = Path(output_dir or settings.output_dir)
        self._high_noise_model_path = high_noise_model_path or settings.turbodiffusion_high_noise_model_path
        self._low_noise_model_path = low_noise_model_path or settings.turbodiffusion_low_noise_model_path
        self._boundary = boundary if boundary is not None else settings.turbodiffusion_boundary

        # Wan2.2-A14B == image-to-video; all others are text-to-video
        self._is_i2v = self._model_name == "Wan2.2-A14B"

        # populated during load()
        self._net: torch.nn.Module | None = None             # T2V DiT (stays on GPU)
        self._high_noise_net: torch.nn.Module | None = None  # I2V high-noise DiT
        self._low_noise_net: torch.nn.Module | None = None   # I2V low-noise DiT
        self._tokenizer: Wan2pt1VAEInterface | None = None   # shared VAE

    # ------------------------------------------------------------------
    def _make_args(self) -> argparse.Namespace:
        """Namespace consumed by create_model() inside modify_model.py."""
        return argparse.Namespace(
            model=self._model_name,
            attention_type=self._attention_type,
            sla_topk=self._sla_topk,
            quant_linear=self._quant_linear,
            default_norm=self._default_norm,
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        if self.is_loaded:
            return
        self._info.status = ModelStatus.LOADING
        logger.info(f"Loading TurboDiffusion ({self._model_name})")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        args = self._make_args()

        logger.info(f"Loading VAE from {self._vae_path}")
        self._tokenizer = Wan2pt1VAEInterface(vae_pth=self._vae_path)

        if self._is_i2v:
            logger.info("Loading I2V high-noise DiT")
            self._high_noise_net = create_model(dit_path=self._high_noise_model_path, args=args).cpu()
            torch.cuda.empty_cache()
            logger.info("Loading I2V low-noise DiT")
            self._low_noise_net = create_model(dit_path=self._low_noise_model_path, args=args).cpu()
            torch.cuda.empty_cache()
        else:
            logger.info(f"Loading T2V DiT from {self._dit_path}")
            # stays on CUDA for repeated server requests
            self._net = create_model(dit_path=self._dit_path, args=args)

        self._info.status = ModelStatus.READY
        self._info.device = "cuda"
        if torch.cuda.is_available():
            self._info.memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024
        logger.info("TurboDiffusion loaded")

    def unload(self) -> None:
        self._net = None
        self._high_noise_net = None
        self._low_noise_net = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._info.status = ModelStatus.UNLOADED
        self._info.memory_usage_mb = 0.0

    # ------------------------------------------------------------------
    # validation & prediction
    # ------------------------------------------------------------------
    def validate_input(self, input_data: TurboDiffusionInput) -> bool:
        if not input_data.prompt or not input_data.prompt.strip():
            return False
        if not (1 <= input_data.num_steps <= 4):
            return False
        if input_data.num_frames < 1:
            return False
        if self._is_i2v and input_data.task == "image2video" and input_data.image is None:
            return False
        return True

    @torch.inference_mode()
    def predict(self, input_data: TurboDiffusionInput) -> TurboDiffusionOutput:
        self._ensure_loaded()
        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input: {input_data}")

        seed = input_data.seed if input_data.seed is not None else 0
        video_path = self._output_dir / f"video_{uuid.uuid4().hex[:8]}.mp4"

        # text embedding — T5 encoder stays resident after first call
        logger.info("Computing text embedding")
        with torch.no_grad():
            text_emb = get_umt5_embedding(
                checkpoint_path=self._text_encoder_path,
                prompts=input_data.prompt,
            ).to(**tensor_kwargs)

        resolution = input_data.resolution or ("720p" if self._is_i2v else "480p")
        aspect_ratio = input_data.aspect_ratio or "16:9"
        w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]

        if self._is_i2v and input_data.task == "image2video":
            self._generate_i2v(text_emb, input_data, seed, video_path, w, h)
        else:
            self._generate_t2v(text_emb, input_data, seed, video_path, w, h)

        metadata = {
            "prompt": input_data.prompt,
            "num_steps": input_data.num_steps,
            "num_frames": input_data.num_frames,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "seed": seed,
            "model": self._model_name,
            "attention_type": self._attention_type,
        }
        return TurboDiffusionOutput(video_path=str(video_path), metadata=metadata)

    # ------------------------------------------------------------------
    # sampling loops
    # ------------------------------------------------------------------
    def _generate_t2v(self, text_emb, input_data, seed, video_path, w, h) -> None:
        """rCM text-to-video sampling (1-4 steps)."""
        tokenizer = self._tokenizer
        condition = {"crossattn_emb": text_emb}

        state_shape = [
            tokenizer.latent_ch,
            tokenizer.get_latent_num_frames(input_data.num_frames),
            h // tokenizer.spatial_compression_factor,
            w // tokenizer.spatial_compression_factor,
        ]

        generator = torch.Generator(device=tensor_kwargs["device"]).manual_seed(seed)
        x = torch.randn(1, *state_shape, dtype=torch.float32,
                        device=tensor_kwargs["device"], generator=generator)

        t_steps = _build_t_steps(sigma_max=80.0, num_steps=input_data.num_steps, device=x.device)
        x = x.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

        logger.info(f"T2V sampling: {input_data.num_steps} steps")
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            with torch.no_grad():
                v = self._net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                    **condition,
                ).to(torch.float64)
                x = (1 - t_next) * (x - t_cur * v) + t_next * torch.randn(
                    *x.shape, dtype=torch.float32,
                    device=tensor_kwargs["device"], generator=generator,
                )

        self._decode_and_save(x.float(), video_path)

    def _generate_i2v(self, text_emb, input_data, seed, video_path, w, h) -> None:
        """rCM image-to-video sampling with high/low noise model switch."""
        import torchvision.transforms.v2 as T
        from PIL import Image

        tokenizer = self._tokenizer
        F = input_data.num_frames
        lat_h = h // tokenizer.spatial_compression_factor
        lat_w = w // tokenizer.spatial_compression_factor
        lat_t = tokenizer.get_latent_num_frames(F)

        # --- write image bytes to temp file if needed ---
        if isinstance(input_data.image, bytes):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(input_data.image)
            tmp.close()
            image_path = tmp.name
        else:
            image_path = str(input_data.image)

        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = T.Compose([
                T.ToImage(),
                T.Resize(size=(h, w), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])(img).unsqueeze(0).to(device=tensor_kwargs["device"], dtype=torch.float32)

            with torch.no_grad():
                video_frames = torch.cat(
                    [img_tensor.unsqueeze(2),
                     torch.zeros(1, 3, F - 1, h, w, device=img_tensor.device)],
                    dim=2,
                )
                encoded = tokenizer.encode(video_frames)
                del video_frames
                torch.cuda.empty_cache()
        finally:
            if isinstance(input_data.image, bytes):
                os.unlink(image_path)

        msk = torch.zeros(1, 4, lat_t, lat_h, lat_w,
                          device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
        msk[:, :, 0, :, :] = 1.0
        y = torch.cat([msk, encoded.to(**tensor_kwargs)], dim=1)

        condition = {
            "crossattn_emb": text_emb,
            "y_B_C_T_H_W": y,
        }

        state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]
        generator = torch.Generator(device=tensor_kwargs["device"]).manual_seed(seed)
        x = torch.randn(1, *state_shape, dtype=torch.float32,
                        device=tensor_kwargs["device"], generator=generator)

        t_steps = _build_t_steps(sigma_max=200.0, num_steps=input_data.num_steps, device=x.device)
        x = x.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

        # high-noise model first; swap to low-noise at boundary
        self._high_noise_net.cuda()
        net = self._high_noise_net
        switched = False

        logger.info(f"I2V sampling: {input_data.num_steps} steps (boundary={self._boundary})")
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            if t_cur.item() < self._boundary and not switched:
                self._high_noise_net.cpu()
                torch.cuda.empty_cache()
                self._low_noise_net.cuda()
                net = self._low_noise_net
                switched = True
                logger.info("Switched to low-noise model")

            with torch.no_grad():
                v = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                    **condition,
                ).to(torch.float64)
                x = (1 - t_next) * (x - t_cur * v) + t_next * torch.randn(
                    *x.shape, dtype=torch.float32,
                    device=tensor_kwargs["device"], generator=generator,
                )

        # move whichever model is still on GPU back to CPU
        (self._low_noise_net if switched else self._high_noise_net).cpu()
        torch.cuda.empty_cache()

        self._decode_and_save(x.float(), video_path)

    # ------------------------------------------------------------------
    # shared decode + save
    # ------------------------------------------------------------------
    def _decode_and_save(self, latents: torch.Tensor, video_path: Path) -> None:
        logger.info("VAE decode + save")
        with torch.no_grad():
            video = self._tokenizer.decode(latents)
        video_out = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0
        save_image_or_video(
            rearrange(video_out.unsqueeze(0), "n b c t h w -> c t (n h) (b w)"),
            str(video_path),
            fps=16,
        )
        logger.info(f"Saved {video_path}")
