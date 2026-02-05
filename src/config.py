"""Configuration settings for the inference service."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Model Settings
    device: Literal["cuda", "cpu", "auto"] = "auto"
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "float16"

    # Q-Align Settings
    qalign_model_id: str = "q-future/one-align"
    qalign_enabled: bool = True

    # TurboDiffusion Settings
    turbodiffusion_enabled: bool = True
    turbodiffusion_model: str = "Wan2.1-1.3B"
    turbodiffusion_dit_path: str = "./checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth"
    turbodiffusion_vae_path: str = "./checkpoints/Wan2.1_VAE.pth"
    turbodiffusion_text_encoder_path: str = "./checkpoints/models_t5_umt5-xxl-enc-bf16.pth"
    turbodiffusion_attention_type: str = "sagesla"
    turbodiffusion_sla_topk: float = 0.1
    turbodiffusion_quant_linear: bool = True
    turbodiffusion_default_norm: bool = False
    # TurboDiffusion I2V (only used when model is Wan2.2-A14B)
    turbodiffusion_high_noise_model_path: str = ""
    turbodiffusion_low_noise_model_path: str = ""
    turbodiffusion_boundary: float = 0.9

    # FastVideo Settings
    fastvideo_enabled: bool = True
    fastvideo_model_id: str = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"

    # Cache Settings
    model_cache_dir: str = "./model_cache"
    output_dir: str = "./outputs"

    model_config = {
        "env_prefix": "VLGM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
