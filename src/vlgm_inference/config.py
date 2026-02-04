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
    turbodiffusion_checkpoint_dir: str = "./checkpoints"
    turbodiffusion_num_steps: int = 4

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
