#!/usr/bin/env bash
# =============================================================================
# Download TurboDiffusion model checkpoints from HuggingFace
#
# Usage:
#   bash docker/download_checkpoints.sh                  # default: ./checkpoints
#   bash docker/download_checkpoints.sh /path/to/dir     # custom directory
#
# Downloads (default T2V-1.3B quantized set ≈ 6 GB total):
#   • Wan2.1_VAE.pth                              — shared VAE (Wan2.1 & Wan2.2)
#   • models_t5_umt5-xxl-enc-bf16.pth             — shared umT5 text encoder
#   • TurboWan2.1-T2V-1.3B-480P-quant.pth         — quantized DiT  (RTX 4090/5090)
#
# For H100 / GPUs with 40 GB+ VRAM, uncomment the unquantized variant below.
# For I2V (Wan2.2-A14B), uncomment the I2V section at the bottom.
# =============================================================================
set -euo pipefail

CHECKPOINT_DIR="${1:-./checkpoints}"
mkdir -p "$CHECKPOINT_DIR"

# ---------------------------------------------------------------------------
# Shared weights (used by every TurboDiffusion model variant)
# ---------------------------------------------------------------------------
echo "▸ Downloading Wan2.1 VAE …"
wget -q --show-progress \
    -O "$CHECKPOINT_DIR/Wan2.1_VAE.pth" \
    "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth"

echo "▸ Downloading umT5 text encoder …"
wget -q --show-progress \
    -O "$CHECKPOINT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth"

# ---------------------------------------------------------------------------
# T2V 1.3B — quantized  (RTX 4090 / RTX 5090, < 24 GB VRAM)
# ---------------------------------------------------------------------------
echo "▸ Downloading TurboWan2.1-T2V-1.3B-480P  (quantized) …"
wget -q --show-progress \
    -O "$CHECKPOINT_DIR/TurboWan2.1-T2V-1.3B-480P-quant.pth" \
    "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P-quant.pth"

# ---------------------------------------------------------------------------
# T2V 1.3B — UNQUANTIZED  (H100 / GPUs with 40 GB+ VRAM)
# Uncomment the block below and remove --quant_linear from your config if using
# this variant.  Also update VLGM_TURBODIFFUSION_DIT_PATH accordingly.
# ---------------------------------------------------------------------------
# echo "▸ Downloading TurboWan2.1-T2V-1.3B-480P  (unquantized) …"
# wget -q --show-progress \
#     -O "$CHECKPOINT_DIR/TurboWan2.1-T2V-1.3B-480P.pth" \
#     "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P.pth"

# ---------------------------------------------------------------------------
# I2V (Wan2.2-A14B) — quantized  (RTX 4090 / RTX 5090)
# Uncomment when using image-to-video mode.  Update docker-compose env vars:
#   VLGM_TURBODIFFUSION_MODEL=Wan2.2-A14B
#   VLGM_TURBODIFFUSION_HIGH_NOISE_MODEL_PATH=/app/checkpoints/TurboWan2.2-I2V-A14B-high-720P-quant.pth
#   VLGM_TURBODIFFUSION_LOW_NOISE_MODEL_PATH=/app/checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth
# ---------------------------------------------------------------------------
# echo "▸ Downloading TurboWan2.2-I2V-A14B  high-noise (quantized) …"
# wget -q --show-progress \
#     -O "$CHECKPOINT_DIR/TurboWan2.2-I2V-A14B-high-720P-quant.pth" \
#     "https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-high-720P-quant.pth"
#
# echo "▸ Downloading TurboWan2.2-I2V-A14B  low-noise (quantized) …"
# wget -q --show-progress \
#     -O "$CHECKPOINT_DIR/TurboWan2.2-I2V-A14B-low-720P-quant.pth" \
#     "https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-low-720P-quant.pth"

echo ""
echo "✓  All checkpoints saved to  $CHECKPOINT_DIR"
ls -lh "$CHECKPOINT_DIR"
