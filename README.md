# VLGM Local Inference Service

A FastAPI-based local inference service for hosting Visual-Language Generative Models, specifically **Q-Align** (video quality assessment) and **TurboDiffusion** (fast video generation).

## Features

- **Q-Align**: Video quality assessment (1-5 scale)
- **TurboDiffusion**: Fast text-to-video generation with 100-200x acceleration
- Simple REST API with just two endpoints
- Docker support with GPU acceleration
- Auto-loading models on first request

## Quick Start

### Prerequisites

- Python 3.10-3.12
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU with CUDA 12.1+ (required)
- Docker & Docker Compose (optional)

### Installation with uv

```bash
# Clone the repository
git clone https://github.com/yourusername/vlgm-local-inference.git
cd vlgm-local-inference

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
uv pip install -e .
```

### Running the Service

```bash
# Start the server
vlgm-serve

# Or with uvicorn directly
uvicorn vlgm_inference.api.app:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs

## Docker Setup

Requires NVIDIA Docker runtime and drivers >= 525.60.13.

```bash
docker compose up
```

## API Endpoints

### Q-Align: Video Quality Evaluation

**`POST /qalign/evaluate`**

Evaluate the quality of a video.

**Request:**
```json
{
  "video_url": "https://example.com/video.mp4",
  "task": "quality"
}
```

Or with base64:
```json
{
  "video_base64": "/9j/4AAQSkZJRg...",
  "task": "quality"
}
```

**Response:**
```json
{
  "score": 4.23,
  "confidence": {
    "excellent": 0.35,
    "good": 0.45,
    "fair": 0.15,
    "poor": 0.04,
    "bad": 0.01
  },
  "task": "quality"
}
```

**Score interpretation:**
- 5: Excellent
- 4: Good
- 3: Fair
- 2: Poor
- 1: Bad

**Example with curl:**
```bash
curl -X POST http://localhost:8000/qalign/evaluate \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "task": "quality"}'
```

---

### TurboDiffusion: Video Generation

**`POST /turbodiffusion/generate`**

Generate a video from a text prompt.

**Request:**
```json
{
  "prompt": "A cat walking through a sunlit garden with flowers",
  "negative_prompt": "blurry, low quality",
  "num_steps": 4,
  "guidance_scale": 7.5,
  "width": 832,
  "height": 480,
  "num_frames": 81,
  "fps": 16,
  "seed": 42
}
```

**Response:**
```json
{
  "video_url": "/outputs/video_abc12345.mp4",
  "video_path": "./outputs/video_abc12345.mp4",
  "metadata": {
    "prompt": "A cat walking through a sunlit garden with flowers",
    "num_steps": 4,
    "width": 832,
    "height": 480,
    "seed": 42
  }
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the video |
| `negative_prompt` | string | "" | Elements to avoid |
| `num_steps` | int | 4 | Inference steps (4 recommended) |
| `guidance_scale` | float | 7.5 | Prompt adherence strength |
| `width` | int | 832 | Output width in pixels |
| `height` | int | 480 | Output height in pixels |
| `num_frames` | int | 81 | Number of frames |
| `fps` | int | 16 | Frames per second |
| `seed` | int | null | Random seed for reproducibility |

**Example with curl:**
```bash
curl -X POST http://localhost:8000/turbodiffusion/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A dolphin jumping out of the ocean at sunset"}'
```

## Configuration

Configuration via environment variables (prefix: `VLGM_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VLGM_HOST` | `0.0.0.0` | Server host |
| `VLGM_PORT` | `8000` | Server port |
| `VLGM_DEVICE` | `auto` | Device: `cuda` or `auto` |
| `VLGM_TORCH_DTYPE` | `float16` | Model dtype |
| `VLGM_QALIGN_ENABLED` | `true` | Enable Q-Align |
| `VLGM_TURBODIFFUSION_ENABLED` | `true` | Enable TurboDiffusion |
| `VLGM_TURBODIFFUSION_MODEL` | `Wan2.1-1.3B` | Model variant |
| `VLGM_OUTPUT_DIR` | `./outputs` | Output directory |

Create a `.env` file:

```env
VLGM_DEVICE=cuda
VLGM_TORCH_DTYPE=float16
```

## Model Checkpoints

### Q-Align

Automatically downloaded from HuggingFace on first use.

### TurboDiffusion

Download checkpoints from [TurboDiffusion releases](https://github.com/thu-ml/TurboDiffusion/releases) and place in `checkpoints/`:

```
checkpoints/
├── TurboWan2.1-T2V-1.3B-480P.pth
├── Wan2.1_VAE.pth
└── models_t5_umt5-xxl-enc-bf16.pth
```

## Project Structure

```
vlgm-local-inference/
├── src/vlgm_inference/
│   ├── __init__.py
│   ├── config.py
│   ├── models/
│   │   ├── base.py            # ABC base class
│   │   ├── qalign.py          # Q-Align implementation
│   │   └── turbodiffusion.py  # TurboDiffusion implementation
│   └── api/
│       ├── app.py             # FastAPI application
│       ├── schemas.py         # Pydantic models
│       └── routes/
│           ├── qalign.py
│           └── turbodiffusion.py
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Hardware Requirements

- NVIDIA GPU with 8GB+ VRAM (Q-Align)
- NVIDIA GPU with 24GB+ VRAM (TurboDiffusion 1.3B)
- 32GB RAM
- CUDA 12.1+

## License

Please refer to the original repositories for model licenses:
- [Q-Align](https://github.com/Q-Future/Q-Align): Apache 2.0
- [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion): Check repository
