# Demo Clients

Demo scripts to test the VLGM Local Inference API.

## Prerequisites

```bash
pip install httpx
```

## Q-Align: Video Quality Assessment

Evaluate the quality of a video file or URL.

```bash
# Evaluate a local video file
python demo/qalign.py path/to/video.mp4

# Evaluate a video from URL
python demo/qalign.py --url "https://example.com/video.mp4"

# Use a different API server
python demo/qalign.py --api-url http://192.168.1.100:8000 video.mp4
```

### Example Output

```
Evaluating video from file: sample.mp4

==================================================
Q-Align Video Quality Assessment Result
==================================================
Task: quality
Score: 4.23 / 5.00

Confidence Distribution:
  excellent : 35.00% ██████████
  good      : 45.00% █████████████
  fair      : 15.00% ████
  poor      :  4.00% █
  bad       :  1.00%
==================================================
```

## TurboDiffusion: Video Generation

Generate videos from text prompts.

```bash
# Basic generation
python demo/turbodiffusion.py "A cat walking through a garden"

# With options
python demo/turbodiffusion.py "A dolphin jumping at sunset" \
    --negative-prompt "blurry, distorted" \
    --steps 4 \
    --seed 42 \
    --output generated_video.mp4

# Full options
python demo/turbodiffusion.py "A rocket launching into space" \
    --negative-prompt "low quality" \
    --steps 4 \
    --guidance 7.5 \
    --width 832 \
    --height 480 \
    --frames 81 \
    --fps 16 \
    --seed 12345 \
    --output rocket.mp4
```

### Example Output

```
Generating video for prompt: A cat walking through a garden
This may take a while...

==================================================
TurboDiffusion Video Generation Result
==================================================
Video URL: /outputs/video_abc12345.mp4
Video Path: ./outputs/video_abc12345.mp4

Metadata:
  prompt: A cat walking through a garden
  num_steps: 4
  width: 832
  height: 480
  seed: 42
==================================================
Video saved to: generated_video.mp4
```

## Python Usage

You can also import the functions directly:

```python
from demo.qalign import evaluate_video_from_file, evaluate_video_from_url
from demo.turbodiffusion import generate_video, download_video

# Q-Align
result = evaluate_video_from_file("video.mp4")
print(f"Quality score: {result['score']}")

# TurboDiffusion
result = generate_video(prompt="A beautiful sunset over the ocean")
download_video(result["video_url"], "sunset.mp4")
```
