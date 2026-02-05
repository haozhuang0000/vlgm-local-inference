"""Simple example to call FastVideo API."""

import requests
import base64

# API endpoint
URL = "http://localhost:8002/fastvideo/generate"

# Request payload
payload = {
    "prompt": "A cat walking through a garden",
    "num_frames": 49,
    "height": 480,
    "width": 832,
    "num_inference_steps": 8,
}

# Make request
response = requests.post(URL, json=payload)
response.raise_for_status()

# Save video
data = response.json()
video_bytes = base64.b64decode(data["video_base64"])
with open("output.mp4", "wb") as f:
    f.write(video_bytes)

print(f"Video saved to output.mp4")
print(f"Metadata: {data['metadata']}")
