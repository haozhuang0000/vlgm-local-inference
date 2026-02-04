"""Demo client for TurboDiffusion video generation API."""

import httpx
import argparse
from pathlib import Path


API_BASE_URL = "http://localhost:8000"


def generate_video(
    prompt: str,
    negative_prompt: str = "",
    num_steps: int = 4,
    guidance_scale: float = 7.5,
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    fps: int = 16,
    seed: int | None = None,
) -> dict:
    """
    Generate a video from a text prompt.

    Args:
        prompt: Text description of the video to generate
        negative_prompt: Elements to avoid in generation
        num_steps: Number of inference steps (4 recommended)
        guidance_scale: How closely to follow the prompt
        width: Output video width
        height: Output video height
        num_frames: Number of frames to generate
        fps: Frames per second
        seed: Random seed for reproducibility

    Returns:
        API response with video URL and metadata
    """
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
    }
    if seed is not None:
        payload["seed"] = seed

    response = httpx.post(
        f"{API_BASE_URL}/turbodiffusion/generate",
        json=payload,
        timeout=600.0,  # Video generation can take a while
    )
    response.raise_for_status()
    return response.json()


def download_video(video_url: str, output_path: str) -> None:
    """Download the generated video to a local file."""
    full_url = f"{API_BASE_URL}{video_url}" if video_url.startswith("/") else video_url
    response = httpx.get(full_url, timeout=60.0)
    response.raise_for_status()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Video saved to: {output_path}")


def print_result(result: dict) -> None:
    """Pretty print the generation result."""
    print("\n" + "=" * 50)
    print("TurboDiffusion Video Generation Result")
    print("=" * 50)
    print(f"Video URL: {result['video_url']}")
    print(f"Video Path: {result['video_path']}")
    print("\nMetadata:")
    for key, value in result["metadata"].items():
        print(f"  {key}: {value}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="TurboDiffusion Video Generation Demo Client"
    )
    parser.add_argument(
        "prompt",
        help="Text description of the video to generate",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, distorted",
        help="Elements to avoid (default: 'blurry, low quality, distorted')",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of inference steps (default: 4)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (default: 832)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (default: 480)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=81,
        help="Number of frames (default: 81)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Download video to this path",
    )
    parser.add_argument(
        "--api-url",
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})",
    )

    args = parser.parse_args()

    global API_BASE_URL
    API_BASE_URL = args.api_url

    try:
        print(f"Generating video for prompt: {args.prompt}")
        print("This may take a while...")

        result = generate_video(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            fps=args.fps,
            seed=args.seed,
        )

        print_result(result)

        if args.output:
            download_video(result["video_url"], args.output)

    except httpx.HTTPStatusError as e:
        print(f"API Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
