"""Demo client for Q-Align video quality assessment API."""

import base64
import argparse
from pathlib import Path

import httpx


API_BASE_URL = "http://localhost:8000"


def evaluate_video(video_path: str, task: str = "quality") -> dict:
    """
    Evaluate video quality by uploading a local file to the API.

    Args:
        video_path: Path to the video file
        task: Evaluation task ("quality")

    Returns:
        API response with score
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = httpx.post(
        f"{API_BASE_URL}/qalign/evaluate",
        json={
            "video_base64": video_base64,
            "task": task,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def print_result(result: dict) -> None:
    """Pretty print the evaluation result."""
    print("\n" + "=" * 50)
    print("Q-Align Video Quality Assessment Result")
    print("=" * 50)
    print(f"Task:  {result['task']}")
    print(f"Score: {result['score']:.2f} / 5.00")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Q-Align Video Quality Assessment Demo Client"
    )
    parser.add_argument(
        "video",
        help="Path to the video file to evaluate",
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
        print(f"Evaluating video: {args.video}")
        result = evaluate_video(args.video)
        print_result(result)

    except httpx.HTTPStatusError as e:
        print(f"API Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
