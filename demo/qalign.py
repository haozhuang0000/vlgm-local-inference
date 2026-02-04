# """Demo client for Q-Align video quality assessment API."""
#
# import base64
# import httpx
# import argparse
# from pathlib import Path
#
#
# API_BASE_URL = "http://localhost:8000"
#
#
# def evaluate_video_from_file(video_path: str, task: str = "quality") -> dict:
#     """
#     Evaluate video quality from a local file.
#
#     Args:
#         video_path: Path to the video file
#         task: Evaluation task ("quality")
#
#     Returns:
#         API response with score and confidence
#     """
#     video_path = Path(video_path)
#     if not video_path.exists():
#         raise FileNotFoundError(f"Video file not found: {video_path}")
#
#     # Read and encode video as base64
#     with open(video_path, "rb") as f:
#         video_data = f.read()
#     video_base64 = base64.b64encode(video_data).decode("utf-8")
#
#     # Call API
#     response = httpx.post(
#         f"{API_BASE_URL}/qalign/evaluate",
#         json={
#             "video_base64": video_base64,
#             "task": task,
#         },
#         timeout=120.0,  # Video processing may take time
#     )
#     response.raise_for_status()
#     return response.json()
#
#
# def evaluate_video_from_url(video_url: str, task: str = "quality") -> dict:
#     """
#     Evaluate video quality from a URL.
#
#     Args:
#         video_url: URL of the video
#         task: Evaluation task ("quality")
#
#     Returns:
#         API response with score and confidence
#     """
#     response = httpx.post(
#         f"{API_BASE_URL}/qalign/evaluate",
#         json={
#             "video_url": video_url,
#             "task": task,
#         },
#         timeout=120.0,
#     )
#     response.raise_for_status()
#     return response.json()
#
#
# def print_result(result: dict) -> None:
#     """Pretty print the evaluation result."""
#     print("\n" + "=" * 50)
#     print("Q-Align Video Quality Assessment Result")
#     print("=" * 50)
#     print(f"Task: {result['task']}")
#     print(f"Score: {result['score']:.2f} / 5.00")
#     print("\nConfidence Distribution:")
#     for level, confidence in result["confidence"].items():
#         bar = "█" * int(confidence * 30)
#         print(f"  {level:10s}: {confidence:.2%} {bar}")
#     print("=" * 50)
#
#
# def main():
#     try:
#         result = evaluate_video_from_file("./m2-res_1080p.mp4")
#
#         print_result(result)
#
#     except httpx.HTTPStatusError as e:
#         print(f"API Error: {e.response.status_code} - {e.response.text}")
#     except Exception as e:
#         print(f"Error: {e}")
#
#
# if __name__ == "__main__":
#     main()

from q_align import QAlignVideoScorer, load_video

scorer = QAlignVideoScorer()
video_list = [load_video("m2-res_1080p.mp4")]
print(scorer(video_list).tolist())