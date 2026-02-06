from contextlib import contextmanager
from PIL import Image


@contextmanager
def video_reader_context(video_file):
    """Context manager for VideoReader to ensure proper resource cleanup."""
    from decord import VideoReader
    vr = VideoReader(video_file)
    try:
        yield vr
    finally:
        del vr


def load_video(video_file):
    with video_reader_context(video_file) as vr:
        # Get video frame rate
        fps = vr.get_avg_fps()

        # Calculate frame indices for 1fps
        frame_indices = [int(fps * i) for i in range(int(len(vr) / fps))]
        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(frames[i]) for i in range(int(len(vr) / fps))]