"""
Video processing module.
Extracts frames from MP4 at configurable intervals (time-based or frame-based).
"""

import cv2
from pathlib import Path
from typing import Generator, Tuple, Union


def extract_frames(
    video_path: Union[str, Path],
    interval_seconds: float = None,
    interval_frames: int = None,
    start_time: float = 0.0,
    end_time: float = None,
) -> Generator[Tuple[int, float, "cv2.Mat"], None, None]:
    """
    Extract frames from an MP4 video at regular intervals.

    Args:
        video_path: Path to the MP4 file
        interval_seconds: Extract 1 frame every N seconds (e.g., 5.0 = every 5 sec)
        interval_frames: Extract 1 frame every N frames (alternative to interval_seconds)
        start_time: Start from this timestamp in seconds
        end_time: Stop at this timestamp (None = end of video)

    Yields:
        Tuples of (frame_number, timestamp_seconds, frame_bgr)
    """
    if interval_seconds is None and interval_frames is None:
        interval_frames = 30  # default: every 30 frames (~1 sec at 30fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if interval_seconds is not None:
        interval_frame_count = max(1, int(interval_seconds * fps))
    else:
        interval_frame_count = max(1, interval_frames)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time is not None else total_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        yield frame_idx, timestamp, frame
        frame_idx += interval_frame_count
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()
