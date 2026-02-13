"""
Video processing module.
Extracts frames from MP4 at configurable intervals (time-based or frame-based).
"""

import os
import cv2
from pathlib import Path
from typing import Generator, Tuple, Union


def _read_frame_quiet(cap) -> Tuple[bool, "cv2.Mat"]:
    """Read a frame while suppressing FFmpeg H.264 NAL warnings on stderr."""
    try:
        stderr_fd = 2
        devnull = open(os.devnull, "w")
        saved_stderr = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stderr_fd)
    except OSError:
        return cap.read()
    try:
        return cap.read()
    finally:
        try:
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stderr)
            devnull.close()
        except OSError:
            pass


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

    # Read sequentially (no seeking) to avoid H.264 NAL unit errors
    next_yield_frame = start_frame
    frame_idx = 0

    while frame_idx < end_frame:
        ret, frame = _read_frame_quiet(cap)
        if not ret:
            break
        if frame_idx >= next_yield_frame:
            timestamp = frame_idx / fps
            yield frame_idx, timestamp, frame
            next_yield_frame += interval_frame_count
        frame_idx += 1

    cap.release()
