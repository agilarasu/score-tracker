"""
Main pipeline: MP4 → frame extraction → text detection → txt storage.
Future: swap storage for LLM.
"""

from pathlib import Path
from typing import Optional, Union

from .video_processor import extract_frames
from .text_detector import TextDetector
from .storage import write_detection, DELIMITER


def run_pipeline(
    video_path: Union[str, Path],
    output_txt_path: Union[str, Path],
    *,
    interval_seconds: Optional[float] = None,
    interval_frames: Optional[int] = 30,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    yolo_model: Optional[str] = None,
    use_gpu: bool = True,
) -> None:
    """
    Process MP4 video: detect overlay text every n sec/frame, store in txt.

    Args:
        video_path: Path to MP4 file
        output_txt_path: Path to output .txt file
        interval_seconds: Extract 1 frame every N seconds
        interval_frames: Extract 1 frame every N frames (used if interval_seconds is None)
        start_time: Start timestamp (seconds)
        end_time: End timestamp (None = full video)
        yolo_model: Optional YOLO model for text-region detection
        use_gpu: Use GPU for OCR/YOLO
    """
    video_path = Path(video_path)
    output_txt_path = Path(output_txt_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    detector = TextDetector(yolo_model=yolo_model, use_gpu=use_gpu)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for frame_idx, timestamp, frame in extract_frames(
            video_path,
            interval_seconds=interval_seconds,
            interval_frames=interval_frames,
            start_time=start_time,
            end_time=end_time,
        ):
            texts = detector.detect(frame)
            write_detection(f, frame_idx, timestamp, texts, delimiter=DELIMITER)
