#!/usr/bin/env python3
"""
CLI entrypoint for Football Score Tracker.
Flow: MP4 → detect overlay text every n sec/frame → store in txt.
"""

import argparse
from pathlib import Path

from score_tracker.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Detect overlay text from football match videos using YOLO"
    )
    parser.add_argument("video", type=Path, help="Path to MP4 video file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output txt file (default: <video_stem>_detections.txt)",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=None,
        help="Extract 1 frame every N seconds (e.g., 5.0)",
    )
    parser.add_argument(
        "--interval-frames",
        type=int,
        default=30,
        help="Extract 1 frame every N frames (default: 30, ~1 sec at 30fps)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds (default: full video)",
    )

    args = parser.parse_args()
    video_path = args.video
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output = args.output or video_path.with_name(
        f"{video_path.stem}_detections.txt"
    )
    run_pipeline(
        video_path,
        output,
        interval_seconds=args.interval_sec,
        interval_frames=args.interval_frames,
        start_time=args.start,
        end_time=args.end,
    )
    print(f"Done. Output: {output}")


if __name__ == "__main__":
    main()
