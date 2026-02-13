"""
Storage module for detected overlay text.
Writes to a text file with delimiter and time/frame metadata.
"""

from pathlib import Path
from typing import List, Optional, TextIO


DELIMITER = "============="


def write_detection(
    f: TextIO,
    frame_number: int,
    timestamp_seconds: float,
    texts: List[str],
    delimiter: str = DELIMITER,
) -> None:
    """
    Append one detection block to an open file.

    Args:
        f: Open file handle (text mode)
        frame_number: Frame index in the video
        timestamp_seconds: Timestamp in seconds
        texts: List of detected text strings
        delimiter: Separator between blocks (default: =============)
    """
    block = [
        delimiter,
        f"frame={frame_number}",
        f"time={timestamp_seconds:.2f}s",
        "",
        *[t for t in texts if t],
        "",
    ]
    f.write("\n".join(block) + "\n")
    f.flush()


def save_to_file(
    output_path: Path,
    entries: List[dict],
    delimiter: str = DELIMITER,
) -> None:
    """
    Write all detection entries to a file (batch mode).

    Args:
        output_path: Path to output .txt file
        entries: List of dicts with keys: frame_number, timestamp_seconds, texts
        delimiter: Separator between blocks
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for e in entries:
            write_detection(
                f,
                e["frame_number"],
                e["timestamp_seconds"],
                e["texts"],
                delimiter,
            )
