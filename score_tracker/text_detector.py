"""
Text detection module.
YOLO first layer for region detection, EasyOCR for text extraction.
Reduces stadium ad text by focusing on overlay zones.
"""

import warnings
import numpy as np
from typing import List, Optional, Tuple

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Default YOLO model for text/overlay region detection (Hugging Face)
DEFAULT_YOLO_MODEL = "keremberke/yolov8n-table-extraction"


class TextDetector:
    """
    Detect and extract overlay text from video frames.
    YOLO detects regions first, EasyOCR extracts text from crops.
    """

    def __init__(
        self,
        yolo_model: Optional[str] = DEFAULT_YOLO_MODEL,
        use_gpu: bool = True,
        overlay_zones_only: bool = False,
        overlay_zone_ratio: float = 0.25,
    ):
        self._yolo = None
        self._reader = None
        self._overlay_zones_only = overlay_zones_only
        self._overlay_zone_ratio = overlay_zone_ratio

        if easyocr is None:
            raise ImportError("Install easyocr: pip install easyocr")
        self._reader = easyocr.Reader(
            ["en"], gpu=use_gpu, verbose=False
        )

        if yolo_model is not None and YOLO is not None:
            try:
                self._yolo = YOLO(yolo_model)
            except Exception as e:
                warnings.warn(
                    f"YOLO model '{yolo_model}' failed to load ({e}). "
                    "Falling back to full-frame OCR. Use --no-yolo to skip.",
                    UserWarning,
                )
                self._yolo = None

    def _in_overlay_zone(self, y1: int, y2: int, frame_h: int) -> bool:
        """Check if box center is in top or bottom overlay zones."""
        if not self._overlay_zones_only:
            return True
        center_y = (y1 + y2) / 2
        top_threshold = frame_h * self._overlay_zone_ratio
        bottom_threshold = frame_h * (1 - self._overlay_zone_ratio)
        return center_y < top_threshold or center_y > bottom_threshold

    def _get_boxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Run YOLO and return filtered boxes (x1, y1, x2, y2)."""
        results = self._yolo(frame, verbose=False)
        h = frame.shape[0]
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if self._in_overlay_zone(y1, y2, h):
                    boxes.append((x1, y1, x2, y2))
        return boxes

    def detect(self, frame: np.ndarray) -> List[str]:
        """
        Detect overlay text in a frame and return extracted strings.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            List of detected text strings
        """
        texts: List[str] = []

        def _run_ocr(img: np.ndarray) -> List[str]:
            result = self._reader.readtext(img)
            return [t[1] for t in result if t[1] and str(t[1]).strip()]

        if self._yolo is not None:
            for x1, y1, x2, y2 in self._get_boxes(frame):
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    texts.extend(_run_ocr(crop))
        else:
            texts.extend(_run_ocr(frame))

        return texts


def detect_overlay_text(
    frame: np.ndarray,
    detector: Optional["TextDetector"] = None,
) -> List[str]:
    """
    Convenience function to detect overlay text in a frame.

    Args:
        frame: BGR image
        detector: Optional pre-instantiated TextDetector

    Returns:
        List of detected text strings
    """
    if detector is None:
        detector = TextDetector()
    return detector.detect(frame)
