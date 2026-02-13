"""
Text detection module.
Uses EasyOCR for overlay text (avoids PaddlePaddle compatibility issues).
YOLO optional for text-region detection.
"""

import numpy as np
from typing import List, Optional

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class TextDetector:
    """
    Detect and extract overlay text from video frames.
    Uses EasyOCR (PyTorch-based, no PaddlePaddle).
    """

    def __init__(
        self,
        yolo_model: Optional[str] = None,
        use_gpu: bool = True,
    ):
        self._yolo = None
        self._reader = None

        if easyocr is None:
            raise ImportError("Install easyocr: pip install easyocr")
        self._reader = easyocr.Reader(
            ["en"], gpu=use_gpu, verbose=False
        )

        if yolo_model and YOLO is not None:
            self._yolo = YOLO(yolo_model)

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
            results = self._yolo(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
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
