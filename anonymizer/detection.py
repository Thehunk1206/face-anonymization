"""Single-pass face detection on full frames and padded black patches."""

import numpy as np

from .config import Settings, resolve_device
from .models import CHECKER_MODEL, DETECTOR_MODEL, validated_model


class YOLOFace:
    """Share inference code; each worker owns its own model and settings."""

    def __init__(self, settings: Settings, *, checking: bool = False):
        # Set Ultralytics' local/offline settings before importing the predictor.
        from .tracking import FaceTracker  # noqa: F401
        from ultralytics import YOLO
        self.settings = settings
        self.device = resolve_device(settings.checker_device if checking else settings.detector_device)
        self.size = settings.checker_size if checking else settings.detector_size
        self.threshold = (settings.checker_threshold if checking else
                          settings.tracker_low_threshold if settings.tracker_max_gap_seconds else settings.detector_threshold)
        name = CHECKER_MODEL if checking else DETECTOR_MODEL
        self.model = YOLO(str(validated_model(settings.model_dir, name)), task="detect")

    def detect_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Run one full-frame prediction per image, grouped into a single batch."""
        if not images:
            return []
        results = self.model.predict(
            images, device=self.device, imgsz=self.size,
            conf=self.threshold, iou=0.5, max_det=1000, augment=False, half=False,
            verbose=False, save=False,
        )
        # Ultralytics maps boxes back to the original image dimensions.
        return [np.column_stack((result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()))
                .astype(np.float32).reshape(-1, 5) for result in results]


def mask_faces(image: np.ndarray, boxes: np.ndarray, padding: float) -> None:
    """Replace each padded face region with solid black pixels before encoding."""
    height, width = image.shape[:2]
    for x1, y1, x2, y2, _ in boxes:
        face_width, face_height = x2 - x1, y2 - y1
        left, top = max(0, int(x1 - padding * face_width)), max(0, int(y1 - padding * face_height))
        right, bottom = min(width, int(np.ceil(x2 + padding * face_width))), min(height, int(np.ceil(y2 + padding * face_height)))
        if right <= left or bottom <= top:
            continue
        image[top:bottom, left:right] = 0
