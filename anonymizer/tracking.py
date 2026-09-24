"""BoT-SORT with camera-motion compensation and short gap masks, per video."""

import math
import os
from types import SimpleNamespace

import cv2
import numpy as np

from .config import SETTINGS, Settings

# Runtime downloads and automatic dependency installation stay disabled.
# Keep the library's settings inside the application's private data directory.
SETTINGS.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
os.environ.setdefault("YOLO_CONFIG_DIR", str(SETTINGS.data_dir / ".ultralytics"))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.gmc import GMC


class CameraMotion(GMC):
    """Recover cleanly when a frame cannot provide a valid motion estimate."""

    def __init__(self, method: str):
        super().__init__(method=method)
        self.errors = 0

    def apply(self, image, detections=None):
        try:
            warp = super().apply(image, detections)
            if np.shape(warp) != (2, 3) or not np.isfinite(warp).all():
                raise ValueError("Invalid camera-motion estimate")
            return warp
        except (cv2.error, TypeError, ValueError):
            # Re-seed features on the next frame instead of retaining old ones.
            self.errors += 1
            self.reset_params()
            return np.eye(2, 3)


class FaceTracker:
    def __init__(self, settings: Settings):
        if not 0 < settings.tracker_low_threshold < settings.detector_threshold <= 1:
            raise ValueError("Require 0 < tracker_low_threshold < detector_threshold <= 1")
        if settings.tracker_max_gap_seconds < 0:
            raise ValueError("tracker_max_gap_seconds must be nonnegative")
        if settings.tracker_gmc_method not in ("sparseOptFlow", "none"):
            raise ValueError('tracker_gmc_method must be "sparseOptFlow" or "none"')
        self.settings = settings
        self.tracker = BOTSORT(SimpleNamespace(
            track_high_thresh=settings.detector_threshold,
            track_low_thresh=float(np.nextafter(np.float32(settings.tracker_low_threshold), -np.float32(np.inf))),
            new_track_thresh=settings.detector_threshold,
            track_buffer=30, match_thresh=0.8, fuse_score=True,
            gmc_method=settings.tracker_gmc_method, with_reid=False,
            # Required constructor fields; appearance matching remains disabled.
            proximity_thresh=0.5, appearance_thresh=0.8, model="auto",
        ))
        self.tracker.gmc = CameraMotion(settings.tracker_gmc_method)
        # Our timestamp pruning below replaces the library's frame-count expiry,
        # which can end a track too early in variable-rate footage.
        self.tracker.max_time_lost = math.inf
        self.last_seen: dict[int, float] = {}
        self.previous_view = None
        self.previous_time = None
        self.recovered_detections = self.predicted_boxes = self.scene_resets = 0

    def update(self, image: np.ndarray, boxes: np.ndarray, timestamp: float) -> np.ndarray:
        strong = boxes[boxes[:, 4] >= self.settings.detector_threshold]
        if self.settings.tracker_max_gap_seconds == 0:
            return strong

        # Look at original pixels, before black patches. This inexpensive scene
        # change heuristic can also reset tracks on abrupt camera/lighting changes.
        view = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
        cut = self.previous_view is not None and cv2.absdiff(view, self.previous_view).mean() > 45
        gap = (self.previous_time is not None
               and timestamp - self.previous_time > self.settings.tracker_max_gap_seconds + 1e-9)
        if cut or gap:
            self.tracker.reset()
            self.last_seen.clear()
            self.scene_resets += int(cut)
        self.previous_view, self.previous_time = view, timestamp

        # Expire by media timestamps too: a frame-count buffer alone is not
        # sufficient for variable-rate footage. Prune BEFORE matching a new frame.
        def recent(track):
            return timestamp - self.last_seen[track.track_id] <= self.settings.tracker_max_gap_seconds + 1e-9

        self.tracker.tracked_stracks = [track for track in self.tracker.tracked_stracks if recent(track)]
        self.tracker.lost_stracks = [track for track in self.tracker.lost_stracks
                                    if track.state == TrackState.Lost and recent(track)]
        candidates = boxes[boxes[:, 4] >= self.settings.tracker_low_threshold]
        detections = Boxes(np.column_stack((candidates, np.zeros(len(candidates), dtype=np.float32))), image.shape[:2])
        # Pass original pixels on EVERY frame, including empty detections, so
        # camera motion also corrects predicted boxes during a detection gap.
        self.tracker.update(detections, img=image)

        # Always preserve current strong detections, even when association fails.
        # Low-score boxes may add masks only after matching an existing track.
        masks = list(strong.copy())

        active = self.tracker.tracked_stracks
        lost = [track for track in self.tracker.lost_stracks if track.state == TrackState.Lost]
        for track in active:
            detection = candidates[int(track.idx)]
            if detection[4] < self.settings.detector_threshold:
                masks.append(detection.copy())
                self.recovered_detections += 1
            # A new strong detection is sufficient for a privacy mask immediately,
            # including when a face first appears after frame one.
            track.is_activated = True
            self.last_seen[track.track_id] = timestamp
        for track in active + lost:
            # Keep the filtered/predicted box as well as the current detection:
            # smoothing must never shrink coverage of a newly detected face.
            box = track.xyxy
            height, width = image.shape[:2]
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]:
                masks.append(np.append(box, track.score))
                self.predicted_boxes += int(track.state == TrackState.Lost)
        retained = {track.track_id for track in active + lost}
        self.last_seen = {key: value for key, value in self.last_seen.items() if key in retained}
        return np.asarray(masks, dtype=np.float32).reshape(-1, 5)
