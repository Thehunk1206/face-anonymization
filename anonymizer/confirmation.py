"""Confirm repeated checker detections without dropping brief privacy findings."""

from collections import Counter, defaultdict, deque
from copy import deepcopy

import cv2
import numpy as np


class TemporalConfirmation:
    """Hold only one window of boxes; count actual detections, never predictions."""

    def __init__(self, window=6, hits=5, context=None):
        if not 2 <= hits <= window:
            raise ValueError("Require 2 <= checker_confirmation_hits <= checker_window_frames")
        self.window, self.hits = window, hits
        self.pending = deque(deepcopy((context or {}).get("tail", [])))
        histogram = (context or {}).get("last_histogram")
        self.last_histogram = None if histogram is None else np.asarray(histogram, dtype=np.float32)

    def recent(self):
        rows = []
        for row in reversed(self.pending):
            rows.append(row)
            if row["scene_cut"]:
                break
        return list(reversed(rows))

    @staticmethod
    def match_score(box, history, frame):
        """Overlap or nearby motion, with a size gate to avoid distant matches."""
        last_frame, last_box = history[-1]
        old = np.asarray(last_box, dtype=float)
        new = np.asarray(box, dtype=float)
        if len(history) > 1:
            before_frame, before_box = history[-2]
            velocity = (old - before_box) / (last_frame - before_frame)
            old = old + velocity * (frame - last_frame)
        old_size, new_size = old[2:] - old[:2], new[2:] - new[:2]
        if min(*old_size, *new_size) <= 0:
            return None
        ratio = new_size / old_size
        if np.any(ratio < 0.5) or np.any(ratio > 2):
            return None
        overlap = np.maximum(0, np.minimum(old[2:], new[2:]) - np.maximum(old[:2], new[:2]))
        intersection = float(np.prod(overlap))
        iou = intersection / (np.prod(old_size) + np.prod(new_size) - intersection)
        distance = np.linalg.norm((old[:2] + old[2:] - new[:2] - new[2:]) / 2)
        distance /= max(np.linalg.norm(old_size), np.linalg.norm(new_size))
        if iou < 0.2 and distance > 0.6:
            return None
        return float(iou - distance)

    def add(self, row, image):
        view = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
        # Persist color statistics, not a frame thumbnail, in downloadable reports.
        histogram = np.concatenate([cv2.calcHist([view], [channel], None, [16], [0, 256]).ravel()
                                    for channel in range(3)]) / view.size
        row["scene_cut"] = bool(self.last_histogram is not None and
                                cv2.compareHist(histogram, self.last_histogram, cv2.HISTCMP_BHATTACHARYYA) > 0.65)
        self.last_histogram = histogram
        history = defaultdict(list)
        if not row["scene_cut"]:
            for previous in self.recent():
                for face in previous["faces"]:
                    history[face["candidate_id"]].append((previous["global_frame"], face["box"]))

        matches = []
        for candidate, observations in history.items():
            # Permit one missed frame, but never link across a longer absence.
            if row["global_frame"] - observations[-1][0] > 2:
                continue
            for index, face in enumerate(row["faces"]):
                score = self.match_score(face["box"], observations[-2:], row["global_frame"])
                if score is not None:
                    matches.append((score, candidate, index))
        used_candidates, used_faces = set(), set()
        for _, candidate, index in sorted(matches, reverse=True):
            if candidate not in used_candidates and index not in used_faces:
                row["faces"][index]["candidate_id"] = candidate
                used_candidates.add(candidate)
                used_faces.add(index)
        for index, face in enumerate(row["faces"]):
            face.setdefault("candidate_id", f"{row['global_frame']}:{index}")
            face["confirmation"] = "unconfirmed"

        self.pending.append(row)
        recent = self.recent()
        counts = Counter(face["candidate_id"] for item in recent for face in item["faces"])
        for item in recent:
            for face in item["faces"]:
                if counts[face["candidate_id"]] >= self.hits:
                    face["confirmation"] = "persistent"
        # The first frame cannot gain further support after this window closes.
        return self.pending.popleft() if len(self.pending) >= self.window else None

    def context(self):
        return {"tail": deepcopy(list(self.pending)),
                "last_histogram": self.last_histogram.tolist() if self.last_histogram is not None else None}

    def finish(self):
        while self.pending:
            yield self.pending.popleft()


def confirmation_counts(findings):
    counts = {"persistent_frames": 0, "unconfirmed_frames": 0,
              "persistent_detections": 0, "unconfirmed_detections": 0}
    for finding in findings:
        labels = Counter(face.get("confirmation", "unconfirmed") for face in finding["faces"])
        for label in ("persistent", "unconfirmed"):
            counts[label + "_frames"] += int(labels[label] > 0)
            counts[label + "_detections"] += labels[label]
    return counts
