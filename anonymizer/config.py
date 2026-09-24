"""Small, deployment-level settings; processing defaults are deliberately conservative."""

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", "models"))
    detector_device: str = os.getenv("DETECTOR_DEVICE", os.getenv("DEVICE", "auto"))
    checker_device: str = os.getenv("CHECKER_DEVICE", os.getenv("DEVICE", "auto"))
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_MB", "4096")) * 1024 * 1024
    detector_threshold: float = 0.65
    checker_threshold: float = 0.7
    detector_size: int = 960
    checker_size: int = 960
    chunk_seconds: float = float(os.getenv("CHUNK_SECONDS", "30"))
    padding: float = 0.2  # Add 20% of the box width/height on EACH side.
    detection_batch_size: int = 4
    checking_batch_size: int = 4
    checker_window_frames: int = 6
    checker_confirmation_hits: int = 5
    tracker_low_threshold: float = 0.2  # Only helps match an existing face track.
    tracker_max_gap_seconds: float = 0.3  # Set to 0 to disable tracking.
    tracker_gmc_method: str = "sparseOptFlow"  # Camera-motion compensation; "none" disables it.

    @property
    def database(self) -> Path:
        return self.data_dir / "jobs.sqlite3"

    def prepare(self) -> None:
        if self.chunk_seconds <= 0 or min(self.detection_batch_size, self.checking_batch_size) < 1:
            raise ValueError("Chunk duration and batch sizes must be positive")
        if not 2 <= self.checker_confirmation_hits <= self.checker_window_frames:
            raise ValueError("Require 2 <= checker_confirmation_hits <= checker_window_frames")
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for directory in ("uploads", "outputs", "workers"):
            (self.data_dir / directory).mkdir(exist_ok=True, mode=0o700)

    def processing(self) -> dict:
        """Persist inference choices, keeping storage paths deployment-specific."""
        return {key: value for key, value in asdict(self).items()
                if key not in {"data_dir", "model_dir", "max_upload_bytes", "detector_device", "checker_device"}}

    def for_job(self, values: dict):
        return replace(self, **{key: value for key, value in values.items() if key in self.processing()})


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; select cpu for this worker")
    return requested


SETTINGS = Settings()
