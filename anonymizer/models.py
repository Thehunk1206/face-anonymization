"""Download once; validate model bytes before inference. Videos never leave this machine."""

import hashlib
import shutil
from pathlib import Path
from urllib.request import urlopen

from .config import SETTINGS

DETECTOR_MODEL = "yolov12s-face.pt"
CHECKER_MODEL = "yolov12m-face.pt"
MODELS = {
    DETECTOR_MODEL: (
        "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov12s-face.pt",
        "2dc59c2b90fed383f7d5ca9d202efd7f147a25f1d688c15eeb7779426c09e63e",
    ),
    CHECKER_MODEL: (
        "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov12m-face.pt",
        "a55f2842ef97b09ec4e6fb166ff6fb27af8515e9dd93bda878b4eab98abb2bc2",
    ),
}

# Recognize unfinished jobs from the previous checker without loading that model.
PREVIOUS_MODELS = {
    DETECTOR_MODEL: MODELS[DETECTOR_MODEL][1],
    "detection_mobilenet0.25_Final.pth": "2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220",
}


def snapshot(settings) -> dict:
    return {"settings": settings.processing(), "models": {name: checksum for name, (_, checksum) in MODELS.items()}}


def validated_model(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}. Run: python -m anonymizer.models")
    with path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    if checksum != MODELS[name][1]:
        raise ValueError(f"Checksum mismatch for {name}; download the model again")
    return path


def main() -> None:
    SETTINGS.model_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, _) in MODELS.items():
        path = SETTINGS.model_dir / name
        if not path.exists():
            temporary = path.with_suffix(".download")
            try:
                print(f"Downloading {name}", flush=True)
                with urlopen(url, timeout=60) as response, temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
                with temporary.open("rb") as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != MODELS[name][1]:
                        raise ValueError(f"Checksum mismatch for downloaded {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        validated_model(SETTINGS.model_dir, name)
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
