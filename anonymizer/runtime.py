"""Local process ownership and small status files for the dashboard."""

import fcntl
import json
import os
from pathlib import Path


class RoleLock:
    def __init__(self, directory: Path, role: str):
        self.directory, self.role = directory, role
        self.file = None

    def __enter__(self):
        self.file = (self.directory / f"{self.role}.lock").open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError(f"Another {self.role} process is already using this DATA_DIR") from None
        self.status("starting")
        return self

    def status(self, state, **details):
        temporary = self.directory / f"{self.role}.partial.json"
        temporary.write_text(json.dumps({"pid": os.getpid(), "state": state, **details}))
        temporary.replace(self.directory / f"{self.role}.json")

    def __exit__(self, *_):
        self.status("stopped")
        self.file.close()


def worker_status(directory: Path, role: str) -> dict:
    path = directory / f"{role}.lock"
    if not path.exists():
        return {"state": "stopped"}
    with path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                return json.loads((directory / f"{role}.json").read_text())
            except (FileNotFoundError, ValueError):
                return {"state": "starting"}
    return {"state": "stopped"}
