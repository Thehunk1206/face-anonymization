"""Two small worker loops: upload anonymization and independent chunk checking."""

import json
import logging
import signal
from pathlib import Path
from threading import Event

import cv2
import torch

from .config import SETTINGS, Settings
from .database import PIPELINE_VERSION, Jobs
from .detection import YOLOFace
from .media import assemble_chunks, chunk_video, publish_file
from .models import PREVIOUS_MODELS, snapshot
from .quality import check_video, combine_reports, update_previous_reports
from .runtime import RoleLock
from .video import anonymize

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, role: str, jobs: Jobs, settings: Settings, ownership: RoleLock):
        self.role, self.jobs, self.settings, self.ownership = role, jobs, settings, ownership
        self.stop = Event()
        self.model = self.model_settings = None

    def load_model(self, job):
        saved = json.loads(job["settings_json"])
        if job["pipeline_version"] != PIPELINE_VERSION or saved["models"] != snapshot(self.settings)["models"]:
            raise ValueError("Job uses a different model/pipeline version; restore it or submit a new upload ID")
        settings = self.settings.for_job(saved["settings"])
        if saved["settings"] != self.model_settings:
            self.ownership.status("loading", upload_id=job["upload_id"])
            self.model = None
            self.model = YOLOFace(settings, checking=self.role == "check")
            self.model_settings = saved["settings"]
        self.ownership.status("running", upload_id=job["upload_id"], device=self.model.device)
        return settings

    def redact(self, job):
        upload_id = job["upload_id"]
        directory = self.settings.data_dir / "outputs" / upload_id
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        settings = self.load_model(job)
        stats = anonymize(
            Path(job["input_path"]), directory, self.model, settings, self.stop,
            lambda done, total: self.jobs.update(upload_id, frames_done=done, total_frames=total),
            job["chunks"], lambda index, stats: self.jobs.publish_chunk(upload_id, index, stats),
        )
        assemble_chunks(directory, self.jobs.chunks(upload_id), stats, self.stop)
        self.jobs.update(upload_id, redaction_status="completed", redaction_stats=json.dumps(stats))
        logger.info("Anonymized %s: %s frames in %s chunks", upload_id, stats["frames"], stats["chunk_count"])

    def check(self, item):
        upload_id, index = item["upload_id"], item["chunk_index"]
        job = self.jobs.get(upload_id)
        self.load_model(job)
        chunk = job["chunks"][index]
        directory = self.settings.data_dir / "outputs" / upload_id
        report = chunk_video(directory, index).with_suffix(".json")
        temporary = report.with_suffix(".partial.json")
        context = None
        if index and job["chunks"][index - 1]["status"] in ("passed", "flagged"):
            previous = json.loads(chunk_video(directory, index - 1).with_suffix(".json").read_text())
            context = previous.get("temporal_context")
        try:
            summary = check_video(
                chunk_video(directory, index), temporary, self.model, self.stop,
                lambda done, _: self.jobs.update_chunk(upload_id, index, frames_done=done),
                expected=chunk["stats"], context=context, chunk_index=index,
            )
            publish_file(temporary, report)
            update_previous_reports(directory, summary["boundary_updates"])
            self.jobs.update_chunk(upload_id, index, status="flagged" if summary["flagged_frames"] else "passed",
                                   flagged_frames=summary["flagged_frames"], suspected_faces=summary["suspected_faces"])
            logger.info("Checked %s chunk %s: %s flags", upload_id, index, summary["flagged_frames"])
        finally:
            temporary.unlink(missing_ok=True)

    def run(self):
        self.jobs.upgrade_checker(snapshot(self.settings), PREVIOUS_MODELS)
        self.jobs.recover(self.role)
        while not self.stop.is_set():
            self.ownership.status("idle")
            # Finish ready jobs before taking another chunk from a busy queue.
            item = self.jobs.ready_to_finalize() if self.role == "check" else None
            finalizing = item is not None
            if item is None:
                item = (self.jobs.claim_upload(snapshot(self.settings)) if self.role == "redact"
                        else self.jobs.claim_chunk())
            if item is None:
                self.stop.wait(0.5)
                continue
            upload_id = item["upload_id"]
            try:
                if finalizing:
                    self.ownership.status("finalizing", upload_id=upload_id)
                    summary = combine_reports(self.settings.data_dir / "outputs" / upload_id, item, self.stop)
                    self.jobs.update(upload_id, status="needs_review" if summary["flagged_frames"] else "completed")
                elif self.role == "redact":
                    self.redact(item)
                else:
                    self.check(item)
            except InterruptedError:
                return  # Active work is reclaimed by this role at its next startup.
            except Exception as error:
                logger.exception("%s failed for %s", self.role, upload_id)
                message = str(error)[:1000]
                if self.role == "redact":
                    self.jobs.update(upload_id, status="failed", redaction_status="failed", redaction_error=message, error=message)
                else:
                    if not finalizing:
                        self.jobs.update_chunk(upload_id, item["chunk_index"], status="failed", error=message)
                    self.jobs.update(upload_id, status="failed", checking_error=message, error=message)


def run_worker(role):
    SETTINGS.prepare()
    jobs = Jobs(SETTINGS.database)
    jobs.initialize()
    with RoleLock(SETTINGS.data_dir / "workers", role) as ownership:
        cv2.setNumThreads(2)
        torch.set_num_threads(2)
        worker = Worker(role, jobs, SETTINGS, ownership)
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: worker.stop.set())
        worker.run()
