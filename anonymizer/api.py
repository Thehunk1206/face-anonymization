"""Upload, progress, local previews, and checked downloads; workers run separately."""

import os
import re
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import SETTINGS
from .database import Jobs
from .media import chunk_video
from .models import snapshot
from .runtime import RoleLock, worker_status

jobs = Jobs(SETTINGS.database)


@asynccontextmanager
async def lifespan(app: FastAPI):
    SETTINGS.prepare()
    jobs.initialize()
    with RoleLock(SETTINGS.data_dir / "workers", "api"):
        yield  # API startup never reclaims worker-owned tasks.


app = FastAPI(title="Face Anonymizer", lifespan=lifespan, description=(
    "Upload with a stable upload_id. YOLO publishes silent 30-second chunks; "
    "YOLOv12m-face checks the encoded chunks. Early previews are local operator review only. "
    "Downloads require a complete check without flags. A clear scan does not guarantee anonymity."
))
static_dir = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class ChunkStatus(BaseModel):
    chunk_index: int
    status: Literal["queued", "checking", "passed", "flagged", "failed"]
    start_seconds: float
    duration_seconds: float
    start_frame: int
    total_frames: int
    frames_done: int
    flagged_frames: int
    suspected_faces: int
    error: str | None


class JobStatus(BaseModel):
    upload_id: str
    filename: str | None = None
    pipeline_version: str | None = None
    status: Literal["queued", "processing", "checking", "completed", "needs_review", "failed"]
    frames_done: int
    total_frames: int | None
    redaction_status: str | None = None
    checking_status: str
    checking_frames: int
    checked_chunks: int
    flagged_frames: int
    full_preview_ready: bool
    redaction_error: str | None = None
    checking_error: str | None = None
    error: str | None
    created_at: str
    updated_at: str
    chunks: list[ChunkStatus]


def existing_job(upload_id):
    job = jobs.get(upload_id)
    if job is None:
        raise HTTPException(404, "Unknown upload ID")
    return job


def public_job(job):
    return {**job, "full_preview_ready": job["redaction_status"] == "completed" or (
                job["pipeline_version"] is None and job["status"] in ("completed", "needs_review")),
            "chunks": [{**chunk, **{key: chunk["stats"][key] for key in
                        ("start_seconds", "duration_seconds", "start_frame")},
                        "total_frames": chunk["stats"]["frames"]} for chunk in job["chunks"]]}


def require_local(request):
    try:
        local = request.client is not None and ip_address(request.client.host).is_loopback
    except ValueError:
        local = False
    if not local:
        raise HTTPException(403, "Video review is available only on this computer")


def available_file(path, media_type, filename=None):
    if not path.is_file():
        raise HTTPException(410, "This artifact is no longer available. Upload again with a new upload ID.")
    return FileResponse(path, media_type=media_type, filename=filename, headers={"Cache-Control": "no-store"})


def existing_chunk(upload_id, index):
    job = existing_job(upload_id)
    chunk = next((chunk for chunk in job["chunks"] if chunk["chunk_index"] == index), None)
    if chunk is None:
        raise HTTPException(404, "Chunk is not published yet")
    return chunk


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/jobs", response_model=list[JobStatus])
def recent_jobs():
    return [public_job(job) for job in jobs.recent()]


@app.post("/jobs", response_model=JobStatus, status_code=202)
def submit(upload_id: Annotated[str, Form(description="Stable event ID; 1–80 letters, digits, _ or -")],
           video: Annotated[UploadFile, File()]):
    path = None
    queued = False
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", upload_id):
            raise HTTPException(422, "upload_id must contain 1–80 letters, numbers, underscores or hyphens")
        previous = jobs.get(upload_id)
        if previous:
            return public_job(previous)
        path = SETTINGS.data_dir / "uploads" / f"{uuid4().hex}.video"
        size = 0
        with path.open("xb") as target:
            while chunk := video.file.read(1024 * 1024):
                size += len(chunk)
                if size > SETTINGS.max_upload_bytes:
                    raise HTTPException(413, "Video exceeds MAX_UPLOAD_MB")
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if size == 0:
            raise HTTPException(422, "The video is empty")
        filename = Path((video.filename or "video").replace("\\", "/")).name[:255]
        queued = jobs.add(upload_id, path, snapshot(SETTINGS), filename)
        return public_job(existing_job(upload_id))
    finally:
        video.file.close()
        if path is not None and not queued:
            path.unlink(missing_ok=True)


@app.get("/jobs/{upload_id}", response_model=JobStatus)
def status(upload_id: str):
    return public_job(existing_job(upload_id))


@app.get("/jobs/{upload_id}/report")
def report(upload_id: str):
    if existing_job(upload_id)["status"] not in ("completed", "needs_review"):
        raise HTTPException(409, "The full quality check has not completed; published chunk reports remain available")
    return available_file(SETTINGS.data_dir / "outputs" / upload_id / "report.json",
                          "application/json", f"{upload_id}-report.json")


@app.get("/jobs/{upload_id}/video")
def download(upload_id: str):
    if existing_job(upload_id)["status"] != "completed":
        raise HTTPException(409, "Download blocked: every chunk must pass the quality check")
    return available_file(SETTINGS.data_dir / "outputs" / upload_id / "anonymized.mp4",
                          "video/mp4", f"{upload_id}-anonymized.mp4")


@app.get("/jobs/{upload_id}/preview")
def preview(upload_id: str, request: Request):
    require_local(request)
    if not public_job(existing_job(upload_id))["full_preview_ready"]:
        raise HTTPException(409, "Full preview is available after anonymization; select a published chunk first")
    return available_file(SETTINGS.data_dir / "outputs" / upload_id / "anonymized.mp4", "video/mp4")


@app.get("/jobs/{upload_id}/chunks/{chunk_index}/preview")
def preview_chunk(upload_id: str, chunk_index: int, request: Request):
    require_local(request)
    existing_chunk(upload_id, chunk_index)
    return available_file(chunk_video(SETTINGS.data_dir / "outputs" / upload_id, chunk_index), "video/mp4")


@app.get("/jobs/{upload_id}/chunks/{chunk_index}/report")
def chunk_report(upload_id: str, chunk_index: int):
    chunk = existing_chunk(upload_id, chunk_index)
    if chunk["status"] not in ("passed", "flagged"):
        raise HTTPException(409, "This chunk's quality check has not completed")
    return available_file(chunk_video(SETTINGS.data_dir / "outputs" / upload_id, chunk_index).with_suffix(".json"),
                          "application/json", f"{upload_id}-chunk-{chunk_index}-report.json")


@app.get("/queues")
def queues():
    return jobs.queues()


@app.get("/health")
def health():
    workers = {role: worker_status(SETTINGS.data_dir / "workers", role) for role in ("redact", "check")}
    healthy = all(worker["state"] != "stopped" for worker in workers.values())
    return JSONResponse({"status": "ok" if healthy else "degraded", "workers": workers,
                         "devices": {"redact": SETTINGS.detector_device, "check": SETTINGS.checker_device}},
                        status_code=200 if healthy else 503)
