"""Detect in batches, track continuously, and publish silent chunks as they finish."""

import time
from fractions import Fraction
from itertools import batched
from pathlib import Path
from threading import Event

import av

from .config import Settings
from .database import PIPELINE_VERSION
from .detection import mask_faces
from .media import (chunk_video, create_encoder, encode_frame, frame_image, open_video,
                    publish_file, timed_frames, validate_coverage, verify_encoded)
from .models import DETECTOR_MODEL, MODELS
from .tracking import FaceTracker


def anonymize(source: Path, directory: Path, detector, settings: Settings, stop: Event,
              progress, published: list[dict], on_chunk) -> dict:
    started = time.monotonic()
    (directory / "chunks").mkdir(parents=True, exist_ok=True, mode=0o700)
    for temporary in (directory / "chunks").glob("*.partial.mp4"):
        temporary.unlink()
    for index, chunk in enumerate(published):
        if chunk["chunk_index"] != index or not chunk_video(directory, index).is_file():
            raise ValueError("A previously published chunk is missing; upload again with a new ID")
    tracker = FaceTracker(settings)
    count = detections = index = chunk_frames = 0
    writer = encoder = None
    first_time = previous_time = chunk_start = None
    dimensions = time_base = None
    chunk_first_pts = first_pts = chunk_frame_start = 0
    end_time = Fraction(0)
    durations = {}
    chunk_started = started
    metadata = {
        "pipeline_version": PIPELINE_VERSION, "detector": "YOLOv12s-face",
        "detector_model_sha256": MODELS[DETECTOR_MODEL][1], "detector_device": detector.device,
        "detector_threshold": settings.detector_threshold, "detector_size": settings.detector_size,
        "detection_batch_size": settings.detection_batch_size,
        "padding_each_side": settings.padding, "redaction_method": "solid_black", "audio_removed": True,
        "tracker": "BoT-SORT / ultralytics 8.3.161" if settings.tracker_max_gap_seconds else "disabled",
        "tracker_gmc_method": settings.tracker_gmc_method, "tracker_with_reid": False,
        "tracker_low_threshold": settings.tracker_low_threshold,
        "tracker_max_gap_seconds": settings.tracker_max_gap_seconds,
    }

    def finish_chunk():
        nonlocal writer, encoder, index, chunk_frames
        stats = {**metadata, "frames": chunk_frames, "width": dimensions[1], "height": dimensions[0],
                 "duration_seconds": float(end_time - chunk_start),
                 "start_seconds": float(chunk_start - first_time), "start_frame": chunk_frame_start,
                 "start_pts": chunk_first_pts - first_pts, "time_base": str(time_base),
                 "processing_seconds": round(time.monotonic() - chunk_started, 3)}
        if index < len(published):
            # Replay restores BoT-SORT; immutable published files/reports stay in use.
            old = published[index]["stats"]
            if any(stats[key] != old[key] for key in ("frames", "start_pts", "time_base", "start_frame")):
                raise ValueError("Source timing changed while recovering the job")
        else:
            encode_frame(writer, encoder, None, durations)
            if durations:
                raise ValueError("Encoder did not emit every frame")
            writer.close()
            writer = encoder = None
            path = chunk_video(directory, index)
            temporary = path.with_suffix(".partial.mp4")
            verify_encoded(temporary, stats, stop)
            publish_file(temporary, path)
            on_chunk(index, stats)
        index += 1
        chunk_frames = 0

    try:
        with open_video(source) as reader:
            stream = reader.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            if rate is None or rate <= 0:
                raise ValueError("Video has no usable frame rate")
            for batch in batched(timed_frames(reader, stream), settings.detection_batch_size):
                if stop.is_set():
                    raise InterruptedError("Worker is shutting down")
                images = [frame_image(frame) for frame, _ in batch]
                predictions = detector.detect_batch(images)
                for (frame, duration), image, boxes in zip(batch, images, predictions, strict=True):
                    if stop.is_set():
                        raise InterruptedError("Worker is shutting down")
                    timestamp = frame.pts * frame.time_base
                    if first_time is None:
                        first_time, first_pts = timestamp, frame.pts
                        dimensions, time_base = image.shape[:2], frame.time_base
                    if image.shape[:2] != dimensions or frame.time_base != time_base:
                        raise ValueError("Video changes dimensions or time base mid-stream")
                    if previous_time is not None and timestamp <= previous_time:
                        raise ValueError("Video timestamps are not strictly increasing")
                    if chunk_frames and timestamp - chunk_start >= Fraction(str(settings.chunk_seconds)):
                        finish_chunk()
                    if chunk_frames == 0:
                        chunk_start, chunk_first_pts, chunk_frame_start = timestamp, frame.pts, count
                        chunk_started = time.monotonic()
                        if index >= len(published):
                            temporary = chunk_video(directory, index).with_suffix(".partial.mp4")
                            writer = av.open(str(temporary), "w", options={"movflags": "+faststart"})
                            encoder = create_encoder(writer, stream, frame, image, rate)
                    masks = tracker.update(image, boxes, float(timestamp))
                    detections += int((boxes[:, 4] >= settings.detector_threshold).sum())
                    if writer is not None:
                        mask_faces(image, masks, settings.padding)
                        encoded = av.VideoFrame.from_ndarray(image, format="bgr24")
                        encoded.pts = frame.pts - chunk_first_pts
                        encoded.time_base = time_base
                        encoded.duration = round(duration / time_base)
                        encode_frame(writer, encoder, encoded, durations)
                    count += 1
                    chunk_frames += 1
                    previous_time, end_time = timestamp, timestamp + duration
                    if count == 1 or count % 10 == 0:
                        progress(count, stream.frames or None)
            if count == 0:
                raise ValueError("Video contains no decodable frames")
            validate_coverage(stream, count, float(first_time), float(end_time), float(duration))
            finish_chunk()
            if index < len(published):
                raise ValueError("Source ends before its published chunks")
    finally:
        if writer is not None:
            writer.close()
        for temporary in (directory / "chunks").glob("*.partial.mp4"):
            temporary.unlink(missing_ok=True)
    progress(count, count)
    return {**metadata, "frames": count, "width": dimensions[1], "height": dimensions[0],
            "duration_seconds": float(end_time - first_time), "source_first_timestamp": float(first_time),
            "chunk_count": index, "face_detections": detections,
            "tracker_gmc_errors": tracker.tracker.gmc.errors,
            "tracker_recovered_detections": tracker.recovered_detections,
            "tracker_predicted_boxes": tracker.predicted_boxes, "tracker_scene_resets": tracker.scene_resets,
            "processing_seconds": round(time.monotonic() - started, 3)}
