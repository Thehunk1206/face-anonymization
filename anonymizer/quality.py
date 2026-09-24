"""YOLOv12m-face audit of encoded video, also usable as a standalone command."""

import argparse
import json
import time
from itertools import batched
from pathlib import Path
from threading import Event

import torch

from .config import SETTINGS
from .confirmation import TemporalConfirmation, confirmation_counts
from .detection import YOLOFace
from .models import CHECKER_MODEL, MODELS
from .media import chunk_video, frame_image, open_video, publish_file, timed_frames, validate_coverage


def check_video(source: Path, report_path: Path, checker: YOLOFace, stop: Event, progress,
                expected: dict | None = None, context=None, chunk_index=0) -> dict:
    started = time.monotonic()
    count = flagged_frames = faces = 0
    first_time = last_time = None
    last_duration = 0.0
    confirmation = TemporalConfirmation(checker.settings.checker_window_frames,
                                        checker.settings.checker_confirmation_hits, context)
    start_frame = expected["start_frame"] if expected else 0
    if confirmation.pending and confirmation.pending[-1]["global_frame"] != start_frame - 1:
        raise ValueError("Checker context does not precede this chunk")
    totals = confirmation_counts([])
    boundary_updates = []
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Stream findings to disk: hours of flagged footage should not fill RAM.
    with open_video(source) as reader, report_path.open("w") as report:
        report.write('{"findings":[')

        def write_finding(row):
            nonlocal flagged_frames, faces
            if row is None or not row["faces"]:
                return
            if row["chunk_index"] != chunk_index:
                boundary_updates.append(row)
                return
            finding = {key: row[key] for key in ("frame", "timestamp_seconds", "faces")}
            if flagged_frames:
                report.write(",")
            json.dump(finding, report)
            flagged_frames += 1
            faces += len(finding["faces"])
            for key, value in confirmation_counts([finding]).items():
                totals[key] += value

        stream = reader.streams.video[0]
        for batch in batched(timed_frames(reader, stream), checker.settings.checking_batch_size):
            if stop.is_set():
                raise InterruptedError("Application is shutting down")
            images = []
            for frame, duration in batch:
                image = frame_image(frame)
                timestamp = float(frame.pts * frame.time_base)
                if last_time is not None and timestamp <= last_time:
                    raise ValueError("Quality check found invalid frame timestamps")
                if first_time is None:
                    first_time = timestamp
                if expected and (image.shape[1], image.shape[0]) != (expected["width"], expected["height"]):
                    raise ValueError("Encoded output has unexpected dimensions")
                images.append(image)
                last_time = timestamp
            predictions = checker.detect_batch(images)
            for (frame, duration), boxes, image in zip(batch, predictions, images, strict=True):
                if stop.is_set():
                    raise InterruptedError("Application is shutting down")
                timestamp = float(frame.pts * frame.time_base)
                row = {
                    "frame": count, "global_frame": start_frame + count, "chunk_index": chunk_index,
                    "timestamp_seconds": round(timestamp - first_time, 6),
                    "faces": [{"box": [round(float(value), 2) for value in box[:4]],
                               "confidence": round(float(box[4]), 4)} for box in boxes],
                }
                write_finding(confirmation.add(row, image))
                count += 1
                last_duration = float(duration)
                if count == 1 or count % 10 == 0:
                    progress(count, expected["frames"] if expected else (stream.frames or None))
        if count == 0:
            raise ValueError("Quality check could not decode any frames")
        validate_coverage(stream, count, first_time, last_time + last_duration, last_duration)
        if expected:
            if count != expected["frames"]:
                raise ValueError(f"Incomplete output: expected {expected['frames']} frames, checked {count}")
            if abs(last_time - first_time + last_duration - expected["duration_seconds"]) > max(0.15, last_duration * 2):
                raise ValueError("Encoded output duration differs from the source")
        tail = confirmation.context()
        for row in confirmation.finish():
            write_finding(row)
        summary = {
            "verdict": "needs_review" if flagged_frames else "no_residual_faces_detected",
            "scan_complete": True, "checked_frames": count, "flagged_frames": flagged_frames,
            "suspected_faces": faces, "checking_seconds": round(time.monotonic() - started, 3),
            "checker": "YOLOv12m-face / Ultralytics 8.3.161",
            "checker_mode": "single_full_frame",
            "checker_device": checker.device,
            "checker_model_sha256": MODELS[CHECKER_MODEL][1],
            "checker_size": checker.settings.checker_size,
            "confidence_threshold": checker.settings.checker_threshold,
            "checking_batch_size": checker.settings.checking_batch_size,
            **totals,
            "temporal_confirmation": {"window_frames": confirmation.window, "required_hits": confirmation.hits,
                                      "brief_findings_block_download": True},
            # A bounded tail makes chunk boundaries and checker restarts equivalent
            # to a continuous scan. The next chunk can upgrade earlier findings.
            "temporal_context": tail, "boundary_updates": boundary_updates,
            "processing": expected,
            "limitations": "Both stages use YOLO-face and may share missed detections. No detections is not proof of anonymity. Other identifying content may remain. Pipeline outputs have no audio.",
        }
        report.write("]")
        report.write("," + json.dumps(summary)[1:])
    progress(count, count)
    return summary


def update_previous_reports(directory: Path, updates: list) -> None:
    """Atomically backfill confirmation when support arrives in a later chunk."""
    by_chunk = {}
    for row in updates:
        by_chunk.setdefault(row["chunk_index"], {})[row["frame"]] = row["faces"]
    for index, frames in by_chunk.items():
        path = chunk_video(directory, index).with_suffix(".json")
        report = json.loads(path.read_text())
        for finding in report["findings"]:
            if finding["frame"] in frames:
                finding["faces"] = frames[finding["frame"]]
        for row in report.get("temporal_context", {}).get("tail", []):
            if row["chunk_index"] == index and row["frame"] in frames:
                row["faces"] = frames[row["frame"]]
        report.update(confirmation_counts(report["findings"]))
        temporary = path.with_suffix(".confirmation.partial.json")
        try:
            temporary.write_text(json.dumps(report))
            publish_file(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def combine_reports(directory: Path, job: dict, stop: Event) -> dict:
    """Stream chunk findings into one report using original-video offsets."""
    stats = json.loads(job["redaction_stats"])
    checked = flagged = faces = 0
    seconds = 0.0
    totals = confirmation_counts([])
    temporary = directory / "report.partial.json"
    try:
        with temporary.open("w") as target:
            target.write('{"findings":[')
            first = True
            for chunk in job["chunks"]:
                if stop.is_set():
                    raise InterruptedError("Worker is shutting down")
                report = json.loads(chunk_video(directory, chunk["chunk_index"]).with_suffix(".json").read_text())
                if not report["scan_complete"] or report["checked_frames"] != chunk["stats"]["frames"]:
                    raise ValueError("A chunk quality check is incomplete")
                checked += report["checked_frames"]
                flagged += report["flagged_frames"]
                faces += report["suspected_faces"]
                seconds += report["checking_seconds"]
                for key, value in confirmation_counts(report["findings"]).items():
                    totals[key] += value
                for finding in report["findings"]:
                    finding.update(chunk_index=chunk["chunk_index"], chunk_frame=finding["frame"],
                                   chunk_timestamp_seconds=finding["timestamp_seconds"])
                    finding["frame"] += chunk["stats"]["start_frame"]
                    finding["timestamp_seconds"] = round(finding["timestamp_seconds"] + chunk["stats"]["start_seconds"], 6)
                    if not first:
                        target.write(",")
                    json.dump(finding, target)
                    first = False
            if checked != stats["frames"]:
                raise ValueError("Combined check does not cover every input frame")
            summary = {key: value for key, value in report.items()
                       if key not in ("findings", "processing", "temporal_context", "boundary_updates")}
            summary.update(verdict="needs_review" if flagged else "no_residual_faces_detected",
                           checked_frames=checked, flagged_frames=flagged, suspected_faces=faces,
                           checking_seconds=round(seconds, 3), processing=stats, chunk_count=len(job["chunks"]), **totals)
            target.write("]," + json.dumps(summary)[1:])
        publish_file(temporary, directory / "report.json")
        return summary
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--report", type=Path, default=Path("quality-report.json"))
    args = parser.parse_args()
    temporary = args.report.with_name(args.report.name + ".partial")
    if args.video.resolve() in (args.report.resolve(), temporary.resolve()):
        parser.error("The report or its temporary file must not overwrite the input video")
    torch.set_num_threads(2)
    try:
        summary = check_video(args.video, temporary, YOLOFace(SETTINGS, checking=True), Event(), lambda *_: None)
        publish_file(temporary, args.report)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("temporal_context", "boundary_updates")}, indent=2))
    raise SystemExit(2 if summary["flagged_frames"] else 0)


if __name__ == "__main__":
    main()
