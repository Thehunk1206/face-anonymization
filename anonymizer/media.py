"""Video timing, orientation, silent encoding, and lossless chunk assembly."""

import os
from fractions import Fraction
from pathlib import Path
from threading import Event

import av
import numpy as np


def open_video(path: Path):
    container = av.open(str(path), options={"err_detect": "explode"})
    if len(container.streams.video) != 1:
        container.close()
        raise ValueError("Provide a video with exactly one video stream")
    return container


def frame_image(frame: av.VideoFrame) -> np.ndarray:
    if frame.is_corrupt:
        raise ValueError("A corrupt frame was encountered")
    if frame.pts is None:
        raise ValueError("Video frame has no timestamp")
    image = frame.to_ndarray(format="bgr24")
    angle = frame.rotation
    if angle % 90:
        raise ValueError("Only 90-degree display rotations are supported")
    return np.ascontiguousarray(np.rot90(image, int(angle / 90) % 4))


def timed_frames(reader, stream):
    """One-frame lookahead gives true VFR durations; decoded duration can be nominal."""
    previous = None
    first_time = None
    packet_end = None
    for packet in reader.demux(stream):
        if packet.pts is not None and packet.duration:
            end = (packet.pts + packet.duration) * packet.time_base
            packet_end = max(packet_end, end) if packet_end is not None else end
        for frame in packet.decode():
            if frame.pts is None:
                raise ValueError("Video frame has no timestamp")
            timestamp = frame.pts * frame.time_base
            if first_time is None:
                first_time = timestamp
            if previous is not None:
                duration = timestamp - previous.pts * previous.time_base
                if duration <= 0:
                    raise ValueError("Video timestamps are not strictly increasing")
                yield previous, duration
            previous = frame
    if previous is not None:
        last_time = previous.pts * previous.time_base
        ends = [packet_end] if packet_end is not None else []
        if stream.duration:
            start = stream.start_time * stream.time_base if stream.start_time is not None else first_time
            ends.append(start + stream.duration * stream.time_base)
        ends = [end for end in ends if end > last_time]
        if ends:
            # A container may trim the last packet, or overstate video duration
            # using the longer audio track. Honor the earlier valid boundary.
            duration = min(ends) - last_time
        else:
            duration = previous.duration * previous.time_base or 1 / (stream.average_rate or stream.guessed_rate)
        if duration <= 0:
            raise ValueError("Video has an invalid final frame duration")
        yield previous, duration


def encode_frame(writer, encoder, frame, durations: dict) -> None:
    # H.264 may buffer several frames. Preserve their durations until packets emerge.
    if frame is not None:
        durations[frame.pts * frame.time_base] = frame.duration * frame.time_base
    for packet in encoder.encode(frame):
        duration = durations.pop(packet.pts * packet.time_base)
        packet.duration = round(duration / packet.time_base)
        writer.mux(packet)


def validate_coverage(stream, count: int, first_time: float, end_time: float, frame_duration: float) -> None:
    if count == 0:
        raise ValueError("Video contains no decodable frames")
    if stream.frames and count != stream.frames:
        raise ValueError(f"Incomplete decoding: expected {stream.frames} frames, decoded {count}")
    if stream.duration:
        declared = float(stream.duration * stream.time_base)
        if end_time - first_time < declared - max(0.15, 2 * frame_duration):
            raise ValueError("Decoded video ends before its declared duration")


def create_encoder(writer, stream, frame, image, rate):
    """Keep source geometry and timestamp precision in the H.264 output."""
    encoder = writer.add_stream("libx264", rate=rate)
    encoder.width, encoder.height = image.shape[1], image.shape[0]
    # yuv444p preserves odd dimensions; ordinary videos use yuv420p.
    encoder.pix_fmt = "yuv420p" if encoder.width % 2 == encoder.height % 2 == 0 else "yuv444p"
    encoder.time_base = frame.time_base
    encoder.codec_context.time_base = frame.time_base
    aspect = stream.sample_aspect_ratio or Fraction(1, 1)
    encoder.codec_context.sample_aspect_ratio = 1 / aspect if frame.rotation % 180 else aspect
    encoder.options = {"crf": "18", "preset": "fast", "bf": "0"}
    return encoder


def publish_file(temporary: Path, destination: Path) -> None:
    """Publish only closed, complete files on the same filesystem."""
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def chunk_video(directory: Path, index: int) -> Path:
    return directory / "chunks" / f"{index:06d}.mp4"


def verify_encoded(path: Path, expected: dict, stop: Event) -> None:
    """H.264 without B-frames has one ordered video packet per encoded frame."""
    with open_video(path) as reader:
        stream = reader.streams.video[0]
        if reader.streams.audio:
            raise ValueError("Generated video unexpectedly contains audio")
        if (stream.width, stream.height) != (expected["width"], expected["height"]):
            raise ValueError("Encoded dimensions differ from the source")
        count, previous, end = 0, None, None
        for packet in reader.demux(stream):
            if stop.is_set():
                raise InterruptedError("Worker is shutting down")
            if not packet.size:
                continue
            if packet.pts is None or packet.duration <= 0:
                raise ValueError("Encoded frame has missing timing")
            timestamp = packet.pts * packet.time_base
            if (previous is None and timestamp != 0) or (previous is not None and timestamp <= previous):
                raise ValueError("Encoded timestamps are not ordered from zero")
            previous = timestamp
            end = timestamp + packet.duration * packet.time_base
            count += 1
        if count != expected["frames"] or end is None:
            raise ValueError(f"Incomplete encoding: expected {expected['frames']} frames, got {count}")
        if abs(float(end) - expected["duration_seconds"]) > max(0.001, float(stream.time_base) * 2):
            raise ValueError("Encoded duration differs from the source")


def assemble_chunks(directory: Path, chunks: list[dict], expected: dict, stop: Event) -> None:
    """Remux already encoded frames at their original offsets; never add audio."""
    temporary = directory / "anonymized.partial.mp4"
    try:
        with av.open(str(temporary), "w", options={"movflags": "+faststart"}) as writer:
            output = None
            for chunk in chunks:
                stats = chunk["stats"]
                offset = stats["start_pts"] * Fraction(stats["time_base"])
                with open_video(chunk_video(directory, chunk["chunk_index"])) as reader:
                    stream = reader.streams.video[0]
                    if output is None:
                        output = writer.add_stream_from_template(stream)
                    for packet in reader.demux(stream):
                        if stop.is_set():
                            raise InterruptedError("Worker is shutting down")
                        if packet.dts is None:
                            continue
                        shift = offset / packet.time_base
                        if shift.denominator != 1:
                            raise ValueError("Chunk time base cannot represent its original offset")
                        packet.pts += int(shift)
                        packet.dts += int(shift)
                        packet.stream = output
                        writer.mux(packet)
        verify_encoded(temporary, expected, stop)
        publish_file(temporary, directory / "anonymized.mp4")
    finally:
        temporary.unlink(missing_ok=True)
