#!/usr/bin/env python3
"""
Synchronized 4-Camera Session Recorder (Full FFV1 Lossless MKV Edition)

Records color video + depth frames from 4 RealSense cameras:
  - Cameras 1, 2 (tripod): Capture body pose
  - Cameras 3, 4 (table): Capture gaze

Color is stored during recording as lossless FFV1 in color.mkv.
Depth is stored as lossless FFV1 video in MKV container (16-bit grayscale).

Output structure:
    recordings/session_YYYY-MM-DD_HH:MM/
        cam1/color.mkv  cam1/depth.mkv
        cam2/color.mkv  cam2/depth.mkv
        cam3/color.mkv  cam3/depth.mkv
        cam4/color.mkv  cam4/depth.mkv
        metadata.json

Controls:
    C: Validate live multi-camera depth/PCL alignment
    E: Executive bypass of the pre-check and unlock recording
    R: Toggle recording on/off
    Q: Quit
    Ctrl+C: Stop and quit (headless/no-gui mode)

Usage:
    python felfelfeci3.py --output-dir ./recordings

    Serial numbers are read from camera_config.json (see --cam-config).

Requirements:
    ffmpeg CLI available on PATH, or imageio-ffmpeg installed
"""

import argparse
import cv2
import csv
import numpy as np
import pyrealsense2 as rs
import os
import sys
import json
import time
import threading
import queue
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent directory (devel/) to sys.path so 'import utils' works from anywhere
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def get_ffmpeg_executable():
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg as iio_ff
        return iio_ff.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg was not found on PATH and imageio_ffmpeg is not installed"
        ) from exc


class FFmpegStdinWriter:
    """Small rawvideo-to-ffmpeg stdin writer used by the realtime recorder."""

    def __init__(
        self,
        output_path,
        size,
        fps,
        pix_fmt_in,
        pix_fmt_out,
        codec,
        output_params=None,
        log_path=None,
    ):
        width, height = size
        self.output_path = output_path
        self._owns_log_file = log_path is not None
        self.log_file = open(log_path, 'w') if log_path else subprocess.DEVNULL
        cmd = [
            get_ffmpeg_executable(),
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            '-f', 'rawvideo',
            '-pix_fmt', pix_fmt_in,
            '-s:v', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',
            '-an',
            '-c:v', codec,
            '-pix_fmt', pix_fmt_out,
        ]
        if output_params:
            cmd.extend(output_params)
        cmd.append(output_path)

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.log_file,
        )

    def send(self, data):
        if data is None:
            return
        returncode = self.process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"FFmpeg exited early for {self.output_path} "
                f"(return code {returncode})"
            )
        if self.process.stdin is not None:
            try:
                self.process.stdin.write(data)
            except BrokenPipeError as exc:
                raise RuntimeError(
                    f"FFmpeg pipe closed for {self.output_path}"
                ) from exc

    def close(self):
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        returncode = self.process.wait()
        if self._owns_log_file:
            self.log_file.close()
        if returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed for {self.output_path} "
                f"(return code {returncode})"
            )


class CameraThread:
    """Manages a single RealSense camera in its own capture thread."""

    def __init__(
        self,
        cam_idx,
        serial,
        width,
        height,
        fps,
        enable_depth=True,
        align_depth_live=False,
    ):
        self.cam_idx = cam_idx
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.align_depth_live = bool(align_depth_live and enable_depth)

        self.pipeline = None
        self.align = None
        self.intrinsics_data = None
        self.calibration_data = None
        self.depth_scale = None  # meters per depth unit (z16)
        self.usb_type = None
        self.physical_port = None

        # Thread communication
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None

        # Recording state (controlled by main thread)
        self.record_event = None  # shared threading.Event for synchronized start
        self.record_barrier = None  # shared threading.Barrier for synchronized first frame
        self.recording = False
        self._barrier_passed = False  # True once this thread has passed the barrier
        self._color_writer = None
        self.cam_dir = None       # per-camera subdirectory inside session_dir
        self.session_dir = None
        self.frame_count = 0
        self.capture_count = 0
        self.writer_frame_count = 0
        self.queue_drop_count = 0
        self.duplicate_frame_count = 0
        self.max_writer_queue_depth = 0
        self.writer_queue_capacity = 90
        self.capture_timestamps = []
        self._last_recorded_color_frame_number = None
        self._writer_error = None

        # Timestamp storage: separate lists for color and depth
        # Each entry: (frame_idx, hw_timestamp_ms, host_timestamp_ms, timestamp_domain)
        self.color_timestamps = []
        self.depth_timestamps = []

        # FFV1 depth video writer
        self._depth_writer = None

        # Writer thread: decouples disk I/O from the capture loop so that
        # FFmpeg color/depth writes never block frame capture.
        self._write_queue = None
        self._writer_thread = None

    @staticmethod
    def _intrinsics_to_dict(intr):
        return {
            'fx': intr.fx, 'fy': intr.fy,
            'ppx': intr.ppx, 'ppy': intr.ppy,
            'width': intr.width, 'height': intr.height,
            'model': str(intr.model),
            'coeffs': list(intr.coeffs),
        }

    @staticmethod
    def _extrinsics_to_dict(extr):
        return {
            'rotation': list(extr.rotation),
            'translation': list(extr.translation),
        }

    def start(self):
        """Initialize RealSense pipeline and start capture thread."""
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_depth:
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = self.pipeline.start(cfg)

        # Configure sensors.
        # NOTE: We intentionally do NOT enable global_time_enabled — it causes
        # segfaults in many librealsense versions. Instead we use
        # backend_timestamp metadata for cross-camera comparable host-clock
        # timestamps (see _capture_loop).
        device = profile.get_device()
        try:
            usb_descriptor = getattr(
                rs.camera_info, "usb_type_descriptor", None
            )
            if usb_descriptor is not None and device.supports(usb_descriptor):
                self.usb_type = device.get_info(
                    usb_descriptor
                )
        except (AttributeError, RuntimeError):
            self.usb_type = None
        try:
            physical_port = getattr(rs.camera_info, "physical_port", None)
            if physical_port is not None and device.supports(physical_port):
                self.physical_port = device.get_info(
                    physical_port
                )
        except (AttributeError, RuntimeError):
            self.physical_port = None

        for sensor in device.sensors:
            # Force frame-rate priority: prevent FPS drops in low light
            if sensor.supports(rs.option.auto_exposure_priority):
                sensor.set_option(rs.option.auto_exposure_priority, 0)

        # Optional live depth->color alignment (CPU-heavy for multi-cam setups)
        if self.enable_depth:
            if self.align_depth_live:
                self.align = rs.align(rs.stream.color)
            try:
                depth_sensor = device.first_depth_sensor()
                self.depth_scale = float(depth_sensor.get_depth_scale())
            except RuntimeError:
                self.depth_scale = None

        # Extract camera calibration params
        color_stream_profile = profile.get_stream(rs.stream.color)
        color_video_profile = color_stream_profile.as_video_stream_profile()
        color_intr = color_video_profile.get_intrinsics()
        color_intrinsics = self._intrinsics_to_dict(color_intr)

        depth_intrinsics = None
        depth_to_color_extrinsics = None
        color_to_depth_extrinsics = None

        if self.enable_depth:
            depth_stream_profile = profile.get_stream(rs.stream.depth)
            depth_video_profile = depth_stream_profile.as_video_stream_profile()
            depth_intr = depth_video_profile.get_intrinsics()
            depth_intrinsics = self._intrinsics_to_dict(depth_intr)

            depth_to_color_extrinsics = self._extrinsics_to_dict(
                depth_video_profile.get_extrinsics_to(color_video_profile)
            )
            color_to_depth_extrinsics = self._extrinsics_to_dict(
                color_video_profile.get_extrinsics_to(depth_video_profile)
            )

        # Keep legacy field for backward compatibility
        self.intrinsics_data = color_intrinsics
        self.calibration_data = {
            'color_intrinsics': color_intrinsics,
            'depth_intrinsics': depth_intrinsics,
            'depth_to_color_extrinsics': depth_to_color_extrinsics,
            'color_to_depth_extrinsics': color_to_depth_extrinsics,
            'depth_scale_meters_per_unit': self.depth_scale,
        }

        # Warm up
        for _ in range(15):
            self.pipeline.wait_for_frames()

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    @staticmethod
    def _get_hw_timestamp(frame):
        """Extract hardware timestamp and domain from a RealSense frame.

        Returns:
            (hw_ts_ms, domain_str): hardware timestamp in ms and the clock domain name.
        """
        hw_ts = frame.get_timestamp()  # ms, from the frame's timestamp domain
        domain = frame.frame_timestamp_domain
        domain_map = {
            rs.timestamp_domain.hardware_clock: 'hardware_clock',
            rs.timestamp_domain.system_time: 'system_time',
            rs.timestamp_domain.global_time: 'global_time',
        }
        domain_str = domain_map.get(domain, str(domain))
        return hw_ts, domain_str

    def _capture_loop(self):
        """Continuous frame capture in background thread.

        IMPORTANT: Frame data is copied with np.array() (not np.asanyarray)
        because the zero-copy view into librealsense's internal buffer can be
        recycled before the writer thread finishes reading it, causing a
        segfault.  All slow disk I/O is offloaded to _writer_loop via
        _write_queue so that this loop keeps up with wait_for_frames().

        Timestamps are captured from the RealSense hardware clock immediately
        after wait_for_frames() returns — before any processing — for maximum
        accuracy.  A host-side perf_counter reference is also recorded.
        """
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue

            # ── Capture host timestamp immediately after frame arrival ──
            host_ts = time.perf_counter() * 1000.0  # ms with µs precision

            if self.enable_depth:
                if self.align is not None:
                    aligned = self.align.process(frames)
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                else:
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
            else:
                color_frame = frames.get_color_frame()
                depth_frame = None

            if not color_frame:
                continue

            # ── Extract hardware timestamps from RealSense frames ──
            color_hw_ts, color_ts_domain = self._get_hw_timestamp(color_frame)
            color_frame_number = int(color_frame.get_frame_number())

            depth_hw_ts = None
            depth_ts_domain = None
            if depth_frame:
                depth_hw_ts, depth_ts_domain = self._get_hw_timestamp(depth_frame)

            # COPY frame data — np.asanyarray returns a view into RS's buffer
            # which can be freed before write() finishes, causing a segfault.
            color_image = np.array(color_frame.get_data())

            depth_image = None
            if depth_frame:
                depth_image = np.array(depth_frame.get_data())

            # Check for synchronized recording start (barrier-gated)
            if not self.recording and self._color_writer is not None:
                if self.record_event is not None and self.record_event.is_set():
                    # Wait at barrier so all cameras start on the same frame
                    if not self._barrier_passed and self.record_barrier is not None:
                        self.record_barrier.wait()
                        self._barrier_passed = True
                    self.recording = True

            # Record if active — hand off to writer thread (no I/O here)
            if self.recording:
                if (
                    self._last_recorded_color_frame_number == color_frame_number
                ):
                    self.duplicate_frame_count += 1
                self._last_recorded_color_frame_number = color_frame_number
                capture_idx = self.capture_count
                self.capture_count += 1
                self.capture_timestamps.append(
                    (
                        capture_idx,
                        color_frame_number,
                        color_hw_ts,
                        host_ts,
                        color_ts_domain,
                    )
                )
                if self._write_queue is not None:
                    try:
                        self._write_queue.put_nowait((
                            color_image, depth_image, self.frame_count,
                            host_ts,
                            color_hw_ts, color_ts_domain,
                            depth_hw_ts, depth_ts_domain,
                        ))
                        self.frame_count += 1
                        self.max_writer_queue_depth = max(
                            self.max_writer_queue_depth,
                            self._write_queue.qsize(),
                        )
                    except queue.Full:
                        self.queue_drop_count += 1

            # Always update display queue (use put-replace so it's never stale)
            try:
                self.frame_queue.put_nowait((color_image, depth_image))
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait((color_image, depth_image))

    def _writer_loop(self):
        """Disk I/O thread — writes color and depth as FFV1 MKV.

        Runs in a separate thread so that FFmpeg writes
        never block the capture loop.  Depth frames are written one at a time
        via FFmpeg stdin; RAM usage stays constant regardless of session
        length.

        Timestamps from the RealSense hardware clock are accumulated here and
        written to CSV files when recording stops.
        """
        while True:
            item = self._write_queue.get()
            try:
                if item is None:  # sentinel from stop_recording
                    break

                if self._writer_error is not None:
                    continue

                (color_image, depth_image, frame_idx,
                 host_ts,
                 color_hw_ts, color_ts_domain,
                 depth_hw_ts, depth_ts_domain) = item

                if self._color_writer is not None:
                    color_bgr = color_image.astype(np.uint8, copy=False)
                    if not color_bgr.flags.c_contiguous:
                        color_bgr = np.ascontiguousarray(color_bgr)
                    self._color_writer.send(color_bgr.tobytes())

                if depth_image is not None and self._depth_writer is not None:
                    depth_z16 = depth_image.astype(np.uint16, copy=False)
                    if not depth_z16.flags.c_contiguous:
                        depth_z16 = np.ascontiguousarray(depth_z16)
                    self._depth_writer.send(depth_z16.tobytes())

                self.color_timestamps.append(
                    (frame_idx, color_hw_ts, host_ts, color_ts_domain)
                )
                if depth_hw_ts is not None:
                    self.depth_timestamps.append(
                        (frame_idx, depth_hw_ts, host_ts, depth_ts_domain)
                    )
                self.writer_frame_count += 1
            except Exception as exc:
                self._writer_error = repr(exc)
                print(
                    f"\n[cam{self.cam_idx + 1}] Recording writer failed: {exc}"
                )
            finally:
                self._write_queue.task_done()

    def prepare_recording(self, session_dir):
        """Set up writers and directories, but don't start recording yet.

        Actual recording begins when self.record_event is set (event-gated
        synchronization across cameras).
        """
        self.session_dir = session_dir
        cam_name = f"cam{self.cam_idx + 1}"

        # Per-camera subdirectory
        self.cam_dir = os.path.join(session_dir, cam_name)
        os.makedirs(self.cam_dir, exist_ok=True)

        # Color writer: FFV1 lossless MKV. The RealSense color stream is
        # 8-bit BGR; bgr0 keeps those channels lossless and adds one constant
        # unused byte because this FFmpeg FFV1 encoder does not support bgr24.
        video_path = os.path.join(self.cam_dir, "color.mkv")
        self._color_writer = FFmpegStdinWriter(
            video_path,
            (self.width, self.height),
            pix_fmt_in='bgr24',
            pix_fmt_out='bgr0',
            codec='ffv1',
            fps=self.fps,
            log_path=os.path.join(self.cam_dir, "color_writer.log"),
        )

        # Depth video writer — FFV1 lossless in MKV container (true 16-bit grayscale)
        if self.enable_depth:
            depth_mkv_path = os.path.join(self.cam_dir, "depth.mkv")
            self._depth_writer = FFmpegStdinWriter(
                depth_mkv_path,
                (self.width, self.height),
                pix_fmt_in='gray16le',
                pix_fmt_out='gray16le',
                codec='ffv1',
                fps=self.fps,
                log_path=os.path.join(self.cam_dir, "depth_writer.log"),
            )

            # Save depth metadata as a small sidecar JSON (replaces h5 attrs)
            depth_meta = {
                'format': 'ffv1_mkv',
                'pixel_format': 'gray16le',
                'dtype': 'uint16',
                'unit': 'z16_raw',
                'source_stream_format': 'z16',
                'depth_scale_meters_per_unit': self.depth_scale or 0.0,
                'cam_idx': self.cam_idx + 1,
                'aligned_to': 'color' if self.align_depth_live else 'depth',
                'alignment_mode': (
                    'live_realsense_align' if self.align_depth_live
                    else 'none_raw_depth'
                ),
                'codec': 'ffv1',
                'container': 'mkv',
                'lossless': True,
                'note': (
                    'depth_meters = frame.astype(float32) * depth_scale; '
                    'read with: imageio_ffmpeg.read_frames("depth.mkv", pix_fmt="gray16le", bits_per_pixel=16)'
                ),
            }
            depth_meta_path = os.path.join(self.cam_dir, "depth_meta.json")
            with open(depth_meta_path, 'w') as f:
                json.dump(depth_meta, f, indent=2)

        self.frame_count = 0
        self.capture_count = 0
        self.writer_frame_count = 0
        self.queue_drop_count = 0
        self.duplicate_frame_count = 0
        self.max_writer_queue_depth = 0
        self.capture_timestamps = []
        self._last_recorded_color_frame_number = None
        self._writer_error = None
        self.color_timestamps = []
        self.depth_timestamps = []
        self._barrier_passed = False

        # Start writer thread (I/O decoupled from capture)
        self._write_queue = queue.Queue(maxsize=90)  # ~3 seconds at 30fps
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def stop_recording(self):
        """Stop recording, flush writer thread, release video writer, save timestamps."""
        self.recording = False

        # Flush and stop writer thread
        if self._write_queue is not None:
            self._write_queue.put(None)  # sentinel
        if self._writer_thread is not None:
            self._writer_thread.join()
            self._writer_thread = None
        self._write_queue = None

        if self._color_writer is not None:
            try:
                self._color_writer.close()
            except Exception as exc:
                self._writer_error = self._writer_error or repr(exc)
            self._color_writer = None

        # Close FFV1 depth video writer
        if self._depth_writer is not None:
            try:
                self._depth_writer.close()
            except Exception as exc:
                self._writer_error = self._writer_error or repr(exc)
            self._depth_writer = None

        # Save per-frame timestamps as CSV (separate files for color and depth)
        cam_name = f"cam{self.cam_idx + 1}"
        color_ts_copy = list(self.color_timestamps)
        depth_ts_copy = list(self.depth_timestamps)

        if color_ts_copy and self.cam_dir is not None:
            color_ts_path = os.path.join(
                self.cam_dir, f"{cam_name}_color_timestamps.csv")
            with open(color_ts_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame_idx', 'hw_timestamp_ms',
                    'host_timestamp_ms', 'timestamp_domain'
                ])
                for row in color_ts_copy:
                    writer.writerow(row)

        if depth_ts_copy and self.cam_dir is not None:
            depth_ts_path = os.path.join(
                self.cam_dir, f"{cam_name}_depth_timestamps.csv")
            with open(depth_ts_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame_idx', 'hw_timestamp_ms',
                    'host_timestamp_ms', 'timestamp_domain'
                ])
                for row in depth_ts_copy:
                    writer.writerow(row)

        frame_count = self.writer_frame_count
        # Extract hw timestamps for the timing report (color stream as reference)
        hw_timestamps_ms = [row[1] for row in color_ts_copy]
        recording_stats = {
            'capture_frames': self.capture_count,
            'queued_frames': self.frame_count,
            'written_frames': self.writer_frame_count,
            'queue_dropped_frames': self.queue_drop_count,
            'duplicate_frame_numbers_observed': self.duplicate_frame_count,
            'max_writer_queue_depth': self.max_writer_queue_depth,
            'writer_queue_capacity': self.writer_queue_capacity,
            'writer_error': self._writer_error,
            '_capture_samples': list(self.capture_timestamps),
        }
        self.color_timestamps = []
        self.depth_timestamps = []
        self.capture_timestamps = []
        return frame_count, hw_timestamps_ms, recording_stats

    def stop(self):
        """Stop capture thread and release pipeline."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=3)
        if self.recording:
            self.stop_recording()
        if self.pipeline is not None:
            self.pipeline.stop()

    def get_latest_frame(self):
        """Get most recent frame (non-blocking)."""
        frame = None
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        return frame


def generate_frame_timing_report(
    session_dir,
    all_timestamps,
    target_fps,
    roles,
    recording_stats=None,
):
    """Generate human-readable frame-timing diagnostics for all cameras.

    Creates two files in *session_dir*:
        frame_timing_report.csv  – per-frame data (cam, frame_idx, timestamp_ms, delta_ms)
        frame_timing_summary.json – aggregate statistics per camera

    Parameters
    ----------
    session_dir : str
        Directory of the current recording session.
    all_timestamps : dict[int, list[float]]
        Mapping cam_id (1-based) → list of backend_timestamps in milliseconds.
    target_fps : int
        Desired frame rate (e.g. 30).
    roles : list[str]
        Per-camera role labels (e.g. ['pose', 'pose', 'gaze', 'gaze']).
    """
    expected_interval_ms = 1000.0 / target_fps  # e.g. 33.33 ms for 30 fps
    drop_threshold_ms = expected_interval_ms * 1.6   # >60% longer → likely dropped frame
    dup_threshold_ms = expected_interval_ms * 0.4    # <40% of expected → likely duplicate
    recording_stats = recording_stats or {}

    def calculate_timing(timestamp_values):
        values = np.asarray(
            [float(timestamp) for timestamp in timestamp_values],
            dtype=np.float64,
        )
        if values.size < 2:
            return None
        deltas = np.diff(values)
        mean_ms = float(np.mean(deltas))
        return {
            "frame_count": int(values.size),
            "duration_seconds": float((values[-1] - values[0]) / 1000.0),
            "fps": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
            "mean_ms": mean_ms,
            "std_ms": float(np.std(deltas)),
            "min_ms": float(np.min(deltas)),
            "max_ms": float(np.max(deltas)),
            "late_count": int(np.sum(deltas > drop_threshold_ms)),
            "fast_count": int(np.sum(deltas < dup_threshold_ms)),
        }

    csv_path = os.path.join(session_dir, "frame_timing_report.csv")
    summary = {}

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow([
            "camera", "role", "frame_idx", "timestamp_ms",
            "delta_ms", "flag"
        ])

        for cam_id in sorted(all_timestamps.keys()):
            ts_list = [float(t) for t in all_timestamps[cam_id]]
            role = roles[cam_id - 1] if cam_id - 1 < len(roles) else "unknown"
            stats = dict(recording_stats.get(cam_id, {}))
            capture_samples = stats.pop("_capture_samples", [])
            capture_host_ts = [sample[3] for sample in capture_samples]
            unique_capture_hw_ts = []
            last_frame_number = None
            for sample in capture_samples:
                frame_number = sample[1]
                if frame_number == last_frame_number:
                    continue
                unique_capture_hw_ts.append(sample[2])
                last_frame_number = frame_number

            delivery_timing = calculate_timing(capture_host_ts)
            capture_timing = calculate_timing(unique_capture_hw_ts)
            saved_timing = calculate_timing(ts_list)

            for i, t in enumerate(ts_list):
                if i == 0:
                    writer.writerow([f"cam{cam_id}", role, i, round(t, 3), "", ""])
                else:
                    delta = float(t - ts_list[i - 1])
                    flag = ""
                    if delta > drop_threshold_ms:
                        flag = "LATE"
                    elif delta < dup_threshold_ms:
                        flag = "FAST"

                    writer.writerow([
                        f"cam{cam_id}", role, i, round(t, 3),
                        round(delta, 3), flag
                    ])

            capture_fps = (
                capture_timing["fps"] if capture_timing is not None else 0.0
            )
            capture_is_slow = (
                capture_timing is not None
                and capture_fps < target_fps * 0.90
            )
            queue_drops = int(stats.get("queue_dropped_frames", 0))
            queued_frames = int(stats.get("queued_frames", len(ts_list)))
            written_frames = int(stats.get("written_frames", len(ts_list)))
            writer_error = stats.get("writer_error")
            queue_capacity = max(
                1, int(stats.get("writer_queue_capacity", 1))
            )
            max_queue_depth = int(stats.get("max_writer_queue_depth", 0))
            writer_problem = bool(
                writer_error
                or queue_drops
                or written_frames < queued_frames
            )
            writer_backpressure = max_queue_depth >= queue_capacity * 0.80

            if capture_is_slow and (writer_problem or writer_backpressure):
                bottleneck = "combined_capture_and_encoder_or_disk"
            elif writer_problem or writer_backpressure:
                bottleneck = "encoder_or_disk_backpressure"
            elif capture_is_slow:
                bottleneck = "camera_usb_or_capture_side"
            else:
                bottleneck = "none_detected"

            camera_summary = {
                "role": role,
                "total_frames": len(ts_list),
                "target_fps": target_fps,
                "expected_interval_ms": round(expected_interval_ms, 3),
                "capture_frames": int(
                    stats.get("capture_frames", len(capture_samples))
                ),
                "unique_camera_frames": len(unique_capture_hw_ts),
                "queued_frames": queued_frames,
                "written_frames": written_frames,
                "queue_dropped_frames": queue_drops,
                "duplicate_frame_numbers_observed": int(
                    stats.get("duplicate_frame_numbers_observed", 0)
                ),
                "max_writer_queue_depth": max_queue_depth,
                "writer_queue_capacity": queue_capacity,
                "writer_error": writer_error,
                "bottleneck_diagnosis": bottleneck,
            }

            if capture_timing is not None:
                camera_summary.update({
                    "capture_fps_avg": round(capture_timing["fps"], 3),
                    "camera_frame_fps_avg": round(
                        capture_timing["fps"], 3
                    ),
                    "capture_interval_mean_ms": round(
                        capture_timing["mean_ms"], 3
                    ),
                    "capture_interval_std_ms": round(
                        capture_timing["std_ms"], 3
                    ),
                    "capture_late_intervals": capture_timing["late_count"],
                    "capture_fast_intervals": capture_timing["fast_count"],
                })
            else:
                camera_summary["capture_note"] = (
                    "Not enough captured frames for interval analysis"
                )

            if delivery_timing is not None:
                camera_summary["sdk_delivery_fps_avg"] = round(
                    delivery_timing["fps"], 3
                )
            else:
                camera_summary["sdk_delivery_note"] = (
                    "Not enough SDK deliveries for interval analysis"
                )

            if saved_timing is not None:
                camera_summary.update({
                    "duration_seconds": round(
                        saved_timing["duration_seconds"], 3
                    ),
                    "actual_fps_avg": round(saved_timing["fps"], 3),
                    "saved_fps_avg": round(saved_timing["fps"], 3),
                    "interval_mean_ms": round(saved_timing["mean_ms"], 3),
                    "interval_std_ms": round(saved_timing["std_ms"], 3),
                    "interval_min_ms": round(saved_timing["min_ms"], 3),
                    "interval_max_ms": round(saved_timing["max_ms"], 3),
                    "jitter_ms": round(saved_timing["std_ms"], 3),
                    "late_frames (delta > {:.1f}ms)".format(
                        drop_threshold_ms
                    ): saved_timing["late_count"],
                    "fast_frames (delta < {:.1f}ms)".format(
                        dup_threshold_ms
                    ): saved_timing["fast_count"],
                })
            else:
                camera_summary["saved_note"] = (
                    "Not enough written frames for interval analysis"
                )

            summary[f"cam{cam_id}"] = camera_summary

    # Write summary JSON
    summary_path = os.path.join(session_dir, "frame_timing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Frame timing report saved → {csv_path}")
    print(f"Frame timing summary saved → {summary_path}")

    # Print a quick console overview
    print("\n╔══════════════ FRAME TIMING SUMMARY ══════════════╗")
    for cam_key in sorted(summary.keys()):
        s = summary[cam_key]
        capture_fps = s.get("capture_fps_avg", 0.0)
        delivery_fps = s.get("sdk_delivery_fps_avg", 0.0)
        saved_fps = s.get("saved_fps_avg", 0.0)
        print(f"║ {cam_key} ({s['role']}): "
              f"camera {capture_fps:.1f} fps | "
              f"SDK {delivery_fps:.1f} fps | "
              f"saved {saved_fps:.1f} fps | "
              f"written {s['written_frames']} | "
              f"queue-drop {s['queue_dropped_frames']} | "
              f"duplicate-number {s['duplicate_frame_numbers_observed']} | "
              f"queue {s['max_writer_queue_depth']}/"
              f"{s['writer_queue_capacity']} | "
              f"{s['bottleneck_diagnosis']}")
    print("╚══════════════════════════════════════════════════╝\n")


def copy_latest_calibration_to_session(session_dir, recordings_dir):
    """Copy the newest multicam_calibration.npz into the session directory."""
    session_path = Path(session_dir).resolve()
    recordings_path = Path(recordings_dir).resolve()

    candidates = []
    direct = recordings_path / "multicam_calibration.npz"
    if direct.exists():
        candidates.append(direct)

    if recordings_path.exists():
        for path in recordings_path.rglob("multicam_calibration.npz"):
            resolved = path.resolve()
            try:
                resolved.relative_to(session_path)
                continue
            except ValueError:
                pass
            if resolved not in candidates:
                candidates.append(resolved)

    if not candidates:
        print(f"[Calibration Copy] No multicam_calibration.npz found under {recordings_path}")
        return None

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    destination = session_path / "multicam_calibration.npz"
    shutil.copy2(latest, destination)
    print(f"[Calibration Copy] Copied {latest} -> {destination}")
    return str(destination)


def capture_precheck_color_frames(cameras, timeout_s=5.0):
    """Collect one latest color frame per camera from already-started threads."""
    frames = {}
    deadline = time.time() + float(timeout_s)

    while time.time() < deadline and len(frames) < len(cameras):
        for cam in cameras:
            cam_id = cam.cam_idx + 1
            if cam_id in frames:
                continue
            result = cam.get_latest_frame()
            if result is None:
                continue
            color_image, _depth_image = result
            frames[cam_id] = color_image.copy()
        if len(frames) < len(cameras):
            time.sleep(0.02)

    missing = [cam.cam_idx + 1 for cam in cameras if cam.cam_idx + 1 not in frames]
    return frames, missing


def capture_precheck_depth_frames(
    cameras,
    frame_count=5,
    timeout_s=6.0,
    interval_s=0.04,
):
    """Collect a short static depth burst from every running camera."""
    samples = {
        cam.cam_idx + 1: []
        for cam in cameras
    }
    deadline = time.time() + float(timeout_s)
    target_count = max(3, int(frame_count))

    while time.time() < deadline:
        complete = True
        for cam in cameras:
            cam_id = cam.cam_idx + 1
            if len(samples[cam_id]) >= target_count:
                continue
            complete = False
            result = cam.get_latest_frame()
            if result is None:
                continue
            _color_image, depth_image = result
            if depth_image is not None:
                samples[cam_id].append(depth_image.copy())

        if all(len(frames) >= target_count for frames in samples.values()):
            break
        if complete:
            break
        time.sleep(float(interval_s))

    missing = [
        camera_id
        for camera_id, frames in samples.items()
        if len(frames) < target_count
    ]
    return samples, missing


def run_cube_calibration_precheck(args, cameras, output_base):
    """Run the retained AprilTag-cube checker when explicitly requested."""
    from calibration_checker import CalibrationChecker

    if not os.path.exists(args.calib_check_layout):
        raise FileNotFoundError(
            "AprilTag cube layout JSON not found: "
            f"{args.calib_check_layout}. Create it once from "
            "apriltag_cube_layout.example.json using the real CAD coordinates."
        )

    calibration_npz = args.calib_check_npz
    if calibration_npz is None:
        calibration_npz = os.path.join(str(output_base), "multicam_calibration.npz")

    frames_by_camera, missing = capture_precheck_color_frames(
        cameras, timeout_s=args.calib_check_timeout
    )
    if missing:
        raise RuntimeError(f"Could not capture pre-check frames from cameras: {missing}")

    checker = CalibrationChecker(
        calibration_npz,
        args.calib_check_layout,
        families=args.calib_check_family,
        reference_camera=args.calib_check_ref_camera,
        min_tags_per_camera=args.calib_check_min_tags,
        min_points_per_camera=args.calib_check_min_points,
        max_reprojection_error_px=args.calib_check_max_reproj_px,
        min_normal_facing_cos=args.calib_check_min_normal_cos,
        max_rotation_error_deg=args.calib_check_max_rot_deg,
        max_translation_error_mm=args.calib_check_max_trans_mm,
        compare_to_reference_only=not args.calib_check_all_pairs,
        require_all_cameras=not args.calib_check_allow_partial,
    )
    result = checker.check(frames_by_camera)
    checker.print_report(result)

    return result


def run_pcl_alignment_precheck(args, cameras, output_base):
    """Validate live depth alignment without changing the calibration."""
    from pcl_alignment_checker import PCLAlignmentChecker

    calibration_npz = args.calib_check_npz
    if calibration_npz is None:
        calibration_npz = os.path.join(
            str(output_base), "multicam_calibration.npz"
        )

    print(
        f"Capturing {args.pcl_check_frames} static depth frames per camera. "
        "Keep people, chairs, and table objects still..."
    )
    depth_frames, missing = capture_precheck_depth_frames(
        cameras,
        frame_count=args.pcl_check_frames,
        timeout_s=args.calib_check_timeout,
    )
    if missing:
        raise RuntimeError(
            f"Could not collect a complete depth burst from cameras: {missing}"
        )

    camera_models = {}
    for cam in cameras:
        cam_id = cam.cam_idx + 1
        if not cam.calibration_data:
            raise RuntimeError(
                f"Live stream calibration is unavailable for camera {cam_id}"
            )
        camera_models[cam_id] = dict(cam.calibration_data)
        camera_models[cam_id]["aligned_to_color"] = bool(
            cam.align_depth_live
        )

    checker = PCLAlignmentChecker(
        calibration_npz,
        sample_step=args.pcl_check_sample_step,
        pass_median_mm=args.pcl_check_pass_median_mm,
        pass_p75_mm=args.pcl_check_pass_p75_mm,
        pass_inlier_ratio=args.pcl_check_pass_inlier_ratio,
        warn_median_mm=args.pcl_check_warn_median_mm,
        warn_p75_mm=args.pcl_check_warn_p75_mm,
        warn_inlier_ratio=args.pcl_check_warn_inlier_ratio,
    )
    result = checker.check(depth_frames, camera_models)
    checker.print_report(result)
    return result


def run_calibration_precheck(args, cameras, output_base):
    if args.calib_check_method == "cube":
        return run_cube_calibration_precheck(args, cameras, output_base)
    return run_pcl_alignment_precheck(args, cameras, output_base)



def main(args):
    num_cameras = 4
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Load serial mapping from config file
    from utils import load_camera_serials
    config_serials = load_camera_serials(args.cam_config)
    print(f"Loaded camera serials from {args.cam_config}: {config_serials}")

    # Collect serial numbers: config file > auto-detect
    serials = [config_serials.get(i) for i in range(1, num_cameras + 1)]

    # Auto-detect if serials still not specified
    ctx = rs.context()
    devices = ctx.query_devices()
    available_serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]

    if len(available_serials) < num_cameras:
        print(f"Warning: Only {len(available_serials)} RealSense cameras found, need {num_cameras}")

    for i in range(num_cameras):
        if serials[i] is None and i < len(available_serials):
            serials[i] = available_serials[i]

    print(f"Cameras:")
    roles = ['pose', 'pose', 'gaze', 'gaze']
    for i in range(num_cameras):
        print(f"  Cam {i + 1} ({roles[i]}): {serials[i] or 'NOT FOUND'}")
    print("Capture options:")
    print(f"  Live depth alignment: {'ON (depth->color)' if args.align_depth_live else 'OFF (record raw depth)'}")
    print("  Color format: FFV1 lossless MKV (color.mkv, 8-bit BGR stored as bgr0)")
    print(f"  Depth format: FFV1 lossless MKV (16-bit grayscale)")

    # Initialize camera threads
    cameras = []
    for i in range(num_cameras):
        if serials[i] is None:
            print(f"Error: No serial for camera {i + 1}. Add it to {args.cam_config}")
            return 2

        cam = CameraThread(
            i,
            serials[i],
            args.width,
            args.height,
            args.fps,
            align_depth_live=args.align_depth_live,
        )
        cameras.append(cam)

    # Shared synchronization primitives for recording
    record_event = threading.Event()
    record_barrier = threading.Barrier(num_cameras)
    for cam in cameras:
        cam.record_event = record_event
        cam.record_barrier = record_barrier

    print(f"\nStarting {num_cameras} camera streams...")
    for cam in cameras:
        cam.start()
        connection = cam.usb_type or "unknown USB"
        port = cam.physical_port or "unknown port"
        print(
            f"  Camera {cam.cam_idx + 1} ({cam.serial}) started | "
            f"USB {connection} | {port}"
        )

    time.sleep(1)  # Let threads stabilize

    if args.pcl_check_only:
        print("\nRunning one PCL alignment check, then exiting...")
        try:
            result = run_pcl_alignment_precheck(args, cameras, output_base)
            return 0 if result.ok else 2
        except Exception as exc:
            print(f"\n\033[1;31mPCL alignment check failed: {exc}\033[0m")
            return 2
        finally:
            for cam in cameras:
                cam.stop()

    # Runtime mode
    use_gui = (not args.no_gui) and bool(os.environ.get('DISPLAY'))

    if not use_gui:
        if args.no_gui:
            print("\nRunning in no-gui mode.")
        else:
            print("\nDISPLAY not found. Falling back to no-gui mode.")

    # Main loop state
    is_recording = False
    session_dir = None
    session_start_time = None
    session_calibration_path = None
    calib_precheck_ok = not args.calib_check
    calib_precheck_status = "disabled" if not args.calib_check else "pending"
    session_precheck_status = None

    def run_precheck_gate():
        nonlocal calib_precheck_ok, calib_precheck_status
        if args.calib_check_method == "cube":
            print("\nRunning AprilTag cube calibration pre-check...")
        else:
            print("\nRunning live depth/PCL alignment pre-check...")
        try:
            precheck_result = run_calibration_precheck(args, cameras, output_base)
        except Exception as exc:
            print(f"\n\033[1;31mCalibration pre-check failed: {exc}\033[0m")
            calib_precheck_ok = False
            calib_precheck_status = "failed"
            return False

        calib_precheck_ok = bool(precheck_result.ok)
        if calib_precheck_ok:
            calib_precheck_status = "passed"
            print("\n\033[1;32mPre-check OK. Press R to start recording.\033[0m")
        else:
            calib_precheck_status = "failed"
            if args.calib_check_method == "cube":
                remedy = "Adjust cube/cameras"
            else:
                remedy = "Keep the setup still and inspect camera/calibration alignment"
            print(
                f"\n\033[1;31mPre-check failed. {remedy}, "
                "then press C again.\033[0m"
            )
        return calib_precheck_ok

    def bypass_precheck_gate():
        nonlocal calib_precheck_ok, calib_precheck_status
        if not args.calib_check:
            return True
        if calib_precheck_status == "passed":
            print("\n\033[1;32mPre-check already passed. Press R to record.\033[0m")
            return True
        calib_precheck_ok = True
        calib_precheck_status = "bypassed"
        print(
            "\n\033[1;33mExecutive bypass enabled with E. "
            "Press R to record; metadata will mark this session as bypassed."
            "\033[0m"
        )
        return True

    def start_recording_session():
        nonlocal is_recording, session_dir, session_start_time
        nonlocal session_calibration_path, session_precheck_status
        if is_recording:
            return True

        if args.calib_check and not calib_precheck_ok:
            print(
                "\n\033[1;33mRecord locked: press C for the pre-check "
                "or E to bypass it before pressing R.\033[0m"
            )
            return False

        session_start_time = datetime.now()
        timestamp = session_start_time.strftime("%Y-%m-%d_%H:%M")
        session_path = output_base / f"session_{timestamp}"
        collision_index = 2
        while session_path.exists():
            session_path = output_base / (
                f"session_{timestamp}_{collision_index:02d}"
            )
            collision_index += 1
        session_path.mkdir(parents=True, exist_ok=False)
        session_dir = str(session_path)
        session_calibration_path = copy_latest_calibration_to_session(session_dir, output_base)
        session_precheck_status = calib_precheck_status
        for cam in cameras:
            cam.prepare_recording(session_dir)

        record_event.set()
        is_recording = True
        print(f"\nRecording started -> {session_dir}")
        return True

    def stop_recording_session():
        nonlocal is_recording, calib_precheck_ok
        nonlocal calib_precheck_status, session_precheck_status
        if not is_recording:
            return

        record_event.clear()
        for cam in cameras:
            cam.recording = False
        time.sleep(0.02)

        # Reset barrier for potential next recording session
        try:
            record_barrier.reset()
        except threading.BrokenBarrierError:
            pass

        frame_counts = {}
        all_timestamps = {}  # cam_idx+1 -> list of timestamps (ms)
        recording_stats = {}
        for cam in cameras:
            fc, ts, stats = cam.stop_recording()
            frame_counts[cam.cam_idx + 1] = fc
            all_timestamps[cam.cam_idx + 1] = ts
            recording_stats[cam.cam_idx + 1] = stats

        session_record_end = datetime.now()

        metadata = {
            'session_start': session_start_time.isoformat(),
            'session_end': session_record_end.isoformat(),
            'resolution': {'width': args.width, 'height': args.height},
            'fps': args.fps,
            'session_calibration_file': (
                os.path.basename(session_calibration_path)
                if session_calibration_path else None
            ),
            'calibration_precheck': {
                'required': bool(args.calib_check),
                'method': (
                    args.calib_check_method if args.calib_check else None
                ),
                'status': (
                    session_precheck_status or calib_precheck_status
                ),
            },
            'cameras': {}
        }

        for i, cam in enumerate(cameras):
            metadata['cameras'][str(i + 1)] = {
                'serial': cam.serial,
                'role': roles[i],
                'usb_type': cam.usb_type,
                'physical_port': cam.physical_port,
                'intrinsics': cam.intrinsics_data,
                'calibration': cam.calibration_data,
                'frame_count': frame_counts[i + 1],
                'recording_health': {
                    key: value
                    for key, value in recording_stats[i + 1].items()
                    if not key.startswith('_')
                },
                'timestamp_source': 'realsense_hardware_clock',
                'timestamp_host_reference': 'perf_counter',
                'timestamp_file_format': 'csv',
                'color_timestamp_file': f"cam{i + 1}/cam{i + 1}_color_timestamps.csv",
                'depth_timestamp_file': f"cam{i + 1}/cam{i + 1}_depth_timestamps.csv",
                'timestamp_columns': [
                    'frame_idx', 'hw_timestamp_ms',
                    'host_timestamp_ms', 'timestamp_domain',
                ],
                'color_storage': {
                    'format': 'ffv1_mkv',
                    'file': f"cam{i + 1}/color.mkv",
                    'pixel_format_in': 'bgr24',
                    'pixel_format_out': 'bgr0',
                    'source_stream_format': 'bgr8',
                    'codec': 'ffv1',
                    'container': 'mkv',
                    'lossless': True,
                    'note': (
                        'RealSense color stream is 8-bit BGR. FFV1 stores it '
                        'losslessly; bgr0 adds a constant unused padding byte '
                        'because this FFmpeg FFV1 encoder does not support '
                        'bgr24 output.'
                    ),
                },
                'depth_storage': {
                    'format': 'ffv1_mkv',
                    'file': f"cam{i + 1}/depth.mkv",
                    'pixel_format': 'gray16le',
                    'dtype': 'uint16',
                    'unit': 'z16_raw',
                    'codec': 'ffv1',
                    'container': 'mkv',
                    'lossless': True,
                    'source_stream_format': 'z16',
                    'aligned_to': 'color' if cam.align_depth_live else 'depth',
                    'alignment_mode': 'live_realsense_align' if cam.align_depth_live else 'none_raw_depth',
                    'depth_scale_meters_per_unit': cam.depth_scale,
                    'note': (
                        'depth_meters = frame.astype(float32) * depth_scale; '
                        'read with: imageio_ffmpeg.read_frames("depth.mkv", pix_fmt="gray16le", bits_per_pixel=16); '
                        'if alignment_mode=none_raw_depth run devel/align_depth_postprocess.py '
                        'to generate aligned depth'
                    ),
                },
            }

        metadata_path = os.path.join(session_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # --- Frame timing diagnostics ---
        generate_frame_timing_report(
            session_dir,
            all_timestamps,
            args.fps,
            roles,
            recording_stats=recording_stats,
        )

        is_recording = False
        if args.calib_check:
            calib_precheck_ok = False
            calib_precheck_status = "pending"
        session_precheck_status = None
        print(f"Recording stopped. Frames: {frame_counts}")
        print(f"Metadata saved to {metadata_path}")

    window_name = "4-Camera Session (C=Check, E=Executive, R=Record, Q=Quit)"
    if use_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        except cv2.error:
            print(
                "[GUI] Full-screen mode is unavailable; "
                "using a resizable window."
            )

    # Cache last good frame per camera to avoid black-frame flicker
    last_frames = None
    if use_gui:
        last_frames = [np.zeros((args.height, args.width, 3), dtype=np.uint8)
                       for _ in range(num_cameras)]

    if use_gui:
        if args.calib_check:
            print(
                "\nReady. Check camera views first. Press C to run the "
                "pre-check, E to bypass it, R to record when unlocked, "
                "or Q to quit."
            )
        else:
            print("\nReady. Press R to start recording, Q to quit.")
    else:
        if args.calib_check:
            print("\nReady. Recording will run pre-check automatically. Press Ctrl+C to stop.")
        else:
            print("\nReady. Recording will start automatically. Press Ctrl+C to stop.")

    try:
        last_status_log = 0.0
        while True:
            if use_gui and not is_recording:
                # --- Full camera preview ---
                display_images = []
                for i, cam in enumerate(cameras):
                    result = cam.get_latest_frame()
                    if result is not None:
                        color, _depth = result
                        last_frames[i] = color
                    display_images.append(last_frames[i])

                grid_images = []
                for i, img in enumerate(display_images[:num_cameras]):
                    resized = cv2.resize(img, (640, 360))
                    label = f"Cam {i + 1} ({roles[i]})"
                    cv2.putText(resized, label, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    grid_images.append(resized)

                while len(grid_images) < 4:
                    grid_images.append(np.zeros((360, 640, 3), dtype=np.uint8))

                row1 = np.hstack(grid_images[:2])
                row2 = np.hstack(grid_images[2:4])
                combined = np.vstack([row1, row2])

                if args.calib_check:
                    if calib_precheck_status == "passed":
                        status_text = "Calibration OK - Press R to start recording | Q quit"
                        status_color = (0, 255, 0)
                    elif calib_precheck_status == "bypassed":
                        status_text = "Pre-check BYPASSED (E) - Press R to record | Q quit"
                        status_color = (0, 165, 255)
                    else:
                        status_text = "C: pre-check | E: executive bypass | R locked | Q: quit"
                        status_color = (0, 255, 255)
                    cv2.rectangle(combined, (0, 0), (combined.shape[1], 42), (0, 0, 0), -1)
                    cv2.putText(combined, status_text, (18, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

                cv2.imshow(window_name, combined)
                key = cv2.waitKey(1) & 0xFF
            elif use_gui:
                # --- Minimal status during recording (no camera frame processing) ---
                elapsed = (datetime.now() - session_start_time).total_seconds()
                status = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(status, "RECORDING", (200, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(status, f"Elapsed: {elapsed:.1f}s", (220, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                for i, cam in enumerate(cameras):
                    y = 180 + i * 40
                    cv2.putText(status,
                                (
                                    f"Cam {i+1} ({roles[i]}): "
                                    f"{cam.writer_frame_count} written, "
                                    f"{cam.queue_drop_count} dropped"
                                ),
                                (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 1)
                cv2.putText(status, "Press R to stop", (220, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow(window_name, status)
                key = cv2.waitKey(100) & 0xFF
            else:
                # --- Headless mode: auto-start and periodic terminal status ---
                if not is_recording:
                    if args.calib_check and not calib_precheck_ok:
                        if not run_precheck_gate():
                            break
                    if not start_recording_session():
                        break

                now = time.time()
                if now - last_status_log >= 1.0 and session_start_time is not None:
                    elapsed = (datetime.now() - session_start_time).total_seconds()
                    counts = ", ".join([
                        (
                            f"cam{i+1}:{cam.writer_frame_count}"
                            f"/drop:{cam.queue_drop_count}"
                        )
                        for i, cam in enumerate(cameras)
                    ])
                    print(f"Recording... {elapsed:.1f}s | {counts}", end='\r', flush=True)
                    last_status_log = now

                time.sleep(0.05)
                continue

            if key == ord('q') or key == 27:
                break

            if key == ord('c') and args.calib_check and not is_recording:
                run_precheck_gate()
                continue

            if (
                key in (ord('e'), ord('E'))
                and args.calib_check
                and not is_recording
            ):
                bypass_precheck_gate()
                continue

            if key == ord('r'):
                if not is_recording:
                    start_recording_session()

                else:
                    stop_recording_session()

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        # Stop recording if still active
        if is_recording:
            print("Stopping active recording...")
            stop_recording_session()

        # Stop all cameras
        for cam in cameras:
            cam.stop()

        if use_gui:
            cv2.destroyAllWindows()
        print("Session ended.")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Record synchronized color + depth from 4 RealSense cameras "
            "(FFV1 lossless color.mkv + FFV1 lossless depth.mkv)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--cam-config', type=str, default=os.path.join(script_dir, 'camera_config.json'),
                        help='Path to camera config JSON file mapping cam IDs to serial numbers')
    parser.add_argument('--output-dir', type=str, default=os.path.join(script_dir, 'recordings'),
                        help='Base directory for recordings')
    parser.add_argument('--width', type=int, default=1280, help='Frame width')
    parser.add_argument('--height', type=int, default=720, help='Frame height')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument(
        '--align-depth-live',
        action='store_true',
        help=(
            'Align depth to color during capture. This increases CPU load; '
            'default is OFF (record raw depth and align in post-process).'
        ),
    )
    parser.add_argument('--no-gui', action='store_true',
                        help='Run without OpenCV windows (headless mode)')
    parser.add_argument(
        '--calib-check',
        dest='calib_check',
        action='store_true',
        default=True,
        help='Require a calibration pre-check before recording can start',
    )
    parser.add_argument(
        '--no-calib-check',
        dest='calib_check',
        action='store_false',
        help='Skip the calibration pre-check',
    )
    parser.add_argument(
        '--calib-check-method',
        choices=('pcl', 'cube'),
        default='pcl',
        help=(
            'Pre-check method. pcl directly validates live depth-cloud '
            'alignment; cube retains the legacy AprilTag check.'
        ),
    )
    parser.add_argument(
        '--pcl-check-only',
        action='store_true',
        help=(
            'Start all cameras, run one live PCL alignment check, and exit '
            'without creating a recording session'
        ),
    )
    parser.add_argument(
        '--calib-check-layout',
        type=str,
        default=(
            os.path.join(script_dir, 'apriltag_cube_layout_calibrated.json')
            if os.path.exists(os.path.join(script_dir, 'apriltag_cube_layout_calibrated.json'))
            else os.path.join(script_dir, 'apriltag_cube_layout.json')
        ),
        help='JSON file with measured or ideal cube AprilTag 3D corner coordinates',
    )
    parser.add_argument(
        '--calib-check-npz',
        type=str,
        default=None,
        help='Morning multicam_calibration.npz (default: {output_dir}/multicam_calibration.npz)',
    )
    parser.add_argument('--calib-check-family', type=str, default='tag36h11',
                        help='AprilTag family for the cube tags')
    parser.add_argument('--calib-check-ref-camera', type=int, default=1,
                        help='Reference camera for pair checks')
    parser.add_argument('--calib-check-timeout', type=float, default=5.0,
                        help='Seconds to collect pre-check frames from every camera')
    parser.add_argument(
        '--pcl-check-frames',
        type=int,
        default=5,
        help='Static depth frames per camera used for temporal median filtering',
    )
    parser.add_argument(
        '--pcl-check-sample-step',
        type=int,
        default=6,
        help='Depth pixel sampling step used by the PCL check',
    )
    parser.add_argument(
        '--pcl-check-pass-median-mm',
        type=float,
        default=35.0,
        help='PASS limit for bidirectional median depth disagreement',
    )
    parser.add_argument(
        '--pcl-check-pass-p75-mm',
        type=float,
        default=65.0,
        help='PASS limit for bidirectional 75th-percentile disagreement',
    )
    parser.add_argument(
        '--pcl-check-pass-inlier-ratio',
        type=float,
        default=0.35,
        help='Minimum PASS fraction within the depth-dependent inlier threshold',
    )
    parser.add_argument(
        '--pcl-check-warn-median-mm',
        type=float,
        default=50.0,
        help='WARN limit for bidirectional median depth disagreement',
    )
    parser.add_argument(
        '--pcl-check-warn-p75-mm',
        type=float,
        default=100.0,
        help='WARN limit for bidirectional 75th-percentile disagreement',
    )
    parser.add_argument(
        '--pcl-check-warn-inlier-ratio',
        type=float,
        default=0.20,
        help='Minimum WARN fraction within the depth-dependent inlier threshold',
    )
    parser.add_argument('--calib-check-min-tags', type=int, default=1,
                        help='Minimum known cube tags required per camera')
    parser.add_argument('--calib-check-min-points', type=int, default=4,
                        help='Minimum 2D-3D correspondences required per camera')
    parser.add_argument('--calib-check-max-reproj-px', type=float, default=3.0,
                        help='Maximum per-camera solvePnP reprojection error')
    parser.add_argument('--calib-check-min-normal-cos', type=float, default=0.05,
                        help='Minimum visible-face normal facing score; +1 faces camera, 0 is grazing, negative faces away')
    parser.add_argument('--calib-check-max-rot-deg', type=float, default=4.0,
                        help='Maximum allowed camera-pair rotation delta in degrees')
    parser.add_argument('--calib-check-max-trans-mm', type=float, default=50.0,
                        help='Maximum allowed camera-pair translation delta in millimeters')
    parser.add_argument('--calib-check-all-pairs', action='store_true',
                        help='Compare every visible camera pair instead of only ref-camera pairs')
    parser.add_argument('--calib-check-allow-partial', action='store_true',
                        help='Allow pre-check to pass when only a subset of cameras see the cube')

    args = parser.parse_args()
    raise SystemExit(main(args))
