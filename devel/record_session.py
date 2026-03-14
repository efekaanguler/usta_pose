#!/usr/bin/env python3
"""
Synchronized 4-Camera Session Recorder

Records color video + depth frames from 4 RealSense cameras:
  - Cameras 1, 2 (tripod): Capture body pose
  - Cameras 3, 4 (table): Capture gaze

Output structure:
    recordings/session_YYYYMMDD_HHMMSS/
        cam1_color.mp4       cam1_depth/frame_NNNNNN.npy  (float32, meters)
        cam2_color.mp4       cam2_depth/...
        cam3_color.mp4       cam3_depth/...
        cam4_color.mp4       cam4_depth/...
        metadata.json

Controls:
    R: Toggle recording on/off
    Q: Quit
    Ctrl+C: Stop and quit (headless/no-gui mode)

Usage:
    python record_session.py --output-dir ./recordings

    Serial numbers are read from camera_config.json (see --cam-config).
"""

import argparse
import cv2
import h5py
import numpy as np
import pyrealsense2 as rs
import os
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path


class CameraThread:
    """Manages a single RealSense camera in its own capture thread."""

    def __init__(self, cam_idx, serial, width, height, fps, enable_depth=True):
        self.cam_idx = cam_idx
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        self.pipeline = None
        self.align = None
        self.intrinsics_data = None
        self.calibration_data = None
        self.depth_scale = None  # meters per depth unit (z16)

        # Thread communication
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None

        # Recording state (controlled by main thread)
        self.record_event = None  # shared threading.Event for synchronized start
        self.record_barrier = None  # shared threading.Barrier for synchronized first frame
        self.recording = False
        self._barrier_passed = False  # True once this thread has passed the barrier
        self.video_writer = None
        self.cam_dir = None       # per-camera subdirectory inside session_dir
        self.session_dir = None
        self.frame_count = 0
        self.timestamps = []  # host-clock timestamps (ms) per recorded frame

        # HDF5 depth storage (single resizable dataset, one frame at a time)
        self._depth_h5 = None
        self._depth_dataset = None

        # Writer thread: decouples disk I/O from the capture loop so that
        # slow VideoWriter.write / h5py append never block frame capture.
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
        for sensor in device.sensors:
            # Force frame-rate priority: prevent FPS drops in low light
            if sensor.supports(rs.option.auto_exposure_priority):
                sensor.set_option(rs.option.auto_exposure_priority, 0)

        # Align depth to color
        if self.enable_depth:
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

    def _capture_loop(self):
        """Continuous frame capture in background thread.

        IMPORTANT: Frame data is copied with np.array() (not np.asanyarray)
        because the zero-copy view into librealsense's internal buffer can be
        recycled before VideoWriter / imwrite finish reading it, causing a
        segfault.  All slow disk I/O is offloaded to _writer_loop via
        _write_queue so that this loop keeps up with wait_for_frames().
        """
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue

            if self.enable_depth and self.align:
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
            else:
                color_frame = frames.get_color_frame()
                depth_frame = None

            if not color_frame:
                continue

            # COPY frame data — np.asanyarray returns a view into RS's buffer
            # which can be freed before write() finishes, causing a segfault.
            color_image = np.array(color_frame.get_data())

            depth_image = None
            if depth_frame:
                depth_image = np.array(depth_frame.get_data())

            # Check for synchronized recording start (barrier-gated)
            if not self.recording and self.video_writer is not None:
                if self.record_event is not None and self.record_event.is_set():
                    # Wait at barrier so all cameras start on the same frame
                    if not self._barrier_passed and self.record_barrier is not None:
                        self.record_barrier.wait()
                        self._barrier_passed = True
                    self.recording = True

            # Record if active — hand off to writer thread (no I/O here)
            if self.recording:
                # Use time.perf_counter() for sub-millisecond precision.
                # backend_timestamp gives only integer ms (33.0, 34.0 etc.)
                # which hides the real jitter between frames.
                ts = time.perf_counter() * 1000.0  # ms with µs precision

                # Enqueue for the writer thread; drop frame if queue full
                if self._write_queue is not None:
                    try:
                        self._write_queue.put_nowait(
                            (color_image, depth_image, self.frame_count, ts))
                        self.frame_count += 1
                    except queue.Full:
                        pass  # writer can't keep up — drop this frame

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
        """Disk I/O thread — writes video frames and depth into HDF5.

        Runs in a separate thread so that VideoWriter.write and h5py appends
        never block the capture loop.  Depth frames are appended one at a time
        to a resizable dataset; RAM usage stays constant regardless of session
        length.
        """
        while True:
            item = self._write_queue.get()
            if item is None:  # sentinel from stop_recording
                break
            color_image, depth_image, frame_idx, ts = item
            self.timestamps.append(ts)

            if self.video_writer is not None:
                self.video_writer.write(color_image)

            if depth_image is not None and self._depth_dataset is not None:
                # Store raw uint16 z16 values — multiply by depth_scale for
                # meters at read time.  Much smaller than float32.
                depth_z16 = depth_image.astype(np.uint16)
                n = self._depth_dataset.shape[0]
                self._depth_dataset.resize(n + 1, axis=0)
                self._depth_dataset[n] = depth_z16

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

        # Video writer
        video_path = os.path.join(self.cam_dir, "color.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps,
                                            (self.width, self.height))

        # Depth HDF5 file with a resizable dataset — no per-frame files needed
        if self.enable_depth:
            depth_h5_path = os.path.join(self.cam_dir, "depth.h5")
            self._depth_h5 = h5py.File(depth_h5_path, 'w')
            self._depth_dataset = self._depth_h5.create_dataset(
                'depth',
                shape=(0, self.height, self.width),
                maxshape=(None, self.height, self.width),
                dtype=np.uint16,
                chunks=(1, self.height, self.width),
                compression='gzip',
                compression_opts=4,
                shuffle=True,       # byte-shuffle before gzip → much better ratio on uint16
            )
            self._depth_h5.attrs['unit'] = 'z16_raw'
            self._depth_h5.attrs['dtype'] = 'uint16'
            self._depth_h5.attrs['source_stream_format'] = 'z16'
            self._depth_h5.attrs['depth_scale_meters_per_unit'] = self.depth_scale or 0.0
            self._depth_h5.attrs['cam_idx'] = self.cam_idx + 1

        self.frame_count = 0
        self.timestamps = []
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
            self._writer_thread.join(timeout=30)  # allow time to flush
            self._writer_thread = None
        self._write_queue = None

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        # Close depth HDF5 file (all frames already written by writer thread)
        if self._depth_h5 is not None:
            self._depth_h5.close()
            self._depth_h5 = None
            self._depth_dataset = None

        # Save per-frame timestamps as HDF5
        timestamps_copy = list(self.timestamps)
        if timestamps_copy and self.cam_dir is not None:
            ts_path = os.path.join(self.cam_dir, "timestamps.h5")
            ts_array = np.array(timestamps_copy, dtype=np.float64)
            with h5py.File(ts_path, 'w') as hf:
                ds = hf.create_dataset('timestamps', data=ts_array)
                ds.attrs['cam_idx'] = self.cam_idx + 1
                ds.attrs['source'] = 'perf_counter'
                ds.attrs['unit'] = 'milliseconds'
                ds.attrs['dtype'] = 'float64'
                ds.attrs['frame_count'] = len(ts_array)

        frame_count = self.frame_count
        self.timestamps = []
        return frame_count, timestamps_copy

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


def generate_frame_timing_report(session_dir, all_timestamps, target_fps, roles):
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
    import csv

    expected_interval_ms = 1000.0 / target_fps  # e.g. 33.33 ms for 30 fps
    # Thresholds for anomaly detection
    drop_threshold_ms = expected_interval_ms * 1.6   # >60% longer → likely dropped frame
    dup_threshold_ms = expected_interval_ms * 0.4     # <40% of expected → likely duplicate

    csv_path = os.path.join(session_dir, "frame_timing_report.csv")
    summary = {}

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow([
            "camera", "role", "frame_idx", "timestamp_ms",
            "delta_ms", "flag"
        ])

        for cam_id in sorted(all_timestamps.keys()):
            # Force float so deltas are always fractional, never integer
            ts_list = [float(t) for t in all_timestamps[cam_id]]
            role = roles[cam_id - 1] if cam_id - 1 < len(roles) else "unknown"

            if len(ts_list) < 2:
                # Not enough frames for meaningful analysis
                writer.writerow([f"cam{cam_id}", role, 0,
                                 ts_list[0] if ts_list else "", "", ""])
                summary[f"cam{cam_id}"] = {
                    "role": role,
                    "total_frames": len(ts_list),
                    "note": "Not enough frames for interval analysis"
                }
                continue

            deltas = []
            drop_count = 0
            dup_count = 0

            for i, t in enumerate(ts_list):
                if i == 0:
                    writer.writerow([f"cam{cam_id}", role, i, round(t, 3), "", ""])
                else:
                    delta = float(t - ts_list[i - 1])
                    deltas.append(delta)

                    flag = ""
                    if delta > drop_threshold_ms:
                        flag = "LATE"
                        drop_count += 1
                    elif delta < dup_threshold_ms:
                        flag = "FAST"
                        dup_count += 1

                    writer.writerow([
                        f"cam{cam_id}", role, i, round(t, 3),
                        round(delta, 3), flag
                    ])

            deltas_arr = np.array(deltas)
            actual_mean = float(np.mean(deltas_arr))
            actual_std = float(np.std(deltas_arr))
            actual_min = float(np.min(deltas_arr))
            actual_max = float(np.max(deltas_arr))

            actual_fps = 1000.0 / actual_mean if actual_mean > 0 else 0.0
            duration_s = (ts_list[-1] - ts_list[0]) / 1000.0

            summary[f"cam{cam_id}"] = {
                "role": role,
                "total_frames": len(ts_list),
                "duration_seconds": round(duration_s, 3),
                "target_fps": target_fps,
                "expected_interval_ms": round(expected_interval_ms, 3),
                "actual_fps_avg": round(actual_fps, 3),
                "interval_mean_ms": round(actual_mean, 3),
                "interval_std_ms": round(actual_std, 3),
                "interval_min_ms": round(actual_min, 3),
                "interval_max_ms": round(actual_max, 3),
                "jitter_ms": round(actual_std, 3),
                "late_frames (delta > {:.1f}ms)".format(drop_threshold_ms): drop_count,
                "fast_frames (delta < {:.1f}ms)".format(dup_threshold_ms): dup_count,
            }

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
        if "note" in s:
            print(f"║ {cam_key} ({s['role']}): {s['note']}")
            continue
        print(f"║ {cam_key} ({s['role']}): "
              f"{s['total_frames']} frames | "
              f"avg {s['actual_fps_avg']:.1f} fps | "
              f"interval {s['interval_mean_ms']:.1f}±{s['interval_std_ms']:.1f} ms | "
              f"min {s['interval_min_ms']:.1f} max {s['interval_max_ms']:.1f} ms | "
              f"late:{s.get(list(k for k in s if k.startswith('late'))[0], 0)} "
              f"fast:{s.get(list(k for k in s if k.startswith('fast'))[0], 0)}")
    print("╚══════════════════════════════════════════════════╝\n")


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

    # Initialize camera threads
    cameras = []
    for i in range(num_cameras):
        if serials[i] is None:
            print(f"Error: No serial for camera {i + 1}. Add it to {args.cam_config}")
            return

        cam = CameraThread(i, serials[i], args.width, args.height, args.fps)
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
        print(f"  Camera {cam.cam_idx + 1} ({cam.serial}) started")

    time.sleep(1)  # Let threads stabilize

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

    def start_recording_session():
        nonlocal is_recording, session_dir, session_start_time
        if is_recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = str(output_base / f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        session_start_time = datetime.now()
        for cam in cameras:
            cam.prepare_recording(session_dir)

        record_event.set()
        is_recording = True
        print(f"\nRecording started -> {session_dir}")

    def stop_recording_session():
        nonlocal is_recording
        if not is_recording:
            return

        record_event.clear()
        time.sleep(0.05)  # let capture threads see cleared event

        # Reset barrier for potential next recording session
        try:
            record_barrier.reset()
        except threading.BrokenBarrierError:
            pass

        frame_counts = {}
        all_timestamps = {}  # cam_idx+1 -> list of timestamps (ms)
        for cam in cameras:
            fc, ts = cam.stop_recording()
            frame_counts[cam.cam_idx + 1] = fc
            all_timestamps[cam.cam_idx + 1] = ts

        metadata = {
            'session_start': session_start_time.isoformat(),
            'session_end': datetime.now().isoformat(),
            'resolution': {'width': args.width, 'height': args.height},
            'fps': args.fps,
            'cameras': {}
        }

        for i, cam in enumerate(cameras):
            metadata['cameras'][str(i + 1)] = {
                'serial': cam.serial,
                'role': roles[i],
                'intrinsics': cam.intrinsics_data,
                'calibration': cam.calibration_data,
                'frame_count': frame_counts[i + 1],
                'timestamp_source': 'perf_counter',
                'timestamp_file_format': 'hdf5',
                'timestamp_file': f"cam{i + 1}/timestamps.h5",
                'timestamp_dataset': 'timestamps',
                'depth_storage': {
                    'format': 'hdf5',
                    'file': f"cam{i + 1}/depth.h5",
                    'dataset': 'depth',
                    'shape': '(N, height, width)',
                    'dtype': 'uint16',
                    'unit': 'z16_raw',
                    'compression': 'gzip_level4_shuffle',
                    'source_stream_format': 'z16',
                    'depth_scale_meters_per_unit': cam.depth_scale,
                    'note': 'depth_meters = data.astype(float32) * depth_scale',
                },
            }

        metadata_path = os.path.join(session_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # --- Frame timing diagnostics ---
        generate_frame_timing_report(
            session_dir, all_timestamps, args.fps, roles
        )

        is_recording = False
        print(f"Recording stopped. Frames: {frame_counts}")
        print(f"Metadata saved to {metadata_path}")

    window_name = "4-Camera Session (R=Record, Q=Quit)"
    if use_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Cache last good frame per camera to avoid black-frame flicker
    last_frames = None
    if use_gui:
        last_frames = [np.zeros((args.height, args.width, 3), dtype=np.uint8)
                       for _ in range(num_cameras)]

    if use_gui:
        print("\nReady. Press R to start recording, Q to quit.")
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
                                f"Cam {i+1} ({roles[i]}): {cam.frame_count} frames",
                                (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 1)
                cv2.putText(status, "Press R to stop", (220, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow(window_name, status)
                key = cv2.waitKey(100) & 0xFF
            else:
                # --- Headless mode: auto-start and periodic terminal status ---
                if not is_recording:
                    start_recording_session()

                now = time.time()
                if now - last_status_log >= 1.0 and session_start_time is not None:
                    elapsed = (datetime.now() - session_start_time).total_seconds()
                    counts = ", ".join([f"cam{i+1}:{cam.frame_count}" for i, cam in enumerate(cameras)])
                    print(f"Recording... {elapsed:.1f}s | {counts}", end='\r', flush=True)
                    last_status_log = now

                time.sleep(0.05)
                continue

            if key == ord('q') or key == 27:
                break

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Record synchronized color + depth from 4 RealSense cameras",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--cam-config', type=str, default='./camera_config.json',
                        help='Path to camera config JSON file mapping cam IDs to serial numbers')
    parser.add_argument('--output-dir', type=str, default='./recordings',
                        help='Base directory for recordings')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run without OpenCV windows (headless mode)')

    args = parser.parse_args()
    main(args)
