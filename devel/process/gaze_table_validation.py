#!/usr/bin/env python3
"""Project raw Cam3 gaze onto the measured table plane for visual validation."""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


RVIZ_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass
class TimestampTable:
    frame_indices: np.ndarray
    timestamps_ms: np.ndarray

    @classmethod
    def from_csv(cls, path):
        frame_indices = []
        timestamps_ms = []
        with open(path, "r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                frame_indices.append(int(row["frame_idx"]))
                timestamps_ms.append(float(row["hw_timestamp_ms"]))
        if not frame_indices:
            raise RuntimeError(f"Timestamp CSV is empty: {path}")
        order = np.argsort(timestamps_ms)
        return cls(
            np.asarray(frame_indices, dtype=np.int64)[order],
            np.asarray(timestamps_ms, dtype=np.float64)[order],
        )

    def nearest(self, timestamp_ms):
        position = int(np.searchsorted(self.timestamps_ms, timestamp_ms))
        candidates = np.clip(
            np.asarray([position - 1, position], dtype=np.int64),
            0,
            len(self.timestamps_ms) - 1,
        )
        best = candidates[
            int(np.argmin(np.abs(self.timestamps_ms[candidates] - timestamp_ms)))
        ]
        return (
            int(self.frame_indices[best]),
            float(self.timestamps_ms[best]),
            float(abs(self.timestamps_ms[best] - timestamp_ms)),
        )


@dataclass
class TablePlane:
    origin: np.ndarray
    normal: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    table_u_min: float
    table_u_max: float
    table_v_min: float
    table_v_max: float
    texture_points: np.ndarray
    texture_colors: np.ndarray
    inlier_count: int
    residual_mm: float

    def coordinates(self, points):
        values = np.asarray(points, dtype=np.float64)
        relative = values - self.origin
        return np.column_stack(
            [relative @ self.axis_u, relative @ self.axis_v]
        )

    def intersect(self, ray_origin, ray_direction, max_distance_m):
        denominator = float(np.dot(self.normal, ray_direction))
        if abs(denominator) < 1e-8:
            return None, None, "PARALLEL TO TABLE"
        distance = float(
            np.dot(self.normal, self.origin - ray_origin) / denominator
        )
        if distance <= 0.05:
            return None, distance, "INTERSECTION BEHIND EYES"
        if distance > max_distance_m:
            return None, distance, "INTERSECTION TOO FAR"
        point = ray_origin + distance * ray_direction
        return point, distance, "VALID"

    def contains(self, point, tolerance_m):
        uv = self.coordinates(np.asarray(point).reshape(1, 3))[0]
        return bool(
            self.table_u_min - tolerance_m
            <= uv[0]
            <= self.table_u_max + tolerance_m
            and self.table_v_min - tolerance_m
            <= uv[1]
            <= self.table_v_max + tolerance_m
        )


class SequentialVideoReader:
    def __init__(self, path):
        self.path = str(path)
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.next_frame_index = None

    def read(self, frame_index):
        frame_index = int(frame_index)
        if (
            self.next_frame_index is None
            or frame_index < self.next_frame_index
            or frame_index - self.next_frame_index > 120
        ):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self.next_frame_index = frame_index

        frame = None
        while self.next_frame_index <= frame_index:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read frame {frame_index} from {self.path}"
                )
            self.next_frame_index += 1
        return frame

    def close(self):
        self.capture.release()


class VideoSink:
    def __init__(self, path, size, fps):
        self.backend = "opencv-mp4v"
        self.generator = None
        self.writer = None
        if imageio_ffmpeg is not None:
            self.backend = "ffmpeg-libx264"
            self.generator = imageio_ffmpeg.write_frames(
                str(path),
                size,
                fps=fps,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                output_params=[
                    "-crf",
                    "18",
                    "-preset",
                    "medium",
                    "-movflags",
                    "+faststart",
                ],
            )
            self.generator.send(None)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(path), fourcc, fps, size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not create output video: {path}")

    def write(self, frame_bgr):
        if self.generator is not None:
            self.generator.send(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            self.writer.write(frame_bgr)

    def close(self):
        if self.generator is not None:
            self.generator.close()
        if self.writer is not None:
            self.writer.release()


def require_file(path, description):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def find_calibration(session_dir, explicit_path):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.append(session_dir / "multicam_calibration.npz")
    try:
        metadata = json.loads((session_dir / "metadata.json").read_text())
        session_calibration = metadata.get("session_calibration_file")
        if session_calibration:
            candidates.append(session_dir / session_calibration)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    for candidate in candidates:
        if candidate.expanduser().exists():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "multicam_calibration.npz was not found in the session. "
        "Provide it with --calib."
    )


def find_color_video(session_dir, metadata, camera_id):
    camera = metadata["cameras"][str(camera_id)]
    storage_file = camera.get("color_storage", {}).get("file")
    candidates = []
    if storage_file:
        candidates.append(session_dir / storage_file)
    candidates.extend(
        [
            session_dir / f"cam{camera_id}" / "color.mkv",
            session_dir / f"cam{camera_id}" / "color.mp4",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Color video for cam{camera_id} was not found.")


def timestamp_path(session_dir, metadata, camera_id, stream):
    camera = metadata["cameras"][str(camera_id)]
    key = f"{stream}_timestamp_file"
    configured = camera.get(key)
    if configured:
        candidate = session_dir / configured
    else:
        candidate = (
            session_dir
            / f"cam{camera_id}"
            / f"cam{camera_id}_{stream}_timestamps.csv"
        )
    return require_file(candidate, f"Cam{camera_id} {stream} timestamps")


def load_gaze_rows(session_dir, camera_id, explicit_path):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    camera_dir = session_dir / f"cam{camera_id}"
    candidates.extend(
        [
            camera_dir / f"cam{camera_id}_gaze_model_raw.csv",
            camera_dir / f"cam{camera_id}_gaze_raw.csv",
        ]
    )
    gaze_path = next(
        (candidate.expanduser().resolve() for candidate in candidates if candidate.expanduser().exists()),
        None,
    )
    if gaze_path is None:
        raise FileNotFoundError(
            f"Cam{camera_id} gaze CSV was not found. Run "
            "devel/revised_process/extract_gaze_independent.py first."
        )
    gaze = pd.read_csv(gaze_path)
    required = {
        "frame_idx",
        "hw_timestamp_ms",
        "gaze_x",
        "gaze_y",
        "gaze_z",
        "face_detected",
    }
    missing = sorted(required.difference(gaze.columns))
    if missing:
        raise RuntimeError(f"{gaze_path} is missing columns: {missing}")
    gaze = gaze.sort_values("hw_timestamp_ms").reset_index(drop=True)
    return gaze_path, gaze


def nearest_row_index(sorted_timestamps, timestamp_ms):
    position = int(np.searchsorted(sorted_timestamps, timestamp_ms))
    candidates = np.clip(
        np.asarray([position - 1, position], dtype=np.int64),
        0,
        len(sorted_timestamps) - 1,
    )
    return int(
        candidates[
            int(np.argmin(np.abs(sorted_timestamps[candidates] - timestamp_ms)))
        ]
    )


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9 or not np.all(np.isfinite(vector)):
        return None
    return vector / norm


def camera_to_reference_transform(calibration, camera_id, reference_camera):
    explicit_key = f"T_cam{camera_id}_to_ref"
    if explicit_key in calibration:
        transform = np.asarray(calibration[explicit_key], dtype=np.float64)
        return transform[:3, :3], transform[:3, 3]
    if camera_id == reference_camera:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    rotation_ref_to_camera = np.asarray(
        calibration[f"R_{camera_id}_to_ref"], dtype=np.float64
    )
    translation_ref_to_camera = np.asarray(
        calibration[f"t_{camera_id}_to_ref"], dtype=np.float64
    ).reshape(3)
    rotation_camera_to_ref = rotation_ref_to_camera.T
    translation_camera_to_ref = (
        -rotation_camera_to_ref @ translation_ref_to_camera
    )
    return rotation_camera_to_ref, translation_camera_to_ref


def reference_to_camera_transform(calibration, camera_id, reference_camera):
    rotation_camera_to_ref, translation_camera_to_ref = (
        camera_to_reference_transform(
            calibration, camera_id, reference_camera
        )
    )
    rotation_ref_to_camera = rotation_camera_to_ref.T
    translation_ref_to_camera = (
        -rotation_ref_to_camera @ translation_camera_to_ref
    )
    return rotation_ref_to_camera, translation_ref_to_camera


def camera_vector_to_world(
    calibration, camera_id, reference_camera, vector_camera
):
    rotation_camera_to_ref, _ = camera_to_reference_transform(
        calibration, camera_id, reference_camera
    )
    vector_ref = rotation_camera_to_ref @ vector_camera
    return normalize(RVIZ_ROTATION @ vector_ref)


def world_to_camera(
    calibration, camera_id, reference_camera, points_world
):
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    points_ref = points_world @ RVIZ_ROTATION
    rotation_ref_to_camera, translation_ref_to_camera = (
        reference_to_camera_transform(
            calibration, camera_id, reference_camera
        )
    )
    return (
        points_ref @ rotation_ref_to_camera.T
        + translation_ref_to_camera
    )


def project_world_points(
    calibration, camera_id, reference_camera, points_world
):
    points_camera = world_to_camera(
        calibration, camera_id, reference_camera, points_world
    )
    intrinsic = np.asarray(
        calibration[f"K{camera_id}"], dtype=np.float64
    )
    distortion = np.asarray(
        calibration[f"dist{camera_id}"], dtype=np.float64
    ).reshape(-1)
    projected, _ = cv2.projectPoints(
        points_camera.reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        intrinsic,
        distortion,
    )
    return projected.reshape(-1, 2), points_camera[:, 2]


def eye_origin_from_row(row, person_id):
    eyes = []
    for keypoint in (1, 2):
        point = np.asarray(
            [
                row[f"p{person_id}_kpt{keypoint}_world_x"],
                row[f"p{person_id}_kpt{keypoint}_world_y"],
                row[f"p{person_id}_kpt{keypoint}_world_z"],
            ],
            dtype=np.float64,
        )
        if np.all(np.isfinite(point)):
            eyes.append(point)
    if eyes:
        return np.mean(eyes, axis=0)
    nose = np.asarray(
        [
            row[f"p{person_id}_kpt0_world_x"],
            row[f"p{person_id}_kpt0_world_y"],
            row[f"p{person_id}_kpt0_world_z"],
        ],
        dtype=np.float64,
    )
    return nose if np.all(np.isfinite(nose)) else None


def load_interaction_geometry(path, gaze_camera_id):
    base_columns = [
        "timestamp_ms",
        "dyad_ref_x",
        "dyad_ref_y",
        "dyad_ref_z",
        "p1_gaze_cam_id",
        "p2_gaze_cam_id",
    ]
    for person_id in (1, 2):
        for axis in "xyz":
            base_columns.append(f"p{person_id}_ref_{axis}")
        for keypoint in (0, 1, 2):
            for axis in "xyz":
                base_columns.append(
                    f"p{person_id}_kpt{keypoint}_world_{axis}"
                )
    interaction = pd.read_parquet(path, columns=base_columns)
    if interaction.empty:
        raise RuntimeError(f"Interaction parquet is empty: {path}")
    person_scores = {}
    for person_id in (1, 2):
        camera_values = pd.to_numeric(
            interaction[f"p{person_id}_gaze_cam_id"], errors="coerce"
        )
        person_scores[person_id] = int(
            np.sum(camera_values.to_numpy() == gaze_camera_id)
        )
    person_id = max(person_scores, key=person_scores.get)
    if person_scores[person_id] == 0:
        raise RuntimeError(
            f"No participant in {path} is associated with gaze cam"
            f"{gaze_camera_id}."
        )
    interaction = interaction.sort_values("timestamp_ms").reset_index(drop=True)
    return interaction, person_id


def interaction_center(interaction, person_id):
    for prefix in ("dyad_ref", f"p{person_id}_ref"):
        columns = [f"{prefix}_{axis}" for axis in "xyz"]
        values = interaction[columns].to_numpy(dtype=np.float64)
        finite_rows = np.all(np.isfinite(values), axis=1)
        if np.any(finite_rows):
            return np.median(values[finite_rows], axis=0), prefix
    raise RuntimeError(
        "Could not recover a valid dyad or focal-participant center."
    )


def import_pcl_helpers():
    postprocess_dir = Path(__file__).resolve().parents[1] / "postprocess"
    sys.path.insert(0, str(postprocess_dir))
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "usta_mpl")
    )
    import create_session_pcl

    return create_session_pcl


def synchronized_pcl(
    session_dir,
    metadata,
    calibration,
    timestamp_tables,
    timestamp_ms,
    camera_ids,
    step,
):
    pcl_helpers = import_pcl_helpers()
    reference_camera = int(
        np.asarray(calibration["ref_camera"]).item()
        if "ref_camera" in calibration
        else 1
    )
    all_points = []
    all_colors = []
    diagnostics = []
    for camera_id in camera_ids:
        color_frame, _, color_delta = timestamp_tables[
            (camera_id, "color")
        ].nearest(timestamp_ms)
        depth_frame, _, depth_delta = timestamp_tables[
            (camera_id, "depth")
        ].nearest(timestamp_ms)
        points_camera, colors, _ = pcl_helpers.extract_camera_pcl(
            str(session_dir),
            camera_id,
            metadata,
            calibration,
            frame_idx=color_frame,
            depth_frame_idx=depth_frame,
            step=step,
            min_z=0.25,
            max_z=3.5,
        )
        if len(points_camera) == 0:
            continue
        rotation, translation = camera_to_reference_transform(
            calibration, camera_id, reference_camera
        )
        points_ref = points_camera @ rotation.T + translation
        points_world = points_ref @ RVIZ_ROTATION.T
        finite = np.all(np.isfinite(points_world), axis=1)
        all_points.append(points_world[finite])
        all_colors.append(np.asarray(colors, dtype=np.uint8)[finite])
        diagnostics.append(
            {
                "camera_id": camera_id,
                "point_count": int(np.sum(finite)),
                "color_sync_ms": color_delta,
                "depth_sync_ms": depth_delta,
            }
        )
    if not all_points:
        raise RuntimeError("No point cloud could be extracted from pose cameras.")
    return (
        np.concatenate(all_points, axis=0),
        np.concatenate(all_colors, axis=0),
        diagnostics,
    )


def fit_table_plane(points, colors, dyad_center, random_seed):
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    horizontal_distance = np.linalg.norm(
        points[:, :2] - dyad_center[:2], axis=1
    )
    candidate_mask = (
        np.all(np.isfinite(points), axis=1)
        & (horizontal_distance < 1.0)
        & (points[:, 2] > dyad_center[2] - 0.55)
        & (points[:, 2] < dyad_center[2] + 0.06)
    )
    candidates = points[candidate_mask]
    candidate_colors = colors[candidate_mask]
    if len(candidates) < 500:
        raise RuntimeError(
            "Too few task-space points to estimate the table plane."
        )

    rng = np.random.default_rng(random_seed)
    best_mask = None
    best_score = -np.inf
    expected_height = float(dyad_center[2] - 0.16)
    for _ in range(900):
        sample = candidates[
            rng.choice(len(candidates), size=3, replace=False)
        ]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        normal = normalize(normal)
        if normal is None or abs(normal[2]) < 0.88:
            continue
        if normal[2] < 0:
            normal = -normal
        offset = -float(np.dot(normal, sample[0]))
        center_height = -(
            normal[0] * dyad_center[0]
            + normal[1] * dyad_center[1]
            + offset
        ) / normal[2]
        if not (
            dyad_center[2] - 0.48
            <= center_height
            <= dyad_center[2] + 0.02
        ):
            continue
        distances = np.abs(candidates @ normal + offset)
        inliers = distances < 0.012
        inlier_count = int(np.sum(inliers))
        height_penalty = 2500.0 * abs(center_height - expected_height)
        score = inlier_count - height_penalty
        if score > best_score:
            best_score = score
            best_mask = inliers

    if best_mask is None or int(np.sum(best_mask)) < 300:
        raise RuntimeError(
            "Robust table-plane estimation failed. Check depth and calibration."
        )

    inlier_points = candidates[best_mask]
    origin = np.mean(inlier_points, axis=0)
    _, _, right_vectors = np.linalg.svd(
        inlier_points - origin, full_matrices=False
    )
    normal = normalize(right_vectors[-1])
    if normal[2] < 0:
        normal = -normal
    residuals = np.abs((candidates - origin) @ normal)
    refined_mask = residuals < 0.014
    inlier_points = candidates[refined_mask]
    origin = np.mean(inlier_points, axis=0)
    _, _, right_vectors = np.linalg.svd(
        inlier_points - origin, full_matrices=False
    )
    normal = normalize(right_vectors[-1])
    if normal[2] < 0:
        normal = -normal

    axis_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    axis_u = normalize(axis_u - normal * np.dot(axis_u, normal))
    if axis_u is None:
        axis_u = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis_u = normalize(axis_u - normal * np.dot(axis_u, normal))
    axis_v = normalize(np.cross(normal, axis_u))

    relative = candidates - origin
    all_uv = np.column_stack([relative @ axis_u, relative @ axis_v])
    texture_mask = (
        np.abs(relative @ normal) < 0.04
    )
    texture_points = candidates[texture_mask]
    texture_colors = candidate_colors[texture_mask]
    texture_uv = all_uv[texture_mask]
    if len(texture_points) < 300:
        raise RuntimeError("Too few tabletop texture points were recovered.")

    table_u_min, table_u_max = np.quantile(texture_uv[:, 0], [0.01, 0.99])
    table_v_min, table_v_max = np.quantile(texture_uv[:, 1], [0.01, 0.99])
    residual_mm = float(
        np.median(np.abs((inlier_points - origin) @ normal)) * 1000.0
    )
    return TablePlane(
        origin=origin,
        normal=normal,
        axis_u=axis_u,
        axis_v=axis_v,
        table_u_min=float(table_u_min),
        table_u_max=float(table_u_max),
        table_v_min=float(table_v_min),
        table_v_max=float(table_v_max),
        texture_points=texture_points,
        texture_colors=texture_colors,
        inlier_count=len(inlier_points),
        residual_mm=residual_mm,
    )


def safe_int_point(point):
    if point is None or not np.all(np.isfinite(point)):
        return None
    return tuple(int(round(value)) for value in point)


def draw_marker(frame, point, color, label, radius=15):
    pixel = safe_int_point(point)
    if pixel is None:
        return
    height, width = frame.shape[:2]
    if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
        return
    cv2.circle(frame, pixel, radius + 5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(frame, pixel, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, pixel, radius, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (pixel[0] + radius + 7, pixel[1] - radius - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (pixel[0] + radius + 7, pixel[1] - radius - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_projected_trail(
    frame,
    calibration,
    camera_id,
    reference_camera,
    trail_points,
):
    if len(trail_points) < 2:
        return
    projected, depths = project_world_points(
        calibration,
        camera_id,
        reference_camera,
        np.asarray(trail_points),
    )
    height, width = frame.shape[:2]
    segments = []
    current = []
    for point, depth in zip(projected, depths):
        inside = (
            depth > 0
            and 0 <= point[0] < width
            and 0 <= point[1] < height
        )
        if inside:
            current.append(safe_int_point(point))
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    for segment in segments:
        if len(segment) >= 2:
            cv2.polylines(
                frame,
                [np.asarray(segment, dtype=np.int32)],
                False,
                (45, 45, 220),
                3,
                cv2.LINE_AA,
            )


def render_camera_panel(
    frame,
    title,
    subtitle,
    output_size,
    calibration,
    camera_id,
    reference_camera,
    landing,
    trail_points,
    on_table,
):
    frame = frame.copy()
    if landing is not None:
        draw_projected_trail(
            frame,
            calibration,
            camera_id,
            reference_camera,
            trail_points,
        )
        projected, depths = project_world_points(
            calibration,
            camera_id,
            reference_camera,
            np.asarray(landing).reshape(1, 3),
        )
        if depths[0] > 0:
            color = (40, 40, 230) if on_table else (0, 140, 255)
            draw_marker(
                frame,
                projected[0],
                color,
                "GAZE LANDING" if on_table else "OFF TABLE",
            )
    return panel_with_header(frame, title, subtitle, output_size)


def render_gaze_panel(
    frame,
    title,
    subtitle,
    output_size,
    calibration,
    camera_id,
    reference_camera,
    eye_world,
    gaze_camera,
):
    frame = frame.copy()
    if eye_world is not None and gaze_camera is not None:
        eye_camera = world_to_camera(
            calibration,
            camera_id,
            reference_camera,
            np.asarray(eye_world).reshape(1, 3),
        )[0]
        endpoint_camera = eye_camera + 0.35 * gaze_camera
        intrinsic = np.asarray(
            calibration[f"K{camera_id}"], dtype=np.float64
        )
        distortion = np.asarray(
            calibration[f"dist{camera_id}"], dtype=np.float64
        ).reshape(-1)
        if eye_camera[2] > 0 and endpoint_camera[2] > 0:
            projected, _ = cv2.projectPoints(
                np.asarray([eye_camera, endpoint_camera]).reshape(-1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                intrinsic,
                distortion,
            )
            start, end = projected.reshape(-1, 2)
            start_pixel = safe_int_point(start)
            end_pixel = safe_int_point(end)
            if start_pixel is not None and end_pixel is not None:
                cv2.arrowedLine(
                    frame,
                    start_pixel,
                    end_pixel,
                    (0, 220, 255),
                    5,
                    cv2.LINE_AA,
                    tipLength=0.18,
                )
    return panel_with_header(frame, title, subtitle, output_size)


def letterbox(image, width, height, color=(18, 18, 18)):
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), color, dtype=np.uint8)
    offset_x = (width - resized_width) // 2
    offset_y = (height - resized_height) // 2
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas


def panel_with_header(image, title, subtitle, output_size):
    width, height = output_size
    header_height = 76
    body = letterbox(image, width, height - header_height)
    panel = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    panel[header_height:] = body
    cv2.putText(
        panel,
        title,
        (22, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        subtitle,
        (22, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return panel


def build_table_map_background(table, width, height):
    canvas = np.full((height, width, 3), (242, 241, 237), dtype=np.uint8)
    header_height = 76
    margin = 42
    uv = table.coordinates(table.texture_points)
    display_u_min = table.table_u_min - 0.18
    display_u_max = table.table_u_max + 0.18
    display_v_min = table.table_v_min - 0.18
    display_v_max = table.table_v_max + 0.18
    usable_width = width - 2 * margin
    usable_height = height - header_height - 2 * margin
    scale = min(
        usable_width / max(display_u_max - display_u_min, 1e-6),
        usable_height / max(display_v_max - display_v_min, 1e-6),
    )
    center_u = 0.5 * (display_u_min + display_u_max)
    center_v = 0.5 * (display_v_min + display_v_max)

    def to_pixel(values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        pixel_x = width / 2 + (values[:, 0] - center_u) * scale
        pixel_y = (
            header_height
            + (height - header_height) / 2
            - (values[:, 1] - center_v) * scale
        )
        return np.column_stack([pixel_x, pixel_y])

    pixels = np.rint(to_pixel(uv)).astype(np.int32)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= header_height)
        & (pixels[:, 1] < height)
    )
    pixels = pixels[inside]
    colors_bgr = table.texture_colors[inside][:, ::-1]
    if len(pixels) > 120000:
        selection = np.linspace(
            0, len(pixels) - 1, 120000, dtype=np.int64
        )
        pixels = pixels[selection]
        colors_bgr = colors_bgr[selection]
    for offset_x, offset_y in (
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ):
        shifted_x = np.clip(pixels[:, 0] + offset_x, 0, width - 1)
        shifted_y = np.clip(
            pixels[:, 1] + offset_y, header_height, height - 1
        )
        canvas[shifted_y, shifted_x] = colors_bgr

    corners_uv = np.asarray(
        [
            [table.table_u_min, table.table_v_min],
            [table.table_u_max, table.table_v_min],
            [table.table_u_max, table.table_v_max],
            [table.table_u_min, table.table_v_max],
        ]
    )
    corners = np.rint(to_pixel(corners_uv)).astype(np.int32)
    cv2.polylines(
        canvas,
        [corners],
        True,
        (75, 75, 75),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "MEASURED TABLE TOP VIEW",
        (22, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"raw RGB-D plane | {table.inlier_count:,} inliers | "
            f"median residual {table.residual_mm:.1f} mm"
        ),
        (22, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    return canvas, to_pixel


def render_table_panel(
    background,
    to_pixel,
    table,
    landing,
    trail_points,
    on_table,
    status,
):
    panel = background.copy()
    if trail_points:
        uv = table.coordinates(np.asarray(trail_points))
        pixels = np.rint(to_pixel(uv)).astype(np.int32)
        if len(pixels) >= 2:
            cv2.polylines(
                panel,
                [pixels],
                False,
                (45, 45, 220),
                4,
                cv2.LINE_AA,
            )
    if landing is not None:
        pixel = to_pixel(table.coordinates(np.asarray(landing).reshape(1, 3)))[0]
        color = (40, 40, 230) if on_table else (0, 140, 255)
        draw_marker(
            panel,
            pixel,
            color,
            "GAZE" if on_table else "OFF TABLE",
            radius=17,
        )
    status_color = (40, 120, 40) if on_table else (0, 90, 220)
    cv2.putText(
        panel,
        status,
        (22, panel.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        status_color,
        2,
        cv2.LINE_AA,
    )
    return panel


def choose_clip_bounds(
    gaze_timestamps,
    interaction_timestamps,
    timestamp_tables,
    start_seconds,
    duration_seconds,
):
    minimums = [
        float(np.min(gaze_timestamps)),
        float(np.min(interaction_timestamps)),
    ]
    maximums = [
        float(np.max(gaze_timestamps)),
        float(np.max(interaction_timestamps)),
    ]
    for table in timestamp_tables.values():
        minimums.append(float(table.timestamps_ms[0]))
        maximums.append(float(table.timestamps_ms[-1]))
    available_start = max(minimums)
    available_end = min(maximums)
    clip_start = available_start + start_seconds * 1000.0
    clip_end = clip_start + duration_seconds * 1000.0
    if clip_start >= available_end:
        raise RuntimeError("Requested --start-s is beyond synchronized data.")
    if clip_end > available_end:
        available_duration = (available_end - clip_start) / 1000.0
        raise RuntimeError(
            f"Only {available_duration:.2f}s remain after --start-s; "
            f"{duration_seconds:.2f}s was requested."
        )
    return clip_start, clip_end


def format_vector(vector):
    if vector is None:
        return "raw vector unavailable"
    return (
        f"raw camera vector "
        f"[{vector[0]:+.3f}, {vector[1]:+.3f}, {vector[2]:+.3f}]"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project raw Cam3 gaze onto a measured RGB-D table plane and "
            "render a visual validation video."
        )
    )
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--camera", type=int, default=3)
    parser.add_argument("--pose-cameras", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument("--calib", default=None)
    parser.add_argument("--gaze-csv", default=None)
    parser.add_argument("--interaction-parquet", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pcl-step", type=int, default=4)
    parser.add_argument("--max-ray-distance-m", type=float, default=3.0)
    parser.add_argument("--table-tolerance-m", type=float, default=0.04)
    parser.add_argument("--max-sync-error-ms", type=float, default=120.0)
    parser.add_argument("--trail-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.duration_s <= 0 or args.output_fps <= 0:
        raise ValueError("--duration-s and --output-fps must be positive.")

    session_dir = require_file(args.session_dir, "Session directory")
    if not session_dir.is_dir():
        raise NotADirectoryError(session_dir)
    metadata_path = require_file(
        session_dir / "metadata.json", "Session metadata"
    )
    metadata = json.loads(metadata_path.read_text())
    calibration_path = find_calibration(session_dir, args.calib)
    calibration = np.load(calibration_path)
    reference_camera = int(
        np.asarray(calibration["ref_camera"]).item()
        if "ref_camera" in calibration
        else 1
    )

    interaction_path = (
        Path(args.interaction_parquet).expanduser().resolve()
        if args.interaction_parquet
        else session_dir / "session_interaction_dataset.parquet"
    )
    interaction_path = require_file(
        interaction_path, "Interaction parquet"
    )
    gaze_path, gaze = load_gaze_rows(
        session_dir, args.camera, args.gaze_csv
    )
    interaction, person_id = load_interaction_geometry(
        interaction_path, args.camera
    )

    all_camera_ids = sorted(set(args.pose_cameras + [args.camera]))
    timestamp_tables = {}
    color_paths = {}
    for camera_id in all_camera_ids:
        color_paths[camera_id] = find_color_video(
            session_dir, metadata, camera_id
        )
        timestamp_tables[(camera_id, "color")] = TimestampTable.from_csv(
            timestamp_path(
                session_dir, metadata, camera_id, "color"
            )
        )
    for camera_id in args.pose_cameras:
        timestamp_tables[(camera_id, "depth")] = TimestampTable.from_csv(
            timestamp_path(
                session_dir, metadata, camera_id, "depth"
            )
        )

    gaze_timestamps = pd.to_numeric(
        gaze["hw_timestamp_ms"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    interaction_timestamps = pd.to_numeric(
        interaction["timestamp_ms"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    clip_start, clip_end = choose_clip_bounds(
        gaze_timestamps,
        interaction_timestamps,
        timestamp_tables,
        args.start_s,
        args.duration_s,
    )
    midpoint_ms = 0.5 * (clip_start + clip_end)
    dyad_center, center_source = interaction_center(
        interaction, person_id
    )

    points, colors, pcl_diagnostics = synchronized_pcl(
        session_dir,
        metadata,
        calibration,
        timestamp_tables,
        midpoint_ms,
        args.pose_cameras,
        args.pcl_step,
    )
    table = fit_table_plane(points, colors, dyad_center, args.seed)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else session_dir
        / "gaze_validation"
        / f"cam{args.camera}_table_projection_{args.duration_s:g}s.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")

    output_width = 1920
    output_height = 1080
    panel_size = (output_width // 2, output_height // 2)
    table_panel_size = (
        (output_width, output_height // 2)
        if len(args.pose_cameras) == 1
        else panel_size
    )
    table_background, table_to_pixel = build_table_map_background(
        table, *table_panel_size
    )
    readers = {
        camera_id: SequentialVideoReader(color_paths[camera_id])
        for camera_id in all_camera_ids
    }
    writer = VideoSink(
        output_path,
        (output_width, output_height),
        args.output_fps,
    )

    total_frames = int(math.ceil(args.duration_s * args.output_fps))
    trail_length = max(1, int(round(args.trail_s * args.output_fps)))
    trail_points = []
    output_rows = []
    try:
        for output_frame in range(total_frames):
            timestamp_ms = (
                clip_start + output_frame * 1000.0 / args.output_fps
            )
            gaze_index = nearest_row_index(gaze_timestamps, timestamp_ms)
            gaze_row = gaze.iloc[gaze_index]
            pose_index = nearest_row_index(
                interaction_timestamps, timestamp_ms
            )
            pose_row = interaction.iloc[pose_index]
            pose_sync_ms = float(
                abs(interaction_timestamps[pose_index] - timestamp_ms)
            )
            gaze_sync_ms = float(
                abs(gaze_timestamps[gaze_index] - timestamp_ms)
            )

            frames = {}
            frame_sync = {}
            for camera_id in all_camera_ids:
                frame_index, _, sync_ms = timestamp_tables[
                    (camera_id, "color")
                ].nearest(timestamp_ms)
                frames[camera_id] = readers[camera_id].read(frame_index)
                frame_sync[camera_id] = sync_ms

            face_detected = bool(
                int(float(gaze_row.get("face_detected", 0) or 0))
            )
            gaze_camera = normalize(
                np.asarray(
                    [
                        gaze_row["gaze_x"],
                        gaze_row["gaze_y"],
                        gaze_row["gaze_z"],
                    ],
                    dtype=np.float64,
                )
            )
            eye_world = eye_origin_from_row(pose_row, person_id)
            gaze_world = (
                camera_vector_to_world(
                    calibration,
                    args.camera,
                    reference_camera,
                    gaze_camera,
                )
                if gaze_camera is not None
                else None
            )

            landing = None
            ray_distance = None
            status = "NO FACE"
            if (
                face_detected
                and gaze_world is not None
                and eye_world is not None
                and pose_sync_ms <= args.max_sync_error_ms
                and gaze_sync_ms <= args.max_sync_error_ms
            ):
                landing, ray_distance, status = table.intersect(
                    eye_world,
                    gaze_world,
                    args.max_ray_distance_m,
                )
            elif pose_sync_ms > args.max_sync_error_ms:
                status = "POSE SYNC TOO LARGE"
            elif gaze_sync_ms > args.max_sync_error_ms:
                status = "GAZE SYNC TOO LARGE"
            elif eye_world is None:
                status = "EYE ORIGIN UNAVAILABLE"
            elif gaze_world is None:
                status = "GAZE VECTOR UNAVAILABLE"

            on_table = bool(
                landing is not None
                and table.contains(landing, args.table_tolerance_m)
            )
            if landing is not None:
                status = "ON TABLE" if on_table else "OFF TABLE"
                trail_points.append(landing)
                trail_points = trail_points[-trail_length:]
            else:
                trail_points = []

            cam3_panel = render_gaze_panel(
                frames[args.camera],
                f"CAM{args.camera} RAW GAZE",
                (
                    f"participant P{person_id} | "
                    f"{format_vector(gaze_camera)} | no smoothing"
                ),
                panel_size,
                calibration,
                args.camera,
                reference_camera,
                eye_world,
                gaze_camera,
            )
            pose_panels = []
            for camera_id in args.pose_cameras[:2]:
                pose_panels.append(
                    render_camera_panel(
                        frames[camera_id],
                        f"CAM{camera_id} TABLE PROJECTION",
                        (
                            f"{status} | red dot is the calibrated "
                            "ray-plane intersection"
                        ),
                        panel_size,
                        calibration,
                        camera_id,
                        reference_camera,
                        landing,
                        trail_points,
                        on_table,
                    )
                )
            table_panel = render_table_panel(
                table_background,
                table_to_pixel,
                table,
                landing,
                trail_points,
                on_table,
                status,
            )
            if len(pose_panels) == 1:
                composed = np.vstack(
                    [
                        np.hstack([cam3_panel, pose_panels[0]]),
                        table_panel,
                    ]
                )
            else:
                while len(pose_panels) < 2:
                    pose_panels.append(
                        np.full(
                            (panel_size[1], panel_size[0], 3),
                            30,
                            dtype=np.uint8,
                        )
                    )
                composed = np.vstack(
                    [
                        np.hstack([cam3_panel, pose_panels[0]]),
                        np.hstack([pose_panels[1], table_panel]),
                    ]
                )
            elapsed = (timestamp_ms - clip_start) / 1000.0
            cv2.putText(
                composed,
                f"{elapsed:05.1f}s / {args.duration_s:.1f}s",
                (output_width - 260, output_height - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (30, 30, 30),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                composed,
                f"{elapsed:05.1f}s / {args.duration_s:.1f}s",
                (output_width - 260, output_height - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            writer.write(composed)

            landing_values = (
                np.asarray(landing, dtype=np.float64)
                if landing is not None
                else np.full(3, np.nan)
            )
            eye_values = (
                np.asarray(eye_world, dtype=np.float64)
                if eye_world is not None
                else np.full(3, np.nan)
            )
            world_values = (
                np.asarray(gaze_world, dtype=np.float64)
                if gaze_world is not None
                else np.full(3, np.nan)
            )
            camera_values = (
                np.asarray(gaze_camera, dtype=np.float64)
                if gaze_camera is not None
                else np.full(3, np.nan)
            )
            output_rows.append(
                {
                    "output_frame": output_frame,
                    "elapsed_s": elapsed,
                    "timestamp_ms": timestamp_ms,
                    "gaze_source_frame": int(gaze_row["frame_idx"]),
                    "participant_id": person_id,
                    "face_detected": int(face_detected),
                    "gaze_sync_ms": gaze_sync_ms,
                    "pose_sync_ms": pose_sync_ms,
                    "cam1_sync_ms": frame_sync.get(1, np.nan),
                    "cam2_sync_ms": frame_sync.get(2, np.nan),
                    "cam3_sync_ms": frame_sync.get(3, np.nan),
                    "gaze_cam_x": camera_values[0],
                    "gaze_cam_y": camera_values[1],
                    "gaze_cam_z": camera_values[2],
                    "gaze_world_x": world_values[0],
                    "gaze_world_y": world_values[1],
                    "gaze_world_z": world_values[2],
                    "eye_world_x": eye_values[0],
                    "eye_world_y": eye_values[1],
                    "eye_world_z": eye_values[2],
                    "landing_world_x": landing_values[0],
                    "landing_world_y": landing_values[1],
                    "landing_world_z": landing_values[2],
                    "ray_distance_m": (
                        ray_distance
                        if ray_distance is not None
                        else np.nan
                    ),
                    "on_measured_table": int(on_table),
                    "status": status,
                }
            )
    finally:
        writer.close()
        for reader in readers.values():
            reader.close()

    pd.DataFrame(output_rows).to_csv(csv_path, index=False)
    diagnostics = {
        "session_dir": str(session_dir),
        "calibration": str(calibration_path),
        "interaction_parquet": str(interaction_path),
        "gaze_csv": str(gaze_path),
        "gaze_camera": args.camera,
        "participant_id": person_id,
        "table_fit_center_source": center_source,
        "clip_start_timestamp_ms": clip_start,
        "clip_end_timestamp_ms": clip_end,
        "table_origin_world": table.origin.tolist(),
        "table_normal_world": table.normal.tolist(),
        "table_inlier_count": table.inlier_count,
        "table_median_residual_mm": table.residual_mm,
        "pcl_sources": pcl_diagnostics,
        "video_encoder": writer.backend,
        "video": str(output_path),
        "csv": str(csv_path),
    }
    print(json.dumps(diagnostics, indent=2))
    print(f"\nSaved visual validation video: {output_path}")
    print(f"Saved frame-by-frame audit CSV: {csv_path}")
    print(
        "The gaze vector is never clamped, corrected, or smoothed. "
        "OFF TABLE means the raw calibrated ray missed the measured extent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
