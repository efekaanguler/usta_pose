#!/usr/bin/env python3
"""
RViz visualizer for revised person-based session_ml_dataset.parquet files.

The revised pipeline stores each person's root in the shared reference frame
and each keypoint as a root-relative offset. This visualizer reconstructs every
keypoint directly as:

    keypoint_world = root_world + keypoint_relative

It intentionally does not center, normalize, align, scale, or otherwise
transform either person independently. A fixed display-only rotation is applied
only when converting reconstructed coordinates into RViz marker points.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import rclpy
    from rclpy.node import Node
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point, TransformStamped
    from tf2_ros import StaticTransformBroadcaster
    from std_msgs.msg import ColorRGBA
except ImportError:
    rclpy = None
    StaticTransformBroadcaster = None

    class Node:  # pragma: no cover - only used when ROS is unavailable.
        pass

    class Point:  # pragma: no cover
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class ColorRGBA:  # pragma: no cover
        def __init__(self, r=0.0, g=0.0, b=0.0, a=0.0):
            self.r = r
            self.g = g
            self.b = b
            self.a = a

    class Marker:  # pragma: no cover
        ARROW = 0
        SPHERE = 2
        LINE_LIST = 5
        SPHERE_LIST = 7
        ADD = 0
        DELETE = 2

        def __init__(self):
            self.header = SimpleNamespace(frame_id="", stamp=None)
            self.ns = ""
            self.id = 0
            self.type = 0
            self.action = 0
            self.lifetime = SimpleNamespace(sec=0, nanosec=0)
            self.pose = SimpleNamespace(orientation=SimpleNamespace(w=0.0))
            self.scale = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.color = None
            self.points = []

    class MarkerArray:  # pragma: no cover
        def __init__(self):
            self.markers = []

    class TransformStamped:  # pragma: no cover
        def __init__(self):
            self.header = SimpleNamespace(frame_id="", stamp=None)
            self.child_frame_id = ""
            self.transform = SimpleNamespace(
                translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )


NUM_KEYPOINTS = 133
WORLD_FRAME = "world"
RVIZ_DISPLAY_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)

# Body keypoint connections (kpt 0-16)
BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

LEFT_HAND = [
    (91, 92), (92, 93), (93, 94), (94, 95),
    (91, 96), (96, 97), (97, 98), (98, 99),
    (91, 100), (100, 101), (101, 102), (102, 103),
    (91, 104), (104, 105), (105, 106), (106, 107),
    (91, 108), (108, 109), (109, 110), (110, 111),
]

RIGHT_HAND = [
    (112, 113), (113, 114), (114, 115), (115, 116),
    (112, 117), (117, 118), (118, 119), (119, 120),
    (112, 121), (121, 122), (122, 123), (123, 124),
    (112, 125), (125, 126), (126, 127), (127, 128),
    (112, 129), (129, 130), (130, 131), (131, 132),
]

WRIST_TO_HAND = [
    (9, 91),
    (10, 112),
]

ALL_SKELETON = BODY_SKELETON + LEFT_HAND + RIGHT_HAND + WRIST_TO_HAND

PERSON_COLORS = {
    1: {
        "joint": ColorRGBA(r=1.0, g=0.3, b=0.3, a=1.0),
        "bone": ColorRGBA(r=1.0, g=0.5, b=0.5, a=0.8),
        "root": ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0),
        "gaze": ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    },
    2: {
        "joint": ColorRGBA(r=0.3, g=1.0, b=0.3, a=1.0),
        "bone": ColorRGBA(r=0.5, g=1.0, b=0.5, a=0.8),
        "root": ColorRGBA(r=0.1, g=1.0, b=0.1, a=1.0),
        "gaze": ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
    },
}


def finite_vector(values):
    values = np.asarray(values, dtype=np.float64)
    return values.shape == (3,) and np.all(np.isfinite(values))


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return False
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def row_bool(row, column, default=False):
    if column not in row:
        return default
    value = row[column]
    if is_missing(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def get_root(row, person_id):
    prefix = f"p{person_id}"
    if not row_bool(row, f"{prefix}_root_valid"):
        return None
    root = np.array([
        row.get(f"{prefix}_root_x", np.nan),
        row.get(f"{prefix}_root_y", np.nan),
        row.get(f"{prefix}_root_z", np.nan),
    ], dtype=np.float64)
    if not finite_vector(root):
        return None
    return root


def reconstruct_keypoint(row, person_id, keypoint_idx):
    root = get_root(row, person_id)
    if root is None:
        return None

    prefix = f"p{person_id}_kpt{keypoint_idx}_rel"
    rel = np.array([
        row.get(f"{prefix}_x", np.nan),
        row.get(f"{prefix}_y", np.nan),
        row.get(f"{prefix}_z", np.nan),
    ], dtype=np.float64)
    if not finite_vector(rel):
        return None
    return root + rel


def reconstruct_keypoints(row, person_id, num_keypoints=NUM_KEYPOINTS):
    return [
        reconstruct_keypoint(row, person_id, keypoint_idx)
        for keypoint_idx in range(num_keypoints)
    ]


def to_point(values):
    return Point(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def rotate_for_rviz(point):
    return RVIZ_DISPLAY_ROTATION @ np.asarray(point, dtype=np.float64)


def to_rviz_point(values):
    return to_point(rotate_for_rviz(values))


def gaze_is_available(row, person_id):
    prefix = f"p{person_id}"
    observed_col = f"{prefix}_gaze_observed"
    interpolated_col = f"{prefix}_gaze_interpolated"
    has_provenance = observed_col in row or interpolated_col in row
    if has_provenance and not (
        row_bool(row, observed_col) or row_bool(row, interpolated_col)
    ):
        return False

    vector = np.array([
        row.get(f"{prefix}_gaze_dir_x", np.nan),
        row.get(f"{prefix}_gaze_dir_y", np.nan),
        row.get(f"{prefix}_gaze_dir_z", np.nan),
    ], dtype=np.float64)
    return finite_vector(vector) and np.linalg.norm(vector) > 0.0


def normalized_gaze_vector(row, person_id):
    if not gaze_is_available(row, person_id):
        return None
    prefix = f"p{person_id}"
    vector = np.array([
        row[f"{prefix}_gaze_dir_x"],
        row[f"{prefix}_gaze_dir_y"],
        row[f"{prefix}_gaze_dir_z"],
    ], dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return vector / norm


def gaze_origin(row, person_id, keypoints=None):
    root = get_root(row, person_id)
    if root is None:
        return None
    if keypoints is None:
        keypoints = reconstruct_keypoints(row, person_id)

    if len(keypoints) > 2 and keypoints[1] is not None and keypoints[2] is not None:
        return (keypoints[1] + keypoints[2]) / 2.0
    if keypoints and keypoints[0] is not None:
        return keypoints[0]
    return root


def create_delete_markers(person_id, stamp=None):
    markers = []
    for namespace, marker_id in (
        (f"person{person_id}_joints", 0),
        (f"person{person_id}_bones", 1),
        (f"person{person_id}_root", 2),
        (f"person{person_id}_gaze", 3),
    ):
        marker = Marker()
        marker.header.frame_id = WORLD_FRAME
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.DELETE
        markers.append(marker)
    return markers


def matrix_to_quaternion(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(rotation)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    return [float(qx), float(qy), float(qz), float(qw)]


def resolve_parquet_path(parquet_path=None, session_dir=None):
    if parquet_path:
        return parquet_path
    if session_dir:
        return os.path.join(session_dir, "session_ml_dataset.parquet")
    raise ValueError("Either --parquet or --session-dir must be provided.")


def validate_revised_schema(df):
    columns = set(df.columns)
    if any(column.startswith("cam") for column in columns):
        raise ValueError(
            "This visualizer expects revised p1_/p2_ Parquet columns, not camX_ pose columns."
        )

    missing = []
    for person_id in (1, 2):
        prefix = f"p{person_id}"
        for column in (
            f"{prefix}_root_x",
            f"{prefix}_root_y",
            f"{prefix}_root_z",
            f"{prefix}_root_valid",
        ):
            if column not in columns:
                missing.append(column)

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required revised schema columns: {joined}")


class VisualizerNode(Node):
    def __init__(self, parquet_path, calib_path=None, fps=30.0, gaze_length=0.4):
        if rclpy is None:
            raise RuntimeError("ROS2 Python packages are required to run the RViz visualizer.")
        if pd is None:
            raise RuntimeError("pandas is required to load session_ml_dataset.parquet.")
        if fps <= 0.0:
            raise ValueError("fps must be positive.")

        super().__init__("revised_session_visualizer")
        self.parquet_path = parquet_path
        self.calib_path = calib_path
        self.fps = fps
        self.gaze_length = gaze_length

        self.marker_pub = self.create_publisher(MarkerArray, "visualization_marker_array", 10)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        self.df = pd.read_parquet(parquet_path)
        validate_revised_schema(self.df)
        self.rows = self.df.to_dict("records")
        self.get_logger().info(f"Loaded {len(self.rows)} frames from {parquet_path}")

        if calib_path:
            self.publish_static_tfs(calib_path)

        self.frame_idx = 0
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

    def publish_static_tfs(self, calib_path):
        """Publish stored ref -> camera transforms for RViz display only."""
        try:
            calib = np.load(calib_path)
            tfs = []
            num_cameras = int(np.asarray(calib["num_cameras"]).item()) if "num_cameras" in calib else 4

            for cam_id in range(1, num_cameras + 1):
                r_key = f"R_{cam_id}_to_ref"
                t_key = f"t_{cam_id}_to_ref"
                if r_key not in calib or t_key not in calib:
                    continue

                rotation_ref_to_cam = np.asarray(calib[r_key], dtype=np.float64)
                translation_ref_to_cam = np.asarray(calib[t_key], dtype=np.float64).reshape(3)
                if rotation_ref_to_cam.shape != (3, 3) or not finite_vector(translation_ref_to_cam):
                    self.get_logger().warning(f"Skipping malformed calibration for cam{cam_id}")
                    continue

                quaternion = matrix_to_quaternion(rotation_ref_to_cam)
                tf_msg = TransformStamped()
                tf_msg.header.stamp = self.get_clock().now().to_msg()
                tf_msg.header.frame_id = WORLD_FRAME
                tf_msg.child_frame_id = f"cam{cam_id}"
                tf_msg.transform.translation.x = float(translation_ref_to_cam[0])
                tf_msg.transform.translation.y = float(translation_ref_to_cam[1])
                tf_msg.transform.translation.z = float(translation_ref_to_cam[2])
                tf_msg.transform.rotation.x = quaternion[0]
                tf_msg.transform.rotation.y = quaternion[1]
                tf_msg.transform.rotation.z = quaternion[2]
                tf_msg.transform.rotation.w = quaternion[3]
                tfs.append(tf_msg)

            if tfs:
                self.tf_static_broadcaster.sendTransform(tfs)
                self.get_logger().info(f"Published {len(tfs)} static camera transforms")
            else:
                self.get_logger().warning("No camera transforms found in calibration file")
        except Exception as exc:
            self.get_logger().error(f"Failed to load calibration TFs: {exc}")

    def timer_callback(self):
        if self.frame_idx >= len(self.rows):
            self.get_logger().info("Session playback finished. Restarting.")
            self.frame_idx = 0
            return

        row = self.rows[self.frame_idx]
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for person_id in (1, 2):
            root = get_root(row, person_id)
            if root is None:
                markers.markers.extend(create_delete_markers(person_id, stamp))
                continue

            keypoints = reconstruct_keypoints(row, person_id)
            markers.markers.extend(self.create_person_markers(person_id, root, keypoints, stamp))
            gaze_marker = self.create_gaze_marker(row, person_id, keypoints, stamp)
            if gaze_marker is not None:
                markers.markers.append(gaze_marker)
            else:
                markers.markers.extend(self.create_gaze_delete_marker(person_id, stamp))

        self.marker_pub.publish(markers)
        self.frame_idx += 1

    def create_person_markers(self, person_id, root, keypoints, stamp):
        markers = []
        colors = PERSON_COLORS[person_id]
        lifetime_ns = int(1e9 / self.fps * 2)

        joint_marker = Marker()
        joint_marker.header.frame_id = WORLD_FRAME
        joint_marker.header.stamp = stamp
        joint_marker.ns = f"person{person_id}_joints"
        joint_marker.id = 0
        joint_marker.type = Marker.SPHERE_LIST
        joint_marker.action = Marker.ADD
        joint_marker.lifetime.nanosec = lifetime_ns
        joint_marker.pose.orientation.w = 1.0
        joint_marker.scale.x = 0.025
        joint_marker.scale.y = 0.025
        joint_marker.scale.z = 0.025
        joint_marker.color = colors["joint"]
        joint_marker.points = [to_rviz_point(point) for point in keypoints if point is not None]
        if joint_marker.points:
            markers.append(joint_marker)

        bone_marker = Marker()
        bone_marker.header.frame_id = WORLD_FRAME
        bone_marker.header.stamp = stamp
        bone_marker.ns = f"person{person_id}_bones"
        bone_marker.id = 1
        bone_marker.type = Marker.LINE_LIST
        bone_marker.action = Marker.ADD
        bone_marker.lifetime.nanosec = lifetime_ns
        bone_marker.pose.orientation.w = 1.0
        bone_marker.scale.x = 0.008
        bone_marker.color = colors["bone"]

        for start_idx, end_idx in ALL_SKELETON:
            start = keypoints[start_idx]
            end = keypoints[end_idx]
            if start is None or end is None:
                continue
            bone_marker.points.append(to_rviz_point(start))
            bone_marker.points.append(to_rviz_point(end))

        if bone_marker.points:
            markers.append(bone_marker)

        root_marker = Marker()
        root_marker.header.frame_id = WORLD_FRAME
        root_marker.header.stamp = stamp
        root_marker.ns = f"person{person_id}_root"
        root_marker.id = 2
        root_marker.type = Marker.SPHERE
        root_marker.action = Marker.ADD
        root_marker.lifetime.nanosec = lifetime_ns
        root_marker.pose.orientation.w = 1.0
        root_marker.pose.position = to_rviz_point(root)
        root_marker.scale.x = 0.06
        root_marker.scale.y = 0.06
        root_marker.scale.z = 0.06
        root_marker.color = colors["root"]
        markers.append(root_marker)

        return markers

    def create_gaze_marker(self, row, person_id, keypoints, stamp):
        vector = normalized_gaze_vector(row, person_id)
        if vector is None:
            return None

        origin = gaze_origin(row, person_id, keypoints)
        if origin is None:
            return None

        end = origin + vector * self.gaze_length
        marker = Marker()
        marker.header.frame_id = WORLD_FRAME
        marker.header.stamp = stamp
        marker.ns = f"person{person_id}_gaze"
        marker.id = 3
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.lifetime.nanosec = int(1e9 / self.fps * 2)
        marker.scale.x = 0.015
        marker.scale.y = 0.03
        marker.scale.z = 0.05
        marker.color = PERSON_COLORS[person_id]["gaze"]
        marker.points = [to_rviz_point(origin), to_rviz_point(end)]
        return marker

    def create_gaze_delete_marker(self, person_id, stamp):
        marker = Marker()
        marker.header.frame_id = WORLD_FRAME
        marker.header.stamp = stamp
        marker.ns = f"person{person_id}_gaze"
        marker.id = 3
        marker.action = Marker.DELETE
        return [marker]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=None, help="Path to session_ml_dataset.parquet")
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Session directory containing session_ml_dataset.parquet",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Optional multicam_calibration.npz for camera TF display only",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--gaze-length", type=float, default=0.4)
    args = parser.parse_args()

    try:
        parquet_path = resolve_parquet_path(args.parquet, args.session_dir)
    except ValueError as exc:
        parser.error(str(exc))

    if not os.path.exists(parquet_path):
        parser.error(f"Parquet file not found: {parquet_path}")

    if rclpy is None:
        print("ROS2 Python packages are required to run this visualizer.", file=sys.stderr)
        return 1

    rclpy.init()
    node = VisualizerNode(
        parquet_path,
        calib_path=args.calib,
        fps=args.fps,
        gaze_length=args.gaze_length,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
