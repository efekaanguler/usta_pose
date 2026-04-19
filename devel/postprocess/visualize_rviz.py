#!/usr/bin/env python3
"""
RViz Visualization for 3D Pose and Gaze

Reads session_output.csv and multicam_calibration.npz to replay 
the captured session in RViz.

Usage:
    python3 visualize_rviz.py --csv /path/to/session_output.csv --calib /path/to/multicam_calibration.npz
"""

import argparse
import csv
import os
import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped, Point
from tf2_ros import StaticTransformBroadcaster
from std_msgs.msg import ColorRGBA

# ---------------------------------------------------------------------------
# SKELETON CONNECTIONS (133 keypoints RTMPose-L Wholebody)
# ---------------------------------------------------------------------------

# Body keypoint connections (kpt 0-16)
BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 6),                                 # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 11), (6, 12), (11, 12),            # torso
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
]

# Hand connections
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

# Wrist-to-hand connections
WRIST_TO_HAND = [
    (9, 91),   # left wrist -> left hand root
    (10, 112), # right wrist -> right hand root
]

ALL_SKELETON = BODY_SKELETON + LEFT_HAND + RIGHT_HAND + WRIST_TO_HAND

# Person colors
PERSON_COLORS = {
    1: {
        'joint': ColorRGBA(r=1.0, g=0.3, b=0.3, a=1.0),  # Red
        'bone':  ColorRGBA(r=1.0, g=0.5, b=0.5, a=0.8),   # Light red
        'gaze':  ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),   # Yellow
    },
    2: {
        'joint': ColorRGBA(r=0.3, g=1.0, b=0.3, a=1.0),  # Green
        'bone':  ColorRGBA(r=0.5, g=1.0, b=0.5, a=0.8),   # Light green
        'gaze':  ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),   # Cyan
    },
}


def get_eye_midpoint(row, cam_id):
    """Get the midpoint between left eye (kpt1) and right eye (kpt2).
    Falls back to nose (kpt0), then any available face keypoint."""
    
    # Try eye midpoint first
    lx = row.get(f"cam{cam_id}_kpt1_x")
    ly = row.get(f"cam{cam_id}_kpt1_y")
    lz = row.get(f"cam{cam_id}_kpt1_z")
    rx = row.get(f"cam{cam_id}_kpt2_x")
    ry = row.get(f"cam{cam_id}_kpt2_y")
    rz = row.get(f"cam{cam_id}_kpt2_z")
    
    if lx and ly and lz and rx and ry and rz:
        return Point(
            x=(float(lx) + float(rx)) / 2.0,
            y=(float(ly) + float(ry)) / 2.0,
            z=(float(lz) + float(rz)) / 2.0,
        )
    
    # Fallback: single eye
    if lx and ly and lz:
        return Point(x=float(lx), y=float(ly), z=float(lz))
    if rx and ry and rz:
        return Point(x=float(rx), y=float(ry), z=float(rz))
    
    # Fallback: nose (kpt0)
    nx = row.get(f"cam{cam_id}_kpt0_x")
    ny = row.get(f"cam{cam_id}_kpt0_y")
    nz = row.get(f"cam{cam_id}_kpt0_z")
    if nx and ny and nz:
        return Point(x=float(nx), y=float(ny), z=float(nz))
    
    return None


class VisualizerNode(Node):
    def __init__(self, csv_path, calib_path, fps=30.0):
        super().__init__('session_visualizer')
        self.csv_path = csv_path
        self.calib_path = calib_path
        self.fps = fps

        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Load data
        self.rows = self.load_csv(csv_path)
        self.get_logger().info(f"Loaded {len(self.rows)} frames from CSV")
        
        if calib_path:
            self.publish_static_tfs(calib_path)

        self.frame_idx = 0
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

    def load_csv(self, path):
        rows = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def publish_static_tfs(self, calib_path):
        """Publish camera extrinsics as static transforms to ROS TF."""
        try:
            calib = np.load(calib_path)
            tfs = []
            
            for cam_id in [1, 2, 3, 4]:
                R_key = f'R_{cam_id}_to_ref'
                t_key = f't_{cam_id}_to_ref'
                
                if R_key in calib and t_key in calib:
                    R = calib[R_key]
                    t = calib[t_key].flatten()
                    q = self.matrix_to_quaternion(R)

                    t_msg = TransformStamped()
                    t_msg.header.stamp = self.get_clock().now().to_msg()
                    t_msg.header.frame_id = 'world'
                    t_msg.child_frame_id = f'cam{cam_id}'
                    t_msg.transform.translation.x = float(t[0])
                    t_msg.transform.translation.y = float(t[1])
                    t_msg.transform.translation.z = float(t[2])
                    t_msg.transform.rotation.x = q[0]
                    t_msg.transform.rotation.y = q[1]
                    t_msg.transform.rotation.z = q[2]
                    t_msg.transform.rotation.w = q[3]
                    tfs.append(t_msg)
            
            self.tf_static_broadcaster.sendTransform(tfs)
            self.get_logger().info("Published static camera transforms")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {e}")

    def timer_callback(self):
        if self.frame_idx >= len(self.rows):
            self.get_logger().info("Session playback finished. Restarting...")
            self.frame_idx = 0
            return

        row = self.rows[self.frame_idx]
        markers = MarkerArray()

        # For each person (cam1=Person1, cam2=Person2)
        for cam_id in [1, 2]:
            markers.markers.extend(self.create_person_markers(row, cam_id))

        # Gaze arrows
        markers.markers.extend(self.create_gaze_markers(row))

        # Clear old markers that may have disappeared
        markers.markers.extend(self.create_cleanup_markers(row))

        self.marker_pub.publish(markers)
        self.frame_idx += 1

    def create_person_markers(self, row, cam_id):
        """Create joint spheres and bone lines for one person."""
        markers = []
        now = self.get_clock().now().to_msg()
        colors = PERSON_COLORS[cam_id]

        # --- Joint spheres ---
        joint_marker = Marker()
        joint_marker.header.frame_id = "world"
        joint_marker.header.stamp = now
        joint_marker.ns = f"person{cam_id}_joints"
        joint_marker.id = 0
        joint_marker.type = Marker.SPHERE_LIST
        joint_marker.action = Marker.ADD
        joint_marker.lifetime.sec = 0
        joint_marker.lifetime.nanosec = int(1e9 / self.fps * 2)
        joint_marker.pose.orientation.w = 1.0
        joint_marker.scale.x = 0.025
        joint_marker.scale.y = 0.025
        joint_marker.scale.z = 0.025
        joint_marker.color = colors['joint']
        
        for i in range(133):
            col_x = f"cam{cam_id}_kpt{i}_x"
            col_y = f"cam{cam_id}_kpt{i}_y"
            col_z = f"cam{cam_id}_kpt{i}_z"
            
            if row.get(col_x) and row.get(col_y) and row.get(col_z):
                p = Point()
                p.x = float(row[col_x])
                p.y = float(row[col_y])
                p.z = float(row[col_z])
                joint_marker.points.append(p)
        
        if joint_marker.points:
            markers.append(joint_marker)

        # --- Bone lines ---
        bone_marker = Marker()
        bone_marker.header.frame_id = "world"
        bone_marker.header.stamp = now
        bone_marker.ns = f"person{cam_id}_bones"
        bone_marker.id = 1
        bone_marker.type = Marker.LINE_LIST
        bone_marker.action = Marker.ADD
        bone_marker.lifetime.sec = 0
        bone_marker.lifetime.nanosec = int(1e9 / self.fps * 2)
        bone_marker.pose.orientation.w = 1.0
        bone_marker.scale.x = 0.008
        bone_marker.color = colors['bone']
        
        for start, end in ALL_SKELETON:
            sx = f"cam{cam_id}_kpt{start}_x"
            sy = f"cam{cam_id}_kpt{start}_y"
            sz = f"cam{cam_id}_kpt{start}_z"
            ex = f"cam{cam_id}_kpt{end}_x"
            ey = f"cam{cam_id}_kpt{end}_y"
            ez = f"cam{cam_id}_kpt{end}_z"
            
            if (row.get(sx) and row.get(sy) and row.get(sz) and 
                row.get(ex) and row.get(ey) and row.get(ez)):
                p_start = Point(x=float(row[sx]), y=float(row[sy]), z=float(row[sz]))
                p_end = Point(x=float(row[ex]), y=float(row[ey]), z=float(row[ez]))
                bone_marker.points.append(p_start)
                bone_marker.points.append(p_end)
        
        if bone_marker.points:
            markers.append(bone_marker)

        return markers

    def create_gaze_markers(self, row):
        """Create gaze arrows originating from the eye midpoint."""
        markers = []
        now = self.get_clock().now().to_msg()

        # cam3 gaze -> Person 1 (cam1 eyes)
        # cam4 gaze -> Person 2 (cam2 eyes)
        gaze_mapping = {3: 1, 4: 2}

        for gaze_cam, pose_cam in gaze_mapping.items():
            gx = f"cam{gaze_cam}_gaze_x"
            gy = f"cam{gaze_cam}_gaze_y"
            gz = f"cam{gaze_cam}_gaze_z"
            
            if not (row.get(gx) and row.get(gy) and row.get(gz)):
                continue

            # Get eye midpoint as gaze origin
            origin = get_eye_midpoint(row, pose_cam)
            if origin is None:
                continue

            vec = np.array([float(row[gx]), float(row[gy]), float(row[gz])])
            arrow_length = 0.4  # 40cm arrow
            end_p = Point(
                x=origin.x + vec[0] * arrow_length,
                y=origin.y + vec[1] * arrow_length,
                z=origin.z + vec[2] * arrow_length,
            )

            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = now
            marker.ns = f"person{pose_cam}_gaze"
            marker.id = gaze_cam
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = int(1e9 / self.fps * 2)
            marker.scale.x = 0.015  # shaft diameter
            marker.scale.y = 0.03   # head diameter
            marker.scale.z = 0.05   # head length
            marker.color = PERSON_COLORS[pose_cam]['gaze']
            marker.points = [origin, end_p]
            markers.append(marker)
        
        return markers

    def create_cleanup_markers(self, row):
        """Delete markers for persons that have no data this frame."""
        markers = []
        now = self.get_clock().now().to_msg()
        
        for cam_id in [1, 2]:
            has_data = any(
                row.get(f"cam{cam_id}_kpt{i}_x")
                for i in range(17)  # check body keypoints only
            )
            if not has_data:
                for ns, mid in [(f"person{cam_id}_joints", 0), 
                                (f"person{cam_id}_bones", 1)]:
                    m = Marker()
                    m.header.frame_id = "world"
                    m.header.stamp = now
                    m.ns = ns
                    m.id = mid
                    m.action = Marker.DELETE
                    markers.append(m)
        return markers

    @staticmethod
    def matrix_to_quaternion(R):
        """Rotation matrix to quaternion [x, y, z, w]."""
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return [qx, qy, qz, qw]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to session_output.csv')
    parser.add_argument('--calib', type=str, help='Path to multicam_calibration.npz')
    parser.add_argument('--fps', type=float, default=30.0)
    args = parser.parse_args()

    rclpy.init()
    node = VisualizerNode(args.csv, args.calib, fps=args.fps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
