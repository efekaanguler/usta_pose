#!/usr/bin/env python3
"""
Step 1: Independent Pose Processing (Cam-Specific)

Runs RTMPose-L wholebody (133 keypoints, 2D) on ALL frames of a single pose
camera based on its hardware timestamps. Deprojects each keypoint to 3D 
in the LOCAL CAMERA FRAME using the depth.mkv video.
NO global transformation is applied at this stage.

Usage:
    python extract_pose_independent.py \
        --session-dir ./recordings/session_YYYYMMDD_HHMMSS \
        --cam-id 1

Output:
    {session_dir}/cam1/cam1_pose_raw.csv
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# MMPose imports (deferred so --help works without torch)
# ---------------------------------------------------------------------------
_mmpose_loaded = False

def _ensure_mmpose():
    global _mmpose_loaded
    if _mmpose_loaded:
        return
    try:
        from mmpose.utils import register_all_modules
        register_all_modules()
        _mmpose_loaded = True
    except ImportError:
        print("Warning: mmpose not found. Make sure you are in the correct environment.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_timestamps(csv_path):
    """Return list of dicts from camX_color_timestamps.csv."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'frame_idx': int(row['frame_idx']),
                'hw_timestamp_ms': float(row['hw_timestamp_ms'])
            })
    rows.sort(key=lambda x: x['frame_idx'])
    return rows


def deproject_pixel_to_3d(x, y, depth_image, K, depth_scale, patch_radius=2):
    """Deproject 2D pixel + depth -> 3D point in camera frame.

    Uses a median of a small patch around (x, y) for robustness.
    Returns [X, Y, Z] or [nan, nan, nan] if invalid.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]

    h, w = depth_image.shape
    ix, iy = int(round(x)), int(round(y))

    y_lo = max(0, iy - patch_radius)
    y_hi = min(h, iy + patch_radius + 1)
    x_lo = max(0, ix - patch_radius)
    x_hi = min(w, ix + patch_radius + 1)

    patch = depth_image[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
    valid = patch[patch > 0]

    if len(valid) == 0:
        return np.array([np.nan, np.nan, np.nan])

    Z = float(np.median(valid)) * depth_scale
    if Z < 0.1 or Z > 10.0:
        return np.array([np.nan, np.nan, np.nan])

    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return np.array([X, Y, Z])


# ---------------------------------------------------------------------------
# Frame seeking helpers
# ---------------------------------------------------------------------------

class VideoFrameReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        self.cap.release()


class DepthFrameReader:
    def __init__(self, path):
        import av
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self._frames = {}
        self._iter = self.container.decode(self.stream)
        self._next_idx = 0

    def read_frame(self, idx):
        if idx in self._frames:
            return self._frames[idx]

        while self._next_idx <= idx:
            try:
                frame_av = next(self._iter)
                arr = frame_av.to_ndarray()
                self._frames[self._next_idx] = arr
                self._next_idx += 1
            except StopIteration:
                return None

        return self._frames.get(idx)

    def clear_cache_before(self, idx):
        keys_to_remove = [k for k in self._frames if k < idx]
        for k in keys_to_remove:
            del self._frames[k]

    def close(self):
        self.container.close()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_camera(session_dir, cam_id, cfg_path, ckpt_path, device):
    """Run RTMPose-L on a single pose camera and deproject keypoints to local 3D."""

    import torch
    from mmpose.apis import init_model, inference_topdown

    _ensure_mmpose()

    # Load model
    model = init_model(cfg_path, ckpt_path, device=device)
    print(f"RTMPose-L model loaded on {device}")

    # Load metadata for intrinsics
    meta_path = os.path.join(session_dir, "metadata.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(os.path.dirname(session_dir), "metadata.json")
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    cam_meta = meta['cameras'][str(cam_id)]
    depth_scale = cam_meta['depth_storage']['depth_scale_meters_per_unit']
    intr = cam_meta['intrinsics']
    K = np.array([
        [intr['fx'], 0, intr['ppx']],
        [0, intr['fy'], intr['ppy']],
        [0, 0, 1]
    ])

    cam_dir = os.path.join(session_dir, f"cam{cam_id}")
    ts_csv_path = os.path.join(cam_dir, f"cam{cam_id}_color_timestamps.csv")
    
    if not os.path.exists(ts_csv_path):
        raise FileNotFoundError(f"Timestamps file not found: {ts_csv_path}")

    color_reader = VideoFrameReader(os.path.join(cam_dir, "color.mp4"))
    depth_reader = DepthFrameReader(os.path.join(cam_dir, "depth.mkv"))
    timestamps = load_timestamps(ts_csv_path)

    w = int(color_reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(color_reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bbox = np.array([[0.0, 0.0, float(w - 1), float(h - 1)]], dtype=np.float32)

    NUM_KEYPOINTS = 133
    results = []

    for row in tqdm(timestamps, desc=f"Pose processing Cam {cam_id}"):
        frame_idx = row['frame_idx']
        hw_ts = row['hw_timestamp_ms']
        
        result_row = {
            'frame_idx': frame_idx,
            'hw_timestamp_ms': hw_ts
        }

        # Read frames
        color_frame = color_reader.read_frame(frame_idx)
        depth_frame = depth_reader.read_frame(frame_idx)
        depth_reader.clear_cache_before(frame_idx - 5)

        if color_frame is None:
            # Fill NaN for all keypoints
            for kpt_i in range(NUM_KEYPOINTS):
                for axis in ('x', 'y', 'z', 'score'):
                    result_row[f'kpt{kpt_i}_{axis}'] = ''
            results.append(result_row)
            continue

        # Run inference
        rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        pose_results = inference_topdown(model, rgb_frame, bboxes=bbox)

        if not pose_results or pose_results[0].pred_instances is None:
            for kpt_i in range(NUM_KEYPOINTS):
                for axis in ('x', 'y', 'z', 'score'):
                    result_row[f'kpt{kpt_i}_{axis}'] = ''
            results.append(result_row)
            continue

        pred = pose_results[0].pred_instances
        keypoints_2d = pred.keypoints[0]  # (133, 2) or (133, 3)
        scores = pred.keypoint_scores[0] if pred.keypoint_scores.ndim > 1 else pred.keypoint_scores

        # Deproject each keypoint
        for kpt_i in range(NUM_KEYPOINTS):
            px, py = keypoints_2d[kpt_i, 0], keypoints_2d[kpt_i, 1]
            score = float(scores[kpt_i]) if kpt_i < len(scores) else 0.0

            # DO NOT TRANSFORM TO GLOBAL (LOCAL CAMERA FRAME ONLY)
            if depth_frame is not None and score > 0.3:
                p3d_cam = deproject_pixel_to_3d(px, py, depth_frame, K, depth_scale)
            else:
                p3d_cam = np.array([np.nan, np.nan, np.nan])

            result_row[f'kpt{kpt_i}_x'] = round(p3d_cam[0], 6) if not np.isnan(p3d_cam[0]) else ''
            result_row[f'kpt{kpt_i}_y'] = round(p3d_cam[1], 6) if not np.isnan(p3d_cam[1]) else ''
            result_row[f'kpt{kpt_i}_z'] = round(p3d_cam[2], 6) if not np.isnan(p3d_cam[2]) else ''
            result_row[f'kpt{kpt_i}_score'] = round(score, 4)

        results.append(result_row)

    # Cleanup
    color_reader.close()
    depth_reader.close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run RTMW2D pose on all frames independently (Local Coords)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--session-dir', type=str, required=True)
    parser.add_argument('--cam-id', type=int, required=True, help='Camera ID (e.g., 1 or 2)')
    parser.add_argument('--cfg-path', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    session_dir = args.session_dir
    cam_id = args.cam_id
    output_path = os.path.join(session_dir, f"cam{cam_id}", f"cam{cam_id}_pose_raw.csv")

    # Default model paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    devel_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(devel_dir)
    rtmw2d_dir = os.path.join(project_root, "models", "pose", "rtmw2d")

    cfg_path = args.cfg_path or os.path.join(
        rtmw2d_dir, "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py")
    ckpt_path = args.ckpt_path or os.path.join(
        rtmw2d_dir, "rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth")

    # Device
    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"Session:    {session_dir}")
    print(f"Camera ID:  {cam_id}")
    print(f"Config:     {cfg_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device:     {device}")
    print()

    results = process_camera(session_dir, cam_id, cfg_path, ckpt_path, device)

    # Write output CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nPose results saved: {output_path}")
    else:
        print("No results generated.")


if __name__ == '__main__':
    main()
