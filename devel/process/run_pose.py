#!/usr/bin/env python3
"""
Step 2: Pose Processing (cam1, cam2)

Runs RTMPose-L wholebody (133 keypoints, 2D) on matched frames from pose
cameras, then deprojects each keypoint to 3D using the depth.mkv video and
transforms to global (reference camera) coordinates.

Usage:
    python run_pose.py \\
        --session-dir ./recordings/session_YYYYMMDD_HHMMSS \\
        --matched-csv ./recordings/session_YYYYMMDD_HHMMSS/matched_frames.csv \\
        --calib-npz  ./calib_data/multicam_calibration.npz

Output:
    {session_dir}/pose_results.csv
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
    from mmpose.utils import register_all_modules
    register_all_modules()
    _mmpose_loaded = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_matched_csv(csv_path):
    """Return list of dicts from matched_frames.csv."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_calibration(calib_path, cam_num):
    """Load R, t to transform from camera coords to reference (global) coords."""
    calib = np.load(calib_path)
    R_key = f'R_{cam_num}_to_ref'
    t_key = f't_{cam_num}_to_ref'
    if R_key in calib and t_key in calib:
        R_cam_to_ref = calib[R_key]
        t_cam_to_ref = calib[t_key].flatten()
        return R_cam_to_ref, t_cam_to_ref
    print(f"Warning: Calibration for cam {cam_num} not found. Using identity.")
    return np.eye(3), np.zeros(3)


def deproject_pixel_to_3d(x, y, depth_image, K, depth_scale, patch_radius=2):
    """Deproject 2D pixel + depth → 3D point in camera frame.

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
    """Random-access reader for color.mp4 using OpenCV."""

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
    """Sequential reader for depth.mkv via PyAV (16-bit gray)."""

    def __init__(self, path):
        import av
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self._frames = {}
        self._iter = self.container.decode(self.stream)
        self._next_idx = 0

    def read_frame(self, idx):
        """Read frame at index idx. Supports forward seeking only."""
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
        """Free memory for frames we no longer need."""
        keys_to_remove = [k for k in self._frames if k < idx]
        for k in keys_to_remove:
            del self._frames[k]

    def close(self):
        self.container.close()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_pose_cameras(session_dir, matched_rows, calib_npz_path,
                         cfg_path, ckpt_path, device, pose_cams):
    """Run RTMPose-L on pose cameras and deproject keypoints to 3D global."""

    import torch
    from mmpose.apis import init_model, inference_topdown

    _ensure_mmpose()

    # Load model
    model = init_model(cfg_path, ckpt_path, device=device)
    print(f"RTMPose-L model loaded on {device}")

    # Load metadata for intrinsics
    meta_path = os.path.join(session_dir, "metadata.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Prepare per-camera data
    cam_data = {}
    for cam_id in pose_cams:
        cam_meta = meta['cameras'][str(cam_id)]
        depth_scale = cam_meta['depth_storage']['depth_scale_meters_per_unit']
        intr = cam_meta['intrinsics']
        K = np.array([
            [intr['fx'], 0, intr['ppx']],
            [0, intr['fy'], intr['ppy']],
            [0, 0, 1]
        ])
        R, t = load_calibration(calib_npz_path, cam_id)

        cam_dir = os.path.join(session_dir, f"cam{cam_id}")
        color_reader = VideoFrameReader(os.path.join(cam_dir, "color.mp4"))
        depth_reader = DepthFrameReader(os.path.join(cam_dir, "depth.mkv"))

        w = int(color_reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(color_reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bbox = np.array([[0.0, 0.0, float(w - 1), float(h - 1)]], dtype=np.float32)

        cam_data[cam_id] = {
            'depth_scale': depth_scale,
            'K': K,
            'R': R,
            't': t,
            'color_reader': color_reader,
            'depth_reader': depth_reader,
            'bbox': bbox,
        }

    NUM_KEYPOINTS = 133
    results = []

    for row_i, row in enumerate(tqdm(matched_rows, desc="Pose processing")):
        result_row = {'master_frame_idx': int(row['master_frame_idx'])}

        for cam_id in pose_cams:
            frame_idx = int(row[f'cam{cam_id}_idx'])
            cd = cam_data[cam_id]

            # Read frames
            color_frame = cd['color_reader'].read_frame(frame_idx)
            depth_frame = cd['depth_reader'].read_frame(frame_idx)
            cd['depth_reader'].clear_cache_before(frame_idx - 5)

            if color_frame is None:
                # Fill NaN for all keypoints
                for kpt_i in range(NUM_KEYPOINTS):
                    for axis in ('x', 'y', 'z', 'score'):
                        result_row[f'cam{cam_id}_kpt{kpt_i}_{axis}'] = ''
                continue

            # Run inference
            rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            pose_results = inference_topdown(model, rgb_frame, bboxes=cd['bbox'])

            if not pose_results or pose_results[0].pred_instances is None:
                for kpt_i in range(NUM_KEYPOINTS):
                    for axis in ('x', 'y', 'z', 'score'):
                        result_row[f'cam{cam_id}_kpt{kpt_i}_{axis}'] = ''
                continue

            pred = pose_results[0].pred_instances
            keypoints_2d = pred.keypoints[0]  # (133, 2) or (133, 3)
            scores = pred.keypoint_scores[0] if pred.keypoint_scores.ndim > 1 else pred.keypoint_scores

            # Deproject each keypoint
            for kpt_i in range(NUM_KEYPOINTS):
                px, py = keypoints_2d[kpt_i, 0], keypoints_2d[kpt_i, 1]
                score = float(scores[kpt_i]) if kpt_i < len(scores) else 0.0

                if depth_frame is not None and score > 0.3:
                    p3d_cam = deproject_pixel_to_3d(
                        px, py, depth_frame, cd['K'], cd['depth_scale'])
                    if not np.any(np.isnan(p3d_cam)):
                        p3d_global = cd['R'] @ p3d_cam + cd['t']
                    else:
                        p3d_global = np.array([np.nan, np.nan, np.nan])
                else:
                    p3d_global = np.array([np.nan, np.nan, np.nan])

                result_row[f'cam{cam_id}_kpt{kpt_i}_x'] = round(p3d_global[0], 6) if not np.isnan(p3d_global[0]) else ''
                result_row[f'cam{cam_id}_kpt{kpt_i}_y'] = round(p3d_global[1], 6) if not np.isnan(p3d_global[1]) else ''
                result_row[f'cam{cam_id}_kpt{kpt_i}_z'] = round(p3d_global[2], 6) if not np.isnan(p3d_global[2]) else ''
                result_row[f'cam{cam_id}_kpt{kpt_i}_score'] = round(score, 4)

        results.append(result_row)

    # Cleanup
    for cam_id in pose_cams:
        cam_data[cam_id]['color_reader'].close()
        cam_data[cam_id]['depth_reader'].close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run RTMW2D pose on matched frames and deproject to 3D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--session-dir', type=str, required=True)
    parser.add_argument('--matched-csv', type=str, default=None,
                        help='Path to matched_frames.csv (default: {session_dir}/matched_frames.csv)')
    parser.add_argument('--calib-npz', type=str, default=None,
                        help='Path to multicam_calibration.npz')
    parser.add_argument('--pose-cams', type=int, nargs='+', default=[1, 2],
                        help='Camera IDs for pose processing')
    parser.add_argument('--cfg-path', type=str,
                        default=None,
                        help='Path to RTMPose config .py')
    parser.add_argument('--ckpt-path', type=str,
                        default=None,
                        help='Path to RTMPose checkpoint .pth')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: cuda:0 if available)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: {session_dir}/pose_results.csv)')
    args = parser.parse_args()

    # Resolve defaults
    session_dir = args.session_dir
    matched_csv = args.matched_csv or os.path.join(session_dir, "matched_frames.csv")
    output_path = args.output or os.path.join(session_dir, "pose_results.csv")

    # Default calib path: look in session_dir, then in calib_data
    if args.calib_npz:
        calib_npz = args.calib_npz
    else:
        calib_in_session = os.path.join(session_dir, "multicam_calibration.npz")
        calib_in_calib_data = os.path.join(session_dir, "calib_data", "multicam_calibration.npz")
        if os.path.exists(calib_in_session):
            calib_npz = calib_in_session
        elif os.path.exists(calib_in_calib_data):
            calib_npz = calib_in_calib_data
        else:
            print("Error: Cannot find multicam_calibration.npz. Use --calib-npz.")
            sys.exit(1)

    # Default model paths: relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    process_dir = script_dir  # usta_pose/devel/process
    devel_dir = os.path.dirname(process_dir)  # usta_pose/devel
    project_root = os.path.dirname(devel_dir)  # usta_pose
    rtmw2d_dir = os.path.join(project_root, "models", "pose", "rtmw2d")

    cfg_path = args.cfg_path or os.path.join(
        rtmw2d_dir, "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py")
    ckpt_path = args.ckpt_path or os.path.join(
        rtmw2d_dir, "rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth")

    if not os.path.exists(cfg_path):
        print(f"Error: Config not found: {cfg_path}")
        sys.exit(1)
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Device
    if args.device:
        device = args.device
    else:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Session:    {session_dir}")
    print(f"Matched:    {matched_csv}")
    print(f"Calib:      {calib_npz}")
    print(f"Config:     {cfg_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device:     {device}")
    print(f"Pose cams:  {args.pose_cams}")
    print()

    # Load matched frames
    matched_rows = load_matched_csv(matched_csv)
    print(f"Loaded {len(matched_rows)} matched frame sets\n")

    # Process
    results = process_pose_cameras(
        session_dir, matched_rows, calib_npz,
        cfg_path, ckpt_path, device, args.pose_cams,
    )

    # Write output CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nPose results saved: {output_path}")
        print(f"  Rows: {len(results)}, Columns: {len(fieldnames)}")
    else:
        print("No results generated.")


if __name__ == '__main__':
    main()
