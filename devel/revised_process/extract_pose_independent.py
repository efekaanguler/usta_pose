#!/usr/bin/env python3
"""
Step 1: Independent Pose Processing (Cam-Specific)

Runs a selectable top-down wholebody pose model on ALL frames of a single
pose camera based on its hardware timestamps. The default is RTMPose-L 2D
because metric depth deprojection requires reliable image-space keypoints.
RTMW3D remains available explicitly, but it should be used only with a tight
person bbox for this depth-deprojection path. NO global transformation is
applied at this stage.

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


NUM_KEYPOINTS = 133
MIN_KEYPOINT_SCORE = 0.3
CAM1_DEFAULT_RIGHT_CROP_FRACTION = 0.2
CAM1_REJECT_CLOSE_DEPTH_M = 0.7
CAM1_REJECT_MAX_BODY_AREA_RATIO = 0.32
BODY_KEYPOINTS = tuple(range(17))
POSE_MODEL_RTMPOSE2D = "rtmpose2d"
POSE_MODEL_RTMW3D = "rtmw3d"
SUPPORTED_POSE_MODELS = (POSE_MODEL_RTMPOSE2D, POSE_MODEL_RTMW3D)


def default_model_paths(project_root, pose_model):
    if pose_model == POSE_MODEL_RTMPOSE2D:
        model_dir = os.path.join(project_root, "models", "pose", "rtmw2d")
        return (
            os.path.join(model_dir, "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"),
            os.path.join(model_dir, "rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"),
        )
    if pose_model == POSE_MODEL_RTMW3D:
        model_dir = os.path.join(project_root, "models", "pose", "rtmw3d")
        return (
            os.path.join(model_dir, "rtmw3d-l_8xb64_cocktail14-384x288.py"),
            os.path.join(model_dir, "rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"),
        )
    raise ValueError(f"Unsupported pose model: {pose_model}")

# ---------------------------------------------------------------------------
# MMPose imports (deferred so --help works without torch)
# ---------------------------------------------------------------------------
_mmpose_loaded = False

def _ensure_mmpose(project_root=None, require_rtmw3d=False):
    global _mmpose_loaded
    if _mmpose_loaded and not require_rtmw3d:
        return
    if require_rtmw3d:
        rtmpose3d_candidates = [
            os.environ.get("RTMPOSE3D_PROJECT_DIR"),
            "/workspace/mmpose/projects/rtmpose3d",
        ]
        if project_root:
            rtmpose3d_candidates.append(os.path.join(project_root, "mmpose", "projects", "rtmpose3d"))
        for candidate in rtmpose3d_candidates:
            if candidate and os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)
    try:
        from mmpose.utils import register_all_modules
        if require_rtmw3d:
            try:
                import rtmpose3d  # noqa: F401
            except ImportError:
                print(
                    "Warning: rtmpose3d custom module not found. Set "
                    "RTMPOSE3D_PROJECT_DIR if init_model cannot import it."
                )
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


def intrinsics_to_matrix(intr):
    return np.array([
        [intr['fx'], 0.0, intr['ppx']],
        [0.0, intr['fy'], intr['ppy']],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def flatten_depth_image(depth_image):
    if depth_image is None:
        return None
    if depth_image.ndim == 3:
        return depth_image[:, :, 0]
    return depth_image


def build_depth_ray_grid(depth_intr, shape):
    h, w = shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    x_norm = (uu - float(depth_intr['ppx'])) / float(depth_intr['fx'])
    y_norm = (vv - float(depth_intr['ppy'])) / float(depth_intr['fy'])
    return x_norm.astype(np.float32), y_norm.astype(np.float32)


def align_depth_to_color(depth_image, x_norm, y_norm, rotation, translation, color_intr, depth_scale):
    """Project raw depth-frame z16 pixels into the color camera image plane."""
    depth_image = flatten_depth_image(depth_image)
    color_h = int(color_intr['height'])
    color_w = int(color_intr['width'])
    out = np.zeros((color_h, color_w), dtype=np.uint16)

    z_m = depth_image.astype(np.float32) * float(depth_scale)
    valid = z_m > 0.0
    if not np.any(valid):
        return out

    z = z_m[valid]
    x_d = x_norm[valid] * z
    y_d = y_norm[valid] * z

    x_c = rotation[0, 0] * x_d + rotation[0, 1] * y_d + rotation[0, 2] * z + translation[0]
    y_c = rotation[1, 0] * x_d + rotation[1, 1] * y_d + rotation[1, 2] * z + translation[1]
    z_c = rotation[2, 0] * x_d + rotation[2, 1] * y_d + rotation[2, 2] * z + translation[2]

    valid_z = z_c > 1e-6
    if not np.any(valid_z):
        return out

    x_c = x_c[valid_z]
    y_c = y_c[valid_z]
    z_c = z_c[valid_z]

    u = np.rint((x_c / z_c) * float(color_intr['fx']) + float(color_intr['ppx'])).astype(np.int32)
    v = np.rint((y_c / z_c) * float(color_intr['fy']) + float(color_intr['ppy'])).astype(np.int32)
    in_bounds = (u >= 0) & (u < color_w) & (v >= 0) & (v < color_h)
    if not np.any(in_bounds):
        return out

    u = u[in_bounds]
    v = v[in_bounds]
    z_c = z_c[in_bounds]
    z16 = np.rint(z_c / float(depth_scale)).astype(np.int32)

    valid_range = (z16 > 0) & (z16 <= np.iinfo(np.uint16).max)
    if not np.any(valid_range):
        return out

    u = u[valid_range]
    v = v[valid_range]
    z16 = z16[valid_range].astype(np.uint16)

    flat_idx = v * color_w + u
    flat = np.full(color_h * color_w, np.iinfo(np.uint16).max, dtype=np.uint16)
    np.minimum.at(flat, flat_idx, z16)
    out = flat.reshape(color_h, color_w)
    out[out == np.iinfo(np.uint16).max] = 0
    return out


class DepthProjector:
    """Provides color-frame depth for keypoints detected in color images."""

    def __init__(self, cam_meta):
        depth_storage = cam_meta.get('depth_storage', {})
        calibration = cam_meta.get('calibration', {})

        self.depth_scale = float(
            depth_storage.get(
                'depth_scale_meters_per_unit',
                calibration.get('depth_scale_meters_per_unit', 0.001),
            )
        )
        self.color_intr = calibration.get('color_intrinsics') or cam_meta.get('intrinsics')
        if self.color_intr is None:
            raise KeyError("Missing color intrinsics in metadata.")

        self.K = intrinsics_to_matrix(self.color_intr)
        self.aligned_to = str(depth_storage.get('aligned_to', 'unknown')).lower()
        self.alignment_mode = str(depth_storage.get('alignment_mode', 'unknown'))
        self._x_norm = None
        self._y_norm = None
        self._depth_shape = None

        self.needs_alignment = self.aligned_to != 'color'
        if self.needs_alignment:
            self.depth_intr = calibration.get('depth_intrinsics')
            extr = calibration.get('depth_to_color_extrinsics')
            if self.depth_intr is None or extr is None:
                raise ValueError(
                    "Depth is not color-aligned, but metadata lacks depth_intrinsics "
                    "or depth_to_color_extrinsics."
                )
            self.rotation = np.array(extr['rotation'], dtype=np.float32).reshape(3, 3)
            self.translation = np.array(extr['translation'], dtype=np.float32).reshape(3)
        else:
            self.depth_intr = None
            self.rotation = None
            self.translation = None

    def depth_for_color(self, depth_image):
        depth_image = flatten_depth_image(depth_image)
        if depth_image is None or not self.needs_alignment:
            return depth_image

        if self._depth_shape != depth_image.shape:
            self._depth_shape = depth_image.shape
            self._x_norm, self._y_norm = build_depth_ray_grid(self.depth_intr, depth_image.shape)

        return align_depth_to_color(
            depth_image,
            self._x_norm,
            self._y_norm,
            self.rotation,
            self.translation,
            self.color_intr,
            self.depth_scale,
        )


def deproject_pixel_to_3d(x, y, depth_image, K, depth_scale, patch_radius=2):
    """Deproject 2D pixel + depth -> 3D point in camera frame.

    Uses a median of a small patch around (x, y) for robustness.
    Returns [X, Y, Z] or [nan, nan, nan] if invalid.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    depth_image = flatten_depth_image(depth_image)

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


def normalize_model_outputs(pred_instances):
    keypoints = np.asarray(pred_instances.keypoints, dtype=np.float64)
    keypoints = np.squeeze(keypoints)
    if keypoints.ndim != 2 or keypoints.shape[0] < NUM_KEYPOINTS:
        return None, None

    scores = getattr(pred_instances, "keypoint_scores", None)
    if scores is None:
        scores = np.ones(keypoints.shape[0], dtype=np.float64)
    else:
        scores = np.asarray(scores, dtype=np.float64)
        scores = np.squeeze(scores)
        if scores.ndim != 1 or scores.shape[0] < NUM_KEYPOINTS:
            scores = np.ones(keypoints.shape[0], dtype=np.float64)

    return keypoints[:NUM_KEYPOINTS], scores[:NUM_KEYPOINTS]


def default_pose_bbox(cam_id, width, height):
    if cam_id == 1:
        x2 = float(int(round(width * (1.0 - CAM1_DEFAULT_RIGHT_CROP_FRACTION))) - 1)
        return [0.0, 0.0, max(0.0, x2), float(height - 1)]
    return [0.0, 0.0, float(width - 1), float(height - 1)]


def set_empty_keypoints(row, scores=None):
    for kpt_i in range(NUM_KEYPOINTS):
        for axis in ('x', 'y', 'z'):
            row[f'kpt{kpt_i}_{axis}'] = ''
        if scores is not None and kpt_i < len(scores) and np.isfinite(scores[kpt_i]):
            row[f'kpt{kpt_i}_score'] = round(float(scores[kpt_i]), 4)
        else:
            row[f'kpt{kpt_i}_score'] = ''


def pose_quality_stats(keypoints_2d, scores, local_points, frame_width, frame_height):
    body_idx = np.array(BODY_KEYPOINTS, dtype=np.int32)
    body_scores = scores[body_idx]
    body_points = local_points[body_idx]
    body_pixels = keypoints_2d[body_idx, :2]

    body_valid_3d = (
        np.isfinite(body_scores)
        & (body_scores >= MIN_KEYPOINT_SCORE)
        & np.all(np.isfinite(body_points), axis=1)
    )
    body_valid_2d = (
        np.isfinite(body_scores)
        & (body_scores >= MIN_KEYPOINT_SCORE)
        & np.all(np.isfinite(body_pixels), axis=1)
        & (body_pixels[:, 0] >= 0.0)
        & (body_pixels[:, 0] < frame_width)
        & (body_pixels[:, 1] >= 0.0)
        & (body_pixels[:, 1] < frame_height)
    )

    median_depth = np.nan
    if np.any(body_valid_3d):
        median_depth = float(np.nanmedian(body_points[body_valid_3d, 2]))

    bbox_area = np.nan
    if np.sum(body_valid_2d) >= 5:
        valid_pixels = body_pixels[body_valid_2d]
        bbox_w = float(np.nanmax(valid_pixels[:, 0]) - np.nanmin(valid_pixels[:, 0]))
        bbox_h = float(np.nanmax(valid_pixels[:, 1]) - np.nanmin(valid_pixels[:, 1]))
        bbox_area = bbox_w * bbox_h

    return {
        "body_valid_keypoints": int(np.sum(body_valid_3d)),
        "body_median_depth_m": median_depth,
        "body_bbox_area_px": bbox_area,
    }


def rejection_reason(cam_id, stats, frame_width, frame_height, reject_close_depth_m, reject_max_body_area_ratio):
    if cam_id != 1:
        return ""

    valid_count = stats["body_valid_keypoints"]
    median_depth = stats["body_median_depth_m"]
    bbox_area = stats["body_bbox_area_px"]
    max_area = float(frame_width * frame_height) * reject_max_body_area_ratio

    if valid_count >= 5 and np.isfinite(median_depth) and median_depth < reject_close_depth_m:
        return "cam1_foreground_depth"
    if valid_count >= 5 and np.isfinite(bbox_area) and bbox_area > max_area:
        return "cam1_foreground_bbox"
    return ""


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

def process_camera(
    session_dir,
    cam_id,
    cfg_path,
    ckpt_path,
    device,
    bbox_xyxy=None,
    foreground_rejection=True,
    reject_close_depth_m=CAM1_REJECT_CLOSE_DEPTH_M,
    reject_max_body_area_ratio=CAM1_REJECT_MAX_BODY_AREA_RATIO,
    pose_model=POSE_MODEL_RTMPOSE2D,
):
    """Run the selected pose model and deproject image keypoints to local 3D."""

    import torch
    from mmpose.apis import init_model, inference_topdown

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    _ensure_mmpose(project_root, require_rtmw3d=(pose_model == POSE_MODEL_RTMW3D))

    model = init_model(cfg_path, ckpt_path, device=device)
    if pose_model == POSE_MODEL_RTMW3D:
        try:
            model.cfg.model.test_cfg.mode = "vis"
        except Exception:
            pass
        print("Warning: RTMW3D metric deprojection is unreliable with broad/full-frame bboxes. Use a tight --bbox.")
    model_label = "RTMW3D" if pose_model == POSE_MODEL_RTMW3D else "RTMPose-L 2D"
    print(f"{model_label} model loaded on {device}")

    meta_path = os.path.join(session_dir, "metadata.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(os.path.dirname(session_dir), "metadata.json")

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    cam_meta = meta['cameras'][str(cam_id)]
    depth_projector = DepthProjector(cam_meta)
    K = depth_projector.K
    depth_scale = depth_projector.depth_scale
    if depth_projector.needs_alignment:
        print(
            f"Cam {cam_id}: depth is {depth_projector.aligned_to} "
            f"({depth_projector.alignment_mode}); aligning depth to color before deprojection."
        )

    cam_dir = os.path.join(session_dir, f"cam{cam_id}")
    ts_csv_path = os.path.join(cam_dir, f"cam{cam_id}_color_timestamps.csv")

    if not os.path.exists(ts_csv_path):
        raise FileNotFoundError(f"Timestamps file not found: {ts_csv_path}")

    color_reader = VideoFrameReader(os.path.join(cam_dir, "color.mp4"))
    depth_reader = DepthFrameReader(os.path.join(cam_dir, "depth.mkv"))
    timestamps = load_timestamps(ts_csv_path)

    w = int(color_reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(color_reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if pose_model == POSE_MODEL_RTMW3D and bbox_xyxy is None:
        raise ValueError(
            "RTMW3D metric deprojection requires an explicit tight --bbox; "
            "broad/default boxes produced invalid metric skeletons."
        )
    if bbox_xyxy is None:
        bbox_values = default_pose_bbox(cam_id, w, h)
    else:
        x1, y1, x2, y2 = bbox_xyxy
        bbox_values = [
            float(np.clip(x1, 0, w - 1)),
            float(np.clip(y1, 0, h - 1)),
            float(np.clip(x2, 0, w - 1)),
            float(np.clip(y2, 0, h - 1)),
        ]
        if bbox_values[2] <= bbox_values[0] or bbox_values[3] <= bbox_values[1]:
            raise ValueError(f"Invalid bbox for cam {cam_id}: {bbox_xyxy}")
    bbox = np.array([bbox_values], dtype=np.float32)
    print(f"Cam {cam_id}: pose bbox xyxy={bbox_values}")
    if cam_id == 1 and bbox_xyxy is None:
        print("Cam 1: default ROI excludes the rightmost 1/5 of the image.")
    if cam_id == 1 and foreground_rejection:
        print(
            "Cam 1: foreground rejection enabled "
            f"(depth < {reject_close_depth_m:.2f}m or body area > "
            f"{reject_max_body_area_ratio:.2f} of frame)."
        )

    results = []

    for row in tqdm(timestamps, desc=f"Pose processing Cam {cam_id}"):
        frame_idx = row['frame_idx']
        hw_ts = row['hw_timestamp_ms']

        result_row = {
            'frame_idx': frame_idx,
            'hw_timestamp_ms': hw_ts,
            'pose_rejected': 0,
            'pose_reject_reason': '',
            'body_valid_keypoints': '',
            'body_median_depth_m': '',
            'body_bbox_area_px': '',
        }

        color_frame = color_reader.read_frame(frame_idx)
        depth_frame = depth_reader.read_frame(frame_idx)
        depth_reader.clear_cache_before(frame_idx - 5)
        depth_for_color = depth_projector.depth_for_color(depth_frame) if depth_frame is not None else None

        if color_frame is None:
            set_empty_keypoints(result_row)
            results.append(result_row)
            continue

        rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        pose_results = inference_topdown(model, rgb_frame, bboxes=bbox)

        if not pose_results or pose_results[0].pred_instances is None:
            set_empty_keypoints(result_row)
            results.append(result_row)
            continue

        pred = pose_results[0].pred_instances
        keypoints, scores = normalize_model_outputs(pred)
        if keypoints is None or scores is None:
            set_empty_keypoints(result_row)
            results.append(result_row)
            continue

        local_points = np.full((NUM_KEYPOINTS, 3), np.nan, dtype=np.float64)
        for kpt_i in range(NUM_KEYPOINTS):
            px, py = keypoints[kpt_i, 0], keypoints[kpt_i, 1]
            score = float(scores[kpt_i]) if kpt_i < len(scores) else 0.0

            if depth_for_color is not None and score >= MIN_KEYPOINT_SCORE:
                local_points[kpt_i] = deproject_pixel_to_3d(px, py, depth_for_color, K, depth_scale)

        stats = pose_quality_stats(keypoints, scores, local_points, w, h)
        result_row['body_valid_keypoints'] = stats["body_valid_keypoints"]
        if np.isfinite(stats["body_median_depth_m"]):
            result_row['body_median_depth_m'] = round(stats["body_median_depth_m"], 6)
        if np.isfinite(stats["body_bbox_area_px"]):
            result_row['body_bbox_area_px'] = round(stats["body_bbox_area_px"], 3)

        reason = ""
        if foreground_rejection:
            reason = rejection_reason(
                cam_id,
                stats,
                w,
                h,
                reject_close_depth_m,
                reject_max_body_area_ratio,
            )
        if reason:
            result_row['pose_rejected'] = 1
            result_row['pose_reject_reason'] = reason
            set_empty_keypoints(result_row, scores=scores)
            results.append(result_row)
            continue

        for kpt_i in range(NUM_KEYPOINTS):
            p3d_cam = local_points[kpt_i]
            score = float(scores[kpt_i]) if kpt_i < len(scores) else 0.0

            result_row[f'kpt{kpt_i}_x'] = round(p3d_cam[0], 6) if not np.isnan(p3d_cam[0]) else ''
            result_row[f'kpt{kpt_i}_y'] = round(p3d_cam[1], 6) if not np.isnan(p3d_cam[1]) else ''
            result_row[f'kpt{kpt_i}_z'] = round(p3d_cam[2], 6) if not np.isnan(p3d_cam[2]) else ''
            result_row[f'kpt{kpt_i}_score'] = round(score, 4)

        results.append(result_row)

    color_reader.close()
    depth_reader.close()

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Run pose on all frames independently (Local Coords)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--session-dir', type=str, required=True)
    parser.add_argument('--cam-id', type=int, required=True, help='Camera ID (e.g., 1 or 2)')
    parser.add_argument('--cfg-path', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument(
        '--pose-model',
        choices=SUPPORTED_POSE_MODELS,
        default=os.environ.get('POSE_MODEL', POSE_MODEL_RTMPOSE2D),
        help='Pose backbone used for image-space keypoints before depth deprojection.',
    )
    parser.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        default=None,
        help='Optional person-specific ROI in color-image pixel coordinates.',
    )
    parser.add_argument(
        '--disable-foreground-rejection',
        action='store_true',
        help='Disable cam1 close/large-body rejection for debugging.',
    )
    parser.add_argument(
        '--reject-close-depth-m',
        type=float,
        default=CAM1_REJECT_CLOSE_DEPTH_M,
        help='Cam1 frame rejection threshold for foreground median body depth.',
    )
    parser.add_argument(
        '--reject-max-body-area-ratio',
        type=float,
        default=CAM1_REJECT_MAX_BODY_AREA_RATIO,
        help='Cam1 frame rejection threshold for projected body bbox area / frame area.',
    )
    args = parser.parse_args()

    session_dir = args.session_dir
    cam_id = args.cam_id
    output_path = os.path.join(session_dir, f"cam{cam_id}", f"cam{cam_id}_pose_raw.csv")

    # Default model paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    devel_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(devel_dir)
    default_cfg_path, default_ckpt_path = default_model_paths(project_root, args.pose_model)

    cfg_path = args.cfg_path or default_cfg_path
    ckpt_path = args.ckpt_path or default_ckpt_path

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
    print(f"Pose model: {args.pose_model}")
    print(f"Config:     {cfg_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device:     {device}")
    print()

    results = process_camera(
        session_dir,
        cam_id,
        cfg_path,
        ckpt_path,
        device,
        bbox_xyxy=args.bbox,
        foreground_rejection=not args.disable_foreground_rejection,
        reject_close_depth_m=args.reject_close_depth_m,
        reject_max_body_area_ratio=args.reject_max_body_area_ratio,
        pose_model=args.pose_model,
    )

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
