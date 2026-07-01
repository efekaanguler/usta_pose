#!/usr/bin/env python3
"""
Create 3D Point Cloud (PCL) files and visualizations from recorded RGB-D sessions.

Extracts colored 3D point clouds from Cam 1 and Cam 2, applies multi-camera calibration
transformation to align them in the global world frame (Glob PCL), and generates:
  1. cam1_pointcloud.ply
  2. cam2_pointcloud.ply
  3. global_combined_pointcloud.ply
  4. pcl_summary_visualization.png (multi-view projection for presentation slides)

Usage:
    python3 devel/postprocess/create_session_pcl.py \
        --session-dir /path/to/session_YYYYMMDD_HHMMSS \
        --calib /path/to/multicam_calibration.npz \
        --frame 100 --step 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add revised_process to path for DepthProjector
script_dir = os.path.dirname(os.path.abspath(__file__))
revised_dir = os.path.join(os.path.dirname(script_dir), "revised_process")
if revised_dir not in sys.path:
    sys.path.append(revised_dir)

try:
    from extract_pose_independent import DepthProjector
except ImportError:
    # Fallback if standalone
    DepthProjector = None


def write_ply(filename, points, colors):
    """Write ASCII PLY file from 3D points (N, 3) and RGB colors (N, 3)."""
    n_points = len(points)
    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(n_points):
            p = points[i]
            c = colors[i]
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    print(f"[PLY Export] Saved {n_points:,} points -> {filename}")


def read_lossless_depth_frame(depth_path, frame_idx):
    """Read a specific 16-bit depth frame losslessly using imageio_ffmpeg or fallback to cv2."""
    try:
        import imageio_ffmpeg as iio_ff
        rgen = iio_ff.read_frames(depth_path, pix_fmt='gray16le', bits_per_pixel=16)
        meta = next(rgen)
        w, h = meta['size']
        curr = 0
        raw_depth = None
        for raw_bytes in rgen:
            if curr == frame_idx:
                raw_depth = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(h, w).copy()
                break
            curr += 1
        rgen.close()
        if raw_depth is not None:
            return True, raw_depth
    except Exception as e:
        pass

    cap_d = cv2.VideoCapture(depth_path, cv2.CAP_ANY)
    cap_d.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap_d.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret_d, raw_depth = cap_d.read()
    cap_d.release()
    return ret_d, raw_depth


def load_npz_dict(path):
    with np.load(path) as data:
        return {key: np.array(data[key]) for key in data.files}


def default_recordings_dir():
    devel_dir = os.path.dirname(script_dir)
    return os.path.join(devel_dir, "record", "recordings")


def find_multicam_calibration(session_dir, explicit_path=None):
    parent_dir = os.path.dirname(session_dir)
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend([
        os.path.join(session_dir, "multicam_calibration.npz"),
        os.path.join(session_dir, "calib_data", "multicam_calibration.npz"),
        os.path.join(parent_dir, "multicam_calibration.npz"),
        os.path.join(parent_dir, "calib_data", "multicam_calibration.npz"),
        os.path.join(default_recordings_dir(), "multicam_calibration.npz"),
    ])
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def find_intrinsics_npz(session_dir, preferred_calib=None):
    parent_dir = os.path.dirname(session_dir)
    recordings_dir = default_recordings_dir()
    candidates = [
        preferred_calib,
        os.path.join(session_dir, "multicam_calibration.npz"),
        os.path.join(session_dir, "calib_data", "multicam_calibration.npz"),
        os.path.join(session_dir, "calib_data", "master_intrinsics.npz"),
        os.path.join(parent_dir, "multicam_calibration.npz"),
        os.path.join(parent_dir, "calib_data", "master_intrinsics.npz"),
        os.path.join(recordings_dir, "multicam_calibration.npz"),
        os.path.join(recordings_dir, "calib_data", "master_intrinsics.npz"),
    ]

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not os.path.exists(candidate):
            continue
        seen.add(candidate)
        try:
            with np.load(candidate) as data:
                if "K1" in data:
                    return candidate
        except Exception:
            continue
    return None


def intrinsics_from_npz(intrinsics_data, cam_id):
    if intrinsics_data is None:
        return None
    K = intrinsics_data.get(f"K{cam_id}")
    if K is None:
        return None
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "ppx": float(K[0, 2]),
        "ppy": float(K[1, 2]),
    }


def extract_camera_pcl(session_dir, cam_id, meta, intrinsics_data=None, frame_idx=100, step=3, min_z=0.3, max_z=3.5):
    """Extract 3D points (N, 3) and RGB colors (N, 3) for given camera and frame."""
    cam_dir = os.path.join(session_dir, f"cam{cam_id}")
    color_path = os.path.join(cam_dir, "color.mp4")
    depth_path = os.path.join(cam_dir, "depth.mkv")

    if not os.path.exists(color_path) or not os.path.exists(depth_path):
        print(f"Warning: Missing files for cam{cam_id}")
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None

    cap_c = cv2.VideoCapture(color_path)
    cap_c.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret_c, color_bgr = cap_c.read()
    cap_c.release()

    ret_d, raw_depth = read_lossless_depth_frame(depth_path, frame_idx)

    if not ret_c or not ret_d:
        print(f"Error: Could not read frame {frame_idx} from cam{cam_id}")
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    cam_meta = meta['cameras'][str(cam_id)]
    intr = intrinsics_from_npz(intrinsics_data, cam_id)
    if intr is None:
        intr = cam_meta.get('calibration', {}).get('color_intrinsics') or cam_meta['intrinsics']
    fx, fy, ppx, ppy = intr['fx'], intr['fy'], intr['ppx'], intr['ppy']

    if DepthProjector is not None:
        dp = DepthProjector(cam_meta)
        aligned_depth = dp.depth_for_color(raw_depth)
        depth_scale = dp.depth_scale
    else:
        aligned_depth = raw_depth
        depth_scale = float(cam_meta.get('depth_storage', {}).get('depth_scale_meters_per_unit', 0.001))

    z_meters = aligned_depth.astype(np.float32) * depth_scale
    h, w = z_meters.shape

    u, v = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    z = z_meters[v, u]

    valid = (z >= min_z) & (z <= max_z)
    z_val = z[valid]
    u_val = u[valid]
    v_val = v[valid]

    x_val = (u_val - ppx) * z_val / fx
    y_val = (v_val - ppy) * z_val / fy

    points = np.stack([x_val, y_val, z_val], axis=1)
    colors = color_rgb[v_val, u_val]

    return points, colors, color_rgb


def main():
    parser = argparse.ArgumentParser(description="Create PCL files and presentation slides visuals.")
    parser.add_argument("--session-dir", required=True, help="Path to session directory")
    parser.add_argument("--calib", default=None, help="Path to multicam_calibration.npz")
    parser.add_argument("--frame", type=int, default=100, help="Frame index to extract")
    parser.add_argument("--step", type=int, default=3, help="Pixel subsampling step (higher=faster, lower=denser)")
    parser.add_argument("--out-dir", default=None, help="Output folder for PLY and PNG files")
    args = parser.parse_args()

    session_dir = os.path.abspath(args.session_dir)
    meta_path = os.path.join(session_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata.json in {session_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    calib_path = find_multicam_calibration(session_dir, args.calib)

    if calib_path and os.path.exists(calib_path):
        print(f"[Calibration] Loaded {calib_path}")
        calib_data = np.load(calib_path)
    else:
        print("\n" + "!" * 80)
        print("[UYARI] Session klasörü içinde 'multicam_calibration.npz' BULUNAMADI!")
        print("[UYARI] Tekil kameralar (cam1/cam2) doğru üretilecek ANCAK birleşik bulut")
        print("[UYARI] ('global_combined_pointcloud.ply') hizalanmadan üst üste binecektir!")
        print("!" * 80 + "\n")
        calib_data = None

    intrinsics_path = find_intrinsics_npz(session_dir, calib_path)
    if intrinsics_path:
        print(f"[Intrinsics] Using high-precision intrinsics from {intrinsics_path}")
        intrinsics_data = load_npz_dict(intrinsics_path)
    else:
        print("[Intrinsics] High-precision intrinsics not found; falling back to metadata.json")
        intrinsics_data = None

    out_dir = args.out_dir or os.path.join(session_dir, "pcl_output")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- Extracting Point Clouds for Frame {args.frame} ---")

    # 1. Extract Cam 1 PCL
    pts1, col1, rgb1 = extract_camera_pcl(
        session_dir, 1, meta, intrinsics_data=intrinsics_data,
        frame_idx=args.frame, step=args.step,
    )
    ply1_path = os.path.join(out_dir, "cam1_pointcloud.ply")
    write_ply(ply1_path, pts1, col1)

    # 2. Extract Cam 2 PCL
    pts2, col2, rgb2 = extract_camera_pcl(
        session_dir, 2, meta, intrinsics_data=intrinsics_data,
        frame_idx=args.frame, step=args.step,
    )
    ply2_path = os.path.join(out_dir, "cam2_pointcloud.ply")
    write_ply(ply2_path, pts2, col2)

    # 3. Transform & Combine (Task 2: 2 tane pcl calib dönüşümü)
    #
    # IMPORTANT: The NPZ keys "R_2_to_ref" / "t_2_to_ref" are misleadingly named.
    # multicam_calibrate.py stores the REF -> CAM transform there:
    #     P_cam = R_stored @ P_ref + t_stored
    # To convert cam2 points INTO the reference (world) frame we must INVERT:
    #     P_ref = R_stored^T @ P_cam  +  (- R_stored^T @ t_stored)
    # This matches exactly what resample_and_transform.py does in
    # load_camera_to_ref_transform().
    if calib_data is not None and len(pts2) > 0:
        R_ref_to_cam2 = np.asarray(calib_data["R_2_to_ref"], dtype=np.float64)
        t_ref_to_cam2 = np.asarray(calib_data["t_2_to_ref"], dtype=np.float64).reshape(3)
        R_cam2_to_world = R_ref_to_cam2.T
        t_cam2_to_world = -R_cam2_to_world @ t_ref_to_cam2
        pts2_world = (R_cam2_to_world @ pts2.T).T + t_cam2_to_world
        print(f"[Calibration] Applied inverted transform (cam2 -> world)")
    else:
        pts2_world = pts2
        t_cam2_to_world = None

    pts1_world = pts1  # Cam 1 is reference

    combined_pts = np.vstack([pts1_world, pts2_world])
    combined_col = np.vstack([col1, col2])
    # Track which camera each point came from for color-coded overlay
    cam_labels = np.array([1]*len(pts1_world) + [2]*len(pts2_world))
    glob_ply_path = os.path.join(out_dir, "global_combined_pointcloud.ply")
    write_ply(glob_ply_path, combined_pts, combined_col)

    # 4. Create Presentation Slide Visualization (PNG)
    print("\n[Visualization] Generating presentation slide summary diagram...")
    fig = plt.figure(figsize=(20, 12), facecolor='#1a1a2e')

    n_sample_small = 25000
    n_sample_large = 50000

    # --- Row 1, Col 1: Cam 1 RGB ---
    ax1 = fig.add_subplot(2, 3, 1)
    if rgb1 is not None:
        ax1.imshow(rgb1)
    ax1.set_title(f"Cam 1 — RGB (Frame {args.frame})", fontsize=13, fontweight='bold', color='white')
    ax1.axis('off')
    ax1.set_facecolor('#1a1a2e')

    # --- Row 2, Col 1: Cam 2 RGB ---
    ax2 = fig.add_subplot(2, 3, 4)
    if rgb2 is not None:
        ax2.imshow(rgb2)
    ax2.set_title(f"Cam 2 — RGB (Frame {args.frame})", fontsize=13, fontweight='bold', color='white')
    ax2.axis('off')
    ax2.set_facecolor('#1a1a2e')

    # --- Row 1, Col 2: Cam 1 PCL front view (X vs -Y => people stand upright) ---
    ax3 = fig.add_subplot(2, 3, 2)
    ax3.set_facecolor('#0f0f23')
    if len(pts1) > 0:
        sub = np.random.choice(len(pts1), min(n_sample_small, len(pts1)), replace=False)
        ax3.scatter(pts1[sub, 0], -pts1[sub, 1], c=col1[sub]/255.0, s=0.4, alpha=0.8)
    ax3.set_title(f"PCL 1 — Cam 1 Local\n{len(pts1):,} points", fontsize=12, fontweight='bold', color='#66ccff')
    ax3.set_xlabel("X (m)", color='white', fontsize=10)
    ax3.set_ylabel("Y (m, up)", color='white', fontsize=10)
    ax3.tick_params(colors='white')
    ax3.grid(True, linestyle='--', alpha=0.2, color='gray')

    # --- Row 2, Col 2: Cam 2 PCL front view ---
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.set_facecolor('#0f0f23')
    if len(pts2) > 0:
        sub = np.random.choice(len(pts2), min(n_sample_small, len(pts2)), replace=False)
        ax4.scatter(pts2[sub, 0], -pts2[sub, 1], c=col2[sub]/255.0, s=0.4, alpha=0.8)
    ax4.set_title(f"PCL 2 — Cam 2 Local\n{len(pts2):,} points", fontsize=12, fontweight='bold', color='#ff9966')
    ax4.set_xlabel("X (m)", color='white', fontsize=10)
    ax4.set_ylabel("Y (m, up)", color='white', fontsize=10)
    ax4.tick_params(colors='white')
    ax4.grid(True, linestyle='--', alpha=0.2, color='gray')

    # --- Right half: Global Combined PCL (top-down XZ bird's eye) ---
    ax5 = fig.add_subplot(1, 3, 3)
    ax5.set_facecolor('#0f0f23')
    if len(combined_pts) > 0:
        sub = np.random.choice(len(combined_pts), min(n_sample_large, len(combined_pts)), replace=False)
        ax5.scatter(
            combined_pts[sub, 0], combined_pts[sub, 2],
            c=combined_col[sub]/255.0, s=0.6, alpha=0.7,
        )
        # Add camera origin markers
        ax5.plot(0, 0, 'v', color='#66ccff', markersize=14, label='Cam 1 (ref)', zorder=10)
        if calib_data is not None and t_cam2_to_world is not None:
            ax5.plot(t_cam2_to_world[0], t_cam2_to_world[2], 'v', color='#ff9966', markersize=14, label='Cam 2', zorder=10)
        ax5.legend(loc='upper right', fontsize=10, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')

    ax5.set_title(
        f"Global Combined PCL (World Frame)\n"
        f"Aligned via Calibration Transform\n"
        f"Total: {len(combined_pts):,} points",
        fontsize=13, fontweight='bold', color='#44ff88',
    )
    ax5.set_xlabel("World X (m)", color='white', fontsize=11)
    ax5.set_ylabel("World Z (m)", color='white', fontsize=11)
    ax5.tick_params(colors='white')
    ax5.grid(True, linestyle='--', alpha=0.25, color='gray')

    plt.tight_layout()
    viz_path = os.path.join(out_dir, "pcl_summary_visualization.png")
    plt.savefig(viz_path, dpi=200, bbox_inches="tight", facecolor='#1a1a2e')
    plt.close()
    print(f"[Visualization] Saved presentation slide image -> {viz_path}")
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
