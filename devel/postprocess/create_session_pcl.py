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
import csv
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))


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


def find_color_video(session_dir, cam_id, cam_meta):
    storage_file = cam_meta.get("color_storage", {}).get("file")
    candidates = []
    if storage_file:
        candidates.append(os.path.join(session_dir, storage_file))
    cam_dir = os.path.join(session_dir, f"cam{cam_id}")
    candidates.extend(
        [
            os.path.join(cam_dir, "color.mkv"),
            os.path.join(cam_dir, "color.mp4"),
        ]
    )
    return next((path for path in candidates if os.path.exists(path)), None)


def _deproject_pixels(pixel_x, pixel_y, depth_meters, intrinsics):
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    x = (pixel_x - ppx) * depth_meters / fx
    y = (pixel_y - ppy) * depth_meters / fy
    return np.stack([x, y, depth_meters], axis=1)


def raw_depth_points_in_color_camera(
    raw_depth,
    color_rgb,
    cam_meta,
    step,
    min_z,
    max_z,
):
    calibration = cam_meta.get("calibration", {})
    depth_intrinsics = calibration.get("depth_intrinsics")
    color_intrinsics = calibration.get("color_intrinsics") or cam_meta.get("intrinsics")
    depth_to_color = calibration.get("depth_to_color_extrinsics")
    if not depth_intrinsics or not color_intrinsics or not depth_to_color:
        raise KeyError(
            "metadata.json lacks depth intrinsics, color intrinsics, or "
            "depth-to-color extrinsics required for raw-depth geometry."
        )

    depth_scale = float(
        calibration.get(
            "depth_scale_meters_per_unit",
            cam_meta.get("depth_storage", {}).get(
                "depth_scale_meters_per_unit", 0.001
            ),
        )
    )
    depth_meters = raw_depth.astype(np.float32) * depth_scale
    height, width = depth_meters.shape
    pixel_x, pixel_y = np.meshgrid(
        np.arange(0, width, step, dtype=np.float64),
        np.arange(0, height, step, dtype=np.float64),
    )
    sampled_depth = depth_meters[pixel_y.astype(int), pixel_x.astype(int)]
    valid = (sampled_depth >= min_z) & (sampled_depth <= max_z)
    points_depth = _deproject_pixels(
        pixel_x[valid], pixel_y[valid], sampled_depth[valid], depth_intrinsics
    )

    rotation = np.asarray(
        depth_to_color["rotation"], dtype=np.float64
    ).reshape(3, 3, order="F")
    translation = np.asarray(
        depth_to_color["translation"], dtype=np.float64
    ).reshape(3)
    points_color = (rotation @ points_depth.T).T + translation

    positive = points_color[:, 2] > 0
    points_color = points_color[positive]
    color_x = (
        float(color_intrinsics["fx"]) * points_color[:, 0] / points_color[:, 2]
        + float(color_intrinsics["ppx"])
    )
    color_y = (
        float(color_intrinsics["fy"]) * points_color[:, 1] / points_color[:, 2]
        + float(color_intrinsics["ppy"])
    )
    color_u = np.rint(color_x).astype(np.int32)
    color_v = np.rint(color_y).astype(np.int32)
    color_height, color_width = color_rgb.shape[:2]
    inside = (
        (color_u >= 0)
        & (color_u < color_width)
        & (color_v >= 0)
        & (color_v < color_height)
    )
    points_color = points_color[inside]
    colors = color_rgb[color_v[inside], color_u[inside]]
    return points_color, colors


def extract_camera_pcl(
    session_dir,
    cam_id,
    meta,
    intrinsics_data=None,
    frame_idx=100,
    depth_frame_idx=None,
    step=3,
    min_z=0.3,
    max_z=3.5,
):
    """Extract 3D points (N, 3) and RGB colors (N, 3) for given camera and frame."""
    cam_dir = os.path.join(session_dir, f"cam{cam_id}")
    depth_path = os.path.join(cam_dir, "depth.mkv")
    cam_meta = meta['cameras'][str(cam_id)]
    color_path = find_color_video(session_dir, cam_id, cam_meta)

    if not color_path or not os.path.exists(depth_path):
        print(f"Warning: Missing files for cam{cam_id}")
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None

    cap_c = cv2.VideoCapture(color_path)
    cap_c.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret_c, color_bgr = cap_c.read()
    cap_c.release()

    if depth_frame_idx is None:
        depth_frame_idx = frame_idx
    ret_d, raw_depth = read_lossless_depth_frame(depth_path, depth_frame_idx)

    if not ret_c or not ret_d:
        print(
            f"Error: Could not read color/depth frames "
            f"{frame_idx}/{depth_frame_idx} from cam{cam_id}"
        )
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    storage = cam_meta.get("depth_storage", {})
    raw_depth_geometry = (
        storage.get("aligned_to", "depth") == "depth"
        or storage.get("alignment_mode") == "none_raw_depth"
    )
    if raw_depth_geometry:
        points, colors = raw_depth_points_in_color_camera(
            raw_depth,
            color_rgb,
            cam_meta,
            step=step,
            min_z=min_z,
            max_z=max_z,
        )
    else:
        intr = (
            cam_meta.get("calibration", {}).get("color_intrinsics")
            or cam_meta["intrinsics"]
        )
        depth_scale = float(
            storage.get("depth_scale_meters_per_unit", 0.001)
        )
        depth_meters = raw_depth.astype(np.float32) * depth_scale
        height, width = depth_meters.shape
        pixel_x, pixel_y = np.meshgrid(
            np.arange(0, width, step),
            np.arange(0, height, step),
        )
        sampled_depth = depth_meters[pixel_y, pixel_x]
        valid = (sampled_depth >= min_z) & (sampled_depth <= max_z)
        points = _deproject_pixels(
            pixel_x[valid],
            pixel_y[valid],
            sampled_depth[valid],
            intr,
        )
        colors = color_rgb[pixel_y[valid], pixel_x[valid]]

    return points, colors, color_rgb


def read_timestamp_csv(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    int(row["frame_idx"]),
                    float(row["host_timestamp_ms"]),
                )
            )
    return rows


def nearest_frame(rows, target_host_ms):
    if not rows:
        return None, float("inf")
    frame_index, host_ms = min(rows, key=lambda item: abs(item[1] - target_host_ms))
    return frame_index, abs(host_ms - target_host_ms)


def synchronized_frame_indices(session_dir, metadata, reference_camera, reference_frame):
    reference_meta = metadata["cameras"][str(reference_camera)]
    reference_path = os.path.join(
        session_dir, reference_meta["color_timestamp_file"]
    )
    reference_rows = read_timestamp_csv(reference_path)
    reference_matches = [
        host_ms for frame_index, host_ms in reference_rows
        if frame_index == reference_frame
    ]
    if not reference_matches:
        raise IndexError(
            f"Reference color frame {reference_frame} is absent from {reference_path}."
        )
    target_host_ms = reference_matches[0]

    result = {}
    for camera_text, camera_meta in metadata["cameras"].items():
        camera_id = int(camera_text)
        color_rows = read_timestamp_csv(
            os.path.join(session_dir, camera_meta["color_timestamp_file"])
        )
        depth_rows = read_timestamp_csv(
            os.path.join(session_dir, camera_meta["depth_timestamp_file"])
        )
        color_frame, color_delta = nearest_frame(color_rows, target_host_ms)
        depth_frame, depth_delta = nearest_frame(depth_rows, target_host_ms)
        result[camera_id] = {
            "color_frame": color_frame,
            "depth_frame": depth_frame,
            "color_delta_ms": color_delta,
            "depth_delta_ms": depth_delta,
        }
    return result


def camera_to_reference_transform(calibration, camera_id, reference_camera):
    if camera_id == reference_camera:
        return np.eye(3), np.zeros(3)
    explicit_key = f"T_cam{camera_id}_to_ref"
    if explicit_key in calibration:
        transform = np.asarray(calibration[explicit_key], dtype=np.float64)
        return transform[:3, :3], transform[:3, 3]

    rotation_ref_camera = np.asarray(
        calibration[f"R_{camera_id}_to_ref"], dtype=np.float64
    )
    translation_ref_camera = np.asarray(
        calibration[f"t_{camera_id}_to_ref"], dtype=np.float64
    ).reshape(3)
    rotation_camera_ref = rotation_ref_camera.T
    translation_camera_ref = -rotation_camera_ref @ translation_ref_camera
    return rotation_camera_ref, translation_camera_ref


def main():
    parser = argparse.ArgumentParser(description="Create PCL files and presentation slides visuals.")
    parser.add_argument("--session-dir", required=True, help="Path to session directory")
    parser.add_argument("--calib", default=None, help="Path to multicam_calibration.npz")
    parser.add_argument("--frame", type=int, default=100, help="Frame index to extract")
    parser.add_argument("--step", type=int, default=3, help="Pixel subsampling step (higher=faster, lower=denser)")
    parser.add_argument("--out-dir", default=None, help="Output folder for PLY and PNG files")
    parser.add_argument(
        "--camera-ids",
        type=int,
        nargs="+",
        default=None,
        help="Cameras to fuse. Defaults to every camera in metadata.json.",
    )
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
        calib_data = load_npz_dict(calib_path)
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
    camera_ids = args.camera_ids or sorted(
        int(camera_id) for camera_id in meta["cameras"]
    )
    reference_camera = (
        int(np.asarray(calib_data["ref_camera"]).item())
        if calib_data is not None and "ref_camera" in calib_data
        else camera_ids[0]
    )
    synchronized = synchronized_frame_indices(
        session_dir, meta, reference_camera, args.frame
    )
    print(
        f"\n--- Extracting timestamp-synchronized point clouds at "
        f"camera {reference_camera} color frame {args.frame} ---"
    )

    local_clouds = {}
    world_clouds = {}
    camera_origins = {}
    for camera_id in camera_ids:
        frame_info = synchronized[camera_id]
        print(
            f"  cam{camera_id}: color={frame_info['color_frame']} "
            f"(delta {frame_info['color_delta_ms']:.2f}ms), "
            f"depth={frame_info['depth_frame']} "
            f"(delta {frame_info['depth_delta_ms']:.2f}ms)"
        )
        points, colors, rgb = extract_camera_pcl(
            session_dir,
            camera_id,
            meta,
            intrinsics_data=intrinsics_data,
            frame_idx=frame_info["color_frame"],
            depth_frame_idx=frame_info["depth_frame"],
            step=args.step,
        )
        local_clouds[camera_id] = (points, colors, rgb)
        write_ply(
            os.path.join(out_dir, f"cam{camera_id}_pointcloud.ply"),
            points,
            colors,
        )
        if calib_data is not None and len(points):
            rotation, translation = camera_to_reference_transform(
                calib_data, camera_id, reference_camera
            )
            points_world = (rotation @ points.T).T + translation
            camera_origins[camera_id] = translation
        else:
            points_world = points
            camera_origins[camera_id] = np.zeros(3)
        world_clouds[camera_id] = (points_world, colors)
        write_ply(
            os.path.join(
                out_dir,
                f"cam{camera_id}_pointcloud_in_cam{reference_camera}_frame.ply",
            ),
            points_world,
            colors,
        )

    nonempty_points = [
        world_clouds[camera_id][0]
        for camera_id in camera_ids
        if len(world_clouds[camera_id][0])
    ]
    nonempty_colors = [
        world_clouds[camera_id][1]
        for camera_id in camera_ids
        if len(world_clouds[camera_id][1])
    ]
    combined_pts = (
        np.vstack(nonempty_points) if nonempty_points else np.empty((0, 3))
    )
    combined_col = (
        np.vstack(nonempty_colors)
        if nonempty_colors
        else np.empty((0, 3), dtype=np.uint8)
    )
    glob_ply_path = os.path.join(out_dir, "global_combined_pointcloud.ply")
    write_ply(glob_ply_path, combined_pts, combined_col)

    # 4. Create Presentation Slide Visualization (PNG)
    print("\n[Visualization] Generating presentation slide summary diagram...")
    fig = plt.figure(figsize=(20, 12), facecolor='#1a1a2e')

    n_sample_small = 25000
    n_sample_large = 50000
    display_camera_ids = camera_ids[:2]
    while len(display_camera_ids) < 2:
        display_camera_ids.append(display_camera_ids[0])
    first_camera, second_camera = display_camera_ids
    pts1, col1, rgb1 = local_clouds[first_camera]
    pts2, col2, rgb2 = local_clouds[second_camera]

    # --- Row 1, Col 1: Cam 1 RGB ---
    ax1 = fig.add_subplot(2, 3, 1)
    if rgb1 is not None:
        ax1.imshow(rgb1)
    ax1.set_title(f"Cam {first_camera} — RGB", fontsize=13, fontweight='bold', color='white')
    ax1.axis('off')
    ax1.set_facecolor('#1a1a2e')

    # --- Row 2, Col 1: Cam 2 RGB ---
    ax2 = fig.add_subplot(2, 3, 4)
    if rgb2 is not None:
        ax2.imshow(rgb2)
    ax2.set_title(f"Cam {second_camera} — RGB", fontsize=13, fontweight='bold', color='white')
    ax2.axis('off')
    ax2.set_facecolor('#1a1a2e')

    # --- Row 1, Col 2: Cam 1 PCL front view (X vs -Y => people stand upright) ---
    ax3 = fig.add_subplot(2, 3, 2)
    ax3.set_facecolor('#0f0f23')
    if len(pts1) > 0:
        sub = np.random.choice(len(pts1), min(n_sample_small, len(pts1)), replace=False)
        ax3.scatter(pts1[sub, 0], -pts1[sub, 1], c=col1[sub]/255.0, s=0.4, alpha=0.8)
    ax3.set_title(f"Cam {first_camera} Local\n{len(pts1):,} points", fontsize=12, fontweight='bold', color='#66ccff')
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
    ax4.set_title(f"Cam {second_camera} Local\n{len(pts2):,} points", fontsize=12, fontweight='bold', color='#ff9966')
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
        marker_colors = ("#66ccff", "#ff9966", "#ffee66", "#bb88ff")
        for position, camera_id in enumerate(camera_ids):
            origin = camera_origins[camera_id]
            suffix = " (ref)" if camera_id == reference_camera else ""
            ax5.plot(
                origin[0],
                origin[2],
                'v',
                color=marker_colors[position % len(marker_colors)],
                markersize=11,
                label=f"Cam {camera_id}{suffix}",
                zorder=10,
            )
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
