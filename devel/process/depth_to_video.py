#!/usr/bin/env python3
"""
Depth HDF5 → MP4 Converter

Converts depth.h5 files (recorded by felfeci_record_session.py) into
colorized MP4 videos using the TURBO colormap for easy visual inspection.

Supports both old (float32 meters) and new (uint16 z16_raw) depth formats.

Usage examples:
    # Convert all cameras in a session
    python depth_to_video.py ./recordings/session_YYYYMMDD_HHMMSS

    # Convert only cam1
    python depth_to_video.py ./recordings/session_YYYYMMDD_HHMMSS --cam 1

    # Convert a single depth.h5 file
    python depth_to_video.py ./recordings/session_YYYYMMDD_HHMMSS/cam1/depth.h5

    # Custom FPS, max-depth, and output directory
    python depth_to_video.py ./recordings/session_YYYYMMDD_HHMMSS --fps 30 --max-depth 3.0 --output-dir ./my_videos

    # Without colorbar overlay
    python depth_to_video.py ./recordings/session_YYYYMMDD_HHMMSS --no-colorbar
"""

import argparse
import cv2
import h5py
import json
import numpy as np
import os
import sys
from pathlib import Path


def load_depth_h5(h5_path):
    """Load depth data from HDF5, returning (hf, dataset, depth_scale, is_metric).

    Handles both storage formats:
      - uint16 z16_raw:  needs depth_scale multiplication for meters
      - float32 meters:  already in meters (legacy)
    """
    hf = h5py.File(h5_path, 'r')
    ds = hf['depth']

    unit = hf.attrs.get('unit', 'unknown')
    depth_scale = float(hf.attrs.get('depth_scale_meters_per_unit', 0.0))

    if isinstance(unit, bytes):
        unit = unit.decode()

    is_metric = (unit == 'meters')
    return hf, ds, depth_scale, is_metric


def depth_to_meters(frame, depth_scale, is_metric):
    """Convert a single depth frame to float32 meters."""
    if is_metric:
        return frame.astype(np.float32)
    else:
        return frame.astype(np.float32) * depth_scale


def colorize_depth(depth_meters, min_depth=0.0, max_depth=None,
                   colormap=cv2.COLORMAP_TURBO, add_colorbar=True):
    """Convert a metric depth frame to a colorized BGR image.

    Args:
        depth_meters: (H, W) float32 array in meters
        min_depth: clip minimum depth
        max_depth: clip maximum depth (auto if None)
        colormap: OpenCV colormap constant
        add_colorbar: whether to include a colorbar strip on the right

    Returns:
        (H, W[+60], 3) uint8 BGR image
    """
    valid_mask = depth_meters > 0

    if max_depth is None:
        if valid_mask.any():
            max_depth = float(np.percentile(depth_meters[valid_mask], 98))
        else:
            max_depth = 5.0

    max_depth = max(max_depth, min_depth + 0.01)

    clipped = np.clip(depth_meters, min_depth, max_depth)
    normalized = ((clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    normalized[~valid_mask] = 0

    colored = cv2.applyColorMap(normalized, colormap)
    colored[~valid_mask] = [0, 0, 0]

    if not add_colorbar:
        return colored

    # Colorbar on the right
    h, w = colored.shape[:2]
    bar_w = 60
    bar = np.zeros((h, bar_w, 3), dtype=np.uint8)

    strip_x = 8
    strip_w = 16
    for y in range(h):
        val = int(255 * (1.0 - y / max(h - 1, 1)))
        bar[y, strip_x:strip_x + strip_w] = cv2.applyColorMap(
            np.array([[val]], dtype=np.uint8), colormap)[0, 0]

    cv2.putText(bar, f"{min_depth:.1f}m", (strip_x, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(bar, f"{max_depth:.1f}m", (strip_x, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    mid = (min_depth + max_depth) / 2
    cv2.putText(bar, f"{mid:.1f}m", (strip_x, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return np.hstack([colored, bar])


def find_depth_files(session_dir):
    """Find all cam*/depth.h5 files in a session directory."""
    session = Path(session_dir)
    files = {}
    for cam_dir in sorted(session.iterdir()):
        if cam_dir.is_dir() and cam_dir.name.startswith('cam'):
            h5_path = cam_dir / 'depth.h5'
            if h5_path.exists():
                cam_num = int(cam_dir.name.replace('cam', ''))
                files[cam_num] = str(h5_path)
    return files


def get_session_fps(session_dir):
    """Try to read the recording FPS from metadata.json."""
    meta_path = os.path.join(session_dir, 'metadata.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return int(meta.get('fps', 0))
        except (json.JSONDecodeError, ValueError):
            pass
    return 0


def convert_h5_to_mp4(h5_path, output_path, fps=30, max_depth=None,
                       add_colorbar=True, cam_label=None):
    """Convert a single depth.h5 file to an MP4 video.

    Args:
        h5_path: path to the depth.h5 file
        output_path: path for the output .mp4 file
        fps: video frame rate
        max_depth: max depth for colormap (auto if None)
        add_colorbar: include colorbar in the video
        cam_label: optional label overlay (e.g. "cam1")

    Returns:
        number of frames written
    """
    hf, ds, depth_scale, is_metric = load_depth_h5(h5_path)
    total = ds.shape[0]

    if total == 0:
        print(f"  (empty dataset, skipping)")
        hf.close()
        return 0

    fmt_name = 'meters' if is_metric else 'z16_raw'
    print(f"  Frames: {total}, dtype: {ds.dtype}, format: {fmt_name}, "
          f"scale: {depth_scale}")

    # Pre-compute a global max_depth for consistent colormap across all frames
    if max_depth is None:
        # Sample up to 10 evenly-spaced frames to estimate range
        sample_indices = np.linspace(0, total - 1, min(10, total), dtype=int)
        all_valid = []
        for idx in sample_indices:
            raw = ds[idx]
            meters = depth_to_meters(raw, depth_scale, is_metric)
            valid = meters[meters > 0]
            if valid.size > 0:
                all_valid.append(np.percentile(valid, 98))
        if all_valid:
            max_depth = float(np.max(all_valid))
        else:
            max_depth = 5.0

    # Get frame dimensions from first frame to set up writer
    first_raw = ds[0]
    first_meters = depth_to_meters(first_raw, depth_scale, is_metric)
    first_colored = colorize_depth(first_meters, max_depth=max_depth,
                                   add_colorbar=add_colorbar)
    frame_h, frame_w = first_colored.shape[:2]

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        print(f"  Error: Could not open video writer for {output_path}")
        hf.close()
        return 0

    # Write frames
    for i in range(total):
        raw = ds[i]
        meters = depth_to_meters(raw, depth_scale, is_metric)
        colored = colorize_depth(meters, max_depth=max_depth,
                                 add_colorbar=add_colorbar)

        # Optional label overlay
        if cam_label:
            info = f"{cam_label} [{i}/{total}]"
            valid = meters[meters > 0]
            if valid.size > 0:
                info += f" {valid.min():.2f}-{valid.max():.2f}m"
            cv2.putText(colored, info, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(colored)

        # Progress indicator every 10%
        if total >= 10 and (i + 1) % max(1, total // 10) == 0:
            pct = (i + 1) * 100 // total
            print(f"    {pct}% ({i + 1}/{total})", end='\r')

    writer.release()
    hf.close()

    print(f"  Saved: {output_path}  ({total} frames, {total/fps:.1f}s @ {fps}fps)")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Convert depth.h5 files to colorized MP4 videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'path',
        help='Session directory or specific depth.h5 file',
    )
    parser.add_argument(
        '--cam', type=int, default=None,
        help='Camera number to convert (1-4). Default: all cameras',
    )
    parser.add_argument(
        '--fps', type=int, default=None,
        help='Video FPS. Auto-detected from metadata.json if not set',
    )
    parser.add_argument(
        '--max-depth', type=float, default=None,
        help='Maximum depth for colormap (meters). Auto if not set',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for MP4 files. Default: same as session dir',
    )
    parser.add_argument(
        '--no-colorbar', action='store_true',
        help='Omit the colorbar from the video',
    )
    parser.add_argument(
        '--no-label', action='store_true',
        help='Omit frame info text overlay',
    )

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: {args.path} does not exist")
        sys.exit(1)

    # Handle single h5 file
    if args.path.endswith('.h5'):
        h5_path = args.path
        session_dir = str(Path(h5_path).parent.parent)
        cam_name = Path(h5_path).parent.name
        cam_num = int(cam_name.replace('cam', ''))
        depth_files = {cam_num: h5_path}
    else:
        session_dir = args.path
        depth_files = find_depth_files(session_dir)

    if not depth_files:
        print(f"No depth.h5 files found in {args.path}")
        sys.exit(1)

    # Apply camera filter
    if args.cam is not None:
        depth_files = {k: v for k, v in depth_files.items() if k == args.cam}
        if not depth_files:
            print(f"No depth.h5 found for cam{args.cam}")
            sys.exit(1)

    # Determine FPS
    fps = args.fps
    if fps is None:
        fps = get_session_fps(session_dir)
    if not fps:
        fps = 30
        print(f"Could not detect FPS from metadata, using default: {fps}")

    # Output directory
    output_dir = args.output_dir or session_dir

    add_colorbar = not args.no_colorbar
    add_label = not args.no_label

    print(f"Converting depth.h5 → MP4")
    print(f"  Session : {session_dir}")
    print(f"  FPS     : {fps}")
    print(f"  Max depth: {args.max_depth or 'auto'}")
    print(f"  Output  : {output_dir}")
    print(f"  Cameras : {sorted(depth_files.keys())}")
    print()

    total_frames = 0
    for cam_num, h5_path in sorted(depth_files.items()):
        cam_label = f"cam{cam_num}" if add_label else None
        out_name = f"cam{cam_num}_depth.mp4"
        out_path = os.path.join(output_dir, out_name)

        print(f"--- cam{cam_num}: {h5_path} ---")
        n = convert_h5_to_mp4(
            h5_path, out_path, fps=fps, max_depth=args.max_depth,
            add_colorbar=add_colorbar, cam_label=cam_label,
        )
        total_frames += n
        print()

    print(f"Done! Converted {total_frames} total frames from "
          f"{len(depth_files)} camera(s).")


if __name__ == '__main__':
    main()
