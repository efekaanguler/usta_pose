#!/usr/bin/env python3
"""
Depth Map Visualizer — colorized depth inspection for recorded sessions.

Reads depth.h5 files and produces colorized depth images using the TURBO
colormap for quick visual verification.

Supports both old (float32 meters) and new (uint16 z16_raw) depth formats.

Usage examples:
    # Export first 5 frames of all cameras as PNGs
    python visualize_depth.py ./recordings/session_YYYYMMDD_HHMMSS --export --num-frames 5

    # Export a specific frame from a specific camera
    python visualize_depth.py ./recordings/session_YYYYMMDD_HHMMSS --cam 1 --frame 42 --export

    # Interactive slider (requires display)
    python visualize_depth.py ./recordings/session_YYYYMMDD_HHMMSS --interactive

    # View a single depth.h5 file directly
    python visualize_depth.py ./recordings/session_YYYYMMDD_HHMMSS/cam1/depth.h5 --interactive
"""

import argparse
import cv2
import h5py
import numpy as np
import os
import sys
from pathlib import Path


def load_depth_h5(h5_path):
    """Load depth data from HDF5, returning (data_array, depth_scale, is_metric).

    Handles both storage formats:
      - uint16 z16_raw:  needs depth_scale multiplication for meters
      - float32 meters:  already in meters (legacy)
    """
    hf = h5py.File(h5_path, 'r')
    ds = hf['depth']

    # Detect format from attrs
    unit = hf.attrs.get('unit', 'unknown')
    depth_scale = float(hf.attrs.get('depth_scale_meters_per_unit', 0.0))

    if isinstance(unit, bytes):
        unit = unit.decode()

    is_metric = (unit == 'meters')  # legacy float32 format

    return hf, ds, depth_scale, is_metric


def depth_to_meters(frame, depth_scale, is_metric):
    """Convert a single depth frame to float32 meters."""
    if is_metric:
        return frame.astype(np.float32)
    else:
        return frame.astype(np.float32) * depth_scale


def colorize_depth(depth_meters, min_depth=0.0, max_depth=None, colormap=cv2.COLORMAP_TURBO):
    """Convert a metric depth frame to a colorized BGR image with colorbar.

    Args:
        depth_meters: (H, W) float32 array in meters
        min_depth: clip minimum depth
        max_depth: clip maximum depth (auto if None)
        colormap: OpenCV colormap constant

    Returns:
        (H, W+60, 3) uint8 BGR image with colorbar on the right
    """
    valid_mask = depth_meters > 0

    if max_depth is None:
        if valid_mask.any():
            max_depth = float(np.percentile(depth_meters[valid_mask], 98))
        else:
            max_depth = 5.0  # fallback

    max_depth = max(max_depth, min_depth + 0.01)

    # Normalize to 0-255
    clipped = np.clip(depth_meters, min_depth, max_depth)
    normalized = ((clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    # Invalid pixels (depth=0) → black
    normalized[~valid_mask] = 0

    # Apply colormap
    colored = cv2.applyColorMap(normalized, colormap)
    colored[~valid_mask] = [0, 0, 0]

    # Add colorbar on the right
    h, w = colored.shape[:2]
    bar_w = 60
    bar = np.zeros((h, bar_w, 3), dtype=np.uint8)

    # Gradient strip
    strip_x = 8
    strip_w = 16
    for y in range(h):
        val = int(255 * (1.0 - y / max(h - 1, 1)))
        bar[y, strip_x:strip_x + strip_w] = cv2.applyColorMap(
            np.array([[val]], dtype=np.uint8), colormap)[0, 0]

    # Labels
    cv2.putText(bar, f"{min_depth:.1f}m", (strip_x, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(bar, f"{max_depth:.1f}m", (strip_x, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    mid = (min_depth + max_depth) / 2
    cv2.putText(bar, f"{mid:.1f}m", (strip_x, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    result = np.hstack([colored, bar])
    return result


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


def export_frames(session_dir, cam_filter=None, frame_idx=None, num_frames=5, max_depth=None):
    """Export colorized depth frames as PNGs."""
    # Handle direct h5 file path
    if session_dir.endswith('.h5'):
        h5_path = session_dir
        session_dir = str(Path(h5_path).parent.parent)
        cam_name = Path(h5_path).parent.name
        cam_num = int(cam_name.replace('cam', ''))
        depth_files = {cam_num: h5_path}
    else:
        depth_files = find_depth_files(session_dir)

    if not depth_files:
        print(f"No depth.h5 files found in {session_dir}")
        return

    # Apply camera filter
    if cam_filter is not None:
        depth_files = {k: v for k, v in depth_files.items() if k == cam_filter}

    # Output directory
    out_dir = os.path.join(session_dir, 'depth_viz')
    os.makedirs(out_dir, exist_ok=True)

    for cam_num, h5_path in sorted(depth_files.items()):
        print(f"\n--- cam{cam_num}: {h5_path} ---")
        hf, ds, depth_scale, is_metric = load_depth_h5(h5_path)
        total = ds.shape[0]
        print(f"  Total frames: {total}, dtype: {ds.dtype}, "
              f"format: {'meters' if is_metric else 'z16_raw'}, scale: {depth_scale}")

        if total == 0:
            print("  (empty dataset, skipping)")
            hf.close()
            continue

        # Determine frames to export
        if frame_idx is not None:
            indices = [frame_idx] if frame_idx < total else []
        else:
            indices = list(range(min(num_frames, total)))

        for idx in indices:
            raw = ds[idx]
            meters = depth_to_meters(raw, depth_scale, is_metric)
            colored = colorize_depth(meters, max_depth=max_depth)

            # Add frame info text
            info = f"cam{cam_num} frame:{idx}"
            valid = meters[meters > 0]
            if valid.size > 0:
                info += f"  min:{valid.min():.2f}m  max:{valid.max():.2f}m  mean:{valid.mean():.2f}m"
            cv2.putText(colored, info, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            fname = f"cam{cam_num}_frame{idx:06d}.png"
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, colored)
            print(f"  Saved: {out_path}")

        hf.close()

    print(f"\nAll exports saved to: {out_dir}")


def interactive_viewer(session_dir, cam_filter=None, max_depth=None):
    """Interactive depth viewer with frame slider."""
    # Handle direct h5 file path
    if session_dir.endswith('.h5'):
        h5_path = session_dir
        cam_name = Path(h5_path).parent.name
        cam_num = int(cam_name.replace('cam', ''))
        depth_files = {cam_num: h5_path}
    else:
        depth_files = find_depth_files(session_dir)

    if not depth_files:
        print(f"No depth.h5 files found in {session_dir}")
        return

    if cam_filter is not None:
        depth_files = {k: v for k, v in depth_files.items() if k == cam_filter}

    # Load all datasets (keep files open)
    cam_data = {}
    max_frames = 0
    for cam_num, h5_path in sorted(depth_files.items()):
        hf, ds, depth_scale, is_metric = load_depth_h5(h5_path)
        cam_data[cam_num] = (hf, ds, depth_scale, is_metric)
        max_frames = max(max_frames, ds.shape[0])

    if max_frames == 0:
        print("No frames found.")
        for hf, _, _, _ in cam_data.values():
            hf.close()
        return

    window_name = "Depth Viewer (Q=Quit, S=Save)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_frame = [0]  # mutable for callback

    def on_trackbar(val):
        current_frame[0] = val

    cv2.createTrackbar("Frame", window_name, 0, max_frames - 1, on_trackbar)

    print(f"Interactive viewer: {len(cam_data)} cameras, {max_frames} frames")
    print("  Q = quit, S = save current view as PNG")

    while True:
        idx = current_frame[0]
        panels = []

        for cam_num in sorted(cam_data.keys()):
            hf, ds, depth_scale, is_metric = cam_data[cam_num]
            if idx < ds.shape[0]:
                raw = ds[idx]
                meters = depth_to_meters(raw, depth_scale, is_metric)
                colored = colorize_depth(meters, max_depth=max_depth)

                # Stats overlay
                valid = meters[meters > 0]
                info = f"cam{cam_num} [{idx}/{ds.shape[0]}]"
                if valid.size > 0:
                    info += f" {valid.min():.2f}-{valid.max():.2f}m"
                cv2.putText(colored, info, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panels.append(colored)
            else:
                # Frame out of range for this camera — show black placeholder
                h, w = ds.shape[1], ds.shape[2]
                blank = np.zeros((h, w + 60, 3), dtype=np.uint8)
                cv2.putText(blank, f"cam{cam_num} (no frame {idx})", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                panels.append(blank)

        # Arrange in 2x2 grid (or 1xN for fewer cameras)
        if len(panels) <= 2:
            combined = np.hstack(panels) if panels else np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            # Pad to 4 panels
            while len(panels) < 4:
                panels.append(np.zeros_like(panels[0]))
            row1 = np.hstack(panels[:2])
            row2 = np.hstack(panels[2:4])
            combined = np.vstack([row1, row2])

        cv2.imshow(window_name, combined)
        key = cv2.waitKey(50) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            save_path = f"depth_snapshot_frame{idx:06d}.png"
            cv2.imwrite(save_path, combined)
            print(f"Saved: {save_path}")

    # Cleanup
    for hf, _, _, _ in cam_data.values():
        hf.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize recorded depth maps from HDF5 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('path', help='Session directory or specific depth.h5 file')
    parser.add_argument('--cam', type=int, default=None,
                        help='Camera number to view (1-4). Default: all cameras')
    parser.add_argument('--frame', type=int, default=None,
                        help='Specific frame index to export')
    parser.add_argument('--num-frames', type=int, default=5,
                        help='Number of frames to export (from start)')
    parser.add_argument('--max-depth', type=float, default=None,
                        help='Maximum depth for colormap (meters). Auto if not set')
    parser.add_argument('--export', action='store_true',
                        help='Export colorized depth frames as PNG')
    parser.add_argument('--interactive', action='store_true',
                        help='Open interactive viewer with frame slider')

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: {args.path} does not exist")
        sys.exit(1)

    if args.interactive:
        interactive_viewer(args.path, cam_filter=args.cam, max_depth=args.max_depth)
    elif args.export or args.frame is not None:
        export_frames(args.path, cam_filter=args.cam, frame_idx=args.frame,
                      num_frames=args.num_frames, max_depth=args.max_depth)
    else:
        # Default: export
        export_frames(args.path, cam_filter=args.cam, frame_idx=args.frame,
                      num_frames=args.num_frames, max_depth=args.max_depth)


if __name__ == '__main__':
    main()
