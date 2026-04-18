#!/usr/bin/env python3
"""
Step 1: Cross-Camera Frame Matching

Matches frames across all cameras using hardware timestamps. Selects cam1 as
the master camera (starting from the 10th frame) and finds the nearest
corresponding frame in each slave camera by hw_timestamp_ms.

Usage:
    python match_frames.py --session-dir ./recordings/session_YYYYMMDD_HHMMSS

Output:
    {session_dir}/matched_frames.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


def load_timestamps(csv_path):
    """Load timestamps from a camera CSV file.

    Returns:
        list of (frame_idx, hw_timestamp_ms) tuples, sorted by frame_idx.
    """
    entries = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['frame_idx'])
            hw_ts = float(row['hw_timestamp_ms'])
            entries.append((idx, hw_ts))
    entries.sort(key=lambda x: x[0])
    return entries


def find_nearest_frame(target_ts, slave_timestamps, slave_hw_array):
    """Find the frame in slave_timestamps closest to target_ts.

    Uses binary search on a pre-sorted hw_timestamp array for O(log n).

    Returns:
        (slave_frame_idx, delta_ms)
    """
    pos = np.searchsorted(slave_hw_array, target_ts)

    best_idx = None
    best_delta = float('inf')

    for candidate in [pos - 1, pos]:
        if 0 <= candidate < len(slave_hw_array):
            delta = abs(slave_hw_array[candidate] - target_ts)
            if delta < best_delta:
                best_delta = delta
                best_idx = candidate

    slave_frame_idx = slave_timestamps[best_idx][0]
    return slave_frame_idx, best_delta


def main():
    parser = argparse.ArgumentParser(
        description="Match frames across cameras by hardware timestamps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--session-dir', type=str, required=True,
                        help='Path to session directory')
    parser.add_argument('--master-cam', type=int, default=1,
                        help='Master camera ID (1-based)')
    parser.add_argument('--skip-frames', type=int, default=10,
                        help='Number of initial master frames to skip (warm-up)')
    parser.add_argument('--max-delta-ms', type=float, default=50.0,
                        help='Maximum acceptable timestamp delta (ms). '
                             'Frame pairs exceeding this are flagged.')
    parser.add_argument('--num-cameras', type=int, default=4,
                        help='Total number of cameras')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: {session_dir}/matched_frames.csv)')
    args = parser.parse_args()

    session_dir = args.session_dir
    num_cameras = args.num_cameras
    master_cam = args.master_cam

    # --- Load all timestamp CSVs ---
    all_timestamps = {}  # cam_id -> list of (frame_idx, hw_ts)
    for cam_id in range(1, num_cameras + 1):
        ts_csv = os.path.join(
            session_dir, f"cam{cam_id}", f"cam{cam_id}_color_timestamps.csv")
        if not os.path.exists(ts_csv):
            print(f"Warning: Timestamp file not found: {ts_csv}")
            continue
        all_timestamps[cam_id] = load_timestamps(ts_csv)
        print(f"  cam{cam_id}: {len(all_timestamps[cam_id])} frames loaded")

    if master_cam not in all_timestamps:
        print(f"Error: Master camera {master_cam} timestamps not found.")
        sys.exit(1)

    slave_cams = [c for c in sorted(all_timestamps.keys()) if c != master_cam]
    print(f"\nMaster camera: cam{master_cam}")
    print(f"Slave cameras: {['cam' + str(c) for c in slave_cams]}")
    print(f"Skipping first {args.skip_frames} master frames\n")

    # --- Pre-build sorted hw_timestamp arrays for binary search ---
    slave_hw_arrays = {}
    for cam_id in slave_cams:
        hw_values = np.array([t[1] for t in all_timestamps[cam_id]])
        slave_hw_arrays[cam_id] = hw_values

    # --- Match frames ---
    master_ts = all_timestamps[master_cam]
    output_path = args.output or os.path.join(session_dir, "matched_frames.csv")

    matched_count = 0
    flagged_count = 0

    cam_headers = [f"cam{c}_idx" for c in sorted(all_timestamps.keys())]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['master_frame_idx', 'master_hw_ts_ms'] + cam_headers + [
            'max_delta_ms', 'flag']
        writer.writerow(header)

        for i, (master_idx, master_hw_ts) in enumerate(master_ts):
            if i < args.skip_frames:
                continue

            row = [master_idx, round(master_hw_ts, 3)]

            # Master camera's own index
            matched_indices = {}
            matched_indices[master_cam] = master_idx

            max_delta = 0.0

            for cam_id in slave_cams:
                slave_idx, delta = find_nearest_frame(
                    master_hw_ts,
                    all_timestamps[cam_id],
                    slave_hw_arrays[cam_id],
                )
                matched_indices[cam_id] = slave_idx
                max_delta = max(max_delta, delta)

            # Build row in camera order
            for cam_id in sorted(all_timestamps.keys()):
                row.append(matched_indices[cam_id])

            flag = ''
            if max_delta > args.max_delta_ms:
                flag = 'HIGH_DELTA'
                flagged_count += 1

            row.append(round(max_delta, 3))
            row.append(flag)
            writer.writerow(row)
            matched_count += 1

    print(f"Matched {matched_count} frame sets")
    print(f"Flagged (delta > {args.max_delta_ms}ms): {flagged_count}")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
