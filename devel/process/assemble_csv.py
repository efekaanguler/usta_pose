#!/usr/bin/env python3
"""
Step 4: CSV Assembly

Merges matched_frames.csv, pose_results.csv, and gaze_results.csv into a
single unified session_output.csv.

Usage:
    python assemble_csv.py \\
        --session-dir ./recordings/session_YYYYMMDD_HHMMSS

Output:
    {session_dir}/session_output.csv
"""

import argparse
import csv
import os
import sys


def load_csv_as_dict(csv_path, key_field='master_frame_idx'):
    """Load a CSV into a dict keyed by key_field."""
    data = {}
    if not os.path.exists(csv_path):
        print(f"  Warning: {csv_path} not found, skipping.")
        return data, []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            key = row[key_field]
            data[key] = row
    return data, [f for f in fieldnames if f != key_field]


def main():
    parser = argparse.ArgumentParser(
        description="Assemble pose + gaze results into unified CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--session-dir', type=str, required=True)
    parser.add_argument('--matched-csv', type=str, default=None)
    parser.add_argument('--pose-csv', type=str, default=None)
    parser.add_argument('--gaze-csv', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    session_dir = args.session_dir
    matched_csv = args.matched_csv or os.path.join(session_dir, "matched_frames.csv")
    pose_csv = args.pose_csv or os.path.join(session_dir, "pose_results.csv")
    gaze_csv = args.gaze_csv or os.path.join(session_dir, "gaze_results.csv")
    output_path = args.output or os.path.join(session_dir, "session_output.csv")

    print(f"Session:  {session_dir}")
    print(f"Matched:  {matched_csv}")
    print(f"Pose:     {pose_csv}")
    print(f"Gaze:     {gaze_csv}")
    print(f"Output:   {output_path}")
    print()

    # Load all CSVs
    matched_data, matched_cols = load_csv_as_dict(matched_csv)
    pose_data, pose_cols = load_csv_as_dict(pose_csv)
    gaze_data, gaze_cols = load_csv_as_dict(gaze_csv)

    if not matched_data:
        print("Error: No matched frame data found.")
        sys.exit(1)

    # Build unified fieldnames
    fieldnames = ['master_frame_idx'] + matched_cols + pose_cols + gaze_cols

    # Merge rows
    rows_written = 0
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key in sorted(matched_data.keys(), key=lambda x: int(x)):
            row = {'master_frame_idx': key}

            # Matched frame info
            for col in matched_cols:
                row[col] = matched_data[key].get(col, '')

            # Pose data
            if key in pose_data:
                for col in pose_cols:
                    row[col] = pose_data[key].get(col, '')
            else:
                for col in pose_cols:
                    row[col] = ''

            # Gaze data
            if key in gaze_data:
                for col in gaze_cols:
                    row[col] = gaze_data[key].get(col, '')
            else:
                for col in gaze_cols:
                    row[col] = ''

            writer.writerow(row)
            rows_written += 1

    print(f"Assembled {rows_written} rows, {len(fieldnames)} columns")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
