# Revised Process and Postprocess Usage

This guide explains how to run the revised USTA pose pipeline and how to replay
the result in RViz2.

The workflow has two parts:

1. Run `devel/revised_process` inside the Docker container. This is the
   preprocessing step and it creates the ML-ready Parquet file.
2. Run `devel/postprocess` on the local machine. This publishes ROS2
   visualization data so the session can be observed in RViz2.

## Expected Session Layout

The scripts expect a recorded session directory with camera subdirectories:

```text
session_YYYYMMDD_HHMMSS/
  metadata.json
  multicam_calibration.npz              # optional, can also be in calib_data/
  cam1/
    color.mp4
    depth.mkv
    cam1_color_timestamps.csv
  cam2/
    color.mp4
    depth.mkv
    cam2_color_timestamps.csv
  cam3/
    color.mp4
    cam3_color_timestamps.csv
  cam4/
    color.mp4
    cam4_color_timestamps.csv
```

`cam1` and `cam2` are used for pose extraction. `cam3` and `cam4` are used for
gaze extraction.

## 1. Preprocess Inside Docker

The revised process uses model dependencies that are provided by the Docker
image, so run this part inside the development container.

From the local machine, start the container:

```bash
cd /path/to/usta_pose
bash dev_run.sh
```

Inside the container, run the revised pipeline:

```bash
cd /host_mount/usta_pose
bash devel/revised_process/run_revised_pipeline.sh /path/to/session_YYYYMMDD_HHMMSS
```

If the session is inside the mounted repository, use that mounted path. For
example:

```bash
bash devel/revised_process/run_revised_pipeline.sh /host_mount/usta_pose/testing/session_YYYYMMDD_HHMMSS
```

The pipeline runs three steps:

1. Pose extraction for `cam1` and `cam2`
2. Gaze extraction for `cam3` and `cam4`
3. Adaptive-FPS resampling, short-gap smoothing/interpolation, calibration transform, and robust root-relative export

The final output is written to:

```text
<SESSION_DIR>/session_ml_dataset.parquet
```

Pose extraction defaults to RTMPose-L 2D from `models/pose/rtmw2d/` because
metric depth deprojection depends on reliable image-space keypoints. RTMW3D is
available with `POSE_MODEL=rtmw3d`, but broad/full-frame RTMW3D boxes are not
reliable for metric deprojection; the extractor requires a tight
`POSE_BBOX_CAM#` before using that mode. If no `POSE_BBOX_CAM1` is provided, cam1 automatically excludes the
rightmost one fifth of the image to avoid the near cam2 subject entering cam1.
Cam1 also rejects close/oversized foreground detections by default; use
`--disable-foreground-rejection` only for debugging.

The resampling stage estimates output FPS from usable timestamps instead of
forcing 30 FPS. Override it only when needed:

```bash
python3 devel/revised_process/resample_and_transform.py \
  --session-dir /path/to/session_YYYYMMDD_HHMMSS \
  --target-fps 24 \
  --max-interp-gap-ms 150
```

Interpolation is intentionally limited to short gaps and does not fill before
the first valid observation or after the last valid observation. Root metadata
uses numeric source codes: `0=missing`, `1=hips`, `2=torso`, `3=shoulders`.

Intermediate outputs are written next to each camera:

```text
<SESSION_DIR>/cam1/cam1_pose_raw.csv
<SESSION_DIR>/cam2/cam2_pose_raw.csv
<SESSION_DIR>/cam3/cam3_gaze_raw.csv
<SESSION_DIR>/cam4/cam4_gaze_raw.csv
```

### Optional Preprocess Commands

To run only one stage manually:

```bash
python3 devel/revised_process/extract_pose_independent.py \
  --session-dir /path/to/session_YYYYMMDD_HHMMSS \
  --cam-id 1

python3 devel/revised_process/extract_gaze_independent.py \
  --session-dir /path/to/session_YYYYMMDD_HHMMSS \
  --cam-id 3

python3 devel/revised_process/resample_and_transform.py \
  --session-dir /path/to/session_YYYYMMDD_HHMMSS
```

To pass a specific calibration file:

```bash
python3 devel/revised_process/resample_and_transform.py \
  --session-dir /path/to/session_YYYYMMDD_HHMMSS \
  --calib-npz /path/to/multicam_calibration.npz
```

If no calibration file is found, identity transforms are used. For correct
multi-camera geometry, provide `multicam_calibration.npz`.

## 2. Create the Interaction Parquet

After `session_ml_dataset.parquet` is generated, create the RViz-aligned dyadic
interaction dataset locally:

```bash
python3 devel/postprocess/create_interaction_parquet.py \
  --session-dir /path/to/session_YYYYMMDD_HHMMSS
```

This writes:

```text
<SESSION_DIR>/session_interaction_dataset.parquet
```

All coordinates in this derived file use the same display convention as RViz2:
`x=world_x`, `y=world_z`, `z=-world_y`, so `+z` is up. The file includes
absolute keypoints, person-relative keypoints, dyad-relative keypoints, gaze
relations, distance/motion features, quality flags, and flattened calibration
metadata.

To process and collect many raw sessions into dataset-level folders, run:

```bash
python3 devel/postprocess/collect_dataset_parquets.py /path/to/dataset_root --require-four
```

The collector discovers every `session_YYYYMMDD_HHMMSS` folder, runs
`devel/revised_process/run_revised_pipeline.sh` when `session_ml_dataset.parquet`
is missing, then writes default pipeline parquets to `default_parquets/` and
RViz-aligned interaction parquets to `final_dataset_parquets/`. Session folders
are ordered within their direct parent by timestamp, and output names use
`YYYYMMDD_HHMMSS_orderN.parquet`. Use `--dry-run` to inspect work without
running the pipeline, `--skip-processing` to require already generated default
parquets, and `--force-reprocess` to rerun the revised pipeline for every
session.

## 3. Postprocess Locally and View in RViz2

Run the postprocess visualization on the local machine, not inside the Docker
preprocessing environment. The postprocess script reads
`session_ml_dataset.parquet` and publishes ROS2 `visualization_marker_array`
messages.

Use one terminal for the ROS publisher:

```bash
cd /path/to/usta_pose
source /opt/ros/<ros_distro>/setup.bash
python3 devel/postprocess/revised_visualize_rviz.py \
  --session-dir /path/to/session_YYYYMMDD_HHMMSS \
  --calib /path/to/session_YYYYMMDD_HHMMSS/multicam_calibration.npz \
  --fps 30
```

Replace `<ros_distro>` with the ROS2 distribution installed on the local
machine, for example `humble` or `jazzy`.

The `--calib` argument is optional. It is used only to publish camera TF frames
for display in RViz2. The person pose data is already stored in the shared
world frame inside the Parquet file.

You can also pass the Parquet file directly:

```bash
python3 devel/postprocess/revised_visualize_rviz.py \
  --parquet /path/to/session_YYYYMMDD_HHMMSS/session_ml_dataset.parquet \
  --fps 30
```

## 4. Launch RViz2 in a Separate Terminal

Open a second local terminal while the postprocess script is still running:

```bash
source /opt/ros/<ros_distro>/setup.bash
rviz2
```

In RViz2:

1. Set the fixed frame to `world`.
2. Add a `MarkerArray` display.
3. Set the marker topic to `/visualization_marker_array`.
4. If `--calib` was provided, add a `TF` display to inspect camera frames.

The visualizer replays the session continuously. When it reaches the last
frame, it starts again from the beginning.

## Troubleshooting

If `session_ml_dataset.parquet` is missing, the Docker preprocessing step did
not finish successfully. Re-run:

```bash
bash devel/revised_process/run_revised_pipeline.sh /path/to/session_YYYYMMDD_HHMMSS
```

If the local postprocess script says ROS2 Python packages are missing, make
sure ROS2 is installed locally and that the correct setup file is sourced:

```bash
source /opt/ros/<ros_distro>/setup.bash
```

If RViz2 opens but nothing is visible, check:

1. The postprocess terminal is still running.
2. RViz2 fixed frame is `world`.
3. The display type is `MarkerArray`.
4. The topic is `/visualization_marker_array`.
5. The session path points to the same session that contains
   `session_ml_dataset.parquet`.

