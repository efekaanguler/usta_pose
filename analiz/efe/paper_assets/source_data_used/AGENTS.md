# Codex Agent Instructions: USTA Pose & Gaze Pipeline

Welcome! This document provides an overview of the workspace to help you understand the architecture and workflow of our machine learning dataset generation pipeline.

## 1. Data and Source Code Directories
- **`./amildak_ramazan`**: This directory contains our sample dataset (recordings and sessions).
- **`./usta_pose`**: This directory contains the core source code for processing data, running inference, and generating datasets. We mainly work within the `./usta_pose/devel/` subdirectory.

## 2. Pipeline Overview (`./usta_pose/devel/`)

The core dataset processing pipeline is orchestrated by **`./revised_process/run_revised_pipeline.sh`**. It takes a session directory as input and processes the raw camera data in three main steps to output a machine learning-ready tabular dataset.

### Step 1: Independent Pose Processing (`./revised_process/extract_pose_independent.py`)
- Runs on pose cameras (typically `cam1` and `cam2`).
- Uses **RTMPose-L 2D** by default to extract 133 whole-body image-space keypoints for every frame.
- Deprojects the image-space keypoints into metric 3D within the **local camera coordinate frame** using the synchronized depth video (`depth.mkv`). No global transformations are applied at this stage.
- `POSE_MODEL=rtmw3d` is available only as an explicit experimental mode; broad/full-frame RTMW3D boxes are blocked for metric deprojection, so a tight person bbox must be provided.
- For `cam1`, the default ROI excludes the rightmost 1/5 of the image and rejects close/oversized foreground detections to reduce wrong-person switches.
- Outputs raw pose data to `{session_dir}/cam{id}/cam{id}_pose_raw.csv`.

### Step 2: Independent Gaze Processing (`./revised_process/extract_gaze_independent.py`)
- Runs on gaze cameras (typically `cam3` and `cam4`).
- Uses MediaPipe Face Detection to crop the head and the **PureGaze** model to estimate gaze pitch and yaw.
- Converts the pitch and yaw into a 3D unit gaze vector.
- Outputs raw gaze data to `{session_dir}/cam{id}/cam{id}_gaze_raw.csv`.

### Step 3: Resample, Transform, and Export (`./revised_process/resample_and_transform.py`)
This is the final and most crucial step where all independent camera data is synchronized and transformed:
1. **Synchronization**: Reads the raw CSV files and creates a globally synchronized timeline at an adaptive session FPS estimated from the usable camera timestamps.
2. **Smoothing & Interpolation**: Interpolates only short pose/gaze gaps (default max 150 ms) and applies **Savitzky-Golay filtering** only within finite continuous segments. It does not forward/back-fill long missing regions.
3. **Root-Relative Transformation**: 
   - Uses multi-camera calibration data to transform all local 3D points and gaze vectors into a unified reference frame (Cam1).
   - Calculates a robust **root** using plausible hips when available, falling back to torso or shoulder midpoint when hips are missing/unreliable.
   - Computes all 133 keypoint coordinates relative to this root and stores root provenance metadata.
4. **Export**: Saves the synchronized, root-relative dataset containing 3D keypoints, gaze directions, and metadata into a final **`session_ml_dataset.parquet`** file.

### Interaction Dataset (`./postprocess/create_interaction_parquet.py`)
- Reads `session_ml_dataset.parquet`, reconstructs each skeleton, applies the same coordinate rotation used for RViz, and writes `session_interaction_dataset.parquet`.
- The derived parquet stores RViz-aligned absolute, person-relative, and dyad-relative keypoints, gaze/interaction features, quality flags, and calibration provenance.

### Visualization (`./postprocess/revised_visualize_rviz.py`)
- This script reads the processed dataset and broadcasts ROS (Robot Operating System) messages.
- It is used to visualize the synchronized 3D skeletons, gaze vectors, and camera transformations in **RViz**.

## 3. Environment Setup
You can activate the project's Python virtual environment to access and use the required libraries by sourcing it:
```bash
source /home/efekaan/Desktop/torch/torch_env/bin/activate
```
