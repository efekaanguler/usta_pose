# Final Dataset Parquet Explanation

This document explains the files generated in `final_dataset_parquets/`.

These files are created by:

```bash
python3 usta_pose/devel/postprocess/collect_dataset_parquets.py /path/to/dataset_root
```

For each raw session, the collector first runs the revised process pipeline to
create `session_ml_dataset.parquet`, then creates a richer final interaction
parquet with `create_interaction_parquet.py`.

## 1. File Naming

Each final parquet file represents one recorded interaction session.

Example:

```text
20260507_154437_order1.parquet
20260507_154803_order2.parquet
20260507_155406_order3.parquet
20260507_155542_order4.parquet
```

The filename format is:

```text
YYYYMMDD_HHMMSS_orderN.parquet
```

Meaning:

- `YYYYMMDD_HHMMSS`: timestamp from the original session folder name.
- `orderN`: temporal order inside the direct parent group folder.
- `order1`: earliest session in that group.
- `order4`: latest session in that group.

For example, if a group folder contains:

```text
session_20260507_154437
session_20260507_154803
session_20260507_155406
session_20260507_155542
```

then the generated files become:

```text
20260507_154437_order1.parquet
20260507_154803_order2.parquet
20260507_155406_order3.parquet
20260507_155542_order4.parquet
```

## 2. What One Row Means

Each row is one synchronized frame/time step.

The row count depends on session duration and the adaptive FPS selected by the
revised process pipeline.

Example:

```text
rows=801
cols=3436
```

means:

- `801` synchronized frames.
- `3436` tabular features/columns per frame.

These files are wide tabular files. They are designed for ML training, feature
analysis, clustering, and nonverbal communication pattern discovery.

## 3. Coordinate System

The final parquets use the same coordinate representation as RViz2.

The original `session_ml_dataset.parquet` stores coordinates in the calibrated
camera/world reference frame. RViz applies a display rotation before showing the
skeletons. The final parquet stores coordinates after that same rotation.

Conversion:

```text
final_x = world_x
final_y = world_z
final_z = -world_y
```

So in `final_dataset_parquets/`:

- `+z` points upward.
- The head/body vertical direction should be visible in positive `z`.
- Coordinates match the visual orientation used in RViz2.

This is important: use the final parquet directly for analysis if you want the
same orientation that you visually inspect in RViz2.

## 4. Main Column Groups

The final parquet contains several column groups:

1. Session metadata
2. Calibration metadata
3. Dyad reference features
4. Per-person reference/root features
5. Per-person keypoint coordinates
6. Gaze features
7. Interaction distance features
8. Motion features
9. Quality and validity flags

The two people are represented as:

```text
p1
p2
```

These IDs come from the revised pipeline mapping:

- `p1`: person assigned to pose cam2 / gaze cam4.
- `p2`: person assigned to pose cam1 / gaze cam3.

## 5. Session Metadata Columns

Typical columns:

```text
schema_version
session_id
timestamp_ms
frame_idx
source_parquet_path
coordinate_frame
pose_model
gaze_model
```

Meaning:

- `schema_version`: version of the final parquet schema.
- `session_id`: original session folder name.
- `timestamp_ms`: synchronized hardware timestamp in milliseconds.
- `frame_idx`: zero-based row index in this final parquet.
- `source_parquet_path`: original `session_ml_dataset.parquet` path.
- `coordinate_frame`: expected to be `rviz_world`.
- `pose_model`: pose model used by the revised process, usually `rtmpose2d`.
- `gaze_model`: gaze model name, usually `puregaze`.

## 6. Calibration Metadata Columns

The final parquet stores calibration provenance directly inside the file so it
can be analyzed later without searching for the original `.npz`.

Typical columns:

```text
calib_path
calib_sha256
calib_ref_camera
calib_num_cameras
calib_image_width
calib_image_height
```

Per camera:

```text
calib_cam1_K_fx
calib_cam1_K_fy
calib_cam1_K_cx
calib_cam1_K_cy
calib_cam1_dist_0
...
calib_cam1_dist_4
```

Transform columns:

```text
calib_cam1_R_ref_to_cam_00
...
calib_cam1_R_ref_to_cam_22
calib_cam1_t_ref_to_cam_x
calib_cam1_t_ref_to_cam_y
calib_cam1_t_ref_to_cam_z

calib_cam1_R_cam_to_ref_00
...
calib_cam1_R_cam_to_ref_22
calib_cam1_t_cam_to_ref_x
calib_cam1_t_cam_to_ref_y
calib_cam1_t_cam_to_ref_z
```

Also stored:

```text
rviz_rotation_00
...
rviz_rotation_22
```

This is the matrix used to convert world coordinates into the RViz-aligned
coordinate frame.

## 7. Dyad Reference Columns

The dyad reference is the midpoint between the two people's reference/root
points.

Columns:

```text
dyad_ref_x
dyad_ref_y
dyad_ref_z
dyad_ref_valid
```

Meaning:

- `dyad_ref_x/y/z`: midpoint between `p1_ref` and `p2_ref`.
- `dyad_ref_valid`: true only when both people have valid references.

Distance/orientation columns:

```text
dyad_root_distance
dyad_root_horizontal_distance
dyad_vertical_offset_p1_minus_p2
dyad_p1_to_p2_unit_x
dyad_p1_to_p2_unit_y
dyad_p1_to_p2_unit_z
dyad_p2_to_p1_unit_x
dyad_p2_to_p1_unit_y
dyad_p2_to_p1_unit_z
```

Meaning:

- `dyad_root_distance`: 3D distance between the two reference points.
- `dyad_root_horizontal_distance`: distance in the horizontal plane.
- `dyad_vertical_offset_p1_minus_p2`: vertical offset between people.
- `dyad_p1_to_p2_unit_*`: unit vector from person 1 to person 2.
- `dyad_p2_to_p1_unit_*`: unit vector from person 2 to person 1.

These are useful for detecting interpersonal distance, approach/withdrawal, and
spatial relation.

## 8. Person Reference Columns

For each person:

```text
p1_ref_x
p1_ref_y
p1_ref_z
p1_ref_valid
p1_ref_source
p1_ref_observed
p1_ref_interpolated
p1_ref_to_dyad_x
p1_ref_to_dyad_y
p1_ref_to_dyad_z
p1_pose_cam_id
p1_gaze_cam_id
```

Same structure exists for `p2`.

Meaning:

- `p1_ref_x/y/z`: RViz-aligned person reference point.
- `p1_ref_valid`: true if reference is usable.
- `p1_ref_source`: source code used to compute the reference:
  - `0`: missing
  - `1`: hips
  - `2`: torso
  - `3`: shoulders
- `p1_ref_observed`: reference came from observed keypoints.
- `p1_ref_interpolated`: reference used interpolated keypoints.
- `p1_ref_to_dyad_x/y/z`: person reference relative to dyad midpoint.
- `p1_pose_cam_id`: source pose camera ID.
- `p1_gaze_cam_id`: source gaze camera ID.

## 9. Keypoint Coordinate Columns

Each person has 133 keypoints.

For each keypoint:

```text
p1_kpt0_world_x
p1_kpt0_world_y
p1_kpt0_world_z

p1_kpt0_person_rel_x
p1_kpt0_person_rel_y
p1_kpt0_person_rel_z

p1_kpt0_dyad_rel_x
p1_kpt0_dyad_rel_y
p1_kpt0_dyad_rel_z

p1_kpt0_score
p1_kpt0_observed
p1_kpt0_interpolated
```

Same pattern exists for:

```text
p1_kpt0 ... p1_kpt132
p2_kpt0 ... p2_kpt132
```

Meaning:

- `world_x/y/z`: absolute RViz-aligned coordinate.
- `person_rel_x/y/z`: keypoint relative to that person's reference point.
- `dyad_rel_x/y/z`: keypoint relative to the midpoint between both people.
- `score`: pose model confidence.
- `observed`: keypoint came from an actual model observation near that frame.
- `interpolated`: keypoint was filled by interpolation.

### Which Coordinate Type Should You Use?

Use `world_*` when you care about absolute layout:

- Where are people in the room?
- How far are they from each other?
- Is someone moving toward the other person?

Use `person_rel_*` when you care about body posture independent of location:

- arm raised
- head tilted
- hand near face
- torso pose

Use `dyad_rel_*` when you care about interaction geometry:

- hand moving toward the other person
- both people leaning toward the table center
- gestures relative to the shared interaction space

For most nonverbal communication analysis, use a combination of:

```text
person_rel_*
dyad_rel_*
gaze features
distance/motion features
quality flags
```

## 10. Gaze Columns

Per person:

```text
p1_gaze_dir_x
p1_gaze_dir_y
p1_gaze_dir_z
p1_gaze_observed
p1_gaze_interpolated
p1_face_detected
p1_gaze_yaw
p1_gaze_pitch
```

Derived gaze relation features:

```text
p1_gaze_to_other_head_cos
p1_gaze_to_other_head_angle_deg
p1_gaze_to_dyad_ref_cos
p1_gaze_to_dyad_ref_angle_deg
```

Same columns exist for `p2`.

Meaning:

- `p1_gaze_dir_x/y/z`: normalized RViz-aligned gaze direction.
- `p1_gaze_observed`: gaze was directly observed.
- `p1_gaze_interpolated`: gaze was interpolated.
- `p1_face_detected`: face detector succeeded.
- `p1_gaze_yaw`, `p1_gaze_pitch`: raw gaze angles from gaze model.
- `p1_gaze_to_other_head_cos`: cosine similarity between gaze direction and
  vector toward the other person's head.
- `p1_gaze_to_other_head_angle_deg`: angle in degrees between gaze and the
  other person's head.
- `p1_gaze_to_dyad_ref_*`: gaze relation to the dyad midpoint.

Small `gaze_to_other_head_angle_deg` means the person is likely looking toward
the other person's head.

## 11. Interaction Distance Features

Examples:

```text
p1_head_to_p2_head_distance
p1_left_wrist_to_p2_right_wrist_distance
p1_right_wrist_to_p2_left_wrist_distance
p1_left_wrist_to_p2_head_distance
p1_right_wrist_to_p2_head_distance
p2_left_wrist_to_p1_head_distance
p2_right_wrist_to_p1_head_distance
```

Meaning:

- head-to-head distance
- cross-person wrist-to-wrist distance
- each person's hands relative to the other person's head

These are useful for detecting:

- reaching
- pointing
- touch-like moments
- personal space changes
- hand-near-face interaction patterns

## 12. Motion Features

Examples:

```text
p1_motion_speed
p2_motion_speed
p1_motion_energy_body
p2_motion_energy_body
motion_energy_ratio_p1_over_p2
```

Meaning:

- `p1_motion_speed`: speed of the person's reference point.
- `p1_motion_energy_body`: average body keypoint speed.
- `motion_energy_ratio_p1_over_p2`: relative movement dominance.

These are useful for:

- detecting who is more active
- turn-taking analysis
- stillness vs movement
- synchrony / asymmetry analysis

## 13. Quality Columns

Per person:

```text
p1_body_valid_keypoint_count
p1_body_valid_keypoint_ratio
p1_hand_valid_keypoint_count
p1_hand_valid_keypoint_ratio
p1_any_pose_interpolated
p1_any_gaze_interpolated
```

Frame-level:

```text
frame_pose_valid
frame_gaze_valid
frame_interaction_valid
```

Meaning:

- `body_valid_keypoint_count`: valid body keypoints among keypoints `0..16`.
- `body_valid_keypoint_ratio`: valid body keypoint ratio.
- `hand_valid_keypoint_count`: valid hand keypoints among keypoints `91..132`.
- `hand_valid_keypoint_ratio`: valid hand keypoint ratio.
- `any_pose_interpolated`: at least one pose keypoint was interpolated.
- `any_gaze_interpolated`: gaze was interpolated.
- `frame_pose_valid`: both people have valid pose references and enough body keypoints.
- `frame_gaze_valid`: both people have usable observed gaze.
- `frame_interaction_valid`: frame is usable for pose-based dyadic interaction features.

For training or analysis, usually filter with:

```text
frame_interaction_valid == True
```

For gaze-specific analysis, additionally require:

```text
frame_gaze_valid == True
```

## 14. Suggested Use for Training

For pose/posture models:

Use:

```text
p1_kpt*_person_rel_*
p2_kpt*_person_rel_*
p1_body_valid_keypoint_ratio
p2_body_valid_keypoint_ratio
frame_pose_valid
```

For dyadic interaction models:

Use:

```text
p1_kpt*_dyad_rel_*
p2_kpt*_dyad_rel_*
dyad_root_distance
dyad_p1_to_p2_unit_*
dyad_p2_to_p1_unit_*
interaction distance features
motion features
frame_interaction_valid
```

For gaze/nonverbal attention models:

Use:

```text
p1_gaze_dir_*
p2_gaze_dir_*
p1_gaze_to_other_head_angle_deg
p2_gaze_to_other_head_angle_deg
mutual_gaze_cos_min
frame_gaze_valid
```

For nonverbal vocabulary discovery:

Recommended approach:

1. Load all files in `final_dataset_parquets/`.
2. Filter invalid frames.
3. Split into temporal windows, for example 1-5 seconds.
4. Aggregate each window:
   - mean
   - standard deviation
   - min/max
   - velocity summaries
   - gaze angle summaries
5. Cluster windows.
6. Inspect clusters in RViz/video.
7. Assign human-readable labels such as:
   - mutual gaze
   - gaze avoidance
   - lean-in
   - reach toward other
   - hand-near-face
   - synchronized stillness
   - one-person dominant motion

## 15. Practical Notes

- Do not assume every session has the same number of rows.
- Some sessions are much longer than others.
- For ML training, use sliding windows or sequence batching.
- Do not train directly on invalid frames without masks.
- Prefer `person_rel_*` for body pose.
- Prefer `dyad_rel_*` and interaction features for nonverbal communication.
- Prefer `world_*` only when absolute room/table layout matters.
- Always keep quality columns in the model input or use them for filtering.

