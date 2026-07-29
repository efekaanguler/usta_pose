# felfelfeci3 Full Lossless Recorder

`felfelfeci3.py` records synchronized 4-camera RealSense sessions with both
RGB and depth stored as lossless FFV1 video in MKV containers.

This recorder is intended for scientific master recordings where preserving the
raw camera information matters more than file size.

## Output

Each session is written under:

```text
recordings/session_YYYYMMDD_HHMMSS/
```

Each camera folder contains:

```text
camX/color.mkv
camX/depth.mkv
camX/camX_color_timestamps.csv
camX/camX_depth_timestamps.csv
```

The session root also contains:

```text
metadata.json
frame_timing_report.csv
frame_timing_summary.json
```

## Stored Formats

RGB color:

```text
source stream: RealSense bgr8
ffmpeg input:  bgr24
ffmpeg output: bgr0
codec:         FFV1
container:     MKV
file:          color.mkv
lossless:      yes, for the original 8-bit BGR channels
```

Depth:

```text
source stream: RealSense z16
ffmpeg input:  gray16le
ffmpeg output: gray16le
codec:         FFV1
container:     MKV
file:          depth.mkv
lossless:      yes, uint16 z16 values are preserved
```

`color.mkv` uses `bgr0` because this FFmpeg FFV1 encoder does not support
`bgr24` output directly. The original 8-bit BGR channels are preserved
losslessly; the fourth byte is an unused constant padding channel added by the
pixel format. This is not 16-bit RGB.

`depth.mkv` keeps the raw `uint16` z16 depth units. Convert to meters with the
`depth_scale_meters_per_unit` value saved in `metadata.json` and
`depth_meta.json`.

## Usage

Activate the project environment if needed:

```bash
source /home/efekaan/Desktop/torch/torch_env/bin/activate
```

Run a recording with the default camera config:

```bash
python3 usta_pose/devel/record/felfelfeci3.py
```

Headless mode:

```bash
python3 usta_pose/devel/record/felfelfeci3.py \
  --no-gui
```

Common options:

```text
--output-dir DIR          Base recordings directory
--cam-config FILE         Camera serial config JSON
--width N                Frame width, default 1280
--height N               Frame height, default 720
--fps N                  Capture FPS, default 30
--align-depth-live       Align depth to color during capture; higher CPU load
--no-gui                 Run without OpenCV preview windows
```

## Default Depth/PCL Calibration Pre-check

Before recording starts, `felfelfeci3.py` captures five live raw-depth frames
from every camera. It takes a temporal median, rejects moving pixels and depth
edges, then projects each camera's measured depth into its paired camera through
the same transform chain used by `create_session_pcl.py`:

```text
source depth
  -> source color camera
  -> morning reference frame
  -> target color camera
  -> target depth
```

The checker measures bidirectional median/P75 depth disagreement, inlier ratio,
and common depth support. It never runs ICP and never changes the calibration.
`PASS` and a connected-graph `WARN` unlock recording; severe mismatch or a
disconnected camera blocks it.

Default gates are intentionally operational rather than micron-level: `PASS`
uses 35 mm median and 65 mm P75 limits; `WARN` extends these to 50 mm and
100 mm while still requiring a connected camera graph. This allows normal D400
depth noise and small surface/occlusion effects without accepting the visibly
separated clouds seen in bad calibrations. RealSense specifies up to 2% depth
Z error and RMS spatial noise for the relevant D400 family operating range:
[D400 depth-quality specification](https://support.realsenseai.com/hc/en-us/articles/360059129453-Depth-accuracy-for-Intel-RealSense-D400-Series-Cameras).

Run normally:

```bash
python3 usta_pose/devel/record/felfelfeci3.py
```

By default, the script uses:

```text
output-dir: usta_pose/devel/record/recordings
calibration: {output-dir}/multicam_calibration.npz
pre-check: five-frame live depth/PCL consistency
```

Controls:

```text
C  capture a short static depth burst and run the PCL check
R  start recording only after PASS/WARN
Q  quit
```

Keep people, chairs, and tabletop objects still for about one second while
pressing `C`. No point cloud, image, or log file is written by the check.

Run only the live check and exit:

```bash
python3 usta_pose/devel/record/felfelfeci3.py \
  --pcl-check-only \
  --no-gui
```

Emergency/debug recording without any pre-check:

```bash
python3 usta_pose/devel/record/felfelfeci3.py --no-calib-check
```

The AprilTag cube implementation is retained but is no longer the default:

```bash
python3 -m pip install pupil-apriltags
python3 usta_pose/devel/record/felfelfeci3.py \
  --calib-check-method cube
```

## Morning Calibration

The default daily command keeps ChArUco pairwise capture and fixed master
intrinsics. After writing `multicam_calibration.npz`, it automatically starts
the cameras and runs the same PCL check:

```bash
cd usta_pose/devel/calibration
./calibrate.sh
```

Use `--skip-pcl-check` only for offline diagnostics when cameras are not
connected.

## Python Dependencies

```bash
python3 -m pip install -r usta_pose/devel/record/requirements.txt
```

## Downstream Note

The existing revised processing scripts currently look for `camX/color.mp4`.
This full-lossless recorder writes `camX/color.mkv` instead. To process these
sessions directly, add `color.mkv` fallback support to the downstream readers or
create a derived `color.mp4` for compatibility.

Do not treat a derived MP4 as the scientific RGB master: MP4/H.264 conversion is
lossy unless a separate lossless codec/container path is used.

## Quick Verification

After a short test recording, verify the streams with `ffprobe`:

```bash
ffprobe -hide_banner recordings/session_YYYYMMDD_HHMMSS/cam1/color.mkv
ffprobe -hide_banner recordings/session_YYYYMMDD_HHMMSS/cam1/depth.mkv
```

Expected result:

```text
color.mkv: ffv1 video, bgr0, 8-bit BGR data plus unused padding byte
depth.mkv: ffv1 video, gray16le, 16-bit depth values
```
