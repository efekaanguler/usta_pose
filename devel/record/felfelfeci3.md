# felfelfeci3 Full Lossless Recorder

`felfelfeci3.py` records synchronized 4-camera RealSense sessions with both
RGB and depth stored as lossless FFV1 video in MKV containers.

This recorder is intended for scientific master recordings where preserving the
raw camera information matters more than file size.

## Output

Each session is written under:

```text
recordings/session_YYYY-MM-DD_HH:MM/
```

If another recording starts during the same minute, the recorder appends
`_02`, `_03`, and so on instead of reusing an existing directory.

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
If one projection direction passes while the reverse direction contains
non-mutual or occluded surfaces, the pair is reported as a visibility warning
rather than a rigid-transform failure. A bidirectional outlier remains a real
calibration failure and blocks recording.
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
E  Executive bypass: skip the check and unlock R for engineering captures
Q  quit
```

Sessions unlocked with `E` store
`calibration_precheck.status = "bypassed"` in `metadata.json`. This keeps quick
engineering captures distinguishable from sessions that passed the scientific
pre-check. After each recording, the gate locks again and requires either `C`
or `E`.

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

## FPS Diagnostics

`frame_timing_summary.json` separates camera-side capture rate from saved video
rate and records queue drops, duplicate frame numbers, maximum writer queue
depth, FFmpeg errors, USB type, and physical USB port. The
`bottleneck_diagnosis` field distinguishes camera/USB capture loss from
encoder/disk backpressure.

These diagnostics do not change the scientific recording path. Every camera
still writes `color.mkv` as FFV1/bgr0 and `depth.mkv` as FFV1/gray16le. Frame
numbers that repeat are counted for diagnosis but are not removed from the
recording.

The main summary fields mean:

```text
camera_frame_fps_avg             unique RealSense frame numbers on hardware time
sdk_delivery_fps_avg             all framesets returned to Python on host time
saved_fps_avg                    paired color/depth frames successfully written
queue_dropped_frames             frames rejected because the 90-frame writer queue filled
max_writer_queue_depth           highest observed queue occupancy
duplicate_frame_numbers_observed repeated RealSense frame numbers; frames are retained
writer_error                     FFmpeg/pipe failure, otherwise null
bottleneck_diagnosis             likely capture/USB, writer/disk, combined, or none
```

`camera_usb_or_capture_side` points to the camera/USB/host-controller side:
inspect USB 3 mode, physical port/controller distribution, cable quality,
power, and RealSense warnings. `encoder_or_disk_backpressure` means capture was
healthy but the unchanged eight-stream FFV1 writer path could not drain its
queue fast enough; use a faster recording disk or a stronger recording PC
rather than changing the master codec.

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
ffprobe -hide_banner 'recordings/session_YYYY-MM-DD_HH:MM/cam1/color.mkv'
ffprobe -hide_banner 'recordings/session_YYYY-MM-DD_HH:MM/cam1/depth.mkv'
```

Expected result:

```text
color.mkv: ffv1 video, bgr0, 8-bit BGR data plus unused padding byte
depth.mkv: ffv1 video, gray16le, 16-bit depth values
```
