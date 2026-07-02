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
python usta_pose/devel/record/felfelfeci3.py \
  --output-dir usta_pose/devel/recordings
```

Headless mode:

```bash
python usta_pose/devel/record/felfelfeci3.py \
  --output-dir usta_pose/devel/recordings \
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
