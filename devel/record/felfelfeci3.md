# felfelfeci3 Recording Variants

This folder contains two experimental recorder variants derived from
`felfelfeci2.py`. Both keep the existing depth path unchanged:

- `depth.mkv`: realtime FFV1 lossless, 16-bit `gray16le`
- final RGB file expected by the existing processing pipeline: `color.mp4`

The difference is only how RGB is recorded during the realtime capture phase
before it is converted to final `color.mp4`.

Both scripts use the `ffmpeg` CLI for realtime video writers and for the
offline transcode step. `ffmpeg` must be available on `PATH`.

Recommended environment:

```bash
source /home/efekaan/Desktop/torch/torch_env/bin/activate
```

## Variant 1: `felfelfeci3_1.py`

Realtime RGB is written as lossless FFV1:

```text
camX/color_lossless.mkv
camX/depth.mkv
```

When recording stops, the script runs an offline transcode:

```text
camX/color_lossless.mkv -> camX/color.mp4
```

The final MP4 uses `libx264`, `yuv420p`, and the configured offline CRF/preset.
By default this is:

```text
CRF 20, preset slow
```

Use this variant when RGB quality during capture is important and the NVMe disk
has enough free space for large temporary files. It moves pressure away from
realtime H.264 encoding and toward disk write bandwidth.

Example:

```bash
python usta_pose/devel/record/felfelfeci3_1.py \
  --output-dir usta_pose/devel/recordings \
  --offline-crf 20 \
  --no-gui
```

Delete the FFV1 RGB intermediate after successful conversion:

```bash
python usta_pose/devel/record/felfelfeci3_1.py \
  --output-dir usta_pose/devel/recordings \
  --offline-crf 20 \
  --delete-intermediate \
  --no-gui
```

## Variant 2: `felfelfeci3_2.py`

Realtime RGB is written as fast H.264:

```text
camX/color_realtime.mp4
camX/depth.mkv
```

When recording stops, the script runs an offline transcode:

```text
camX/color_realtime.mp4 -> camX/color.mp4
```

Realtime RGB defaults to:

```text
libx264, CRF 18, preset ultrafast, tune zerolatency
```

Offline final RGB defaults to:

```text
libx264, CRF 20, preset slow
```

Use this variant when temporary disk space matters more than perfect RGB
intermediate quality. The realtime file is already lossy, so the offline step
can make the final file smaller/more consistent, but it cannot recover detail
lost during realtime encoding.

Example:

```bash
python usta_pose/devel/record/felfelfeci3_2.py \
  --output-dir usta_pose/devel/recordings \
  --realtime-crf 18 \
  --offline-crf 20 \
  --no-gui
```

Delete the realtime RGB intermediate after successful conversion:

```bash
python usta_pose/devel/record/felfelfeci3_2.py \
  --output-dir usta_pose/devel/recordings \
  --realtime-crf 18 \
  --offline-crf 20 \
  --delete-intermediate \
  --no-gui
```

## Options

Common options:

```text
--output-dir DIR          Base recordings directory
--cam-config FILE         Camera serial config JSON
--width N                Frame width
--height N               Frame height
--fps N                  Capture FPS
--offline-crf N          Final color.mp4 CRF, default 20
--offline-preset NAME    Final color.mp4 preset, default slow
--delete-intermediate    Delete RGB intermediate only after successful transcode
--align-depth-live       Align depth during capture; higher CPU load
--no-gui                 Headless mode
```

`felfelfeci3_2.py` also has:

```text
--realtime-crf N         Realtime ultrafast RGB CRF, default 18
```

## Output Contract

Both scripts end with the same files required by the downstream pose/gaze
pipeline:

```text
camX/color.mp4
camX/depth.mkv
camX/camX_color_timestamps.csv
camX/camX_depth_timestamps.csv
metadata.json
frame_timing_report.csv
frame_timing_summary.json
```

Intermediate files are kept by default. Passing `--delete-intermediate` removes
them only after ffmpeg successfully creates a non-empty `color.mp4`.

## Suggested Test

Start with a short 2-5 minute test at the target resolution/FPS. Compare:

- `frame_timing_summary.json`: late/fast frame counts and average FPS
- session size before and after deleting intermediates
- visual quality of final `color.mp4`

With a 512 GB NVMe and about 400 GB free, `felfelfeci3_1.py` is reasonable for
short tests, but long 4-camera 720p/30 sessions can still consume space quickly.
For longer sessions, `felfelfeci3_2.py` is usually the safer first choice.
