# ChArUco-Based Stereo Camera Calibration

Complete workflow for calibrating two RealSense cameras using ChArUco boards. This method provides superior accuracy compared to individual AprilTag markers.

## Overview

This calibration system computes:
1. **Intrinsic parameters** for each camera (focal length, principal point, distortion coefficients)
2. **Extrinsic parameters** (relative rotation R and translation T between cameras)

## Why ChArUco?

**Advantages over AprilTag calibration:**
- Sub-pixel corner accuracy (better than marker centroids)
- Robust to partial occlusion
- More calibration points per image
- Single planar target (easier to manufacture accurately)

**Your current 3×4 board:**
- Provides only 6 internal corners (limited but usable)
- Requires **careful capture procedure** with many diverse views
- Consider upgrading to a larger board (6×8 or 8×11) for better results

## Calibration Workflows

There are two approaches to stereo calibration, each with different trade-offs:

### Workflow A: Two-Stage Calibration (RECOMMENDED for Best Quality)

**Why this is better:**
- ✓ Full FOV coverage for intrinsic calibration (including edges and corners)
- ✓ Better distortion parameter estimation
- ✓ More accurate for precision applications (like gaze estimation)
- ✗ Requires 3 capture sessions (longer process)

**When the board is at the edge of one camera:**
- Stage 1 (intrinsic): Board may only be visible to ONE camera - that's OK!
- Stage 2 (extrinsic): Board must be visible to BOTH cameras

### Workflow B: Single-Stage Calibration (Faster but Less Accurate)

**Why you might use this:**
- ✓ Faster - only one capture session needed
- ✓ Simpler workflow
- ✗ Cannot capture board at edges (must be visible to both cameras)
- ✗ Limited FOV coverage → less accurate intrinsics
- ✗ May have higher distortion errors at image edges

**Recommendation:** Use Workflow A (two-stage) for best results, especially for gaze estimation where accuracy is critical.

## Quick Start

### Workflow A: Two-Stage Calibration (Recommended)

```bash
# Step 1a: Capture intrinsic images for Camera 1 (cover ALL edges and corners!)
python charuco_intrinsic_capture.py \
  --camera-id 1 \
  --output-dir ./intrinsic_cam1 \
  --num-captures 30 \
  --auto-capture \
  --squares-x 3 --squares-y 4 \
  --square-length 0.063 --marker-length 0.047

# Step 1b: Capture intrinsic images for Camera 2 (cover ALL edges and corners!)
python charuco_intrinsic_capture.py \
  --camera-id 2 \
  --output-dir ./intrinsic_cam2 \
  --num-captures 30 \
  --auto-capture \
  --squares-x 3 --squares-y 4 \
  --square-length 0.063 --marker-length 0.047

# Step 2: Capture stereo pairs (board visible in BOTH cameras)
python charuco_stereo_capture.py \
  --output-dir ./stereo_captures \
  --num-captures 25 \
  --auto-capture \
  --squares-x 3 --squares-y 4 \
  --square-length 0.063 --marker-length 0.047

# Step 3: Two-stage calibration
python charuco_two_stage_calibrate.py \
  --intrinsic-dir-1 ./intrinsic_cam1 \
  --intrinsic-dir-2 ./intrinsic_cam2 \
  --stereo-dir ./stereo_captures \
  --output charuco_calibration.npz \
  --squares-x 3 --squares-y 4 \
  --square-length 0.063 --marker-length 0.047
```

### Workflow B: Single-Stage Calibration (Faster)

```bash
# Step 1: Capture stereo pairs only
python charuco_stereo_capture.py \
  --output-dir ./stereo_captures \
  --num-captures 30 \
  --auto-capture \
  --squares-x 3 --squares-y 4 \
  --square-length 0.063 --marker-length 0.047

# Step 2: Single-stage calibration
python charuco_two_stage_calibrate.py \
  --stereo-dir ./stereo_captures \
  --output charuco_calibration.npz \
  --squares-x 3 --squares-y 4 \
  --square-length 0.063 --marker-length 0.047

# Or use the original calibration script
python charuco_stereo_calibrate.py \
  --input-dir ./stereo_captures \
  --output charuco_calibration.npz \
  --squares-x 3 --squares-y 4 \
  --square-length 0.063 --marker-length 0.047
```

## Detailed Instructions

### 1. Generate a Better Board (Recommended)

```bash
# Generate a 6×8 board optimized for A4 paper
python generate_charuco_board.py --squares-x 6 --squares-y 8 --output charuco_6x8.png

# Generate an 8×11 board for A3 paper (even better)
python generate_charuco_board.py --squares-x 8 --squares-y 11 --paper-size A3 --output charuco_8x11.png
```

**Printing tips:**
- Print at 100% scale (disable "fit to page")
- Use matte paper to reduce glare
- Mount on rigid flat surface (foam board, plywood)
- Verify printed square size with a ruler

### 2. Intrinsic Calibration Image Capture (Two-Stage Workflow Only)

**For two-stage workflow, capture images for each camera independently.**

```bash
# Camera 1 - Cover entire FOV including edges!
python charuco_intrinsic_capture.py \
  --camera-id 1 \
  --output-dir ./intrinsic_cam1 \
  --num-captures 30 \
  --auto-capture \
  --squares-x 3 --squares-y 4 \
  --square-length 0.063 --marker-length 0.047

# Camera 2 - Cover entire FOV including edges!
python charuco_intrinsic_capture.py \
  --camera-id 2 \
  --output-dir ./intrinsic_cam2 \
  --num-captures 30 \
  --auto-capture \
  --squares-x 3 --squares-y 4 \
  --square-length 0.063 --marker-length 0.047
```

**Critical for intrinsic calibration:**
- Position board at **all four corners** of the camera view
- Include **all edges** (left, right, top, bottom)
- Include center positions
- Vary distance (near and far)
- Tilt board at different angles

**Visual guide:** The display shows a 3×3 grid overlay and corner markers - make sure you capture images with the board in each grid region.

### 3. Stereo Calibration Image Capture

**RECOMMENDED: Auto-capture mode for single person operation**

```bash
# Using your current 3×4 board
python charuco_stereo_capture.py \
  --output-dir ./charuco_captures \
  --num-captures 30 \
  --auto-capture \
  --squares-x 3 \
  --squares-y 4 \
  --square-length 0.063 \
  --marker-length 0.047 \
  --aruco-dict 4X4_50
```

**Alternative: Manual mode (requires more coordination)**

```bash
python charuco_stereo_capture.py \
  --output-dir ./charuco_captures \
  --num-captures 30 \
  --squares-x 3 \
  --squares-y 4 \
  --square-length 0.063 \
  --marker-length 0.047 \
  --aruco-dict 4X4_50
```

**How auto-capture works:**
1. Position the board so it's visible in both camera views
2. A **countdown timer (3...2...1)** appears when board is detected
3. Hold the board **perfectly still** during countdown
4. Image captures automatically with green flash
5. Move to next position and repeat
6. Process exits automatically when target is reached

**Advantages of auto-capture:**
- ✓ Single person can operate (no need to press keys while holding board)
- ✓ Consistent timing reduces motion blur
- ✓ Countdown allows you to stabilize the board
- ✓ Faster workflow
- ✓ Better quality (less camera shake)

**During capture (both modes):**
- Both cameras display side-by-side in real-time
- Move the board to different positions:
  - Left, right, top, bottom, center
  - Near and far (vary depth)
  - Tilted at different angles (not just frontal)
- Press **Q** to quit anytime
- In manual mode: Press **SPACE** when ready

**Critical for accuracy:**
- Capture 30+ image pairs minimum
- Ensure board is fully visible in BOTH cameras
- Hold steady during countdown/capture (avoid motion blur)
- Cover the entire field of view with different board positions

### 3. Calibrate Cameras

```bash
# Using your current 3×4 board
python charuco_stereo_calibrate.py \
  --input-dir ./charuco_captures \
  --output charuco_calibration.npz \
  --squares-x 3 \
  --squares-y 4 \
  --square-length 0.063 \
  --marker-length 0.047 \
  --aruco-dict 4X4_50
```

**Outputs:**
- `charuco_calibration.npz` - NumPy format for use with `process.py`
- `charuco_calibration.yaml` - YAML format for inspection/compatibility

**Expected quality:**
- **Excellent**: Reprojection error < 0.5 pixels
- **Good**: Reprojection error < 1.0 pixels
- **Acceptable**: Reprojection error < 2.0 pixels (consider recapturing)

## Finding Camera Serial Numbers

For two-stage calibration, you need to know which camera is which. Find serial numbers using:

```bash
# List all connected RealSense devices
rs-enumerate-devices

# Or let the script show available cameras
python charuco_intrinsic_capture.py --camera-id 1 --output-dir ./test
# (Press Q immediately after seeing camera list)
```

**Save these serial numbers** - you'll need them for all three capture sessions to ensure consistency.

## Understanding the Parameters

### Board Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--squares-x` | Number of squares horizontally | 3 (your board) or 6 (recommended) |
| `--squares-y` | Number of squares vertically | 4 (your board) or 8 (recommended) |
| `--square-length` | Square side length in **meters** | 0.063 (63mm) |
| `--marker-length` | ArUco marker side length in **meters** | 0.047 (47mm) |
| `--aruco-dict` | ArUco dictionary type | 4X4_50 |

**IMPORTANT:** Parameters must **exactly match** your physical board. Measure with a ruler!

### Camera Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--cam-config` | Path to camera config JSON file | `./camera_config.json` |
| `--width` | Image width | 1280 |
| `--height` | Image height | 720 |
| `--fps` | Frames per second | 30 |

## Output Format

### NPZ File Contents

```python
import numpy as np

calib = np.load('charuco_calibration.npz')

# Intrinsic parameters
K1 = calib['K1']          # Camera 1 intrinsic matrix (3×3)
dist1 = calib['dist1']    # Camera 1 distortion coefficients (5,)
K2 = calib['K2']          # Camera 2 intrinsic matrix (3×3)
dist2 = calib['dist2']    # Camera 2 distortion coefficients (5,)

# Extrinsic parameters (relative pose)
R = calib['R']            # Rotation matrix from cam1 to cam2 (3×3)
T = calib['T']            # Translation vector from cam1 to cam2 (3,1)

# Image size
image_size = calib['image_size']  # (width, height)
```

### Using with `process.py`

The NPZ calibration file is directly compatible with your existing pipeline:

```bash
python process.py \
  --video1 recording1.mp4 \
  --video2 recording2.mp4 \
  --calib charuco_calibration.npz \
  --model ./models/pose_landmarker_heavy.task \
  --output-dir ./outputs \
  --vis-3d
```

## Adjusting Auto-Capture Timing

The default auto-capture settings work well for most scenarios:
- **3 second countdown**: Time to stabilize the board after detection
- **4 second interval**: Total time between captures (includes countdown)

To adjust timing:

```bash
# Faster capture (2 second interval) - use if you can move board quickly
python charuco_stereo_capture.py \
  --auto-capture \
  --capture-interval 2.0 \
  ... other parameters ...

# Slower capture (6 second interval) - use for more deliberate positioning
python charuco_stereo_capture.py \
  --auto-capture \
  --capture-interval 6.0 \
  ... other parameters ...
```

**Note:** The countdown is always 3 seconds. The `--capture-interval` sets the minimum time between successive captures.

## Troubleshooting

### "Found only 1 RealSense camera"

**Problem:** Script can't detect both cameras
**Solutions:**
- Connect both cameras to different USB 3.0 ports
- Check `rs-enumerate-devices` command to verify both are detected
- Try unplugging and reconnecting cameras
- Ensure cameras are not being used by another process

### "No board detected"

**Problem:** ChArUco corners not being detected
**Solutions:**
- Ensure good lighting (avoid strong shadows or glare)
- Check that board parameters match your physical board
- Verify ArUco dictionary is correct
- Make sure board is flat and not curved/wrinkled
- Move closer to the camera

### "Only X corners detected (need Y)"

**Problem:** Partial board detection
**Solutions:**
- Move closer to camera to make board larger in frame
- Ensure entire board is visible (not cut off at edges)
- Improve lighting conditions
- Use a larger board with more squares

### High reprojection error (> 2.0 pixels)

**Problem:** Poor calibration quality
**Solutions:**
- Capture more images (30+ pairs minimum)
- Ensure better coverage of field of view
- Include more tilted/angled views (not just frontal)
- Verify board measurements are accurate
- Use a larger board (6×8 or 8×11 instead of 3×4)
- Re-capture with steadier holds (avoid motion blur)

## Comparison: AprilTag vs ChArUco

| Aspect | AprilTag (your current method) | ChArUco (this method) |
|--------|-------------------------------|----------------------|
| **Accuracy** | Centroid-based (good) | Sub-pixel corners (excellent) |
| **Robustness** | Each tag independent | Can detect partial boards |
| **Points per capture** | 4-8 tag centroids | 6-80+ corner points |
| **Manufacturing** | Multiple tags, precise placement needed | Single printout |
| **Partial occlusion** | Affected | Robust |
| **Setup complexity** | Need tag layout definition | Just print and measure |

## Advanced: Verifying Calibration Quality

### Check intrinsic parameters

```python
import numpy as np

calib = np.load('charuco_calibration.npz')

print("Camera 1 Intrinsics:")
print(f"  Focal length: fx={calib['K1'][0,0]:.2f}, fy={calib['K1'][1,1]:.2f}")
print(f"  Principal point: cx={calib['K1'][0,2]:.2f}, cy={calib['K1'][1,2]:.2f}")
print(f"  Distortion: {calib['dist1'].flatten()}")

print("\nCamera 2 Intrinsics:")
print(f"  Focal length: fx={calib['K2'][0,0]:.2f}, fy={calib['K2'][1,1]:.2f}")
print(f"  Principal point: cx={calib['K2'][0,2]:.2f}, cy={calib['K2'][1,2]:.2f}")
print(f"  Distortion: {calib['dist2'].flatten()}")
```

### Check stereo extrinsics

```python
import cv2
import numpy as np

calib = np.load('charuco_calibration.npz')

R = calib['R']
T = calib['T']

# Baseline (distance between cameras)
baseline = np.linalg.norm(T)
print(f"Baseline: {baseline:.4f} meters ({baseline*100:.2f} cm)")

# Rotation angle
rvec, _ = cv2.Rodrigues(R)
angle = np.linalg.norm(rvec) * 180 / np.pi
axis = rvec.flatten() / np.linalg.norm(rvec)
print(f"Rotation: {angle:.2f}° around axis {axis}")
```

**Sanity checks:**
- Baseline should match approximate physical distance between cameras
- If cameras are roughly parallel, rotation angle should be small (< 10°)
- Focal lengths should be similar for both cameras (same model)

## File Organization

```
calibration/
├── charuco_intrinsic_capture.py       # NEW: Single-camera capture for intrinsics
├── charuco_stereo_capture.py          # Capture synchronized stereo pairs
├── charuco_two_stage_calibrate.py     # NEW: Two-stage calibration (recommended)
├── charuco_stereo_calibrate.py        # Single-stage calibration (legacy)
├── generate_charuco_board.py          # Generate larger boards
├── README_CHARUCO.md                  # This file
│
├── intrinsic_cam1/                    # Intrinsic captures for camera 1
│   ├── capture_000_*.png
│   ├── capture_001_*.png
│   └── ...
│
├── intrinsic_cam2/                    # Intrinsic captures for camera 2
│   ├── capture_000_*.png
│   ├── capture_001_*.png
│   └── ...
│
├── stereo_captures/                   # Stereo pair captures
│   ├── camera_1/
│   │   ├── capture_000_*.png
│   │   └── ...
│   └── camera_2/
│       ├── capture_000_*.png
│       └── ...
│
├── charuco_calibration.npz            # Final calibration (NPZ)
├── charuco_calibration.yaml           # Final calibration (YAML)
│
└── boards/                            # Generated boards
    ├── charuco_6x8.png
    ├── charuco_6x8.txt                # Parameters file
    ├── charuco_8x11.png
    └── charuco_8x11.txt
```

## Recommendations for Best Results

### Calibration Method:
1. ✓ **Use two-stage workflow** for best accuracy (especially for gaze estimation)
2. ✓ Auto-capture mode (single person friendly, less camera shake)
3. ✓ Ensure you cover ALL edges and corners during intrinsic capture
4. ✗ Avoid single-stage workflow unless you need quick calibration

### For your current 3×4 board (with two-stage workflow):
1. ✓ Intrinsic capture: **30 images per camera** with full FOV coverage
2. ✓ Stereo capture: **25 image pairs** in overlapping region
3. ✓ Include extreme angles (tilted up to 45° in all directions)
4. ✓ Vary depth significantly (board filling 30%-80% of frame)

### When you get a larger board:
1. ✓ **6×8 on A4**: Good balance (35 corners, fits standard printer)
2. ✓ **8×11 on A3**: Excellent (70 corners, professional quality)
3. ✓ Two-stage workflow becomes even more accurate
4. ✓ Can reduce capture counts to 25 per camera (intrinsic) + 20 pairs (stereo)

## Integration with Existing Workflow

Your existing scripts use AprilTag calibration:
- `calibration/stero_calibr.py`
- `calibration/stereo_calibr_w_coord.py`

**ChArUco with two-stage workflow advantages:**
- ✓ Best sub-pixel accuracy (critical for gaze estimation precision)
- ✓ Full FOV calibration (better distortion modeling at edges)
- ✓ Easier board manufacturing (just print, no physical tag layout needed)
- ✓ More robust detection (handles partial occlusion)
- ✓ More calibration points per capture

**To switch to ChArUco (two-stage workflow):**
1. Capture intrinsic images: `charuco_intrinsic_capture.py` (camera 1 and 2)
2. Capture stereo images: `charuco_stereo_capture.py`
3. Calibrate: `charuco_two_stage_calibrate.py`
4. Use the resulting NPZ file directly with `process.py` (same format!)

**To switch to ChArUco (single-stage fallback):**
1. Capture stereo images: `charuco_stereo_capture.py`
2. Calibrate: `charuco_stereo_calibrate.py` or `charuco_two_stage_calibrate.py`
3. Use with `process.py`

No code changes needed in `process.py` - the calibration format is identical.
