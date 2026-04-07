"""
Process MKV Depth + MP4 Color Pose Pipeline

This script loops through 4-camera recordings, detects human pose keypoints 
using MediaPipe, and deprojects them into 3D using the 16-bit MKV depth video.
It outputs a JSON array containing the 3D values into a new CSV column, mapped
to the global standard coordinate frame.

Usage:
  python devel/process_mkv_pose.py --session-dir devel/recordings/session_YYYYMMDD_HHMMSS
  
By default, this will look for `multicam_calibration.npz` in your session folder.
"""
import cv2
import numpy as np
import imageio.v3 as iio
import mediapipe as mp
import json
import os
import argparse
import pandas as pd
from tqdm import tqdm

def get_calib(calib_path, cam_num):
    """Load calibration matrices for transforming cam coordinate to global (ref) coordinate"""
    calib = np.load(calib_path)
    # Check if this cam has R and t
    R_key = f'R_{cam_num}_to_ref'
    t_key = f't_{cam_num}_to_ref'
    if R_key in calib and t_key in calib:
        R_ref_to_cam = calib[R_key]
        t_ref_to_cam = calib[t_key]
        # Invert to go from cam to ref
        R_cam_to_ref = R_ref_to_cam.T
        t_cam_to_ref = -R_ref_to_cam.T @ t_ref_to_cam
        return R_cam_to_ref, t_cam_to_ref
    print(f"Warning: Calibration for cam {cam_num} not found. Returning Identity.")
    return np.eye(3), np.zeros(3)

def deproject_pixel_to_3d(x, y, depth_image, K, depth_scale):
    """Deproject 2D pixel to 3D point in camera frame using depth image."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Squeeze out extra channel dimension if present (e.g. from (H, W, 1) to (H, W))
    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]
        
    h, w = depth_image.shape
    ix, iy = int(round(x)), int(round(y))
    
    r = 2 # 5x5 patch roughly
    y_lo = max(0, iy - r)
    y_hi = min(h, iy + r + 1)
    x_lo = max(0, ix - r)
    x_hi = min(w, ix + r + 1)
    
    patch = depth_image[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
    valid = patch[patch > 0]
    
    if len(valid) == 0:
        # Debug why no valid pixels
        if x_lo == x_hi or y_lo == y_hi:
            pass # out of bounds completely
        else:
            # Patch is entirely zeroes!
            pass
        return [np.nan, np.nan, np.nan, 0.0]
        
    depth_val = np.median(valid)
    Z = depth_val * depth_scale
    
    if Z < 0.1 or Z > 10.0:
        return [np.nan, np.nan, np.nan, Z]
        
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return [X, Y, Z, Z]

def process_camera(cam_num, session_dir, calib_npz_path, model_path, save_video=False):
    print(f"\nProcessing Camera {cam_num}...")
    cam_dir = os.path.join(session_dir, f"cam{cam_num}")
    color_vid = os.path.join(cam_dir, "color.mp4")
    depth_vid = os.path.join(cam_dir, "depth.mkv")
    ts_csv = os.path.join(cam_dir, f"cam{cam_num}_color_timestamps.csv")
    meta_path = os.path.join(session_dir, "metadata.json")
    
    if not all(os.path.exists(p) for p in [color_vid, depth_vid, ts_csv, meta_path]):
        print(f"Missing files for cam{cam_num}, skipping. Make sure session dir is correct.")
        return
        
    df = pd.read_csv(ts_csv)
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    cam_meta = meta['cameras'][str(cam_num)]
    depth_scale = cam_meta['depth_storage']['depth_scale_meters_per_unit']
    intr = cam_meta['intrinsics']
    K = np.array([
        [intr['fx'], 0, intr['ppx']],
        [0, intr['fy'], intr['ppy']],
        [0, 0, 1]
    ])
    
    R_cam_to_ref, t_cam_to_ref = get_calib(calib_npz_path, cam_num)
    
    # Initialize MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
    from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
    from mediapipe.tasks.python.vision import pose_landmarker as mp_pose_landmarker
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    pose_arrays = []
    
    import av
    cap = cv2.VideoCapture(color_vid)
    try:
        depth_container = av.open(depth_vid)
        depth_stream = depth_container.streams.video[0]
        depth_iter = depth_container.decode(depth_stream)
    except Exception as e:
        print(f"Error opening depth_vid {depth_vid}: {e}")
        return
        
    out_video = None
    if save_video:
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        out_vid_path = os.path.join(cam_dir, f"cam{cam_num}_pose.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_vid_path, fourcc, vid_fps, (vid_w, vid_h))
        
    with PoseLandmarker.create_from_options(options) as landmarker:
        for i in tqdm(range(len(df)), desc=f"Cam {cam_num} Frames"):
            ret, frame = cap.read()
            try:
                depth_frame_av = next(depth_iter)
                depth_frame = depth_frame_av.to_ndarray()
            except (StopIteration, av.AVError):
                depth_frame = None
                
            if not ret or depth_frame is None:
                pose_arrays.append(json.dumps([[None, None, None]] * 33))
                if save_video and out_video is not None and ret:
                    out_video.write(frame)
                continue
                
            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            result = landmarker.detect(mp_image)
            
            frame_pose_3d = []
            if not result.pose_landmarks:
                if i < 30:
                    print(f"Frame {i}: No pose detected by MediaPipe.")
                frame_pose_3d = [[None, None, None]] * 33
            else:
                landmarks = result.pose_landmarks[0]
                h, w, _ = frame.shape
                
                local_3d_points = []
                for lm in landmarks:
                    px = lm.x * w
                    py = lm.y * h
                    p3d = deproject_pixel_to_3d(px, py, depth_frame, K, depth_scale)
                    local_3d_points.append(p3d[:3])
                
                local_3d_points = np.array(local_3d_points)
                global_3d_points = np.full_like(local_3d_points, np.nan)
                
                # Transform to global
                for idx in range(33):
                    if not np.isnan(local_3d_points[idx, 0]):
                        p_global = R_cam_to_ref @ local_3d_points[idx] + t_cam_to_ref
                        global_3d_points[idx] = p_global
                        
                frame_pose_3d = global_3d_points.tolist()
                
                # Debug print for first few frames
                if i < 30:
                    valid_pts = np.sum(~np.isnan(local_3d_points[:, 0]))
                    if valid_pts == 0:
                        print(f"Frame {i}: MediaPipe found pose, but local_3d_points are ALL NaNs (Z out of bounds).")
                        
                if save_video and out_video is not None:
                    # Natively feed the normalized landmark components directly provided by modern tasks API
                    mp_drawing.draw_landmarks(
                        frame,
                        landmarks,
                        mp_pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
                
            if save_video and out_video is not None:
                out_video.write(frame)
                
            # Convert the array to a JSON string formatted payload
            pose_arrays.append(json.dumps(frame_pose_3d))
            
    cap.release()
    if out_video is not None:
        out_video.release()
    depth_container.close()
    
    # Save to CSV
    df['pose_3d_array_global'] = pose_arrays
    out_csv = os.path.join(cam_dir, f"cam{cam_num}_pose_timestamps.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv} containing the [33 x 3] 3D array in the 'pose_3d_array_global' column.\n")

def main():
    parser = argparse.ArgumentParser(description="Process depth MKVs + Pose")
    parser.add_argument('--session-dir', type=str, required=True, help='Path to the session_YYMMDD_HHMMSS folder')
    parser.add_argument('--calib-npz', type=str, default=None, help='Path to multi_camera_calibration.npz (defaults to session-dir/multicam_calibration.npz)')
    parser.add_argument('--model', type=str, default='/workspace/mediapipe/pose_landmarker_full.task', help='Path to MediaPipe model.task')
    parser.add_argument('--save-video', action='store_true', help='Generate .mp4 visualization videos in each cam directory')
    args = parser.parse_args()
    
    # If calib-npz is not provided, default to the one inside session-dir
    if args.calib_npz is None:
        args.calib_npz = os.path.join(args.session_dir, "multicam_calibration.npz")
    
    # Process pose exclusively for cameras 1 and 2
    for cam_num in [1, 2]:
        process_camera(cam_num, args.session_dir, args.calib_npz, args.model, args.save_video)
        
    print("Process complete! Check the pose_timestamps.csv in each camera's directory.")

if __name__ == '__main__':
    main()
