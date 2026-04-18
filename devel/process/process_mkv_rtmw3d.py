import cv2
import numpy as np
import json
import os
import argparse
import pandas as pd
from tqdm import tqdm
import sys

# Import AV directly for the MKV depth frames
import av

# Import PyTorch and MMPose required modules
import torch
import mmcv
from mmpose.utils import register_all_modules
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples

# Add RTMPose3D specific paths
sys.path.append('/workspace/mmpose/projects/rtmpose3d')
try:
    import rtmpose3d
except ImportError as e:
    print("Warning: rtmpose3d module could not be imported. Custom modules for RTMW3D may be missing.")

def get_calib(calib_path, cam_num):
    """Load calibration matrices for transforming cam coordinate to global (ref) coordinate"""
    calib = np.load(calib_path)
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
    """Deproject 2D pixel to 3D point using 16-bit depth image."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
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
        return [np.nan, np.nan, np.nan, 0.0]
        
    depth_val = np.median(valid)
    Z = depth_val * depth_scale
    
    if Z < 0.1 or Z > 10.0:
        return [np.nan, np.nan, np.nan, Z]
        
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return [X, Y, Z, Z]

def process_camera(cam_num, session_dir, calib_npz_path, cfg_path, ckpt_path, device):
    print(f"\nProcessing Camera {cam_num} with RTMW3D...")
    cam_dir = os.path.join(session_dir, f"cam{cam_num}")
    color_vid = os.path.join(cam_dir, "color.mp4")
    depth_vid = os.path.join(cam_dir, "depth.mkv")
    ts_csv = os.path.join(cam_dir, f"cam{cam_num}_color_timestamps.csv")
    meta_path = os.path.join(session_dir, "metadata.json")
    
    if not all(os.path.exists(p) for p in [color_vid, depth_vid, ts_csv, meta_path]):
        print(f"Missing files for cam{cam_num}, skipping.")
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
    
    # Initialize MMPose RTMW3D Model
    register_all_modules()
    model = init_model(cfg_path, ckpt_path, device=device)
    model.cfg.model.test_cfg.mode = 'vis' # Crucial for 3D output mode
    print("RTMW3D Model loaded.")
    
    rtmw3d_2d_arrays = []
    rtmw3d_native_3d_arrays = []
    rtmw3d_global_3d_arrays = []
    
    cap = cv2.VideoCapture(color_vid)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bboxes = np.array([[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32)
    
    try:
        depth_container = av.open(depth_vid)
        depth_stream = depth_container.streams.video[0]
        depth_iter = depth_container.decode(depth_stream)
    except Exception as e:
        print(f"Error opening depth_vid {depth_vid}: {e}")
        return
        
    for i in tqdm(range(len(df)), desc=f"Cam {cam_num} Frames"):
        ret, frame = cap.read()
        try:
            depth_frame_av = next(depth_iter)
            depth_frame = depth_frame_av.to_ndarray()
        except (StopIteration, av.AVError):
            depth_frame = None
            
        if not ret or depth_frame is None:
            # Append empty frame formats
            empty_133_3d = [[None, None, None]] * 133
            empty_133_2d = [[None, None]] * 133
            rtmw3d_2d_arrays.append(json.dumps(empty_133_2d))
            rtmw3d_native_3d_arrays.append(json.dumps(empty_133_3d))
            rtmw3d_global_3d_arrays.append(json.dumps(empty_133_3d))
            continue
            
        # MMPose uses RGB frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        pose_est_results = inference_topdown(model, rgb_frame, bboxes=bboxes)
        
        if not pose_est_results or pose_est_results[0].pred_instances is None:
            empty_133_3d = [[None, None, None]] * 133
            empty_133_2d = [[None, None]] * 133
            rtmw3d_2d_arrays.append(json.dumps(empty_133_2d))
            rtmw3d_native_3d_arrays.append(json.dumps(empty_133_3d))
            rtmw3d_global_3d_arrays.append(json.dumps(empty_133_3d))
            continue
        
        pred_instances = pose_est_results[0].pred_instances
        keypoints = pred_instances.keypoints # (1, 133, 3) where Z is natively predicted
        
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1) # Deal with extra dimensions
            
        # Person 0 (since bbox is whole image, there should be max 1)
        kpts = keypoints[0] # (133, 3) 
        
        # 2D Array is just X,Y at pixel space
        arr_2d = kpts[:, :2].tolist()
        
        # Native 3D transformations for RTMW3D specific coordinate frame (taken from original demo)
        kpts_native_3d = -kpts[..., [0, 2, 1]]  # (x,y,z) → (-x,z,y)
        kpts_native_3d[..., 2] -= np.nanmin(kpts_native_3d[..., 2]) # Rebase touches ground
        arr_native_3d = kpts_native_3d.tolist()
        
        # Now create the Global Deprojected 3D using real MKV 16-bit depth
        local_3d_points = []
        for point_2d in kpts[:, :2]:
            px, py = point_2d[0], point_2d[1]
            if np.isnan(px) or np.isnan(py):
                local_3d_points.append([np.nan, np.nan, np.nan])
            else:
                p3d = deproject_pixel_to_3d(px, py, depth_frame, K, depth_scale)
                local_3d_points.append(p3d[:3])
                
        local_3d_points = np.array(local_3d_points)
        global_3d_points = np.full_like(local_3d_points, np.nan)
        
        # Transform to global
        for idx in range(133):
            if not np.isnan(local_3d_points[idx, 0]):
                p_global = R_cam_to_ref @ local_3d_points[idx] + t_cam_to_ref
                global_3d_points[idx] = p_global
                
        arr_global_3d = global_3d_points.tolist()
        
        # Save to JSON outputs
        rtmw3d_2d_arrays.append(json.dumps(arr_2d))
        rtmw3d_native_3d_arrays.append(json.dumps(arr_native_3d))
        rtmw3d_global_3d_arrays.append(json.dumps(arr_global_3d))
        
    cap.release()
    depth_container.close()
    
    # Save arrays to a NEW CSV mapping specifically for RTMW3D
    df['pose_rtmw3d_2d_array'] = rtmw3d_2d_arrays
    df['pose_rtmw3d_native_3d_array'] = rtmw3d_native_3d_arrays
    df['pose_rtmw3d_3d_array_global'] = rtmw3d_global_3d_arrays
    
    out_csv = os.path.join(cam_dir, f"cam{cam_num}_rtmw3d_timestamps.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv} containing 3 sets of 133 keypoint arrays.\n")

def main():
    parser = argparse.ArgumentParser(description="Process depth MKVs + RTMW3D Pose")
    parser.add_argument('--session-dir', type=str, required=True, help='Path to the session_YYMMDD_HHMMSS folder')
    parser.add_argument('--calib-npz', type=str, default=None, help='Path to multi_camera_calibration.npz')
    parser.add_argument('--cfg-path', type=str, default='/workspace/rtmw3dl/rtmw3dl_config.py', help='Path to RTMW3D py config file')
    parser.add_argument('--ckpt-path', type=str, default='/workspace/rtmw3dl/rtmw3dl_ckpt.pth', help='Path to RTMW3D pth checkpont file')
    args = parser.parse_args()
    
    if args.calib_npz is None:
        args.calib_npz = os.path.join(args.session_dir, "multicam_calibration.npz")
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    for cam_num in [1, 2]:
        process_camera(cam_num, args.session_dir, args.calib_npz, args.cfg_path, args.ckpt_path, device)
        
    print("Process complete! Check the rtmw3d_timestamps.csv in each camera's directory.")

if __name__ == '__main__':
    main()
