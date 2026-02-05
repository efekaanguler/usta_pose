import os
import glob
import cv2
import numpy as np
import torch
import mmcv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from mmpose.utils import register_all_modules
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples

# Model paths
CFG_PATH = "./../../../rtmw3dl/rtmw3dl_config.py"
CKPT_PATH = "./../../../rtmw3dl/rtmw3dl_ckpt.pth"
INPUT_DIR = "./../../testing/UDIVA/"
OUTPUT_DIR = "./../../testing/rtmw3dl/"

# Import custom modules
from rtmpose3d import *  # noqa: F401, F403

# COCO-WholeBody skeleton connections (133 keypoints)
SKELETON = [
    # Body (17 keypoints)
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    # Face connections (simplified)
    (0, 17), (0, 18), (17, 18),
    # Left hand connections (simplified)
    (91, 92), (92, 93), (93, 94), (94, 95),
    (91, 96), (96, 97), (97, 98), (98, 99),
    (91, 100), (100, 101), (101, 102), (102, 103),
    (91, 104), (104, 105), (105, 106), (106, 107),
    (91, 108), (108, 109), (109, 110), (110, 111),
    # Right hand connections (simplified)
    (112, 113), (113, 114), (114, 115), (115, 116),
    (112, 117), (117, 118), (118, 119), (119, 120),
    (112, 121), (121, 122), (122, 123), (123, 124),
    (112, 125), (125, 126), (126, 127), (127, 128),
    (112, 129), (129, 130), (130, 131), (131, 132),
]


def pick_attr(obj, names):
    """Helper to pick first available attribute"""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def draw_keypoints(frame, keypoints_2d, scores=None, threshold=0.3):
    """Draw keypoints and skeleton on frame"""
    h, w = frame.shape[:2]
    
    if keypoints_2d is None or len(keypoints_2d) == 0:
        return frame
    
    # Draw skeleton
    for start_idx, end_idx in SKELETON:
        if start_idx >= len(keypoints_2d) or end_idx >= len(keypoints_2d):
            continue
        
        if scores is not None:
            if (start_idx >= len(scores) or end_idx >= len(scores) or 
                scores[start_idx] < threshold or scores[end_idx] < threshold):
                continue
        
        start_point = (int(keypoints_2d[start_idx][0]), int(keypoints_2d[start_idx][1]))
        end_point = (int(keypoints_2d[end_idx][0]), int(keypoints_2d[end_idx][1]))
        
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    
    # Draw keypoints
    for i, kpt in enumerate(keypoints_2d):
        if scores is not None and i < len(scores) and scores[i] < threshold:
            continue
        
        x, y = int(kpt[0]), int(kpt[1])
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    
    return frame


def draw_3d_skeleton(keypoints_3d, scores, h, w, threshold=0.3, axis_limits=None):
    """Draw 3D skeleton projection using matplotlib"""
    # Create blank canvas
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    if keypoints_3d is None or len(keypoints_3d) == 0:
        return canvas
    
    # Create matplotlib figure
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract 3D coordinates
    xs = keypoints_3d[:, 0]
    ys = keypoints_3d[:, 1]
    zs = keypoints_3d[:, 2]
    
    # Draw connections
    for start_idx, end_idx in SKELETON:
        if start_idx >= len(keypoints_3d) or end_idx >= len(keypoints_3d):
            continue
        
        if scores is not None:
            if (start_idx >= len(scores) or end_idx >= len(scores) or 
                scores[start_idx] < threshold or scores[end_idx] < threshold):
                continue
        
        ax.plot([xs[start_idx], xs[end_idx]], 
               [ys[start_idx], ys[end_idx]], 
               [zs[start_idx], zs[end_idx]], 'b-', linewidth=2)
    
    # Draw keypoints
    valid_points = []
    for i in range(len(keypoints_3d)):
        if scores is None or (i < len(scores) and scores[i] >= threshold):
            valid_points.append([xs[i], ys[i], zs[i]])
    
    if valid_points:
        valid_points = np.array(valid_points)
        ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], 
                  c='g', marker='o', s=20)
    
    # Set labels and viewing angle
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=15, azim=70)  # Demo'daki açılar
    
    # Use fixed axis limits if provided, otherwise calculate dynamically
    if axis_limits is not None:
        ax.set_xlim(axis_limits['x'])
        ax.set_ylim(axis_limits['y'])
        ax.set_zlim(axis_limits['z'])
    else:
        # Set reasonable limits based on data
        if len(keypoints_3d) > 0:
            x_range = max(xs.max() - xs.min(), 0.5)
            y_range = max(ys.max() - ys.min(), 0.5)
            z_range = max(zs.max() - zs.min(), 0.5)
            max_range = max(x_range, y_range, z_range) * 0.6
            
            x_center = (xs.max() + xs.min()) / 2
            y_center = (ys.max() + ys.min()) / 2
            z_center = (zs.max() + zs.min()) / 2
            
            ax.set_xlim([x_center - max_range, x_center + max_range])
            ax.set_ylim([y_center - max_range, y_center + max_range])
            ax.set_zlim([z_center - max_range, z_center + max_range])
    
    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Resize to match frame size
    img = cv2.resize(img, (w, h))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def process_video(video_path, output_path, model, device):
    """Process a single video file with side-by-side view"""
    # Fixed axis limits for consistent 3D visualization
    axis_limits = {
        'x': [-220, -120],
        'y': [-200, -100],
        'z': [0, 80]
    }
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer for side-by-side output (2x width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    # Full image bbox (single person assumption)
    bboxes = np.array([[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32)
    
    frame_count = 0
    detected_poses = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for mmpose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        pose_est_results = inference_topdown(model, rgb_frame, bboxes=bboxes)
        
        # Post-process results (like in demo)
        for idx, pose_est_result in enumerate(pose_est_results):
            pred_instances = pose_est_result.pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            
            # Handle score dimensions
            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pose_est_results[idx].pred_instances.keypoint_scores = keypoint_scores
            
            # Handle keypoint dimensions
            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)
            
            # !! CRITICAL: Transform axes (like in demo)
            # Model outputs in different coordinate system
            keypoints_3d = -keypoints[..., [0, 2, 1]]  # (x,y,z) → (-x,z,y)
            
            # Rebase height (make lowest point touch ground)
            keypoints_3d[..., 2] -= np.min(keypoints_3d[..., 2], axis=-1, keepdims=True)
            
            pose_est_results[idx].pred_instances.keypoints = keypoints_3d
        
        # Merge all results
        pred_3d_data_samples = merge_data_samples(pose_est_results)
        pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)
        
        # DEBUG: First frame
        if frame_count == 0:
            print(f"  DEBUG - First frame:")
            print(f"    Results count: {len(pose_est_results)}")
            if pred_3d_instances is not None:
                k3d = pred_3d_instances.keypoints
                sc = pred_3d_instances.keypoint_scores
                print(f"    3D Keypoints shape: {k3d.shape}")
                print(f"    3D range: x=[{k3d[..., 0].min():.2f}, {k3d[..., 0].max():.2f}]")
                print(f"              y=[{k3d[..., 1].min():.2f}, {k3d[..., 1].max():.2f}]")
                print(f"              z=[{k3d[..., 2].min():.2f}, {k3d[..., 2].max():.2f}]")
                print(f"    Scores shape: {sc.shape}")
                print(f"    Score range: [{sc.min():.3f}, {sc.max():.3f}]")
        
        # Left side: Original + 2D overlay (project 3D to 2D)
        left_frame = frame.copy()
        right_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        has_detection = False
        
        if pred_3d_instances is not None:
            k3d = pred_3d_instances.keypoints
            sc = pred_3d_instances.keypoint_scores
            
            if k3d is not None and len(k3d) > 0:
                has_detection = True
                
                # For 2D visualization, use original x,y from 3D (before transform)
                # Or project 3D back to 2D
                for person_idx in range(len(k3d)):
                    k3d_person = k3d[person_idx]
                    sc_person = sc[person_idx] if sc is not None else None
                    
                    # Simple 2D projection (orthographic)
                    k2d = k3d_person[:, [0, 1]]  # Just take x,y
                    # Scale to image size (assuming normalized coords)
                    if k2d[:, 0].max() <= 10:  # If small values
                        k2d[:, 0] = (k2d[:, 0] + 2) * width / 4
                        k2d[:, 1] = (k2d[:, 1] + 2) * height / 4
                    
                    left_frame = draw_keypoints(left_frame, k2d, sc_person)
                    right_frame = draw_3d_skeleton(k3d_person, sc_person, height, width, axis_limits=axis_limits)
        
        if has_detection:
            detected_poses += 1
        
        # Combine side-by-side
        combined_frame = np.hstack([left_frame, right_frame])
        
        # Save first frame as debug sample
        if frame_count == 0:
            debug_output = os.path.join(OUTPUT_DIR, f"debug_{os.path.basename(video_path).replace('.mp4', '_frame0.jpg')}")
            cv2.imwrite(debug_output, combined_frame)
            print(f"    💾 Saved first frame to {debug_output}\n")
        
        # Write frame
        out.write(combined_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames (poses detected: {detected_poses})")
    
    cap.release()
    out.release()
    print(f"  Total frames processed: {frame_count}, Frames with poses: {detected_poses}")


def main():
    # Setup device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all mp4 files
    video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    video_files.sort()
    
    if not video_files:
        print(f"No mp4 files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(video_files)} video(s) to process")
    print(f"Config: {CFG_PATH}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Initialize model
    register_all_modules()
    model = init_model(CFG_PATH, CKPT_PATH, device=device)
    
    # !! CRITICAL: Set test mode to 'vis' for 3D output
    model.cfg.model.test_cfg.mode = 'vis'
    
    print("Model loaded successfully!")
    print(f"Test mode: {model.cfg.model.test_cfg.mode}\n")
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        output_path = os.path.join(OUTPUT_DIR, video_name)
        
        print(f"[{i}/{len(video_files)}] Processing: {video_name}")
        process_video(video_path, output_path, model, device)
        print(f"[{i}/{len(video_files)}] Saved to: {output_path}\n")
    
    print("All videos processed successfully!")


if __name__ == "__main__":
    main()