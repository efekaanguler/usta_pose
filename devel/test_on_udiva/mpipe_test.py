import os
import glob
import cv2
import mediapipe as mp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Model path
MODEL_PATH = "./../../../mediapipe/pose_landmarker_full.task"
INPUT_DIR = "./../../testing/UDIVA/"
OUTPUT_DIR = "./../../testing/mediapipe/"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks(frame, landmarks, h, w):
    """Draw pose landmarks on frame"""
    # MediaPipe pose connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
        (27, 31), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
    ]
    
    for lm in landmarks:
        # Draw keypoints
        for landmark in lm:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(lm) and end_idx < len(lm):
                start_x = int(lm[start_idx].x * w)
                start_y = int(lm[start_idx].y * h)
                end_x = int(lm[end_idx].x * w)
                end_y = int(lm[end_idx].y * h)
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
    
    return frame


def draw_3d_skeleton(world_landmarks, h, w):
    """Draw 3D skeleton projection using matplotlib"""
    # Create blank canvas
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    if not world_landmarks or len(world_landmarks) == 0:
        return canvas
    
    # MediaPipe pose connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
        (27, 31), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
    ]
    
    # Create matplotlib figure
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    for lm in world_landmarks:
        # Extract 3D coordinates
        xs = [landmark.x for landmark in lm]
        ys = [landmark.y for landmark in lm]
        zs = [landmark.z for landmark in lm]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(lm) and end_idx < len(lm):
                ax.plot([xs[start_idx], xs[end_idx]], 
                       [ys[start_idx], ys[end_idx]], 
                       [zs[start_idx], zs[end_idx]], 'b-', linewidth=2)
        
        # Draw keypoints
        ax.scatter(xs, ys, zs, c='g', marker='o', s=20)
    
    # Set labels and viewing angle
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=15, azim=45)
    
    # Set equal aspect ratio
    max_range = 1.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    
    # Resize to match frame size
    img = cv2.resize(img, (w, h))
    return img


def process_video(video_path, output_path, landmarker):
    """Process a single video file with side-by-side view"""
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
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect poses
        result = landmarker.detect(mp_image)
        
        # Left side: Original + 2D overlay
        left_frame = frame.copy()
        if result.pose_landmarks:
            left_frame = draw_landmarks(left_frame, result.pose_landmarks, height, width)
        
        # Right side: 3D skeleton projection
        right_frame = draw_3d_skeleton(result.pose_world_landmarks, height, width)
        
        # Combine side-by-side
        combined_frame = np.hstack([left_frame, right_frame])
        
        # Write frame
        out.write(combined_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    print(f"  Total frames processed: {frame_count}")


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all mp4 files
    video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    video_files.sort()
    
    if not video_files:
        print(f"No mp4 files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(video_files)} video(s) to process")
    print(f"Model: {MODEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Initialize MediaPipe Pose Landmarker
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=5,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        for i, video_path in enumerate(video_files, 1):
            video_name = os.path.basename(video_path)
            output_path = os.path.join(OUTPUT_DIR, video_name)
            
            print(f"[{i}/{len(video_files)}] Processing: {video_name}")
            process_video(video_path, output_path, landmarker)
            print(f"[{i}/{len(video_files)}] Saved to: {output_path}\n")
    
    print("All videos processed successfully!")


if __name__ == "__main__":
    main()
