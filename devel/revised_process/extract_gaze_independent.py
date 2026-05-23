#!/usr/bin/env python3
"""
Step 2: Independent Gaze Processing (Cam-Specific)

Runs PureGaze and FaceDetection on ALL frames of a single gaze camera 
based on its hardware timestamps. Extracts the 3D unit gaze vector.
NO global transformation is applied at this stage.

Usage:
    python extract_gaze_independent.py \
        --session-dir ./recordings/session_YYYYMMDD_HHMMSS \
        --cam-id 3

Output:
    {session_dir}/cam3/cam3_gaze_raw.csv
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PureGaze model wrapper
# ---------------------------------------------------------------------------

class PureGazeInference:
    def __init__(self, weights_path, model_module_dir, device="cuda:0"):
        import torch
        from torchvision import transforms

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if model_module_dir not in sys.path:
            sys.path.insert(0, model_module_dir)

        try:
            from model.model import Model
        except ImportError:
            print("Warning: PureGaze model module not found. Make sure USTA-Human-Interaction-Analysis is accessible.")
            self.model = None
            return

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"PureGaze weights not found: {weights_path}")

        net = Model()
        state_dict = torch.load(weights_path, map_location=self.device)
        net.load_state_dict(state_dict)
        net.to(self.device)
        net.eval()
        self.model = net

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.torch = torch
        print(f"PureGaze model loaded on {self.device}")

    def estimate(self, head_crop_bgr):
        if self.model is None or head_crop_bgr is None or head_crop_bgr.size == 0:
            return None, None

        head_crop_rgb = cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(head_crop_rgb).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            outputs, _ = self.model({"face": input_tensor}, require_img=False)
            pred = outputs.cpu().numpy()[0]

        if pred.shape[0] != 2:
            return None, None

        eth_pitch, eth_yaw = pred
        return float(eth_yaw), float(eth_pitch)

    @staticmethod
    def pitch_yaw_to_unit_vector(pitch, yaw):
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        return np.array([x, y, z], dtype=np.float64)


# ---------------------------------------------------------------------------
# Face detection via MediaPipe
# ---------------------------------------------------------------------------

class FaceDetector:
    def __init__(self, min_confidence=0.5):
        import mediapipe as mp
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1, 
            min_detection_confidence=min_confidence,
        )

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None

        best = max(results.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box

        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def close(self):
        self.detector.close()


# ---------------------------------------------------------------------------
# Frame reader & helpers
# ---------------------------------------------------------------------------

class VideoFrameReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

    def read_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        self.cap.release()

def load_timestamps(csv_path):
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'frame_idx': int(row['frame_idx']),
                'hw_timestamp_ms': float(row['hw_timestamp_ms'])
            })
    rows.sort(key=lambda x: x['frame_idx'])
    return rows


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_gaze_camera(session_dir, cam_id, gaze_model, face_detector):
    cam_dir = os.path.join(session_dir, f"cam{cam_id}")
    ts_csv_path = os.path.join(cam_dir, f"cam{cam_id}_color_timestamps.csv")
    
    if not os.path.exists(ts_csv_path):
        raise FileNotFoundError(f"Timestamps file not found: {ts_csv_path}")

    video_reader = VideoFrameReader(os.path.join(cam_dir, "color.mp4"))
    timestamps = load_timestamps(ts_csv_path)

    results = []

    for row in tqdm(timestamps, desc=f"Gaze processing Cam {cam_id}"):
        frame_idx = row['frame_idx']
        hw_ts = row['hw_timestamp_ms']

        result_row = {
            'frame_idx': frame_idx,
            'hw_timestamp_ms': hw_ts
        }

        frame = video_reader.read_frame(frame_idx)

        if frame is None:
            result_row['gaze_yaw'] = ''
            result_row['gaze_pitch'] = ''
            result_row['gaze_x'] = ''
            result_row['gaze_y'] = ''
            result_row['gaze_z'] = ''
            result_row['face_detected'] = 0
            results.append(result_row)
            continue

        face_bbox = face_detector.detect(frame)

        if face_bbox is None:
            result_row['gaze_yaw'] = ''
            result_row['gaze_pitch'] = ''
            result_row['gaze_x'] = ''
            result_row['gaze_y'] = ''
            result_row['gaze_z'] = ''
            result_row['face_detected'] = 0
            results.append(result_row)
            continue

        x1, y1, x2, y2 = face_bbox
        head_crop = frame[y1:y2, x1:x2]

        yaw, pitch = gaze_model.estimate(head_crop)

        if yaw is None:
            result_row['gaze_yaw'] = ''
            result_row['gaze_pitch'] = ''
            result_row['gaze_x'] = ''
            result_row['gaze_y'] = ''
            result_row['gaze_z'] = ''
            result_row['face_detected'] = 1
            results.append(result_row)
            continue

        gaze_vec = PureGazeInference.pitch_yaw_to_unit_vector(pitch, yaw)

        result_row['gaze_yaw'] = round(yaw, 6)
        result_row['gaze_pitch'] = round(pitch, 6)
        result_row['gaze_x'] = round(gaze_vec[0], 6)
        result_row['gaze_y'] = round(gaze_vec[1], 6)
        result_row['gaze_z'] = round(gaze_vec[2], 6)
        result_row['face_detected'] = 1

        results.append(result_row)

    video_reader.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run PureGaze on all frames independently",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--session-dir', type=str, required=True)
    parser.add_argument('--cam-id', type=int, required=True, help='Camera ID (e.g., 3 or 4)')
    parser.add_argument('--puregaze-weights', type=str, default=None)
    parser.add_argument('--puregaze-model-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    session_dir = args.session_dir
    cam_id = args.cam_id
    output_path = os.path.join(session_dir, f"cam{cam_id}", f"cam{cam_id}_gaze_raw.csv")

    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    devel_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(devel_dir)
    gaze_est_dir = os.path.join(project_root, "..", "USTA-Human-Interaction-Analysis", "gaze_estimation")

    weights_path = args.puregaze_weights or os.path.join(
        project_root, "models", "gaze", "res50_puregaze", "Res50_PureGaze_ETH.pt")
    model_dir = args.puregaze_model_dir or gaze_est_dir

    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"Session:    {session_dir}")
    print(f"Camera ID:  {cam_id}")
    print(f"Weights:    {weights_path}")
    print(f"Model dir:  {model_dir}")
    print(f"Device:     {device}")
    print()

    # Initialize models
    gaze_model = PureGazeInference(weights_path, model_dir, device=device)
    face_detector = FaceDetector(min_confidence=0.5)

    results = process_gaze_camera(session_dir, cam_id, gaze_model, face_detector)
    face_detector.close()

    # Write output
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nGaze results saved: {output_path}")
    else:
        print("No results generated.")

if __name__ == '__main__':
    main()
