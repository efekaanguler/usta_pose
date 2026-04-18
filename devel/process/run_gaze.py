#!/usr/bin/env python3
"""
Step 3: Gaze Processing (cam3, cam4)

Runs PureGaze on matched frames from gaze cameras. Uses MediaPipe
FaceDetection to locate the head, then feeds the cropped face region
to PureGaze for (yaw, pitch) estimation → 3D unit gaze vector.

Usage:
    python run_gaze.py \\
        --session-dir ./recordings/session_YYYYMMDD_HHMMSS \\
        --matched-csv ./recordings/session_YYYYMMDD_HHMMSS/matched_frames.csv

Output:
    {session_dir}/gaze_results.csv
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PureGaze model wrapper (standalone, no class inheritance needed)
# ---------------------------------------------------------------------------

class PureGazeInference:
    """Lightweight PureGaze wrapper for single-image inference."""

    def __init__(self, weights_path, model_module_dir, device="cuda:0"):
        import torch
        from torchvision import transforms

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Add the model module directory to sys.path so `from model.model import Model` works
        if model_module_dir not in sys.path:
            sys.path.insert(0, model_module_dir)

        from model.model import Model

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
        """Estimate gaze from a BGR head crop.

        Returns:
            (yaw, pitch) in radians, or (None, None) if failed.
        """
        if head_crop_bgr is None or head_crop_bgr.size == 0:
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
        """Convert (pitch, yaw) to a 3D unit gaze vector."""
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        return np.array([x, y, z], dtype=np.float64)


# ---------------------------------------------------------------------------
# Face detection via MediaPipe
# ---------------------------------------------------------------------------

class FaceDetector:
    """MediaPipe-based face detector for extracting head crops."""

    def __init__(self, min_confidence=0.5):
        import mediapipe as mp
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,  # full-range model
            min_detection_confidence=min_confidence,
        )

    def detect(self, frame_bgr):
        """Detect the largest face and return (x1, y1, x2, y2) or None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None

        # Pick the detection with highest score
        best = max(results.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box

        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        # Add padding (20px)
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
# Frame reader
# ---------------------------------------------------------------------------

class VideoFrameReader:
    """Random-access reader for color.mp4."""

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


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_gaze_cameras(session_dir, matched_rows, gaze_model, face_detector,
                         gaze_cams):
    """Run PureGaze on gaze cameras for all matched frames."""

    results = []

    # Open video readers
    cam_readers = {}
    for cam_id in gaze_cams:
        cam_dir = os.path.join(session_dir, f"cam{cam_id}")
        cam_readers[cam_id] = VideoFrameReader(
            os.path.join(cam_dir, "color.mp4"))

    for row in tqdm(matched_rows, desc="Gaze processing"):
        result_row = {'master_frame_idx': int(row['master_frame_idx'])}

        for cam_id in gaze_cams:
            frame_idx = int(row[f'cam{cam_id}_idx'])
            frame = cam_readers[cam_id].read_frame(frame_idx)

            prefix = f'cam{cam_id}'

            if frame is None:
                result_row[f'{prefix}_gaze_yaw'] = ''
                result_row[f'{prefix}_gaze_pitch'] = ''
                result_row[f'{prefix}_gaze_x'] = ''
                result_row[f'{prefix}_gaze_y'] = ''
                result_row[f'{prefix}_gaze_z'] = ''
                result_row[f'{prefix}_face_detected'] = 0
                continue

            # Detect face
            face_bbox = face_detector.detect(frame)

            if face_bbox is None:
                result_row[f'{prefix}_gaze_yaw'] = ''
                result_row[f'{prefix}_gaze_pitch'] = ''
                result_row[f'{prefix}_gaze_x'] = ''
                result_row[f'{prefix}_gaze_y'] = ''
                result_row[f'{prefix}_gaze_z'] = ''
                result_row[f'{prefix}_face_detected'] = 0
                continue

            x1, y1, x2, y2 = face_bbox
            head_crop = frame[y1:y2, x1:x2]

            # Run PureGaze
            yaw, pitch = gaze_model.estimate(head_crop)

            if yaw is None:
                result_row[f'{prefix}_gaze_yaw'] = ''
                result_row[f'{prefix}_gaze_pitch'] = ''
                result_row[f'{prefix}_gaze_x'] = ''
                result_row[f'{prefix}_gaze_y'] = ''
                result_row[f'{prefix}_gaze_z'] = ''
                result_row[f'{prefix}_face_detected'] = 1
                continue

            gaze_vec = PureGazeInference.pitch_yaw_to_unit_vector(pitch, yaw)

            result_row[f'{prefix}_gaze_yaw'] = round(yaw, 6)
            result_row[f'{prefix}_gaze_pitch'] = round(pitch, 6)
            result_row[f'{prefix}_gaze_x'] = round(gaze_vec[0], 6)
            result_row[f'{prefix}_gaze_y'] = round(gaze_vec[1], 6)
            result_row[f'{prefix}_gaze_z'] = round(gaze_vec[2], 6)
            result_row[f'{prefix}_face_detected'] = 1

        results.append(result_row)

    # Cleanup
    for reader in cam_readers.values():
        reader.close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run PureGaze on matched frames from gaze cameras",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--session-dir', type=str, required=True)
    parser.add_argument('--matched-csv', type=str, default=None)
    parser.add_argument('--gaze-cams', type=int, nargs='+', default=[3, 4])
    parser.add_argument('--puregaze-weights', type=str, default=None,
                        help='Path to PureGaze .pt weights')
    parser.add_argument('--puregaze-model-dir', type=str, default=None,
                        help='Directory containing model/ package for PureGaze')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    session_dir = args.session_dir
    matched_csv = args.matched_csv or os.path.join(session_dir, "matched_frames.csv")
    output_path = args.output or os.path.join(session_dir, "gaze_results.csv")

    # Default PureGaze paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    gaze_est_dir = os.path.join(project_root, "USTA-Human-Interaction-Analysis", "gaze_estimation")

    weights_path = args.puregaze_weights or os.path.join(
        gaze_est_dir, "models", "Res50_PureGaze_ETH.pt")
    model_dir = args.puregaze_model_dir or gaze_est_dir

    if not os.path.exists(weights_path):
        print(f"Error: PureGaze weights not found: {weights_path}")
        sys.exit(1)

    # Device
    if args.device:
        device = args.device
    else:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Session:    {session_dir}")
    print(f"Matched:    {matched_csv}")
    print(f"Weights:    {weights_path}")
    print(f"Model dir:  {model_dir}")
    print(f"Device:     {device}")
    print(f"Gaze cams:  {args.gaze_cams}")
    print()

    # Load matched frames
    rows = []
    with open(matched_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} matched frame sets\n")

    # Initialize models
    gaze_model = PureGazeInference(weights_path, model_dir, device=device)
    face_detector = FaceDetector(min_confidence=0.5)

    # Process
    results = process_gaze_cameras(session_dir, rows, gaze_model, face_detector,
                                   args.gaze_cams)

    face_detector.close()

    # Write output
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nGaze results saved: {output_path}")
        print(f"  Rows: {len(results)}, Columns: {len(fieldnames)}")
    else:
        print("No results generated.")


if __name__ == '__main__':
    main()
