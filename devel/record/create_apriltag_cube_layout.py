#!/usr/bin/env python3
"""
Create apriltag_cube_layout.json with a normal webcam.

The wizard scans one visible cube face at a time, reads the AprilTag ID, and
stores exact 3D tag-corner coordinates in the same corner order returned by the
AprilTag detector. The generated JSON is consumed by calibration_checker.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


FACE_ORDER = ["front", "right", "back", "left", "top", "bottom"]
MISSING_FACE_CHOICES = ["none", *FACE_ORDER]

FACE_DEFS = {
    "front": {
        "normal": (0.0, 0.0, 1.0),
        "up": (0.0, 1.0, 0.0),
        "tr": "ÖN yüz kameraya baksın; ÜST yüz ekranın üstüne baksın.",
    },
    "right": {
        "normal": (1.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "tr": "SAĞ yüz kameraya baksın; ÜST yüz ekranın üstüne baksın.",
    },
    "back": {
        "normal": (0.0, 0.0, -1.0),
        "up": (0.0, 1.0, 0.0),
        "tr": "ARKA yüz kameraya baksın; ÜST yüz ekranın üstüne baksın.",
    },
    "left": {
        "normal": (-1.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "tr": "SOL yüz kameraya baksın; ÜST yüz ekranın üstüne baksın.",
    },
    "top": {
        "normal": (0.0, 1.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "tr": "ÜST yüz kameraya baksın; ÖN yüz ekranın üstüne baksın.",
    },
    "bottom": {
        "normal": (0.0, -1.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "tr": "ALT yüz kameraya baksın; ÖN yüz ekranın üstüne baksın.",
    },
}

SCREEN_CORNER_NAMES = ["tl", "tr", "br", "bl"]


def load_detector(families: str):
    try:
        from pupil_apriltags import Detector

        return Detector(families=families)
    except ImportError:
        pass

    try:
        from dt_apriltags import Detector

        return Detector(families=families)
    except ImportError as exc:
        raise ImportError(
            "pupil-apriltags veya dt-apriltags kurulu değil. "
            "Örn: python3 -m pip install pupil-apriltags opencv-python"
        ) from exc


def prompt_float_mm(label: str, value: Optional[float]) -> float:
    if value is not None:
        return float(value)

    while True:
        raw = input(f"{label} (mm): ").strip().replace(",", ".")
        try:
            parsed = float(raw)
        except ValueError:
            print("Sayı gir reis, örn: 80")
            continue
        if parsed <= 0:
            print("Pozitif değer lazım.")
            continue
        return parsed


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        raise ValueError("Zero vector cannot be normalized.")
    return vec / norm


def face_screen_corners(face: str, cube_size_m: float, tag_size_m: float) -> Dict[str, np.ndarray]:
    """Return physical TL/TR/BR/BL corners for the prompted face orientation."""
    face_def = FACE_DEFS[face]
    normal = normalize(np.asarray(face_def["normal"], dtype=np.float64))
    screen_up = normalize(np.asarray(face_def["up"], dtype=np.float64))
    camera_forward = -normal
    screen_right = normalize(np.cross(camera_forward, screen_up))

    center = normal * (cube_size_m / 2.0)
    half_tag = tag_size_m / 2.0

    return {
        "tl": center - screen_right * half_tag + screen_up * half_tag,
        "tr": center + screen_right * half_tag + screen_up * half_tag,
        "br": center + screen_right * half_tag - screen_up * half_tag,
        "bl": center - screen_right * half_tag - screen_up * half_tag,
    }


def classify_image_corners(corners: np.ndarray) -> Dict[int, str]:
    """
    Map detector corner index -> screen corner label while the face is held
    fronto-parallel-ish according to the prompt.
    """
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]

    mapping = {
        int(np.argmin(sums)): "tl",
        int(np.argmax(diffs)): "tr",
        int(np.argmax(sums)): "br",
        int(np.argmin(diffs)): "bl",
    }
    if len(mapping) != 4 or set(mapping.values()) != set(SCREEN_CORNER_NAMES):
        raise RuntimeError("Corner sırası okunamadı; yüzü kameraya daha düz gösterip tekrar dene.")
    return mapping


def quad_area(corners: np.ndarray) -> float:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return float(abs(cv2.contourArea(pts)))


def largest_detection(detections):
    if not detections:
        return None
    return max(detections, key=lambda det: quad_area(det.corners))


def draw_detections(frame: np.ndarray, detections, active_face: str, instruction: str) -> np.ndarray:
    overlay = frame.copy()
    cv2.putText(
        overlay,
        f"{active_face.upper()} - {instruction}",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "s: kaydet | q: cik | yuzu mumkun oldugunca duz ve buyuk goster",
        (24, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for detection in detections:
        pts = np.asarray(detection.corners, dtype=np.int32).reshape(4, 2)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        center = tuple(np.asarray(detection.center, dtype=np.int32).reshape(2))
        cv2.putText(
            overlay,
            f"ID {int(detection.tag_id)}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        for idx, corner in enumerate(pts):
            cv2.circle(overlay, tuple(corner), 5, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                str(idx),
                tuple(corner + np.array([6, -6])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return overlay


def active_face_order(missing_face: str) -> List[str]:
    if missing_face == "none":
        return list(FACE_ORDER)
    if missing_face not in FACE_ORDER:
        raise ValueError(f"Unknown missing face: {missing_face}")
    return [face for face in FACE_ORDER if face != missing_face]


def scan_cube_faces(args, detector) -> Dict[str, object]:
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam açılamadı: camera index {args.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    scans = {}
    window_name = "AprilTag Cube Layout Wizard"

    try:
        faces_to_scan = active_face_order(args.missing_face)
        if args.missing_face != "none":
            print(
                f"\n5 yüzlü küp modu: {args.missing_face.upper()} yüz okutulmayacak. "
                "Eksik yüzü kayıt/pre-check sırasında tabana koy."
            )

        for face in faces_to_scan:
            instruction = FACE_DEFS[face]["tr"]
            print("\n" + "=" * 72)
            print(f"{face.upper()} yüzü okutuluyor")
            print(instruction)
            print("Tag tek/büyük görününce 's' bas. Çıkmak için 'q'.")

            saved = False
            last_detection = None
            while not saved:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Webcam frame alınamadı.")

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(gray)
                last_detection = largest_detection(detections)

                overlay = draw_detections(frame, detections, face, instruction)
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    raise KeyboardInterrupt
                if key != ord("s"):
                    continue

                if last_detection is None:
                    print("Tag göremedim; yüzü kameraya biraz daha düz/yakın göster.")
                    continue

                tag_id = int(last_detection.tag_id)
                if any(scan["id"] == tag_id for scan in scans.values()):
                    print(f"ID {tag_id} daha önce kaydedilmiş; aynı tag iki yüzde olamaz.")
                    continue

                try:
                    index_to_screen = classify_image_corners(last_detection.corners)
                except RuntimeError as exc:
                    print(str(exc))
                    continue

                scans[face] = {
                    "id": tag_id,
                    "detector_corner_to_screen_corner": {
                        str(idx): screen_corner for idx, screen_corner in index_to_screen.items()
                    },
                }
                print(f"Kaydedildi: {face} -> ID {tag_id}")
                saved = True
                time.sleep(0.5)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return scans


def build_layout(
    scans: Dict[str, object],
    cube_size_mm: float,
    tag_size_mm: float,
    family: str,
    missing_face: str,
    missing_tag_id: Optional[int],
) -> Dict[str, object]:
    cube_size_m = cube_size_mm / 1000.0
    tag_size_m = tag_size_mm / 1000.0

    tags: List[Dict[str, object]] = []
    faces: Dict[str, int] = {}

    for face in active_face_order(missing_face):
        scan = scans[face]
        screen_corners = face_screen_corners(face, cube_size_m, tag_size_m)
        detector_mapping = scan["detector_corner_to_screen_corner"]
        corners = [
            screen_corners[detector_mapping[str(idx)]].round(9).tolist()
            for idx in range(4)
        ]

        tag_id = int(scan["id"])
        faces[face] = tag_id
        face_normal = normalize(np.asarray(FACE_DEFS[face]["normal"], dtype=np.float64))
        face_center = face_normal * (cube_size_m / 2.0)
        tags.append({
            "id": tag_id,
            "family": family,
            "face": face,
            "normal": face_normal.round(9).tolist(),
            "center": face_center.round(9).tolist(),
            "corners": corners,
        })

    missing_info = None
    if missing_face != "none":
        missing_info = {"face": missing_face}
        if missing_tag_id is not None:
            missing_info["expected_tag_id"] = int(missing_tag_id)

    return {
        "unit": "m",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": (
            "Generated by create_apriltag_cube_layout.py. Cube frame: origin at cube "
            "center, +X=right, +Y=top, +Z=front. Tags are assumed centered on each face. "
            "If missing_face is set, keep that physical face on the table/bottom during checks."
        ),
        "cube_size_mm": cube_size_mm,
        "tag_size_mm": tag_size_mm,
        "missing_face": missing_info,
        "faces": faces,
        "tags": tags,
    }


def write_json(path: Path, data: Dict[str, object], force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        answer = input(f"{path} zaten var. Üzerine yazılsın mı? [y/N]: ").strip().lower()
        if answer not in ("y", "yes", "e", "evet"):
            raise RuntimeError("Yazma iptal edildi.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Use a webcam to create apriltag_cube_layout.json for felfelfeci3 pre-check."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Mac webcam index")
    parser.add_argument("--family", type=str, default="tag36h11", help="AprilTag family")
    parser.add_argument("--cube-size-mm", type=float, default=None, help="Cube edge length in millimeters")
    parser.add_argument(
        "--tag-size-mm",
        type=float,
        default=None,
        help="Detected AprilTag square side length in millimeters, not the paper size",
    )
    parser.add_argument("--width", type=int, default=1280, help="Webcam preview width")
    parser.add_argument("--height", type=int, default=720, help="Webcam preview height")
    parser.add_argument(
        "--missing-face",
        type=str,
        default="bottom",
        choices=MISSING_FACE_CHOICES,
        help=(
            "Face without a usable tag. Default bottom for the 5-face cube; "
            "use 'none' for a full 6-face cube."
        ),
    )
    parser.add_argument(
        "--missing-tag-id",
        type=int,
        default=1,
        help="Expected ID on the missing face, stored only as metadata",
    )
    parser.add_argument(
        "--six-face-cube",
        action="store_true",
        help="Shortcut for --missing-face none",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "apriltag_cube_layout.json",
        help="Output layout JSON path",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output without asking")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.six_face_cube:
        args.missing_face = "none"
    missing_tag_id = None if args.missing_face == "none" else args.missing_tag_id

    print("\nAprilTag Cube Layout Wizard")
    print("Varsayım: her tag kendi yüzünün TAM ORTASINDA.")
    print("Ölçü: tag-size, detector'ın çizdiği AprilTag karenin kenarı; kağıt kenarı değil.\n")
    if args.missing_face != "none":
        print(
            f"5 yüz modu aktif: {args.missing_face.upper()} yüz skip edilecek; "
            f"eksik tag ID metadata={missing_tag_id}."
        )

    cube_size_mm = prompt_float_mm("Küp kenarı", args.cube_size_mm)
    tag_size_mm = prompt_float_mm("AprilTag kare kenarı", args.tag_size_mm)

    detector = load_detector(args.family)
    scans = scan_cube_faces(args, detector)
    layout = build_layout(
        scans,
        cube_size_mm,
        tag_size_mm,
        args.family,
        args.missing_face,
        missing_tag_id,
    )
    write_json(args.output, layout, args.force)

    print("\n" + "=" * 72)
    print(f"OK: layout yazıldı -> {args.output}")
    print("Yüz -> ID eşleşmesi:")
    for face in FACE_ORDER:
        if face in layout["faces"]:
            print(f"  {face:>6}: {layout['faces'][face]}")
        else:
            print(f"  {face:>6}: MISSING/SKIPPED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nİptal edildi.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nHATA: {exc}", file=sys.stderr)
        raise SystemExit(1)
