import pyrealsense2 as rs
import cv2
import numpy as np
import apriltag
import yaml
import os
import json

# ---------- Ayarlar ----------
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FRAME_SKIP = 20       # 30 FPS -> efektif 3 FPS
MAX_FRAMES =  50     # Toplam işlenecek frame sayısı
SAVE_YAML = "stereo_camera_romer14.yaml"
TAG_JSON_PATH = "tag.json"  # AprilTag layout JSON dosyası

# ---------- JSON'dan AprilTag layout oku ----------
with open(TAG_JSON_PATH, "r") as f:
    tag_data = json.load(f)

# JSON içinden tag bilgilerini bir dictionary'ye al
APRIL_TAG_LAYOUT = {}
for tag in tag_data["tagCoordinates"][0]:
    APRIL_TAG_LAYOUT[tag["id"]] = {
        "center": tag["center"],
        "corners": tag["corners"],
        "unit": tag.get("unit", "mm")
    }

print(f"{len(APRIL_TAG_LAYOUT)} adet tag yüklendi JSON'dan.")

# ---------- RealSense pipeline başlat ----------
def start_pipeline(serial):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)
    pipe.start(cfg)
    return pipe

# ---------- AprilTag detector ----------
detector = apriltag.Detector()

# ---------- Tag'in dünyadaki 4 köşesini hesapla ----------
def get_tag_world_coords(tag_id):
    if tag_id not in APRIL_TAG_LAYOUT:
        raise ValueError(f"Tag ID {tag_id} JSON layout'ta yok!")

    corners = APRIL_TAG_LAYOUT[tag_id]["corners"]

    # JSON mm olarak kaydedilmiş, metreye çeviriyoruz
    corners_m = np.array([[x/1000.0, y/1000.0, 0] for [x, y] in corners], dtype=np.float32)
    return corners_m

# ---------- Tekli kamera kalibrasyonu ----------
def calibrate_camera(objpoints, imgpoints, img_size):
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None
    )
    # Reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        imgpoints2 = imgpoints2.reshape(-1, 2)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    return K, D, rvecs, tvecs, mean_error

# ---------- Kamera seri numaraları ----------
serial1 = "318122303397"  # sol kamera
serial0 = "313522300887"  # sağ kamera

pipe0 = start_pipeline(serial0)
pipe1 = start_pipeline(serial1)

# ---------- Frame toplama ----------
collected_obj_points = []
collected_imgpoints0 = []
collected_imgpoints1 = []

frame_counter = 0
processed_frames = 0

print("Frame toplama başlıyor... Bitirmek için 'q' tuşuna basın.")

try:
    while processed_frames < MAX_FRAMES:
        frames0 = pipe0.wait_for_frames()
        frames1 = pipe1.wait_for_frames()

        color0 = frames0.get_color_frame()
        color1 = frames1.get_color_frame()
        if not color0 or not color1:
            continue

        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue

        img0 = np.asanyarray(color0.get_data())
        img1 = np.asanyarray(color1.get_data())
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # AprilTag tespitleri
        results0 = detector.detect(gray0)
        results1 = detector.detect(gray1)

        if len(results0) > 0 and len(results1) > 0:
            # Ortak görülen tag ID'lerini bul
            ids0 = set(r.tag_id for r in results0)
            ids1 = set(r.tag_id for r in results1)
            common_ids = ids0.intersection(ids1)
            print("Camera 0 IDs:", list(ids0))
            print("Camera 1 IDs:", list(ids1))
            print("Common IDs:", sorted(list(common_ids)))
            print("--------------")

            if len(common_ids) > 0:
                frame_obj_points = []
                frame_img_points0 = []
                frame_img_points1 = []

                for tag_id in common_ids:
                    r0 = next(r for r in results0 if r.tag_id == tag_id)
                    r1 = next(r for r in results1 if r.tag_id == tag_id)

                    # Tag'in JSON'dan dünyadaki koordinatlarını al
                    world_corners = get_tag_world_coords(tag_id)

                    # Tüm köşeleri ekle
                    frame_obj_points.extend(world_corners)
                    frame_img_points0.extend(r0.corners.astype(np.float32))
                    frame_img_points1.extend(r1.corners.astype(np.float32))

                collected_obj_points.append(np.array(frame_obj_points, dtype=np.float32))
                collected_imgpoints0.append(np.array(frame_img_points0, dtype=np.float32))
                collected_imgpoints1.append(np.array(frame_img_points1, dtype=np.float32))

                processed_frames += 1
                print(f"[Frame {frame_counter}] {len(common_ids)} ortak tag işlendi.")

        # Görselleştirme
        for r in results0:
            for c in r.corners.astype(int):
                cv2.circle(img0, tuple(c), 5, (0,255,0), -1)
        for r in results1:
            for c in r.corners.astype(int):
                cv2.circle(img1, tuple(c), 5, (0,255,0), -1)

        combined = np.hstack((img0, img1))
        cv2.imshow("Stereo Cameras", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Kullanıcı durdurdu")
            break

except KeyboardInterrupt:
    print("Ctrl+C ile durduruldu")

finally:
    pipe0.stop()
    pipe1.stop()
    cv2.destroyAllWindows()

# ---------- Tekli kamera kalibrasyonları ----------
print("Kamera 0 kalibrasyonu...")
K0, D0, rvecs0, tvecs0, error0 = calibrate_camera(collected_obj_points, collected_imgpoints0, (IMAGE_WIDTH, IMAGE_HEIGHT))
print("Kamera 0 Mean Reprojection Error:", error0)

print("Kamera 1 kalibrasyonu...")
K1, D1, rvecs1, tvecs1, error1 = calibrate_camera(collected_obj_points, collected_imgpoints1, (IMAGE_WIDTH, IMAGE_HEIGHT))
print("Kamera 1 Mean Reprojection Error:", error1)

# ---------- Stereo kalibrasyon ----------
print("Stereo kalibrasyon (extrinsics) yapılıyor...")
flags = cv2.CALIB_USE_INTRINSIC_GUESS
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate(
    collected_obj_points,
    collected_imgpoints0,
    collected_imgpoints1,
    K0, D0,
    K1, D1,
    (IMAGE_WIDTH, IMAGE_HEIGHT),
    criteria=criteria,
    flags=flags
)
print("Stereo extrinsics tamamlandı.")
print("R:\n", R)
print("T:\n", T)

# ---------- YAML olarak kaydet ----------
data = {
    'camera0': {'K': K0.tolist(), 'D': D0.tolist(), 'reprojection_error': float(error0)},
    'camera1': {'K': K1.tolist(), 'D': D1.tolist(), 'reprojection_error': float(error1)},
    'stereo': {'R': R.tolist(), 'T': T.tolist(), 'E': E.tolist(), 'F': F.tolist()}
}

with open(SAVE_YAML, "w") as f:
    yaml.dump(data, f)

print(f"Kalibrasyon verisi kaydedildi: {os.path.abspath(SAVE_YAML)}")
