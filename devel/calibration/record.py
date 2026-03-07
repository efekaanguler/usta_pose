import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import os
import threading

# ---------- Kamera seri numaraları ----------
serial0 = "313522300887"
serial1 = "318122303397"  # İkinci kameranın seri numarasını buraya yaz

# ---------- Kayıt klasörü ----------
save_dir = "recordings"
os.makedirs(save_dir, exist_ok=True)

# ---------- Zaman etiketi ----------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------- Kayıt fonksiyonu ----------
def record_camera(serial, filename):
    print(f"[INFO] Kamera {serial} başlatılıyor...")

    # Pipeline ve config oluştur
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Pipeline başlat
    pipeline.start(config)

    # Video kaydedici
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 30, (1280, 720))

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Görüntüyü göster
            window_name = f"Camera {serial}"
            cv2.imshow(window_name, color_image)

            # Dosyaya kaydet
            out.write(color_image)

            # Çıkış kontrolü
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"[INFO] Kamera {serial} kaydı durduruldu.")
                break
    finally:
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Kamera {serial} kaydedildi: {filename}")

# ---------- Dosya adları ----------
file0 = os.path.join(save_dir, f"color_{serial0}_{timestamp}.avi")
file1 = os.path.join(save_dir, f"color_{serial1}_{timestamp}.avi")

# ---------- Thread başlat (eşzamanlı kayıt için) ----------
t1 = threading.Thread(target=record_camera, args=(serial0, file0))
t2 = threading.Thread(target=record_camera, args=(serial1, file1))

t1.start()
t2.start()

t1.join()
t2.join()

print("[INFO] Tüm kameraların kayıtları tamamlandı.")
