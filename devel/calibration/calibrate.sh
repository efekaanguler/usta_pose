#!/usr/bin/env bash
# =============================================================================
#  calibrate.sh  —  4-Camera Full Calibration Pipeline
# =============================================================================
#
#  AŞAMALAR
#  --------
#  1. Intrinsic capture   : Her kamera için ayrı ayrı görüntü çekimi (4×)
#  2. Extrinsic capture   : 5 stereo oturumu ile kameralar arası görüntü çekimi
#       Oturum A : cam1 + cam3  (tripod sol  ↔ masa sol)
#       Oturum B : cam2 + cam4  (tripod sağ  ↔ masa sağ)
#       Oturum C : cam1 + cam2  (iki tripod kamerası)
#       Oturum D : cam2 + cam3
#       Oturum E : cam1 + cam4
#  3. Çok-kamera kalibrasyon : multicam_calibrate.py ile
#       → intrinsic (K, dist) + extrinsic (R,T) tüm kameralar için
#       → çıktı: $OUTPUT_FILE
#
#  KULLANIM
#  --------
#  chmod +x calibrate.sh
#  ./calibrate.sh                          # varsayılan parametrelerle
#  ./calibrate.sh --skip-intrinsic         # intrinsic çekim zaten varsa atla
#  ./calibrate.sh --skip-capture           # tüm çekim zaten varsa sadece kalibre et
#  ./calibrate.sh --output /path/to/out.npz
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# BOARD PARAMETRELERİ  (her yerde aynı board kullanılmalı)
# ---------------------------------------------------------------------------
SQUARES_X=4
SQUARES_Y=3
SQUARE_LENGTH=0.063      # metre
MARKER_LENGTH=0.047      # metre
ARUCO_DICT="4X4_50"

# ---------------------------------------------------------------------------
# KALİBRASYON ÇALIŞMA DİZİNİ VE ÇIKTI DOSYASI
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB_DIR="${SCRIPT_DIR}/calib_data"      # tüm ara veriler buraya
OUTPUT_FILE="${CALIB_DIR}/multicam_calibration.npz"

# Kamera config dosyası (record_session ile aynı)
CAM_CONFIG="${SCRIPT_DIR}/../camera_config.json"

# Çekim başına hedef görüntü sayısı
NUM_CAPTURES=20
CAPTURE_INTERVAL=4.0     # saniye (auto-capture arası)

# Referans kamera (extrinsic grafinin merkezlendiği kamera)
REF_CAMERA=1

# ---------------------------------------------------------------------------
# ARGÜMAN AYRIŞTIRICISI
# ---------------------------------------------------------------------------
SKIP_INTRINSIC=false
SKIP_CAPTURE=false

for arg in "$@"; do
    case $arg in
        --skip-intrinsic)  SKIP_INTRINSIC=true ;;
        --skip-capture)    SKIP_CAPTURE=true; SKIP_INTRINSIC=true ;;
        --output=*)        OUTPUT_FILE="${arg#*=}" ;;
        --output)          shift; OUTPUT_FILE="$1" ;;
        --calib-dir=*)     CALIB_DIR="${arg#*=}" ;;
        --ref-camera=*)    REF_CAMERA="${arg#*=}" ;;
        --num-captures=*)  NUM_CAPTURES="${arg#*=}" ;;
        -h|--help)
            grep '^#' "$0" | head -40 | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# ORTAK BOARD ARGÜMANLARI (string olarak)
# ---------------------------------------------------------------------------
BOARD_ARGS="--squares-x ${SQUARES_X} --squares-y ${SQUARES_Y} \
--square-length ${SQUARE_LENGTH} --marker-length ${MARKER_LENGTH} \
--aruco-dict ${ARUCO_DICT}"

# ---------------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------------------------
log()  { echo -e "\n\033[1;36m[calibrate.sh]\033[0m $*"; }
ok()   { echo -e "\033[1;32m  ✓ $*\033[0m"; }
warn() { echo -e "\033[1;33m  ⚠ $*\033[0m"; }
die()  { echo -e "\033[1;31m  ✗ $*\033[0m"; exit 1; }

# Kullanıcıdan onay bekle
confirm() {
    local msg="${1:-Devam etmek için Enter\'a basın...}"
    read -rp "$(echo -e "\033[0;33m  → ${msg}\033[0m")"
}

python_cmd() {
    # Sanal ortam etkinleştirildiyse python3, değilse python3
    python3 "$@"
}

# ---------------------------------------------------------------------------
mkdir -p "${CALIB_DIR}"

LOG_FILE="${CALIB_DIR}/calibration_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
log "Tüm çıktılar ${LOG_FILE} konumuna kaydediliyor..."

# Intrinsic ve extrinsic çekim dizinleri
INTR_DIR="${CALIB_DIR}/intrinsic"
EXTR_DIR="${CALIB_DIR}/extrinsic"

INTR_CAM1="${INTR_DIR}/cam1"
INTR_CAM2="${INTR_DIR}/cam2"
INTR_CAM3="${INTR_DIR}/cam3"
INTR_CAM4="${INTR_DIR}/cam4"

SESSION_A="${EXTR_DIR}/session_cam1_cam3"   # tripod sol  ↔ masa sol
SESSION_B="${EXTR_DIR}/session_cam2_cam4"   # tripod sağ  ↔ masa sağ
SESSION_C="${EXTR_DIR}/session_cam1_cam2"   # iki tripod
SESSION_D="${EXTR_DIR}/session_cam2_cam3"   # kamera 2 + 3
SESSION_E="${EXTR_DIR}/session_cam1_cam4"   # kamera 1 + 4

# =============================================================================
#  AŞAMA 1 — INTRINSIC ÇEKIM (4 kamera ayrı ayrı)
# =============================================================================
if [ "${SKIP_INTRINSIC}" = false ]; then
    log "AŞAMA 1 — Intrinsic görüntü çekimi başlıyor (4 kamera)"
    echo "  Her kamera için ChArUco board'u tüm FOV'u kaplayacak şekilde hareket ettirin."
    echo "  Hedef: ${NUM_CAPTURES} görüntü / kamera  |  Auto-capture: ${CAPTURE_INTERVAL}s arayla"
    echo ""

    for CAM_ID in 1 2 3 4; do
        INTR_OUT_VAR="INTR_CAM${CAM_ID}"
        INTR_OUT="${!INTR_OUT_VAR}"
        mkdir -p "${INTR_OUT}"

        log "Kamera ${CAM_ID} intrinsic çekimi"
        confirm "Hazır olduğunuzda Enter'a basın (pencere açılacak)..."
        python_cmd "${SCRIPT_DIR}/charuco_intrinsic_capture.py" \
            --cam-config  "${CAM_CONFIG}" \
            --camera-id   "${CAM_ID}" \
            --output-dir  "${INTR_OUT}" \
            --num-captures "${NUM_CAPTURES}" \
            --auto-capture \
            --capture-interval "${CAPTURE_INTERVAL}" \
            ${BOARD_ARGS}
        ok "Kamera ${CAM_ID} intrinsic çekimi tamamlandı → ${INTR_OUT}"
    done
else
    warn "Intrinsic çekim atlandı (--skip-intrinsic)."
fi

# =============================================================================
#  AŞAMA 2 — EXTRINSIC STEREO ÇEKIM (5 oturum)
# =============================================================================
if [ "${SKIP_CAPTURE}" = false ]; then
    log "AŞAMA 2 — Extrinsic stereo çekim (5 oturum)"
    echo "  Board'u iki kameranın ortak görüş alanında tutun."
    echo ""

    declare -A SESSION_CAMERAS=(
        ["${SESSION_A}"]="1,3"
        ["${SESSION_B}"]="2,4"
        ["${SESSION_C}"]="1,2"
        ["${SESSION_D}"]="2,3"
        ["${SESSION_E}"]="1,4"
    )

    for SESSION_DIR in "${SESSION_A}" "${SESSION_B}" "${SESSION_C}" "${SESSION_D}" "${SESSION_E}"; do
        CAM_IDS="${SESSION_CAMERAS[${SESSION_DIR}]}"
        mkdir -p "${SESSION_DIR}"

        log "Oturum: kamera ${CAM_IDS}  →  ${SESSION_DIR}"
        confirm "Hazır olduğunuzda Enter'a basın (pencere açılacak)..."
        python_cmd "${SCRIPT_DIR}/multicam_capture.py" \
            --cam-config  "${CAM_CONFIG}" \
            --output-dir  "${SESSION_DIR}" \
            --camera-ids  "${CAM_IDS}" \
            --num-captures "${NUM_CAPTURES}" \
            --auto-capture \
            --capture-interval "${CAPTURE_INTERVAL}" \
            ${BOARD_ARGS}
        ok "Oturum tamamlandı → ${SESSION_DIR}"
    done
else
    warn "Extrinsic çekim atlandı (--skip-capture)."
fi

# =============================================================================
#  AŞAMA 3 — ÇOK-KAMERA KALİBRASYON
# =============================================================================
log "AŞAMA 3 — Çok-kamera kalibrasyon hesaplanıyor..."

# Çıktı dizinini oluştur (farklı bir konum belirtilmişse)
mkdir -p "$(dirname "${OUTPUT_FILE}")"

python_cmd "${SCRIPT_DIR}/multicam_calibrate.py" \
    --intrinsic-dir-1 "${INTR_CAM1}" \
    --intrinsic-dir-2 "${INTR_CAM2}" \
    --intrinsic-dir-3 "${INTR_CAM3}" \
    --intrinsic-dir-4 "${INTR_CAM4}" \
    --session-dirs "${SESSION_A}" "${SESSION_B}" "${SESSION_C}" "${SESSION_D}" "${SESSION_E}" \
    --output        "${OUTPUT_FILE}" \
    --num-cameras   4 \
    --ref-camera    "${REF_CAMERA}" \
    ${BOARD_ARGS}

# =============================================================================
#  ÖZET
# =============================================================================
if [ -f "${OUTPUT_FILE}" ]; then
    ok "Kalibrasyon tamamlandı!"
    echo ""
    echo "  Çıktı dosyası : ${OUTPUT_FILE}"
    echo "  Ara veriler   : ${CALIB_DIR}"
    echo ""
    echo "  NPZ içeriği   :"
    echo "    K1..K4         — 3×3 kamera matrisleri"
    echo "    dist1..dist4   — distorsiyon katsayıları"
    echo "    R_1_to_ref .. R_4_to_ref  — rotasyon (dünya→kamera)"
    echo "    t_1_to_ref .. t_4_to_ref  — öteleme (metre)"
    echo "    ref_camera, num_cameras"
else
    die "Çıktı dosyası oluşturulamadı: ${OUTPUT_FILE}"
fi
