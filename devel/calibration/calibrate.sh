#!/usr/bin/env bash
# =============================================================================
#  calibrate.sh - Daily 4-Camera Extrinsic Calibration Flow
# =============================================================================
#
#  This script does NOT calculate intrinsics. Run record_intrinsic.py once to
#  create:
#
#      ../record/recordings/calib_data/master_intrinsics.npz
#
#  Daily flow:
#      1. record_extrinsic.py captures synchronized stereo ChArUco pairs
#      2. calculate.py combines those captures with fixed master intrinsics
#
#  Final output is always overwritten at:
#      ../record/recordings/multicam_calibration.npz
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORD_DIR="${SCRIPT_DIR}/../record"
RECORDINGS_DIR="${RECORD_DIR}/recordings"
CALIB_DIR="${RECORDINGS_DIR}/calib_data"
EXTRINSIC_DIR="${CALIB_DIR}/extrinsic"
MASTER_INTRINSICS="${CALIB_DIR}/master_intrinsics.npz"
OUTPUT_FILE="${RECORDINGS_DIR}/multicam_calibration.npz"
CAM_CONFIG="${RECORD_DIR}/camera_config.json"

SQUARES_X=4
SQUARES_Y=3
SQUARE_LENGTH=0.063
MARKER_LENGTH=0.047
ARUCO_DICT="4X4_50"

NUM_CAPTURES=20
CAPTURE_INTERVAL=4.0
REF_CAMERA=1
MIN_PAIRS=5
MANUAL=false
SKIP_CAPTURE=false
PAIRS=()

log()  { echo -e "\n\033[1;36m[calibrate.sh]\033[0m $*"; }
ok()   { echo -e "\033[1;32m  ✓ $*\033[0m"; }
warn() { echo -e "\033[1;33m  ⚠ $*\033[0m"; }
die()  { echo -e "\033[1;31m  ✗ $*\033[0m"; exit 1; }

usage() {
    sed -n '1,35p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --skip-capture             Use latest extrinsic run and only calculate
  --manual                   Use manual SPACE capture instead of auto-capture
  --pairs 1,2 1,3 ...        Override daily camera-pair sessions
  --num-captures N           Captures per camera pair
  --capture-interval SEC     Auto-capture interval hint
  --ref-camera N             Reference camera, 1-indexed
  --min-pairs N              Minimum shared captures for stereo calibration
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-capture)
            SKIP_CAPTURE=true
            shift
            ;;
        --manual)
            MANUAL=true
            shift
            ;;
        --pairs)
            shift
            PAIRS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                PAIRS+=("$1")
                shift
            done
            ;;
        --num-captures)
            shift
            NUM_CAPTURES="$1"
            shift
            ;;
        --num-captures=*)
            NUM_CAPTURES="${1#*=}"
            shift
            ;;
        --capture-interval)
            shift
            CAPTURE_INTERVAL="$1"
            shift
            ;;
        --capture-interval=*)
            CAPTURE_INTERVAL="${1#*=}"
            shift
            ;;
        --ref-camera)
            shift
            REF_CAMERA="$1"
            shift
            ;;
        --ref-camera=*)
            REF_CAMERA="${1#*=}"
            shift
            ;;
        --min-pairs)
            shift
            MIN_PAIRS="$1"
            shift
            ;;
        --min-pairs=*)
            MIN_PAIRS="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

python_cmd() {
    python3 "$@"
}

mkdir -p "${CALIB_DIR}" "${EXTRINSIC_DIR}" "${RECORDINGS_DIR}"

LOG_FILE="${CALIB_DIR}/calibration_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log "Daily calibration log: ${LOG_FILE}"
log "Recordings dir: ${RECORDINGS_DIR}"

if [[ ! -f "${MASTER_INTRINSICS}" ]]; then
    die "Missing ${MASTER_INTRINSICS}. Run: python3 ${SCRIPT_DIR}/record_intrinsic.py"
fi

if [[ ! -f "${CAM_CONFIG}" ]]; then
    die "Missing camera config: ${CAM_CONFIG}"
fi

BOARD_ARGS=(
    --squares-x "${SQUARES_X}"
    --squares-y "${SQUARES_Y}"
    --square-length "${SQUARE_LENGTH}"
    --marker-length "${MARKER_LENGTH}"
    --aruco-dict "${ARUCO_DICT}"
)

if [[ "${SKIP_CAPTURE}" == false ]]; then
    log "Step 1/2: capturing daily extrinsics only"
    RECORD_ARGS=(
        "${SCRIPT_DIR}/record_extrinsic.py"
        --cam-config "${CAM_CONFIG}"
        --output-dir "${EXTRINSIC_DIR}"
        --num-captures "${NUM_CAPTURES}"
        --capture-interval "${CAPTURE_INTERVAL}"
        "${BOARD_ARGS[@]}"
    )
    if [[ "${MANUAL}" == true ]]; then
        RECORD_ARGS+=(--manual)
    fi
    if [[ ${#PAIRS[@]} -gt 0 ]]; then
        RECORD_ARGS+=(--pairs "${PAIRS[@]}")
    fi

    python_cmd "${RECORD_ARGS[@]}"
    ok "Extrinsic capture completed"
else
    warn "Extrinsic capture skipped; using the latest recorded extrinsic run"
fi

log "Step 2/2: calculating multicam calibration with fixed intrinsics"
python_cmd "${SCRIPT_DIR}/calculate.py" \
    --master-intrinsics "${MASTER_INTRINSICS}" \
    --extrinsic-dir "${EXTRINSIC_DIR}" \
    --output "${OUTPUT_FILE}" \
    --num-cameras 4 \
    --ref-camera "${REF_CAMERA}" \
    --min-pairs "${MIN_PAIRS}" \
    "${BOARD_ARGS[@]}"

if [[ -f "${OUTPUT_FILE}" ]]; then
    ok "Calibration completed and overwritten: ${OUTPUT_FILE}"
    echo ""
    echo "  Fixed intrinsics : ${MASTER_INTRINSICS}"
    echo "  Calib data       : ${CALIB_DIR}"
    echo "  Final NPZ        : ${OUTPUT_FILE}"
    echo ""
    echo "  NPZ keys include:"
    echo "    K1..K4, dist1..dist4"
    echo "    R_1_to_ref..R_4_to_ref"
    echo "    t_1_to_ref..t_4_to_ref"
else
    die "Output file was not created: ${OUTPUT_FILE}"
fi
