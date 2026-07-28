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
#  Daily default flow:
#      1. record_extrinsic.py captures the known cube in legacy camera pairs
#      2. calculate.py jointly optimizes cameras and cube poses
#
#  The legacy pairwise ChArUco flow remains available with --method charuco.
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
IDEAL_CUBE_LAYOUT="${RECORD_DIR}/apriltag_cube_layout.json"
CALIBRATED_CUBE_LAYOUT="${RECORD_DIR}/apriltag_cube_layout_calibrated.json"
if [[ -f "${CALIBRATED_CUBE_LAYOUT}" ]]; then
    CUBE_LAYOUT="${CALIBRATED_CUBE_LAYOUT}"
else
    CUBE_LAYOUT="${IDEAL_CUBE_LAYOUT}"
fi

SQUARES_X=4
SQUARES_Y=3
SQUARE_LENGTH=0.063
MARKER_LENGTH=0.047
ARUCO_DICT="4X4_50"

METHOD="cube"
CUBE_CAPTURE_MODE="pairwise"
CUBE_SOLVER="pairwise"
NUM_CAPTURES=30
CAPTURE_INTERVAL=1.0
REF_CAMERA=1
MIN_PAIRS=5
MIN_CAMERAS=2
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
  --method cube|charuco      Daily extrinsic method
  --skip-capture             Reuse the current extrinsic capture and calculate
  --manual                   Use manual SPACE capture instead of auto-capture
  --cube-capture-mode MODE   pairwise (default) or all
  --cube-solver SOLVER       pairwise (default) or joint-ba
  --pairs 1,2 1,3 ...        Override cube/ChArUco camera pairs
  --num-captures N           Accepted captures per pair/session
  --min-cameras N            Cube-visible cameras in all-camera mode
  --capture-interval SEC     Automatic pre-capture countdown duration
  --ref-camera N             Reference camera, 1-indexed
  --min-pairs N              Minimum shared captures for stereo calibration
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            shift
            METHOD="$1"
            shift
            ;;
        --method=*)
            METHOD="${1#*=}"
            shift
            ;;
        --skip-capture)
            SKIP_CAPTURE=true
            shift
            ;;
        --manual)
            MANUAL=true
            shift
            ;;
        --cube-capture-mode)
            shift
            CUBE_CAPTURE_MODE="$1"
            shift
            ;;
        --cube-capture-mode=*)
            CUBE_CAPTURE_MODE="${1#*=}"
            shift
            ;;
        --cube-solver)
            shift
            CUBE_SOLVER="$1"
            shift
            ;;
        --cube-solver=*)
            CUBE_SOLVER="${1#*=}"
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
        --min-cameras)
            shift
            MIN_CAMERAS="$1"
            shift
            ;;
        --min-cameras=*)
            MIN_CAMERAS="${1#*=}"
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

if [[ "${METHOD}" != "cube" && "${METHOD}" != "charuco" ]]; then
    die "--method must be cube or charuco"
fi
if [[ "${CUBE_CAPTURE_MODE}" != "pairwise" && "${CUBE_CAPTURE_MODE}" != "all" ]]; then
    die "--cube-capture-mode must be pairwise or all"
fi
if [[ "${CUBE_SOLVER}" != "pairwise" && "${CUBE_SOLVER}" != "joint-ba" ]]; then
    die "--cube-solver must be pairwise or joint-ba"
fi

log "Recordings dir: ${RECORDINGS_DIR}"

if [[ ! -f "${MASTER_INTRINSICS}" ]]; then
    die "Missing ${MASTER_INTRINSICS}. Run: python3 ${SCRIPT_DIR}/record_intrinsic.py"
fi

if [[ ! -f "${CAM_CONFIG}" ]]; then
    die "Missing camera config: ${CAM_CONFIG}"
fi

if [[ "${METHOD}" == "cube" ]]; then
    [[ -f "${CUBE_LAYOUT}" ]] || die "Missing cube layout: ${CUBE_LAYOUT}"
    python3 -c "import scipy" >/dev/null 2>&1 || \
        die "Cube calibration requires scipy in the active python3 environment"
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
        --method "${METHOD}"
        --cam-config "${CAM_CONFIG}"
        --output-dir "${EXTRINSIC_DIR}"
        --num-captures "${NUM_CAPTURES}"
        --capture-interval "${CAPTURE_INTERVAL}"
    )
    if [[ "${METHOD}" == "cube" ]]; then
        RECORD_ARGS+=(
            --cube-layout "${CUBE_LAYOUT}"
            --cube-capture-mode "${CUBE_CAPTURE_MODE}"
            --num-cameras 4
            --min-cameras "${MIN_CAMERAS}"
        )
    else
        RECORD_ARGS+=("${BOARD_ARGS[@]}")
    fi
    if [[ "${MANUAL}" == true ]]; then
        RECORD_ARGS+=(--manual)
    fi
    if [[ ${#PAIRS[@]} -gt 0 ]]; then
        RECORD_ARGS+=(--pairs "${PAIRS[@]}")
    fi

    python_cmd "${RECORD_ARGS[@]}"
    ok "Extrinsic capture completed"
else
    warn "Extrinsic capture skipped; using the current recorded extrinsic data"
fi

log "Step 2/2: calculating multicam calibration with fixed intrinsics"
python_cmd "${SCRIPT_DIR}/calculate.py" \
    --method "${METHOD}" \
    --master-intrinsics "${MASTER_INTRINSICS}" \
    --extrinsic-dir "${EXTRINSIC_DIR}" \
    --output "${OUTPUT_FILE}" \
    --num-cameras 4 \
    --ref-camera "${REF_CAMERA}" \
    --min-pairs "${MIN_PAIRS}" \
    --cube-layout "${CUBE_LAYOUT}" \
    --cube-solver "${CUBE_SOLVER}" \
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
    echo "    T_ref_to_cam1..T_ref_to_cam4"
    echo "    T_cam1_to_ref..T_cam4_to_ref"
else
    die "Output file was not created: ${OUTPUT_FILE}"
fi
