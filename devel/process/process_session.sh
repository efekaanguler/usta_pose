#!/usr/bin/env bash
# ======================================================================
# process_session.sh — Orchestrate the full processing pipeline
#
# Usage:
#   bash process_session.sh <SESSION_DIR> [CALIB_NPZ]
#
# Steps:
#   1. match_frames.py   → matched_frames.csv
#   2. run_pose.py       → pose_results.csv   (cam1, cam2)
#   3. run_gaze.py       → gaze_results.csv   (cam3, cam4)
#   4. assemble_csv.py   → session_output.csv  (unified)
# ======================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <SESSION_DIR> [CALIB_NPZ]"
    exit 1
fi

SESSION_DIR="$1"
CALIB_NPZ="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================="
echo "  PROCESS PIPELINE"
echo "======================================================================="
echo "  Session:    $SESSION_DIR"
echo "  Script dir: $SCRIPT_DIR"
if [ -n "$CALIB_NPZ" ]; then
    echo "  Calib NPZ:  $CALIB_NPZ"
fi
echo "======================================================================="
echo ""

# ── Step 1: Frame Matching ──────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║  STEP 1/4 — Frame Matching                         ║"
echo "╚══════════════════════════════════════════════════════╝"
START=$(date +%s)

python3 "$SCRIPT_DIR/match_frames.py" \
    --session-dir "$SESSION_DIR"

END=$(date +%s)
echo "  ⏱ Step 1 completed in $((END - START))s"
echo ""

# ── Step 2: Pose Processing ─────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║  STEP 2/4 — Pose Processing (cam1, cam2)           ║"
echo "╚══════════════════════════════════════════════════════╝"
START=$(date +%s)

POSE_ARGS=(
    --session-dir "$SESSION_DIR"
    --matched-csv "$SESSION_DIR/matched_frames.csv"
)
if [ -n "$CALIB_NPZ" ]; then
    POSE_ARGS+=(--calib-npz "$CALIB_NPZ")
fi

python3 "$SCRIPT_DIR/run_pose.py" "${POSE_ARGS[@]}"

END=$(date +%s)
echo "  ⏱ Step 2 completed in $((END - START))s"
echo ""

# ── Step 3: Gaze Processing ─────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║  STEP 3/4 — Gaze Processing (cam3, cam4)           ║"
echo "╚══════════════════════════════════════════════════════╝"
START=$(date +%s)

python3 "$SCRIPT_DIR/run_gaze.py" \
    --session-dir "$SESSION_DIR" \
    --matched-csv "$SESSION_DIR/matched_frames.csv"

END=$(date +%s)
echo "  ⏱ Step 3 completed in $((END - START))s"
echo ""

# ── Step 4: CSV Assembly ────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║  STEP 4/4 — CSV Assembly                           ║"
echo "╚══════════════════════════════════════════════════════╝"
START=$(date +%s)

python3 "$SCRIPT_DIR/assemble_csv.py" \
    --session-dir "$SESSION_DIR"

END=$(date +%s)
echo "  ⏱ Step 4 completed in $((END - START))s"
echo ""

echo "======================================================================="
echo "  ✓ Pipeline complete!"
echo "  Final output: $SESSION_DIR/session_output.csv"
echo "======================================================================="
