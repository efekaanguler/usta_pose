#!/usr/bin/env bash
# ======================================================================
# run_revised_pipeline.sh — Revised USTA Pose Pipeline
#
# Usage:
#   bash run_revised_pipeline.sh <SESSION_DIR>
# ======================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <SESSION_DIR>"
    exit 1
fi

SESSION_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================="
echo "  REVISED PROCESS PIPELINE (ML DATASET GENERATOR)"
echo "======================================================================="
echo "  Session:    $SESSION_DIR"
echo "  Script dir: $SCRIPT_DIR"
echo "======================================================================="
echo ""

# ── Step 1: Pose Processing (cam1, cam2) ──────────────────────────────
if [ -z "${SKIP_POSE:-}" ]; then
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  STEP 1 — Independent Pose Processing                ║"
    echo "╚══════════════════════════════════════════════════════╝"
    
    for cam_id in 1 2; do
        if [ -d "$SESSION_DIR/cam$cam_id" ]; then
            bbox_var="POSE_BBOX_CAM${cam_id}"
            bbox_args=()
            if [ -n "${!bbox_var:-}" ]; then
                read -r -a bbox_values <<< "${!bbox_var}"
                if [ "${#bbox_values[@]}" -ne 4 ]; then
                    echo "Error: ${bbox_var} must contain four values: x1 y1 x2 y2"
                    exit 1
                fi
                bbox_args=(--bbox "${bbox_values[@]}")
            fi
            python3 "$SCRIPT_DIR/extract_pose_independent.py" \
                --session-dir "$SESSION_DIR" \
                --cam-id $cam_id \
                "${bbox_args[@]}"
        else
            echo "Warning: cam$cam_id directory not found, skipping."
        fi
    done
else
    echo "⏭ Skipping Pose Processing..."
fi
echo ""

# ── Step 2: Gaze Processing (cam3, cam4) ──────────────────────────────
if [ -z "${SKIP_GAZE:-}" ]; then
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  STEP 2 — Independent Gaze Processing                ║"
    echo "╚══════════════════════════════════════════════════════╝"
    
    for cam_id in 3 4; do
        if [ -d "$SESSION_DIR/cam$cam_id" ]; then
            python3 "$SCRIPT_DIR/extract_gaze_independent.py" \
                --session-dir "$SESSION_DIR" \
                --cam-id $cam_id
        else
            echo "Warning: cam$cam_id directory not found, skipping."
        fi
    done
else
    echo "⏭ Skipping Gaze Processing..."
fi
echo ""

# ── Step 3: Resample & ML Tabular Transformation ───────────────────────
if [ -z "${SKIP_RESAMPLE:-}" ]; then
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  STEP 3 — Savitzky-Golay & Root Relative        ║"
    echo "╚══════════════════════════════════════════════════════╝"
    
    python3 "$SCRIPT_DIR/resample_and_transform.py" \
        --session-dir "$SESSION_DIR"
else
    echo "⏭ Skipping Resample and Transform..."
fi
echo ""

echo "======================================================================="
echo "  ✓ Pipeline complete!"
echo "  Final output: $SESSION_DIR/session_ml_dataset.parquet"
echo "======================================================================="
