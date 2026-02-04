#!/bin/bash

# Script dizini ve repo adını al
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="$(basename "$SCRIPT_DIR")"

# GPU kontrolü yap
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU bulundu, GPU desteğiyle container başlatılıyor..."
    GPU_FLAG="--gpus all"
else
    echo "GPU bulunamadı, CPU ile container başlatılıyor..."
    GPU_FLAG=""
fi

# Container'ı çalıştır
docker run -it --rm \
    $GPU_FLAG \
    -v "$SCRIPT_DIR:/workspace/$REPO_NAME" \
    -w "/workspace/$REPO_NAME" \
    usta_pose_models:latest \
    /bin/bash
