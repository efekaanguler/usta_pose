#!/bin/bash

# Script dizini ve repo adını al
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="$(basename "$SCRIPT_DIR")"

# Parent dizini al (bir üst dizin)
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# GPU kontrolü yap
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU bulundu, GPU desteğiyle container başlatılıyor..."
    GPU_FLAG="--gpus all"
else
    echo "GPU bulunamadı, CPU ile container başlatılıyor..."
    GPU_FLAG=""
fi

# X11 Erişimi (GUI için)
xhost +local:root &> /dev/null

# Container'ı çalıştır
# Parent dizin host_mount olarak mount ediliyor
# Working directory container içinde host_mount/$REPO_NAME olacak
docker run -it --rm \
    $GPU_FLAG \
    --privileged \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev:/dev \
    -v /dev/bus/usb:/dev/bus/usb \
    --group-add video \
    -v "$PARENT_DIR:/host_mount" \
    -w "/host_mount/$REPO_NAME" \
    usta_pose_models:latest \
    /bin/bash
