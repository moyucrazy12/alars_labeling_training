#!/usr/bin/env bash
set -e

CKPT_PATH="labeling_pipeline/models/sam2/checkpoints/sam2.1_hiera_large.pt"
CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

if [ ! -f "$CKPT_PATH" ]; then
    echo "[INFO] SAM2 checkpoint not found."
    echo "[INFO] Downloading to: $CKPT_PATH"
    mkdir -p "$(dirname "$CKPT_PATH")"
    wget -O "$CKPT_PATH" "$CKPT_URL"
else
    echo "[INFO] SAM2 checkpoint found: $CKPT_PATH"
fi

exec "$@"
