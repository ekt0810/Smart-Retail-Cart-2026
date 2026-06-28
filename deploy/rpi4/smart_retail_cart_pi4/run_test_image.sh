#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-test_images}"

python3 src/predict_image_pi4.py \
  --model models/best_320_ncnn_model \
  --source "$SOURCE" \
  --imgsz 320 \
  --conf 0.35
