#!/usr/bin/env bash
set -euo pipefail

python3 src/predict_camera_pi4.py \
  --model models/best_320_ncnn_model \
  --imgsz 320 \
  --camera 0 \
  --conf 0.35 \
  --width 640 \
  --height 480 \
  "$@"
