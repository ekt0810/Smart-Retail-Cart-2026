#!/usr/bin/env bash
set -euo pipefail

python3 src/predict_camera.py \
  --model runs/detect/train/weights/best.onnx \
  --camera 0 \
  --conf 0.35 \
  --imgsz 416 \
  "$@"
