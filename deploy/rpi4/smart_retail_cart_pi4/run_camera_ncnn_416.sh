#!/usr/bin/env bash
set -euo pipefail

python3 src/predict_camera_pi4.py \
  --model models/best_416_ncnn_model \
  --backend csi \
  --imgsz 416 \
  --camera 0 \
  --conf 0.35 \
  --width 640 \
  --height 480 \
  "$@"
