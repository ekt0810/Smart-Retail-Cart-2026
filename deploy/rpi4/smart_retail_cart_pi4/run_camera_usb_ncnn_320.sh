#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python3
fi

exec "$PYTHON_BIN" "$PROJECT_DIR/src/predict_camera_pi4.py" \
  --model "$PROJECT_DIR/models/best_320_ncnn_model" \
  --backend usb \
  --imgsz 320 \
  --camera 0 \
  --conf 0.35 \
  --width 640 \
  --height 480 \
  "$@"
