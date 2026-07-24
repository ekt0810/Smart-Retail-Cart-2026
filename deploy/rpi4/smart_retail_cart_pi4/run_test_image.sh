#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python3
fi
SOURCE="${1:-test_images}"
if [[ "$SOURCE" != /* ]]; then
  SOURCE="$PROJECT_DIR/$SOURCE"
fi

exec "$PYTHON_BIN" "$PROJECT_DIR/src/predict_image_pi4.py" \
  --model "$PROJECT_DIR/models/best_320_ncnn_model" \
  --source "$SOURCE" \
  --imgsz 320 \
  --conf 0.35
