#!/usr/bin/env bash
set -euo pipefail

python src/export_model.py \
  --model runs/detect/train/weights/best.pt \
  "$@"
