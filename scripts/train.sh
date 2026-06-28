#!/usr/bin/env bash
set -euo pipefail

python src/train.py \
  --data data/smart_retail_cart.yaml \
  --model yolo11n.pt \
  --imgsz 416 \
  --batch 8 \
  --workers 2 \
  "$@"
