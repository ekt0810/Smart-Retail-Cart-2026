#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname -s)" != "Linux" ]] || ! grep -qi "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
  echo "This installer must be run on Raspberry Pi OS." >&2
  exit 1
fi

sudo apt update
sudo apt install -y --no-install-recommends \
  python3-venv python3-pip python3-picamera2 libgl1 libglib2.0-0

if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  python3 -m venv --system-site-packages "$PROJECT_DIR/.venv"
fi

"$PROJECT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$PROJECT_DIR/.venv/bin/python" -m pip install --prefer-binary -r "$PROJECT_DIR/requirements-rpi.txt"

echo
echo "Installation complete. Test the packaged model with:"
echo "  cd $PROJECT_DIR"
echo "  .venv/bin/python src/predict_image_pi4.py --source test_images"
echo "Then run the CSI camera with:"
echo "  bash run_camera_ncnn_320.sh"
