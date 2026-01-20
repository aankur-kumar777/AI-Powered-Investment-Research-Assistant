#!/usr/bin/env bash
set -euo pipefail

# quick dev setup script for Ubuntu-like systems
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# hint: install tesseract on debian/ubuntu
if command -v apt-get >/dev/null 2>&1; then
  echo "installing system OCR deps (poppler-utils, tesseract)"
  sudo apt-get update && sudo apt-get install -y poppler-utils tesseract-ocr
fi

echo "Setup complete. Activate venv with: source .venv/bin/activate"