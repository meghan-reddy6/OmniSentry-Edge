#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  OmniSentry-Edge: Rubik Pi 3 Auto-Setup (Qualcomm NPU)   "
echo "=========================================================="

# 1. System Dependencies
echo "[1/4] Installing system packages..."
sudo apt-get update
sudo apt-get install -y git git-lfs python3-pip python3-venv portaudio19-dev libasound2-dev python3-smbus

# 2. Pull Git LFS Binary Weights
echo "[2/4] Fetching ONNX model weights..."
git lfs install
git lfs fetch --all
git lfs checkout

if [ ! -s "models/yolov8_det.onnx" ]; then
    echo "ERROR: models/yolov8_det.onnx is missing."
    exit 1
fi

if head -c 100 "models/yolov8_det.onnx" | grep -q "git-lfs.github.com/spec"; then
    echo "ERROR: models/yolov8_det.onnx is still a Git LFS pointer."
    echo "Run: git lfs pull"
    exit 1
fi

echo "OK: models/yolov8_det.onnx verified ($(stat -c%s "models/yolov8_det.onnx") bytes)."

# 3. Python Virtual Environment Setup
echo "[3/4] Initializing Python Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv --system-site-packages venv
fi
source venv/bin/activate

# 4. Install Dependencies & Resolve ONNX Runtime
echo "[4/4] Installing Python requirements and ONNX Runtime..."
pip install --upgrade pip
pip install -r requirements.txt

# Search for hardware-specific Qualcomm wheels first
LOCAL_WHEEL=$(find /opt /usr -name "*onnxruntime*.whl" 2>/dev/null | head -n 1)
if [ -n "$LOCAL_WHEEL" ]; then
    echo "Found local Qualcomm ONNX Runtime wheel: $LOCAL_WHEEL"
    pip install "$LOCAL_WHEEL" --force-reinstall
else
    echo "Attempting to install onnxruntime-qnn..."
    pip install onnxruntime-qnn --extra-index-url https://download.onnxruntime.ai/ || pip install onnxruntime
fi

export LD_LIBRARY_PATH=/usr/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

echo ""
echo "=== Provider Verification ==="
python -c "import onnxruntime as ort; print('Available EPs:', ort.get_available_providers())"

echo ""
echo "=========================================================="
echo "  Setup Complete! Run: source venv/bin/activate && python src/main.py"
echo "=========================================================="
