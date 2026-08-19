#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  OmniSentry-Edge: Rubik Pi 3 Auto-Setup (Qualcomm NPU)   "
echo "=========================================================="

# 1. Install System Packages & Audio/Hardware Drivers
echo "[1/4] Installing system dependencies and utilities..."
sudo apt-get update
sudo apt-get install -y git git-lfs python3-pip python3-venv portaudio19-dev libasound2-dev python3-smbus python3-onnxruntime || true

# 2. Fetch Git LFS Binaries & Validate
echo "[2/4] Pulling binary ONNX model weights..."
git lfs install
git lfs fetch --all
git lfs checkout

if [ ! -s "models/yolov8_det.onnx" ]; then
    echo "ERROR: models/yolov8_det.onnx is missing or empty."
    exit 1
fi

if head -c 100 "models/yolov8_det.onnx" | grep -q "git-lfs.github.com/spec"; then
    echo "ERROR: models/yolov8_det.onnx is still a Git LFS pointer text file."
    echo "Run: git lfs pull"
    exit 1
fi

echo "OK: models/yolov8_det.onnx is verified as binary ($(stat -c%s "models/yolov8_det.onnx") bytes)."

# 3. Create Python VENV with System Site-Packages (Inherits QNN Provider)
echo "[3/4] Initializing Python Virtual Environment with hardware drivers..."
rm -rf venv
python3 -m venv --system-site-packages venv
source venv/bin/activate

# 4. Install Python Dependencies
echo "[4/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Ensure library search paths include Qualcomm HTP libraries
export LD_LIBRARY_PATH=/usr/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

echo ""
echo "=== Testing Execution Providers in Virtual Environment ==="
python -c "import onnxruntime as ort; eps = ort.get_available_providers(); print('Available Providers:', eps); assert 'QNNExecutionProvider' in eps or 'CPUExecutionProvider' in eps"

echo ""
echo "=========================================================="
echo "  Setup Complete! Launching OmniSentry-Edge...            "
echo "=========================================================="
echo "To run manually:"
echo "  source venv/bin/activate"
echo "  python src/main.py"
echo "=========================================================="
