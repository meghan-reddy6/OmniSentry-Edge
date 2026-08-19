#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  OmniSentry-Edge: Rubik Pi 3 Auto-Setup (Qualcomm QNN)   "
echo "=========================================================="

# 1. System Dependencies & Git LFS
echo "[1/5] Installing system packages and Git LFS..."
sudo apt-get update
sudo apt-get install -y git git-lfs portaudio19-dev libasound2-dev python3-pip python3-venv

# 2. Pull Binary ONNX Weights
echo "[2/5] Fetching binary ONNX models via Git LFS..."
git lfs install
git lfs pull

# Verify model binary sizes
if [ ! -s "models/yolov8_det.onnx" ] || [ $(stat -c%s "models/yolov8_det.onnx") -lt 1000000 ]; then
    echo "ERROR: models/yolov8_det.onnx is invalid or missing. Ensure Git LFS pulled real binary weights."
    exit 1
fi

# 3. Python Virtual Environment Setup
echo "[3/5] Initializing Python Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 4. Install Dependencies & Qualcomm QNN Execution Provider Wheel
echo "[4/5] Installing dependencies and Qualcomm ONNX Runtime..."
pip install --upgrade pip
pip install -r requirements.txt

# Ensure QNN Execution Provider wheel is installed
pip uninstall -y onnxruntime || true
pip install onnxruntime-qnn --extra-index-url https://download.onnxruntime.ai/ || true

# Copy system QNN wheel if available as distro package
if [ -d "/usr/lib/python3/dist-packages/onnxruntime" ]; then
    echo "Linking system Qualcomm QNN ONNX Runtime distribution..."
    cp -r /usr/lib/python3/dist-packages/onnxruntime* venv/lib/python3.12/site-packages/ || true
fi

# 5. Environment Variables & Provider Verification
echo "[5/5] Configuring Library Paths and Testing Providers..."
export LD_LIBRARY_PATH=/usr/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

python3 -c "import onnxruntime as ort; eps = ort.get_available_providers(); print('\n>>> Available Providers:', eps); assert 'QNNExecutionProvider' in eps or 'CPUExecutionProvider' in eps"

echo ""
echo "=========================================================="
echo "  Setup Complete! Launching OmniSentry-Edge...            "
echo "=========================================================="
echo "To manually run later:"
echo "  source venv/bin/activate"
echo "  python src/main.py"
echo "=========================================================="
