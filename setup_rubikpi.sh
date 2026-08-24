#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  OmniSentry-Edge: Rubik Pi 3 Auto-Setup (Qualcomm NPU)   "
echo "=========================================================="

# 1. System packages & Qualcomm QNN Tools
sudo apt update
sudo apt install -y git git-lfs python3-pip python3-venv portaudio19-dev libasound2-dev python3-smbus qnn-tools snpe-tools || true

# 2. Pull Git LFS Weights
git lfs install
git lfs fetch --all
git lfs checkout

# 3. Create Virtual Environment with system site packages
if [ ! -d "venv" ]; then
    python3 -m venv --system-site-packages venv
fi
source venv/bin/activate

# 4. Install Dependencies & Thundercomm QNN Wheel
echo "[4/4] Installing Python dependencies and Qualcomm QNN wheel..."
pip install --upgrade pip
pip install -r requirements.txt

# Ensure CPU onnxruntime is removed
pip uninstall -y onnxruntime || true

# Download and install the official aarch64 Python 3.12 QNN wheel
WHEEL_URL="https://cdn.edgeimpulse.com/qc-ai-docs/wheels/onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl"
WHEEL_FILE="onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl"

if ! python3 -c "import onnxruntime as ort; assert 'QNNExecutionProvider' in ort.get_available_providers()" 2>/dev/null; then
    echo "Fetching official QNN ONNX Runtime wheel for Python 3.12..."
    wget -q --show-progress -O "$WHEEL_FILE" "$WHEEL_URL"
    pip install "$WHEEL_FILE" --force-reinstall
    rm -f "$WHEEL_FILE"
fi

export LD_LIBRARY_PATH=/usr/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

echo ""
echo "=== Provider Verification ==="
python3 -c "import onnxruntime as ort; eps = ort.get_available_providers(); print('Available Providers:', eps); assert 'QNNExecutionProvider' in eps, 'ERROR: QNNExecutionProvider failed to register!'"

echo ""
echo "=========================================================="
echo "  Setup Complete! Launch with:"
echo "    source venv/bin/activate"
echo "    python src/main.py"
echo "=========================================================="
