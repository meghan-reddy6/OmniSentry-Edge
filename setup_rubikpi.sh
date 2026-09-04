#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  OmniSentry-Edge: Rubik Pi 3 Auto-Setup (Qualcomm NPU)   "
echo "=========================================================="

sudo apt update || true
sudo apt install -y git git-lfs python3-pip python3-venv portaudio19-dev libasound2-dev python3-smbus qnn-tools snpe-tools wget i2c-tools || true
sudo usermod -aG i2c,video,audio $USER || true

git lfs install
git lfs fetch --all
git lfs checkout

if [ ! -d "venv" ]; then
    python3 -m venv --system-site-packages venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Download official QNN ONNX Runtime wheel if not registered
WHEEL_URL="https://cdn.edgeimpulse.com/qc-ai-docs/wheels/onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl"
WHEEL_NAME="onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl"

if ! python3 -c "import onnxruntime as ort; assert 'QNNExecutionProvider' in ort.get_available_providers()" 2>/dev/null; then
    echo "Installing official QNN ONNX Runtime wheel..."
    wget -q --show-progress -O "$WHEEL_NAME" "$WHEEL_URL"
    pip install "$WHEEL_NAME" --force-reinstall
    rm -f "$WHEEL_NAME"
fi

export LD_LIBRARY_PATH=/usr/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Pre-compile QNN context binary
if [ -f "scripts/compile_qnn_ctx.py" ] && [ -f "models/yolov8_det.onnx" ]; then
    echo "Pre-compiling QNN context binary for Hexagon NPU..."
    python3 scripts/compile_qnn_ctx.py || true
fi

echo ""
echo "=== Verification ==="
python3 -c "import onnxruntime as ort; eps = ort.get_available_providers(); print('Available Providers:', eps); assert 'QNNExecutionProvider' in eps, 'QNN EP registration failed!'"

echo ""
echo "Setup complete! Run:"
echo "  source venv/bin/activate"
echo "  python src/main.py"
