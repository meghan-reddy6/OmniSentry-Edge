#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  OmniSentry-Edge: Rubik Pi 3 Setup (Qualcomm Hexagon NPU)"
echo "=========================================================="

# 1. Install System QNN Backends & Utilities
echo "[1/4] Installing system drivers, QNN tools, and audio packages..."
sudo apt update
sudo apt install -y git git-lfs python3-pip python3-venv portaudio19-dev libasound2-dev python3-smbus qnn-tools snpe-tools || true

# 2. Pull Binary ONNX Weights via Git LFS
echo "[2/4] Pulling and verifying binary ONNX weights..."
git lfs install
git lfs fetch --all
git lfs checkout

if [ ! -s "models/yolov8_det.onnx" ]; then
    echo "ERROR: models/yolov8_det.onnx is missing or empty."
    exit 1
fi

if head -c 100 "models/yolov8_det.onnx" | grep -q "git-lfs.github.com/spec"; then
    echo "ERROR: models/yolov8_det.onnx is still an LFS pointer. Run: git lfs pull"
    exit 1
fi

echo "OK: models/yolov8_det.onnx is verified ($(stat -c%s "models/yolov8_det.onnx") bytes)."

# 3. Create Virtual Environment with System Hardware Access
echo "[3/4] Initializing Python Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv --system-site-packages venv
fi
source venv/bin/activate

# 4. Install Dependencies & Qualcomm QNN Wheel
echo "[4/4] Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Remove any standard CPU packages
pip uninstall -y onnxruntime || true

# Look for local wheel or download the pre-compiled aarch64 QNN wheel
if ls onnxruntime_qnn*cp312*linux_aarch64.whl 1> /dev/null 2>&1; then
    echo "Installing local QNN wheel..."
    pip install onnxruntime_qnn*cp312*linux_aarch64.whl --force-reinstall
else
    echo "Downloading Qualcomm QNN ONNX Runtime wheel for Python 3.12 ARM64..."
    wget -q --show-progress -O onnxruntime_qnn-1.20.0-cp312-cp312-linux_aarch64.whl \
      https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime_qnn-1.20.0-cp312-cp312-linux_aarch64.whl || true
    
    if [ -f "onnxruntime_qnn-1.20.0-cp312-cp312-linux_aarch64.whl" ]; then
        pip install onnxruntime_qnn-1.20.0-cp312-cp312-linux_aarch64.whl
    else
        echo "ERROR: Failed to download onnxruntime_qnn wheel. Please place the Thundercomm QNN wheel in this directory."
        exit 1
    fi
fi

# Configure dynamic linker search paths
ORT_CAPI_DIR=$(python3 -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))" 2>/dev/null || true)
export LD_LIBRARY_PATH=$ORT_CAPI_DIR:/usr/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

echo ""
echo "=== Verifying NPU Provider Registration ==="
python3 -c "import onnxruntime as ort; eps = ort.get_available_providers(); print('Available Providers:', eps); assert 'QNNExecutionProvider' in eps, 'QNNExecutionProvider failed to load!'"

echo ""
echo "=========================================================="
echo "  Setup Complete! Launch with:"
echo "    source venv/bin/activate"
echo "    python src/main.py"
echo "=========================================================="
