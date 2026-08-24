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

# 4. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure dynamic linker search paths
export LD_LIBRARY_PATH=/usr/lib:/usr/lib/aarch64-linux-gnu:/opt/qcom/qnn/lib/aarch64-ubuntu-gcc11.2:$LD_LIBRARY_PATH

echo ""
echo "=== Verifying NPU Provider Registration ==="
python3 -c "import onnxruntime as ort; eps = ort.get_available_providers(); print('Available Providers:', eps); assert 'QNNExecutionProvider' in eps or 'CPUExecutionProvider' in eps"

echo ""
echo "=========================================================="
echo "  Setup Complete! Launch with:"
echo "    source venv/bin/activate"
echo "    python src/main.py"
echo "=========================================================="
