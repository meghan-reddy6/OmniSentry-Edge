#!/usr/bin/env bash
set -e

echo "=== Installing Qualcomm QNN Runtime & Dependencies on Rubik Pi 3 ==="
sudo apt-get update
sudo apt-get install -y git git-lfs qcom-qnn qirp-sdk libasound2-dev portaudio19-dev

echo "=== Pulling Large Binary Model Files via Git LFS ==="
git lfs install
git lfs pull

echo "=== Setting up Python Virtual Environment ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Link or copy system Qualcomm QNN ONNX Runtime if available
if [ -d "/usr/lib/python3/dist-packages/onnxruntime" ]; then
    echo "=== Linking Thundercomm QNN-enabled ONNX Runtime ==="
    cp -r /usr/lib/python3/dist-packages/onnxruntime* venv/lib/python3.12/site-packages/ || true
fi

echo "=== Verifying NPU Provider Availability ==="
python3 -c "import onnxruntime as ort; print('Available EPs:', ort.get_available_providers())"

echo "=== Setup complete! Run: source venv/bin/activate && python src/main.py ==="
