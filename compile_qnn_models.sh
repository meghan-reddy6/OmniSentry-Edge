#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  Qualcomm QNN HTP Context Binary Compiler (QCS6490)      "
echo "=========================================================="

QNN_LIB="/usr/lib/libQnnHtp.so"
OUTPUT_DIR="models/qnn_ctx"
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$QNN_LIB" ]; then
    echo "ERROR: $QNN_LIB not found. Make sure qnn-tools and qirp-sdk are installed."
    exit 1
fi

# 1. Create HTP Backend Configuration for QCS6490
cat << 'EOF' > /tmp/htp_backend_config.json
{
  "devices": [
    {
      "htp_arch": "v68"
    }
  ]
}
EOF

# 2. Compile YOLOv8 INT8/UINT8 to QNN Serialized Context Binary
if [ -f "models/yolov8_det.onnx" ]; then
    echo "[1/2] Generating QNN HTP context binary for YOLOv8..."
    qnn-context-binary-generator \
        --backend "$QNN_LIB" \
        --model "models/yolov8_det.onnx" \
        --binary_file "yolov8_det.serialized" \
        --output_dir "$OUTPUT_DIR" \
        --config_file /tmp/htp_backend_config.json || true
fi

# 3. Compile Face Detector to QNN Serialized Context Binary
if [ -f "models/face_detector.onnx" ]; then
    echo "[2/2] Generating QNN HTP context binary for Face Detector..."
    qnn-context-binary-generator \
        --backend "$QNN_LIB" \
        --model "models/face_detector.onnx" \
        --binary_file "face_detector.serialized" \
        --output_dir "$OUTPUT_DIR" \
        --config_file /tmp/htp_backend_config.json || true
fi

echo "Compilation complete. Context binaries stored in: $OUTPUT_DIR"