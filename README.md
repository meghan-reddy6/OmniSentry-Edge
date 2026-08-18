# OmniSentry-Edge (Mark-1)

Edge AI audio-visual tracking head optimized for the **Thundercomm Rubik Pi 3** (Qualcomm QCS6490 SoC, ARM64 Ubuntu 22.04) and cross-platform simulation.

## Key Features
- **Zero-Setup Offline Boot:** Bundled with hardware-quantized Qualcomm AI Hub ONNX models (`yolov8_det.onnx`, `face_detector.onnx`, `whisper_tiny_en_int8.onnx`).
- **Real-Time VLM Object & Face Tracking:** Runs multi-class object detection and head tracking directly via ONNX Runtime with Qualcomm QNN EP acceleration (CPU fallback for development).
- **Diagnostics Web Stream:** Live MJPEG HUD overlay stream at `http://localhost:8080/stream`.
- **Adaptive Audio Sensing:** Real-time VAD speech command grounding.

## Quick Start

### 1. Clone with Git LFS
```bash
git clone <repo-url>
cd mark1
git lfs pull
```

### 2. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run the Stack
```bash
python src/main.py
```

## CLI Commands
- `track person` - Track full-body presence
- `track face` - Track head/face region
- `track <object>` - Track COCO objects (e.g. `track chair`, `track cell phone`)
- `home` - Center pan/tilt servos to (0, 0)
- `exit` - Gracefully shutdown all agents
