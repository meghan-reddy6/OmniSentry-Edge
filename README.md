# OmniSentry-Edge (Mark-1)

Edge AI audio-visual tracking head optimized for the **Thundercomm Rubik Pi 3** (Qualcomm QCS6490 SoC, ARM64 Ubuntu 22.04) and cross-platform simulation.

## Key Features
- **Zero-Setup Offline Boot:** Bundled with hardware-quantized Qualcomm AI Hub ONNX models (`yolov8_det.onnx`, `face_detector.onnx`, `whisper_tiny_en_int8.onnx`).
- **Real-Time VLM Object & Face Tracking:** Runs multi-class object detection and head tracking directly via ONNX Runtime with Qualcomm QNN EP acceleration (CPU fallback for development).
- **Diagnostics Web Stream:** Live MJPEG HUD overlay stream at `http://localhost:8080/stream`.
- **Adaptive Audio Sensing:** Real-time VAD speech command grounding.

## Quick Start

### Option A: Automated Hardware Setup (Thundercomm Rubik Pi 3)
The easiest way to get started on the Rubik Pi 3 is to run the automated setup script. This script installs all required system packages (including the Qualcomm QNN & QIRP SDKs), pulls the large binary models using Git LFS, creates a Python virtual environment, and installs the runtime dependencies.

```bash
# 1. Clone the repository
git clone <repo-url>
cd mark1

# 2. Run the deployment script
chmod +x setup_rubikpi.sh
./setup_rubikpi.sh

# 3. Activate the environment and run the stack
source venv/bin/activate
python src/main.py
```

### Option B: Manual Setup (Local Development / PC)
If you are running the project in a cross-platform simulation environment (Windows/Linux PC) without Qualcomm QNN, follow these steps:

```bash
# 1. Clone with Git LFS
git clone <repo-url>
cd mark1
git lfs pull

# 2. Install Dependencies
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Run the Stack
python src/main.py
```

## CLI Commands
- `track person` - Track full-body presence
- `track face` - Track head/face region
- `track <object>` - Track COCO objects (e.g. `track chair`, `track cell phone`)
- `home` - Center pan/tilt servos to (0, 0)
- `exit` - Gracefully shutdown all agents
