# OmniSentry-Edge (Mark-1)

Hardware-accelerated edge AI audio-visual tracking head optimized for the **Thundercomm Rubik Pi 3** (Qualcomm QCS6490 SoC, Hexagon NPU).

## Features
- **Qualcomm Hexagon NPU Acceleration:** Executes quantized YOLOv8 object detection on NPU via `QNNExecutionProvider`.
- **Diagnostics Web Preview:** Live MJPEG HUD stream with real-time target error telemetry at `http://<RUBIK_PI_IP>:8080/stream`.
- **Low-Latency Tracking Engine:** Pure detector-based inference with temporal box smoothing.

## Step-by-Step Setup (Rubik Pi 3)

### 1. Clone & Run Setup
```bash
git clone <repo-url>
cd OmniSentry-Edge
chmod +x setup_rubikpi.sh
./setup_rubikpi.sh
```

### 2. Run the Stack
```bash
source venv/bin/activate
python src/main.py
```

## Interactive CLI Commands
- `track person` - Track full-body presence
- `track face` - Track head/face region
- `track chair` - Track chair objects
- `home` - Center pan/tilt servos (0, 0)
- `exit` - Gracefully shutdown all agents
