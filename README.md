# OmniSentry-Edge (Mark-1)

Hardware-accelerated edge AI audio-visual tracking head optimized for the **Thundercomm Rubik Pi 3** (Qualcomm QCS6490 SoC, Hexagon NPU, Ubuntu 24.04 ARM64).

## Architecture & Features
- **Qualcomm Hexagon NPU Acceleration:** Executes quantized YOLOv8 object detection and face detection directly on the NPU via `QNNExecutionProvider` (`libQnnHtp.so`).
- **Pre-Compiled QNN Context Binaries:** Natively loads `*.serialized.bin` context files for instant inference startup.
- **Asynchronous Thread Decoupling:** Video capture and hardware inference are cleanly decoupled using background workers. The main loop uses fast `cv2.INTER_NEAREST` downscaling, ensuring a stable ~30 FPS video feed without being bottlenecked by NPU inference.
- **Smooth Kinematics:** Implements Exponential Moving Average (EMA) bounding box smoothing ($\alpha=0.65$) to eliminate servo jitter.
- **Directional Acoustic Seeking:** Utilizes GCC-PHAT for sound source localization, aggressively filtered with a confidence floor ($<0.40$) and a $250\text{ ms}$ debounce timer to prevent event bus flooding.
- **Strict Prompt Handover:** Safely transitions from `ACOUSTIC_SEEK` back to `IDLE` unless a user explicitly provides a tracking prompt (prevents phantom latching on empty spaces or ambient noise).
- **Diagnostics Web Preview:** Live MJPEG HUD stream with real-time target error telemetry, state transitions, and audio levels available at `http://<RUBIK_PI_IP>:8080/stream`.

## Deployment (Rubik Pi 3)

### 1. Clone & Auto-Setup
The automated setup script installs dependencies, pulls ONNX weights via Git LFS, downloads the QNN Execution Provider wheel, and preemptively compiles QNN serialized context binaries.
```bash
git clone <repo-url>
cd OmniSentry-Edge
chmod +x setup_rubikpi.sh compile_qnn_models.sh
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

