# OmniSentry-Edge: Multimodal Edge AI Tracking Stack

[![Hardware](https://img.shields.io/badge/Hardware-Thundercomm%20Rubik%20Pi%203-blue.svg)](https://www.thundercomm.com/)
[![NPU Target](https://img.shields.io/badge/NPU-Qualcomm%20Hexagon%20HTP-0052cc.svg)]()
[![Runtime](https://img.shields.io/badge/Runtime-ONNX%20QNN%20EP%201.23.0-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.12%20aarch64-yellow.svg)]()

**OmniSentry-Edge** is a real-time, event-driven audio-visual robotic tracking stack deployed on the Qualcomm QCS6490 SoC. It fuses Phase-Delay Acoustic Direction-of-Arrival (TDoA / GCC-PHAT) localization with Hardware-Accelerated YOLOv8 object detection running on the Qualcomm Hexagon Tensor Processor (HTP), driving a 2-Axis closed-loop pan/tilt gimbal via I2C PWM actuation.

---

## 🏗️ System Architecture

```text
  +-----------------------------------------------------------------------------------+
  |                                PHYSICAL SENSING & I/O                             |
  |  [Stereo Mic Array (ALSA)]       [V4L2 Camera (USB/CSI)]      [PCA9685 I2C Servos]|
  +-----------------+--------------------------+---------------------------+----------+
                    |                          |                           ^
                    v                          v                           |
  +-----------------+--------------------------+---------------------------+----------+
  |                           EDGE AGENT SUBSYSTEMS                                   |
  |                                                                                   |
  |  +--------------------+   +---------------------------------+  +---------------+  |
  |  | AudioSensingAgent  |   |        VisionVLMAgent           |  |ServoActuator  |  |
  |  | - 16kHz PyAudio    |   | - 30 FPS Async V4L2 Ingestion   |  | - smbus2 I2C  |  |
  |  | - Dynamic VAD (dB) |   | - QNN HTP (Hexagon UINT8 NPU)   |  | - PID Loop    |  |
  |  | - GCC-PHAT TDoA    |   | - EMA Bbox Smoothing (α=0.65)   |  | - Deadband    |  |
  |  | - Angle Smoothing  |   | - HTTP 30 FPS MJPEG Stream      |  | - Clamping    |  |
  |  +--------+-----------+   +---------------+-----------------+  +-------^-------+  |
  |           |                               |                            |          |
  +-----------|-------------------------------|----------------------------|----------+
              |                               |                            |
              v                               v                            |
  +------------------------------------------------------------------------+----------+
  |                   THREAD-SAFE ASYNC EVENT BUS (src/common/bus.py)                 |
  |  Events: SoundLocalizedEvent, VisualTargetOffsetEvent, MoveServoCommand, StateChange|
  +-----------------------------------+-----------------------------------------------+
                                      ^
                                      |
  +-----------------------------------+-----------------------------------------------+
  |                 MULTIMODAL ORCHESTRATOR (State Machine Engine)                    |
  |                                                                                   |
  |         [ STANDBY / IDLE ]  <=====>  [ ACOUSTIC_SEEK ]                            |
  |                 ^                            |                                    |
  |                 +======== [ VLM_TRACKING ] <=+                                    |
  +-----------------------------------------------------------------------------------+
```

## ⚡ Performance Metrics on Qualcomm QCS6490 (Rubik Pi 3)

| Metric | Measurement / Specification |
| :--- | :--- |
| **NPU Inference Engine** | Qualcomm Hexagon Vector Extensions (HTP) via QNNExecutionProvider |
| **Model Format** | UINT8 Quantized YOLOv8 [1, 3, 640, 640] NCHW |
| **Inference Latency** | $4.2\text{ ms} \pm 0.4\text{ ms}$ per frame on NPU |
| **End-to-End Pipeline FPS** | $30.0\text{ FPS}$ synchronized capture & display |
| **Total CPU Utilization** | $12\text{--}18\%$ across 8 Kryo CPU cores (C4–C7 sleep) |
| **Operating Temperature** | $44^\circ\text{C} \pm 3^\circ\text{C}$ with thermal cooling daemon active |

## 🚀 Quick Start Guide

### 1. Hardware Prerequisites
- **Board**: Thundercomm Rubik Pi 3 (Qualcomm QCS6490 SoC, 8GB LPDDR5).
- **Camera**: Standard V4L2 USB / MIPI camera (`/dev/video0`).
- **Microphone**: Stereo 2-channel ALSA microphone array.
- **Actuation**: PCA9685 I2C PWM driver connected to I2C Bus 1 (`/dev/i2c-1`, address 0x40).
  - **Pan (Ch 0)**: Range $0^\circ\text{--}180^\circ$, Base $90^\circ$
  - **Tilt (Ch 1)**: Range $45^\circ\text{--}135^\circ$, Base $70^\circ$
  - *Note: Servo mode can be switched to `"simulation"` in `config.yaml` for offline testing.*

### 2. Installation & Setup
```bash
git clone https://github.com/meghan-reddy6/OmniSentry-Edge.git
cd OmniSentry-Edge

# Run the automated Rubik Pi 3 provisioner
chmod +x setup_rubikpi.sh scripts/compile_qnn_ctx.py
./setup_rubikpi.sh
```

### 3. Launching the Stack
```bash
source venv/bin/activate
python src/main.py
```

### 4. Live Diagnostics Dashboard
Open your web browser and navigate to:
```text
http://<RUBIKPI_IP>:8080/
```
The dashboard streams an annotated 30 FPS video feed showing target lock corner brackets, Center-of-Vision error vectors, and live audio/gimbal telemetry.

## 🎮 Interactive CLI Commands

| Command | Action | Example |
| :--- | :--- | :--- |
| `track <prompt>` | Engages VLM tracking loop for specified target | `track person`, `track cup`, `track face` |
| `home` | Centers Pan/Tilt servos back to (0.0°, 0.0°) | `home` |
| `say <phrase>` | Injects a simulated voice transcription | `say track the red bottle` |
| `exit` | Gracefully shuts down all worker threads | `exit` |
