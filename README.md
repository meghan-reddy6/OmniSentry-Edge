# 👁️ OmniSentry-Edge: Multimodal Audio-Visual Directional Sensing & VLM Tracking Head

[![Hardware](https://img.shields.io/badge/Hardware-RubikPi%203%20%7C%20Qualcomm%20QCS6490-orange.svg)](#)
[![OS](https://img.shields.io/badge/OS-Ubuntu%2024.04%20%2F%20Qualcomm%20Linux-blue.svg)](#)
[![NPU Acceleration](https://img.shields.io/badge/NPU-Hexagon%20770%20%2812%20TOPS%29-green.svg)](#)
[![Actuation](https://img.shields.io/badge/Actuation-2--DOF%20Pan%2FTilt%20PID-red.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](#)

**OmniSentry-Edge** is an offline, real-time edge AI robotics framework built for the **Thundercomm RubikPi 3 (Qualcomm QCS6490 SoC)**.

It pairs multi-microphone array **Direction of Arrival (DoA)** sound localization and offline **Whisper / Keyword ASR** with **Hexagon NPU-accelerated computer vision** (YuNet & YOLO-World) to drive a 2-DOF Pan/Tilt camera platform at 30+ FPS.

---

## 🚀 Key Features

* 🎙️ **Acoustic Localization & Voice Assistant**:
  * Real-time **GCC-PHAT** Time Difference of Arrival (TDoA) calculation for sound source localization.
  * Adaptive ambient noise floor auto-calibration on startup.
  * Wake word activation (**"Sentry"**) with synthesized audio feedback tone and hands-free command parsing (*"track cup"*, *"track face"*, *"home"*).
* 👤 **Disambiguated Single-Person Face Tracking**:
  * Offloads face detection to **YuNet** with Non-Maximum Suppression (NMS).
  * Enforces the **Single-Person Rule**: locks on when 1 person is present; avoids jitter and aborts to home if multiple faces conflict.
* 🔍 **Open-Vocabulary VLM Object Tracking**:
  * Zero-shot text prompt grounding via **YOLO-World**.
  * Fast visual handoff to a 30+ FPS **CSRT/ByteTrack/KCF** engine with non-destructive periodic re-anchoring and 1.5s lost-target recovery.
* 🕹️ **2-DOF Closed-Loop PID Kinematics**:
  * Linear $2.5\text{ ticks/deg}$ mapping for standard 180° servos on **PCA9685** over `/dev/i2c-1`.
  * Direct SMBus2 I2C fallback driver for Qualcomm Linux boards.
* 📊 **Zero-Dependency Diagnostic Web HUD**:
  * Live HTTP MJPEG preview stream on `http://<rubikpi-ip>:8080/stream`.
  * Real-time bounding boxes, center crosshairs, dB audio meters, and voice transcript banners.

---

## 🛠️ Hardware Requirements & Pinout

| Component | Hardware Specification |
| :--- | :--- |
| **Compute Board** | Thundercomm RubikPi 3 (Qualcomm QCS6490, 8-Core Kryo, 12 TOPS NPU) |
| **Camera** | USB UVC Webcam (typically binds to `/dev/video2` or `/dev/video4`) |
| **Audio** | USB Multi-Microphone Array or I2S Microphone HAT |
| **Actuator Driver**| PCA9685 16-Channel PWM Servo Controller (Address: `0x40`) |
| **Servos** | 2-DOF Pan/Tilt SG90 or MG996R Servos (Pan: Ch 0, Tilt: Ch 1) |

### 40-Pin Header I2C Wiring for PCA9685
* **Pin 1**: 3.3V / VCC (Logic)
* **Pin 3**: I2C1_SDA
* **Pin 5**: I2C1_SCL
* **Pin 6 / 9**: GND
* **External 5V 2A-3A**: Connect to PCA9685 Servo V+ terminal block.

---

## 📦 Installation on RubikPi 3 (Qualcomm Linux / Ubuntu)

### 1. System Dependencies & User Permissions
```bash
sudo apt update
sudo apt install -y portaudio19-dev libasound2-dev i2c-tools v4l-utils python3-dev build-essential
sudo usermod -aG i2c,video,audio $USER
```

*(Log out and log back in for group permissions to take effect).*

### 2. Python Environment Setup

```bash
git clone https://github.com/<your-org>/OmniSentry-Edge.git
cd OmniSentry-Edge

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install onnxruntime pyaudio smbus2 adafruit-circuitpython-pca9685
```

### 3. Download ONNX Models

```bash
python scripts/download_models.py
```

---

## 🚦 Quick Start & Usage

### 1. Identify USB Camera Index

```bash
v4l2-ctl --list-devices
```

Ensure `config.yaml` has `camera_index` set to your USB device (usually `2` or `4`).

### 2. Launch the System

```bash
python src/main.py --config config.yaml
```

### 3. Access Web Diagnostics

Find your RubikPi IP with `hostname -I` and navigate to:

```
http://<RUBIKPI_IP>:8080/stream
```

### 4. Interaction Modes

* **Spoken Wake Word**: Say *"Sentry"* $\rightarrow$ wait for the chime $\rightarrow$ say *"track cell phone"* or *"track person"*.
* **Compound Voice Command**: Say *"Sentry track cup"*.
* **CLI Commands**:
  * `track <prompt>` (e.g. `track book`, `track face`, `track bottle`)
  * `home` (Return servos to 0°, 0°)
  * `say <phrase>` (Simulate spoken input, e.g. `say sentry track face`)
  * `exit`

---

## 🧪 Running Tests

```bash
pytest src/tests/test_agents.py -v
```
