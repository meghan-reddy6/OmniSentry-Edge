# OmniSentry-Edge Agent Subsystems Specification

## 1. VisionVLMAgent (`src/agents/vision_agent.py`)
- **Execution Model:** Fully decoupled multi-threaded pipeline.
  - `_camera_capture_worker`: Continuously drains the V4L2 ring buffer at 30 FPS.
  - `_async_npu_worker`: Runs INT8 tensor inference on Qualcomm Hexagon NPU at $\le 22\text{ FPS}$ throttle.
- **Quantization & De-Quantization:**
  - Input: `[1, 3, 640, 640]` UINT8 `[0..255]` RGB.
  - Outputs: Slices QAI-Hub 3-tensor signature: `boxes` $[1, 8400, 4]$, `scores` $[1, 8400]$, and `class_idx` $[1, 8400]$.
- **Target Smoothing:** Exponential Moving Average (EMA) bounding box filter ($\alpha = 0.65$).
- **Diagnostics Web Server:** Multi-threaded HTTP server broadcasting `multipart/x-mixed-replace` MJPEG stream with HUD telemetry on port 8080.

## 2. AudioSensingAgent (`src/agents/audio_agent.py`)
- **Acoustic Localization:** GCC-PHAT (Generalized Cross-Correlation Phase Transform) Time-Difference-Of-Arrival (TDoA):
  $$\tau = \arg\max_t \left( \mathcal{F}^{-1} \left[ \frac{X_1(f) X_2^*(f)}{\vert{}X_1(f) X_2^*(f)\vert{}} \right] \right)$$
  $$\theta = \arcsin\left(\frac{\tau \cdot v_{\text{sound}}}{d_{\text{mic}}}\right)$$
- **Adaptive VAD:** Dynamic noise floor calibration on startup with threshold gating ($\text{SNR} > 7\text{ dB}$).
- **De-Jitter:** Debounced cooldown timer ($0.25\text{s}$) with confidence filtering ($\ge 0.45$).

## 3. ServoActuatorAgent (`src/agents/servo_agent.py`)
- **Driver:** Native I2C `adafruit_pca9685` / `busio` driver with seamless mock fallback for local simulation.
- **Kinematic Constraints:**
  - Pan Range: $[-90.0^\circ, +90.0^\circ]$
  - Tilt Range: $[-30.0^\circ, +45.0^\circ]$
- **Visual Closed-Loop PID:** Proportional-derivative tracking loop responding to normalized camera center error vectors.

## 4. OrchestratorAgent (`src/agents/orchestrator.py`)
- **Finite State Machine:**
  - `IDLE`: Listens for acoustic localization triggers or CLI prompts.
  - `ACOUSTIC_SEEK`: Sells gimbal towards localized sound angle $(\theta)$.
  - `VLM_TRACKING`: Locks visual neural tracking onto target prompt and commands closed-loop tracking.
