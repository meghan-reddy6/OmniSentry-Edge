# Project Architecture & Agent Specification: RubikPi 3 Audio-Visual Directional & VLM Tracking System

This document outlines the software architecture, agent responsibilities, inter-process communication, state machines, and hardware execution strategies for building an audio-visual sensing and tracking head on the **RubikPi 3 (Qualcomm QCS6490)** running **Yocto / Qualcomm Linux**.

---

## 1. System Overview & Core Mission

The system drives a 2-DOF Pan/Tilt camera head using multimodal sensing (microphone array + UVC webcam) accelerated on the **Qualcomm Hexagon 770 NPU**.

The board operates in two primary modes:
1. **Directional Sound Seeking & Visual Verification Mode (Acoustic Face-Tracking)**
   * **Trigger**: Acoustic energy spike above decibel/VAD threshold.
   * **Action**: Estimate Direction of Arrival (DoA) via TDoA/GCC-PHAT, rotate pan/tilt servos toward the sound source.
   * **Verification**: Capture camera frames to run fast, NPU-accelerated face detection (e.g., INT8 face detector via ONNX Runtime QNN EP).
   * **Fallback**: If no face is visually verified within a configurable timeout (default $3.0\text{s}$), return servos to the home neutral position $(0^\circ, 0^\circ)$.
2. **Vision-Language Model (VLM) & Open-Vocabulary Tracking Mode**
   * **Trigger**: High-level natural language prompt (e.g., `"find the red mug"` or `"track the blue bottle"`).
   * **Action**: Run zero-shot grounding via an INT8 quantized model (e.g., YOLO-World / Grounding DINO on QNN EP) **once** to obtain the target bounding box.
   * **Control Loop**: Hand off bounding box coordinates to a high-speed visual tracker (e.g., ByteTrack / CSRT running at $30+\text{ FPS}$) driving dual-axis PID servo controls to keep the target centered in the field of view.

---

## 2. Hardware Architecture & Hardware Interfaces

```
 [ USB Mic Array / I2S HAT ] ----(USB/I2S)----+
                                             |
 [ USB UVC Camera ] -----------(UVC/V4L2)----+---> [ RubikPi 3 (QCS6490) ]
                                             |     ├── Kryo 670 CPU (Orchestration)
 [ CLI / Web / ZeroMQ Input ] --------------+     └── Hexagon 770 NPU (QNN / INT8)
                                                        |
                                                    (I2C / PWM)
                                                        v
                                             [ PCA9685 Servo Driver ]
                                                        |
                                             [ 2-DOF Pan/Tilt Servos ]
```

---

## 3. Asynchronous Multi-Agent Architecture

To maximize performance and prevent frame drop on the RubikPi 3, the software operates as a set of decoupled, asynchronous agent micro-services communicating over an internal event bus or ZeroMQ channels.

```
+-----------------------------------------------------------------------------------+
|                                 ORCHESTRATOR AGENT                                |
|                   (Global State Machine, Mode Switching, Timeouts)                |
+--------+--------------------------+--------------------------+--------------------+
         ^                          ^                          ^
         | Events                   | Servo Commands           | Bounding Boxes
         v                          v                          v
+------------------+      +-------------------+      +-------------------+
| AUDIO SENSING    |      | SERVO ACTUATOR    |      | VISION & VLM      |
| AGENT            |      | AGENT             |      | AGENT             |
| - GCC-PHAT DoA   |      | - Dual PID Loop   |      | - QNN EP Face Det |
| - Noise Threshold|      | - I2C / PCA9685   |      | - YOLO-World INT8 |
| - VAD Filtering  |      | - Home / Sweeps   |      | - ByteTrack (30FPS|
+------------------+      +-------------------+      +-------------------+
```

---

## 4. Agent Specifications

### 4.1 Orchestrator Agent (`OrchestratorAgent`)
* **Role**: System state machine coordinator and priority arbiter.
* **Responsibilities**:
  * Maintain system states: `IDLE`, `ACOUSTIC_SEEK`, `VISUAL_VERIFYING`, `VLM_TRACKING`, `RESETTING`.
  * Arbitrate priorities (e.g., user VLM tracking commands preempt audio-seeking mode).
  * Manage timeout timers: If `VISUAL_VERIFYING` fails to find a face within $3.0\text{s}$, send a `MOVE_HOME` command to `ServoActuatorAgent`.
* **State Transition Logic**:
  $$\text{IDLE} \xrightarrow{\text{Sound Detected}} \text{ACOUSTIC\_SEEK} \xrightarrow{\text{Motion Complete}} \text{VISUAL\_VERIFYING}$$
  $$\text{VISUAL\_VERIFYING} \xrightarrow{\text{Face Confirmed}} \text{TRACKING}$$
  $$\text{VISUAL\_VERIFYING} \xrightarrow{\text{Timeout (3.0s)}} \text{RESETTING} \rightarrow \text{IDLE}$$

### 4.2 Audio Sensing Agent (`AudioSensingAgent`)
* **Role**: Low-latency multi-channel audio acquisition and Sound Source Localization (SSL).
* **Responsibilities**:
  * Stream multi-channel PCM audio from the mic array via PyAudio / `sounddevice`.
  * Compute Generalized Cross-Correlation with Phase Transform (GCC-PHAT) to estimate azimuth angle $\theta_{\text{azimuth}}$.
  * Apply decibel and Voice Activity Detection (VAD) gates to ignore background ambient noise.
* **Emitted Events**:
  * `SOUND_LOCALIZED(angle_degrees, confidence)`

### 4.3 Vision & VLM Agent (`VisionVLMAgent`)
* **Role**: Hardware-accelerated vision pipeline running on Qualcomm Hexagon NPU.
* **Responsibilities**:
  * **Face Verification Mode**: Execute fast INT8 face detection on incoming video frames using ONNX Runtime with Qualcomm QNN EP (`libqnn_hp.so`).
  * **VLM Grounding Mode**: Execute open-vocabulary detection (YOLO-World / Grounding DINO) ONCE upon receiving text prompts to obtain target bounding box $(x_c, y_c, w, h)$.
  * **High-Speed Tracking**: Hand off initial target box to a lightweight feature tracker (ByteTrack/CSRT) running at $30+\text{ FPS}$ on CPU/GPU to maintain real-time PID responsiveness.
* **Emitted Events**:
  * `TARGET_VERIFIED(center_x, center_y)`
  * `TARGET_NOT_FOUND`
  * `TRACKING_ERROR(dx, dy)`

### 4.4 Servo Actuator Agent (`ServoActuatorAgent`)
* **Role**: Multi-axis motion control and PID kinematic smoothing.
* **Responsibilities**:
  * Communicate with PCA9685 servo driver board over I2C (`/dev/i2c-1`).
  * Translate continuous tracking spatial error signals $(dx, dy)$ into pan/tilt angle adjustments using dual PID controllers.
  * Translate absolute acoustic angles $\theta$ into pan servo positions.
  * Enforce physical safety constraints (e.g., Pan: $-90^\circ$ to $+90^\circ$, Tilt: $-30^\circ$ to $+45^\circ$).
  * Drive smooth return-to-home movement routines.

---

## 5. Sequence Workflows

### 5.1 Directional Sound Seeking & Verification Loop
```
[Audio Agent]         [Orchestrator]         [Servo Agent]         [Vision Agent]
      |                     |                      |                      |
      |-- SOUND_LOCALIZED ->|                      |                      |
      |   (Angle: +45°)     |                      |                      |
      |                     |-- MOVE_TO(+45°) ---->|                      |
      |                     |                      |-- (Pan to +45°)      |
      |                     |<-- MOTION_DONE ------|                      |
      |                     |                                             |
      |                     |-- VERIFY_FACE ----------------------------->|
      |                     |   (Start 3.0s Timer)                        |
      |                     |                                             |
      |                     |              [ Case A: Face Verified ]      |
      |                     |<-- TARGET_VERIFIED -------------------------|
      |                     |    (Engage Focus State)                     |
      |                     |                                             |
      |                     |              [ Case B: Timeout / No Face ]  |
      |                     |<-- TARGET_NOT_FOUND ------------------------|
      |                     |-- MOVE_HOME ------------------------------->|
      |                     |                      |-- (Pan to 0°, 0°)    |
```

### 5.2 Open-Vocabulary VLM Grounding & Tracking
```
[User Input]          [Orchestrator]         [Vision Agent]        [Servo Agent]
     |                      |                      |                     |
     |-- TRACK("blue cup")->|                      |                     |
     |                      |-- RUN_GROUNDING ---->|                     |
     |                      |                      |-- (QNN INT8 Ground) |
     |                      |                      |-- Box: (x, y, w, h) |
     |                      |                      |-- Init ByteTrack    |
     |                      |                      |                     |
     |                      |                      |-- ERR(dx, dy) ----->|
     |                      |                      |                     |-- PID Step & Move
     |                      |                      |-- ERR(dx, dy) ----->|
     |                      |                      |                     |-- PID Step & Move
```

---

## 6. Qualcomm Edge Optimization Guidelines

1. **Zero-Copy Frame Pipeline**: Ensure video capture (V4L2) shares frame buffers directly with ONNX Runtime memory allocations to prevent CPU memory copies.
2. **Two-Stage VLM Strategy**: Never run multi-billion parameter VLM inferences every frame. Run zero-shot grounding once, then track the bounding box with lightweight CV algorithms at $30+\text{ FPS}$.
3. **NPU Quantization**: All vision models must be converted to INT8 precision using Qualcomm AI Hub / QNN SDK to utilize Hexagon 770 NPU tensor processing.
