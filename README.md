# RubikPi 3 Audio-Visual Directional & VLM Tracking System

This project implements a 2-DOF Pan/Tilt audio-visual sensing and tracking head accelerated on the **Qualcomm Hexagon 770 NPU** of the **RubikPi 3 (QCS6490)** running **Yocto / Qualcomm Linux**.

---

## System Architecture

The software is built as an asynchronous multi-agent system coordinating over an internal pub/sub event bus:
*   **Orchestrator Agent**: Coordinates global state transitions (`IDLE`, `ACOUSTIC_SEEK`, `VISUAL_VERIFYING`, `VLM_TRACKING`, `RESETTING`) and coordinates command priorities.
*   **Audio Sensing Agent**: Streams multi-channel audio to calculate sound Direction of Arrival (DoA) via GCC-PHAT and RMS decibel voice activity detection.
*   **Vision & VLM Agent**: Performs fast face verification (YuNet ONNX) and open-vocabulary object grounding (YOLO-World ONNX) linked to a high-speed local CSRT tracker.
*   **Servo Actuator Agent**: Controls Pan/Tilt servos via the PCA9685 driver board over I2C and runs dual-axis PID loops to center targets.

---

## Installation & Setup

1.  **Initialize Virtual Environment & Install Core Requirements**:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate      # On Windows
    source .venv/bin/activate    # On Linux/macOS
    pip install -r requirements.txt
    ```

2.  **Download Model Weights**:
    Run the automated model downloader script to fetch the face detection and VLM models. If running in a virtual or offline test environment, it will automatically compile lightweight mock ONNX graphs:
    ```bash
    python scripts/download_models.py
    ```
    *To force-generate the lightweight mock models instantly without downloading full weights:*
    ```bash
    python scripts/download_models.py --dummy
    ```

---

## Real-Time Diagnostics HUD & Video Preview

The system features an integrated visual diagnostic suite that overlays tracking graphics onto camera frames.

### Configurations (`config.yaml`)
You can control the preview parameters in the `vision` section:
```yaml
vision:
  enable_preview: true
  preview_mode: "web"       # Options: "web" (MJPEG HTTP stream) or "gui" (cv2.imshow window)
  web_port: 8080            # Port to serve the MJPEG stream
  draw_hud: true            # Overlays crosshairs, target bounding boxes, and error vectors
```

### Accessing the Preview
*   **Web Stream Mode (`web`)**: Navigate to `http://localhost:8080/stream` (or your RubikPi's IP address) in any browser to watch the live camera preview with HUD overlays. This mode is fully non-blocking and ideal for headless Yocto Linux systems over SSH.
*   **GUI Mode (`gui`)**: Displays a local OpenCV window. If the script is running headlessly (no display server / missing `$DISPLAY`), the system automatically prints a warning and falls back to Web Stream Mode to maintain preview functionality.

### HUD Elements
*   **Red Crosshair**: Fixed at the camera frame center.
*   **Target Bounding Box**: Color-coded outline highlighting the locked target (Green for faces, Orange/Blue for VLM objects).
*   **Spatial Error Vector**: A line drawn from the center crosshair to the target center, visualizing the direction the servos are correcting.
*   **Top-Left Diagnostic Panel**: Real-time display of the active state name, processing FPS, target coordinates, and exact `dx`, `dy` PID error values.

---

## Verification & Testing

### Running the Unit Tests
Execute the automated test suite to verify the DSP mathematics, PID controllers, and Orchestrator state transitions:
```bash
python -m pytest src/tests/test_agents.py -v
```

### Running the Main System
Start the complete agent stack in simulation mode:
```bash
python src/main.py --config config.yaml
```
Use the interactive command-line interface to trigger manual actions:
*   Type `track <prompt>` (e.g., `track cup`) to seek and lock onto a target.
*   Type `home` to return the servos to center.
*   Type `exit` to cleanly stop all agents.
