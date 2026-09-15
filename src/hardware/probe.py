import os
import sys
import argparse
import logging
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HardwareProbe")

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import smbus2
except ImportError:
    smbus2 = None


def probe_cameras(max_index=6):
    """Probes available video devices and reports properties."""
    print("\n--- [1] SCANNING VIDEO / CAMERA DEVICES ---")
    available_cameras = []
    
    # Try V4L2 first on Linux, default fallback on Windows/Mac
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if hasattr(cv2, "CAP_V4L2") and os.name != "nt" else [cv2.CAP_ANY]

    for idx in range(max_index):
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    backend_name = "V4L2" if backend == getattr(cv2, "CAP_V4L2", -1) else "Default/DShow"
                    
                    info = {
                        "index": idx,
                        "backend": backend_name,
                        "resolution": f"{w}x{h}",
                        "fps": fps if fps > 0 else 30
                    }
                    available_cameras.append(info)
                    print(f"  [+] Found Camera Index {idx}: {w}x{h} @ ~{info['fps']}FPS via {backend_name}")
                    cap.release()
                    break
                cap.release()

    if not available_cameras:
        print("  [-] No functional camera sensors found.")
    return available_cameras


def probe_microphones():
    """Queries and categorizes audio input devices."""
    print("\n--- [2] SCANNING AUDIO INPUT / MICROPHONES ---")
    if sd is None:
        print("  [-] sounddevice not installed. Run: pip install sounddevice")
        return []

    input_devices = []
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]

        for i, dev in enumerate(devices):
            max_in = dev.get("max_input_channels", 0)
            if max_in > 0:
                is_default = (i == default_in)
                samplerate = int(dev.get("default_samplerate", 48000))
                
                # Check channel capability
                ch_desc = "Stereo (2ch)" if max_in >= 2 else "Mono (1ch)"
                tag = " [DEFAULT]" if is_default else ""
                
                info = {
                    "index": i,
                    "name": dev.get("name"),
                    "channels": max_in,
                    "samplerate": samplerate,
                    "is_default": is_default
                }
                input_devices.append(info)
                print(f"  [+] Mic Index {i}: \"{dev.get('name')}\" | {ch_desc} | {samplerate}Hz{tag}")

    except Exception as e:
        print(f"  [-] Audio probe error: {e}")

    if not input_devices:
        print("  [-] No active recording hardware detected.")
    return input_devices


def probe_i2c():
    """Checks for I2C bus and PCA9685 PWM chip availability."""
    print("\n--- [3] SCANNING I2C / PCA9685 SERVO BUS ---")
    if os.name == "nt":
        print("  [*] Running on Windows: I2C hardware unavailable -> SIMULATION mode required.")
        return False

    if not os.path.exists("/dev/i2c-1"):
        print("  [-] /dev/i2c-1 not present on filesystem -> Fallback to SIMULATION.")
        return False

    if smbus2 is None:
        print("  [-] smbus2 not installed. Run: pip install smbus2")
        return False

    try:
        bus = smbus2.SMBus(1)
        # Test probe PCA9685 default address 0x40
        bus.read_byte_data(0x40, 0x00)
        bus.close()
        print("  [+] I2C Bus 1 accessible & PCA9685 responded at address 0x40!")
        return True
    except Exception as e:
        print(f"  [-] I2C bus found, but PCA9685 probe failed at 0x40 ({e}).")
        return False


def run_full_probe(update_config=False):
    cams = probe_cameras()
    mics = probe_microphones()
    i2c_ok = probe_i2c()

    print("\n================ HARDWARE PROBE SUMMARY ================")
    rec_cam = cams[0]["index"] if cams else None
    
    # Prefer stereo default, else default, else first available
    rec_mic = None
    mic_channels = 1
    if mics:
        stereos = [m for m in mics if m["channels"] >= 2]
        if stereos:
            rec_mic = stereos[0]["index"]
            mic_channels = 2
        else:
            defaults = [m for m in mics if m["is_default"]]
            rec_mic = defaults[0]["index"] if defaults else mics[0]["index"]
            mic_channels = mics[0]["channels"]

    print(f"Recommended Camera Device Index : {rec_cam}")
    print(f"Recommended Audio Device Index  : {rec_mic} ({mic_channels}ch)")
    print(f"Servo Actuator Mode             : {'PHYSICAL (I2C 0x40)' if i2c_ok else 'SIMULATED (MOCK)'}")
    print("========================================================\n")

    if update_config:
        _update_config_file(rec_cam, rec_mic, mic_channels)


def _update_config_file(cam_idx, mic_idx, channels, config_path="config.yaml"):
    import yaml
    
    # Resolve relative to project root
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    full_path = os.path.join(root_dir, config_path)
    
    if not os.path.exists(full_path):
        print(f"[!] {full_path} not found. Skipping auto-write.")
        return

    try:
        with open(full_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        if cam_idx is not None:
            cfg.setdefault("vision", {}).setdefault("camera", {})["index"] = cam_idx
        if mic_idx is not None:
            cfg.setdefault("audio", {})["hardware"] = cfg.get("audio", {}).get("hardware", {})
            cfg["audio"]["hardware"]["device_index"] = mic_idx
            cfg["audio"]["hardware"]["channels"] = channels

        with open(full_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"[+] Updated {full_path} with probed device indices.")
    except Exception as e:
        print(f"[-] Failed to update {full_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniSentry-Edge Hardware Discovery Tool")
    parser.add_argument("--update-config", action="store_true", help="Auto-write discovered indices to config.yaml")
    args = parser.parse_args()
    run_full_probe(update_config=args.update_config)
