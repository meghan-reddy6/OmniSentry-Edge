"""
Configuration module for the RubikPi 3 Audio-Visual Directional & VLM Tracking System.
Loads parameters from config.yaml and provides default fallback values.
"""
import os
import logging

logger = logging.getLogger(__name__)

# Fallback defaults if config file is missing or values are undefined
DEFAULT_CONFIG = {
    "simulation_mode": True,
    "audio": {
        "sample_rate": 16000,
        "channels": 2,
        "chunk_size": 1024,
        "vad_threshold_db": -45.0,  # dB relative to full scale
        "mic_distance": 0.08,       # meters (8 cm)
        "speed_of_sound": 343.0,    # m/s
    },
    "servo": {
        "i2c_bus": 1,
        "pca9685_address": 0x40,
        "pan_channel": 0,
        "tilt_channel": 1,
        "pan_min_angle": -90.0,
        "pan_max_angle": 90.0,
        "tilt_min_angle": -30.0,
        "tilt_max_angle": 45.0,
        "pan_pid": {
            "kp": 0.05,
            "ki": 0.005,
            "kd": 0.001
        },
        "tilt_pid": {
            "kp": 0.05,
            "ki": 0.005,
            "kd": 0.001
        },
        "home_pan": 0.0,
        "home_tilt": 0.0
    },
    "vision": {
        "camera_index": 0,
        "frame_width": 640,
        "frame_height": 480,
        "face_model_path": "models/face_detector.onnx",
        "vlm_model_path": "models/yolo_world.onnx",
        "tracking_timeout": 3.0,     # seconds
        "verify_threshold": 0.5,     # Confidence threshold
    }
}

class SystemConfig:
    """System-wide configuration registry."""
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = DEFAULT_CONFIG.copy()
        self.load_config()

    def load_config(self):
        """Attempts to load and parse the configuration file."""
        if not os.path.exists(self.config_path):
            logger.warning(
                f"Config file not found at {self.config_path}. Using system defaults."
            )
            return

        try:
            import yaml
            with open(self.config_path, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    self._update_recursive(self._config, loaded)
                    logger.info(f"Successfully loaded configuration from {self.config_path}")
                else:
                    logger.warning("Empty or invalid config file. Using default settings.")
        except ImportError:
            logger.warning("PyYAML not installed. Falling back to default configuration dictionary.")
        except Exception as e:
            logger.error(f"Error reading configuration file: {e}. Falling back to default settings.")

    def _update_recursive(self, base_dict: dict, update_dict: dict):
        """Recursively updates nested dictionaries."""
        for k, v in update_dict.items():
            if isinstance(v, dict) and k in base_dict and isinstance(base_dict[k], dict):
                self._update_recursive(base_dict[k], v)
            else:
                base_dict[k] = v

    def get(self, key: str, default=None):
        """Fetches a top-level configuration section or value."""
        return self._config.get(key, default)

    @property
    def simulation_mode(self) -> bool:
        return self._config.get("simulation_mode", True)

    @property
    def audio(self) -> dict:
        return self._config.get("audio", DEFAULT_CONFIG["audio"])

    @property
    def servo(self) -> dict:
        return self._config.get("servo", DEFAULT_CONFIG["servo"])

    @property
    def vision(self) -> dict:
        return self._config.get("vision", DEFAULT_CONFIG["vision"])
