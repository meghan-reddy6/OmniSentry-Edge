import time
import logging
from src.common.bus import Event

logger = logging.getLogger(__name__)

class ServoTargetReachedEvent(Event):
    def __init__(self, pan: float, tilt: float):
        self.pan = pan
        self.tilt = tilt

class ServoActuatorAgent:
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config

        srv_cfg = self.config.get("servo", {})
        self.driver_type = srv_cfg.get("driver", "pca9685")
        self.i2c_bus_num = srv_cfg.get("i2c_bus", 1)
        self.i2c_addr = srv_cfg.get("i2c_address", 0x40)

        self.pan_cfg = srv_cfg.get("pan", {})
        self.tilt_cfg = srv_cfg.get("tilt", {})

        self.current_pan = self.pan_cfg.get("home_angle_deg", 0.0)
        self.current_tilt = self.tilt_cfg.get("home_angle_deg", 0.0)

        self._hardware_ready = False
        self._init_pca9685()

        self.bus.subscribe("MoveServoCommand", self.handle_move_command)

    def _init_pca9685(self):
        try:
            import busio
            from board import SCL, SDA
            from adafruit_pca9685 import PCA9685
            from adafruit_motor import servo

            i2c = busio.I2C(SCL, SDA)
            self.pca = PCA9685(i2c, address=self.i2c_addr)
            self.pca.frequency = self.config.get("servo", {}).get("pwm_frequency", 50)

            self.pan_servo = servo.Servo(self.pca.channels[self.pan_cfg.get("channel", 0)])
            self.tilt_servo = servo.Servo(self.pca.channels[self.tilt_cfg.get("channel", 1)])
            self._hardware_ready = True
            logger.info("[ServoAgent]: Native PCA9685 hardware initialized on I2C.")
        except Exception as e:
            logger.warning(f"[ServoAgent]: PCA9685 hardware init failed ({e}). Operating in simulation.")
            self._hardware_ready = False

    def handle_move_command(self, event):
        target_pan = getattr(event, 'pan', self.current_pan)
        target_tilt = getattr(event, 'tilt', self.current_tilt)

        # Clamp against physical limits
        target_pan = max(self.pan_cfg.get("min_angle_deg", -90.0), min(self.pan_cfg.get("max_angle_deg", 90.0), target_pan))
        target_tilt = max(self.tilt_cfg.get("min_angle_deg", -30.0), min(self.tilt_cfg.get("max_angle_deg", 45.0), target_tilt))

        self.current_pan = target_pan
        self.current_tilt = target_tilt

        if self._hardware_ready:
            try:
                # Map [-90, +90] to [0, 180] for Adafruit servo API
                self.pan_servo.angle = target_pan + 90.0
                self.tilt_servo.angle = target_tilt + 90.0
            except Exception as e:
                logger.error(f"[ServoAgent]: Servo write error: {e}")

        self.bus.publish(ServoTargetReachedEvent(pan=self.current_pan, tilt=self.current_tilt))

    async def start(self):
        logger.info("[ServoAgent]: Servo actuator agent started.")
        return True

    async def stop(self):
        logger.info("[ServoAgent]: Servo actuator stopped.")
