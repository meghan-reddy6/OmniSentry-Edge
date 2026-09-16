import time
import logging
from src.common.bus import MoveServoCommand, ServoTargetReachedEvent
from src.hardware.pca9685 import PCA9685Direct

logger = logging.getLogger(__name__)

class ServoActuatorAgent:
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config
        servo_cfg = self.config.get("servos", {})

        # Mode configuration: "hardware" or "simulation"
        self.mode = servo_cfg.get("mode", "hardware").lower()
        self.bus_num = servo_cfg.get("i2c_bus", 1)
        self.address = servo_cfg.get("i2c_address", 0x40)

        self.pan_cfg = servo_cfg.get("pan", {})
        self.tilt_cfg = servo_cfg.get("tilt", {})

        self.min_angle_change = float(servo_cfg.get("min_angle_change_deg", 1.5))
        self.min_interval = 1.0 / float(servo_cfg.get("update_rate_hz", 15))

        # Initial live angles
        self.current_pan = float(self.pan_cfg.get("base_angle", 90))
        self.current_tilt = float(self.tilt_cfg.get("base_angle", 75))
        self._last_write_time = 0.0

        self.driver = None
        self._init_hardware()

        self.bus.subscribe("MoveServoCommand", self.handle_move_command)
        self.bus.subscribe("HomeServosCommand", self.handle_home_command)

    def _init_hardware(self):
        if self.mode == "simulation":
            logger.info("[ServoAgent]: Mode set to SIMULATION. Hardware I2C disabled.")
            return

        try:
            self.driver = PCA9685Direct(bus_num=self.bus_num, address=self.address)
            self.set_angles(self.current_pan, self.current_tilt)
            logger.info(f"[ServoAgent]: PCA9685 hardware active on /dev/i2c-{self.bus_num} (Address: 0x{self.address:02X})")
        except Exception as e:
            logger.warning(f"[ServoAgent]: smbus2 hardware init failed: {e}. Falling back to SIMULATION.")
            self.mode = "simulation"

    def handle_move_command(self, event):
        target_pan = getattr(event, "pan", self.current_pan)
        target_tilt = getattr(event, "tilt", self.current_tilt)
        self.set_angles(target_pan, target_tilt)

    def handle_home_command(self, event):
        self.home()

    def set_angles(self, pan_angle, tilt_angle):
        now = time.time()
        if (now - self._last_write_time) < self.min_interval:
            return

        # 1. Clamp to mechanical physical limits
        clamped_pan = max(self.pan_cfg.get("min_angle", 10), min(self.pan_cfg.get("max_angle", 170), float(pan_angle)))
        clamped_tilt = max(self.tilt_cfg.get("min_angle", 40), min(self.tilt_cfg.get("max_angle", 125), float(tilt_angle)))

        # 2. Anti-shiver threshold: suppress micro-jitter smaller than deadband
        delta_p = abs(clamped_pan - self.current_pan)
        delta_t = abs(clamped_tilt - self.current_tilt)
        if delta_p < self.min_angle_change and delta_t < self.min_angle_change:
            return

        self.current_pan = clamped_pan
        self.current_tilt = clamped_tilt
        self._last_write_time = now

        # 3. Apply centralized hardware axis inversion right before physical bus write
        hw_pan = (180.0 - clamped_pan) if self.pan_cfg.get("invert", False) else clamped_pan
        hw_tilt = (180.0 - clamped_tilt) if self.tilt_cfg.get("invert", False) else clamped_tilt

        self._write_pca9685(int(round(hw_pan)), int(round(hw_tilt)))

    def _write_pca9685(self, pan: int, tilt: int):
        if self.mode == 'simulation':
            logger.debug(f"[SERVO ] Write -> Pan:{pan:3d} deg Tilt:{tilt:3d} deg (was P:{int(self.current_pan):3d} deg T:{int(self.current_tilt):3d} deg)")
        else:
            logger.info(f"[ServoAgent] Hardware Write -> Pan: {pan} deg, Tilt: {tilt} deg (was {self.current_pan} deg, {self.current_tilt} deg)")

        if self.mode == "hardware" and self.driver:
            try:
                # Explicit bounds clamping via hardware driver using raw config bounds since we already clamped
                self.driver.set_servo_angle(self.pan_cfg.get("channel", 0), pan, 
                                      min_angle=self.pan_cfg.get("min_angle", 0), max_angle=self.pan_cfg.get("max_angle", 180))
                self.driver.set_servo_angle(self.tilt_cfg.get("channel", 1), tilt, 
                                      min_angle=self.tilt_cfg.get("min_angle", 0), max_angle=self.tilt_cfg.get("max_angle", 180))
            except Exception as e:
                logger.error(f"[ServoAgent]: I2C write error: {e}")

        # Publish state update to EventBus
        self.bus.publish(ServoTargetReachedEvent(pan=self.current_pan, tilt=self.current_tilt))

    def home(self):
        """Restores pan and tilt servos to default base positions."""
        self.set_angles(self.pan_cfg.get("base_angle", 90), self.tilt_cfg.get("base_angle", 75))
        logger.info(f"[ServoAgent]: Servos homed to Base")

    async def start(self):
        self.home()
        logger.info(f"[ServoAgent]: Servo actuator started in {self.mode.upper()} mode.")
        return True

    async def stop(self):
        self.home()
        if self.driver:
            self.driver.close()
        logger.info("[ServoAgent]: Servo actuator stopped.")
