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

        # Pan Configuration (Channel 0: 0°..180°, base: 90°)
        pan_cfg = servo_cfg.get("pan", {})
        self.pan_channel = pan_cfg.get("channel", 0)
        self.pan_min = pan_cfg.get("min_angle", 0)
        self.pan_max = pan_cfg.get("max_angle", 180)
        self.pan_base = pan_cfg.get("base_angle", 90)

        # Tilt Configuration (Channel 1: 45°..135°, base: 70°)
        tilt_cfg = servo_cfg.get("tilt", {})
        self.tilt_channel = tilt_cfg.get("channel", 1)
        self.tilt_min = tilt_cfg.get("min_angle", 45)
        self.tilt_max = tilt_cfg.get("max_angle", 135)
        self.tilt_base = tilt_cfg.get("base_angle", 70)

        # Initial live angles
        self.current_pan = float(self.pan_base)
        self.current_tilt = float(self.tilt_base)

        self.driver = None
        self._init_hardware()

        self.bus.subscribe("MoveServoCommand", self.handle_move_command)

    def _init_hardware(self):
        if self.mode == "simulation":
            logger.info("[ServoAgent]: Mode set to SIMULATION. Hardware I2C disabled.")
            return

        try:
            self.driver = PCA9685Direct(bus_num=self.bus_num, address=self.address)
            self.set_angles(self.pan_base, self.tilt_base)
            logger.info(f"[ServoAgent]: PCA9685 hardware active on /dev/i2c-{self.bus_num} (Address: 0x{self.address:02X})")
        except Exception as e:
            logger.warning(f"[ServoAgent]: smbus2 hardware init failed: {e}. Falling back to SIMULATION.")
            self.mode = "simulation"

    def handle_move_command(self, event):
        target_pan = getattr(event, "pan", self.current_pan)
        target_tilt = getattr(event, "tilt", self.current_tilt)
        self.set_angles(target_pan, target_tilt)

    def set_angles(self, pan_angle, tilt_angle):
        # 1. Round to strict integer degrees
        int_pan = int(round(float(pan_angle)))
        int_tilt = int(round(float(tilt_angle)))

        # 2. Strict boundary clamping
        clamped_pan = max(self.pan_min, min(self.pan_max, int_pan))
        clamped_tilt = max(self.tilt_min, min(self.tilt_max, int_tilt))

        # Only execute I2C write if angle has changed by at least 1 full degree
        if clamped_pan != self.current_pan or clamped_tilt != self.current_tilt:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[SERVO ] Write -> Pan:{clamped_pan:3d}° Tilt:{clamped_tilt:3d}° (was P:{int(self.current_pan):3d}° T:{int(self.current_tilt):3d}°)")
            else:
                logger.info(f"[ServoAgent] Hardware Write -> Pan: {clamped_pan}°, Tilt: {clamped_tilt}° (was {self.current_pan}°, {self.current_tilt}°)")
            self.current_pan = clamped_pan
            self.current_tilt = clamped_tilt

            if self.mode == "hardware" and self.driver:
                try:
                    # Explicit bounds clamping via hardware driver
                    self.driver.set_servo_angle(self.pan_channel, clamped_pan, 
                                          min_angle=self.pan_min, max_angle=self.pan_max)
                    self.driver.set_servo_angle(self.tilt_channel, clamped_tilt, 
                                          min_angle=self.tilt_min, max_angle=self.tilt_max)
                except Exception as e:
                    logger.error(f"[ServoAgent]: I2C write error: {e}")

            # Publish integer state update to EventBus
            self.bus.publish(ServoTargetReachedEvent(pan=self.current_pan, tilt=self.current_tilt))

    def home(self):
        """Restores pan and tilt servos to default base positions."""
        self.set_angles(self.pan_base, self.tilt_base)
        logger.info(f"[ServoAgent]: Servos homed to Base (Pan: {self.pan_base}°, Tilt: {self.tilt_base}°)")

    async def start(self):
        self.home()
        logger.info(f"[ServoAgent]: Servo actuator started in {self.mode.upper()} mode.")
        return True

    async def stop(self):
        self.home()
        if self.driver:
            self.driver.close()
        logger.info("[ServoAgent]: Servo actuator stopped.")
