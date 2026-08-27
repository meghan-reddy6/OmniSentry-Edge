"""
Servo Actuator Agent Module

Manages physical 2-DOF Pan/Tilt gimbal actuation over the I2C bus using the PCA9685 driver.
Integrates clamping and mechanical limit translation. Evaluates MoveServoCommand events 
broadcasted from the visual-PID loop or acoustic TDoA angle approximations.
"""
import logging
import threading
from src.common.bus import MoveServoCommand, ServoTargetReachedEvent

logger = logging.getLogger(__name__)

class ServoActuatorAgent:
    """
    ServoActuatorAgent:
    - Maintains the kinematic state machine of the robot's physical head (pan, tilt).
    - Interfaces with Adafruit Blinka via SMBus/I2C.
    - Gracefully falls back to simulation mode if executed off-target (e.g. PC/Windows).
    - Broadcasts ServoTargetReachedEvent to keep the UI telemetry in lockstep.
    """
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config
        servo_cfg = self.config.get("servos", {})

        # Mode: 'hardware' or 'simulation'
        self.mode = servo_cfg.get("mode", "hardware").lower()
        self.i2c_bus_num = servo_cfg.get("i2c_bus", 1)
        self.i2c_address = servo_cfg.get("i2c_address", 0x40)
        self.pwm_frequency = servo_cfg.get("pwm_frequency", 50)

        # Pan Channel 0 (0° to 180°, base: 90°)
        pan_cfg = servo_cfg.get("pan", {})
        self.pan_channel = pan_cfg.get("channel", 0)
        self.pan_min = pan_cfg.get("min_angle", 0)
        self.pan_max = pan_cfg.get("max_angle", 180)
        self.pan_base = pan_cfg.get("base_angle", 90)

        # Tilt Channel 1 (45° to 135°, base: 70°)
        tilt_cfg = servo_cfg.get("tilt", {})
        self.tilt_channel = tilt_cfg.get("channel", 1)
        self.tilt_min = tilt_cfg.get("min_angle", 45)
        self.tilt_max = tilt_cfg.get("max_angle", 135)
        self.tilt_base = tilt_cfg.get("base_angle", 70)

        # Initial live angles set to base
        self.current_pan = float(self.pan_base)
        self.current_tilt = float(self.tilt_base)

        self.pca = None
        self._servos = {}
        self._init_hardware()

        self.bus.subscribe("MoveServoCommand", self.handle_move_command)

    def _init_hardware(self):
        if self.mode == "simulation":
            logger.info("[ServoAgent]: Mode set to SIMULATION. Real I2C writes bypassed.")
            return

        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685
            from adafruit_motor import servo

            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c, address=self.i2c_address)
            self.pca.frequency = self.pwm_frequency

            self._servos[self.pan_channel] = servo.Servo(
                self.pca.channels[self.pan_channel], min_pulse=500, max_pulse=2500
            )
            self._servos[self.tilt_channel] = servo.Servo(
                self.pca.channels[self.tilt_channel], min_pulse=500, max_pulse=2500
            )

            # Move to default base position on hardware boot
            self.set_angles(self.pan_base, self.tilt_base)
            logger.info(f"[ServoAgent]: PCA9685 Hardware active on I2C 0x{self.i2c_address:02X} (Pan Ch:{self.pan_channel}, Tilt Ch:{self.tilt_channel})")
        except Exception as e:
            logger.warning(f"[ServoAgent]: Hardware initialization failed ({e}). Auto-falling back to SIMULATION.")
            self.mode = "simulation"

    def handle_move_command(self, event):
        target_pan = getattr(event, "pan", self.current_pan)
        target_tilt = getattr(event, "tilt", self.current_tilt)
        self.set_angles(target_pan, target_tilt)

    def set_angles(self, pan_angle: float, tilt_angle: float):
        # Strict boundary clamping
        clamped_pan = max(self.pan_min, min(self.pan_max, float(pan_angle)))
        clamped_tilt = max(self.tilt_min, min(self.tilt_max, float(tilt_angle)))

        self.current_pan = clamped_pan
        self.current_tilt = clamped_tilt

        if self.mode == "hardware" and self.pca:
            try:
                self._servos[self.pan_channel].angle = clamped_pan
                self._servos[self.tilt_channel].angle = clamped_tilt
            except Exception as e:
                logger.error(f"[ServoAgent]: I2C write error: {e}")

        # Broadcast state update to Vision HUD and Orchestrator
        self.bus.publish(ServoTargetReachedEvent(pan=self.current_pan, tilt=self.current_tilt))

    def home(self):
        """Restores pan and tilt servos to neutral base positions."""
        self.set_angles(self.pan_base, self.tilt_base)
        logger.info(f"[ServoAgent]: Servos homed to Base (Pan: {self.pan_base}°, Tilt: {self.tilt_base}°)")

    async def start(self):
        self.home()
        logger.info(f"[ServoAgent]: Servo actuator agent started (Mode: {self.mode.upper()}).")
        return True

    async def stop(self):
        self.home()
        if self.pca:
            try:
                self.pca.deinit()
            except Exception:
                pass
        logger.info("[ServoAgent]: Servo actuator stopped.")
