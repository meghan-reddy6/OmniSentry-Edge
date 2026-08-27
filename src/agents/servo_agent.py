import time
import logging
from src.common.bus import MoveServoCommand, ServoTargetReachedEvent

logger = logging.getLogger(__name__)

# --- PCA9685 Registers & Constants ---
PCA9685_ADDRESS = 0x40
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

class PCA9685Direct:
    def __init__(self, bus_num=1, address=PCA9685_ADDRESS):
        import smbus2
        self.bus = smbus2.SMBus(bus_num)
        self.address = address
        self.reset()
        self.set_pwm_freq(50)

    def reset(self):
        self.bus.write_byte_data(self.address, MODE1, 0x00)
        time.sleep(0.01)

    def set_pwm_freq(self, freq_hz):
        prescale_val = 25000000.0 / (4096.0 * float(freq_hz)) - 1.0
        prescale = int(prescale_val + 0.5)

        old_mode = self.bus.read_byte_data(self.address, MODE1)
        new_mode = (old_mode & 0x7F) | 0x10  # Enter SLEEP mode
        self.bus.write_byte_data(self.address, MODE1, new_mode)
        self.bus.write_byte_data(self.address, PRESCALE, prescale)
        self.bus.write_byte_data(self.address, MODE1, old_mode)
        time.sleep(0.005)
        # Enable auto-increment (0xA1) for burst register writes
        self.bus.write_byte_data(self.address, MODE1, old_mode | 0xA1)

    def set_pwm(self, channel, on, off):
        reg = LED0_ON_L + 4 * channel
        data = [on & 0xFF, (on >> 8) & 0xFF, off & 0xFF, (off >> 8) & 0xFF]
        self.bus.write_i2c_block_data(self.address, reg, data)

    def set_servo_angle(self, channel, angle_deg, min_us=500, max_us=2400):
        """Map angle (0-180 deg) to 12-bit PCA9685 counter ticks at 50Hz (20ms period)."""
        clamped_angle = max(0.0, min(180.0, float(angle_deg)))
        pulse_us = min_us + (clamped_angle / 180.0) * (max_us - min_us)
        ticks = int(pulse_us * 4096.0 / 20000.0)
        self.set_pwm(channel, 0, ticks)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass


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

    def set_angles(self, pan_angle: float, tilt_angle: float):
        # Strict boundary clamping
        clamped_pan = max(self.pan_min, min(self.pan_max, float(pan_angle)))
        clamped_tilt = max(self.tilt_min, min(self.tilt_max, float(tilt_angle)))

        self.current_pan = clamped_pan
        self.current_tilt = clamped_tilt

        if self.mode == "hardware" and self.driver:
            try:
                self.driver.set_servo_angle(self.pan_channel, self.current_pan)
                self.driver.set_servo_angle(self.tilt_channel, self.current_tilt)
            except Exception as e:
                logger.error(f"[ServoAgent]: I2C write error: {e}")

        # Publish state update to EventBus for HUD sync
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
