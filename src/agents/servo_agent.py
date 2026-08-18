"""
Servo Actuator Agent.
Interfaces with the PCA9685 servo driver board over I2C to control 2-DOF Pan/Tilt servos,
and executes PID feedback control loops for tracking.
"""
import asyncio
import logging
from src.common.bus import BaseAgent, EventBus
from src.common.config import SystemConfig
from src.common.messages import (
    Event, MoveToCommand, MotionDoneEvent, TrackingErrorEvent, MoveHomeCommand,
    ServoPositionEvent
)
from src.utils.pid import PIDController

logger = logging.getLogger(__name__)

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

class SMBusPCA9685:
    """Lightweight, direct smbus2 controller for PCA9685."""
    def __init__(self, bus_num, address=0x40):
        import smbus2
        import time
        self.bus = smbus2.SMBus(bus_num)
        self.address = address
        
        # Mode registers
        MODE1 = 0x00
        PRESCALE = 0xFE
        
        # 1. Set Sleep mode (bit 4 = 1) to write prescale
        self.bus.write_byte_data(self.address, MODE1, 0x10)
        time.sleep(0.005)
        # 2. Set prescale to 0x79 (50Hz PWM frequency)
        self.bus.write_byte_data(self.address, PRESCALE, 0x79)
        time.sleep(0.005)
        # 3. Clear Sleep (bit 4 = 0) and enable Auto-increment (bit 5 = 1)
        self.bus.write_byte_data(self.address, MODE1, 0x20)
        time.sleep(0.005)
        
    def set_pwm(self, channel, on_tick, off_tick):
        # ON register start: 0x06 + 4 * channel
        base = 0x06 + 4 * channel
        self.bus.write_byte_data(self.address, base, on_tick & 0xFF)
        self.bus.write_byte_data(self.address, base + 1, (on_tick >> 8) & 0xFF)
        self.bus.write_byte_data(self.address, base + 2, off_tick & 0xFF)
        self.bus.write_byte_data(self.address, base + 3, (off_tick >> 8) & 0xFF)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass

class ServoActuatorAgent(BaseAgent):
    """
    Controls Pan/Tilt servo positions based on commands and tracking feedback.
    Applies safety limits and manages PID loop steps.
    """
    def __init__(self, bus: EventBus, config: SystemConfig):
        super().__init__("ServoActuator", bus, config)
        
        # Load configuration parameters
        servo_cfg = self.config.servo
        self.pan_min = servo_cfg.get("pan_min_angle", -90.0)
        self.pan_max = servo_cfg.get("pan_max_angle", 90.0)
        self.tilt_min = servo_cfg.get("tilt_min_angle", -30.0)
        self.tilt_max = servo_cfg.get("tilt_max_angle", 45.0)
        
        self.pan_channel = servo_cfg.get("pan_channel", 0)
        self.tilt_channel = servo_cfg.get("tilt_channel", 1)
        
        # Current physical angles (start at home)
        self.current_pan = servo_cfg.get("home_pan", 0.0)
        self.current_tilt = servo_cfg.get("home_tilt", 0.0)
        
        # Initialize PID controllers
        pan_pid_cfg = servo_cfg.get("pan_pid", {"kp": 0.05, "ki": 0.005, "kd": 0.001})
        tilt_pid_cfg = servo_cfg.get("tilt_pid", {"kp": 0.05, "ki": 0.005, "kd": 0.001})
        
        self.pan_pid = PIDController(
            kp=pan_pid_cfg.get("kp", 0.05),
            ki=pan_pid_cfg.get("ki", 0.005),
            kd=pan_pid_cfg.get("kd", 0.001)
        )
        self.tilt_pid = PIDController(
            kp=tilt_pid_cfg.get("kp", 0.05),
            ki=tilt_pid_cfg.get("ki", 0.005),
            kd=tilt_pid_cfg.get("kd", 0.001)
        )
        
        # PCA9685 driver reference
        self.pca = None
        self._move_task = None
        
        # Subscriptions
        self.subscribe(MoveToCommand)
        self.subscribe(MoveHomeCommand)
        self.subscribe(TrackingErrorEvent)

    async def setup(self):
        logger.info("Setting up ServoActuatorAgent...")
        self._init_pca9685()
        self._write_servos()  # Initialize to home position

    async def cleanup(self):
        self._cancel_move()
        logger.info("ServoActuatorAgent cleaned up.")

    async def handle_event(self, event: Event):
        if isinstance(event, MoveToCommand):
            logger.info(f"Commanded to move to Pan: {event.pan:.1f}, Tilt: {event.tilt:.1f}")
            self.pan_pid.reset()
            self.tilt_pid.reset()
            self._start_move(event.pan, event.tilt)
            
        elif isinstance(event, MoveHomeCommand):
            logger.info("Commanded to move to home position.")
            self.pan_pid.reset()
            self.tilt_pid.reset()
            home_pan = self.config.servo.get("home_pan", 0.0)
            home_tilt = self.config.servo.get("home_tilt", 0.0)
            self._start_move(home_pan, home_tilt)
            
        elif isinstance(event, TrackingErrorEvent):
            # PID active visual tracking step
            self._cancel_move()  # Interrupt any pending absolute trajectory movements
            
            # Compute adjustment steps (note: error direction maps to output adjustment)
            # Center-to-target error is given. If target is to the right (dx > 0),
            # we need to rotate pan servo in the positive direction (or vice versa).
            # We assume positive pan turns right, positive tilt turns up.
            pan_step = self.pan_pid.update(event.dx)
            tilt_step = self.tilt_pid.update(event.dy)
            
            # Update angles by steps
            new_pan = clamp(self.current_pan + pan_step, self.pan_min, self.pan_max)
            new_tilt = clamp(self.current_tilt - tilt_step, self.tilt_min, self.tilt_max) # Negative tilt correction for OpenCV y-axis
            
            logger.debug(
                f"Tracking update. dx: {event.dx:+.2f}, dy: {event.dy:+.2f} | "
                f"pan_step: {pan_step:+.2f}, tilt_step: {tilt_step:+.2f} | "
                f"pan: {new_pan:.1f}, tilt: {new_tilt:.1f}"
            )
            
            self.current_pan = new_pan
            self.current_tilt = new_tilt
            self._write_servos()

    def _init_pca9685(self):
        """Initializes the PCA9685 board or falls back to SMBus2 driver, then mock driver."""
        servo_cfg = self.config.servo
        bus_num = servo_cfg.get("i2c_bus", 1)
        address = servo_cfg.get("pca9685_address", 0x40)
        
        use_mock = self.config.simulation_mode
        self._smbus_mode = False
        
        if not use_mock:
            # Try adafruit-circuitpython-pca9685 first
            try:
                import busio
                import board
                from adafruit_pca9685 import PCA9685
                i2c = busio.I2C(board.SCL, board.SDA)
                self.pca = PCA9685(i2c, address=address)
                self.pca.frequency = 50  # Servos usually run at 50Hz
                logger.info("PCA9685 hardware driver initialized at I2C address 0x%02X", address)
                return
            except Exception as e:
                logger.warning(
                    f"Adafruit PCA9685 initialization failed: {e}. "
                    "Attempting SMBus2 driver fallback..."
                )
            
            # Try SMBus2 fallback driver
            try:
                self.pca = SMBusPCA9685(bus_num, address)
                self._smbus_mode = True
                logger.info(f"PCA9685 direct SMBus2 driver initialized on I2C bus {bus_num} at address 0x{address:02X}")
                return
            except Exception as ex:
                logger.warning(
                    f"SMBus2 PCA9685 initialization failed: {ex}. "
                    "Falling back to mock driver."
                )
                use_mock = True

        if use_mock:
            from src.tests.mocks import MockPCA9685
            self.pca = MockPCA9685(bus_num, address)
            logger.info("PCA9685 initialized in SIMULATION mode.")

    def _write_servos(self):
        """Maps physical degrees to duty cycles and writes to PCA9685 channels."""
        if not self.pca:
            return
            
        # Map pan/tilt angles to ticks (12-bit range: 150 to 600 corresponding to -90 to 90 degrees)
        # Pan calibration
        pan_tick = self._angle_to_ticks(self.current_pan, self.pan_min, self.pan_max)
        # Tilt calibration
        tilt_tick = self._angle_to_ticks(self.current_tilt, self.tilt_min, self.tilt_max)
        
        # If using Adafruit library, it expects 16-bit duty cycles (0 to 65535)
        # Mock class handles ticks or 16-bit values. We write 16-bit values for consistency.
        pan_duty = int(pan_tick * 16)
        tilt_duty = int(tilt_tick * 16)
        
        try:
            if getattr(self, "_smbus_mode", False):
                # Direct SMBus2 driver writes
                self.pca.set_pwm(self.pan_channel, 0, pan_tick)
                self.pca.set_pwm(self.tilt_channel, 0, tilt_tick)
            elif hasattr(self.pca, "channels"):
                # Adafruit CircuitPython driver writes
                self.pca.channels[self.pan_channel].duty_cycle = pan_duty
                self.pca.channels[self.tilt_channel].duty_cycle = tilt_duty
            else:
                # Direct write function for custom mocks
                self.pca.set_pwm(self.pan_channel, 0, pan_tick)
                self.pca.set_pwm(self.tilt_channel, 0, tilt_tick)
                
            # Publish current position to the event bus for telemetry/HUD display
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.bus.publish(ServoPositionEvent(pan=self.current_pan, tilt=self.current_tilt)))
            except RuntimeError:
                pass
        except Exception as e:
            logger.error(f"Error writing to PCA9685: {e}")

    def _angle_to_ticks(self, angle, min_angle, max_angle) -> int:
        """Helper to map angle in degrees to 12-bit PCA9685 ticks (150 to 600)."""
        # Standard proportional mapping where center 0.0 degrees maps to 375 ticks
        # and physical servo range is 180 degrees (from -90 to 90 mapping to 150 to 600 ticks).
        # This translates to a direct mapping of (600 - 150) / 180 = 2.5 ticks per degree.
        min_ticks = 150
        max_ticks = 600
        ticks = 375 + angle * 2.5
        return int(clamp(ticks, min_ticks, max_ticks))

    def _start_move(self, target_pan, target_tilt):
        """Starts an async task to simulate physical servo motion transit latency."""
        self._cancel_move()
        self._move_task = asyncio.create_task(self._animate_move(target_pan, target_tilt))

    def _cancel_move(self):
        if self._move_task and not self._move_task.done():
            self._move_task.cancel()
            self._move_task = None

    async def _animate_move(self, target_pan, target_tilt):
        """Linearly interpolates angles over time to simulate servo speed limitations."""
        target_pan = clamp(target_pan, self.pan_min, self.pan_max)
        target_tilt = clamp(target_tilt, self.tilt_min, self.tilt_max)
        
        # Max velocity: 180 degrees per second (0.18 deg per ms)
        deg_per_step = 3.6  # degrees covered in 20ms step
        
        try:
            while True:
                dp = target_pan - self.current_pan
                dt_err = target_tilt - self.current_tilt
                
                # Check if we are close enough to the target
                if abs(dp) < 0.1 and abs(dt_err) < 0.1:
                    break
                    
                # Increment angles towards target
                self.current_pan += clamp(dp, -deg_per_step, deg_per_step)
                self.current_tilt += clamp(dt_err, -deg_per_step, deg_per_step)
                
                self._write_servos()
                await asyncio.sleep(0.02)  # 50Hz update rate
                
            self.current_pan = target_pan
            self.current_tilt = target_tilt
            self._write_servos()
            
            logger.info(f"Servo reached target: Pan={self.current_pan:.1f}, Tilt={self.current_tilt:.1f}")
            await self.bus.publish(MotionDoneEvent())
            
        except asyncio.CancelledError:
            # Movement interrupted by tracking or new command
            pass
