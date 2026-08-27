import time
import sys

try:
    import smbus2
except ImportError:
    print("Error: Required smbus2 library not found.")
    print("Run: pip install smbus2")
    sys.exit(1)

# --- PCA9685 Registers & Constants ---
PCA9685_ADDRESS = 0x40
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

class PCA9685Direct:
    def __init__(self, bus_num=1, address=PCA9685_ADDRESS):
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


def test_servos():
    print("Initializing I2C bus and PCA9685 driver via smbus2...")
    try:
        driver = PCA9685Direct(bus_num=1, address=0x40)
    except Exception as e:
        print(f"Hardware init failed: {e}")
        print("Make sure you are running on the Rubik Pi with the PCA9685 connected to /dev/i2c-1.")
        sys.exit(1)

    # By default, Pan is on Channel 0, Tilt is on Channel 1
    # 90 degrees is the center/home point for a standard 180 degree servo
    angles = [90, 45, 135, 90]

    print("Starting servo movement sweep test...")
    try:
        for angle in angles:
            print(f"Moving Pan (Ch 0) and Tilt (Ch 1) to {angle} degrees...")
            driver.set_servo_angle(0, angle)
            driver.set_servo_angle(1, angle)
            time.sleep(1.5)
            
        print("Servo sweep test completed successfully.")
    except Exception as e:
        print(f"Servo write error: {e}")
    finally:
        # Graceful hardware teardown
        try:
            driver.close()
            print("PCA9685 de-initialized.")
        except:
            pass

if __name__ == "__main__":
    test_servos()
