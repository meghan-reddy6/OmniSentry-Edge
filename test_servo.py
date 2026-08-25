import time
import sys

try:
    import busio
    from board import SCL, SDA
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo
except ImportError:
    print("Error: Required adafruit libraries not found.")
    print("Run: pip install adafruit-blinka adafruit-circuitpython-pca9685")
    sys.exit(1)

def test_servos():
    print("Initializing I2C bus and PCA9685 driver...")
    try:
        i2c = busio.I2C(SCL, SDA)
        pca = PCA9685(i2c, address=0x40)
        pca.frequency = 50
    except Exception as e:
        print(f"Hardware init failed: {e}")
        print("Make sure you are running on the Rubik Pi with the PCA9685 connected to the I2C bus.")
        sys.exit(1)

    # By default, Pan is on Channel 0, Tilt is on Channel 1
    pan_servo = servo.Servo(pca.channels[0])
    tilt_servo = servo.Servo(pca.channels[1])

    # 90 degrees is the center/home point for a standard 180 degree servo
    angles = [90, 45, 135, 90]

    print("Starting servo movement sweep test...")
    try:
        for angle in angles:
            print(f"Moving Pan and Tilt to {angle} degrees...")
            pan_servo.angle = angle
            tilt_servo.angle = angle
            time.sleep(1.5)
            
        print("Servo sweep test completed successfully.")
    except Exception as e:
        print(f"Servo write error: {e}")
    finally:
        # Graceful hardware teardown
        try:
            pca.deinit()
            print("PCA9685 de-initialized.")
        except:
            pass

if __name__ == "__main__":
    test_servos()
