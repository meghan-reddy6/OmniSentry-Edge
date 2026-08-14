"""
High-fidelity mocks for PyAudio, PCA9685/SMBus2, and ONNX Runtime QNN EP.
Allows tests and simulation mode to run on standard computers without actual hardware.
"""
import numpy as np
import time
import math

class MockChannel:
    """Mocks a single PWM channel on PCA9685."""
    def __init__(self, index):
        self.index = index
        self._duty_cycle = 0

    @property
    def duty_cycle(self) -> int:
        return self._duty_cycle

    @duty_cycle.setter
    def duty_cycle(self, val: int):
        self._duty_cycle = val

class MockPCA9685:
    """Mocks the Adafruit PCA9685 driver interface."""
    def __init__(self, i2c_bus=None, address=0x40):
        self.address = address
        self.channels = [MockChannel(i) for i in range(16)]
        self.frequency = 50

    def set_pwm(self, channel: int, on: int, off: int):
        # Fallback raw direct registers writer support
        self.channels[channel].duty_cycle = off * 16

class MockStream:
    """Mocks the PyAudio input streaming channel."""
    def __init__(self, rate, channels, chunk_size):
        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.is_active = True
        self.start_time = time.time()
        self.sound_angle = 45.0      # Simulated sound Direction of Arrival angle
        self.sound_active = False    # Whether a loud noise is currently playing
        self.tone_frequency = 500.0  # Hz

    def read(self, num_frames, exception_on_overflow=False) -> bytes:
        """Generates raw multi-channel PCM bytes with phase shift (representing DOA)."""
        t = np.arange(num_frames) / self.rate
        
        # Calculate time delay of arrival (TDOA) in seconds
        # Using mic distance d=0.08m and speed of sound c=343m/s
        d = 0.08
        c = 343.0
        angle_rad = math.radians(self.sound_angle)
        tdoa = (d * math.sin(angle_rad)) / c
        
        # Generate signal base: pure tone if sound active, else quiet white noise
        if self.sound_active:
            # Mic 1 signal
            sig1 = np.sin(2.0 * np.pi * self.tone_frequency * t)
            # Mic 2 signal (delayed/advanced based on TDOA)
            sig2 = np.sin(2.0 * np.pi * self.tone_frequency * (t - tdoa))
            
            # Amplitude multiplier to exceed VAD threshold (e.g., -15 dB)
            amplitude = 0.3
        else:
            # Low-amplitude white noise (below -45 dB threshold)
            sig1 = np.random.normal(0, 0.002, num_frames)
            sig2 = np.random.normal(0, 0.002, num_frames)
            amplitude = 1.0

        # Apply amplitude scaling and clamp to legal float boundaries
        chan1 = np.clip(sig1 * amplitude, -1.0, 1.0)
        chan2 = np.clip(sig2 * amplitude, -1.0, 1.0)
        
        # Convert back to 16-bit PCM integer values
        chan1_int = (chan1 * 32767).astype(np.int16)
        chan2_int = (chan2 * 32767).astype(np.int16)
        
        # Interleave channels: [L0, R0, L1, R1, ...]
        interleaved = np.empty(num_frames * 2, dtype=np.int16)
        interleaved[0::2] = chan1_int
        interleaved[1::2] = chan2_int
        
        # Pace generation to match real-time sample rates
        time.sleep(num_frames / self.rate)
        
        return interleaved.tobytes()

    def stop_stream(self):
        self.is_active = False

    def close(self):
        pass

class MockPyAudio:
    """Mocks the main PyAudio context."""
    def __init__(self):
        self._streams = []

    def open(self, *args, **kwargs) -> MockStream:
        rate = kwargs.get("rate", 16000)
        channels = kwargs.get("channels", 2)
        chunk_size = kwargs.get("frames_per_buffer", 1024)
        
        stream = MockStream(rate, channels, chunk_size)
        self._streams.append(stream)
        return stream

    def terminate(self):
        pass

class MockInferenceSession:
    """Mocks ONNX Runtime InferenceSession for Face Detection and YOLO-World VLM."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.input_names = ["input", "image", "prompt"]
        self.output_names = ["output"]
        
        # Simulation parameters to return valid detections
        self.simulated_face_x = 350.0  # Slightly offset from center (320, 240)
        self.simulated_face_y = 220.0
        self.simulated_face_size = 60.0
        
        self.simulated_object_x = 400.0
        self.simulated_object_y = 300.0
        self.simulated_object_w = 80.0
        self.simulated_object_h = 80.0
        
        # Flag to toggle detection presence
        self.face_present = True
        self.object_present = True

    def run(self, output_names=None, input_feed=None):
        """Simulates bounding box outputs based on feed types."""
        # Check if Face Detection input (YuNet)
        # Face box outputs format: [ [ [x_min, y_min, w, h, ..., score] ] ] (1, N, 15)
        if "face" in self.model_path.lower():
            if not self.face_present:
                return [np.zeros((1, 0, 15), dtype=np.float32)]
            
            w = self.simulated_face_size
            h = self.simulated_face_size
            x_min = self.simulated_face_x - w / 2
            y_min = self.simulated_face_y - h / 2
            score = 0.95
            
            # YuNet output layout: [x_min, y_min, width, height, landmarks..., score]
            detection = np.zeros((1, 1, 15), dtype=np.float32)
            detection[0, 0, 0] = x_min
            detection[0, 0, 1] = y_min
            detection[0, 0, 2] = w
            detection[0, 0, 3] = h
            detection[0, 0, 14] = score
            return [detection]
            
        # Check if YOLO-World VLM input
        # Object box format: [ [ [x_min, y_min, w, h, score] ] ] (1, N, 5)
        else:
            if not self.object_present:
                return [np.zeros((1, 0, 5), dtype=np.float32)]
                
            w = self.simulated_object_w
            h = self.simulated_object_h
            x = self.simulated_object_x - w / 2
            y = self.simulated_object_y - h / 2
            score = 0.85
            
            detection = np.array([[[x, y, w, h, score]]], dtype=np.float32)
            return [detection]
