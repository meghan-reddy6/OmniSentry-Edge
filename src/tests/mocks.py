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
        
        # Servo angles updated dynamically by VisionAgent
        self.current_pan = 0.0
        self.current_tilt = 0.0
        
        # Session start baselines
        self.start_pan = None
        self.start_tilt = None
        
        # Camera FOV angles
        self.fov_x = 60.0
        self.fov_y = 45.0
        
        # Simulation parameters to return valid detections
        self.simulated_face_size = 60.0
        self.simulated_object_w = 80.0
        self.simulated_object_h = 80.0
        
        # Flag to toggle detection presence
        self.face_present = True
        self.object_present = True
        self.face_count = 1

    def run(self, output_names=None, input_feed=None):
        """Simulates bounding box outputs based on feed types."""
        # Initialize start pan/tilt base lines on first frame
        if self.start_pan is None:
            self.start_pan = self.current_pan
        if self.start_tilt is None:
            self.start_tilt = self.current_tilt
            
        # Target is placed at +5.0 degrees pan and +2.0 degrees tilt from start
        target_offset_pan = 5.0
        target_offset_tilt = 2.0
        
        # Relative angle of target to camera frame
        rel_pan = target_offset_pan - (self.current_pan - self.start_pan)
        rel_tilt = target_offset_tilt - (self.current_tilt - self.start_tilt)
        
        # If target is outside the FOV, consider it not present
        in_fov = (abs(rel_pan) <= self.fov_x / 2.0) and (abs(rel_tilt) <= self.fov_y / 2.0)
        
        # Map relative angles to pixel coordinates in 640x480 frame
        x_offset = (rel_pan / (self.fov_x / 2.0)) * 320.0
        y_offset = (rel_tilt / (self.fov_y / 2.0)) * 240.0
        
        simulated_x = 320.0 + x_offset
        simulated_y = 240.0 - y_offset
        
        # Face box outputs format: [ [ [x_min, y_min, w, h, ..., score] ] ] (1, N, 15)
        if "face" in self.model_path.lower():
            if not self.face_present or self.face_count == 0 or not in_fov:
                # Reset start pan/tilt when session ends/face is lost
                self.start_pan = None
                self.start_tilt = None
                return [np.zeros((1, 0, 15), dtype=np.float32)]
            
            w = self.simulated_face_size
            h = self.simulated_face_size
            
            # YuNet output layout: [x_min, y_min, width, height, landmarks..., score]
            detection = np.zeros((1, self.face_count, 15), dtype=np.float32)
            for i in range(self.face_count):
                x_min = (simulated_x + i * 80.0) - w / 2
                y_min = simulated_y - h / 2
                score = 0.95
                
                detection[0, i, 0] = x_min
                detection[0, i, 1] = y_min
                detection[0, i, 2] = w
                detection[0, i, 3] = h
                detection[0, i, 14] = score
            return [detection]
            
        # Check if YOLO-World VLM input
        # Object box format: [ [ [x_min, y_min, w, h, score] ] ] (1, N, 5)
        else:
            if not self.object_present or not in_fov:
                # Reset start pan/tilt when session ends/object is lost
                self.start_pan = None
                self.start_tilt = None
                return [np.zeros((1, 0, 5), dtype=np.float32)]
                
            w = self.simulated_object_w
            h = self.simulated_object_h
            x = simulated_x - w / 2
            y = simulated_y - h / 2
            score = 0.85
            
            detection = np.array([[[x, y, w, h, score]]], dtype=np.float32)
            return [detection]

class MockASREngine:
    """Mocks Whisper ONNX model or speech-to-text decoder engine."""
    mock_transcription = "track book"
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        return self.mock_transcription
