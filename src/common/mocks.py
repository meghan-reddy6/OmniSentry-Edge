"""
High-fidelity mocks for PyAudio, PCA9685/SMBus2, and ONNX Runtime.
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
            sig1 = np.sin(2.0 * np.pi * self.tone_frequency * t)
            sig2 = np.sin(2.0 * np.pi * self.tone_frequency * (t - tdoa))
            amplitude = 0.3
        else:
            std_dev = 0.0025 + 0.0015 * math.sin(time.time() * 2.0)
            sig1 = np.random.normal(0, std_dev, num_frames)
            sig2 = np.random.normal(0, std_dev, num_frames)
            amplitude = 1.0

        chan1 = np.clip(sig1 * amplitude, -1.0, 1.0)
        chan2 = np.clip(sig2 * amplitude, -1.0, 1.0)
        
        chan1_int = (chan1 * 32767).astype(np.int16)
        chan2_int = (chan2 * 32767).astype(np.int16)
        
        interleaved = np.empty(num_frames * 2, dtype=np.int16)
        interleaved[0::2] = chan1_int
        interleaved[1::2] = chan2_int
        
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

class MockNode:
    def __init__(self, name, shape, type="tensor(uint8)"):
        self.name = name
        self.shape = shape
        self.type = type

class MockInferenceSession:
    """Mocks ONNX Runtime InferenceSession for Face Detection and YOLO Detector."""
    def get_inputs(self):
        return [MockNode("images", [1, 3, 640, 640])]
        
    def get_outputs(self):
        return [MockNode("output0", [1, 84, 8400])]

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.input_names = ["input", "image", "prompt"]
        self.output_names = ["output"]
        
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.start_pan = None
        self.start_tilt = None
        
        self.fov_x = 60.0
        self.fov_y = 45.0
        
        self.simulated_face_size = 60.0
        self.simulated_object_w = 80.0
        self.simulated_object_h = 80.0
        
        self.face_present = True
        self.object_present = True
        self.face_count = 1

    def run(self, output_names=None, input_feed=None):
        """Simulates bounding box outputs based on feed types."""
        if self.start_pan is None:
            self.start_pan = self.current_pan
        if self.start_tilt is None:
            self.start_tilt = self.current_tilt
            
        target_offset_pan = 5.0
        target_offset_tilt = 2.0
        
        rel_pan = target_offset_pan - (self.current_pan - self.start_pan)
        rel_tilt = target_offset_tilt - (self.current_tilt - self.start_tilt)
        
        in_fov = (abs(rel_pan) <= self.fov_x / 2.0) and (abs(rel_tilt) <= self.fov_y / 2.0)
        
        x_offset = (rel_pan / (self.fov_x / 2.0)) * 320.0
        y_offset = (rel_tilt / (self.fov_y / 2.0)) * 240.0
        
        simulated_x = 320.0 + x_offset
        simulated_y = 240.0 - y_offset
        
        if "face" in self.model_path.lower() or "mediapipe" in self.model_path.lower():
            if not self.face_present or self.face_count == 0 or not in_fov:
                self.start_pan = None
                self.start_tilt = None
                # Support both shape output formats
                return [np.zeros((1, 0, 15), dtype=np.float32), np.zeros((1, 0, 16), dtype=np.float32), np.zeros((1, 0, 1), dtype=np.float32), np.zeros((1, 0, 1), dtype=np.float32)]
            
            w = self.simulated_face_size
            h = self.simulated_face_size
            
            # MediaPipe SSD outputs
            box_coords = np.zeros((1, 512, 16), dtype=np.float32)
            box_scores = np.zeros((1, 512, 1), dtype=np.float32)
            
            # Put the best face detection in index 0
            # coords mapping: [y_min, x_min, y_max, x_max, ...] relative to 256x256 image size
            box_coords[0, 0, 0] = (simulated_y - h/2) / 480.0 * 256.0
            box_coords[0, 0, 1] = (simulated_x - w/2) / 640.0 * 256.0
            box_coords[0, 0, 2] = (simulated_y + h/2) / 480.0 * 256.0
            box_coords[0, 0, 3] = (simulated_x + w/2) / 640.0 * 256.0
            box_scores[0, 0, 0] = 0.95
            
            return [box_coords, box_coords, box_scores, box_scores]
            
        else:
            # YOLOv8 outputs
            # boxes shape: [1, 8400, 4]
            # scores shape: [1, 8400]
            # class_idx shape: [1, 8400]
            if not self.object_present or not in_fov:
                self.start_pan = None
                self.start_tilt = None
                return [np.zeros((1, 8400, 4), dtype=np.float32), np.zeros((1, 8400), dtype=np.float32), np.zeros((1, 8400), dtype=np.float32)]
                
            w = self.simulated_object_w
            h = self.simulated_object_h
            cx = simulated_x
            cy = simulated_y
            
            boxes = np.zeros((1, 8400, 4), dtype=np.float32)
            scores = np.zeros((1, 8400), dtype=np.float32)
            class_idx = np.zeros((1, 8400), dtype=np.float32)
            
            # Put detected target class at index 0 (e.g. cup)
            boxes[0, 0] = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
            scores[0, 0] = 0.85
            class_idx[0, 0] = 41 # index 41 is cup
            
            return [boxes, scores, class_idx]

class MockASREngine:
    """Mocks Whisper ONNX model or speech-to-text decoder engine."""
    mock_transcription = "track book"
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        return self.mock_transcription
