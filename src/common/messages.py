"""
Message schema definitions for the RubikPi 3 Audio-Visual Directional & VLM Tracking System.
Defines strongly typed events, commands, and states.
"""
from dataclasses import dataclass
from enum import Enum

class SystemState(str, Enum):
    IDLE = "IDLE"
    ACOUSTIC_SEEK = "ACOUSTIC_SEEK"
    VISUAL_VERIFYING = "VISUAL_VERIFYING"
    VLM_TRACKING = "VLM_TRACKING"
    RESETTING = "RESETTING"

@dataclass
class Event:
    """Base class for all system messages."""
    pass

@dataclass
class SoundLocalizedEvent(Event):
    """Emitted by AudioSensingAgent when directional sound is detected."""
    angle: float       # Azimuth angle in degrees (e.g., -90 to 90)
    volume: float      # Volume in dB
    confidence: float  # VAD/energy confidence score

@dataclass
class MoveToCommand(Event):
    """Sent to ServoActuatorAgent to command movement to a specific coordinate."""
    pan: float         # Target pan angle in degrees
    tilt: float        # Target tilt angle in degrees

@dataclass
class MotionDoneEvent(Event):
    """Emitted by ServoActuatorAgent when a command movement is completed."""
    pass

@dataclass
class VerifyFaceCommand(Event):
    """Sent to VisionVLMAgent to initiate face detection verification."""
    timeout: float = 3.0

@dataclass
class TargetVerifiedEvent(Event):
    """Emitted by VisionVLMAgent when a face/target is successfully verified."""
    center_x: float    # Bounding box center x normalized to [-1.0, 1.0]
    center_y: float    # Bounding box center y normalized to [-1.0, 1.0]

@dataclass
class TargetNotFoundEvent(Event):
    """Emitted by VisionVLMAgent if verification fails or target is lost."""
    reason: str = "lost"

@dataclass
class TrackingErrorEvent(Event):
    """Emitted continuously by VisionVLMAgent during tracking loop."""
    dx: float          # Center-to-target horizontal delta in normalized coordinates [-1.0, 1.0]
    dy: float          # Center-to-target vertical delta in normalized coordinates [-1.0, 1.0]

@dataclass
class TrackCommand(Event):
    """Sent by user/orchestrator to initialize open-vocabulary object tracking."""
    prompt: str        # Text prompt, e.g., "blue mug", "red bottle"

@dataclass
class MoveHomeCommand(Event):
    """Command to return servos back to neutral (0, 0) position."""
    pass

@dataclass
class StateChangedEvent(Event):
    """Emitted by OrchestratorAgent when the system state transitions."""
    old_state: SystemState
    new_state: SystemState

@dataclass
class ServoPositionEvent(Event):
    """Emitted by ServoActuatorAgent when servo angles are updated."""
    pan: float
    tilt: float

@dataclass
class VoiceCommandEvent(Event):
    """Emitted by AudioSensingAgent when a voice command is recognized."""
    transcript: str

@dataclass
class SimulateSpeechCommand(Event):
    """Simulates voice input by injecting a transcription directly into the audio agent."""
    text: str

@dataclass
class AudioLevelEvent(Event):
    """Emitted by AudioSensingAgent to broadcast real-time volume levels."""
    rms_db: float
    noise_floor: float
