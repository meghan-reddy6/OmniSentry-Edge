"""
Multimodal Orchestrator Agent Module

Functions as the primary State Machine Engine for the OmniSentry-Edge stack. 
Arbitrates control authority between acoustic sound-seeking mode and high-level 
VLM target tracking via the async event bus.
"""
import time
import logging
import numpy as np
from enum import Enum, auto
from src.common.bus import (
    Event, VoiceDetectedEvent, TrackCommand, OperatingMode,
    SetOperatingModeCommand, ManualJogCommand, HomeServosCommand,
    VisualTargetOffsetEvent
)

logger = logging.getLogger(__name__)

class SystemState(Enum):
    IDLE = auto()
    ACOUSTIC_SEEK = auto()
    VLM_TRACKING = auto()

class StateChangeEvent(Event):
    def __init__(self, new_state: SystemState):
        self.new_state = new_state



class MoveServoCommand(Event):
    def __init__(self, pan: float, tilt: float):
        self.pan = pan
        self.tilt = tilt

class OrchestratorAgent:
    """
    OrchestratorAgent:
    - Maintains the Global FSM: IDLE <-> ACOUSTIC_SEEK <-> VLM_TRACKING.
    - Prevents chaotic servo contention by prioritizing mode state changes.
    - Maps high-level natural language tracking commands to internal states.
    """
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config
        self.state = SystemState.IDLE
        self.current_prompt = self.config.get("orchestrator", {}).get("default_prompt", None)
        self.current_pan = 90.0
        self.current_tilt = 75.0
        
        self._last_state_change = time.time()
        self._min_state_dwell = 0.500  # 500ms debounce
        self._last_visual_detection_time = 0.0
        
        default_mode = self.config.get("system", {}).get("default_mode", "AUTONOMOUS").upper()
        self.current_mode = OperatingMode(default_mode)

        servo_cfg = self.config.get("servos", {})
        self.pan_min = float(servo_cfg.get("pan", {}).get("min_angle", 0))
        self.pan_max = float(servo_cfg.get("pan", {}).get("max_angle", 180))
        self.tilt_min = float(servo_cfg.get("tilt", {}).get("min_angle", 45))
        self.tilt_max = float(servo_cfg.get("tilt", {}).get("max_angle", 135))

        self.bus.subscribe("SoundLocalizedEvent", self.handle_sound_localized)
        self.bus.subscribe("ServoTargetReachedEvent", self.handle_servo_reached)
        self.bus.subscribe("TrackCommand", self.handle_track_command)
        self.bus.subscribe("VoiceDetectedEvent", self._handle_voice_detected)
        self.bus.subscribe("VisualTargetOffsetEvent", self._handle_visual_offset)
        self.bus.subscribe("SetOperatingModeCommand", self._handle_set_mode)
        self.bus.subscribe("ManualJogCommand", self._handle_manual_jog)

    def transition_to(self, new_state: SystemState):
        now = time.time()
        if new_state != self.state and (now - self._last_state_change) >= self._min_state_dwell:
            logger.info(f"[Orchestrator] State transition: {self.state} -> {new_state}")
            self.state = new_state
            self._last_state_change = now
            self.bus.publish(StateChangeEvent(new_state=new_state))

    def _handle_visual_offset(self, event: VisualTargetOffsetEvent):
        if self.current_mode not in (OperatingMode.AUTONOMOUS, OperatingMode.VISION_ONLY):
            return
        
        self._last_visual_detection_time = time.time()
        self.transition_to(SystemState.VLM_TRACKING)

    def handle_track_command(self, event):
        if self.current_mode not in (OperatingMode.VISION_ONLY, OperatingMode.AUTONOMOUS):
            prompt = getattr(event, 'prompt', None)
            if prompt:
                logger.warning("[Orchestrator] Vision tracking ignored: Active mode does not permit vision lock.")
            return

        prompt = getattr(event, 'prompt', None) or getattr(event, 'target', None)
        if isinstance(prompt, str) and prompt.strip():
            self.current_prompt = prompt.strip()
            self.state = SystemState.VLM_TRACKING
            logger.info(f"[Orchestrator]: State -> VLM_TRACKING for target '{self.current_prompt}'")
            self.bus.publish(StateChangeEvent(new_state=SystemState.VLM_TRACKING))

    def handle_sound_localized(self, event):
        if self.state == SystemState.IDLE:
            logger.info(f"[Orchestrator]: Sound detected at {event.angle:+.1f}°. Transition: IDLE -> ACOUSTIC_SEEK")
            self.state = SystemState.ACOUSTIC_SEEK
            self.bus.publish(StateChangeEvent(new_state=SystemState.ACOUSTIC_SEEK))
            self.bus.publish(MoveServoCommand(pan=event.angle, tilt=0.0))

    def handle_servo_reached(self, event):
        if hasattr(event, 'pan'): self.current_pan = event.pan
        if hasattr(event, 'tilt'): self.current_tilt = event.tilt
        
        if self.state == SystemState.ACOUSTIC_SEEK:
            if self.current_prompt and self.current_prompt.strip():
                logger.info(f"[Orchestrator]: Acoustic orient done. Resuming tracking for '{self.current_prompt}'")
                self.state = SystemState.VLM_TRACKING
                self.bus.publish(StateChangeEvent(new_state=SystemState.VLM_TRACKING))
            else:
                logger.info("[Orchestrator]: Acoustic orient done. Returning to IDLE.")
                self.state = SystemState.IDLE
                self.bus.publish(StateChangeEvent(new_state=SystemState.IDLE))

    def _handle_voice_detected(self, event: VoiceDetectedEvent):
        now = time.time()
        
        # In Autonomous mode: suppress audio cues if actively tracking visual target
        if self.current_mode == OperatingMode.AUTONOMOUS:
            if self.state == SystemState.VLM_TRACKING and (now - self._last_visual_detection_time < 1.0):
                logger.debug("[Orchestrator] Audio cue suppressed: active visual lock.")
                return

        if self.current_mode not in (OperatingMode.AUTONOMOUS, OperatingMode.AUDIO_ONLY):
            return

        self.transition_to(SystemState.ACOUSTIC_SEEK)
        pan_offset = -event.azimuth_deg * 0.70
        new_pan = max(self.pan_min, min(self.pan_max, self.current_pan + pan_offset))

        logger.info(f"[Orchestrator] Acoustic seek: {event.direction} -> Pan={new_pan:.1f}°")
        self.current_pan = new_pan
        self.bus.publish(MoveServoCommand(pan=int(round(new_pan)), tilt=int(round(self.current_tilt))))

    def _handle_set_mode(self, cmd: SetOperatingModeCommand):
        self.current_mode = cmd.mode
        logger.info(f"[Orchestrator] Switched active operating mode to: {self.current_mode.value}")

        if self.current_mode == OperatingMode.TERMINAL:
            # Disengage automated vision locks
            self.bus.publish(TrackCommand(prompt=""))
        elif self.current_mode == OperatingMode.STANDBY:
            self.bus.publish(TrackCommand(prompt=""))
            self.bus.publish(HomeServosCommand())
            
        # Broadcast the new mode back over the bus so WebUI can reflect it
        from src.common.bus import OperatingModeChangedEvent
        self.bus.publish(OperatingModeChangedEvent(mode=self.current_mode.value))

    def _handle_manual_jog(self, cmd: ManualJogCommand):
        """Allows direct nudge movements only when in manual terminal mode."""
        if self.current_mode != OperatingMode.TERMINAL:
            logger.warning("[Orchestrator] ManualJog ignored: Not in TERMINAL mode.")
            return

        new_pan = self.current_pan + cmd.pan_delta
        new_tilt = self.current_tilt + cmd.tilt_delta
        
        clamped_pan = float(np.clip(new_pan, self.pan_min, self.pan_max))
        clamped_tilt = float(np.clip(new_tilt, self.tilt_min, self.tilt_max))

        self.bus.publish(MoveServoCommand(pan=clamped_pan, tilt=clamped_tilt))
        # Wait for actual servo confirmation to update current_pan/tilt natively

    async def start(self):
        logger.info("[Orchestrator]: Orchestrator running.")
        return True

    async def stop(self):
        logger.info("[Orchestrator]: Orchestrator stopped.")
