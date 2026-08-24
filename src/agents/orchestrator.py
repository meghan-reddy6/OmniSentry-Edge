import logging
from enum import Enum, auto
from src.common.bus import Event

logger = logging.getLogger(__name__)

class SystemState(Enum):
    IDLE = auto()
    ACOUSTIC_SEEK = auto()
    VLM_TRACKING = auto()

class StateChangeEvent(Event):
    def __init__(self, new_state: SystemState):
        self.new_state = new_state

class TrackCommand(Event):
    def __init__(self, prompt: str):
        self.prompt = prompt

class MoveServoCommand(Event):
    def __init__(self, pan: float, tilt: float):
        self.pan = pan
        self.tilt = tilt

class OrchestratorAgent:
    def __init__(self, bus, config):
        self.bus = bus
        self.config = config
        self.state = SystemState.IDLE
        self.current_prompt = self.config.get("orchestrator", {}).get("default_prompt", None)

        self.bus.subscribe("SoundLocalizedEvent", self.handle_sound_localized)
        self.bus.subscribe("ServoTargetReachedEvent", self.handle_servo_reached)
        self.bus.subscribe("TrackCommand", self.handle_track_command)

    def handle_track_command(self, event):
        prompt = getattr(event, 'prompt', None) or getattr(event, 'target', None)
        if prompt:
            self.current_prompt = str(prompt)
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
        if self.state == SystemState.ACOUSTIC_SEEK:
            if self.current_prompt and self.current_prompt.strip():
                logger.info(f"[Orchestrator]: Acoustic orient done. Resuming tracking for '{self.current_prompt}'")
                self.state = SystemState.VLM_TRACKING
                self.bus.publish(StateChangeEvent(new_state=SystemState.VLM_TRACKING))
            else:
                logger.info("[Orchestrator]: Acoustic orient done. Returning to IDLE.")
                self.state = SystemState.IDLE
                self.bus.publish(StateChangeEvent(new_state=SystemState.IDLE))

    async def start(self):
        logger.info("[Orchestrator]: Orchestrator running.")
        return True

    async def stop(self):
        logger.info("[Orchestrator]: Orchestrator stopped.")
