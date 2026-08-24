"""
Orchestrator Agent implementation.
Manages the system state transitions and coordinate priorities.
"""
import asyncio
import logging
from src.common.bus import BaseAgent, EventBus
from src.common.config import SystemConfig
from src.common.messages import (
    Event, SystemState, SoundLocalizedEvent, MoveToCommand, MotionDoneEvent,
    VerifyFaceCommand, TargetVerifiedEvent, TargetNotFoundEvent, TrackCommand,
    MoveHomeCommand, StateChangedEvent
)

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """
    Coordinates global state transitions and coordinates priorities.
    States: IDLE, ACOUSTIC_SEEK, VISUAL_VERIFYING, VLM_TRACKING, RESETTING.
    """
    def __init__(self, bus: EventBus, config: SystemConfig):
        super().__init__("Orchestrator", bus, config)
        self.state = SystemState.IDLE
        self._timer_task = None
        self.current_prompt = None

        # Subscribe to relevant events
        self.subscribe(SoundLocalizedEvent)
        self.subscribe(MotionDoneEvent)
        self.subscribe(TargetVerifiedEvent)
        self.subscribe(TargetNotFoundEvent)
        self.subscribe(TrackCommand)
        self.subscribe(MoveHomeCommand)

    async def setup(self):
        logger.info("Orchestrator agent initializing...")
        self.state = SystemState.IDLE

    async def cleanup(self):
        self._cancel_timer()
        logger.info("Orchestrator agent cleaned up.")

    async def handle_event(self, event: Event):
        # Downgraded to debug to prevent log spam (e.g. from continuous AudioLevelEvent)
        logger.debug(f"Orchestrator received {type(event).__name__} in state {self.state.value}")

        # Check for preemption by user command (TrackCommand)
        if isinstance(event, TrackCommand):
            self.current_prompt = event.prompt
            await self._transition_to(SystemState.VLM_TRACKING)
            return

        if isinstance(event, MoveHomeCommand):
            if self.state != SystemState.IDLE and self.state != SystemState.RESETTING:
                await self._transition_to(SystemState.RESETTING)
            return

        if self.state == SystemState.IDLE:
            if isinstance(event, SoundLocalizedEvent):
                # Target localized by microphone array
                # Move pan servo to target angle, tilt remains at neutral (0.0)
                await self._transition_to(SystemState.ACOUSTIC_SEEK)
                await self.bus.publish(MoveToCommand(pan=event.angle, tilt=0.0))

        elif self.state == SystemState.ACOUSTIC_SEEK:
            if isinstance(event, MotionDoneEvent):
                # Servo has completed rotation to the sound location
                if self.current_prompt and self.current_prompt.strip():
                    logger.info(f"[Orchestrator]: Resuming tracking for '{self.current_prompt}' at new acoustic angle.")
                    await self._transition_to(SystemState.VLM_TRACKING)
                else:
                    logger.info("[Orchestrator]: Acoustic re-orientation complete. Returning to IDLE.")
                    await self._transition_to(SystemState.IDLE)

            elif isinstance(event, SoundLocalizedEvent):
                # Update seek target if a stronger sound signal arrives
                logger.info(f"Updating seek angle to {event.angle}")
                await self.bus.publish(MoveToCommand(pan=event.angle, tilt=0.0))

        elif self.state == SystemState.VISUAL_VERIFYING:
            if isinstance(event, TargetVerifiedEvent):
                self._cancel_timer()
                # Face verified at center coordinates
                await self._transition_to(SystemState.VLM_TRACKING)
                # Initiate tracking centering loop
                # The VisionAgent will continuously stream tracking updates now

            elif isinstance(event, TargetNotFoundEvent):
                self._cancel_timer()
                if event.reason == "multiple_persons":
                    logger.warning("Face verification failed: multiple persons detected. Resetting...")
                else:
                    logger.warning("Face verification failed. Resetting...")
                await self._transition_to(SystemState.RESETTING)
                await self.bus.publish(MoveHomeCommand())

        elif self.state == SystemState.VLM_TRACKING:
            if isinstance(event, SoundLocalizedEvent):
                # Ignore ambient sound cues while visually locked to prevent servo jerking
                pass
                
            elif isinstance(event, TargetNotFoundEvent):
                # Target lost in visual tracking loop
                logger.warning("Visual tracking target lost. Returning home and waiting in IDLE...")
                await self._transition_to(SystemState.IDLE)
                await self.bus.publish(MoveHomeCommand())

        elif self.state == SystemState.RESETTING:
            if isinstance(event, MotionDoneEvent):
                # Returned home successfully
                await self._transition_to(SystemState.IDLE)

    async def _transition_to(self, new_state: SystemState):
        if self.state == new_state:
            return
        old_state = self.state
        self.state = new_state
        logger.info(f"State transition: {old_state.value} -> {new_state.value}")
        await self.bus.publish(StateChangedEvent(old_state=old_state, new_state=new_state))

    def _start_timer(self):
        self._cancel_timer()
        self._timer_task = asyncio.create_task(self._verification_timer())

    def _cancel_timer(self):
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            self._timer_task = None

    async def _verification_timer(self):
        timeout = self.config.vision.get("tracking_timeout", 3.0)
        try:
            await asyncio.sleep(timeout)
            logger.warning(f"Verification timer expired after {timeout}s.")
            await self.bus.publish(TargetNotFoundEvent(reason="timeout"))
        except asyncio.CancelledError:
            pass
