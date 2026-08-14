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

        # Subscribe to relevant events
        self.subscribe(SoundLocalizedEvent)
        self.subscribe(MotionDoneEvent)
        self.subscribe(TargetVerifiedEvent)
        self.subscribe(TargetNotFoundEvent)
        self.subscribe(TrackCommand)

    async def setup(self):
        logger.info("Orchestrator agent initializing...")
        self.state = SystemState.IDLE

    async def cleanup(self):
        self._cancel_timer()
        logger.info("Orchestrator agent cleaned up.")

    async def handle_event(self, event: Event):
        logger.info(f"Orchestrator received {type(event).__name__} in state {self.state.value}")

        # Check for preemption by user command (TrackCommand)
        if isinstance(event, TrackCommand):
            await self._transition_to(SystemState.VLM_TRACKING)
            # The VisionAgent will handle initializing model tracking for the given prompt
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
                await self._transition_to(SystemState.VISUAL_VERIFYING)
                await self.bus.publish(VerifyFaceCommand(timeout=self.config.vision.get("tracking_timeout", 3.0)))
                self._start_timer()

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
                logger.warning("Face verification failed. Resetting...")
                await self._transition_to(SystemState.RESETTING)
                await self.bus.publish(MoveHomeCommand())

        elif self.state == SystemState.VLM_TRACKING:
            if isinstance(event, TargetNotFoundEvent):
                # Target lost in visual tracking loop
                logger.warning("Visual tracking target lost. Resetting...")
                await self._transition_to(SystemState.RESETTING)
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
            await self.bus.publish(TargetNotFoundEvent())
        except asyncio.CancelledError:
            pass
