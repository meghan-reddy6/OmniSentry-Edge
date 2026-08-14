"""
Event bus implementation for async inter-agent communication.
Defines EventBus and BaseAgent classes.
"""
import asyncio
import logging
from typing import Dict, List, Callable, Union, Type
from src.common.messages import Event
from src.common.config import SystemConfig

logger = logging.getLogger(__name__)

CallbackType = Callable[[Event], Union[None, asyncio.Future]]

class EventBus:
    """An asynchronous in-memory event bus supporting publish-subscribe pattern."""
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[CallbackType]] = {}

    def subscribe(self, event_type: Type[Event], callback: CallbackType):
        """Subscribes a callback to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed {callback} to {event_type.__name__}")

    async def publish(self, event: Event):
        """Asynchronously publishes an event to all subscribers."""
        event_type = type(event)
        callbacks = self._subscribers.get(event_type, [])
        if not callbacks:
            logger.debug(f"No subscribers for event: {event_type.__name__}")
            return

        logger.debug(f"Publishing {event_type.__name__} to {len(callbacks)} subscribers")
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(
                    f"Error in subscriber callback {callback} for event {event_type.__name__}: {e}",
                    exc_info=True
                )

class BaseAgent:
    """Base class for all system agents. Implements queue-based asynchronous processing."""
    def __init__(self, name: str, bus: EventBus, config: SystemConfig):
        self.name = name
        self.bus = bus
        self.config = config
        self.queue = asyncio.Queue()
        self._task = None
        self._running = False

    def subscribe(self, event_type: Type[Event]):
        """Subscribes this agent's queue to an event type."""
        self.bus.subscribe(event_type, self.queue.put_nowait)

    async def start(self):
        """Starts the agent runner loop."""
        self._running = True
        await self.setup()
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Agent {self.name} started successfully.")

    async def stop(self):
        """Stops the agent runner loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.cleanup()
        logger.info(f"Agent {self.name} stopped.")

    async def setup(self):
        """Hook for agent initialization (e.g. opening device drivers). Override in subclasses."""
        pass

    async def cleanup(self):
        """Hook for agent teardown (e.g. closing drivers). Override in subclasses."""
        pass

    async def _loop(self):
        """Continuous event consumption loop."""
        while self._running:
            try:
                event = await self.queue.get()
                await self.handle_event(event)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Agent {self.name} encountered an error processing an event: {e}",
                    exc_info=True
                )

    async def handle_event(self, event: Event):
        """Handles incoming events. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses of BaseAgent must implement handle_event")
