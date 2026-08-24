import asyncio
import inspect
import logging
from typing import Callable, Dict, List, Type, Any

logger = logging.getLogger(__name__)

class Event:
    pass

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()
        self._loop = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def subscribe(self, event_type: str, handler: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable):
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    def publish(self, event: Any):
        event_name = event.__class__.__name__
        handlers = self._subscribers.get(event_name, [])
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    if self._loop and self._loop.is_running():
                        asyncio.run_coroutine_threadsafe(handler(event), self._loop)
                    else:
                        asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"[EventBus] Error dispatching {event_name} to {handler.__name__}: {e}", exc_info=True)
