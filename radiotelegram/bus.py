import asyncio
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Deque, Dict, List, Type


# Message Bus
class MessageBus:
    def __init__(self):
        self.subscribers: Dict[Type, List[Callable]] = {}
        self.logger = logging.getLogger("MessageBus")

    def subscribe(self, event_type: Type, handler: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def publish(self, event):
        self.logger.info(event)
        event_type = type(event)
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                await handler(event)


# Abstract Worker Class
class Worker(ABC):
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.enabled = True
        self.queue: Deque = deque()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def queue_event(self, event):
        self.queue.append(event)

    async def process_queue(self):
        while True:
            if self.enabled and self.queue:
                event = self.queue.popleft()
                await self.handle_event(event)
            await asyncio.sleep(0.1)

    async def handle_event(self, event):
        pass

    @abstractmethod
    async def start(self):
        pass
