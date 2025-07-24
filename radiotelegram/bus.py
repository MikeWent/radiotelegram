import asyncio
import functools
import logging
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Deque, Dict, List, Type


def cpu_intensive(func):
    """
    Decorator to mark a handler as CPU-intensive.
    These handlers will be executed in a thread pool even if they're async.
    """
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()

            # For async functions, we create a sync wrapper that runs the async function
            def sync_runner():
                # Create a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(func(*args, **kwargs))
                finally:
                    new_loop.close()

            return await loop.run_in_executor(None, sync_runner)

        async_wrapper._cpu_intensive = True
        return async_wrapper
    else:
        func._cpu_intensive = True
        return func


# Message Bus
class MessageBus:
    def __init__(self, max_workers: int = 4):
        self.subscribers: Dict[Type, List[Callable]] = {}
        self.logger = logging.getLogger("MessageBus")
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="MessageBus"
        )
        self._shutdown = False

    def subscribe(self, event_type: Type, handler: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def publish(self, event):
        """Publish an event to all subscribers, processing each concurrently."""
        self.logger.info(f"Publishing event: {event}")
        event_type = type(event)

        if event_type in self.subscribers:
            # Create tasks for all handlers to run concurrently
            tasks = []
            for handler in self.subscribers[event_type]:
                # Each handler runs as a separate task for concurrency
                task = asyncio.create_task(self._run_handler_safely(handler, event))
                tasks.append(task)

            if tasks:
                # Wait for all handlers to complete
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    # Log any exceptions that occurred
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            handler_name = getattr(
                                self.subscribers[event_type][i], "__name__", "unknown"
                            )
                            self.logger.error(
                                f"Handler {handler_name} failed: {result}"
                            )
                except Exception as e:
                    self.logger.error(f"Error in message handlers: {e}")

    async def _run_handler_safely(self, handler: Callable, event):
        """Run a message handler safely with proper error handling."""
        try:
            if self._shutdown:
                return

            # Check if handler is marked as CPU-intensive
            is_cpu_intensive = getattr(handler, "_cpu_intensive", False)

            if is_cpu_intensive:
                # Force CPU-intensive handlers to run in thread pool
                loop = asyncio.get_event_loop()
                if asyncio.iscoroutinefunction(handler):
                    # For CPU-intensive async handlers, the decorator handles thread execution
                    await handler(event)
                else:
                    # For CPU-intensive sync handlers, run in thread pool
                    await loop.run_in_executor(self.executor, handler, event)
            else:
                # For regular handlers
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    # Run sync handlers in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(self.executor, handler, event)

        except Exception as e:
            handler_name = getattr(handler, "__name__", str(handler))
            self.logger.error(f"Error in handler {handler_name}: {e}")
            raise  # Re-raise so gather() can catch it

    async def shutdown(self):
        """Shutdown the message bus and cleanup resources."""
        self.logger.info("Shutting down MessageBus...")
        self._shutdown = True

        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            self.logger.warning(f"Error shutting down executor: {e}")

        self.logger.info("MessageBus shutdown complete")


# Abstract Worker Class
class Worker(ABC):
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.enabled = True
        self.queue: Deque = deque()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix=f"Worker-{self.__class__.__name__}"
        )
        self._processing = False

    async def queue_event(self, event):
        """Add an event to the processing queue."""
        self.queue.append(event)

    async def process_queue(self):
        """Process events from the queue, each in a separate thread."""
        self._processing = True

        while self._processing:
            if self.enabled and self.queue:
                event = self.queue.popleft()
                # Process each event in a separate thread to prevent blocking
                try:
                    await self._handle_event_in_thread(event)
                except Exception as e:
                    self.logger.error(
                        f"Error processing event {type(event).__name__}: {e}"
                    )
            else:
                await asyncio.sleep(0.1)

    async def _handle_event_in_thread(self, event):
        """Handle an event in a separate thread."""
        try:
            loop = asyncio.get_event_loop()

            # Check if handle_event is a coroutine function
            if asyncio.iscoroutinefunction(self.handle_event):
                # For async handle_event, run it directly (it's already async)
                await self.handle_event(event)
            else:
                # For sync handle_event, run it in thread pool
                await loop.run_in_executor(self.executor, self.handle_event, event)

        except Exception as e:
            self.logger.error(f"Error in handle_event for {type(event).__name__}: {e}")

    async def handle_event(self, event):
        """Override this method to handle specific events."""
        pass

    async def stop(self):
        """Stop the worker and cleanup resources."""
        self.logger.info(f"Stopping {self.__class__.__name__}...")
        self._processing = False
        self.enabled = False

        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            self.logger.warning(f"Error shutting down worker executor: {e}")

    @abstractmethod
    async def start(self):
        pass
