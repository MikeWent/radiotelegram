import functools
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Deque, Dict, List, Type


def cpu_intensive(func):
    """
    Decorator to mark a handler as CPU-intensive.
    These handlers will be executed in a thread pool.
    """
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
        self._lock = threading.Lock()

    def subscribe(self, event_type: Type, handler: Callable):
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)

    def publish(self, event):
        """Publish an event to all subscribers, processing each in a separate thread."""
        self.logger.info(f"Publishing event: {event.__class__.__name__}")
        event_type = type(event)

        with self._lock:
            if event_type in self.subscribers:
                # Submit each handler to run in a separate thread for concurrency
                futures = []
                for handler in self.subscribers[event_type]:
                    future = self.executor.submit(
                        self._run_handler_safely, handler, event
                    )
                    futures.append(future)

                # Wait for all handlers to complete (optional)
                # You can remove this if you want fire-and-forget behavior
                for future in futures:
                    try:
                        future.result(timeout=30)  # 30-second timeout per handler
                    except Exception as e:
                        self.logger.error(f"Handler execution failed: {e}")

    def _run_handler_safely(self, handler: Callable, event):
        """Run a message handler safely with proper error handling."""
        try:
            if self._shutdown:
                return

            # Check if handler is marked as CPU-intensive
            is_cpu_intensive = getattr(handler, "_cpu_intensive", False)

            if is_cpu_intensive:
                # CPU-intensive handlers run directly in the thread pool
                handler(event)
            else:
                # Regular handlers also run in thread pool for isolation
                handler(event)

        except Exception as e:
            handler_name = getattr(handler, "__name__", str(handler))
            self.logger.error(f"Error in handler {handler_name}: {e}")
            raise  # Re-raise so futures can catch it

    def shutdown(self):
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
        self._processing_thread = None
        self._stop_event = threading.Event()

    def queue_event(self, event):
        """Add an event to the processing queue."""
        self.queue.append(event)

    def process_queue(self):
        """Process events from the queue, each in a separate thread."""
        self._processing = True

        while self._processing and not self._stop_event.is_set():
            if self.enabled and self.queue:
                try:
                    event = self.queue.popleft()
                    # Process each event in a separate thread to prevent blocking
                    future = self.executor.submit(self._handle_event_in_thread, event)
                    # Optional: wait for completion or let it run in background
                    try:
                        future.result(timeout=30)  # 30-second timeout
                    except Exception as e:
                        self.logger.error(
                            f"Error processing event {type(event).__name__}: {e}"
                        )
                except IndexError:
                    # Queue is empty
                    pass

            # Small sleep to prevent busy waiting
            time.sleep(0.1)

    def _handle_event_in_thread(self, event):
        """Handle an event in the current thread."""
        try:
            self.handle_event(event)
        except Exception as e:
            self.logger.error(f"Error in handle_event for {type(event).__name__}: {e}")

    def handle_event(self, event):
        """Override this method to handle specific events."""
        pass

    def stop(self):
        """Stop the worker and cleanup resources."""
        self.logger.info(f"Stopping {self.__class__.__name__}...")
        self._processing = False
        self._stop_event.set()

        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)

        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            self.logger.warning(f"Error shutting down worker executor: {e}")

    @abstractmethod
    def start(self):
        """Start the worker. This method should start any necessary background threads."""
        pass
