import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class MessageBus:
    def __init__(self, max_workers=4):
        self.subscribers = {}
        self.logger = logging.getLogger("MessageBus")
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="MessageBus"
        )
        self._shutdown = False
        self._lock = threading.Lock()

    def subscribe(self, event_type, handler):
        with self._lock:
            self.subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event):
        with self._lock:
            handlers = list(self.subscribers.get(type(event), []))
        futures = [
            self.executor.submit(self._run_handler, handler, event)
            for handler in handlers
        ]
        for future in futures:
            try:
                future.result(timeout=30)
            except Exception:
                pass

    def _run_handler(self, handler, event):
        if self._shutdown:
            return
        try:
            handler(event)
        except Exception as e:
            self.logger.error(
                f"Handler {getattr(handler, '__name__', handler)} error: {e}"
            )
            raise

    def shutdown(self):
        self._shutdown = True
        self.executor.shutdown(wait=True)


class Worker(ABC):
    def __init__(self, bus):
        self.bus = bus
        self.enabled = True
        self.queue = deque()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop_event = threading.Event()

    def queue_event(self, event):
        self.queue.append(event)

    def process_queue(self):
        while not self._stop_event.is_set():
            if self.enabled and self.queue:
                try:
                    event = self.queue.popleft()
                    self.handle_event(event)
                except IndexError:
                    pass
                except Exception as e:
                    self.logger.error(f"Error handling {type(event).__name__}: {e}")
            else:
                time.sleep(0.1)

    def handle_event(self, event):
        pass

    def stop(self):
        self._stop_event.set()

    @abstractmethod
    def start(self):
        pass
