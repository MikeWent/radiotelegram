"""Tests for radiotelegram.bus — MessageBus pub/sub and Worker queue lifecycle."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from radiotelegram.bus import MessageBus, Worker
from radiotelegram.events import (
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TxMessagePlaybackStartedEvent,
)


# ── MessageBus behaviour ─────────────────────────────────────────────


class TestMessageBusPubSub:
    def setup_method(self):
        self.bus = MessageBus()

    def teardown_method(self):
        self.bus.shutdown()

    def test_handler_only_fires_for_subscribed_type(self):
        started_h = MagicMock()
        ended_h = MagicMock()
        self.bus.subscribe(RxRecordingStartedEvent, started_h)
        self.bus.subscribe(RxRecordingEndedEvent, ended_h)
        self.bus.publish(RxRecordingStartedEvent())
        started_h.assert_called_once()
        ended_h.assert_not_called()

    def test_publish_with_no_subscribers_is_silent(self):
        self.bus.publish(TxMessagePlaybackStartedEvent())

    def test_all_subscribers_receive_the_same_event_object(self):
        received = []
        self.bus.subscribe(RxRecordingStartedEvent, lambda e: received.append(id(e)))
        self.bus.subscribe(RxRecordingStartedEvent, lambda e: received.append(id(e)))
        self.bus.publish(RxRecordingStartedEvent())
        assert len(received) == 2
        assert received[0] == received[1]


class TestMessageBusErrorIsolation:
    def setup_method(self):
        self.bus = MessageBus()

    def teardown_method(self):
        self.bus.shutdown()

    def test_second_handler_runs_despite_first_crashing(self):
        boom = MagicMock(side_effect=ValueError("boom"))
        ok = MagicMock()
        self.bus.subscribe(RxRecordingStartedEvent, boom)
        self.bus.subscribe(RxRecordingStartedEvent, ok)
        self.bus.publish(RxRecordingStartedEvent())
        boom.assert_called_once()
        ok.assert_called_once()


class TestMessageBusShutdown:
    def test_handlers_skipped_after_shutdown(self):
        bus = MessageBus()
        handler = MagicMock()
        bus.subscribe(RxRecordingStartedEvent, handler)
        bus.shutdown()
        bus._run_handler(handler, RxRecordingStartedEvent())
        handler.assert_not_called()


# ── Worker queue lifecycle ───────────────────────────────────────────


class ConcreteWorker(Worker):
    def __init__(self, bus):
        super().__init__(bus)
        self.handled = []

    def handle_event(self, event):
        self.handled.append(event)

    def start(self):
        pass


class TestWorkerQueueDrain:
    def setup_method(self):
        self.bus = MessageBus()
        self.worker = ConcreteWorker(self.bus)

    def teardown_method(self):
        self.worker.stop()
        self.bus.shutdown()

    def test_drains_multiple_events_in_order(self):
        e1 = RxRecordingStartedEvent()
        e2 = RxRecordingEndedEvent()
        self.worker.queue_event(e1)
        self.worker.queue_event(e2)
        t = threading.Thread(target=self.worker.process_queue, daemon=True)
        t.start()
        time.sleep(0.5)
        self.worker.stop()
        t.join(timeout=2)
        assert self.worker.handled == [e1, e2]


class TestWorkerEnableGating:
    def setup_method(self):
        self.bus = MessageBus()
        self.worker = ConcreteWorker(self.bus)

    def teardown_method(self):
        self.worker.stop()
        self.bus.shutdown()

    def test_events_stay_queued_while_disabled(self):
        self.worker.enabled = False
        self.worker.queue_event(RxRecordingStartedEvent())
        t = threading.Thread(target=self.worker.process_queue, daemon=True)
        t.start()
        time.sleep(0.3)
        self.worker.stop()
        t.join(timeout=2)
        assert len(self.worker.handled) == 0
        assert len(self.worker.queue) == 1

    def test_events_drain_after_re_enable(self):
        self.worker.enabled = False
        self.worker.queue_event(RxRecordingStartedEvent())
        t = threading.Thread(target=self.worker.process_queue, daemon=True)
        t.start()
        time.sleep(0.2)
        self.worker.enabled = True
        time.sleep(0.5)
        self.worker.stop()
        t.join(timeout=2)
        assert len(self.worker.handled) == 1


class TestWorkerErrorResilience:
    def setup_method(self):
        self.bus = MessageBus()

    def teardown_method(self):
        self.bus.shutdown()

    def test_queue_continues_after_handler_crash(self):
        class Fragile(Worker):
            def __init__(self, bus):
                super().__init__(bus)
                self.call_count = 0

            def handle_event(self, event):
                self.call_count += 1
                if self.call_count == 1:
                    raise RuntimeError("first call fails")

            def start(self):
                pass

        w = Fragile(self.bus)
        w.queue_event(RxRecordingStartedEvent())
        w.queue_event(RxRecordingEndedEvent())
        t = threading.Thread(target=w.process_queue, daemon=True)
        t.start()
        time.sleep(0.5)
        w.stop()
        t.join(timeout=2)
        assert w.call_count == 2


class TestWorkerGracefulStop:
    def test_stop_unblocks_process_queue(self):
        bus = MessageBus()
        w = ConcreteWorker(bus)
        t = threading.Thread(target=w.process_queue, daemon=True)
        t.start()
        w.stop()
        t.join(timeout=3)
        assert not t.is_alive()
        bus.shutdown()
