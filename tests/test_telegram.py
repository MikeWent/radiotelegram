"""Tests for radiotelegram.telegram module."""

import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from requests import exceptions as requests_exceptions

from radiotelegram.bus import MessageBus
from radiotelegram.events import (
    RxRecordingCompleteEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
)
from radiotelegram.telegram import (
    SendChatActionWorker,
    TelegramBotPollingWorker,
    TelegramMessageFetchWorker,
    VoiceMessageUploadWorker,
    robust_telegram_call,
)


# ── robust_telegram_call ──────────────────────────────────────────────


class TestRobustTelegramCall:
    def test_success_on_first_try(self):
        func = MagicMock(return_value="ok")
        result = robust_telegram_call(func, logging.getLogger("test"))
        assert result == "ok"
        func.assert_called_once()

    def test_retry_on_rate_limit(self):
        from telebot.apihelper import ApiTelegramException
        exc = ApiTelegramException("rate", 429, {"error_code": 429, "description": "Too Many Requests"})
        exc.retry_after = 0
        func = MagicMock(side_effect=[exc, "ok"])
        result = robust_telegram_call(func, logging.getLogger("test"), max_retries=2, base_delay=0)
        assert result == "ok"
        assert func.call_count == 2

    def test_retry_on_server_error(self):
        from telebot.apihelper import ApiTelegramException
        exc = ApiTelegramException("error", 502, {"error_code": 502, "description": "Bad Gateway"})
        func = MagicMock(side_effect=[exc, "ok"])
        result = robust_telegram_call(func, logging.getLogger("test"), max_retries=2, base_delay=0)
        assert result == "ok"

    def test_raises_on_non_retryable_api_error(self):
        from telebot.apihelper import ApiTelegramException
        exc = ApiTelegramException("error", 400, {"error_code": 400, "description": "Bad Request"})
        func = MagicMock(side_effect=exc)
        with pytest.raises(ApiTelegramException):
            robust_telegram_call(func, logging.getLogger("test"), max_retries=2, base_delay=0)

    def test_retry_on_connection_error(self):
        func = MagicMock(side_effect=[ConnectionError("net"), "ok"])
        result = robust_telegram_call(func, logging.getLogger("test"), max_retries=2, base_delay=0)
        assert result == "ok"

    def test_raises_network_error_after_retries(self):
        func = MagicMock(side_effect=ConnectionError("net"))
        with pytest.raises(ConnectionError):
            robust_telegram_call(func, logging.getLogger("test"), max_retries=2, base_delay=0)

    def test_request_timeout_is_retryable(self):
        func = MagicMock(side_effect=[requests_exceptions.ReadTimeout("slow"), "ok"])
        result = robust_telegram_call(func, logging.getLogger("test"), max_retries=2, base_delay=0)
        assert result == "ok"

    def test_raise_errors_false_returns_none_after_retries(self):
        func = MagicMock(side_effect=requests_exceptions.ReadTimeout("slow"))
        result = robust_telegram_call(
            func,
            logging.getLogger("test"),
            max_retries=2,
            base_delay=0,
            raise_errors=False,
        )
        assert result is None

    def test_stop_event_interrupts_retry_sleep(self):
        stop_event = threading.Event()
        func = MagicMock(side_effect=requests_exceptions.ReadTimeout("slow"))

        def wait(timeout):
            stop_event.set()
            return True

        with patch.object(stop_event, "wait", side_effect=wait):
            result = robust_telegram_call(
                func,
                logging.getLogger("test"),
                max_retries=3,
                base_delay=10,
                stop_event=stop_event,
                raise_errors=False,
            )
        assert result is None
        assert func.call_count == 1


# ── SendChatActionWorker ──────────────────────────────────────────────


class TestSendChatActionWorker:
    def setup_method(self):
        self.bus = MessageBus()
        self.bot = MagicMock()
        self.worker = SendChatActionWorker(self.bus, self.bot, "12345", 0)

    def teardown_method(self):
        self.worker.stop()
        self.bus.shutdown()

    def test_activates_on_recording_start_deactivates_on_end(self):
        self.worker.on_recording_started(RxRecordingStartedEvent())
        assert self.worker.active is True
        self.worker.on_recording_finished(RxRecordingEndedEvent())
        assert self.worker.active is False

    def test_bus_delivers_recording_events_to_worker(self):
        self.bus.publish(RxRecordingStartedEvent())
        time.sleep(0.3)
        assert self.worker.active is True
        self.bus.publish(RxRecordingEndedEvent())
        time.sleep(0.3)
        assert self.worker.active is False


# ── TelegramMessageFetchWorker ────────────────────────────────────────


class TestTelegramMessageFetchWorker:
    def setup_method(self):
        self.bus = MessageBus()
        self.bot = MagicMock()
        self.worker = TelegramMessageFetchWorker(self.bus, self.bot, "12345", 99)

    def teardown_method(self):
        self.bus.shutdown()

    def test_start_registers_handler(self):
        self.worker.start()
        self.bot.message_handler.assert_called_once_with(content_types=["voice"])


# ── VoiceMessageUploadWorker ─────────────────────────────────────────


class TestVoiceMessageUploadWorker:
    def setup_method(self):
        self.bus = MessageBus()
        self.bot = MagicMock()
        self.worker = VoiceMessageUploadWorker(self.bus, self.bot, "12345", 0)

    def teardown_method(self):
        self.bus.shutdown()

    @patch("radiotelegram.telegram.os.remove")
    def test_handle_event_uploads_voice(self, mock_remove):
        self.bot.send_voice.return_value = MagicMock()
        event = RxRecordingCompleteEvent(filepath="/tmp/test.ogg")
        with patch("builtins.open", MagicMock()):
            self.worker.handle_event(event)
        assert self.bot.send_voice.called

    def test_handle_event_file_not_found(self):
        event = RxRecordingCompleteEvent(filepath="/tmp/nonexistent.ogg")
        with patch("builtins.open", side_effect=FileNotFoundError):
            self.worker.handle_event(event)

    @patch("radiotelegram.telegram.os.remove")
    def test_handle_event_upload_failure_cleans_up(self, mock_remove):
        self.bot.send_voice.side_effect = RuntimeError("upload failed")
        event = RxRecordingCompleteEvent(filepath="/tmp/test.ogg")
        with patch("builtins.open", MagicMock()):
            self.worker.handle_event(event)
        mock_remove.assert_called()

    def test_network_upload_failure_is_requeued_without_cleanup(self):
        self.bot.send_voice.side_effect = requests_exceptions.ReadTimeout("slow")
        self.worker.upload_retry_delay = 0
        event = RxRecordingCompleteEvent(filepath="/tmp/test.ogg")
        with (
            patch("builtins.open", MagicMock()),
            patch("radiotelegram.telegram.os.remove") as mock_remove,
            patch.object(self.worker, "queue_event") as queue_event,
            patch("radiotelegram.telegram.threading.Timer") as timer_cls,
        ):
            timer = MagicMock()
            timer_cls.return_value = timer
            self.worker.handle_event(event)

        mock_remove.assert_not_called()
        timer_cls.assert_called_once()
        assert timer.daemon is True
        timer.start.assert_called_once()


# ── TelegramBotPollingWorker ─────────────────────────────────────────


class TestTelegramBotPollingWorker:
    def setup_method(self):
        self.bus = MessageBus()
        self.bot = MagicMock()
        self.worker = TelegramBotPollingWorker(self.bus, self.bot)

    def teardown_method(self):
        self.worker.stop()
        self.bus.shutdown()

    def test_is_network_error_true(self):
        from telebot.apihelper import ApiTelegramException
        assert self.worker._is_network_error(
            ApiTelegramException("x", 502, {"error_code": 502, "description": "Bad Gateway"})
        )
        assert self.worker._is_network_error(ConnectionError())
        assert self.worker._is_network_error(TimeoutError())

    def test_is_network_error_false(self):
        from telebot.apihelper import ApiTelegramException
        assert not self.worker._is_network_error(
            ApiTelegramException("x", 400, {"error_code": 400, "description": "Bad Request"})
        )
        assert not self.worker._is_network_error(ValueError("nope"))

    def test_stop_signals_event_and_stops_bot(self):
        with patch.object(threading.Thread, "start"):
            self.worker.start()
        self.worker.stop()
        assert self.worker._stop_event.is_set()
        self.bot.stop_polling.assert_called_once()

    def test_polling_loop_resets_session_after_max_failures(self):
        self.bot.get_updates.side_effect = ConnectionError("net")
        self.worker._max_failures = 1
        with (
            patch.object(self.worker, "_skip_pending_updates"),
            patch.object(self.worker._stop_event, "wait", return_value=True),
        ):
            self.worker._polling_loop()
        self.bot.stop_polling.assert_called_once()
        assert self.worker._consecutive_failures == 0

    def test_polling_loop_dispatches_updates_and_advances_offset(self):
        update = MagicMock()
        update.update_id = 42
        self.bot.get_updates.return_value = [update]

        def stop_after_dispatch(updates):
            self.worker._stop_event.set()

        self.bot.process_new_updates.side_effect = stop_after_dispatch
        with patch.object(self.worker, "_skip_pending_updates"):
            self.worker._polling_loop()

        self.bot.process_new_updates.assert_called_once_with([update])
        assert self.worker._last_update_id == 42
