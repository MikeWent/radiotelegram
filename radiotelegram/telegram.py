import os
import threading
import time

from telebot import types

from radiotelegram.bus import Worker
from radiotelegram.events import (
    RxRecordingCompleteEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
)
from radiotelegram.telegram_retry import (
    TelegramRetry,
    is_retryable_telegram_error,
    robust_telegram_call,
    telegram_call,
)


CHAT_ACTION_RETRY = TelegramRetry(2, 1, 5, False)
BOT_ID_RETRY = TelegramRetry(2, 1, 5, False)
DOWNLOAD_RETRY = TelegramRetry(3, 2, 15, False)
UPLOAD_RETRY = TelegramRetry(3, 2, 15)


def _start_daemon(target, name):
    thread = threading.Thread(target=target, daemon=True, name=name)
    thread.start()
    return thread


class TelegramWorker(Worker):
    def __init__(self, bus, bot, chat_id=None, topic_id=0):
        super().__init__(bus)
        self.bot = bot
        self.chat_id = chat_id
        self.topic_id = topic_id

    @property
    def api_topic_id(self):
        return self.topic_id if self.topic_id != 0 else None

    def call(self, func, policy):
        return telegram_call(func, self.logger, policy, self._stop_event)


class SendChatActionWorker(TelegramWorker):
    def __init__(self, bus, bot, chat_id, topic_id):
        super().__init__(bus, bot, chat_id, topic_id)
        self.active = False
        self.recording_start_time = None
        self.bus.subscribe(RxRecordingStartedEvent, self.on_recording_started)
        self.bus.subscribe(RxRecordingEndedEvent, self.on_recording_finished)

    def on_recording_started(self, event):
        self.recording_start_time = time.time()
        self.active = True

    def on_recording_finished(self, event):
        self.active = False
        self.recording_start_time = None

    def _run_loop(self):
        while not self._stop_event.is_set():
            if not self.active:
                self._stop_event.wait(0.1)
            elif time.time() - (self.recording_start_time or 0) < 3:
                self._stop_event.wait(1)
            else:
                self._send_chat_action()
                self._stop_event.wait(5)

    def _send_chat_action(self):
        self.call(
            lambda: self.bot.send_chat_action(
                chat_id=self.chat_id,
                action="record_voice",
                message_thread_id=self.api_topic_id,
                timeout=15,
            ),
            CHAT_ACTION_RETRY,
        )

    def start(self):
        _start_daemon(self._run_loop, "ChatActionLoop")


class TelegramMessageFetchWorker(TelegramWorker):
    def __init__(self, bus, bot, chat_id, topic_id):
        super().__init__(bus, bot, chat_id, topic_id)
        self._bot_id = None

    def start(self):
        @self.bot.message_handler(content_types=["voice"])
        def handle_voice_message(message: types.Message):
            self._handle_voice_message(message)

    def _handle_voice_message(self, message):
        try:
            if not self._message_allowed(message):
                return
            data = self._download_voice(message.voice.file_id)
            if data:
                filepath = f"/tmp/voice_{message.message_id}.ogg"
                self._save_voice(filepath, data)
                self.logger.info(f"Downloaded voice message: {filepath}")
                self.bus.publish(TelegramVoiceMessageDownloadedEvent(filepath))
        except Exception as error:
            self.logger.error(f"Error processing voice message: {error}")

    def _message_allowed(self, message):
        return bool(
            self._get_bot_id() is not None
            and str(message.chat.id) == self.chat_id
            and message.from_user
            and message.from_user.id != self._bot_id
            and (self.topic_id is None or message.message_thread_id == self.topic_id)
            and message.voice
            and message.voice.file_id
        )

    def _get_bot_id(self):
        if self._bot_id is None:
            self._bot_id = getattr(self.call(self.bot.get_me, BOT_ID_RETRY), "id", None)
        return self._bot_id

    def _download_voice(self, file_id):
        def download():
            file_info = self.bot.get_file(file_id)
            if not file_info.file_path:
                raise ValueError("No file path")
            return self.bot.download_file(file_info.file_path)

        return self.call(download, DOWNLOAD_RETRY)

    @staticmethod
    def _save_voice(filepath, data):
        partial = f"{filepath}.part"
        with open(partial, "wb") as file:
            file.write(data)
        os.replace(partial, filepath)


class VoiceMessageUploadWorker(TelegramWorker):
    def __init__(self, bus, bot, chat_id, topic_id):
        super().__init__(bus, bot, chat_id, topic_id)
        self.max_upload_attempts = 5
        self.upload_retry_delay = 30
        self._retry_counts = {}
        self.bus.subscribe(RxRecordingCompleteEvent, self.queue_event)

    def handle_event(self, event):
        self.logger.info(f"Uploading: {event.filepath}")
        try:
            result = self.call(lambda: self._send_voice(event.filepath), UPLOAD_RETRY)
            if self._stop_event.is_set() and result is None:
                return
            self.logger.info(f"Uploaded: {event.filepath}")
            self._retry_counts.pop(event.filepath, None)
        except FileNotFoundError:
            self.logger.warning(f"File not found: {event.filepath}")
            return
        except Exception as error:
            if is_retryable_telegram_error(error) and self._retry_upload_later(event, error):
                return
            self.logger.error(f"Upload failed: {error}")
        self._remove_file(event.filepath)

    def _send_voice(self, filepath):
        with open(filepath, "rb") as voice:
            return self.bot.send_voice(
                chat_id=self.chat_id,
                voice=voice,
                message_thread_id=self.api_topic_id,
                timeout=30,
            )

    def _retry_upload_later(self, event, error):
        attempts = self._retry_counts.get(event.filepath, 0) + 1
        self._retry_counts[event.filepath] = attempts
        if attempts >= self.max_upload_attempts:
            self.logger.error(f"Upload retries exhausted for {event.filepath}: {error}")
            self._retry_counts.pop(event.filepath, None)
            return False

        self.logger.warning(
            f"Upload network failure, retrying {event.filepath} later "
            f"({attempts}/{self.max_upload_attempts}): {error}"
        )
        timer = threading.Timer(self.upload_retry_delay, self.queue_event, [event])
        timer.daemon = True
        timer.start()
        return True

    @staticmethod
    def _remove_file(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass

    def start(self):
        _start_daemon(self.process_queue, f"{self.__class__.__name__}Queue")


class TelegramBotPollingWorker(TelegramWorker):
    def __init__(self, bus, bot):
        super().__init__(bus, bot)
        self._consecutive_failures = 0
        self._max_failures = 10
        self._last_update_id = None
        self.poll_timeout = 10
        self.long_polling_timeout = 5
        self.backoff_base = 2
        self.backoff_max = 60
        self._thread = None

    def _is_network_error(self, error):
        return is_retryable_telegram_error(error)

    def _skip_pending_updates(self):
        try:
            updates = self.bot.get_updates(
                offset=-1,
                limit=1,
                timeout=0,
                allowed_updates=["message"],
                long_polling_timeout=0,
            )
            if updates:
                self._last_update_id = max(update.update_id for update in updates)
                self.logger.info(
                    f"Skipped pending Telegram updates through {self._last_update_id}"
                )
        except Exception as error:
            self.logger.warning(f"Could not skip pending Telegram updates: {error}")

    def _polling_loop(self):
        self._skip_pending_updates()
        while not self._stop_event.is_set():
            try:
                self._poll_once()
                self._consecutive_failures = 0
            except Exception as error:
                if self._handle_polling_error(error):
                    break

    def _poll_once(self):
        updates = self.bot.get_updates(
            offset=None if self._last_update_id is None else self._last_update_id + 1,
            timeout=self.poll_timeout,
            long_polling_timeout=self.long_polling_timeout,
            allowed_updates=["message"],
        )
        if not updates:
            return
        try:
            self.bot.process_new_updates(updates)
        except Exception as error:
            self.logger.error(f"Error dispatching Telegram updates: {error}")
        self._last_update_id = max(update.update_id for update in updates)

    def _handle_polling_error(self, error):
        self._consecutive_failures += 1
        log = self.logger.warning if self._is_network_error(error) else self.logger.error
        log(f"Polling error #{self._consecutive_failures}: {error}")
        if self._consecutive_failures >= self._max_failures:
            self.logger.error(
                f"Telegram polling reached {self._consecutive_failures} "
                "consecutive failures; resetting polling session"
            )
            self._reset_telegram_session()
            self._consecutive_failures = 0
        return self._stop_event.wait(self._polling_backoff())

    def _polling_backoff(self):
        return min(
            self.backoff_base * (2 ** min(self._consecutive_failures - 1, 5)),
            self.backoff_max,
        )

    def _reset_telegram_session(self):
        try:
            self.bot.stop_polling()
        except Exception:
            pass

    def start(self):
        if not (self._thread and self._thread.is_alive()):
            self._thread = _start_daemon(self._polling_loop, "TelegramPolling")

    def stop(self):
        self._reset_telegram_session()
        super().stop()
