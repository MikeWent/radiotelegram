import os
import time
import threading
from typing import Optional

import telebot
from telebot import types
from telebot.apihelper import ApiTelegramException

from radiotelegram.bus import MessageBus, Worker
from radiotelegram.events import (
    RxRecordingCompleteEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
)


def robust_telegram_call(func, logger, max_retries=3, base_delay=2):
    for attempt in range(max_retries):
        try:
            return func()
        except ApiTelegramException as e:
            if e.error_code == 429:
                retry_after = getattr(e, "retry_after", base_delay * (2**attempt))
                logger.warning(f"Rate limited. Waiting {retry_after}s")
                time.sleep(retry_after)
            elif e.error_code in [502, 503, 504] and attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(f"Server error {e.error_code}. Retrying in {delay}s")
                time.sleep(delay)
            else:
                raise
        except (ConnectionError, TimeoutError, OSError) as e:
            if isinstance(e, (FileNotFoundError, PermissionError)):
                raise
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(f"Network error: {e}. Retrying in {delay}s")
                time.sleep(delay)
            else:
                raise
    return None


class SendChatActionWorker(Worker):
    def __init__(self, bus, bot, chat_id, topic_id):
        super().__init__(bus)
        self.bot = bot
        self.chat_id = chat_id
        self.topic_id = topic_id
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
            if self.active:
                elapsed = time.time() - (self.recording_start_time or 0)
                if elapsed >= 3:
                    try:
                        robust_telegram_call(
                            lambda: self.bot.send_chat_action(
                                chat_id=self.chat_id,
                                action="record_voice",
                                message_thread_id=(
                                    self.topic_id if self.topic_id != 0 else None
                                ),
                                timeout=15,
                            ),
                            self.logger,
                        )
                    except Exception:
                        pass
                    time.sleep(5)
                else:
                    time.sleep(1)
            else:
                time.sleep(0.1)

    def start(self):
        threading.Thread(
            target=self._run_loop, daemon=True, name="ChatActionLoop"
        ).start()

    def stop(self):
        super().stop()


class TelegramMessageFetchWorker(Worker):
    def __init__(self, bus, bot, chat_id, topic_id):
        super().__init__(bus)
        self.bot = bot
        self.chat_id = chat_id
        self.topic_id = topic_id
        self._bot_id = None

    def start(self):
        @self.bot.message_handler(content_types=["voice"])
        def handle_voice_message(message: types.Message):
            try:
                if self._bot_id is None:
                    self._bot_id = self.bot.get_me().id

                if not (
                    str(message.chat.id) == self.chat_id
                    and message.from_user
                    and message.from_user.id != self._bot_id
                    and (
                        self.topic_id is None
                        or message.message_thread_id == self.topic_id
                    )
                ):
                    return

                if not (message.voice and message.voice.file_id):
                    return

                filepath = f"/tmp/voice_{message.message_id}.ogg"

                def download():
                    file_info = self.bot.get_file(message.voice.file_id)
                    if not file_info.file_path:
                        raise ValueError("No file path")
                    return self.bot.download_file(file_info.file_path)

                data = robust_telegram_call(download, self.logger)
                if data:
                    with open(filepath, "wb") as f:
                        f.write(data)
                    self.logger.info(f"Downloaded voice message: {filepath}")
                    self.bus.publish(
                        TelegramVoiceMessageDownloadedEvent(filepath=filepath)
                    )

            except Exception as e:
                self.logger.error(f"Error processing voice message: {e}")


class VoiceMessageUploadWorker(Worker):
    def __init__(self, bus, bot, chat_id, topic_id):
        super().__init__(bus)
        self.bot = bot
        self.chat_id = chat_id
        self.topic_id = topic_id
        self.bus.subscribe(RxRecordingCompleteEvent, self.queue_event)

    def handle_event(self, event):
        self.logger.info(f"Uploading: {event.filepath}")
        try:
            robust_telegram_call(
                lambda: self.bot.send_voice(
                    chat_id=self.chat_id,
                    voice=open(event.filepath, "rb"),
                    message_thread_id=self.topic_id if self.topic_id != 0 else None,
                    timeout=30,
                ),
                self.logger,
                max_retries=5,
                base_delay=3,
            )
            self.logger.info(f"Uploaded: {event.filepath}")
        except FileNotFoundError:
            self.logger.warning(f"File not found: {event.filepath}")
            return
        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
        try:
            os.remove(event.filepath)
        except Exception:
            pass

    def start(self):
        threading.Thread(
            target=self.process_queue,
            daemon=True,
            name=f"{self.__class__.__name__}Queue",
        ).start()


class TelegramBotPollingWorker(Worker):
    def __init__(self, bus, bot):
        super().__init__(bus)
        self.bot = bot
        self._consecutive_failures = 0
        self._max_failures = 10

    def _is_network_error(self, exception):
        if isinstance(exception, ApiTelegramException):
            return exception.error_code in [429, 502, 503, 504]
        return isinstance(exception, (ConnectionError, TimeoutError, OSError))

    def _polling_loop(self):
        while not self._stop_event.is_set():
            try:
                self.logger.info("Starting Telegram polling...")
                self._consecutive_failures = 0
                self.bot.infinity_polling(
                    timeout=10,
                    long_polling_timeout=5,
                    skip_pending=True,
                    allowed_updates=["message"],
                )
            except Exception as e:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_failures:
                    self.logger.error(f"Too many failures ({self._max_failures})")
                    break
                delay = min(5 * (2 ** min(self._consecutive_failures - 1, 5)), 60)
                if self._is_network_error(e):
                    self.logger.warning(
                        f"Network error #{self._consecutive_failures}: {e}"
                    )
                else:
                    self.logger.error(
                        f"Polling error #{self._consecutive_failures}: {e}"
                    )
                if self._stop_event.wait(timeout=delay):
                    break

    def start(self):
        threading.Thread(
            target=self._polling_loop, daemon=True, name="TelegramPolling"
        ).start()

    def stop(self):
        try:
            self.bot.stop_polling()
        except Exception:
            pass
        super().stop()
