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
    """Helper function to make robust Telegram API calls with retry logic."""
    for attempt in range(max_retries):
        try:
            return func()
        except ApiTelegramException as e:
            if e.error_code == 429:  # Rate limit
                retry_after = getattr(e, "retry_after", base_delay * (2 ** attempt))
                logger.warning(f"Rate limited. Waiting {retry_after}s")
                time.sleep(retry_after)
            elif e.error_code in [502, 503, 504] and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Server error {e.error_code}. Retrying in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"API error: {e}")
                raise
        except (ConnectionError, TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Network error: {e}. Retrying in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"Network error after {max_retries} attempts: {e}")
                raise
    return None


class SendChatActionWorker(Worker):
    def __init__(self, bus: MessageBus, bot: telebot.TeleBot, chat_id: str, topic_id: int):
        super().__init__(bus)
        self.bot = bot
        self.chat_id = chat_id
        self.topic_id = topic_id
        self.active = False
        self.recording_start_time = None
        self._stop_flag = threading.Event()
        self.bus.subscribe(RxRecordingStartedEvent, self.on_recording_started)
        self.bus.subscribe(RxRecordingEndedEvent, self.on_recording_finished)

    def on_recording_started(self, event: RxRecordingStartedEvent):
        self.recording_start_time = time.time()
        self.active = True

    def on_recording_finished(self, event: RxRecordingEndedEvent):
        self.active = False
        self.recording_start_time = None

    def start(self):
        while not self._stop_flag.is_set():
            if self.active:
                elapsed_time = time.time() - (self.recording_start_time or 0)
                if elapsed_time >= 3:  # Only send status if recording is ongoing
                    def send_action():
                        return self.bot.send_chat_action(
                            chat_id=self.chat_id,
                            action="record_voice",
                            message_thread_id=(
                                self.topic_id if self.topic_id != 0 else None
                            ),
                            timeout=15,
                        )
                    
                    try:
                        robust_telegram_call(send_action, self.logger)
                        time.sleep(5)
                    except Exception:
                        time.sleep(5)
                else:
                    time.sleep(1)
            else:
                time.sleep(0.1)

    def stop(self):
        self._stop_flag.set()
        super().stop()


class TelegramMessageFetchWorker(Worker):
    def __init__(self, bus: MessageBus, bot: telebot.TeleBot, chat_id: str, topic_id: int | None):
        super().__init__(bus)
        self.bot = bot
        self.chat_id = chat_id
        self.topic_id = topic_id

    def start(self):
        @self.bot.message_handler(content_types=["voice"])
        def handle_voice_message(message: types.Message):
            try:
                # Check if message is from the correct chat and topic
                bot_info = self.bot.get_me()
                if not (
                    str(message.chat.id) == self.chat_id
                    and message.from_user
                    and message.from_user.id != bot_info.id
                    and (self.topic_id is None or message.message_thread_id == self.topic_id)
                ):
                    return

                if not (message.voice and message.voice.file_id):
                    self.logger.error("No voice or file_id in message")
                    return

                filename = f"voice_{message.message_id}.ogg"
                filepath = os.path.join("/tmp", filename)
                os.makedirs("/tmp", exist_ok=True)

                def download_voice():
                    if not message.voice:
                        raise ValueError("No voice in message")
                    file_info = self.bot.get_file(message.voice.file_id)
                    if not file_info.file_path:
                        raise ValueError("No file path in file info")
                    return self.bot.download_file(file_info.file_path)

                downloaded_file = robust_telegram_call(download_voice, self.logger)
                if downloaded_file:
                    with open(filepath, "wb") as voice_file:
                        voice_file.write(downloaded_file)
                    
                    self.logger.info(f"Downloaded voice message to {filepath}")
                    self.bus.publish(TelegramVoiceMessageDownloadedEvent(filepath=filepath))

            except Exception as e:
                self.logger.error(f"Error processing voice message: {e}")


class VoiceMessageUploadWorker(Worker):
    def __init__(self, bus: MessageBus, bot: telebot.TeleBot, chat_id: str, topic_id: int):
        super().__init__(bus)
        self.bot = bot
        self.chat_id = chat_id
        self.topic_id = topic_id
        self.bus.subscribe(RxRecordingCompleteEvent, self.queue_event)

    def handle_event(self, event: RxRecordingCompleteEvent):
        """Uploads recorded voice messages to Telegram with robust retry logic."""
        self.logger.info(f"Uploading voice message: {event.filepath}")
        
        def upload_voice():
            with open(event.filepath, "rb") as voice_file:
                return self.bot.send_voice(
                    chat_id=self.chat_id,
                    voice=voice_file,
                    message_thread_id=self.topic_id if self.topic_id != 0 else None,
                    timeout=30,
                )

        try:
            robust_telegram_call(upload_voice, self.logger, max_retries=5, base_delay=3)
            self.logger.info(f"Successfully uploaded {event.filepath}")
            os.remove(event.filepath)
        except FileNotFoundError:
            self.logger.warning(f"File not found: {event.filepath}")
        except Exception as e:
            self.logger.error(f"Failed to upload {event.filepath} after retries: {e}")
            # Clean up file even if upload failed
            try:
                os.remove(event.filepath)
            except Exception:
                pass

    def start(self):
        # Start the queue processing thread
        self._processing_thread = threading.Thread(
            target=self.process_queue, name=f"{self.__class__.__name__}ProcessingThread"
        )
        self._processing_thread.daemon = True
        self._processing_thread.start()


class TelegramBotPollingWorker(Worker):
    """Robust Telegram bot polling worker with automatic restart on failures."""

    def __init__(self, bus: MessageBus, bot: telebot.TeleBot):
        super().__init__(bus)
        self.bot = bot
        self._stop_event = threading.Event()
        self._consecutive_failures = 0
        self._max_failures = 10

    def _is_network_error(self, exception: Exception) -> bool:
        """Check if exception is a network-related error."""
        if isinstance(exception, ApiTelegramException):
            return exception.error_code in [429, 502, 503, 504]
        return isinstance(exception, (ConnectionError, TimeoutError, OSError))

    def _polling_loop(self):
        """Main polling loop with automatic restart on failures."""
        while not self._stop_event.is_set():
            try:
                self.logger.info("Starting Telegram bot polling...")
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
                    self.logger.error(f"Too many failures ({self._max_failures}). Stopping.")
                    break
                
                delay = min(5 * (2 ** min(self._consecutive_failures - 1, 5)), 60)
                
                if self._is_network_error(e):
                    self.logger.warning(f"Network error (#{self._consecutive_failures}): {e}")
                else:
                    self.logger.error(f"Polling error (#{self._consecutive_failures}): {e}")
                
                self.logger.info(f"Restarting in {delay}s...")
                if self._stop_event.wait(timeout=delay):
                    break

    def start(self):
        """Start the polling worker."""
        self.logger.info("Starting Telegram bot polling worker...")
        self._polling_thread = threading.Thread(target=self._polling_loop, name="TelegramPolling")
        self._polling_thread.daemon = True
        self._polling_thread.start()

    def stop(self):
        """Stop the polling worker gracefully."""
        self.logger.info("Stopping Telegram bot polling worker...")
        self._stop_event.set()
        
        try:
            self.bot.stop_polling()
        except Exception as e:
            self.logger.warning(f"Error stopping bot polling: {e}")
        
        if hasattr(self, '_polling_thread') and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=10)
        
        super().stop()
        self.logger.info("Telegram bot polling worker stopped")
