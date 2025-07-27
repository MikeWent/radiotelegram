import os
import time
import threading

import telebot
from telebot import types

from radiotelegram.bus import MessageBus, Worker
from radiotelegram.events import (
    RxRecordingCompleteEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
)


class SendChatActionWorker(Worker):
    def __init__(
        self, bus: MessageBus, bot: telebot.TeleBot, chat_id: str, topic_id: int
    ):
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
                    try:
                        self.logger.info("Sending record_voice chat action")
                        self.bot.send_chat_action(
                            chat_id=self.chat_id,
                            action="record_voice",
                            message_thread_id=(
                                self.topic_id if self.topic_id != 0 else None
                            ),
                        )
                        time.sleep(5)
                    except Exception as e:
                        self.logger.error(f"Error sending chat action: {e}")
                        time.sleep(1)
                else:
                    time.sleep(1)
            else:
                time.sleep(0.1)

    def stop(self):
        self._stop_flag.set()
        super().stop()


class TelegramMessageFetchWorker(Worker):
    def __init__(
        self,
        bus: MessageBus,
        bot: telebot.TeleBot,
        chat_id: str,
        topic_id: int | None,
    ):
        super().__init__(bus)
        self.bot = bot
        self.chat_id = chat_id
        self.topic_id = topic_id

    def start(self):
        @self.bot.message_handler(content_types=["voice"])
        def handle_voice_message(message: types.Message):
            # Check if message is from the correct chat and topic
            if (
                str(message.chat.id) == self.chat_id
                and message.from_user.id != self.bot.get_me().id
                and (
                    self.topic_id is None or message.message_thread_id == self.topic_id
                )
            ):

                filename = f"voice_{message.message_id}.ogg"
                filepath = os.path.join("/tmp", filename)
                os.makedirs("/tmp", exist_ok=True)

                try:
                    file_info = self.bot.get_file(message.voice.file_id)
                    downloaded_file = self.bot.download_file(file_info.file_path)

                    with open(filepath, "wb") as voice_file:
                        voice_file.write(downloaded_file)

                    self.logger.info(f"Downloaded voice message to {filepath}")
                    self.bus.publish(
                        TelegramVoiceMessageDownloadedEvent(filepath=filepath)
                    )
                except Exception as e:
                    self.logger.error(f"Error downloading voice message: {e}")


class VoiceMessageUploadWorker(Worker):
    def __init__(
        self, bus: MessageBus, bot: telebot.TeleBot, chat_id: str, topic_id: int
    ):
        super().__init__(bus)
        self.bot = bot
        self.chat_id = chat_id
        self.topic_id = topic_id
        self.bus.subscribe(RxRecordingCompleteEvent, self.queue_event)

    def handle_event(self, event: RxRecordingCompleteEvent):
        """Uploads recorded voice messages to Telegram with fault-tolerance.
        Retries indefinitely if a timeout or any exception occurs.
        """
        self.logger.info(f"Uploading voice message: {event.filepath}")
        retry_delay = 5  # seconds to wait before retrying
        while True:
            try:
                with open(event.filepath, "rb") as voice_file:
                    self.bot.send_voice(
                        chat_id=self.chat_id,
                        voice=voice_file,
                        message_thread_id=self.topic_id if self.topic_id != 0 else None,
                        timeout=10,
                    )
                self.logger.info(f"Successfully uploaded {event.filepath}")
                os.remove(event.filepath)  # Delete the file after a successful upload
                break  # Exit the loop on success
            except FileNotFoundError:
                self.logger.warning(f"File not found: {event.filepath}")
                break
            except Exception as e:
                self.logger.error(
                    f"Failed to upload {event.filepath}: {e}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)

    def start(self):
        # Start the queue processing thread
        self._processing_thread = threading.Thread(
            target=self.process_queue, name=f"{self.__class__.__name__}ProcessingThread"
        )
        self._processing_thread.daemon = True
        self._processing_thread.start()
