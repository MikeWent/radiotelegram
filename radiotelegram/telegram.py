import asyncio
import os

from aiogram import Bot, Dispatcher, F, types
from aiogram.types import FSInputFile
from bus import MessageBus, Worker
from events import (
    RxRecordingCompleteEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
)


class SendChatActionWorker(Worker):
    def __init__(
        self, bus: MessageBus, bot: Bot, dp: Dispatcher, chat_id: str, topic_id: int
    ):
        super().__init__(bus)
        self.bot = bot
        self.dp = dp
        self.chat_id = chat_id
        self.topic_id = topic_id
        self.active = False
        self.recording_start_time = None
        self.bus.subscribe(RxRecordingStartedEvent, self.on_recording_started)
        self.bus.subscribe(RxRecordingEndedEvent, self.on_recording_finished)

    async def on_recording_started(self, event: RxRecordingStartedEvent):
        self.recording_start_time = asyncio.get_event_loop().time()
        self.active = True

    async def on_recording_finished(self, event: RxRecordingEndedEvent):
        self.active = False
        self.recording_start_time = None

    async def start(self):
        while True:
            if self.active:
                elapsed_time = asyncio.get_event_loop().time() - (
                    self.recording_start_time or 0
                )
                if elapsed_time >= 3:  # Only send status if recording is ongoing
                    self.logger.info("Sending record_voice chat action")
                    await self.bot.send_chat_action(
                        chat_id=self.chat_id,
                        action="record_voice",
                        message_thread_id=self.topic_id,
                    )
                    await asyncio.sleep(5)
                await asyncio.sleep(1)
            else:
                await asyncio.sleep(0.1)


class TelegramMessageFetchWorker(Worker):
    def __init__(
        self, bus: MessageBus, bot: Bot, dp: Dispatcher, chat_id: str, topic_id: int | None
    ):
        super().__init__(bus)
        self.bot = bot
        self.dp = dp
        self.chat_id = chat_id
        self.topic_id = topic_id

    async def start(self):
        @self.dp.message(
            lambda message: message.content_type == types.ContentType.VOICE
            and message.from_user.id != self.bot.id
            and message.chat.id == int(self.chat_id)
            and (message.message_thread_id == self.topic_id or self.topic_id is None)
        )
        async def handle_voice_message(message: types.Message):
            filename = f"voice_{message.message_id}.ogg"
            filepath = os.path.join("downloads", filename)
            os.makedirs("downloads", exist_ok=True)
            await self.bot.download(
                file=message.voice.file_id, destination=filepath, timeout=10  # type: ignore
            )
            self.logger.info(f"Downloaded voice message to {filepath}")
            await self.bus.publish(
                TelegramVoiceMessageDownloadedEvent(filepath=filepath)
            )


# VoiceMessageUploadWorker with retry logic
class VoiceMessageUploadWorker(Worker):
    def __init__(self, bus: MessageBus, bot: Bot, chat_id: str, topic_id: int):
        super().__init__(bus)
        self.bot = bot
        self.chat_id = chat_id
        self.topic_id = topic_id
        self.bus.subscribe(RxRecordingCompleteEvent, self.queue_event)

    async def handle_event(self, event: RxRecordingCompleteEvent):
        """Uploads recorded voice messages to Telegram with fault-tolerance.
        Retries indefinitely if a timeout or any exception occurs.
        """
        self.logger.info(f"Uploading voice message: {event.filepath}")
        retry_delay = 5  # seconds to wait before retrying
        while True:
            try:
                with open(event.filepath, "rb") as voice_file:
                    # FSInputFile encapsulates the file for uploading
                    await self.bot.send_voice(
                        chat_id=self.chat_id,
                        voice=FSInputFile(path=event.filepath),
                        message_thread_id=self.topic_id,
                        request_timeout=5,
                    )
                self.logger.info(f"Successfully uploaded {event.filepath}")
                os.remove(event.filepath)  # Delete the file after a successful upload
                break  # Exit the loop on success
            except FileNotFoundError:
                break
            except Exception as e:
                self.logger.error(
                    f"Failed to upload {event.filepath}: {e}. Retrying in {retry_delay} seconds..."
                )
                await asyncio.sleep(retry_delay)

    async def start(self):
        asyncio.create_task(self.process_queue())
