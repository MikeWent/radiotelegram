import asyncio
import logging
import os
import signal

from aiogram import Bot, Dispatcher
from bus import MessageBus
from dotenv import load_dotenv
from radio import EnhancedRxListenWorker, EnhancedTxPlayWorker
from telegram import (
    SendChatActionWorker,
    TelegramMessageFetchWorker,
    VoiceMessageUploadWorker,
)

# Logger setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)


async def main():
    load_dotenv()
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    TOPIC_ID = int(os.getenv("TELEGRAM_TOPIC_ID", "0")) or None
    AUDIO_DEVICE = os.getenv(
        "AUDIO_DEVICE", "pulse"
    )  # Default to PipeWire/PulseAudio compatible device
    assert TELEGRAM_BOT_TOKEN and CHAT_ID

    bus = MessageBus()
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    audio_listener = EnhancedRxListenWorker(bus, audio_device=AUDIO_DEVICE)
    audio_player = EnhancedTxPlayWorker(bus)
    message_fetcher = TelegramMessageFetchWorker(
        bus, bot, dp, chat_id=CHAT_ID, topic_id=TOPIC_ID
    )
    action_sender = SendChatActionWorker(
        bus, bot, dp, chat_id=CHAT_ID, topic_id=TOPIC_ID
    )
    voice_uploader = VoiceMessageUploadWorker(
        bus, bot, chat_id=CHAT_ID, topic_id=TOPIC_ID
    )

    asyncio.create_task(dp.start_polling(bot, handle_signals=False))
    asyncio.create_task(audio_listener.start())
    asyncio.create_task(action_sender.start())
    asyncio.create_task(voice_uploader.start())
    asyncio.create_task(message_fetcher.start())
    asyncio.create_task(audio_player.start())

    while True:
        await asyncio.sleep(0.1)


if __name__ == "__main__":

    def handle_sigterm(_, __):
        logging.info("EXITING...")
        os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGHUP, handle_sigterm)

    logging.info("radiotelegram is starting. copyleft 2025 mike_went.")
    asyncio.run(main())
