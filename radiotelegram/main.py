import asyncio
import logging
import os
import signal

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from radiotelegram.bus import MessageBus
from radiotelegram.radio import EnhancedRxListenWorker, EnhancedTxPlayWorker
from radiotelegram.telegram import (
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

    # Create components
    bus = MessageBus(max_workers=6)  # Increase workers for better parallel processing
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    # Create workers
    audio_listener = EnhancedRxListenWorker(bus, audio_device=AUDIO_DEVICE)
    audio_player = EnhancedTxPlayWorker(bus)

    # Create workers with optional topic_id handling
    message_fetcher = TelegramMessageFetchWorker(
        bus, bot, dp, chat_id=CHAT_ID, topic_id=TOPIC_ID
    )
    action_sender = SendChatActionWorker(
        bus, bot, dp, chat_id=CHAT_ID, topic_id=TOPIC_ID
    )
    voice_uploader = VoiceMessageUploadWorker(
        bus, bot, chat_id=CHAT_ID, topic_id=TOPIC_ID
    )

    # Store workers for cleanup
    workers = [
        audio_listener,
        audio_player,
        message_fetcher,
        action_sender,
        voice_uploader,
    ]

    async def cleanup():
        """Cleanup function for graceful shutdown."""
        logging.info("Shutdown signal received, cleaning up...")

        # Stop all workers
        for worker in workers:
            try:
                await worker.stop()
            except Exception as e:
                logging.warning(
                    f"Error stopping worker {worker.__class__.__name__}: {e}"
                )

        # Shutdown message bus
        await bus.shutdown()

        logging.info("Cleanup complete")

    try:
        # Start all workers
        asyncio.create_task(dp.start_polling(bot, handle_signals=False))
        asyncio.create_task(audio_listener.start())
        asyncio.create_task(action_sender.start())
        asyncio.create_task(voice_uploader.start())
        asyncio.create_task(message_fetcher.start())
        asyncio.create_task(audio_player.start())

        # Main loop
        while True:
            await asyncio.sleep(0.1)

    except (KeyboardInterrupt, SystemExit):
        await cleanup()
    finally:
        # Ensure cleanup runs even if there's an unexpected exit
        try:
            await cleanup()
        except:
            pass


if __name__ == "__main__":

    def handle_signal(signum, frame):
        """Handle shutdown signals."""
        logging.info(f"Received signal {signum}, initiating shutdown...")
        os._exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGHUP, handle_signal)

    logging.info("radiotelegram is starting. copyleft 2025 mike_went.")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
    except SystemExit:
        logging.info("System exit received")
    finally:
        logging.info("Application terminated")
