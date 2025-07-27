import logging
import os
import signal
import threading
import time

import telebot
from dotenv import load_dotenv

from radiotelegram.bus import MessageBus
from radiotelegram.rx import EnhancedRxListenWorker
from radiotelegram.tx import EnhancedTxPlayWorker
from radiotelegram.telegram import (
    SendChatActionWorker,
    TelegramBotPollingWorker,
    TelegramMessageFetchWorker,
    VoiceMessageUploadWorker,
)

# Logger setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)


def main():
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
    bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

    # Create workers
    audio_listener = EnhancedRxListenWorker(bus, audio_device=AUDIO_DEVICE)
    audio_player = EnhancedTxPlayWorker(bus)

    # Create telegram bot polling worker for robust connection management
    bot_polling_worker = TelegramBotPollingWorker(bus, bot)

    # Create workers with optional topic_id handling
    message_fetcher = TelegramMessageFetchWorker(
        bus, bot, chat_id=CHAT_ID, topic_id=TOPIC_ID
    )
    action_sender = SendChatActionWorker(
        bus, bot, chat_id=CHAT_ID, topic_id=TOPIC_ID or 0
    )
    voice_uploader = VoiceMessageUploadWorker(
        bus, bot, chat_id=CHAT_ID, topic_id=TOPIC_ID or 0
    )

    # Store workers for cleanup
    workers = [
        audio_listener,
        audio_player,
        message_fetcher,
        action_sender,
        voice_uploader,
        bot_polling_worker,
    ]

    # Global flag to signal shutdown
    shutdown_flag = threading.Event()

    def cleanup():
        """Cleanup function for graceful shutdown."""
        logging.info("Shutdown signal received, cleaning up...")

        # Set shutdown flag
        shutdown_flag.set()

        # Stop all workers
        for worker in workers:
            try:
                worker.stop()
            except Exception as e:
                logging.warning(
                    f"Error stopping worker {worker.__class__.__name__}: {e}"
                )

        # Shutdown message bus
        bus.shutdown()

        logging.info("Cleanup complete")

    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logging.info(f"Received signal {signum}, initiating shutdown...")
        cleanup()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)

    try:
        # Start all workers in separate threads
        worker_threads = []
        for worker in workers:
            thread = threading.Thread(
                target=worker.start, name=f"{worker.__class__.__name__}Thread"
            )
            thread.daemon = True
            thread.start()
            worker_threads.append(thread)

        # Main loop - wait for shutdown signal
        while not shutdown_flag.is_set():
            try:
                shutdown_flag.wait(timeout=1.0)
            except KeyboardInterrupt:
                break

    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        # Ensure cleanup runs
        try:
            cleanup()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    logging.info("radiotelegram is starting. copyleft 2025 mike_went.")

    try:
        main()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
    except SystemExit:
        logging.info("System exit received")
    finally:
        logging.info("Application terminated")
