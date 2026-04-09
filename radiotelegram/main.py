import logging
import os
import signal
import threading

import telebot
from dotenv import load_dotenv

from radiotelegram.bus import MessageBus
from radiotelegram.rx_worker import EnhancedRxListenWorker
from radiotelegram.tx import EnhancedTxPlayWorker
from radiotelegram.telegram import (
    SendChatActionWorker,
    TelegramBotPollingWorker,
    TelegramMessageFetchWorker,
    VoiceMessageUploadWorker,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)


def main():
    load_dotenv()
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    TOPIC_ID = int(os.getenv("TELEGRAM_TOPIC_ID", "0")) or None
    AUDIO_DEVICE = os.getenv("AUDIO_DEVICE", "pulse")
    assert TELEGRAM_BOT_TOKEN and CHAT_ID

    bus = MessageBus(max_workers=6)
    bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

    workers = [
        EnhancedRxListenWorker(bus, audio_device=AUDIO_DEVICE),
        EnhancedTxPlayWorker(bus),
        TelegramMessageFetchWorker(bus, bot, chat_id=CHAT_ID, topic_id=TOPIC_ID),
        SendChatActionWorker(bus, bot, chat_id=CHAT_ID, topic_id=TOPIC_ID or 0),
        VoiceMessageUploadWorker(bus, bot, chat_id=CHAT_ID, topic_id=TOPIC_ID or 0),
        TelegramBotPollingWorker(bus, bot),
    ]

    shutdown_flag = threading.Event()

    def signal_handler(signum, frame):
        logging.info(f"Signal {signum}, shutting down...")
        shutdown_flag.set()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal_handler)

    try:
        for w in workers:
            w.start()

        while not shutdown_flag.is_set():
            try:
                shutdown_flag.wait(timeout=1.0)
            except KeyboardInterrupt:
                break
    finally:
        for w in workers:
            try:
                w.stop()
            except Exception as e:
                logging.warning(f"Error stopping {w.__class__.__name__}: {e}")
        bus.shutdown()


if __name__ == "__main__":
    logging.info("radiotelegram starting. copyleft 2025 mike_went.")
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        logging.info("Application terminated")
