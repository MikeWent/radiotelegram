"""Tests for radiotelegram.main module."""

import threading
from unittest.mock import MagicMock, patch

import pytest


class TestMain:
    @patch("radiotelegram.main.TelegramBotPollingWorker")
    @patch("radiotelegram.main.VoiceMessageUploadWorker")
    @patch("radiotelegram.main.SendChatActionWorker")
    @patch("radiotelegram.main.TelegramMessageFetchWorker")
    @patch("radiotelegram.main.EnhancedTxPlayWorker")
    @patch("radiotelegram.main.EnhancedRxListenWorker")
    @patch("radiotelegram.main.telebot.TeleBot")
    @patch("radiotelegram.main.MessageBus")
    @patch("radiotelegram.main.load_dotenv")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "12345",
            "TELEGRAM_TOPIC_ID": "99",
            "AUDIO_DEVICE": "hw:1,0",
        },
    )
    def test_main_creates_all_workers(
        self,
        mock_dotenv,
        mock_bus_cls,
        mock_telebot,
        mock_rx,
        mock_tx,
        mock_fetcher,
        mock_action,
        mock_upload,
        mock_polling,
    ):
        from radiotelegram.main import main

        mock_bus = MagicMock()
        mock_bus_cls.return_value = mock_bus

        original_event = threading.Event

        class FakeEvent(original_event):
            def wait(self, timeout=None):
                raise KeyboardInterrupt

        with patch("radiotelegram.main.threading.Event", FakeEvent):
            main()

        mock_rx.assert_called_once()
        mock_tx.assert_called_once()
        mock_fetcher.assert_called_once()
        mock_action.assert_called_once()
        mock_upload.assert_called_once()
        mock_polling.assert_called_once()

    @patch.dict(
        "os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}, clear=True
    )
    def test_main_raises_without_env(self):
        from radiotelegram.main import main

        with patch("radiotelegram.main.load_dotenv"):
            with pytest.raises(AssertionError):
                main()
