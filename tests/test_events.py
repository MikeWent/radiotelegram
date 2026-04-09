"""Tests for event flow through the message bus.

Verifies that events correctly carry data across stages and that the bus
routs each event type to exactly the right subscribers.
"""

import time
from unittest.mock import MagicMock

from radiotelegram.bus import MessageBus
from radiotelegram.events import (
    RxAudioStatsEvent,
    RxRecordingCompleteEvent,
    RxRecordingEndedEvent,
    RxRecordingStartedEvent,
    TelegramVoiceMessageDownloadedEvent,
    TxMessagePlaybackEndedEvent,
    TxMessagePlaybackStartedEvent,
)


class TestEventRouting:
    """The bus must deliver each event only to handlers subscribed to that type."""

    def setup_method(self):
        self.bus = MessageBus()

    def teardown_method(self):
        self.bus.shutdown()

    def test_recording_lifecycle_events_reach_separate_handlers(self):
        started = MagicMock()
        ended = MagicMock()
        complete = MagicMock()
        self.bus.subscribe(RxRecordingStartedEvent, started)
        self.bus.subscribe(RxRecordingEndedEvent, ended)
        self.bus.subscribe(RxRecordingCompleteEvent, complete)

        self.bus.publish(RxRecordingStartedEvent())
        self.bus.publish(RxRecordingEndedEvent())

        started.assert_called_once()
        ended.assert_called_once()
        complete.assert_not_called()  # only emitted after voice analysis passes

    def test_recording_complete_carries_filepath_to_subscriber(self):
        received = []
        self.bus.subscribe(
            RxRecordingCompleteEvent, lambda e: received.append(e.filepath)
        )
        self.bus.publish(RxRecordingCompleteEvent(filepath="/tmp/rec_001.ogg"))
        assert received == ["/tmp/rec_001.ogg"]

    def test_playback_events_do_not_trigger_recording_handlers(self):
        rec_handler = MagicMock()
        self.bus.subscribe(RxRecordingStartedEvent, rec_handler)

        self.bus.publish(TxMessagePlaybackStartedEvent())
        self.bus.publish(TxMessagePlaybackEndedEvent())

        rec_handler.assert_not_called()

    def test_telegram_download_event_carries_filepath(self):
        received = []
        self.bus.subscribe(
            TelegramVoiceMessageDownloadedEvent,
            lambda e: received.append(e.filepath),
        )
        self.bus.publish(
            TelegramVoiceMessageDownloadedEvent(filepath="/tmp/voice_42.ogg")
        )
        assert received == ["/tmp/voice_42.ogg"]

    def test_multiple_subscribers_all_receive_same_event(self):
        h1, h2, h3 = MagicMock(), MagicMock(), MagicMock()
        for h in (h1, h2, h3):
            self.bus.subscribe(RxRecordingStartedEvent, h)

        self.bus.publish(RxRecordingStartedEvent())

        for h in (h1, h2, h3):
            h.assert_called_once()


class TestAudioStatsEventFlow:
    """RxAudioStatsEvent is the sole stats carrier — verify it faithfully
    transports all fields through publish/subscribe."""

    def setup_method(self):
        self.bus = MessageBus()

    def teardown_method(self):
        self.bus.shutdown()

    def test_stats_event_delivers_all_metrics_to_subscriber(self):
        received = []
        self.bus.subscribe(RxAudioStatsEvent, received.append)

        self.bus.publish(
            RxAudioStatsEvent(
                current_db=-32.0,
                noise_floor_db=-50.0,
                level_margin_db=18.0,
                squelch_open=True,
                vad_probability=0.91,
                vad_flag=1,
                level_score=0.6,
                combined_score=0.78,
                avg_process_time_ms=1.2,
                max_process_time_ms=3.4,
            )
        )

        assert len(received) == 1
        e = received[0]
        assert e.squelch_open is True
        assert e.vad_flag == 1
        assert e.combined_score == 0.78
        assert e.avg_process_time_ms == 1.2

    def test_stats_event_scoring_defaults_to_zero(self):
        """When the rx_worker publishes without optional fields the stats
        consumer must still get zeros, not None."""
        received = []
        self.bus.subscribe(RxAudioStatsEvent, received.append)

        self.bus.publish(
            RxAudioStatsEvent(
                current_db=-60.0,
                noise_floor_db=-60.0,
                level_margin_db=0.0,
                squelch_open=False,
                vad_probability=0.0,
                vad_flag=0,
            )
        )

        e = received[0]
        assert e.level_score == 0.0
        assert e.combined_score == 0.0
        assert e.avg_process_time_ms == 0.0
